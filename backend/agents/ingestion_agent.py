"""
backend/agents/ingestion_agent.py

First node in the pipeline. Maps raw columns to the active process
module's canonical schema, resolves any derivable fields (formula-
based, dependency-ordered; llm_parse-based, batched per unique value),
and writes the cleaned data to a temp file.

Never crashes on unfamiliar input: an unmapped column is preserved,
not dropped; a failed derivation just leaves that field absent. The
only hard failure is a required field left unsatisfied after every
resolution attempt - that's reported via state.error, not an exception.
"""

from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field, create_model

from core.llm_client import call_structured
from core.logging_config import get_agent_logger, run_id_ctx
from core.schema_registry.base import FieldSpec, get_process_module
from orchestration.state import FibrionState, FieldResolution
from services.file_parser import FileParseError, parse_file

logger = get_agent_logger("ingestion")


class ColumnMapping(BaseModel):
    mappings: dict[str, str] = Field(
        description="Raw column name -> canonical field name. Only include "
        "a mapping if genuinely confident - leave ambiguous columns out "
        "rather than guessing."
    )


def _build_mapping_prompt(df: pd.DataFrame, module, data_dictionary: Optional[str]) -> str:
    columns_with_samples = [
        f"- {col!r}: sample values {df[col].dropna().unique()[:3].tolist()}"
        for col in df.columns
    ]
    field_list = "\n".join(
        f"- {f.name} ({f.dtype}): {f.description}" for f in module.required_fields
    )
    dict_section = (
        f"\nData dictionary provided by the uploader:\n{data_dictionary}\n"
        if data_dictionary
        else "\nNo data dictionary was provided - infer meaning from column "
        "names and sample values alone.\n"
    )
    return (
        f"You are mapping raw spreadsheet columns to a canonical schema for "
        f"{module.process_name} production data.\n\n"
        f"Domain context: {module.domain_context}\n{dict_section}\n"
        f"Raw columns, with sample values:\n{chr(10).join(columns_with_samples)}\n\n"
        f"Canonical fields to map to:\n{field_list}"
    )


def _resolve_formula_derivations(df, module, field_resolutions):
    pending = [
        f for f in module.required_fields
        if f.derivation and f.derivation.method == "formula" and f.name not in df.columns
    ]
    made_progress = True
    while pending and made_progress:
        made_progress = False
        still_pending = []
        for field in pending:
            rule = field.derivation
            if all(src in df.columns for src in rule.source_fields):
                # Row-wise apply, not vectorized - fine at this dataset's
                # scale, worth revisiting if a much larger file makes it slow.
                df[field.name] = df.apply(
                    lambda row: rule.formula({src: row[src] for src in rule.source_fields})
                    if all(pd.notna(row[src]) for src in rule.source_fields) else None,
                    axis=1,
                )
                field_resolutions[field.name] = FieldResolution(source="derived_formula", confidence=1.0)
                logger.info(f"Derived '{field.name}' via formula from {rule.source_fields}")
                made_progress = True
            else:
                still_pending.append(field)
        pending = still_pending
    return df


def _build_dynamic_parse_schema(fields: list[FieldSpec]):
    type_map = {"float": Optional[float], "str": Optional[str], "int": Optional[int]}
    field_defs = {f.name: (type_map.get(f.dtype, Optional[str]), None) for f in fields}
    ParsedItem = create_model("ParsedItem", original_value=(str, ...), **field_defs)
    return create_model("ParsedBatch", items=(list[ParsedItem], ...))

def _estimate_max_tokens(n_items: int, per_item: int = 60, base: int = 200, ceiling: int = 4096) -> int:
    """Rough budget for a batched llm_parse call: a fixed baseline for
    prompt/schema overhead, plus per-item cost for the JSON output -
    scales with how many distinct values are actually being parsed,
    instead of guessing one flat number regardless of batch size."""
    return min(ceiling, base + n_items * per_item)

def _resolve_llm_parse_derivations(df, module, field_resolutions):
    pending = [
        f for f in module.required_fields
        if f.derivation and f.derivation.method == "llm_parse" and f.name not in df.columns
    ]
    groups: dict[tuple, list[FieldSpec]] = {}
    for f in pending:
        groups.setdefault(tuple(f.derivation.source_fields), []).append(f)

    for source_fields, fields in groups.items():
        if len(source_fields) != 1 or source_fields[0] not in df.columns:
            continue
        source_col = source_fields[0]
        unique_values = df[source_col].dropna().unique().tolist()
        if not unique_values:
            continue

        schema = _build_dynamic_parse_schema(fields)
        instructions = "\n".join(f"- {f.name}: {f.derivation.parse_instruction}" for f in fields)
        prompt = (
            f"Parse each of these {len(unique_values)} distinct values from the "
            f"'{source_col}' column into its component fields.\n\n"
            f"Fields to extract:\n{instructions}\n\nValues:\n{unique_values}\n\n"
            f"Return one entry per input value, keeping the original value in "
            f"'original_value' so results can be matched back."
        )
        result, meta = call_structured(
            tier="fast", prompt=prompt, output_schema=schema,
            max_tokens=_estimate_max_tokens(len(unique_values)),
        )
        if result is None:
            logger.warning(f"llm_parse derivation for {[f.name for f in fields]} failed: {meta}")
            continue

        parsed_map = {item.original_value: item for item in result.items}
        coverage = sum(1 for v in unique_values if v in parsed_map) / len(unique_values)

        for field in fields:
            df[field.name] = df[source_col].map(
                lambda v: getattr(parsed_map[v], field.name, None) if v in parsed_map else None
            )
            field_resolutions[field.name] = FieldResolution(
                source="derived_llm_parse", confidence=round(coverage, 2), raw_column_name=source_col,
            )
        logger.info(f"Derived {[f.name for f in fields]} from '{source_col}', coverage={coverage:.0%}")

    return df


def run_ingestion(state: FibrionState) -> dict:
    run_id_ctx.set(state.run_id)
    module = get_process_module(state.process_type)

    try:
        df = parse_file(state.file_path)
    except FileParseError as e:
        logger.error(f"File parse failed: {e}")
        return {"error": {"type": "ingestion_file_unreadable", "detail": str(e)}}

    logger.info(f"Parsed {len(df)} rows, {len(df.columns)} columns")

    prompt = _build_mapping_prompt(df, module, state.data_dictionary)
    mapping_result, meta = call_structured(
        tier="fast", prompt=prompt, output_schema=ColumnMapping,
        max_tokens=min(4096, 300 + len(df.columns) * 30),
    )
    if mapping_result is None:
        logger.error(f"Column mapping call failed: {meta}")
        return {"error": {"type": "ingestion_mapping_call_failed", "detail": meta}}

    column_mapping_raw = mapping_result.mappings
    valid_field_names = {f.name for f in module.required_fields}
    column_mapping = {}
    invalid_targets = []
    for raw_col, canonical in column_mapping_raw.items():
        if canonical in valid_field_names:
            column_mapping[raw_col] = canonical
        else:
            invalid_targets.append((raw_col, canonical))
    if invalid_targets:
        logger.warning(f"LLM proposed unknown canonical targets, treating as unmapped: {invalid_targets}")

    unmapped_columns = [c for c in df.columns if c not in column_mapping]
    df = df.rename(columns=column_mapping)

    field_resolutions: dict[str, FieldResolution] = {
        canonical: FieldResolution(source="direct_mapping", confidence=1.0, raw_column_name=raw)
        for raw, canonical in column_mapping.items()
    }

    for field in module.required_fields:
        if field.name in df.columns and field.dtype == "float":
            df[field.name] = pd.to_numeric(df[field.name], errors="coerce")

    df = _resolve_formula_derivations(df, module, field_resolutions)
    df = _resolve_llm_parse_derivations(df, module, field_resolutions)

    missing_required = [
        f.name for f in module.required_fields
        if f.required and (f.name not in df.columns or df[f.name].isna().all())
    ]
    if missing_required:
        logger.error(f"Required fields unsatisfied after mapping and derivation: {missing_required}")
        return {
            "column_mapping": column_mapping,
            "unmapped_columns": unmapped_columns,
            "field_resolutions": field_resolutions,
            "error": {"type": "ingestion_missing_required_fields", "missing_fields": missing_required},
        }

    out_dir = Path("outputs/tmp")
    out_dir.mkdir(parents=True, exist_ok=True)
    cleaned_path = out_dir / f"{state.run_id}_cleaned.parquet"
    df.to_parquet(cleaned_path)

    logger.info(f"Ingestion complete: {len(df)} rows, {len(unmapped_columns)} unmapped, -> {cleaned_path}")
    return {
        "cleaned_data_path": str(cleaned_path),
        "column_mapping": column_mapping,
        "unmapped_columns": unmapped_columns,
        "field_resolutions": field_resolutions,
    }