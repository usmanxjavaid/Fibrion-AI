"""
backend/agents/validation_agent.py

Row-level checks on already-ingested data, plus batched LLM
classification of free-text stoppage reasons when that field exists.
Critical issues (a high fraction of impossible values in one field, or
heavy duplication) route to state.error; a small number is logged as
a minor issue and the pipeline continues - a null in a field the
module marks null_is_meaningful is never flagged at all, since the
paper already told us that can be a real state, not an error.
"""

from typing import Optional

import pandas as pd
from pydantic import BaseModel, Field

from core.llm_client import call_structured
from core.logging_config import get_agent_logger, run_id_ctx
from core.schema_registry.base import get_process_module
from orchestration.state import FibrionState

logger = get_agent_logger("validation")

# Fraction of rows with an impossible value in one field, or of exact-
# duplicate rows, above which the whole file is treated as structurally
# suspect rather than a handful of typos. Tune if it proves too tight
# or too loose once run against more real files.
CRITICAL_THRESHOLD = 0.05


class StoppageClassification(BaseModel):
    classifications: dict[str, str] = Field(
        description="Raw stoppage text -> one of the valid categories"
    )


def _check_impossible_values(df: pd.DataFrame, module) -> dict:
    issues = {}
    for field in module.required_fields:
        if not field.required or field.dtype != "float" or field.name not in df.columns:
            continue
        negative_count = int((df[field.name] < 0).sum())
        if negative_count > 0:
            issues[field.name] = {
                "type": "negative_value", "count": negative_count,
                "fraction": round(negative_count / len(df), 4),
            }
    return issues


def _check_nulls(df: pd.DataFrame, module) -> dict:
    issues = {}
    for field in module.required_fields:
        if field.name not in df.columns or field.null_is_meaningful:
            continue
        null_count = int(df[field.name].isna().sum())
        if null_count > 0:
            issues[field.name] = {
                "type": "unexpected_null", "count": null_count,
                "fraction": round(null_count / len(df), 4),
            }
    return issues


def _check_duplicates(df: pd.DataFrame) -> dict:
    dup_count = int(df.duplicated().sum())
    return {"count": dup_count, "fraction": round(dup_count / len(df), 4)} if dup_count else {}


def _classify_stoppages(df: pd.DataFrame, module) -> tuple[pd.DataFrame, Optional[dict]]:
    if "stoppage_cause_raw" not in df.columns:
        return df, None
    unique_reasons = df["stoppage_cause_raw"].dropna().unique().tolist()
    if not unique_reasons:
        return df, None

    prompt = (
        f"Classify each of these {len(unique_reasons)} loom stoppage reasons "
        f"into exactly one of: {module.stoppage_categories}.\n\n"
        f"Domain context: {module.domain_context}\n\nReasons:\n{unique_reasons}"
    )
    result, meta = call_structured(
        tier="fast", prompt=prompt, output_schema=StoppageClassification,
        max_tokens=min(4096, 200 + len(unique_reasons) * 40),
    )
    if result is None:
        logger.warning(f"Stoppage classification failed: {meta}")
        return df, {"type": "stoppage_classification_failed", "detail": meta}

    df["stoppage_cause_category"] = df["stoppage_cause_raw"].map(result.classifications)
    logger.info(f"Classified {len(unique_reasons)} distinct stoppage reasons")
    return df, None


def run_validation(state: FibrionState) -> dict:
    run_id_ctx.set(state.run_id)
    module = get_process_module(state.process_type)
    df = pd.read_parquet(state.cleaned_data_path)

    impossible_values = _check_impossible_values(df, module)
    null_issues = _check_nulls(df, module)
    duplicate_issues = _check_duplicates(df)
    df, classification_error = _classify_stoppages(df, module)

    critical_fields = [n for n, d in impossible_values.items() if d["fraction"] > CRITICAL_THRESHOLD]
    critical_duplicates = duplicate_issues.get("fraction", 0) > CRITICAL_THRESHOLD

    validation_report = {
        "row_count": len(df), "impossible_values": impossible_values,
        "null_issues": null_issues, "duplicates": duplicate_issues,
        "stoppage_classification_error": classification_error,
    }

    if critical_fields or critical_duplicates:
        logger.error(f"Critical validation failure: bad_fields={critical_fields}, duplicates_critical={critical_duplicates}")
        return {
            "validation_report": validation_report,
            "error": {"type": "validation_critical_failure", "bad_fields": critical_fields,
                       "duplicates_critical": critical_duplicates},
        }

    df.to_parquet(state.cleaned_data_path)  # overwrite - may now include stoppage_cause_category
    logger.info(f"Validation passed: {len(df)} rows")
    return {"validation_report": validation_report}