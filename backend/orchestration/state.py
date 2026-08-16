"""
backend/orchestration/state.py

The shared object flowing through every node in the pipeline. Each
agent reads what it needs and returns updates to a subset of these
fields - LangGraph merges those into the running state as the graph
executes. Fields here aren't Annotated with a reducer, since each
field is owned by exactly one node in our graph (no parallel writers
to the same key) - LangGraph's default overwrite-on-update is correct.
"""

from typing import Literal, Optional

from pydantic import BaseModel, Field


class FieldResolution(BaseModel):
    """How one canonical field's value was actually obtained, and how
    much to trust it - carried through to the final report."""
    source: Literal["direct_mapping", "derived_formula", "derived_llm_parse"]
    confidence: float = 1.0
    raw_column_name: Optional[str] = None


class FibrionState(BaseModel):
    # --- set at the start of a run ---
    run_id: str
    file_path: str
    process_type: str = "weaving"
    delivery_channels: list[Literal["telegram", "email", "whatsapp"]] = Field(default_factory=list)
    data_dictionary: Optional[str] = None

    # --- ingestion outputs ---
    cleaned_data_path: Optional[str] = None   # path to a temp parquet file, not the df itself
    column_mapping: dict[str, str] = Field(default_factory=dict)
    unmapped_columns: list[str] = Field(default_factory=list)
    field_resolutions: dict[str, FieldResolution] = Field(default_factory=dict)

    # --- validation outputs ---
    validation_report: dict = Field(default_factory=dict)

    # --- kpi outputs ---
    kpi_results: dict = Field(default_factory=dict)
    anomalies: list[dict] = Field(default_factory=list)

    # --- analysis outputs ---
    analysis_text: Optional[str] = None

    # --- visualization outputs ---
    chart_paths: list[str] = Field(default_factory=list)

    # --- report outputs ---
    report_path: Optional[str] = None

    # --- verification outputs ---
    verification_passed: Optional[bool] = None
    verification_issues: list[str] = Field(default_factory=list)
    retry_count: int = 0

    # --- notification outputs ---
    delivery_status: dict[str, str] = Field(default_factory=dict)

    # --- unified failure signal - every early-exit path sets this,
    # notification checks it once to decide error-message vs real report ---
    error: Optional[dict] = None