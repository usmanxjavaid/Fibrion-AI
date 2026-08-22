"""
backend/agents/kpi_agent.py

Pure computation, no LLM. Calls the active process module's own
compute_kpis - the math already lives in schema_registry - then runs
a threshold-based anomaly pass over whatever group-level breakdown
came back, flagging statistical outliers for the Analysis Agent to
narrate later. Anomaly detection here is arithmetic, not a model
guessing at what looks unusual.
"""

import pandas as pd

from core.logging_config import get_agent_logger, run_id_ctx
from core.schema_registry.base import get_process_module
from orchestration.state import FibrionState

logger = get_agent_logger("kpi")

# Standard deviations from the mean before something's flagged. 2.0 is
# a textbook default, not derived from this dataset - worth revisiting
# once there's enough real run history to check against actual variance.
Z_SCORE_THRESHOLD = 2.0

ANOMALY_METRICS = ["fulfillment_pct", "rejection_pct", "shrink_variance_pct"]


def _detect_anomalies(by_group: list[dict], group_key: str) -> list[dict]:
    if len(by_group) < 3:
        return []  # too few points for a z-score to mean anything

    df = pd.DataFrame(by_group)
    anomalies = []
    for metric in ANOMALY_METRICS:
        if metric not in df.columns:
            continue
        mean, std = df[metric].mean(), df[metric].std()
        if std == 0 or pd.isna(std):
            continue
        z_scores = (df[metric] - mean) / std
        for idx, z in z_scores.items():
            if abs(z) > Z_SCORE_THRESHOLD:
                anomalies.append({
                    "group_key": group_key, "group_value": df.loc[idx, group_key],
                    "metric": metric, "value": round(float(df.loc[idx, metric]), 2),
                    "run_mean": round(float(mean), 2), "z_score": round(float(z), 2),
                })
    return anomalies


def run_kpi(state: FibrionState) -> dict:
    run_id_ctx.set(state.run_id)
    if state.error or not state.cleaned_data_path:
        logger.warning("Skipping KPI computation - no cleaned data available")
        return {}    
    module = get_process_module(state.process_type)
    df = pd.read_parquet(state.cleaned_data_path)

    try:
        kpi_results = module.compute_kpis(df)
    except Exception as e:
        logger.error(f"KPI computation failed: {e}")
        return {"error": {"type": "kpi_computation_failed", "detail": str(e)}}

    anomalies = []
    if "by_order" in kpi_results:
        primary_orders = [o for o in kpi_results["by_order"]
                            if not o.get("is_supplementary") and not o.get("is_non_order_material")]
        anomalies += _detect_anomalies(primary_orders, "order_id")
    if "by_loom" in kpi_results:
        anomalies += _detect_anomalies(kpi_results["by_loom"], "loom_id")

    logger.info(f"KPI computation complete: {len(anomalies)} anomalies flagged")
    return {"kpi_results": kpi_results, "anomalies": anomalies}