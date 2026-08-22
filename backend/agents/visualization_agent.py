"""
backend/agents/visualization_agent.py

Generates chart images for the PDF report. No LLM - this is
rendering, not judgment. Mirrors compute_kpis's own graceful
degradation: only generates a chart for whatever breakdown is
actually present in the KPI results, rather than assuming every
dataset supports every chart.
"""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")  # headless - no display needed
import matplotlib.pyplot as plt
import pandas as pd

from core.logging_config import get_agent_logger, run_id_ctx
from orchestration.state import FibrionState

logger = get_agent_logger("visualization")

plt.rcParams.update({
    "figure.facecolor": "white", "axes.facecolor": "white",
    "axes.edgecolor": "#444444", "axes.labelcolor": "#222222",
    "text.color": "#222222", "font.size": 10,
})


def _save(fig, out_dir: Path, name: str) -> str:
    path = out_dir / f"{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return str(path)


def _chart_fulfillment_distribution(by_order: list[dict], out_dir: Path):
    values = [o["fulfillment_pct"] for o in by_order
              if not o.get("is_supplementary") and o.get("fulfillment_pct") is not None]
    if len(values) < 3:
        return None
    # Clipped display range - a few extreme anomalies would otherwise
    # flatten the histogram into one bar. The anomaly list itself
    # (from kpi_agent) is where those specific outliers actually show.
    display_values = [v for v in values if v <= 300]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(display_values, bins=30, color="#4C72B0", edgecolor="white")
    ax.axvline(100, color="#C44E52", linestyle="--", linewidth=1.5, label="100% (target)")
    ax.set_xlabel("Fulfillment %")
    ax.set_ylabel("Number of orders")
    ax.set_title("Order fulfillment distribution")
    ax.legend()
    return _save(fig, out_dir, "fulfillment_distribution")


def _chart_top_anomalies(anomalies: list[dict], out_dir: Path, n: int = 10):
    if not anomalies:
        return None
    top = sorted(anomalies, key=lambda a: abs(a["z_score"]), reverse=True)[:n]
    labels = [f"{a['group_value']} ({a['metric']})" for a in top][::-1]
    z_scores = [a["z_score"] for a in top][::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(labels, z_scores, color="#DD8452")
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Z-score (standard deviations from mean)")
    ax.set_title(f"Top {len(top)} flagged anomalies")
    return _save(fig, out_dir, "top_anomalies")


def _chart_by_loom(by_loom: list[dict], out_dir: Path):
    if not by_loom:
        return None
    df = pd.DataFrame(by_loom).sort_values("downtime_pct", ascending=False).head(15)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(df["loom_id"].astype(str)[::-1], df["downtime_pct"][::-1], color="#55A868")
    ax.set_xlabel("Downtime %")
    ax.set_title("Loom downtime (highest first)")
    return _save(fig, out_dir, "loom_downtime")


def _chart_defect_breakdown(defect_breakdown: dict, out_dir: Path):
    if not defect_breakdown:
        return None
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.pie(defect_breakdown.values(), labels=defect_breakdown.keys(), autopct="%1.0f%%", colors=plt.cm.Set2.colors)
    ax.set_title("Defect type breakdown")
    return _save(fig, out_dir, "defect_breakdown")

def _chart_rejection_by_construction(by_order: list[dict], out_dir: Path, top_n: int = 10):
    df = pd.DataFrame([o for o in by_order if not o.get("is_supplementary")])
    if "fabric_construction" not in df.columns or df.empty:
        return None
    # top N by order count, not just any N constructions - a rare
    # one-off construction with 2 orders isn't worth the same visual
    # weight as one used across 80 orders.
    top_constructions = df["fabric_construction"].value_counts().head(top_n).index
    grouped = (
        df[df["fabric_construction"].isin(top_constructions)]
        .groupby("fabric_construction")["rejection_pct"].mean()
        .sort_values(ascending=False)
    )
    if len(grouped) < 2:
        return None

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh(grouped.index.astype(str)[::-1], grouped.values[::-1], color="#8172B3")
    ax.set_xlabel("Average rejection %")
    ax.set_title(f"Rejection rate by fabric construction (top {len(grouped)} by order count)")
    return _save(fig, out_dir, "rejection_by_construction")

def run_visualization(state: FibrionState) -> dict:
    run_id_ctx.set(state.run_id)
    if state.error or not state.kpi_results:
        logger.warning("Skipping visualization - no KPI results available")
        return {}

    out_dir = Path("outputs/reports") / state.run_id / "charts"
    out_dir.mkdir(parents=True, exist_ok=True)
    kpi = state.kpi_results
    chart_paths = []

    for chart in [
        _chart_fulfillment_distribution(kpi.get("by_order", []), out_dir),
        _chart_top_anomalies(state.anomalies, out_dir),
        _chart_rejection_by_construction(kpi.get("by_order", []), out_dir),
        _chart_by_loom(kpi.get("by_loom", []), out_dir),
        _chart_defect_breakdown(kpi.get("defect_breakdown", {}), out_dir),
    ]:
        if chart:
            chart_paths.append(chart)

    logger.info(f"Visualization complete: {len(chart_paths)} charts generated")
    return {"chart_paths": chart_paths}