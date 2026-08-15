"""
backend/core/schema_registry/weaving.py

Weaving's concrete implementation of ProcessModule - the first process
type, and the one everything else in phase 1 is validated against.
"""

import pandas as pd

from core.schema_registry.base import FieldSpec, ProcessModule, register_module


class WeavingModule(ProcessModule):

    @property
    def process_name(self) -> str:
        return "weaving"

    @property
    def required_fields(self) -> list[FieldSpec]:
        return [
            FieldSpec("date", "date", True, "Production date for this record"),
            FieldSpec("loom_id", "str", True, "Identifier for the specific loom"),
            FieldSpec("order_id", "str", True, "Customer order or work order identifier"),
            FieldSpec("fabric_construction", "str", False, "Weave spec, e.g. warp x weft count and density"),
            FieldSpec("warp_count", "float", False, "Yarn count of the warp (lengthwise) threads"),
            FieldSpec("weft_count", "float", False, "Yarn count of the weft (crosswise) threads"),
            FieldSpec("epi", "float", False, "Ends per inch - warp thread density"),
            FieldSpec("ppi", "float", False, "Picks per inch - weft thread density"),
            FieldSpec("production_qty_m", "float", True, "Fabric produced, in meters, for this record"),
            FieldSpec("target_qty_m", "float", True, "Target/planned production, in meters"),
            FieldSpec("rejection_qty_m", "float", True, "Fabric rejected as defective, in meters"),
            FieldSpec("shift_duration_min", "float", True, "Total available production minutes for the shift"),
            FieldSpec("stoppage_duration_min", "float", True, "Total loom downtime, in minutes, for this record"),
            FieldSpec("stoppage_cause_raw", "str", False, "Operator's free-text reason the loom stopped"),
            FieldSpec("shift", "str", False, "Production shift identifier"),
            FieldSpec("operator_id", "str", False, "Operator identifier"),
        ]

    @property
    def stoppage_categories(self) -> list[str]:
        return ["mechanical", "electrical", "material", "other"]

    @property
    def domain_context(self) -> str:
        return (
            "This data comes from a weaving shed. A loom interlaces warp "
            "threads (lengthwise, held under tension) with weft threads "
            "(crosswise, inserted by shuttle/rapier/airjet) to form fabric. "
            "EPI (ends per inch) and PPI (picks per inch) describe thread "
            "density. Loom stoppages are commonly caused by warp/weft "
            "breaks, mechanical faults, electrical faults, or material "
            "shortages. Efficiency compares actual production against the "
            "loom's target output for the same period."
        )

    def compute_kpis(self, df: pd.DataFrame) -> dict:
        by_loom = (
            df.groupby("loom_id")
            .agg(
                production_m=("production_qty_m", "sum"),
                target_m=("target_qty_m", "sum"),
                rejection_m=("rejection_qty_m", "sum"),
                downtime_min=("stoppage_duration_min", "sum"),
                available_min=("shift_duration_min", "sum"),
            )
            .reset_index()
        )
        # Ratios computed from summed totals, not averaged from per-row
        # percentages - a row producing 1000m must not count the same as
        # a row producing 10m when the loom-level rate is derived.
        by_loom["efficiency_pct"] = (by_loom["production_m"] / by_loom["target_m"] * 100).round(2)
        by_loom["downtime_pct"] = (by_loom["downtime_min"] / by_loom["available_min"] * 100).round(2)
        by_loom["rejection_pct"] = (by_loom["rejection_m"] / by_loom["production_m"] * 100).round(2)

        overall = {
            "total_production_m": float(df["production_qty_m"].sum()),
            "total_target_m": float(df["target_qty_m"].sum()),
            "overall_efficiency_pct": round(
                df["production_qty_m"].sum() / df["target_qty_m"].sum() * 100, 2
            ),
            "total_downtime_min": float(df["stoppage_duration_min"].sum()),
            "overall_downtime_pct": round(
                df["stoppage_duration_min"].sum() / df["shift_duration_min"].sum() * 100, 2
            ),
            "overall_rejection_pct": round(
                df["rejection_qty_m"].sum() / df["production_qty_m"].sum() * 100, 2
            ),
        }

        return {"overall": overall, "by_loom": by_loom.to_dict(orient="records")}


weaving_module = register_module(WeavingModule())