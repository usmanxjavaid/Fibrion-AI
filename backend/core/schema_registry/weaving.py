"""
backend/core/schema_registry/weaving.py
"""

import pandas as pd

from core.schema_registry.base import (
    DerivationRule,
    FieldSpec,
    ProcessModule,
    register_module,
)


def _grey_fabric_formula(f: dict) -> float:
    return f["req_finish_fabric_yds"] * 100 / (100 - f["fabric_allowance_pct"])


def _beam_length_formula(f: dict) -> float:
    return f["req_grey_fabric_yds"] * 100 / (100 - f["shrink_allow_pct"])


class WeavingModule(ProcessModule):

    @property
    def process_name(self) -> str:
        return "weaving"

    @property
    def required_fields(self) -> list[FieldSpec]:
        return [
            FieldSpec("order_id", "str", True, "Customer order identifier"),

            FieldSpec("req_grey_fabric_yds", "float", True,
                      "Required grey fabric, in yards",
                      derivation=DerivationRule(
                          source_fields=["req_finish_fabric_yds", "fabric_allowance_pct"],
                          method="formula", formula=_grey_fabric_formula)),
            FieldSpec("total_pdn_per_order_yds", "float", True,
                      "Total grey fabric produced for this order, in yards"),
            FieldSpec("rejection_yds", "float", True,
                      "Fabric rejected/cut for defects, in yards"),
            FieldSpec("shrink_allow_pct", "float", True, "Planned shrinkage percent"),
            FieldSpec("act_shrink_pct", "float", True, "Actual measured shrinkage percent"),

            # --- from this specific dataset, still optional/supporting ---
            FieldSpec("month", "str", False, "Production month"),
            FieldSpec("fabric_construction", "str", False, "Compound weave spec string"),
            FieldSpec("req_finish_fabric_yds", "float", False, "Finished fabric order quantity"),
            FieldSpec("fabric_allowance_pct", "float", False, "Planning buffer percent"),
            FieldSpec("rec_beam_length_yds", "float", False, "Beam length actually received"),
            FieldSpec("req_beam_length_yds", "float", False, "Beam length required",
                      derivation=DerivationRule(
                          source_fields=["req_grey_fabric_yds", "shrink_allow_pct"],
                          method="formula", formula=_beam_length_formula)),
            FieldSpec("previous_pdn_yds", "float", False, "Carried forward from previous shift",
                      null_is_meaningful=True),
            FieldSpec("total_pdn_today_yds", "float", False, "Production summed across shifts",
                      null_is_meaningful=True),
            FieldSpec("warp_count", "str", False, "Warp yarn count",
                      derivation=DerivationRule(source_fields=["fabric_construction"],
                          method="llm_parse", parse_instruction=(
                              "Format is warp x weft / EPI x PPI, e.g. '40x40/110x80' means "
                              "warp=40, weft=40, EPI=110, PPI=80. Extract ONLY the first number, "
                              "before the first 'x' - e.g. 40 from '40x40/110x80'."))),
            FieldSpec("weft_count", "float", False, "Weft yarn count",
                      derivation=DerivationRule(source_fields=["fabric_construction"],
                          method="llm_parse", parse_instruction=(
                              "Format is warp x weft / EPI x PPI, e.g. '40x40/110x80' means "
                              "warp=40, weft=40, EPI=110, PPI=80. Extract the second number, "
                              "between the first 'x' and the '/' - e.g. 40 from '40x40/110x80'."))),
            FieldSpec("epi", "float", False, "Ends per inch",
                      derivation=DerivationRule(source_fields=["fabric_construction"],
                          method="llm_parse", parse_instruction=(
                              "Format is warp x weft / EPI x PPI, e.g. '40x40/110x80' means "
                              "warp=40, weft=40, EPI=110, PPI=80. Extract the first number after "
                              "the '/' - e.g. 110 from '40x40/110x80'."))),
            FieldSpec("ppi", "float", False, "Picks per inch",
                      derivation=DerivationRule(source_fields=["fabric_construction"],
                          method="llm_parse", parse_instruction=(
                              "Format is warp x weft / EPI x PPI, e.g. '40x40/110x80' means "
                              "warp=40, weft=40, EPI=110, PPI=80. Extract the last number, "
                              "after the second 'x' - e.g. 80 from '40x40/110x80'."))),

            # --- standard mill fields NOT in this dataset, but common
            # enough across real weaving sheds to be worth defining now.
            # All optional, all zero-cost when absent - compute_kpis
            # below only uses them when they're actually present. ---
            FieldSpec("loom_id", "str", False,
                      "Identifier for the specific loom - enables per-loom breakdown"),
            FieldSpec("loom_speed_rpm", "float", False,
                      "Loom operating speed, revolutions per minute"),
            FieldSpec("stoppage_duration_min", "float", False,
                      "Total loom downtime, in minutes, for this record"),
            FieldSpec("shift_duration_min", "float", False,
                      "Total available production minutes for the shift - "
                      "needed alongside stoppage_duration_min to compute downtime %"),
            FieldSpec("stoppage_cause_raw", "str", False,
                      "Operator's free-text reason the loom stopped - classified "
                      "by the Validation Agent into stoppage_categories below"),
            FieldSpec("defect_type", "str", False,
                      "Category of defect on rejected fabric, e.g. broken pick, "
                      "loose pick, pattern mismatch, floating yarn"),
            FieldSpec("gsm", "float", False,
                      "Fabric weight, grams per square meter"),
            FieldSpec("fabric_width_inches", "float", False,
                      "Fabric width in inches, as set by the loom's reed"),
        ]

    @property
    def stoppage_categories(self) -> list[str]:
        return ["mechanical", "electrical", "material", "other"]

    @property
    def domain_context(self) -> str:
        return (
            "This data tracks weaving orders from grey (unfinished, "
            "pre-shrinkage) fabric production through to shrinkage during "
            "finishing. Grey fabric required is calculated from the "
            "finished-fabric order plus a shrinkage allowance. EPI/PPI "
            "describe thread density; warp/weft counts describe yarn "
            "thickness. Where available, loom-level fields (loom_id, "
            "speed, downtime, defect type) support machine-level analysis "
            "alongside the order-level fulfillment and shrinkage metrics."
        )

    def compute_kpis(self, df: pd.DataFrame) -> dict:
        by_order = (
            df.groupby("order_id")
            .agg(
                produced_grey_yds=("total_pdn_per_order_yds", "sum"),
                required_grey_yds=("req_grey_fabric_yds", "first"),
                rejection_yds=("rejection_yds", "sum"),
                avg_actual_shrink_pct=("act_shrink_pct", "mean"),
                planned_shrink_pct=("shrink_allow_pct", "first"),
            )
            .reset_index()
        )
        by_order["fulfillment_pct"] = (
            by_order["produced_grey_yds"] / by_order["required_grey_yds"] * 100).round(2)
        by_order["rejection_pct"] = (
            by_order["rejection_yds"] / by_order["produced_grey_yds"] * 100).round(2)
        by_order["shrink_variance_pct"] = (
            by_order["avg_actual_shrink_pct"] - by_order["planned_shrink_pct"]).round(2)

        overall = {
            "total_produced_grey_yds": float(df["total_pdn_per_order_yds"].sum()),
            "total_required_grey_yds": float(df["req_grey_fabric_yds"].sum()),
            "overall_fulfillment_pct": round(
                df["total_pdn_per_order_yds"].sum() / df["req_grey_fabric_yds"].sum() * 100, 2),
            "overall_rejection_pct": round(
                df["rejection_yds"].sum() / df["total_pdn_per_order_yds"].sum() * 100, 2),
            "avg_shrink_variance_pct": round(
                (df["act_shrink_pct"] - df["shrink_allow_pct"]).mean(), 2),
        }

        result = {"overall": overall, "by_order": by_order.to_dict(orient="records")}

        # --- graceful use of optional fields: only computed if present ---
        if "loom_id" in df.columns and "stoppage_duration_min" in df.columns \
                and "shift_duration_min" in df.columns:
            by_loom = (
                df.groupby("loom_id")
                .agg(downtime_min=("stoppage_duration_min", "sum"),
                     available_min=("shift_duration_min", "sum"))
                .reset_index()
            )
            by_loom["downtime_pct"] = (
                by_loom["downtime_min"] / by_loom["available_min"] * 100).round(2)
            result["by_loom"] = by_loom.to_dict(orient="records")

        if "defect_type" in df.columns:
            result["defect_breakdown"] = (
                df["defect_type"].value_counts().to_dict()
            )

        return result


weaving_module = register_module(WeavingModule())