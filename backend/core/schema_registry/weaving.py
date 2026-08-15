"""
backend/core/schema_registry/weaving.py

Weaving's concrete implementation of ProcessModule, built around the
18 columns actually verified in the real Mendeley/Evince Textiles
weaving dataset and its accompanying Data in Brief paper.

Note on naming: the paper's own table and its formula text disagree on
one field's name (assump_crimp% vs shrink_allow%), and the raw source
data separately tracked loom_id/stoppage data that this particular
cleaned dataset dropped. Different real weaving exports will look
different from this one - that's handled by the ingestion mapping and
unmapped-column preservation, not by pre-guessing every variant here.
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
            # --- core fields every fulfillment/rejection/shrinkage KPI needs ---
            FieldSpec("order_id", "str", True,
                      "Customer order identifier; multiple rows can share one order_id"),
            FieldSpec("req_grey_fabric_yds", "float", True,
                      "Required grey (unfinished, pre-shrinkage) fabric, in yards - "
                      "the production target in the same units as actual output"),
            FieldSpec("total_pdn_per_order_yds", "float", True,
                      "Total grey fabric produced for this order, in yards"),
            FieldSpec("rejection_yds", "float", True,
                      "Fabric rejected/cut due to defects (damage, floating yarn, "
                      "loose picks, pattern mismatch), in yards"),
            FieldSpec("shrink_allow_pct", "float", True,
                      "Planned/assumed shrinkage percent used when calculating "
                      "required grey fabric and beam length"),
            FieldSpec("act_shrink_pct", "float", True,
                      "Actual measured shrinkage percent for this batch"),

            # --- supporting / planning context, optional ---
            FieldSpec("month", "str", False, "Production month"),
            FieldSpec("fabric_construction", "str", False,
                      "Weave spec string: warp count x weft count / EPI x PPI"),
            FieldSpec("req_finish_fabric_yds", "float", False,
                      "Finished fabric quantity the customer's order requires"),
            FieldSpec("fabric_allowance_pct", "float", False,
                      "Planning buffer percent for wastage/safety margin, "
                      "separate from the shrinkage allowance"),
            FieldSpec("rec_beam_length_yds", "float", False,
                      "Beam length of warp yarn actually received/supplied, in yards"),
            FieldSpec("req_beam_length_yds", "float", False,
                      "Beam length of warp yarn the order requires, in yards"),
            FieldSpec("previous_pdn_yds", "float", False,
                      "Production carried forward from a previous shift",
                      null_is_meaningful=True),
            FieldSpec("total_pdn_today_yds", "float", False,
                      "Production summed across this day's shifts",
                      null_is_meaningful=True),
            FieldSpec("warp_count", "str", False,
                      "Warp yarn count - real data mixes plain numbers with "
                      "construction notes like 'double_80', needs a parsing rule"),
            FieldSpec("weft_count", "float", False, "Weft yarn count"),
            FieldSpec("epi", "float", False, "Ends per inch - warp thread density"),
            FieldSpec("ppi", "float", False, "Picks per inch - weft thread density"),
        ]

    @property
    def stoppage_categories(self) -> list[str]:
        # Not present as a column in this particular dataset - this defines
        # the categories for *if* a future weaving source includes stoppage
        # data, the way the original raw report (per the paper) did before
        # this dataset's authors stripped it out.
        return ["mechanical", "electrical", "material", "other"]

    @property
    def domain_context(self) -> str:
        return (
            "This data tracks weaving orders from grey (unfinished, "
            "pre-shrinkage) fabric production through to shrinkage during "
            "finishing. Grey fabric quantity required is calculated from the "
            "finished-fabric order plus a shrinkage allowance, since fabric "
            "shrinks during dyeing/finishing after weaving. EPI (ends per "
            "inch) and PPI (picks per inch) describe thread density; warp and "
            "weft counts describe yarn thickness. Rejection covers fabric cut "
            "out for defects such as broken picks, loose picks, or pattern "
            "mismatch."
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
            by_order["produced_grey_yds"] / by_order["required_grey_yds"] * 100
        ).round(2)
        by_order["rejection_pct"] = (
            by_order["rejection_yds"] / by_order["produced_grey_yds"] * 100
        ).round(2)
        by_order["shrink_variance_pct"] = (
            by_order["avg_actual_shrink_pct"] - by_order["planned_shrink_pct"]
        ).round(2)

        overall = {
            "total_produced_grey_yds": float(df["total_pdn_per_order_yds"].sum()),
            "total_required_grey_yds": float(df["req_grey_fabric_yds"].sum()),
            "overall_fulfillment_pct": round(
                df["total_pdn_per_order_yds"].sum() / df["req_grey_fabric_yds"].sum() * 100, 2
            ),
            "overall_rejection_pct": round(
                df["rejection_yds"].sum() / df["total_pdn_per_order_yds"].sum() * 100, 2
            ),
            "avg_shrink_variance_pct": round(
                (df["act_shrink_pct"] - df["shrink_allow_pct"]).mean(), 2
            ),
        }

        return {"overall": overall, "by_order": by_order.to_dict(orient="records")}


weaving_module = register_module(WeavingModule())