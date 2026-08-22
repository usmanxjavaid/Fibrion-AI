"""
backend/agents/discovery_agent.py

Statistical profiling, not another hand-written pattern. Detects
whether a numeric field behaves as a periodic cumulative checkpoint
(recommend max() aggregation, not sum()) and whether a recurring
structural pattern in an ID-like field represents a genuinely
distinct category (not just a coincidental naming convention) -
both confirmed by testing the actual data, not assumed from a name.

Deliberately standalone for now, not yet wired into weaving.py's
compute_kpis - proven against real data first, wired in once trusted.
"""

import pandas as pd

from core.logging_config import get_agent_logger

logger = get_agent_logger("discovery")

_CANDIDATE_ID_PATTERNS = {
    "paren_letter_suffix": r"\([A-Z]\)$",
    "dash_letter_suffix": r"-[A-Z]$",
    "dash_number_suffix": r"-\d+$",
    "no_digits": r"^\D+$",
}


def _profile_field_aggregation(
    df: pd.DataFrame, group_key: str, candidate_fields: list[str],
    zero_fraction_threshold: float = 0.5, monotonic_threshold: float = 0.9,
    constant_threshold: float = 0.85,
) -> dict[str, dict]:
    results = {}
    for field in candidate_fields:
        if field not in df.columns:
            continue
        group_stats = []
        for _, group in df.groupby(group_key):
            values = group[field].dropna()
            if len(values) < 3:
                continue
            zero_frac = (values == 0).mean()
            nonzero = values[values != 0]
            monotonic_frac = 1.0 if len(nonzero) < 2 else (nonzero.diff().dropna() >= 0).mean()
            # <=2 distinct values allows one legitimate mid-order revision,
            # not just a perfectly unchanging field
            is_constant_in_group = values.nunique() <= 2
            group_stats.append({
                "zero_frac": zero_frac, "monotonic_frac": monotonic_frac,
                "is_constant": is_constant_in_group,
            })

        if not group_stats:
            continue
        stats_df = pd.DataFrame(group_stats)
        avg_zero = stats_df["zero_frac"].mean()
        avg_monotonic = stats_df["monotonic_frac"].mean()
        constant_fraction = stats_df["is_constant"].mean()

        is_cumulative = avg_zero >= zero_fraction_threshold and avg_monotonic >= monotonic_threshold
        is_constant_per_group = (not is_cumulative) and constant_fraction >= constant_threshold
        # Mostly-but-not-clearly constant is genuinely ambiguous, not a
        # case for "sum" - summing a near-constant value is a far more
        # dangerous wrong guess (multiplies it by row count) than
        # under-aggregating a real incremental field, so the honest
        # answer here is "uncertain", not a silent default toward the
        # riskier mistake.
        is_ambiguous = (not is_cumulative) and (not is_constant_per_group) and constant_fraction >= 0.5

        if is_cumulative:
            recommended, confidence = "max", ("high" if avg_zero > 0.7 else "medium")
        elif is_constant_per_group:
            recommended, confidence = "first_or_last", ("high" if constant_fraction > 0.95 else "medium")
        elif is_ambiguous:
            recommended, confidence = "uncertain_review_needed", "low"
        else:
            recommended, confidence = "sum", ("high" if constant_fraction < 0.3 and avg_zero < 0.2 else "medium")
        results[field] = {
            "avg_zero_fraction": round(float(avg_zero), 3),
            "avg_monotonic_fraction": round(float(avg_monotonic), 3),
            "constant_within_group_fraction": round(float(constant_fraction), 3),
            "recommended_aggregation": recommended,
            "confidence": confidence,
        }
    return results


def _detect_id_subgroups(
    df_with_metric: pd.DataFrame, id_field: str, metric_field: str,
    min_group_size: int = 10, divergence_ratio: float = 2.0,
) -> dict[str, dict]:
    results = {}
    for pattern_name, pattern in _CANDIDATE_ID_PATTERNS.items():
        is_match = df_with_metric[id_field].astype(str).str.contains(pattern, regex=True)
        if is_match.sum() < min_group_size or (~is_match).sum() < min_group_size:
            continue

        subgroup_median = df_with_metric.loc[is_match, metric_field].median()
        rest_median = df_with_metric.loc[~is_match, metric_field].median()
        if not rest_median or pd.isna(rest_median) or pd.isna(subgroup_median):
            continue

        ratio = subgroup_median / rest_median
        is_distinct = ratio >= divergence_ratio or ratio <= (1 / divergence_ratio)

        results[pattern_name] = {
            "match_count": int(is_match.sum()),
            "subgroup_median": round(float(subgroup_median), 2),
            "rest_median": round(float(rest_median), 2),
            "ratio": round(float(ratio), 2),
            "is_distinct_category": bool(is_distinct),
        }
    return results


def run_discovery(
    df: pd.DataFrame, group_key: str, numeric_fields: list[str],
    ratio_numerator: str, ratio_denominator: str,
) -> dict:
    aggregation_profile = _profile_field_aggregation(df, group_key, numeric_fields)

    naive = df.groupby(group_key).agg(
        naive_numerator=(ratio_numerator, "sum"),
        naive_denominator=(ratio_denominator, "first"),
    ).reset_index()
    naive["naive_ratio"] = naive["naive_numerator"] / naive["naive_denominator"]

    id_subgroups = _detect_id_subgroups(naive, group_key, "naive_ratio")

    logger.info(f"Discovery: {len(aggregation_profile)} fields profiled, "
                f"{sum(1 for v in id_subgroups.values() if v['is_distinct_category'])} distinct subgroup patterns found")
    return {"aggregation_candidates": aggregation_profile, "id_subgroup_candidates": id_subgroups}