from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import pandas as pd

from .config import DATA_PATH, RAW_FEATURES
from .service import get_bundle

SIMULATION_END = pd.Timestamp("2026-09-25")
SIMULATION_DAYS = 42
DEFAULT_APPROVE_THRESHOLD = 0.75
DEFAULT_REJECT_THRESHOLD = 0.25
DEFAULT_ANALYSTS = 1
DEFAULT_PRODUCTIVE_HOURS = 6.0
DEFAULT_REVIEW_MINUTES = 24.0
DISCLAIMER = (
    "Business Analysis is an educational project simulation. Operational fields such as channel, "
    "timestamps, analyst assignment, processing time and SLA are synthetic and deterministic. "
    "The historical target is loan approval, not repayment/default risk."
)


def _safe_number(value: Any, fallback: float = 0.0) -> float:
    try:
        out = float(value)
        return fallback if not np.isfinite(out) else out
    except (TypeError, ValueError):
        return fallback


def _normalize01(series: pd.Series) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce")
    lo, hi = float(values.min()), float(values.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return pd.Series(np.zeros(len(series)), index=series.index, dtype=float)
    return ((values - lo) / (hi - lo)).clip(0, 1)


@lru_cache(maxsize=1)
def build_analysis_frame() -> pd.DataFrame:
    """Build a deterministic operational simulation on top of the original project data.

    The underlying applicant features and historical approval labels come from the supplied
    public loan dataset. Operations metadata is synthesized for project demonstration so
    the app can show realistic Business Analytics / Operations workflows without pretending
    the source contains fields it does not.
    """
    raw = pd.read_csv(DATA_PATH)
    bundle = get_bundle()
    champion = bundle["champion"]
    frame = raw.copy()

    x = frame[RAW_FEATURES].copy()
    frame["approval_probability"] = champion.predict_proba(x)[:, 1]
    frame["historical_approved"] = frame["Loan_Status"].astype(str).str.strip().eq("Y")
    frame["predicted_approved_50"] = frame["approval_probability"] >= 0.5
    frame["historical_match_50"] = frame["predicted_approved_50"] == frame["historical_approved"]
    frame["loan_value"] = pd.to_numeric(frame["LoanAmount"], errors="coerce").fillna(
        pd.to_numeric(frame["LoanAmount"], errors="coerce").median()
    ) * 1000.0

    # Deterministic pseudo-random operational layer. Seeded so screenshots/tests are stable.
    rng = np.random.default_rng(20260925)
    offsets = rng.integers(0, SIMULATION_DAYS, size=len(frame))
    hour_offsets = rng.integers(8, 19, size=len(frame))
    minute_offsets = rng.integers(0, 60, size=len(frame))
    frame["created_at"] = (
        SIMULATION_END
        - pd.to_timedelta(offsets, unit="D")
        + pd.to_timedelta(hour_offsets, unit="h")
        + pd.to_timedelta(minute_offsets, unit="m")
    )
    frame["simulation_day"] = frame["created_at"].dt.date.astype(str)
    recent = (SIMULATION_END - frame["created_at"].dt.normalize()).dt.days <= 14

    # Simulate an increasingly digital acquisition mix to create a useful operations trend.
    channel_draw = rng.random(len(frame))
    channel = np.empty(len(frame), dtype=object)
    recent_probs = [("Digital", 0.55), ("Branch", 0.18), ("Partner", 0.17), ("Assisted", 0.10)]
    older_probs = [("Digital", 0.34), ("Branch", 0.30), ("Partner", 0.23), ("Assisted", 0.13)]
    for i, draw in enumerate(channel_draw):
        choices = recent_probs if bool(recent.iloc[i]) else older_probs
        cumulative = 0.0
        selected = choices[-1][0]
        for name, prob in choices:
            cumulative += prob
            if draw <= cumulative:
                selected = name
                break
        channel[i] = selected
    frame["channel"] = channel

    missing_count = frame[RAW_FEATURES].isna().sum(axis=1)
    channel_doc_penalty = frame["channel"].map({"Digital": 0.01, "Branch": 0.04, "Partner": 0.10, "Assisted": 0.06}).astype(float)
    doc_noise = rng.uniform(0.00, 0.08, len(frame))
    frame["document_completeness"] = (1.0 - missing_count * 0.13 - channel_doc_penalty - doc_noise).clip(0.55, 1.0)
    frame["missing_fields"] = missing_count.astype(int)

    uncertainty = (1.0 - (frame["approval_probability"] - 0.5).abs() / 0.5).clip(0, 1)
    frame["uncertainty"] = uncertainty

    # Baseline route is deliberately conservative: ambiguous decisions go to a human queue.
    frame["baseline_route"] = np.select(
        [frame["approval_probability"] >= DEFAULT_APPROVE_THRESHOLD, frame["approval_probability"] <= DEFAULT_REJECT_THRESHOLD],
        ["Auto approve", "Auto reject"],
        default="Manual review",
    )

    analyst_names = np.array(["Analyst A", "Analyst B", "Analyst C", "Analyst D", "Analyst E", "Analyst F", "Analyst G", "Analyst H"])
    analyst_idx = rng.integers(0, len(analyst_names), len(frame))
    frame["analyst"] = np.where(frame["baseline_route"].eq("Manual review"), analyst_names[analyst_idx], "Automation")

    channel_tat = frame["channel"].map({"Digital": 0.0, "Branch": 5.5, "Partner": 9.0, "Assisted": 7.0}).astype(float)
    manual = frame["baseline_route"].eq("Manual review")
    base_tat = np.where(manual, 8.0 + 20.0 * uncertainty, 0.45 + 0.8 * uncertainty)
    document_tat = (1.0 - frame["document_completeness"]) * 42.0
    queue_noise = rng.gamma(shape=2.0, scale=2.0, size=len(frame))
    frame["processing_hours"] = (base_tat + channel_tat + document_tat + queue_noise).clip(0.2, 96.0)
    frame["sla_hours"] = np.where(manual, 24.0, 2.0)
    frame["sla_breached"] = frame["processing_hours"] > frame["sla_hours"]

    review_minutes = (
        DEFAULT_REVIEW_MINUTES
        + uncertainty * 24.0
        + missing_count * 7.0
        + frame["channel"].map({"Digital": 0.0, "Branch": 4.0, "Partner": 8.0, "Assisted": 6.0}).astype(float)
    )
    frame["estimated_review_minutes"] = review_minutes.clip(8.0, 75.0)

    exposure_norm = _normalize01(frame["loan_value"])
    frame["priority_score"] = (
        uncertainty * 45.0
        + frame["sla_breached"].astype(float) * 22.0
        + exposure_norm * 20.0
        + (1.0 - frame["document_completeness"]) * 13.0
    ).clip(0, 100)

    frame["priority_reason"] = np.select(
        [
            frame["sla_breached"],
            uncertainty >= 0.75,
            frame["document_completeness"] < 0.82,
            exposure_norm >= 0.8,
        ],
        ["SLA breach", "Low decision confidence", "Document friction", "High value at stake"],
        default="Standard review",
    )
    frame["application_id"] = frame["Loan_ID"].astype(str)
    return frame


def _route_for_policy(df: pd.DataFrame, approve_threshold: float, reject_threshold: float) -> pd.Series:
    return pd.Series(
        np.select(
            [df["approval_probability"] >= approve_threshold, df["approval_probability"] <= reject_threshold],
            ["Auto approve", "Auto reject"],
            default="Manual review",
        ),
        index=df.index,
        dtype="object",
    )


def _policy_metrics(
    df: pd.DataFrame,
    approve_threshold: float,
    reject_threshold: float,
    analysts: int = DEFAULT_ANALYSTS,
    productive_hours: float = DEFAULT_PRODUCTIVE_HOURS,
    volume_multiplier: float = 1.0,
) -> dict[str, Any]:
    if not 0 < reject_threshold < approve_threshold < 1:
        raise ValueError("Thresholds must satisfy 0 < reject < approve < 1")
    if analysts < 1 or productive_hours <= 0 or volume_multiplier <= 0:
        raise ValueError("Capacity inputs must be positive")

    route = _route_for_policy(df, approve_threshold, reject_threshold)
    manual_mask = route.eq("Manual review")
    auto_approve = route.eq("Auto approve")
    auto_reject = route.eq("Auto reject")
    automated = ~manual_mask

    historical = df["historical_approved"].to_numpy(dtype=bool)
    auto_prediction = np.where(auto_approve, True, False)
    auto_agreement = float((auto_prediction[automated.to_numpy()] == historical[automated.to_numpy()]).mean()) if automated.any() else 1.0

    n = len(df)
    manual_count = int(manual_mask.sum())
    auto_count = int(automated.sum())
    days = max(1, df["simulation_day"].nunique())
    review_hours = float(df.loc[manual_mask, "estimated_review_minutes"].sum() / 60.0) * volume_multiplier
    daily_review_hours = review_hours / days
    capacity_hours_daily = analysts * productive_hours
    utilization = daily_review_hours / capacity_hours_daily if capacity_hours_daily else 0.0
    daily_manual_cases = manual_count * volume_multiplier / days
    avg_review_hours = max(0.05, float(df.loc[manual_mask, "estimated_review_minutes"].mean() / 60.0)) if manual_count else DEFAULT_REVIEW_MINUTES / 60
    daily_case_capacity = capacity_hours_daily / avg_review_hours
    backlog_per_day = max(0.0, daily_manual_cases - daily_case_capacity)

    manual_everything_hours = float(df["estimated_review_minutes"].sum() / 60.0) * volume_multiplier
    hours_saved = max(0.0, manual_everything_hours - review_hours)

    historically_approved = df["historical_approved"]
    historically_rejected = ~historically_approved
    approved_capture = float((auto_approve & historically_approved).sum() / max(1, int(historically_approved.sum())))
    rejected_capture = float((auto_reject & historically_rejected).sum() / max(1, int(historically_rejected.sum())))

    return {
        "approve_threshold": round(float(approve_threshold), 4),
        "reject_threshold": round(float(reject_threshold), 4),
        "applications": int(n),
        "auto_approve": int(auto_approve.sum()),
        "auto_reject": int(auto_reject.sum()),
        "manual_review": manual_count,
        "auto_decision_rate": round(auto_count / n, 6),
        "manual_review_rate": round(manual_count / n, 6),
        "auto_historical_agreement": round(auto_agreement, 6),
        "historical_approval_auto_capture": round(approved_capture, 6),
        "historical_rejection_auto_capture": round(rejected_capture, 6),
        "auto_approved_value": round(float(df.loc[auto_approve, "loan_value"].sum()) * volume_multiplier, 2),
        "manual_review_value": round(float(df.loc[manual_mask, "loan_value"].sum()) * volume_multiplier, 2),
        "estimated_review_hours": round(review_hours, 2),
        "manual_everything_hours": round(manual_everything_hours, 2),
        "analyst_hours_saved": round(hours_saved, 2),
        "analysts": int(analysts),
        "productive_hours_per_day": round(float(productive_hours), 2),
        "volume_multiplier": round(float(volume_multiplier), 2),
        "capacity_utilization": round(float(utilization), 6),
        "required_analyst_fte": round(float(daily_review_hours / productive_hours), 3),
        "max_sustainable_volume_multiplier": round(float(volume_multiplier / utilization), 3) if utilization > 0 else 10.0,
        "daily_manual_cases": round(float(daily_manual_cases), 2),
        "daily_case_capacity": round(float(daily_case_capacity), 2),
        "backlog_growth_cases_per_day": round(float(backlog_per_day), 2),
        "capacity_status": "Over capacity" if utilization > 1 else ("Tight" if utilization > 0.85 else "Healthy"),
    }


def optimize_policy(
    df: pd.DataFrame,
    minimum_auto_agreement: float = 0.90,
    analysts: int = DEFAULT_ANALYSTS,
    productive_hours: float = DEFAULT_PRODUCTIVE_HOURS,
    volume_multiplier: float = 1.0,
) -> dict[str, Any]:
    """Maximize automated coverage subject to a historical-agreement guardrail.

    Historical agreement is deliberately named as such: the target is the historical approval
    label, not default or repayment behavior.
    """
    minimum_auto_agreement = float(np.clip(minimum_auto_agreement, 0.5, 0.999))
    best: tuple | None = None
    for approve in np.arange(0.55, 0.951, 0.025):
        for reject in np.arange(0.05, 0.451, 0.025):
            if reject >= approve:
                continue
            metrics = _policy_metrics(df, float(approve), float(reject), analysts, productive_hours, volume_multiplier)
            if metrics["auto_historical_agreement"] + 1e-12 < minimum_auto_agreement:
                continue
            # Automation first, then capacity health, then higher agreement, then symmetry.
            key = (
                metrics["auto_decision_rate"],
                -metrics["backlog_growth_cases_per_day"],
                metrics["auto_historical_agreement"],
                -abs((approve + reject) / 2 - 0.5),
            )
            if best is None or key > best[0]:
                best = (key, metrics)
    if best is None:
        fallback = _policy_metrics(df, 0.90, 0.10, analysts, productive_hours, volume_multiplier)
        return {"feasible": False, "minimum_auto_agreement": minimum_auto_agreement, "policy": fallback}
    return {"feasible": True, "minimum_auto_agreement": minimum_auto_agreement, "policy": best[1]}


def _group_metrics(df: pd.DataFrame, dimension: str, approve_threshold: float, reject_threshold: float) -> list[dict[str, Any]]:
    allowed = {
        "channel": "channel",
        "property_area": "Property_Area",
        "education": "Education",
        "self_employed": "Self_Employed",
        "credit_history": "Credit_History",
    }
    column = allowed.get(dimension)
    if column is None:
        raise ValueError(f"Unsupported dimension: {dimension}")
    route = _route_for_policy(df, approve_threshold, reject_threshold)
    work = df.copy()
    work["policy_route"] = route
    rows = []
    total = len(work)
    dataset_manual = float(route.eq("Manual review").mean())
    dataset_tat = float(work["processing_hours"].mean())
    for value, group in work.groupby(column, dropna=False):
        manual = group["policy_route"].eq("Manual review")
        automated = ~manual
        auto_pred = group["policy_route"].eq("Auto approve").to_numpy(dtype=bool)
        agreement = float((auto_pred[automated.to_numpy()] == group.loc[automated, "historical_approved"].to_numpy(dtype=bool)).mean()) if automated.any() else 1.0
        volume_share = len(group) / total
        manual_rate = float(manual.mean())
        avg_tat = float(group["processing_hours"].mean())
        score = volume_share * max(0.0, manual_rate - dataset_manual + 0.15) * max(0.0, avg_tat / max(dataset_tat, 0.1))
        rows.append({
            "segment": "Missing" if pd.isna(value) else str(value),
            "applications": int(len(group)),
            "volume_share": round(volume_share, 6),
            "historical_approval_rate": round(float(group["historical_approved"].mean()), 6),
            "manual_review_rate": round(manual_rate, 6),
            "auto_historical_agreement": round(agreement, 6),
            "avg_processing_hours": round(avg_tat, 2),
            "sla_breach_rate": round(float(group["sla_breached"].mean()), 6),
            "document_completeness": round(float(group["document_completeness"].mean()), 6),
            "loan_value": round(float(group["loan_value"].sum()), 2),
            "opportunity_score": round(float(score), 6),
        })
    return sorted(rows, key=lambda x: (x["opportunity_score"], x["applications"]), reverse=True)


def _trends(df: pd.DataFrame, approve_threshold: float, reject_threshold: float) -> list[dict[str, Any]]:
    work = df.copy()
    work["policy_route"] = _route_for_policy(work, approve_threshold, reject_threshold)
    daily = []
    for day, group in work.groupby("simulation_day"):
        daily.append({
            "date": str(day),
            "applications": int(len(group)),
            "historical_approval_rate": round(float(group["historical_approved"].mean()), 6),
            "manual_review_rate": round(float(group["policy_route"].eq("Manual review").mean()), 6),
            "sla_breach_rate": round(float(group["sla_breached"].mean()), 6),
            "avg_processing_hours": round(float(group["processing_hours"].mean()), 2),
            "digital_share": round(float(group["channel"].eq("Digital").mean()), 6),
        })
    return sorted(daily, key=lambda x: x["date"])


def _funnel(df: pd.DataFrame, approve_threshold: float, reject_threshold: float) -> list[dict[str, Any]]:
    route = _route_for_policy(df, approve_threshold, reject_threshold)
    complete = df["document_completeness"] >= 0.90
    return [
        {"stage": "Applications received", "count": int(len(df)), "rate": 1.0},
        {"stage": "Documents ≥90% complete", "count": int(complete.sum()), "rate": round(float(complete.mean()), 6)},
        {"stage": "Auto decision eligible", "count": int((~route.eq("Manual review")).sum()), "rate": round(float((~route.eq("Manual review")).mean()), 6)},
        {"stage": "Manual review", "count": int(route.eq("Manual review").sum()), "rate": round(float(route.eq("Manual review").mean()), 6)},
        {"stage": "Historical approvals", "count": int(df["historical_approved"].sum()), "rate": round(float(df["historical_approved"].mean()), 6)},
    ]


def _queue(df: pd.DataFrame, approve_threshold: float, reject_threshold: float, limit: int = 40) -> list[dict[str, Any]]:
    work = df.copy()
    work["policy_route"] = _route_for_policy(work, approve_threshold, reject_threshold)
    manual = work[work["policy_route"].eq("Manual review")].copy()
    manual["policy_priority"] = (
        manual["priority_score"]
        + manual["loan_value"].rank(pct=True).fillna(0) * 8
        + (manual["processing_hours"] / 24).clip(0, 2) * 6
    ).clip(0, 100)
    manual = manual.sort_values(["policy_priority", "processing_hours", "loan_value"], ascending=False).head(max(1, min(int(limit), 200)))
    fields = []
    for _, row in manual.iterrows():
        fields.append({
            "application_id": row["application_id"],
            "priority": round(float(row["policy_priority"]), 1),
            "priority_reason": row["priority_reason"],
            "approval_probability": round(float(row["approval_probability"]), 6),
            "loan_value": round(float(row["loan_value"]), 2),
            "channel": row["channel"],
            "property_area": str(row.get("Property_Area", "")),
            "document_completeness": round(float(row["document_completeness"]), 6),
            "processing_hours": round(float(row["processing_hours"]), 2),
            "sla_breached": bool(row["sla_breached"]),
            "analyst": row["analyst"],
            "historical_outcome": "Approved" if bool(row["historical_approved"]) else "Rejected",
        })
    return fields


def _data_quality(df: pd.DataFrame) -> dict[str, Any]:
    source = pd.read_csv(DATA_PATH)
    missing = []
    for col in source.columns:
        count = int(source[col].isna().sum())
        if count:
            missing.append({"field": col, "missing": count, "rate": round(count / len(source), 6)})
    missing.sort(key=lambda x: x["missing"], reverse=True)
    return {
        "rows": int(len(source)),
        "columns": int(len(source.columns)),
        "duplicate_rows": int(source.duplicated().sum()),
        "duplicate_ids": int(source["Loan_ID"].duplicated().sum()) if "Loan_ID" in source else 0,
        "missing_cells": int(source.isna().sum().sum()),
        "missing_by_field": missing,
        "historical_approval_rate": round(float(df["historical_approved"].mean()), 6),
        "notes": [
            "Gender is retained only as a source-data field and is not used by the rebuilt prediction model.",
            "Operational timestamps, channels, analyst assignments, processing time and SLA are simulated for project demonstration.",
            "The target records historical approval decisions; it is not a default or repayment outcome.",
        ],
    }


def _analysis_summary(df: pd.DataFrame, policy: dict[str, Any], segments: list[dict[str, Any]]) -> list[str]:
    insights = []
    top = segments[0] if segments else None
    auto_pct = policy["auto_decision_rate"] * 100
    agreement_pct = policy["auto_historical_agreement"] * 100
    manual_pct = policy["manual_review_rate"] * 100
    insights.append(
        f"Current policy automates {auto_pct:.1f}% of applications with {agreement_pct:.1f}% agreement against historical decisions on automated cases; {manual_pct:.1f}% still require human review."
    )
    if policy["capacity_status"] == "Over capacity":
        insights.append(
            f"The simulated review team is over capacity and would add about {policy['backlog_growth_cases_per_day']:.1f} cases to backlog per day at the selected volume assumption."
        )
    elif policy["capacity_status"] == "Tight":
        insights.append(
            f"Review capacity is tight; the current staffing supports only about {policy['max_sustainable_volume_multiplier']:.2f}× baseline demand before backlog starts growing."
        )
    else:
        insights.append(
            f"Review capacity is healthy at {policy['capacity_utilization']*100:.0f}% utilization; current staffing can sustain roughly {policy['max_sustainable_volume_multiplier']:.2f}× baseline demand before backlog growth."
        )
    if top:
        insights.append(
            f"The strongest operational improvement candidate is {top['segment']} in the selected segment view: {top['manual_review_rate']*100:.1f}% manual review and {top['avg_processing_hours']:.1f}h average processing time."
        )
    high_priority = int((df["priority_score"] >= 70).sum())
    insights.append(f"There are {high_priority} high-priority simulated cases where uncertainty, SLA pressure, documentation friction or loan value justify early analyst attention.")
    return insights


def run_business_analysis(
    approve_threshold: float = DEFAULT_APPROVE_THRESHOLD,
    reject_threshold: float = DEFAULT_REJECT_THRESHOLD,
    analysts: int = DEFAULT_ANALYSTS,
    productive_hours: float = DEFAULT_PRODUCTIVE_HOURS,
    volume_multiplier: float = 1.0,
    segment_dimension: str = "channel",
) -> dict[str, Any]:
    df = build_analysis_frame()
    policy = _policy_metrics(df, approve_threshold, reject_threshold, analysts, productive_hours, volume_multiplier)
    segments = _group_metrics(df, segment_dimension, approve_threshold, reject_threshold)
    optimized = optimize_policy(df, 0.90, analysts, productive_hours, volume_multiplier)
    return {
        "simulation": True,
        "simulation_period": {"start": str((SIMULATION_END - pd.Timedelta(days=SIMULATION_DAYS - 1)).date()), "end": str(SIMULATION_END.date()), "days": SIMULATION_DAYS},
        "policy": policy,
        "optimized_policy_90pct_guardrail": optimized,
        "funnel": _funnel(df, approve_threshold, reject_threshold),
        "trends": _trends(df, approve_threshold, reject_threshold),
        "segments": segments,
        "queue": _queue(df, approve_threshold, reject_threshold, 40),
        "data_quality": _data_quality(df),
        "analysis_summary": _analysis_summary(df, policy, segments),
        "dataset_summary": {
            "applications": int(len(df)),
            "historical_approvals": int(df["historical_approved"].sum()),
            "historical_approval_rate": round(float(df["historical_approved"].mean()), 6),
            "loan_value": round(float(df["loan_value"].sum()), 2),
            "avg_processing_hours": round(float(df["processing_hours"].mean()), 2),
            "sla_breach_rate": round(float(df["sla_breached"].mean()), 6),
            "document_completeness": round(float(df["document_completeness"].mean()), 6),
        },
        "disclaimer": DISCLAIMER,
    }


def case_detail(application_id: str) -> dict[str, Any]:
    df = build_analysis_frame()
    match = df[df["application_id"].eq(str(application_id))]
    if match.empty:
        raise KeyError(application_id)
    row = match.iloc[0]
    return {
        "application_id": row["application_id"],
        "historical_outcome": "Approved" if bool(row["historical_approved"]) else "Rejected",
        "approval_probability": round(float(row["approval_probability"]), 6),
        "baseline_route": row["baseline_route"],
        "priority_score": round(float(row["priority_score"]), 1),
        "priority_reason": row["priority_reason"],
        "channel": row["channel"],
        "analyst": row["analyst"],
        "processing_hours": round(float(row["processing_hours"]), 2),
        "sla_hours": round(float(row["sla_hours"]), 2),
        "sla_breached": bool(row["sla_breached"]),
        "document_completeness": round(float(row["document_completeness"]), 6),
        "loan_value": round(float(row["loan_value"]), 2),
        "applicant": {
            "married": None if pd.isna(row.get("Married")) else str(row.get("Married")),
            "dependents": None if pd.isna(row.get("Dependents")) else str(row.get("Dependents")),
            "education": None if pd.isna(row.get("Education")) else str(row.get("Education")),
            "self_employed": None if pd.isna(row.get("Self_Employed")) else str(row.get("Self_Employed")),
            "applicant_income": _safe_number(row.get("ApplicantIncome")),
            "coapplicant_income": _safe_number(row.get("CoapplicantIncome")),
            "loan_amount_thousands": _safe_number(row.get("LoanAmount")),
            "loan_amount_term_months": _safe_number(row.get("Loan_Amount_Term")),
            "credit_history": None if pd.isna(row.get("Credit_History")) else int(row.get("Credit_History")),
            "property_area": None if pd.isna(row.get("Property_Area")) else str(row.get("Property_Area")),
        },
        "audit": {
            "model": get_bundle()["metadata"]["champion_model"],
            "model_version": get_bundle()["metadata"]["model_version"],
            "source_target": "historical loan approval",
            "simulation_fields": ["channel", "created_at", "analyst", "processing_hours", "sla_hours"],
        },
        "disclaimer": DISCLAIMER,
    }
