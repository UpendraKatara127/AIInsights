from __future__ import annotations

import math
from dataclasses import asdict
from typing import Any, Dict, List, Tuple

from ..config import Thresholds
from .metrics import linear_regression_slope, sudden_drop_indicator, unstable_indicator, severity_score


def _quantile(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    if q <= 0:
        return float(s[0])
    if q >= 1:
        return float(s[-1])
    idx = int(math.floor(q * (len(s) - 1)))
    return float(s[max(0, min(idx, len(s) - 1))])


def _thresholds_from_dict(thresholds: Dict[str, Any]) -> Thresholds:
    base = asdict(Thresholds())
    merged = {**base, **(thresholds or {})}
    return Thresholds(
        downward_slope_threshold=float(merged.get("downward_slope_threshold")),
        sudden_drop_percentile=int(merged.get("sudden_drop_percentile")),
        minimum_sudden_drop_amount=float(merged.get("minimum_sudden_drop_amount")),
        unstable_z_threshold=float(merged.get("unstable_z_threshold")),
        unstable_ratio_threshold=float(merged.get("unstable_ratio_threshold")),
    )


def recommend_lookback(
    datastore: Any,
    *,
    thresholds: Dict[str, Any],
    top_k: int,
    candidate_days: List[int] | None = None,
    max_probe_days: int = 365,
) -> Dict[str, Any]:
    candidates = candidate_days or [5, 10, 15, 20, 25, 30]
    candidates = sorted({int(x) for x in candidates if int(x) > 1})

    system_ids = list(datastore.list_system_ids())
    lengths: List[int] = []
    for sid in system_ids:
        ts = datastore.fetch_system_timeseries(sid, int(max_probe_days))
        lengths.append(int(len(ts.get("values", []) or [])))

    max_len = max(lengths) if lengths else 0
    min_len = min(lengths) if lengths else 0

    t = _thresholds_from_dict(thresholds)
    rows: List[Dict[str, Any]] = []

    for lb in [c for c in candidates if c <= max_len]:
        slopes: List[float] = []
        drops: List[float] = []
        unstable_ratios: List[float] = []
        any_flags = 0
        down_flags = 0
        sudden_flags = 0
        unstable_flags = 0
        eligible = 0

        for sid in system_ids:
            ts = datastore.fetch_system_timeseries(sid, int(lb))
            values = ts.get("values", []) or []
            if len(values) < lb:
                continue
            eligible += 1
            y = [float(p["v"]) for p in values]
            slope = float(linear_regression_slope(y))
            slopes.append(slope)

            down_flag = slope <= t.downward_slope_threshold
            sudden = sudden_drop_indicator(y, thresholds=t)
            unstable = unstable_indicator(y, thresholds=t)
            sudden_flag = bool(sudden.get("flag"))
            unstable_flag = bool(unstable.get("flag"))
            sev = severity_score(down_flag, sudden_flag, unstable_flag).get("severity", 0)

            drops.append(float(sudden.get("drop_amount") or 0.0))
            unstable_ratios.append(float(unstable.get("unstable_ratio") or 0.0))

            if down_flag:
                down_flags += 1
            if sudden_flag:
                sudden_flags += 1
            if unstable_flag:
                unstable_flags += 1
            if sev and int(sev) > 0:
                any_flags += 1

        rows.append(
            {
                "lookback_days": int(lb),
                "eligible_systems": int(eligible),
                "flag_counts": {
                    "any": int(any_flags),
                    "downtrend_point": int(down_flags),
                    "sudden": int(sudden_flags),
                    "unstable": int(unstable_flags),
                },
                "quantiles": {
                    "slope_point": {
                        "p05": _quantile(slopes, 0.05),
                        "p50": _quantile(slopes, 0.50),
                        "p95": _quantile(slopes, 0.95),
                    },
                    "drop_amount": {
                        "p50": _quantile(drops, 0.50),
                        "p95": _quantile(drops, 0.95),
                    },
                    "unstable_ratio": {
                        "p50": _quantile(unstable_ratios, 0.50),
                        "p95": _quantile(unstable_ratios, 0.95),
                    },
                },
            }
        )

    desired_min = max(1, int(top_k))
    desired_max = max(desired_min, int(top_k) * 10)

    eligible_rows = [r for r in rows if r.get("eligible_systems", 0) > 0]
    in_band = [
        r
        for r in eligible_rows
        if desired_min <= int((r.get("flag_counts") or {}).get("any", 0)) <= desired_max
    ]
    if in_band:
        in_band.sort(key=lambda r: (int((r["flag_counts"] or {}).get("any", 0)), -int(r["lookback_days"])))
        chosen = int(in_band[0]["lookback_days"])
    else:
        ge_min = [
            r for r in eligible_rows if int((r.get("flag_counts") or {}).get("any", 0)) >= desired_min
        ]
        if ge_min:
            ge_min.sort(key=lambda r: (-int(r["lookback_days"]), int((r["flag_counts"] or {}).get("any", 0))))
            chosen = int(ge_min[0]["lookback_days"])
        else:
            chosen = int(max_len) if max_len > 0 else int(candidates[-1] if candidates else 20)

    return {
        "recommended_lookback_days": int(chosen),
        "candidates": rows,
        "series_length": {"min": int(min_len), "max": int(max_len), "n_systems": int(len(system_ids))},
        "selection_band": {"min_any_flags": int(desired_min), "max_any_flags": int(desired_max)},
    }

