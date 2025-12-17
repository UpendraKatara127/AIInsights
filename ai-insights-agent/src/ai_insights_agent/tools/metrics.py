from __future__ import annotations

import math
import random
from typing import Dict, List, Tuple

from ..config import BootstrapConfig, Thresholds


def _values_from_points(values: List[Dict]) -> List[float]:
    return [float(v["v"]) for v in values]


def linear_regression_slope(y: List[float]) -> float:
    n = len(y)
    if n < 2:
        return 0.0
    x_mean = (n + 1) / 2.0
    y_mean = sum(y) / n
    numerator = 0.0
    denominator = 0.0
    for i, yi in enumerate(y, start=1):
        dx = i - x_mean
        numerator += dx * (yi - y_mean)
        denominator += dx * dx
    if denominator == 0.0:
        return 0.0
    return numerator / denominator


def _percentile_element(values_sorted: List[float], percentile: int) -> float:
    n = len(values_sorted)
    if n == 0:
        return 0.0
    idx = int(math.floor(n * percentile / 100.0))
    idx = max(0, min(idx, n - 1))
    return values_sorted[idx]


def _quantile_sorted(values_sorted: List[float], q: float) -> float:
    if not values_sorted:
        return 0.0
    if q <= 0:
        return values_sorted[0]
    if q >= 1:
        return values_sorted[-1]
    idx = int(math.floor(q * (len(values_sorted) - 1)))
    return values_sorted[idx]


def bootstrap_slope(
    y: List[float],
    *,
    threshold: float,
    bootstrap: BootstrapConfig,
) -> Dict:
    n = len(y)
    if n < 2:
        return {
            "enabled": True,
            "iterations": bootstrap.iterations,
            "confidence": bootstrap.confidence,
            "slope_mean": 0.0,
            "ci_low": 0.0,
            "ci_high": 0.0,
            "p_below_threshold": 0.0,
            "p_negative": 0.0,
        }

    rng = random.Random(bootstrap.seed)
    slopes: List[float] = []
    below = 0
    negative = 0

    x = list(range(1, n + 1))
    pairs = list(zip(x, y))

    for _ in range(int(bootstrap.iterations)):
        sample = [pairs[rng.randrange(0, n)] for _ in range(n)]
        xs = [p[0] for p in sample]
        ys = [p[1] for p in sample]
        x_mean = sum(xs) / n
        y_mean = sum(ys) / n
        numerator = 0.0
        denominator = 0.0
        for xi, yi in zip(xs, ys):
            dx = xi - x_mean
            numerator += dx * (yi - y_mean)
            denominator += dx * dx
        slope_b = 0.0 if denominator == 0.0 else numerator / denominator
        slopes.append(slope_b)
        if slope_b <= threshold:
            below += 1
        if slope_b < 0:
            negative += 1

    slopes.sort()
    alpha = (1.0 - float(bootstrap.confidence)) / 2.0
    return {
        "enabled": True,
        "iterations": int(bootstrap.iterations),
        "confidence": float(bootstrap.confidence),
        "slope_mean": sum(slopes) / len(slopes),
        "ci_low": _quantile_sorted(slopes, alpha),
        "ci_high": _quantile_sorted(slopes, 1.0 - alpha),
        "p_below_threshold": below / len(slopes),
        "p_negative": negative / len(slopes),
    }


def downtrend_indicator(
    y: List[float],
    *,
    thresholds: Thresholds,
    bootstrap: BootstrapConfig,
) -> Dict:
    slope_point = linear_regression_slope(y)
    if not bootstrap.enabled:
        flag = slope_point <= thresholds.downward_slope_threshold
        return {
            "flag": bool(flag),
            "slope_point": slope_point,
            "threshold": thresholds.downward_slope_threshold,
            "bootstrap": {"enabled": False},
            "reason": "slope_below_threshold" if flag else "slope_not_below_threshold",
        }

    b = bootstrap_slope(y, threshold=thresholds.downward_slope_threshold, bootstrap=bootstrap)
    flag = b["p_below_threshold"] >= bootstrap.downtrend_confidence_min
    return {
        "flag": bool(flag),
        "slope_point": slope_point,
        "threshold": thresholds.downward_slope_threshold,
        "bootstrap": b,
        "reason": "bootstrap_confident_downtrend" if flag else "bootstrap_not_confident",
    }


def sudden_drop_indicator(y: List[float], *, thresholds: Thresholds) -> Dict:
    n = len(y)
    if n == 0:
        return {
            "flag": False,
            "last": None,
            "percentile_value": None,
            "percentile": thresholds.sudden_drop_percentile,
            "drop_amount": None,
            "min_drop": thresholds.minimum_sudden_drop_amount,
            "reason": "empty",
        }
    last = y[-1]
    sorted_y = sorted(y)
    perc_value = _percentile_element(sorted_y, thresholds.sudden_drop_percentile)
    drop_amount = perc_value - last
    flag = last <= perc_value and drop_amount >= thresholds.minimum_sudden_drop_amount
    return {
        "flag": bool(flag),
        "last": float(last),
        "percentile_value": float(perc_value),
        "percentile": int(thresholds.sudden_drop_percentile),
        "drop_amount": float(drop_amount),
        "min_drop": float(thresholds.minimum_sudden_drop_amount),
        "reason": "last_below_percentile_with_min_drop" if flag else "not_sudden_drop",
    }


def unstable_indicator(y: List[float], *, thresholds: Thresholds) -> Dict:
    n = len(y)
    if n == 0:
        return {
            "flag": False,
            "unstable_days": 0,
            "n": 0,
            "unstable_ratio": 0.0,
            "z_threshold": thresholds.unstable_z_threshold,
            "ratio_threshold": thresholds.unstable_ratio_threshold,
            "reason": "empty",
        }
    mean = sum(y) / n
    variance = sum((yi - mean) ** 2 for yi in y) / n
    std = math.sqrt(variance)
    if std == 0.0:
        return {
            "flag": False,
            "unstable_days": 0,
            "n": n,
            "unstable_ratio": 0.0,
            "z_threshold": thresholds.unstable_z_threshold,
            "ratio_threshold": thresholds.unstable_ratio_threshold,
            "reason": "zero_stddev",
        }
    unstable_days = sum(1 for yi in y if abs((yi - mean) / std) >= thresholds.unstable_z_threshold)
    ratio = unstable_days / n
    flag = unstable_days >= n * thresholds.unstable_ratio_threshold
    return {
        "flag": bool(flag),
        "unstable_days": int(unstable_days),
        "n": int(n),
        "unstable_ratio": float(ratio),
        "z_threshold": float(thresholds.unstable_z_threshold),
        "ratio_threshold": float(thresholds.unstable_ratio_threshold),
        "reason": "too_many_outliers" if flag else "not_unstable",
    }


def severity_score(downward: bool, sudden: bool, unstable: bool) -> Dict:
    weights = {"downward": 2, "sudden": 3, "unstable": 1}
    severity = (weights["downward"] if downward else 0) + (weights["sudden"] if sudden else 0) + (
        weights["unstable"] if unstable else 0
    )
    return {"severity": int(severity), "weights": weights}


def flag_series(values: List[Dict], thresholds: Dict) -> Dict:
    t = Thresholds(
        downward_slope_threshold=float(thresholds.get("downward_slope_threshold", -0.15)),
        sudden_drop_percentile=int(thresholds.get("sudden_drop_percentile", 10)),
        minimum_sudden_drop_amount=float(thresholds.get("minimum_sudden_drop_amount", 2.5)),
        unstable_z_threshold=float(thresholds.get("unstable_z_threshold", 1.5)),
        unstable_ratio_threshold=float(thresholds.get("unstable_ratio_threshold", 0.3)),
    )
    b = BootstrapConfig(
        enabled=bool(thresholds.get("bootstrap_enabled", True)),
        iterations=int(thresholds.get("bootstrap_iterations", 2000)),
        confidence=float(thresholds.get("bootstrap_confidence", 0.95)),
        seed=int(thresholds.get("bootstrap_seed", 42)),
        downtrend_confidence_min=float(thresholds.get("downtrend_confidence_min", 0.80)),
    )
    y = _values_from_points(values)
    if y:
        n = len(y)
        mean = sum(y) / n
        var = sum((yi - mean) ** 2 for yi in y) / n
        std = math.sqrt(var)
        first = float(y[0])
        last = float(y[-1])
        delta = last - first
        pct = (delta / first * 100.0) if first not in (0.0, -0.0) else None
        summary = {
            "n": int(n),
            "first": first,
            "last": last,
            "delta": float(delta),
            "pct_change": float(pct) if pct is not None else None,
            "min": float(min(y)),
            "max": float(max(y)),
            "mean": float(mean),
            "std": float(std),
        }
    else:
        summary = {"n": 0}
    downward = downtrend_indicator(y, thresholds=t, bootstrap=b)
    sudden = sudden_drop_indicator(y, thresholds=t)
    unstable = unstable_indicator(y, thresholds=t)
    sev = severity_score(bool(downward["flag"]), bool(sudden["flag"]), bool(unstable["flag"]))
    return {"summary": summary, "downward": downward, "sudden": sudden, "unstable": unstable, "severity": sev}
