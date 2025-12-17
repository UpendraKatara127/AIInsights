from __future__ import annotations

from ai_insights_agent.tools.metrics import (
    linear_regression_slope,
    sudden_drop_indicator,
    unstable_indicator,
    downtrend_indicator,
    severity_score,
)
from ai_insights_agent.config import BootstrapConfig, Thresholds


def test_slope_formula_matches_doc():
    y = [1.0, 2.0, 3.0, 4.0]
    slope = linear_regression_slope(y)
    assert abs(slope - 1.0) < 1e-9


def test_sudden_drop_percentile_floor_index():
    t = Thresholds(sudden_drop_percentile=10, minimum_sudden_drop_amount=2.5)
    y = [100, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    out = sudden_drop_indicator([float(v) for v in y], thresholds=t)
    assert out["percentile_value"] == 2.0


def test_unstable_zscore_ratio():
    t = Thresholds(unstable_z_threshold=1.0, unstable_ratio_threshold=0.3)
    y = [0.0] * 10 + [10.0] * 10
    out = unstable_indicator(y, thresholds=t)
    assert out["flag"] is True
    assert out["unstable_ratio"] >= 0.3


def test_bootstrap_downtrend_confidence_gate():
    t = Thresholds(downward_slope_threshold=-0.15)
    b = BootstrapConfig(enabled=True, iterations=500, confidence=0.95, seed=42, downtrend_confidence_min=0.8)
    y = [10.0 - 0.5 * i for i in range(20)]
    out = downtrend_indicator(y, thresholds=t, bootstrap=b)
    assert out["bootstrap"]["enabled"] is True
    assert out["bootstrap"]["p_below_threshold"] >= 0.8
    assert out["flag"] is True
    ci_low = out["bootstrap"]["ci_low"]
    ci_high = out["bootstrap"]["ci_high"]
    assert ci_low <= out["slope_point"] <= ci_high


def test_severity_weights():
    out = severity_score(True, True, True)
    assert out["severity"] == 6
    assert out["weights"] == {"downward": 2, "sudden": 3, "unstable": 1}
