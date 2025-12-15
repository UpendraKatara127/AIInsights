from __future__ import annotations

from ai_insights_agent.tools.rank import rank_systems


def _mk(system_id: str, sev: int, drop: float, p_below: float, slope: float, unstable_ratio: float):
    return {
        "system_id": system_id,
        "flags": {
            "severity": {"severity": sev, "weights": {"downward": 2, "sudden": 3, "unstable": 1}},
            "sudden": {"drop_amount": drop},
            "downward": {"slope_point": slope, "bootstrap": {"p_below_threshold": p_below}},
            "unstable": {"unstable_ratio": unstable_ratio},
        },
    }


def test_ranking_tie_breaks():
    # tie-break order:
    # 1) severity desc
    # 2) sudden drop_amount desc
    # 3) p_below_threshold desc
    # 4) abs(slope_point) desc
    # 5) unstable_ratio desc
    a = _mk("A", sev=5, drop=3.0, p_below=0.9, slope=-0.2, unstable_ratio=0.1)
    b = _mk("B", sev=5, drop=3.0, p_below=0.8, slope=-0.9, unstable_ratio=0.9)
    c = _mk("C", sev=4, drop=10.0, p_below=1.0, slope=-10.0, unstable_ratio=1.0)
    ranked = rank_systems([b, c, a], top_k=3)["top"]
    assert [x["system_id"] for x in ranked] == ["A", "B", "C"]

