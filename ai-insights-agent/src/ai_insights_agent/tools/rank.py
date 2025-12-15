from __future__ import annotations

from typing import Any, Dict, List


def _tie_break_key(result: Dict[str, Any]) -> tuple:
    sev = int(result["flags"]["severity"]["severity"])
    sudden_drop = result["flags"]["sudden"].get("drop_amount")
    sudden_drop = float(sudden_drop) if sudden_drop is not None else 0.0
    p_below = result["flags"]["downward"].get("bootstrap", {}).get("p_below_threshold")
    p_below = float(p_below) if p_below is not None else 0.0
    slope = result["flags"]["downward"].get("slope_point")
    slope = float(slope) if slope is not None else 0.0
    unstable_ratio = result["flags"]["unstable"].get("unstable_ratio")
    unstable_ratio = float(unstable_ratio) if unstable_ratio is not None else 0.0
    return (sev, sudden_drop, p_below, abs(slope), unstable_ratio)


def rank_systems(system_results: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    ranked = sorted(system_results, key=_tie_break_key, reverse=True)
    return {"top": ranked[: int(top_k)]}


def rank_devices(device_results: List[Dict[str, Any]], top_k: int) -> Dict[str, Any]:
    ranked = sorted(device_results, key=_tie_break_key, reverse=True)
    return {"top": ranked[: int(top_k)]}

