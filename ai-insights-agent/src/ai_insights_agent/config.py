from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class BootstrapConfig:
    enabled: bool = True
    iterations: int = 2000
    confidence: float = 0.95
    seed: int = 42
    downtrend_confidence_min: float = 0.80


@dataclass(frozen=True)
class Thresholds:
    downward_slope_threshold: float = -0.15
    sudden_drop_percentile: int = 10
    minimum_sudden_drop_amount: float = 2.5
    unstable_z_threshold: float = 1.5
    unstable_ratio_threshold: float = 0.3


@dataclass(frozen=True)
class RankingConfig:
    max_systems_to_flag: int = 5
    top_devices_per_system: int = 3
    lookback_days: int = 20


@dataclass(frozen=True)
class Config:
    thresholds: Thresholds = Thresholds()
    bootstrap: BootstrapConfig = BootstrapConfig()
    ranking: RankingConfig = RankingConfig()

