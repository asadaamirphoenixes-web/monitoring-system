"""Result dataclasses shared between run_eval.py (computes them) and
report.py (renders them)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class CountResult:
    line: str
    cls_name: str
    manual: int
    predicted: int
    abs_error: int
    pct_error: Optional[float]   # None when manual == 0 (can't take a percentage of zero)
    gate_fail: bool


@dataclass
class SpeedCheckResult:
    description: str
    approx_speed_kmh: float
    tolerance_kmh: float
    hint_time_s: Optional[float]
    measured_speed_kmh: Optional[float]
    matched_event_clip_ts: Optional[float]
    status: str                  # "pass" | "fail" | "not measured"
    reason: Optional[str] = None


@dataclass
class WholeClipEstimateResult:
    """A rough range check, NOT a precise measurement - see
    GroundTruth.manual_whole_clip_estimate. Never gates the run."""
    cls_name: str
    predicted_total: int
    range_min: int
    range_max: int
    status: str   # "within range" | "outside range"


@dataclass
class ClipResult:
    clip_name: str
    condition: str
    detector_mode: str
    notes: str
    count_results: List[CountResult] = field(default_factory=list)
    speed_results: List[SpeedCheckResult] = field(default_factory=list)
    fragmentation: Dict[str, Optional[float]] = field(default_factory=dict)
    whole_clip_estimate_results: List[WholeClipEstimateResult] = field(default_factory=list)
    skipped_checks: List[str] = field(default_factory=list)
    gate_failures: List[str] = field(default_factory=list)
    speed_mae: Optional[float] = None
