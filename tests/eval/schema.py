"""
KT-Radar :: eval ground truth schema
--------------------------------------
One JSON file per clip, hand-filled by a human - see tests/eval/README.md
for the authoring guide. Every field except clip/condition/site_config is
optional. Missing fields are skipped, never guessed or defaulted to a
plausible-looking number (a clip with no known_speed_checks must report
speed as "not measured", not 0 or an interpolated value).
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional


@dataclass
class SpeedCheck:
    track_hint: str
    approx_speed_kmh: float
    tolerance_kmh: float
    source: str = ""
    # parsed from a "t=MM:SS" or "t=SS[.s]" pattern embedded in track_hint,
    # e.g. "silver sedan entering frame at t=00:14" - the convention the
    # brief's own example uses. None if track_hint doesn't contain one.
    hint_time_s: Optional[float] = None

    @staticmethod
    def parse_hint_time(track_hint: str) -> Optional[float]:
        m = re.search(r"t=(\d+):(\d+(?:\.\d+)?)", track_hint)
        if m:
            return int(m.group(1)) * 60 + float(m.group(2))
        m = re.search(r"t=(\d+(?:\.\d+)?)\b", track_hint)
        if m:
            return float(m.group(1))
        return None


@dataclass
class GroundTruth:
    clip: str
    condition: str
    site_config: str
    manual_line_counts: Dict[str, Dict[str, int]] = field(default_factory=dict)
    known_speed_checks: List[SpeedCheck] = field(default_factory=list)
    # A weaker, whole-clip-total estimate - {class: [min, max]} - for when
    # nobody has done a careful per-line frame-by-frame count, just a rough
    # sanity check from watching the clip once (e.g. sparse sampling, no
    # line-crossing granularity). Deliberately a SEPARATE field from
    # manual_line_counts, never merged into it: forcing a rough whole-clip
    # guess into the precise per-line-crossing MAE metric would misrepresent
    # what it actually is. See report.py - this is rendered as its own
    # low-confidence range check, never with the same weight as a real
    # count MAE, and it never participates in the pass/fail gate.
    manual_whole_clip_estimate: Dict[str, List[int]] = field(default_factory=dict)
    notes: str = ""
    source_path: Optional[Path] = None

    @staticmethod
    def load(path: Path) -> "GroundTruth":
        raw = json.loads(path.read_text())
        checks = []
        for c in raw.get("known_speed_checks", []):
            track_hint = c.get("track_hint", "")
            checks.append(SpeedCheck(
                track_hint=track_hint,
                approx_speed_kmh=float(c["approx_speed_kmh"]),
                tolerance_kmh=float(c["tolerance_kmh"]),
                source=c.get("source", ""),
                hint_time_s=SpeedCheck.parse_hint_time(track_hint),
            ))
        return GroundTruth(
            clip=raw["clip"],
            condition=raw.get("condition", "unknown"),
            site_config=raw["site_config"],
            manual_line_counts=raw.get("manual_line_counts", {}),
            known_speed_checks=checks,
            manual_whole_clip_estimate=raw.get("manual_whole_clip_estimate", {}),
            notes=raw.get("notes", ""),
            source_path=path,
        )
