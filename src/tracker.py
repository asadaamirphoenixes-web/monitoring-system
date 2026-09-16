"""
KT-Radar :: per-track state
-----------------------------
CLAUDE.md constraint 2: speed must use a robust fit, not a two-point
difference. Detection jitter makes frame-to-frame speed noisy, so
speed_kmh() does a least-squares linear fit over a rolling window of
metric-space history rather than (pos[-1] - pos[-2]) / dt.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import Deque, Optional, Tuple

import numpy as np


@dataclass
class TrackState:
    track_id: int
    cls_name: str
    metric_hist: Deque[Tuple[float, float, float]] = field(
        default_factory=lambda: deque(maxlen=45)
    )  # (t_seconds, x_m, y_m)
    boxes: Deque[Tuple[float, np.ndarray]] = field(default_factory=lambda: deque(maxlen=45))
    lines_crossed: set = field(default_factory=set)
    last_seen: float = 0.0
    plate: Optional[str] = None
    plate_conf: float = 0.0
    best_crop: Optional[np.ndarray] = None
    best_crop_score: float = 0.0
    flags: set = field(default_factory=set)

    def speed_kmh(self, window: float = 1.0) -> Optional[float]:
        """Least-squares speed over the last `window` seconds. Robust to jitter."""
        if len(self.metric_hist) < 4:
            return None
        t_end = self.metric_hist[-1][0]
        pts = [p for p in self.metric_hist if t_end - p[0] <= window]
        if len(pts) < 4:
            pts = list(self.metric_hist)[-4:]
        t = np.array([p[0] for p in pts])
        x = np.array([p[1] for p in pts])
        y = np.array([p[2] for p in pts])
        if t[-1] - t[0] < 1e-3:
            return None
        vx = np.polyfit(t, x, 1)[0]
        vy = np.polyfit(t, y, 1)[0]
        return float(math.hypot(vx, vy) * 3.6)

    def heading(self) -> Optional[np.ndarray]:
        if len(self.metric_hist) < 6:
            return None
        x0, y0 = self.metric_hist[0][1], self.metric_hist[0][2]
        x1, y1 = self.metric_hist[-1][1], self.metric_hist[-1][2]
        v = np.array([x1 - x0, y1 - y0], dtype=np.float32)
        n = np.linalg.norm(v)
        if n < 1.5:      # less than 1.5 m travelled: heading is noise
            return None
        return v / n
