"""
KT-Radar :: perspective / metric transform
-------------------------------------------
CLAUDE.md constraint 1: speed is never computed in pixel space. Pixel
displacement per frame is meaningless under perspective - a vehicle far
from the camera covers fewer pixels per second than the same vehicle
close to the camera at the same real speed. Every speed calculation
goes through this homography to a metric ground plane FIRST.
"""

from __future__ import annotations

from typing import List

import cv2
import numpy as np


class GroundPlane:
    """Maps image pixels -> bird's-eye metric coordinates via homography."""

    def __init__(self, source_polygon: List[List[int]], target_size_m: List[float]):
        src = np.array(source_polygon, dtype=np.float32)
        w, h = float(target_size_m[0]), float(target_size_m[1])
        dst = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
        self.M = cv2.getPerspectiveTransform(src, dst)
        self.area_m2 = w * h
        self.width_m, self.length_m = w, h

    def to_metric(self, pts_xy: np.ndarray) -> np.ndarray:
        if pts_xy.size == 0:
            return pts_xy.reshape(-1, 2)
        p = pts_xy.astype(np.float32).reshape(-1, 1, 2)
        return cv2.perspectiveTransform(p, self.M).reshape(-1, 2)
