"""Synthetic detection sequences - no trained model or GPU needed to drive
the pipeline through detector -> tracker -> homography -> speed."""

from __future__ import annotations

from typing import List, Tuple

import numpy as np
import supervision as sv


def moving_box_sequence(
    start_xy: Tuple[float, float],
    velocity_xy_per_frame: Tuple[float, float],
    n_frames: int,
    box_size: float = 40.0,
    class_id: int = 0,
    confidence: float = 0.9,
) -> List[sv.Detections]:
    """One sv.Detections per frame: a single box moving at a constant pixel
    velocity. Keep step size small relative to box_size (roughly <= 10%) or
    ByteTrack's IOU matching will lose the track between frames."""
    x, y = start_xy
    vx, vy = velocity_xy_per_frame
    seq = []
    for _ in range(n_frames):
        box = np.array([[x, y, x + box_size, y + box_size]], dtype=np.float32)
        seq.append(sv.Detections(
            xyxy=box,
            confidence=np.array([confidence], dtype=np.float32),
            class_id=np.array([class_id]),
        ))
        x += vx
        y += vy
    return seq
