"""
KT-Radar :: detector interface
--------------------------------
CLAUDE.md constraint 6: the model does not exist yet. Every module that
depends on inference is written against this interface, not a concrete
YOLO call, so pipeline.py is testable today with MockDetector and swaps
to a real model later without changing anything downstream. YoloDetector
lazy-imports ultralytics so this module - and everything that only needs
the interface - stays importable without torch/a GPU present.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional

import numpy as np
import supervision as sv


class Detector(ABC):
    @property
    @abstractmethod
    def names(self) -> Dict[int, str]:
        """int class id -> class name, as produced by predict()."""

    @abstractmethod
    def predict(self, frame: np.ndarray) -> sv.Detections:
        """Run inference on one BGR frame, return detections in image pixel space."""


class YoloDetector(Detector):
    """Ultralytics YOLO (>= 8.3, YOLO11/YOLO26) behind the Detector interface."""

    def __init__(
        self,
        weights: str,
        imgsz: int = 960,
        conf: float = 0.30,
        iou: float = 0.55,
        device: str = "cuda:0",
    ):
        from ultralytics import YOLO  # lazy: no torch import until a real model is used

        self._model = YOLO(weights)
        self.imgsz, self.conf, self.iou, self.device = imgsz, conf, iou, device

    @property
    def names(self) -> Dict[int, str]:
        return self._model.names

    def predict(self, frame: np.ndarray) -> sv.Detections:
        res = self._model.predict(
            frame, imgsz=self.imgsz, conf=self.conf,
            iou=self.iou, device=self.device, verbose=False,
        )[0]
        return sv.Detections.from_ultralytics(res)


class MockDetector(Detector):
    """Fixture detections instead of real inference.

    Pass `script` (one sv.Detections per call, e.g. a synthetic track
    moving through frames) for sequence tests, or `detections` for a
    fixed return value every call. Defaults to an empty detection set.
    """

    def __init__(
        self,
        names: Dict[int, str],
        script: Optional[List[sv.Detections]] = None,
        detections: Optional[sv.Detections] = None,
    ):
        self._names = names
        self._script = list(script) if script is not None else None
        self._fixed = detections if detections is not None else sv.Detections.empty()
        self._i = 0

    @property
    def names(self) -> Dict[int, str]:
        return self._names

    def predict(self, frame: np.ndarray) -> sv.Detections:
        if self._script is not None:
            det = self._script[min(self._i, len(self._script) - 1)]
            self._i += 1
            return det
        return self._fixed
