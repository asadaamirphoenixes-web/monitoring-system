"""
KT-Radar :: Pakistani ANPR
--------------------------
Two-stage: plate localisation (YOLO) -> text recognition, then a temporal
voting layer that beats single-frame OCR by a wide margin on real CCTV.

Why a custom module instead of an off-the-shelf ANPR SDK:
  * Sindh plates are heterogeneous - green government, white private, yellow
    commercial, old hand-painted, new reflective, Urdu-only vanity plates,
    Karachi "AAA-123" vs "ABC-1234" vs older "ABC-123" formats.
  * Bikes carry a small rear plate, often bent, often covered by a flap.
  * Published Pakistani ANPR work reports ~99% mAP on plate *localisation*
    but only ~73% end-to-end accuracy on low-res frames - the gap is OCR.
    Temporal voting is where you close it.

Recogniser options, in order of practicality:
  1. PaddleOCR PP-OCRv5 with a fine-tuned rec head (best CPU/GPU tradeoff)
  2. A small CRNN/CTC trained on synthetic Pakistani plates + real crops
  3. TrOCR-small fine-tuned (heaviest, best on bent/blurred plates)
"""

from __future__ import annotations

import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import cv2
import numpy as np

# Sindh / Karachi civil formats seen in the wild
PLATE_PATTERNS = [
    re.compile(r"^[A-Z]{3}[- ]?\d{3}$"),     # ABC-123  (older Karachi)
    re.compile(r"^[A-Z]{3}[- ]?\d{4}$"),     # ABC-1234 (current)
    re.compile(r"^[A-Z]{2}[- ]?\d{4}$"),     # AB-1234
    re.compile(r"^[A-Z]{4}[- ]?\d{3}$"),     # KHI series
    re.compile(r"^\d{4}[- ]?[A-Z]{2,3}$"),   # commercial reversed
]

# OCR confusion pairs specific to embossed/reflective plates
CONFUSIONS = {"O": "0", "Q": "0", "I": "1", "L": "1", "S": "5",
              "Z": "2", "B": "8", "G": "6", "D": "0"}


def normalise(raw: str) -> Optional[str]:
    s = re.sub(r"[^A-Z0-9]", "", raw.upper())
    if not (5 <= len(s) <= 8):
        return None
    # split into alpha prefix + numeric suffix, fix confusions per segment
    m = re.match(r"^([A-Z0-9]{2,4})([A-Z0-9]{3,4})$", s)
    if not m:
        return None
    head, tail = m.groups()
    head = "".join(k for c in head for k, v in [(c, c)])   # keep letters as-is
    tail = "".join(CONFUSIONS.get(c, c) for c in tail)     # digits win in tail
    cand = f"{head}-{tail}"
    flat = cand.replace("-", "")
    for p in PLATE_PATTERNS:
        if p.match(flat) or p.match(cand):
            return cand
    return None


@dataclass
class PlateRead:
    text: str
    conf: float
    crop: np.ndarray


class PlateReader:
    def __init__(self, det_weights: str, ocr=None, min_plate_px: int = 26):
        from ultralytics import YOLO
        self.det = YOLO(det_weights)
        self.min_plate_px = min_plate_px
        if ocr is None:
            from paddleocr import PaddleOCR
            ocr = PaddleOCR(use_angle_cls=True, lang="en", show_log=False)
        self.ocr = ocr
        self.votes: Dict[int, Counter] = defaultdict(Counter)
        self.weight: Dict[int, Dict[str, float]] = defaultdict(dict)

    # ---------------- single crop ----------------

    def read(self, vehicle_crop: np.ndarray) -> Optional[PlateRead]:
        if vehicle_crop is None or vehicle_crop.size == 0:
            return None
        r = self.det.predict(vehicle_crop, conf=0.35, verbose=False)[0]
        if not len(r.boxes):
            return None
        # biggest plate box = closest / most readable
        areas = [(float((b[2]-b[0])*(b[3]-b[1])), i)
                 for i, b in enumerate(r.boxes.xyxy.cpu().numpy())]
        _, idx = max(areas)
        x1, y1, x2, y2 = r.boxes.xyxy.cpu().numpy()[idx].astype(int)
        if (x2 - x1) < self.min_plate_px:
            return None
        plate = vehicle_crop[max(0, y1):y2, max(0, x1):x2]
        plate = self._enhance(plate)
        res = self.ocr.ocr(plate, cls=True)
        if not res or not res[0]:
            return None
        # join the text lines top-to-bottom (two-line plates are common)
        lines = sorted(res[0], key=lambda x: x[0][0][1])
        raw = "".join(ln[1][0] for ln in lines)
        conf = float(np.mean([ln[1][1] for ln in lines]))
        txt = normalise(raw)
        if txt is None:
            return None
        return PlateRead(txt, conf * float(r.boxes.conf[idx]), plate)

    @staticmethod
    def _enhance(plate: np.ndarray) -> np.ndarray:
        h, w = plate.shape[:2]
        if h < 48:
            scale = 48 / h
            plate = cv2.resize(plate, (int(w*scale), 48), interpolation=cv2.INTER_CUBIC)
        g = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)
        g = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8)).apply(g)
        g = cv2.bilateralFilter(g, 7, 55, 55)
        return cv2.cvtColor(g, cv2.COLOR_GRAY2BGR)

    # ---------------- temporal voting ----------------

    def accumulate(self, track_id: int, pr: Optional[PlateRead]) -> Tuple[Optional[str], float]:
        """Confidence-weighted vote across every frame a vehicle is visible.

        Empirically this is the single biggest accuracy lever: a plate that
        OCRs correctly in 6 of 14 frames and differently in the rest still
        resolves correctly, because the correct read carries higher conf and
        clusters, while errors scatter.
        """
        if pr is not None:
            self.votes[track_id][pr.text] += 1
            w = self.weight[track_id]
            w[pr.text] = w.get(pr.text, 0.0) + pr.conf
        if not self.votes[track_id]:
            return None, 0.0
        w = self.weight[track_id]
        best = max(w, key=lambda k: w[k])
        total = sum(w.values()) or 1e-9
        agreement = w[best] / total
        n = self.votes[track_id][best]
        # require 2+ agreeing frames before ever returning a plate
        conf = agreement * min(1.0, n / 3.0)
        return (best, conf) if n >= 2 else (None, conf)

    def drop(self, track_id: int):
        self.votes.pop(track_id, None)
        self.weight.pop(track_id, None)
