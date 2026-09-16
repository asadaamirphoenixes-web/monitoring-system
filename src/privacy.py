"""
KT-Radar :: privacy layer
-------------------------
This module is not decoration. Pakistan's Personal Data Protection Bill has
been in draft for years, PECA 2016 governs the digital side, and there is no
settled jurisprudence on municipal video analytics. That legal vacuum is a
liability, not a licence: a student/civic system that scrapes identities from
public CCTV is the fastest way to get shut down or sued.

Design stance:
  * The system is VEHICLE-centric. No facial recognition. Faces are blurred
    on write, always.
  * Plates are stored as a keyed hash (HMAC-SHA256) by default. The raw plate
    exists in RAM only, and is written in clear ONLY when an event has been
    promoted to `enforceable` AND the deployment is operating under a written
    authority from the relevant traffic authority.
  * Raw video is retained 72 h then deleted. Only derived counts and
    hashed-plate events survive past that.
  * Every clear-text plate read is written to an append-only audit log.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

_KEY = os.environ.get("KTR_PLATE_HMAC_KEY", "").encode() or None
AUDIT = Path(os.environ.get("KTR_AUDIT_LOG", "logs/plate_access.jsonl"))
ENFORCEMENT_AUTHORISED = os.environ.get("KTR_ENFORCEMENT_AUTHORISED", "0") == "1"


def plate_token(plate: str) -> str:
    """Stable pseudonym. Lets you do repeat-offender and journey-time analytics
    without ever storing who the vehicle belongs to."""
    if _KEY is None:
        raise RuntimeError("set KTR_PLATE_HMAC_KEY before storing any plate data")
    return hmac.new(_KEY, plate.encode(), hashlib.sha256).hexdigest()[:24]


def release_plate_with_authorization(
    plate: str, event_id: str, reason: str, actor: str
) -> Optional[str]:
    """Return the clear-text plate only under authority, and log it."""
    if not ENFORCEMENT_AUTHORISED:
        return None
    AUDIT.parent.mkdir(parents=True, exist_ok=True)
    with AUDIT.open("a") as f:
        f.write(json.dumps({
            "ts": time.time(), "event_id": event_id,
            "actor": actor, "reason": reason,
            "plate_token": plate_token(plate),
        }) + "\n")
    return plate


class FaceBlur:
    """Blur faces and pedestrians' heads before any frame is persisted."""

    def __init__(self, weights: str = "models/face_yolov8n.pt", conf: float = 0.25):
        from ultralytics import YOLO
        self.m = YOLO(weights)
        self.conf = conf

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        out = frame.copy()
        r = self.m.predict(frame, conf=self.conf, verbose=False)[0]
        for b in r.boxes.xyxy.cpu().numpy().astype(int):
            x1, y1, x2, y2 = b
            roi = out[max(0, y1):y2, max(0, x1):x2]
            if roi.size == 0:
                continue
            k = max(9, (int(min(roi.shape[:2]) / 2) * 2) + 1)
            out[max(0, y1):y2, max(0, x1):x2] = cv2.GaussianBlur(roi, (k, k), 0)
        return out


def redact_evidence(frame: np.ndarray, plate_box, blur: FaceBlur) -> np.ndarray:
    """Evidence clip: faces blurred, plate left readable (it is the evidence)."""
    out = blur(frame)
    if plate_box is not None:
        x1, y1, x2, y2 = [int(v) for v in plate_box]
        out[y1:y2, x1:x2] = frame[y1:y2, x1:x2]
    return out


def purge_expired(video_dir: str = "data/clips", hours: int = 72) -> int:
    cutoff = time.time() - hours * 3600
    n = 0
    for p in Path(video_dir).glob("**/*"):
        if p.is_file() and p.stat().st_mtime < cutoff:
            p.unlink()
            n += 1
    return n
