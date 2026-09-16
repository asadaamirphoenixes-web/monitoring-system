"""
KT-Radar :: hand-annotated detections sidecar
-----------------------------------------------
When a clip has no matching trained weights under models/*.pt, run_eval.py
replays a human's per-frame bounding-box annotations through MockDetector
instead (CLAUDE.md constraint 6). This is that sidecar file's I/O: one
JSON file named <clip_stem>.detections.json next to the clip.

Schema:
{
  "names": {"0": "car"},
  "frames": [
    [{"xyxy": [x1, y1, x2, y2], "class_id": 0, "confidence": 0.9}],
    []
  ]
}
`frames` has exactly one entry per video frame that will be processed
(after the pipeline's fps_target stride, same as a real detector would
see); an empty list means no detections in that frame.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import supervision as sv


def save_detections_script(
    path: Path, names: Dict[int, str], frames: List[sv.Detections]
) -> None:
    payload = {
        "names": {str(k): v for k, v in names.items()},
        "frames": [
            [
                {
                    "xyxy": [float(v) for v in box],
                    "class_id": int(cid),
                    "confidence": float(conf),
                }
                for box, cid, conf in zip(det.xyxy, det.class_id, det.confidence)
            ]
            for det in frames
        ],
    }
    path.write_text(json.dumps(payload, indent=2))


def load_detections_script(path: Path) -> Tuple[Dict[int, str], List[sv.Detections]]:
    raw = json.loads(path.read_text())
    names = {int(k): v for k, v in raw["names"].items()}
    script: List[sv.Detections] = []
    for frame_boxes in raw["frames"]:
        if not frame_boxes:
            script.append(sv.Detections.empty())
            continue
        script.append(sv.Detections(
            xyxy=np.array([b["xyxy"] for b in frame_boxes], dtype=np.float32),
            confidence=np.array([b["confidence"] for b in frame_boxes], dtype=np.float32),
            class_id=np.array([b["class_id"] for b in frame_boxes]),
        ))
    return names, script
