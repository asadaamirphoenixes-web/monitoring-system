"""
KT-Radar :: synthetic placeholder clip generator
----------------------------------------------------
Deterministic - no randomness - so it always matches
tests/eval/ground_truth/synth_day_normal_01.json. The clip video itself is
gitignored (too large for git even at this tiny size, by policy), so this
regenerates it on demand.

The synthetic vehicle is a single 40x40 px box, drifting downward at a
known constant pixel velocity, crossing config/site_example.yaml's
"cordon" line (a horizontal line at y=50 across a 100x100 px patch that
maps to a 10x10 m ground plane, i.e. 1 px = 0.1 m) exactly once:

    step        = 3 px/frame vertically, at 15 fps
    real speed  = 3 px * 0.1 m/px * 15 fps * 3.6 = 16.2 km/h
    anchor (bottom-center) crosses y=50 around frame 30 (t ~ 2.07s)
    sv.LineZone's count event fires later, on the clip's last processed
      frame (t = 3.00s) - by default it requires all 4 box corners past
      the line, not just the bottom-center anchor, so the box's top
      edge (40 px above the anchor) has to clear y=50 too before the
      crossing "counts". See tests/eval/ground_truth/synth_day_normal_01.json's
      track_hint (t=00:03), which reflects the actual emitted event, not
      the anchor-crossing frame.

(step=3 px against a 40 px box is also the largest step that keeps
ByteTrack's IOU-based matching above its threshold and the track ID
stable across frames - see tests/fixtures/detections.py. The same
40 px box height is what makes the corner-vs-anchor crossing delay
above unusually large here - a realistically-sized vehicle box would
show only a frame or two of delay, not 14.)

The video frames themselves are blank placeholders (a frame-index
overlay for sanity, nothing else) - detection comes entirely from the
sidecar JSON below, exactly as it would for a human's hand-annotated
real footage, so the visual content of this clip doesn't need to
resemble the annotated boxes at all.

Run:
    python -m tests.eval.clips.generate_placeholder_clip
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import supervision as sv

from tests.eval.detections_io import save_detections_script

FPS = 15
N_FRAMES = 45
FRAME_SIZE = (100, 100)   # matches config/site_example.yaml's 100x100 px patch
BOX_SIZE = 40.0
STEP_PX = 3.0
X_FIXED = 10.0            # box x is constant -> anchor_x = 30 px = 3.0 m, inside the patch
Y_START = -80.0           # anchor_y = y + BOX_SIZE starts at -40 px (above the patch),
                           # crosses the cordon line (y=50 px) at frame 30 = t=2.00s

CLIP_NAME = "synth_day_normal_01.mp4"
SIDECAR_NAME = "synth_day_normal_01.detections.json"


def build_script():
    names = {0: "car"}
    frames = []
    for i in range(N_FRAMES):
        y = Y_START + STEP_PX * i
        box = np.array([[X_FIXED, y, X_FIXED + BOX_SIZE, y + BOX_SIZE]], dtype=np.float32)
        frames.append(sv.Detections(
            xyxy=box,
            confidence=np.array([0.90], dtype=np.float32),
            class_id=np.array([0]),
        ))
    return names, frames


def write_clip(path: Path) -> None:
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    w, h = FRAME_SIZE
    out = cv2.VideoWriter(str(path), fourcc, FPS, (w, h))
    for i in range(N_FRAMES):
        frame = np.full((h, w, 3), 32, dtype=np.uint8)
        cv2.putText(frame, str(i), (5, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1)
        out.write(frame)
    out.release()


def main() -> None:
    clips_dir = Path(__file__).parent
    clip_path = clips_dir / CLIP_NAME
    sidecar_path = clips_dir / SIDECAR_NAME

    write_clip(clip_path)
    names, frames = build_script()
    save_detections_script(sidecar_path, names, frames)

    print(f"wrote {clip_path} ({N_FRAMES} frames @ {FPS} fps)")
    print(f"wrote {sidecar_path}")


if __name__ == "__main__":
    main()
