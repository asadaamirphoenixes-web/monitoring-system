"""
KT-Radar :: core perception pipeline
------------------------------------
RTSP/file -> YOLO detection -> ByteTrack -> perspective speed -> zone counting
                                         -> violation rule engine -> event bus

Designed for Karachi mixed traffic: motorcycle-dominant, rickshaws, Suzuki
pickups, water tankers, dumpers, donkey carts, pedestrians in carriageway.

Run:
    python -m src.pipeline --config config/site_shahrah_faisal.yaml
"""

from __future__ import annotations

import argparse
import json
import math
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Deque, Dict, List, Optional, Tuple

import cv2
import numpy as np
import yaml

# Ultralytics >= 8.3 (YOLO11 / YOLO26) and supervision >= 0.25
from ultralytics import YOLO
import supervision as sv

from .violations import RuleEngine, promote
from .anpr import PlateReader
from .privacy import FaceBlur, plate_token, release_plate


# --------------------------------------------------------------------------
# Config
# --------------------------------------------------------------------------

@dataclass
class SiteConfig:
    site_id: str
    name: str
    rtsp_url: str
    fps_target: int = 15
    imgsz: int = 960
    conf: float = 0.30
    iou: float = 0.55
    device: str = "cuda:0"
    weights: str = "models/ktr_vehicles_yolo11s.pt"

    # 4 points in the image (clockwise from top-left of the road patch)
    source_polygon: List[List[int]] = field(default_factory=list)
    # real-world size of that patch in metres: [width_m, length_m]
    target_size_m: List[float] = field(default_factory=lambda: [12.0, 45.0])

    # counting lines: {"name": [[x1,y1],[x2,y2]]}
    count_lines: Dict[str, List[List[int]]] = field(default_factory=dict)
    # lane polygons: {"lane_1": [[x,y], ...]}
    lanes: Dict[str, List[List[int]]] = field(default_factory=dict)
    # stop line for red-light running
    stop_line: Optional[List[List[int]]] = None
    # direction of legal travel as a unit vector in target (metric) space
    legal_heading: List[float] = field(default_factory=lambda: [0.0, -1.0])

    speed_limit_kmh: float = 60.0
    min_track_len: int = 8          # frames before speed is trusted
    congestion_density_thr: float = 0.09   # vehicles per m^2
    congestion_speed_thr: float = 12.0     # km/h

    # violation rule engine (see src/violations.py)
    restricted_lanes: Dict[str, List[str]] = field(default_factory=dict)  # {"lane_bus": ["bus"]}
    no_uturn_zone: Optional[List[List[int]]] = None
    calib_quality: float = 1.0      # 0-1, degrades R1 confidence if homography is shaky
    violation_min_conf: float = 0.80
    rider_model_weights: Optional[str] = None  # helmet/occupant classifier for R2/R3

    # ANPR + privacy (see src/anpr.py, src/privacy.py)
    plate_weights: Optional[str] = None        # plate detector; unset disables ANPR
    face_blur_weights: Optional[str] = "models/face_yolov8n.pt"  # unset disables face blur

    @staticmethod
    def load(path: str | Path) -> "SiteConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return SiteConfig(**raw)


# --------------------------------------------------------------------------
# Perspective / metric transform  (the thing most repos get wrong)
# --------------------------------------------------------------------------

class GroundPlane:
    """Maps image pixels -> bird's-eye metric coordinates via homography.

    Speed computed in pixel space is meaningless: a bike 200 px away moves
    far fewer pixels per second than the same bike 40 px away. We rectify to
    a metric plane first, then differentiate.
    """

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


# --------------------------------------------------------------------------
# Track state
# --------------------------------------------------------------------------

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


# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

VEHICLE_CLASSES = {
    "motorcycle", "car", "rickshaw", "bus", "truck", "tanker",
    "minibus", "pickup", "tractor", "bicycle", "cart",
}


class TrafficRadar:
    def __init__(self, cfg: SiteConfig, on_event=None, signal_state_fn=None, rider_model=None):
        self.cfg = cfg
        self.model = YOLO(cfg.weights)
        self.tracker = sv.ByteTrack(
            track_activation_threshold=cfg.conf,
            lost_track_buffer=45,
            minimum_matching_threshold=0.85,
            frame_rate=cfg.fps_target,
        )
        self.plane = GroundPlane(cfg.source_polygon, cfg.target_size_m)
        self.tracks: Dict[int, TrackState] = {}
        self.on_event = on_event or (lambda e: print(json.dumps(e)))
        self.frame_idx = 0
        self.congested = False

        if rider_model is None and cfg.rider_model_weights:
            rider_model = YOLO(cfg.rider_model_weights)
        self.rules = RuleEngine(cfg, signal_state_fn=signal_state_fn, rider_model=rider_model)

        self.plates = PlateReader(cfg.plate_weights) if cfg.plate_weights else None
        self.face_blur = FaceBlur(cfg.face_blur_weights) if cfg.face_blur_weights else None

        self.lines = {
            name: sv.LineZone(
                start=sv.Point(*pts[0]), end=sv.Point(*pts[1])
            ) for name, pts in cfg.count_lines.items()
        }
        self.lane_zones = {
            name: sv.PolygonZone(polygon=np.array(poly))
            for name, poly in cfg.lanes.items()
        }
        self.legal = np.array(cfg.legal_heading, dtype=np.float32)
        self.legal /= (np.linalg.norm(self.legal) + 1e-9)

    # ---------------- main loop ----------------

    def run(self, source: Optional[str] = None, display: bool = False):
        src = source or self.cfg.rtsp_url
        cap = cv2.VideoCapture(src)
        if not cap.isOpened():
            raise RuntimeError(f"cannot open stream: {src}")
        native_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
        stride = max(1, int(round(native_fps / self.cfg.fps_target)))

        t0 = time.time()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            self.frame_idx += 1
            if self.frame_idx % stride:
                continue
            ts = time.time() - t0
            annotated = self.process(frame, ts)
            if display:
                cv2.imshow(self.cfg.site_id, annotated)
                if cv2.waitKey(1) & 0xFF == 27:
                    break
        cap.release()
        cv2.destroyAllWindows()

    # ---------------- per-frame ----------------

    def process(self, frame: np.ndarray, ts: float) -> np.ndarray:
        res = self.model.predict(
            frame, imgsz=self.cfg.imgsz, conf=self.cfg.conf,
            iou=self.cfg.iou, device=self.cfg.device, verbose=False,
        )[0]
        det = sv.Detections.from_ultralytics(res)
        det = det[np.isin(
            [self.model.names[c] for c in det.class_id], list(VEHICLE_CLASSES)
        )] if len(det) else det
        det = self.tracker.update_with_detections(det)

        # blur faces before this frame is used for any stored crop, evidence,
        # or display - detection above already ran on the unblurred frame
        frame_pub = self.face_blur(frame) if self.face_blur else frame

        if len(det):
            # anchor = bottom-centre of box == tyre contact patch on the road
            anchors = det.get_anchors_coordinates(sv.Position.BOTTOM_CENTER)
            metric = self.plane.to_metric(anchors)
        else:
            metric = np.empty((0, 2))

        live_speeds, in_patch = [], 0

        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            cls = self.model.names[int(det.class_id[i])]
            st = self.tracks.get(tid) or TrackState(tid, cls)
            st.last_seen = ts
            mx, my = float(metric[i][0]), float(metric[i][1])
            inside = (0 <= mx <= self.plane.width_m) and (0 <= my <= self.plane.length_m)
            if inside:
                st.metric_hist.append((ts, mx, my))
                in_patch += 1
            st.boxes.append((ts, det.xyxy[i].copy()))

            # keep the sharpest, largest crop for evidence, and feed every
            # frame's crop to ANPR - temporal voting needs one read per frame,
            # not repeated reads of a single cached crop
            x1, y1, x2, y2 = det.xyxy[i].astype(int)
            crop = frame_pub[max(0, y1):y2, max(0, x1):x2]
            if crop.size:
                score = crop.shape[0] * crop.shape[1] * float(det.confidence[i])
                if score > st.best_crop_score:
                    st.best_crop_score, st.best_crop = score, crop.copy()
                if self.plates is not None:
                    st.plate, st.plate_conf = self.plates.accumulate(
                        tid, self.plates.read(crop)
                    )

            self.tracks[tid] = st
            sp = st.speed_kmh()
            if sp is not None and len(st.metric_hist) >= self.cfg.min_track_len:
                live_speeds.append(sp)

        density = in_patch / max(self.plane.area_m2, 1.0)
        mean_v = float(np.mean(live_speeds)) if live_speeds else 0.0
        self.congested = bool(
            density > self.cfg.congestion_density_thr
            and mean_v < self.cfg.congestion_speed_thr
        )
        ctx = {
            "calib_quality": self.cfg.calib_quality,
            "restricted_lanes": self.cfg.restricted_lanes,
            "no_uturn_zone": self.cfg.no_uturn_zone,
            "congested": self.congested,
        }
        self._emit_violations(det, frame, ctx)
        self._update_counts(det)
        self._emit_congestion(ts, in_patch, density, mean_v)
        self._gc(ts)

        return self._annotate(frame_pub, det, live_speeds)

    # ---------------- analytics ----------------

    def _update_counts(self, det: sv.Detections):
        for name, lz in self.lines.items():
            if len(det):
                crossed_in, crossed_out = lz.trigger(det)
                for i, flag in enumerate(crossed_in | crossed_out):
                    if not flag:
                        continue
                    tid = int(det.tracker_id[i])
                    key = f"line:{name}"
                    st = self.tracks.get(tid)
                    if st and key not in st.lines_crossed:
                        st.lines_crossed.add(key)
                        self.on_event({
                            "type": "count",
                            "site": self.cfg.site_id,
                            "line": name,
                            "track_id": tid,
                            "class": st.cls_name,
                            "speed_kmh": st.speed_kmh(),
                            "ts": time.time(),
                        })

    def _emit_violations(self, det: sv.Detections, frame: np.ndarray, ctx: Dict):
        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            st = self.tracks.get(tid)
            if st is None:
                continue
            cands = self.rules.evaluate(st, frame, ctx)
            if not cands:
                continue
            promote(cands, st, min_conf=self.cfg.violation_min_conf)
            for c in cands:
                self.on_event({
                    "type": "violation",
                    "site": self.cfg.site_id,
                    "rule": c.rule,
                    "track_id": c.track_id,
                    "class": c.cls_name,
                    "confidence": round(c.confidence, 3),
                    "enforceable": c.enforceable,
                    "plate": self._plate_for_event(st, c),
                    "detail": c.detail,
                    "ts": c.ts,
                })

    def _plate_for_event(self, st: TrackState, c) -> Optional[str]:
        """Never leak a raw plate into an event. Clear text only via
        release_plate() (gated + audited); otherwise an HMAC token, or
        nothing at all if no key is configured."""
        if not st.plate:
            return None
        if c.enforceable:
            event_id = f"{self.cfg.site_id}:{st.track_id}:{c.rule}:{int(c.ts * 1000)}"
            released = release_plate(st.plate, event_id=event_id, reason=c.rule, actor=self.cfg.site_id)
            if released is not None:
                return released
        try:
            return plate_token(st.plate)
        except RuntimeError:
            return None

    def _emit_congestion(self, ts: float, in_patch: int, density: float, mean_v: float):
        if self.frame_idx % (self.cfg.fps_target * 5):   # every ~5 s
            return
        los = self._level_of_service(density, mean_v)
        self.on_event({
            "type": "state",
            "site": self.cfg.site_id,
            "ts": time.time(),
            "vehicles_in_zone": in_patch,
            "density_veh_per_m2": round(density, 4),
            "mean_speed_kmh": round(mean_v, 1),
            "los": los,
            "congested": self.congested,
        })

    @staticmethod
    def _level_of_service(density: float, mean_v: float) -> str:
        if density < 0.02 and mean_v > 40: return "A"
        if density < 0.04 and mean_v > 30: return "B"
        if density < 0.06 and mean_v > 22: return "C"
        if density < 0.09 and mean_v > 15: return "D"
        if density < 0.13: return "E"
        return "F"

    def _gc(self, ts: float, ttl: float = 6.0):
        dead = [k for k, v in self.tracks.items() if ts - v.last_seen > ttl]
        for k in dead:
            self.tracks.pop(k, None)
            if self.plates is not None:
                self.plates.drop(k)

    # ---------------- overlay ----------------

    def _annotate(self, frame, det, speeds) -> np.ndarray:
        box = sv.BoxAnnotator(thickness=2)
        lab = sv.LabelAnnotator(text_scale=0.4, text_thickness=1)
        labels = []
        for i in range(len(det)):
            tid = int(det.tracker_id[i])
            st = self.tracks.get(tid)
            sp = st.speed_kmh() if st else None
            flags = "!" + ",".join(sorted(st.flags)) if st and st.flags else ""
            labels.append(
                f"#{tid} {self.model.names[int(det.class_id[i])]}"
                + (f" {sp:.0f}km/h" if sp else "") + flags
            )
        out = box.annotate(frame.copy(), det)
        out = lab.annotate(out, det, labels)
        cv2.polylines(out, [np.array(self.cfg.source_polygon)], True, (0, 255, 255), 2)
        for pts in self.cfg.count_lines.values():
            cv2.line(out, tuple(pts[0]), tuple(pts[1]), (255, 0, 255), 2)
        cv2.putText(out, f"n={len(det)}  v={np.mean(speeds) if speeds else 0:.0f} km/h",
                    (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", default=None, help="override rtsp with a video file")
    ap.add_argument("--display", action="store_true")
    a = ap.parse_args()
    cfg = SiteConfig.load(a.config)
    TrafficRadar(cfg).run(a.source, a.display)


if __name__ == "__main__":
    main()
