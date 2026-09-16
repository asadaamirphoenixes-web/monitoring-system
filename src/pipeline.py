"""
KT-Radar :: core perception pipeline
------------------------------------
RTSP/file -> detection (src/detector.py) -> ByteTrack -> perspective speed
(src/homography.py, src/tracker.py) -> zone counting -> violation rule
engine (src/violations.py) -> event bus -> control-room API (src/api.py)

Designed for Karachi mixed traffic: motorcycle-dominant, rickshaws, Suzuki
pickups, water tankers, dumpers, donkey carts, pedestrians in carriageway.

Detection goes through the Detector interface (src/detector.py), not a
concrete YOLO call, so this module runs and is testable without a trained
model or GPU - see CLAUDE.md constraint 6.

Run:
    python -m src.pipeline --config config/site_shahrah_faisal.yaml
    python -m src.pipeline --config config/site_example.yaml \\
        --source tests/fixtures/some_clip.mp4 --mock-detector
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Dict, Optional, cast

import cv2
import numpy as np
import supervision as sv

from .anpr import PlateReader
from .config import SiteConfig
from .detector import Detector, MockDetector, YoloDetector, validate_vehicle_classes
from .homography import GroundPlane
from .privacy import FaceBlur, plate_token, release_plate_with_authorization
from .tracker import TrackState
from .violations import RuleEngine, promote

# --------------------------------------------------------------------------
# Pipeline
# --------------------------------------------------------------------------

# supervision types class_id/confidence/tracker_id as Optional[np.ndarray]
# on Detections in general (an empty/uninitialised set has none of them),
# but by the time we read them here they've always come from a Detector
# (which sets class_id/confidence) or ByteTrack.update_with_detections
# (which sets tracker_id) - never from an empty Detections. These narrow
# that contract in one place instead of asserting it at every call site.
def _class_ids(det: sv.Detections) -> np.ndarray:
    assert det.class_id is not None
    return det.class_id


def _confidences(det: sv.Detections) -> np.ndarray:
    assert det.confidence is not None
    return det.confidence


def _tracker_ids(det: sv.Detections) -> np.ndarray:
    assert det.tracker_id is not None
    return det.tracker_id


class TrafficRadar:
    def __init__(
        self,
        cfg: SiteConfig,
        on_event=None,
        signal_state_fn=None,
        rider_model=None,
        detector: Optional[Detector] = None,
    ):
        self.cfg = cfg
        if detector is None:
            # "the detector loader": only this default-construction path
            # (from cfg.weights) hard-fails on a custom-taxonomy mismatch -
            # an explicitly-passed detector (MockDetector in tests, or a
            # caller's own choice) is never hard-failed, only warned about
            # below via validate_vehicle_classes.
            detector = YoloDetector(
                cfg.weights, imgsz=cfg.imgsz, conf=cfg.conf, iou=cfg.iou, device=cfg.device
            )
            if cfg.class_source == "custom":
                missing = sorted(set(cfg.vehicle_classes) - set(detector.names.values()))
                if missing:
                    raise RuntimeError(
                        f"site '{cfg.site_id}' has class_source: custom but its weights "
                        f"({cfg.weights}) don't know these expected vehicle_classes: "
                        f"{missing} - fix the weights file or vehicle_classes in the "
                        "site config."
                    )
        self.detector = detector
        validate_vehicle_classes(cfg, self.detector)

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
            from ultralytics import YOLO  # lazy: only needed if this feature is enabled

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
        if display:
            cv2.destroyAllWindows()

    # ---------------- per-frame ----------------

    def process(self, frame: np.ndarray, ts: float) -> np.ndarray:
        det = self.detector.predict(frame)
        if len(det):
            class_names = [self.detector.names[c] for c in _class_ids(det)]
            det = cast(sv.Detections, det[np.isin(class_names, self.cfg.vehicle_classes)])
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
            tid = int(_tracker_ids(det)[i])
            cls = self.detector.names[int(_class_ids(det)[i])]
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
                score = crop.shape[0] * crop.shape[1] * float(_confidences(det)[i])
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
                    tid = int(_tracker_ids(det)[i])
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
                            # journey/re-ID token only - a crossing is never
                            # an enforcement action, so raw plate never applies
                            "plate": self._token_for_track(st),
                            "ts": time.time(),
                        })

    def _token_for_track(self, st: TrackState) -> Optional[str]:
        if not st.plate:
            return None
        try:
            return plate_token(st.plate)
        except RuntimeError:
            return None

    def _emit_violations(self, det: sv.Detections, frame: np.ndarray, ctx: Dict):
        for i in range(len(det)):
            tid = int(_tracker_ids(det)[i])
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
        release_plate_with_authorization() (gated + audited); otherwise an
        HMAC token, or nothing at all if no key is configured."""
        if not st.plate:
            return None
        if c.enforceable:
            event_id = f"{self.cfg.site_id}:{st.track_id}:{c.rule}:{int(c.ts * 1000)}"
            released = release_plate_with_authorization(
                st.plate, event_id=event_id, reason=c.rule, actor=self.cfg.site_id
            )
            if released is not None:
                return released
        return self._token_for_track(st)

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
        if density < 0.02 and mean_v > 40:
            return "A"
        if density < 0.04 and mean_v > 30:
            return "B"
        if density < 0.06 and mean_v > 22:
            return "C"
        if density < 0.09 and mean_v > 15:
            return "D"
        if density < 0.13:
            return "E"
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
            tid = int(_tracker_ids(det)[i])
            st = self.tracks.get(tid)
            sp = st.speed_kmh() if st else None
            flags = "!" + ",".join(sorted(st.flags)) if st and st.flags else ""
            labels.append(
                f"#{tid} {self.detector.names[int(_class_ids(det)[i])]}"
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
    ap.add_argument("--mock-detector", action="store_true",
                     help="use MockDetector instead of real YOLO weights - no model/GPU needed")
    a = ap.parse_args()
    cfg = SiteConfig.load(a.config)

    detector = None
    if a.mock_detector:
        names = {i: c for i, c in enumerate(sorted(cfg.vehicle_classes))}
        detector = MockDetector(names=names)

    TrafficRadar(cfg, detector=detector).run(a.source, a.display)


if __name__ == "__main__":
    main()
