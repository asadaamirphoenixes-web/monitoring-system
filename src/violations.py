"""
KT-Radar :: violation rule engine
---------------------------------
Every rule is a pure function over TrackState + context so it can be unit
tested against labelled clips. Nothing here fires a challan directly: rules
emit *candidate* events with a confidence, and only candidates above the
site threshold AND with a readable plate AND with 3 evidence frames get
promoted to `enforceable`. That two-stage design is what keeps false
challans (the thing that killed public trust in every South Asian ANPR
rollout) under control.

Karachi-specific rules included:
  R1  overspeed                       R7  wrong-way / opposing flow
  R2  helmet-less rider               R8  stopped vehicle / breakdown
  R3  triple riding (3+ on a bike)    R9  pedestrian in carriageway
  R4  red-light / stop-line jump      R10 illegal U-turn
  R5  lane-discipline (bus/bike lane) R11 encroachment (static object in lane)
  R6  no-plate / obscured plate       R12 overloaded / protruding load
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np


@dataclass
class Candidate:
    rule: str
    track_id: int
    cls_name: str
    confidence: float
    detail: Dict
    ts: float
    enforceable: bool = False


# ---------------------------------------------------------------- helpers

def _seg_side(p, a, b) -> float:
    return (b[0] - a[0]) * (p[1] - a[1]) - (b[1] - a[1]) * (p[0] - a[0])


def crossed_segment(prev, curr, a, b) -> bool:
    """Did the path prev->curr cross segment a->b?"""
    d1, d2 = _seg_side(prev, a, b), _seg_side(curr, a, b)
    d3, d4 = _seg_side(a, prev, curr), _seg_side(b, prev, curr)
    return (d1 * d2 < 0) and (d3 * d4 < 0)


def point_in_poly(p, poly) -> bool:
    x, y = p
    inside = False
    n = len(poly)
    for i in range(n):
        x1, y1 = poly[i]
        x2, y2 = poly[(i + 1) % n]
        if (y1 > y) != (y2 > y):
            xint = (x2 - x1) * (y - y1) / (y2 - y1 + 1e-9) + x1
            if x < xint:
                inside = not inside
    return inside


# ---------------------------------------------------------------- rules

class RuleEngine:
    def __init__(self, cfg, signal_state_fn=None, rider_model=None):
        """
        signal_state_fn() -> {"phase": "red"|"amber"|"green", "since": ts}
          Supply from the signal controller if you have a wire into it, or
          from a lamp-classifier CNN cropped on the signal head if you do not
          (Karachi signals are mostly not instrumented, so the CNN path is
          the realistic one).
        rider_model: optional YOLO pose/classifier for helmet + occupant count.
        """
        self.cfg = cfg
        self.signal_state_fn = signal_state_fn
        self.rider_model = rider_model
        self._stopped_since: Dict[int, float] = {}
        self._cooldown: Dict[tuple, float] = {}

    # ---- de-dupe: one candidate per (track, rule) per N seconds
    def _fresh(self, tid: int, rule: str, ttl: float = 20.0) -> bool:
        key = (tid, rule)
        now = time.time()
        if now - self._cooldown.get(key, 0) < ttl:
            return False
        self._cooldown[key] = now
        return True

    def evaluate(self, st, frame, ctx: Dict) -> List[Candidate]:
        out: List[Candidate] = []
        for fn in (self.r1_overspeed, self.r2_helmet, self.r3_triple_riding,
                   self.r4_red_light, self.r5_lane, self.r7_wrong_way,
                   self.r8_stopped, self.r10_u_turn):
            try:
                c = fn(st, frame, ctx)
            except Exception:
                c = None
            if c:
                out.append(c)
                st.flags.add(c.rule)
        return out

    # R1 -----------------------------------------------------------------
    def r1_overspeed(self, st, frame, ctx) -> Optional[Candidate]:
        v = st.speed_kmh(window=1.5)
        if v is None or len(st.metric_hist) < self.cfg.min_track_len:
            return None
        # legal tolerance: 10% + 3 km/h absorbs homography error
        limit = self.cfg.speed_limit_kmh
        if v <= limit * 1.10 + 3:
            return None
        if not self._fresh(st.track_id, "R1"):
            return None
        # confidence falls off if the track is short or the plane fit is poor
        conf = min(1.0, 0.55 + 0.02 * len(st.metric_hist)) * ctx.get("calib_quality", 1.0)
        return Candidate("R1_overspeed", st.track_id, st.cls_name, conf,
                         {"speed_kmh": round(v, 1), "limit": limit}, time.time())

    # R2 -----------------------------------------------------------------
    def r2_helmet(self, st, frame, ctx) -> Optional[Candidate]:
        if st.cls_name != "motorcycle" or self.rider_model is None:
            return None
        if st.best_crop is None or st.best_crop.shape[0] < 80:
            return None
        if not self._fresh(st.track_id, "R2", ttl=30):
            return None
        r = self.rider_model.predict(st.best_crop, verbose=False)[0]
        names = [self.rider_model.names[int(c)] for c in r.boxes.cls]
        heads = names.count("head_nohelmet")
        if heads == 0:
            return None
        conf = float(max(r.boxes.conf).item())
        return Candidate("R2_no_helmet", st.track_id, st.cls_name, conf,
                         {"bare_heads": heads}, time.time())

    # R3 -----------------------------------------------------------------
    def r3_triple_riding(self, st, frame, ctx) -> Optional[Candidate]:
        if st.cls_name != "motorcycle" or self.rider_model is None:
            return None
        if st.best_crop is None or not self._fresh(st.track_id, "R3", ttl=30):
            return None
        r = self.rider_model.predict(st.best_crop, verbose=False)[0]
        names = [self.rider_model.names[int(c)] for c in r.boxes.cls]
        riders = sum(1 for n in names if n.startswith("head_"))
        if riders < 3:
            return None
        return Candidate("R3_overloading", st.track_id, st.cls_name, 0.75,
                         {"occupants": riders}, time.time())

    # R4 -----------------------------------------------------------------
    def r4_red_light(self, st, frame, ctx) -> Optional[Candidate]:
        if not self.cfg.stop_line or self.signal_state_fn is None:
            return None
        sig = self.signal_state_fn()
        if sig.get("phase") != "red":
            return None
        if len(st.boxes) < 2:
            return None
        (_, b0), (_, b1) = st.boxes[-2], st.boxes[-1]
        prev = ((b0[0] + b0[2]) / 2, b0[3])
        curr = ((b1[0] + b1[2]) / 2, b1[3])
        a, b = self.cfg.stop_line
        if not crossed_segment(prev, curr, a, b):
            return None
        if not self._fresh(st.track_id, "R4", ttl=60):
            return None
        # require the light to have been red for >0.7 s: kills amber-flicker FPs
        if time.time() - sig.get("since", 0) < 0.7:
            return None
        return Candidate("R4_red_light", st.track_id, st.cls_name, 0.88,
                         {"red_for_s": round(time.time() - sig["since"], 2)}, time.time())

    # R5 -----------------------------------------------------------------
    def r5_lane(self, st, frame, ctx) -> Optional[Candidate]:
        restricted = ctx.get("restricted_lanes", {})   # {"lane_bus": ["bus"]}
        if not restricted or not st.boxes:
            return None
        _, b = st.boxes[-1]
        p = ((b[0] + b[2]) / 2, b[3])
        for lane, allowed in restricted.items():
            poly = self.cfg.lanes.get(lane)
            if not poly or not point_in_poly(p, poly):
                continue
            if st.cls_name in allowed:
                continue
            if not self._fresh(st.track_id, "R5", ttl=25):
                return None
            return Candidate("R5_lane_violation", st.track_id, st.cls_name, 0.7,
                             {"lane": lane, "allowed": allowed}, time.time())
        return None

    # R7 -----------------------------------------------------------------
    def r7_wrong_way(self, st, frame, ctx) -> Optional[Candidate]:
        h = st.heading()
        if h is None:
            return None
        legal = np.array(self.cfg.legal_heading, dtype=np.float32)
        legal /= np.linalg.norm(legal) + 1e-9
        cosang = float(np.dot(h, legal))
        if cosang > -0.65:            # not clearly opposing
            return None
        if not self._fresh(st.track_id, "R7", ttl=45):
            return None
        return Candidate("R7_wrong_way", st.track_id, st.cls_name,
                         min(1.0, abs(cosang)),
                         {"cos_to_legal": round(cosang, 2)}, time.time())

    # R8 -----------------------------------------------------------------
    def r8_stopped(self, st, frame, ctx, min_s: float = 45.0) -> Optional[Candidate]:
        v = st.speed_kmh(window=3.0)
        if v is None:
            return None
        if v > 2.0:
            self._stopped_since.pop(st.track_id, None)
            return None
        t0 = self._stopped_since.setdefault(st.track_id, time.time())
        dur = time.time() - t0
        if dur < min_s or ctx.get("congested"):   # don't alarm in a jam
            return None
        if not self._fresh(st.track_id, "R8", ttl=120):
            return None
        return Candidate("R8_stopped_vehicle", st.track_id, st.cls_name, 0.8,
                         {"stopped_s": round(dur)}, time.time())

    # R10 ----------------------------------------------------------------
    def r10_u_turn(self, st, frame, ctx) -> Optional[Candidate]:
        if len(st.metric_hist) < 20:
            return None
        pts = np.array([[p[1], p[2]] for p in st.metric_hist])
        v_early = pts[len(pts)//4] - pts[0]
        v_late = pts[-1] - pts[-len(pts)//4]
        n1, n2 = np.linalg.norm(v_early), np.linalg.norm(v_late)
        if n1 < 2 or n2 < 2:
            return None
        cosang = float(np.dot(v_early / n1, v_late / n2))
        if cosang > -0.75:
            return None
        zone = ctx.get("no_uturn_zone")
        if zone and not point_in_poly((pts[-1][0], pts[-1][1]), zone):
            return None
        if not self._fresh(st.track_id, "R10", ttl=60):
            return None
        return Candidate("R10_illegal_uturn", st.track_id, st.cls_name, 0.72,
                         {"turn_cos": round(cosang, 2)}, time.time())


# ------------------------------------------------------- promotion gate

def promote(cands: List[Candidate], st, min_conf: float = 0.80) -> List[Candidate]:
    """A candidate becomes enforceable only with plate + confidence + evidence."""
    for c in cands:
        c.enforceable = (
            c.confidence >= min_conf
            and st.plate is not None
            and st.plate_conf >= 0.85
            and len(st.boxes) >= 3
        )
    return cands
