# KT-Radar :: Prompt Pack for AI Coding Agents

Use these with Claude Code, Cursor, or any agentic coder. They are written to be
pasted one at a time, in order. Each assumes the repo layout in the build guide.

Rule of thumb: a prompt that says "build a traffic system" gets you a demo.
A prompt that names the failure mode you are trying to avoid gets you a system.

---

## P0 — Repo bootstrap

```
You are setting up a production edge-vision repo called kt-radar.

Create this structure with working stubs, type hints, and pytest tests:

kt-radar/
  src/ pipeline.py anpr.py violations.py privacy.py api.py congestion.py
       calibrate.py edge_agent.py
  config/ site_*.yaml
  models/            (gitignored, downloaded via scripts/fetch_models.sh)
  tests/ fixtures/   (10 short clips + expected-event JSON)
  docker/ Dockerfile.edge Dockerfile.server compose.yml
  scripts/ fetch_models.sh calibrate_gui.py export_trt.sh

Constraints:
- Python 3.11, ruff + mypy clean, no global state outside a Settings object.
- Every module importable without a GPU present (lazy-load torch).
- tests/ must run in CI on CPU using the fixture clips, no network.
- Config is pydantic-settings, env-overridable, one YAML per camera site.

Output the full tree and every file. Do not summarise; write the code.
```

---

## P1 — Detector fine-tuning for Karachi classes

```
Write a training pipeline that fine-tunes YOLO11s for Karachi road users.

Classes (11): motorcycle, car, rickshaw, qingqi, minibus, bus, pickup,
truck, water_tanker, cart, pedestrian.

Requirements:
- Start from COCO-pretrained weights; COCO has no rickshaw/qingqi/tanker, so
  those need new heads' worth of data. Build a data recipe that mixes:
  (a) ~4k self-labelled Karachi frames, (b) COCO vehicle subset for regularisation,
  (c) heavy augmentation for the long-tail classes.
- Augmentations must match the deployment domain: motion blur, JPEG artifacts at
  q=40-70 (CCTV re-encode), low-light gamma shift, monsoon rain streaks, dust haze,
  and 4x downscale-upscale to mimic a far camera. Use albumentations.
- Handle extreme class imbalance: motorcycles will be ~60% of instances. Implement
  class-balanced sampling and report per-class AP, not just mAP50-95.
- Add a small-object path: motorcycles at 30 px height must be detected. Train at
  imgsz=1280, evaluate at 960 and 1280, and report the delta.
- Emit a model card with per-class AP, per-condition AP (day/night/rain), and the
  latency on Jetson Orin Nano after TensorRT FP16 and INT8 export.

Write: dataset.py, train.py, evaluate.py, export.py, and a Makefile.
```

---

## P2 — Calibration GUI

```
Build scripts/calibrate_gui.py: an OpenCV+Tkinter tool where an operator loads a
frame from an RTSP stream, clicks 4 ground-plane points, enters the real-world
width and length in metres, and immediately sees:

  1. the warped bird's-eye view,
  2. a 1-metre grid overlaid back onto the original frame,
  3. a residual error readout from 2 optional check-points of known separation.

Then let them draw counting lines, lane polygons, a stop line, and a legal-heading
arrow. Save everything to config/site_<id>.yaml in the schema used by SiteConfig.

Critical: warn loudly if the 4 points are near-collinear or if the implied
metres-per-pixel varies more than 4x across the patch — both produce garbage speeds.
```

---

## P3 — Pakistani ANPR

```
Implement a two-stage ANPR for Sindh plates with a temporal voting layer.

Stage 1: YOLO plate localiser fine-tuned on Pakistani plates (Roboflow Pakistani
License Plates + PLPD-style data + your own crops). Must handle: white private,
yellow commercial, green government, hand-painted legacy, two-line bike plates,
and partially occluded plates.

Stage 2: recogniser. Implement three interchangeable backends behind one interface:
  - PaddleOCR PP-OCRv5 with fine-tuned rec head
  - a CRNN+CTC trained on synthetic plates (generate 200k synthetic Sindh plates
    with correct fonts, spacing, dirt, glare, bending, and motion blur)
  - TrOCR-small fine-tuned
Benchmark all three on the same held-out real set and report end-to-end accuracy,
not just character accuracy.

Then: a voting layer that accumulates confidence-weighted reads across every frame
of a track and only emits a plate when >=2 frames agree and weighted agreement
exceeds 0.85. Published Pakistani ANPR work hits ~99% mAP on localisation but only
~73% end-to-end on low-res images; voting is where you recover that gap. Prove it:
report accuracy single-frame vs voted on the same clips.

Also implement format normalisation (ABC-123 / ABC-1234 / AB-1234) and a
segment-aware confusion fixer (O/0, I/1, S/5 resolved differently in the letter
block vs the digit block).
```

---

## P4 — Violation engine with an appeals-grade evidence pack

```
Extend src/violations.py so every promoted violation writes an evidence pack:

  - 3 stills: approach, violation instant, departure (faces blurred, plate clear)
  - a 6-second H.264 clip, 3 s before to 3 s after
  - a JSON manifest: rule id, site, timestamp (NTP-synced, UTC + PKT), speed with
    its 95% confidence interval, calibration hash, model version + weights SHA256,
    tracker config, and the exact frame indices used
  - an SHA256 of the pack plus an append-only hash chain, so tampering is detectable

Rationale: the failure mode of every automated-enforcement rollout in South Asia is
disputed challans. An evidence pack that a citizen can view and a magistrate can
verify is the difference between a pilot and a dead pilot. Build an /appeal
endpoint that renders the pack for a given challan reference.
```

---

## P5 — Congestion, queues, and prediction

```
Build src/congestion.py:

1. Per-lane queue length in metres: cluster stopped/slow tracks along the metric
   y-axis, take the furthest contiguous cluster boundary from the stop line.
2. Level of Service A-F from joint density + mean speed, computed per 30 s.
3. Turning-movement counts: for an intersection camera, classify each track's
   entry and exit approach and output a standard TMC matrix — this is the exact
   artefact a transport consultant pays for, and no free tool produces it for
   Karachi.
4. A 15/30/60-minute congestion forecast per site. Baseline: seasonal-naive on
   time-of-day + day-of-week. Then a GNN or a temporal-fusion transformer over the
   camera graph where edges are road adjacency. You MUST beat the baseline on
   held-out weeks or ship the baseline — report both.
5. Anomaly detection: flag when observed flow deviates >3 sigma from the
   time-of-day profile (accident, protest, VIP movement, flooded underpass).
```

---

## P6 — Adaptive signal control digital twin

```
Build a SUMO digital twin of one Karachi corridor (5-7 signalised intersections
on Shahrah-e-Faisal) and train an RL controller against it.

- Import geometry from OpenStreetMap via netconvert; hand-fix the U-turns and
  service roads, which OSM gets wrong in Karachi.
- Calibrate demand from KT-Radar counts: use the turning-movement matrices from
  P5 to build OD flows, then calibrate with SUMO's routeSampler.
- Model the actual fleet mix: ~75% two-wheelers nationally. Motorcycle
  lane-filtering breaks SUMO's default car-following; use sublane resolution and
  tune impatience/lcSublane per vClass, then validate against observed queue
  lengths from the cameras.
- Use sumo-rl (Gymnasium/PettingZoo) with PPO or IPPO for multi-intersection
  control. Compare against fixed-time, max-pressure, and actuated baselines.
- Report: mean delay, queue length, stops per vehicle, and CO2 proxy. If RL does
  not beat max-pressure, say so — max-pressure is a strong, deployable baseline
  and is far easier to get a government to approve.
- Ship the controller as an advisory feed first (recommended phase timings to a
  human operator), not direct signal actuation.
```

---

## P7 — Edge agent

```
Write src/edge_agent.py to run one camera on a Jetson Orin Nano:

- TensorRT FP16 engine, DeepStream or direct GStreamer with nvv4l2decoder
- Watchdog: reconnect RTSP with exponential backoff, restart the engine on CUDA
  OOM, and post a heartbeat to the server every 15 s
- Store-and-forward: events go to a local SQLite WAL queue and drain to the server
  over HTTPS; survives the 6-12 hour connectivity and power gaps that are normal
  in Karachi. Cap the queue at 500 MB and drop oldest `count` events first,
  never violations.
- Local retention: 72 h of clips on an NVMe, auto-purged by src/privacy.py
- Resource guard: if GPU temp >85C (Karachi summer in a roadside cabinet is brutal),
  drop fps_target to 8 and log a degraded-mode event rather than crashing.
- Expose Prometheus metrics: fps, detections/s, track count, queue depth, temp.
```

---

## P8 — Control room UI

```
Build the operator dashboard as a single-page app:

- A Leaflet map of Karachi with one marker per camera, coloured by LOS A-F,
  updating over the /ws/live websocket.
- Click a site: live annotated stream (WebRTC or HLS), flow/speed sparklines,
  queue lengths per lane, and the last 20 events.
- A violations queue with a human-in-the-loop review step: every candidate shows
  the evidence pack and the operator clicks approve/reject. Track the rejection
  rate per rule per site and surface it — a rule rejecting >8% of candidates is
  miscalibrated and should auto-suspend.
- A planner view: congestion-hours hotspot ranking, corridor journey times, weekly
  and monthly trend, exportable to CSV/GeoJSON.
- Urdu/English toggle. RTL layout for Urdu. This is not optional for a Karachi
  deployment.
```

---

## P9 — Evaluation harness (do this before you demo anything)

```
Build tests/eval/ that produces a one-page honest scorecard:

- 20 hand-labelled 2-minute clips: 5 day, 5 night, 5 rain, 5 peak-jam.
- Metrics: MOTA/IDF1 for tracking, count MAE vs manual count, speed MAE vs a
  GPS-logged probe vehicle driven through the scene, per-rule precision/recall,
  end-to-end plate accuracy.
- Report per-condition, never just the aggregate. An 88% aggregate that is 96%
  by day and 61% at night is a night-time problem, not an 88% system.
- Fail the build if speed MAE > 5 km/h or violation precision < 0.90.
```
