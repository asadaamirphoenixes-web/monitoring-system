# KT-Radar

Core perception pipeline for Karachi traffic monitoring: RTSP/file ->
detection -> ByteTrack -> perspective-corrected speed -> zone counting ->
violation rule engine -> event bus -> control-room API.

Tuned for mixed traffic (motorcycles, rickshaws, Suzuki pickups, water
tankers, dumpers, donkey carts, pedestrians in the carriageway). See
`CLAUDE.md` for the non-negotiable design constraints (speed is never
computed in pixel space, nothing is enforceable on one rule firing,
privacy-by-default, calibration is never hardcoded, every module is
testable without a trained model).

## Setup

```bash
pip install -r requirements.txt          # or requirements-dev.txt for tests/lint/types
```

Place trained detector weights under `models/` (see `weights` in the site
config) and configure a site in `config/`. `config/site_example.yaml` is
a fake/labelled-as-such fixture for tests, not a real camera.

## Run

```bash
python -m src.pipeline --config config/site_shahrah_faisal.yaml
python -m src.pipeline --config config/site_shahrah_faisal.yaml --source path/to/video.mp4 --display

# no trained weights or GPU needed - runs against a MockDetector instead:
python -m src.pipeline --config config/site_example.yaml \
    --source tests/fixtures/some_clip.mp4 --mock-detector
```

## Modules

- `src/config.py` - `SiteConfig`, loaded from per-camera YAML
- `src/homography.py` - `GroundPlane`, the pixel -> metric ground-plane transform
- `src/tracker.py` - `TrackState`, including the least-squares `speed_kmh()`
- `src/detector.py` - the `Detector` interface, `YoloDetector`, and
  `MockDetector` (lets everything downstream be developed/tested without a
  trained model or GPU)
- `src/violations.py` - the rule engine and the candidate -> enforceable
  promotion gate
- `src/anpr.py` - two-stage plate localisation + OCR with temporal voting
- `src/privacy.py` - face blurring, HMAC plate tokens, gated+audited release
- `src/pipeline.py` - `TrafficRadar`, wiring all of the above together
- `src/api.py` - the control-room ingest/query/live-feed API

## Tests

```bash
pytest              # CPU-only, no network, no GPU - uses MockDetector + config/site_example.yaml
ruff check .
mypy src/
```

## Site configuration

Each site is a YAML file (see `config/site_shahrah_faisal.yaml`) defining:

- the RTSP source and model parameters (`imgsz`, `conf`, `iou`, `device`, `weights`)
- `source_polygon` / `target_size_m`: the image-to-metric ground-plane homography
- `count_lines` and `lanes`: counting and lane-occupancy zones
- `stop_line` and `legal_heading`: for violation detection
- `speed_limit_kmh`, `congestion_density_thr`, `congestion_speed_thr`: analytics thresholds
- `restricted_lanes`, `no_uturn_zone`, `rider_model_weights`: violation rule engine (see `src/violations.py`)
- `plate_weights`, `face_blur_weights`: ANPR and face blurring (see `src/anpr.py`, `src/privacy.py`)

## Privacy

Faces are blurred before any frame is used for a stored crop, evidence, or
display (`face_blur_weights`, on by default). Plates are never written to an
event in clear text - only an HMAC-SHA256 token, unless a violation has been
promoted to `enforceable` and the deployment is explicitly authorised to
release it. This is controlled by environment variables, not config, since
they gate legal authority rather than site behaviour:

- `KTR_PLATE_HMAC_KEY`: required to compute plate tokens at all; without it,
  `plate` is omitted from every event rather than falling back to raw text.
- `KTR_ENFORCEMENT_AUTHORISED=1`: required before any clear-text plate is
  ever released, and only for candidates already promoted to `enforceable`.
- `KTR_AUDIT_LOG`: append-only log path (default `logs/plate_access.jsonl`)
  that every clear-text release is written to.

See `src/privacy.py` for the full design rationale.

## Roadmap

`PROMPTS.md` is the prompt pack for the full project (P0-P9): repo bootstrap,
detector fine-tuning, a calibration GUI, ANPR, an appeals-grade evidence pack,
congestion/queue prediction, a SUMO signal-control digital twin, the edge
agent, the control-room UI, and an evaluation harness. Built so far:
`src/pipeline.py`, `src/violations.py`, `src/anpr.py`, `src/privacy.py`,
`src/api.py`, `src/config.py`, `src/homography.py`, `src/tracker.py`,
`src/detector.py`, `tests/` (P0/P3/P4-partial). Not yet started:
`src/congestion.py`, `scripts/calibrate_gui.py`, `src/edge_agent.py`, the
evidence-pack/appeal flow, and `docker/`.

## Control-room API

```bash
uvicorn src.api:app --host 0.0.0.0 --port 8080
```

SQLite-backed by default (`data/ktradar.db`); point an edge node's
`on_event` at `POST /ingest/event` to feed it - see `src/api.py` for the
full endpoint list (`/api/sites`, `/api/flow`, `/api/violations`,
`/api/hotspots`, `/api/journey`, `/ws/live`). The ingest model accepts the
flat event dicts `src/pipeline.py` already emits (`class`, `plate`, and any
rule- or state-specific fields) and normalises them into its stored schema.
