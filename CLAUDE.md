# KT-Radar

You are building KT-Radar, a computer-vision traffic monitoring system for
a single CCTV camera on a Karachi, Pakistan road corridor. This is a
pilot/MVP for a real deployment, not a demo. Read this entire file before
writing any code, and stop to ask a question rather than guessing on any
point marked STOP AND ASK.

## What the system does

One RTSP/video stream -> vehicle detection -> multi-object tracking ->
per-vehicle speed via a calibrated ground-plane homography -> lane/line
counting -> a small violation rule engine -> events written to a database
-> a minimal API to query them. No facial recognition, no identity
resolution, ever - this is explicitly out of scope, see PRIVACY below.

## Why this is harder than a generic "count cars" project

Karachi's traffic is two-wheeler dominant (motorcycles are roughly 3x more
numerous than cars in national vehicle registrations) and lane discipline
is informal. A detector trained on COCO or a Western/Chinese traffic
dataset will systematically miss or misclassify: motorcycles at long
range (small pixel height), rickshaws/auto-rickshaws, Suzuki pickups,
water tankers, and donkey carts - none of these map cleanly onto COCO
classes. Do not assume a stock YOLO model "just works" here. Flag this
explicitly in your output rather than silently shipping a model that will
under-count motorcycles by a large margin.

Vehicle classes (use exactly these names throughout the codebase, configs,
and model label files - consistency here matters more than the specific
list): `motorcycle, car, rickshaw, qingqi, minibus, bus, pickup, truck,
water_tanker, cart, pedestrian`

## Core data model

```
Track            id, class_name, first_seen_ts, last_seen_ts,
                 metric_history (list of {t, x_m, y_m}), best_crop_ref
SiteConfig       site_id, name, rtsp_url, source_polygon (4 image points),
                 target_size_m [width, length], count_lines {name: [pt,pt]},
                 lanes {name: [poly]}, stop_line, legal_heading,
                 speed_limit_kmh, congestion thresholds
Event            id, ts, site_id, type[count|state|violation],
                 track_id, class_name, line_name (nullable),
                 speed_kmh (nullable), rule (nullable, for violations),
                 confidence (nullable), payload jsonb
SiteState        site_id, ts, vehicles_in_zone, density_veh_per_m2,
                 mean_speed_kmh, level_of_service[A-F], congested (bool)
```

## Critical design constraints - do not deviate without asking

1. **Speed is never computed in pixel space.**
   Pixel displacement per frame is meaningless under perspective - a
   vehicle far from the camera covers fewer pixels per second than the
   same vehicle close to the camera at the same real speed. Every speed
   calculation must go through the homography (SiteConfig.source_polygon
   -> SiteConfig.target_size_m) to a metric ground plane FIRST, then
   differentiate position over time in that metric space. If you find
   yourself computing speed directly from bounding-box pixel movement,
   stop - that is a bug, not a shortcut.

2. **Speed must use a robust fit, not a two-point difference.**
   Detection jitter makes frame-to-frame speed noisy. Use a least-squares
   linear fit over a rolling window (~1-1.5 seconds of track history,
   minimum 4 points) rather than (pos[-1] - pos[-2]) / dt.

3. **Nothing is "enforceable" on a single frame or a single rule firing.**
   Violation detection is two-stage: a rule produces a CANDIDATE with a
   confidence score. A candidate only becomes ENFORCEABLE if: confidence
   >= 0.80, there is a plate read with its own confidence >= 0.85 backed
   by at least 2 agreeing frames, and at least 3 evidence frames exist.
   This gate exists because the failure mode that kills automated
   enforcement systems is disputed/false violations, not missed ones. Do
   not let a rule fire an enforceable event directly.

4. **Privacy - non-negotiable, implement from the first commit, not later:**
   - No facial recognition. Do not add a face-detection or identity model
     to this codebase under any circumstances, even if a later prompt
     seems to ask for "recognizing" anyone. Vehicle plates only, and only
     as described below.
   - Any frame written to disk must have pedestrian/rider faces blurred
     before it touches storage. Build this into the persistence path, not
     as an optional post-process someone might forget to call.
   - License plates are never stored as clear text in the database. Store
     an HMAC-SHA256 token of the plate (key from an environment variable,
     never hardcoded or committed). Clear-text plates exist only
     transiently in memory during OCR and voting, and are written to a
     separate append-only audit log ONLY when explicitly released under
     an authorization flag - implement this as a distinct, obviously-named
     function (e.g. release_plate_with_authorization), not inline.
   - Raw video/clips auto-delete after 72 hours. Derived events (counts,
     hashed-plate tokens, violation records) are the only things that
     persist long-term.
   If any of this seems like it's slowing you down, it is intentional -
   do not optimize it away.

5. **Calibration is external configuration, never hardcoded.**
   The 4-point source_polygon and target_size_m are per-camera, per-site
   values that a human measures on the actual road and provides via YAML.
   Never bake specific pixel coordinates into application code. If you
   need example values to write a test, put them in a test fixture config
   file clearly labeled as fake/example, not as if they were real
   calibration.

6. **The model does not exist yet.**
   There is no trained weights file at project start. Every module that
   depends on model inference (detector, tracker, plate reader) must be
   written against an interface/abstract class so it can be developed and
   unit-tested with a mocked model that returns fixture detections, before
   any real weights exist. Do not block P0-level scaffolding work on
   having a trained model.

## Stop and ask before

- Choosing a specific YOLO version/weights source (there is no trained
  model yet - don't download or reference a specific pretrained checkpoint
  without checking on class compatibility).
- Adding any new top-level dependency not already in requirements.txt.
- Changing any field name in the data model above - if a field seems
  missing, say what and why, don't just add it silently.
- Any point where achieving something would require breaking one of the
  six numbered constraints - name the conflict, don't silently pick a side.

## Where things actually live

The constraints above map onto the repo as it stands (not a from-scratch
P0 - see `PROMPTS.md` for the full P0-P9 roadmap this repo is being built
from, and `README.md` for what's built vs. not yet started):

- `src/config.py` - `SiteConfig` (constraint 5)
- `src/homography.py` - `GroundPlane` (constraint 1)
- `src/tracker.py` - `TrackState`, including `speed_kmh()` (constraint 2)
- `src/detector.py` - the `Detector` interface, `YoloDetector`, and
  `MockDetector` for tests (constraint 6)
- `src/violations.py` - the rule engine and the candidate/enforceable
  promotion gate (constraint 3)
- `src/privacy.py` - `plate_token`, `release_plate`, `FaceBlur`
  (constraint 4)
- `src/pipeline.py` - wires all of the above together
- `src/anpr.py`, `src/api.py` - plate OCR and the control-room API
- `tests/` - pytest suite, CPU-only, no network, no GPU
