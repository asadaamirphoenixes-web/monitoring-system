# KT-Radar

Core perception pipeline for Karachi traffic monitoring: RTSP/file -> YOLO
detection -> ByteTrack -> perspective-corrected speed -> zone counting ->
violation rule engine -> event bus.

Tuned for mixed traffic (motorcycles, rickshaws, Suzuki pickups, water
tankers, dumpers, donkey carts, pedestrians in the carriageway).

## Setup

```bash
pip install -r requirements.txt
```

Place trained detector weights under `models/` (see `weights` in the site
config) and configure a site in `config/`.

## Run

```bash
python -m src.pipeline --config config/site_shahrah_faisal.yaml
python -m src.pipeline --config config/site_shahrah_faisal.yaml --source path/to/video.mp4 --display
```

## Site configuration

Each site is a YAML file (see `config/site_shahrah_faisal.yaml`) defining:

- the RTSP source and model parameters (`imgsz`, `conf`, `iou`, `device`, `weights`)
- `source_polygon` / `target_size_m`: the image-to-metric ground-plane homography
- `count_lines` and `lanes`: counting and lane-occupancy zones
- `stop_line` and `legal_heading`: for violation detection
- `speed_limit_kmh`, `congestion_density_thr`, `congestion_speed_thr`: analytics thresholds
