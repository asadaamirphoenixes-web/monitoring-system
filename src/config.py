"""
KT-Radar :: site configuration
-------------------------------
Calibration (source_polygon, target_size_m) and every other per-camera
setting live in external YAML, never in application code - see
CLAUDE.md constraint 5. Load with SiteConfig.load(path).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional

import yaml

# Karachi road-user classes - use exactly these names throughout the
# codebase, configs, and model label files (CLAUDE.md), unless a site's
# vehicle_classes says otherwise (e.g. a region with no fine-tuned
# detector yet - see class_source below).
KARACHI_VEHICLE_CLASSES: List[str] = [
    "motorcycle", "car", "rickshaw", "qingqi", "minibus", "bus",
    "pickup", "truck", "water_tanker", "cart", "pedestrian",
]


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

    # vehicle taxonomy this site expects to see (see src/detector.py's
    # validate_vehicle_classes). Defaults to the Karachi list so every
    # config written before this field existed keeps working unchanged.
    vehicle_classes: List[str] = field(default_factory=lambda: list(KARACHI_VEHICLE_CLASSES))
    # "custom": vehicle_classes is a fine-tuned taxonomy - the detector's
    #   weights MUST know every one of these classes exactly, or the
    #   detector loader hard-fails rather than silently running with a
    #   partial/wrong class list.
    # "coco_subset": vehicle_classes is limited to what a generic
    #   COCO-pretrained detector can already see (car, bus, truck,
    #   motorcycle, bicycle, and COCO's "person" if you relabel it to
    #   "pedestrian" - the cross-check is a literal string match, nothing
    #   translates "person" for you). For a region with no fine-tuned
    #   model yet - see config/site_example_coco_only.yaml.
    class_source: str = "custom"

    def __post_init__(self) -> None:
        if self.class_source not in ("custom", "coco_subset"):
            raise ValueError(
                f"class_source must be 'custom' or 'coco_subset', got {self.class_source!r}"
            )

    @staticmethod
    def load(path: str | Path) -> "SiteConfig":
        raw = yaml.safe_load(Path(path).read_text())
        return SiteConfig(**raw)
