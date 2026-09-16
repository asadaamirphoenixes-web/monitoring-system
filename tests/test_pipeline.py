from pathlib import Path

import numpy as np
import supervision as sv

from src.config import SiteConfig
from src.detector import MockDetector
from src.pipeline import TrafficRadar
from tests.fixtures.detections import moving_box_sequence

EXAMPLE_CONFIG = Path(__file__).resolve().parents[1] / "config" / "site_example.yaml"


def _fake_frame() -> np.ndarray:
    return np.zeros((100, 100, 3), dtype=np.uint8)


def test_pipeline_runs_end_to_end_with_mock_detector_and_no_crash():
    # CLAUDE.md constraint 6: testable today with a mocked model, no
    # trained weights or GPU required.
    cfg = SiteConfig.load(str(EXAMPLE_CONFIG))
    det = sv.Detections(
        xyxy=np.array([[40.0, 40.0, 60.0, 60.0]], dtype=np.float32),
        confidence=np.array([0.9], dtype=np.float32),
        class_id=np.array([0]),
    )
    mock = MockDetector(names={0: "car"}, detections=det)
    events = []
    radar = TrafficRadar(cfg, on_event=events.append, detector=mock)

    frame = _fake_frame()
    for i in range(10):
        radar.process(frame, ts=i / cfg.fps_target)

    assert events
    assert all(e["type"] in ("count", "state", "violation") for e in events)


def test_pipeline_recovers_known_speed_through_mock_detector():
    cfg = SiteConfig.load(str(EXAMPLE_CONFIG))
    # config/site_example.yaml: a 100x100 px patch == 10x10 m, so 1 px = 0.1 m
    px_per_frame = 3.0
    seq = moving_box_sequence(
        start_xy=(10.0, 30.0), velocity_xy_per_frame=(px_per_frame, 0.0),
        n_frames=20, box_size=40.0,
    )
    mock = MockDetector(names={0: "car"}, script=seq)
    radar = TrafficRadar(cfg, on_event=lambda e: None, detector=mock)

    frame = _fake_frame()
    for i in range(len(seq)):
        radar.process(frame, ts=i / cfg.fps_target)

    assert len(radar.tracks) == 1, "ByteTrack should keep this a single track"
    st = next(iter(radar.tracks.values()))
    got_kmh = st.speed_kmh()

    scale_m_per_px = cfg.target_size_m[0] / 100.0
    expected_kmh = px_per_frame * cfg.fps_target * scale_m_per_px * 3.6

    assert got_kmh is not None
    assert abs(got_kmh - expected_kmh) / expected_kmh < 0.05
