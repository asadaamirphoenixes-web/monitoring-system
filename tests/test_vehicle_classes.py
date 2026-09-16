import warnings

import pytest

from src.config import SiteConfig
from src.detector import MockDetector, validate_vehicle_classes


def _cfg(vehicle_classes, class_source) -> SiteConfig:
    return SiteConfig(
        site_id="test", name="test site", rtsp_url="unused",
        vehicle_classes=vehicle_classes, class_source=class_source,
    )


def test_coco_subset_matching_names_fires_no_warning():
    cfg = _cfg(["car", "bus", "truck", "motorcycle", "pedestrian"], "coco_subset")
    names = {i: c for i, c in enumerate(cfg.vehicle_classes)}
    detector = MockDetector(names=names)

    with warnings.catch_warnings():
        warnings.simplefilter("error")   # any warning becomes a failure
        validate_vehicle_classes(cfg, detector)


def test_custom_with_missing_class_warns_and_names_it():
    cfg = _cfg(
        ["motorcycle", "car", "rickshaw", "water_tanker"], "custom",
    )
    # detector doesn't know water_tanker at all
    names = {0: "motorcycle", 1: "car", 2: "rickshaw"}
    detector = MockDetector(names=names)

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        validate_vehicle_classes(cfg, detector)

    messages = [str(w.message) for w in caught]
    assert any("water_tanker" in m for m in messages), messages


def test_class_source_rejects_invalid_value():
    with pytest.raises(ValueError, match="class_source"):
        _cfg(["car"], "not_a_real_source")


def test_custom_site_hard_fails_when_default_yolo_detector_is_missing_a_class(monkeypatch):
    """The 'detector loader' hard-fail (point 2) only applies when
    TrafficRadar builds its own YoloDetector from cfg.weights - a
    stand-in for a real fine-tuned model whose weights don't actually
    know the site's full taxonomy. Faked here (no real weights/GPU
    available) by swapping in a fake ultralytics.YOLO before the lazy
    import inside YoloDetector runs."""
    import sys
    import types

    fake_module = types.ModuleType("ultralytics")

    class FakeYOLO:
        def __init__(self, weights):
            self.names = {0: "motorcycle", 1: "car"}   # no water_tanker

    fake_module.YOLO = FakeYOLO
    monkeypatch.setitem(sys.modules, "ultralytics", fake_module)

    from src.pipeline import TrafficRadar

    cfg = _cfg(["motorcycle", "car", "water_tanker"], "custom")
    with pytest.raises(RuntimeError, match="water_tanker"):
        TrafficRadar(cfg)
