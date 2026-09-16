"""Regression coverage for the P9 eval harness itself (tests/eval/) - not
for the perception pipeline. Regenerates the synthetic placeholder clip if
it's missing (it's gitignored, like every clip) and runs the real harness
functions against it, including the deliberately-bad-ground-truth case
the harness's own Definition of Done calls for."""

from __future__ import annotations

import copy
import json
from pathlib import Path

import pytest

from tests.eval.clips.generate_placeholder_clip import main as generate_placeholder_clip
from tests.eval.run_eval import evaluate_clip

EVAL_DIR = Path(__file__).resolve().parent / "eval"
CLIPS_DIR = EVAL_DIR / "clips"
GT_PATH = EVAL_DIR / "ground_truth" / "synth_day_normal_01.json"


@pytest.fixture(scope="module", autouse=True)
def ensure_placeholder_clip():
    if not (CLIPS_DIR / "synth_day_normal_01.mp4").exists():
        generate_placeholder_clip()


def test_synthetic_placeholder_uses_mock_replay_and_passes_the_gate():
    result = evaluate_clip(GT_PATH, CLIPS_DIR)

    assert result.detector_mode.startswith("mock-replay")
    assert not result.gate_failures, result.gate_failures

    assert any(cr.line == "cordon" and cr.cls_name == "car" and cr.abs_error == 0
               for cr in result.count_results)

    assert len(result.speed_results) == 1
    assert result.speed_results[0].status == "pass"
    assert result.speed_results[0].measured_speed_kmh == pytest.approx(16.2, abs=0.5)

    assert result.fragmentation.get("cordon") == pytest.approx(1.0)


def test_gate_fails_on_a_deliberately_bad_manual_count(tmp_path):
    good = json.loads(GT_PATH.read_text())
    bad = copy.deepcopy(good)
    bad["manual_line_counts"] = {"cordon": {"car": 99}}   # absurd vs. the real single crossing
    bad_path = tmp_path / "bad_ground_truth.json"
    bad_path.write_text(json.dumps(bad))

    result = evaluate_clip(bad_path, CLIPS_DIR)

    assert result.gate_failures, "expected the count-error gate to fail on an absurd manual count"
    assert any("car" in f and "cordon" in f for f in result.gate_failures)


def test_missing_clip_is_reported_not_crashed(tmp_path):
    gt_path = tmp_path / "ghost.json"
    gt_path.write_text(json.dumps({
        "clip": "does_not_exist.mp4",
        "condition": "day",
        "site_config": "config/site_example.yaml",
    }))

    result = evaluate_clip(gt_path, CLIPS_DIR)

    assert result.detector_mode == "skipped: video not found"
    assert result.gate_failures == []
    assert any("missing" in s for s in result.skipped_checks)


def test_speed_check_without_a_parseable_hint_is_not_measured_not_guessed(tmp_path):
    gt_path = tmp_path / "no_hint.json"
    gt_path.write_text(json.dumps({
        "clip": "synth_day_normal_01.mp4",
        "condition": "day",
        "site_config": "config/site_example.yaml",
        "known_speed_checks": [
            {"track_hint": "some vehicle, no timestamp given",
             "approx_speed_kmh": 40, "tolerance_kmh": 5},
        ],
    }))

    result = evaluate_clip(gt_path, CLIPS_DIR)

    assert len(result.speed_results) == 1
    assert result.speed_results[0].status == "not measured"
    assert result.speed_results[0].measured_speed_kmh is None
