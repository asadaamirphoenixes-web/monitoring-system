"""
KT-Radar :: P9 eval harness
------------------------------
The 14 unit tests in tests/ prove the CODE runs correctly against
synthetic detections. They prove nothing about whether the tracker,
homography, and speed logic hold up on real, messy video - occlusion,
jitter, low light, a camera that isn't perfectly still. This is the
harness that answers that question honestly, and reports the truth even
when the truth is bad. See tests/eval/README.md to add a clip.

Run:
    python -m tests.eval.run_eval
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.config import SiteConfig  # noqa: E402
from src.detector import Detector, MockDetector, YoloDetector  # noqa: E402
from src.pipeline import TrafficRadar  # noqa: E402
from tests.eval.detections_io import load_detections_script  # noqa: E402
from tests.eval.report import render_markdown, render_terminal  # noqa: E402
from tests.eval.results import (  # noqa: E402
    ClipResult,
    CountResult,
    SpeedCheckResult,
    WholeClipEstimateResult,
)
from tests.eval.schema import GroundTruth, SpeedCheck  # noqa: E402

# fail if any clip's per-class count error exceeds this share of the manual count
COUNT_GATE_PCT = 0.20
SPEED_GATE_KMH = 5.0       # fail if a clip's mean speed error exceeds this
FRAGMENTATION_GAP_S = 2.0  # crossings on the same line within this gap count as one vehicle


# --------------------------------------------------------------------------
# detector selection - never silently fall back without saying so
# --------------------------------------------------------------------------

def select_detector(cfg: SiteConfig, clip_path: Path) -> Tuple[Optional[Detector], str]:
    weights_path = Path(cfg.weights)
    if weights_path.exists():
        detector = YoloDetector(cfg.weights, imgsz=cfg.imgsz, conf=cfg.conf,
                                 iou=cfg.iou, device=cfg.device)
        return detector, f"real-weights ({weights_path.name})"

    sidecar = clip_path.with_name(clip_path.stem + ".detections.json")
    if sidecar.exists():
        names, script = load_detections_script(sidecar)
        return MockDetector(names=names, script=script), f"mock-replay ({sidecar.name})"

    return None, ("skipped: no real weights at "
                   f"{weights_path} and no hand-annotated sidecar at {sidecar}")


# --------------------------------------------------------------------------
# per-clip evaluation
# --------------------------------------------------------------------------

def evaluate_clip(gt_path: Path, clips_dir: Path) -> ClipResult:
    gt = GroundTruth.load(gt_path)
    clip_path = clips_dir / gt.clip

    skipped: List[str] = []
    if not gt.manual_line_counts:
        skipped.append("no manual_line_counts provided - count metrics not measured")
    if not gt.known_speed_checks:
        skipped.append("no known_speed_checks provided - speed metrics not measured")
    if not gt.manual_whole_clip_estimate:
        skipped.append(
            "no manual_whole_clip_estimate provided - whole-clip range check not measured"
        )

    if not clip_path.exists():
        skipped.append(f"video file missing - drop it at {clip_path} to evaluate this clip")
        return ClipResult(
            clip_name=gt.clip, condition=gt.condition,
            detector_mode="skipped: video not found",
            notes=gt.notes, skipped_checks=skipped,
        )

    cfg = SiteConfig.load(gt.site_config)
    detector, mode = select_detector(cfg, clip_path)
    if detector is None:
        skipped.append(mode)
        return ClipResult(
            clip_name=gt.clip, condition=gt.condition, detector_mode=mode,
            notes=gt.notes, skipped_checks=skipped,
        )

    events: List[dict] = _run_pipeline(cfg, detector, clip_path)

    count_events = [e for e in events if e["type"] == "count"]
    predicted = _predicted_counts(count_events)
    count_results = _count_results(gt.manual_line_counts, predicted)
    speed_results = _resolve_speed_checks(gt.known_speed_checks, count_events)
    fragmentation = _fragmentation_ratios(count_events)
    whole_clip_results = _whole_clip_estimate_results(
        gt.manual_whole_clip_estimate, count_events
    )

    gate_failures: List[str] = []
    for cr in count_results:
        if cr.gate_fail:
            gate_failures.append(
                f"{gt.clip}: count error for '{cr.cls_name}' on line '{cr.line}' = "
                f"{cr.abs_error} ({cr.pct_error:.0%} of manual {cr.manual}) "
                f"- exceeds {COUNT_GATE_PCT:.0%} gate"
            )

    measured_errors = [
        abs(r.measured_speed_kmh - r.approx_speed_kmh)
        for r in speed_results if r.status in ("pass", "fail")
    ]
    speed_mae = (sum(measured_errors) / len(measured_errors)) if measured_errors else None
    if speed_mae is not None and speed_mae > SPEED_GATE_KMH:
        gate_failures.append(
            f"{gt.clip}: speed MAE {speed_mae:.1f} km/h exceeds {SPEED_GATE_KMH:.0f} km/h gate"
        )

    return ClipResult(
        clip_name=gt.clip, condition=gt.condition, detector_mode=mode, notes=gt.notes,
        count_results=count_results, speed_results=speed_results,
        fragmentation=fragmentation, whole_clip_estimate_results=whole_clip_results,
        skipped_checks=skipped,
        # deliberately not folded into gate_failures - see WholeClipEstimateResult
        gate_failures=gate_failures, speed_mae=speed_mae,
    )


def _run_pipeline(cfg: SiteConfig, detector: Detector, clip_path: Path) -> List[dict]:
    """Reads the clip and calls TrafficRadar.process() directly (not
    TrafficRadar.run()) with a frame-index-derived timestamp, not a
    wall-clock one. See the note in the P9 build summary: run()'s
    `ts = time.time() - t0` is correct for a live RTSP stream but wrong
    for offline evaluation of a recorded file, since the eval loop doesn't
    execute in real time - it would make every speed_kmh() reading here
    depend on how fast this script runs, not on the video's own timeline.
    Flagged to the user rather than changed in src/pipeline.py itself.
    """
    events: List[dict] = []
    cap = cv2.VideoCapture(str(clip_path))
    if not cap.isOpened():
        raise RuntimeError(f"cannot open clip: {clip_path}")
    native_fps = cap.get(cv2.CAP_PROP_FPS) or cfg.fps_target
    stride = max(1, round(native_fps / cfg.fps_target))

    radar = TrafficRadar(cfg, on_event=events.append, detector=detector)
    frame_idx = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if frame_idx % stride:
            continue
        ts = frame_idx / native_fps
        before = len(events)
        radar.process(frame, ts)
        for e in events[before:]:
            e["_eval_clip_ts"] = ts
    cap.release()
    return events


# --------------------------------------------------------------------------
# metrics
# --------------------------------------------------------------------------

def _predicted_counts(count_events: List[dict]) -> Dict[str, Dict[str, int]]:
    counts: Dict[str, Dict[str, int]] = {}
    for e in count_events:
        by_class = counts.setdefault(e["line"], {})
        by_class[e["class"]] = by_class.get(e["class"], 0) + 1
    return counts


def _count_results(
    manual: Dict[str, Dict[str, int]], predicted: Dict[str, Dict[str, int]]
) -> List[CountResult]:
    out = []
    for line in sorted(set(manual) | set(predicted)):
        classes = set(manual.get(line, {})) | set(predicted.get(line, {}))
        for cls in sorted(classes):
            m = manual.get(line, {}).get(cls, 0)
            p = predicted.get(line, {}).get(cls, 0)
            err = abs(p - m)
            pct = (err / m) if m > 0 else None
            gate_fail = pct is not None and pct > COUNT_GATE_PCT
            out.append(CountResult(line, cls, m, p, err, pct, gate_fail))
    return out


def _resolve_speed_checks(
    checks: List[SpeedCheck], count_events: List[dict]
) -> List[SpeedCheckResult]:
    candidates = [e for e in count_events if e.get("speed_kmh") is not None]
    results = []
    for chk in checks:
        if chk.hint_time_s is None:
            results.append(SpeedCheckResult(
                description=chk.track_hint, approx_speed_kmh=chk.approx_speed_kmh,
                tolerance_kmh=chk.tolerance_kmh, hint_time_s=None,
                measured_speed_kmh=None, matched_event_clip_ts=None,
                status="not measured",
                reason='could not parse a time from track_hint (expected "t=MM:SS" or "t=SS")',
            ))
            continue
        if not candidates:
            results.append(SpeedCheckResult(
                description=chk.track_hint, approx_speed_kmh=chk.approx_speed_kmh,
                tolerance_kmh=chk.tolerance_kmh, hint_time_s=chk.hint_time_s,
                measured_speed_kmh=None, matched_event_clip_ts=None,
                status="not measured",
                reason="no line-crossing event with a speed reading in this clip",
            ))
            continue
        nearest = min(candidates, key=lambda e: abs(e["_eval_clip_ts"] - chk.hint_time_s))
        measured = nearest["speed_kmh"]
        status = "pass" if abs(measured - chk.approx_speed_kmh) <= chk.tolerance_kmh else "fail"
        results.append(SpeedCheckResult(
            description=chk.track_hint, approx_speed_kmh=chk.approx_speed_kmh,
            tolerance_kmh=chk.tolerance_kmh, hint_time_s=chk.hint_time_s,
            measured_speed_kmh=measured, matched_event_clip_ts=nearest["_eval_clip_ts"],
            status=status,
        ))
    return results


def _whole_clip_estimate_results(
    manual_estimate: Dict[str, List[int]], count_events: List[dict]
) -> List[WholeClipEstimateResult]:
    """Compares predicted whole-clip totals (summed across every counting
    line - this estimate has no per-line granularity) against a rough,
    sparse-sampling range a human gave for the whole clip. This is a much
    weaker check than _count_results: no line, no MAE, just "did the total
    land in the ballpark a quick watch-through suggested." Never used for
    the pass/fail gate."""
    predicted_totals: Dict[str, int] = {}
    for e in count_events:
        predicted_totals[e["class"]] = predicted_totals.get(e["class"], 0) + 1

    out = []
    for cls, (lo, hi) in manual_estimate.items():
        predicted = predicted_totals.get(cls, 0)
        status = "within range" if lo <= predicted <= hi else "outside range"
        out.append(WholeClipEstimateResult(cls, predicted, lo, hi, status))
    return out


def _fragmentation_ratios(count_events: List[dict]) -> Dict[str, Optional[float]]:
    """Heuristic proxy, no ground truth of its own: crossings on the same
    line within FRAGMENTATION_GAP_S of each other are clustered as
    probably-one-real-vehicle re-acquired under a new track_id. ratio =
    crossings / clusters; near 1.0 is good, 3+ suggests the tracker is
    losing and re-acquiring vehicles near that line."""
    by_line: Dict[str, List[float]] = {}
    for e in count_events:
        by_line.setdefault(e["line"], []).append(e["_eval_clip_ts"])
    ratios: Dict[str, Optional[float]] = {}
    for line, times in by_line.items():
        times.sort()
        episodes = 1
        for prev, cur in zip(times, times[1:]):
            if cur - prev > FRAGMENTATION_GAP_S:
                episodes += 1
        ratios[line] = len(times) / episodes if episodes else None
    return ratios


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------

def discover_cases(eval_dir: Path) -> List[Path]:
    return sorted((eval_dir / "ground_truth").glob("*.json"))


def all_gate_failures(results: List[ClipResult]) -> List[str]:
    return [f for r in results for f in r.gate_failures]


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-dir", default=str(Path(__file__).parent),
                     help="defaults to tests/eval")
    a = ap.parse_args(argv)
    eval_dir = Path(a.eval_dir)
    clips_dir = eval_dir / "clips"

    gt_paths = discover_cases(eval_dir)
    if not gt_paths:
        print(f"no ground truth files found under {eval_dir / 'ground_truth'}")
        return 0

    results = [evaluate_clip(p, clips_dir) for p in gt_paths]

    print(render_terminal(results))

    results_dir = eval_dir / "results"
    results_dir.mkdir(exist_ok=True)
    out_path = results_dir / f"report_{time.strftime('%Y%m%d_%H%M%S')}.md"
    out_path.write_text(render_markdown(results))
    print(f"\nfull report written to {out_path}")

    failures = all_gate_failures(results)
    if failures:
        print("\nGATE: FAIL")
        for f in failures:
            print(f"  - {f}")
        return 1
    print("\nGATE: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
