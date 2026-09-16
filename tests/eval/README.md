# KT-Radar eval harness

This checks whether the traffic-counting system actually gets the right
answer on real video, not just whether the code runs. You don't need to
know how to program to add a clip - follow the steps below.

## What you need before you start

- A short video clip (a phone recording is fine), ideally 1-3 minutes.
- For each vehicle-counting line you care about in that clip, a manual
  count: how many of each vehicle type actually crossed it, counted by
  eye (or by watching the video and tallying on paper).
- Optionally, one or two moments where you know roughly how fast a
  vehicle was going (e.g. you paced it in your own car and read your
  speedometer, or you know the road's design speed at that point).

You do **not** need both a count and a speed check - either one alone is
enough to add a clip. Start with whatever you have.

## Step 1: add the video file

Drop your clip into `tests/eval/clips/`, for example:

```
tests/eval/clips/day_normal_01.mp4
```

Naming convention: `<condition>_<description>_<number>.mp4`, where
`<condition>` is one of `day`, `night`, `rain`, `jam`. It doesn't have to
be exact - it's just for grouping clips in the report.

Video files are not tracked in git (they're too large) - they just live
on your machine, or wherever you're running the harness.

## Step 2: write the ground truth file

Create a matching JSON file in `tests/eval/ground_truth/`, named after
the clip: `tests/eval/ground_truth/day_normal_01.json`.

The smallest useful version is just this - one manual count for one line:

```json
{
  "clip": "day_normal_01.mp4",
  "condition": "day",
  "site_config": "config/site_example.yaml",
  "manual_line_counts": {
    "cordon": {"car": 12, "motorcycle": 30}
  }
}
```

- `clip`: the exact filename you used in Step 1.
- `condition`: `day`, `night`, `rain`, or `jam` (whatever best describes it).
- `site_config`: which camera calibration file to use. If you don't have
  a real calibrated site yet, leave this as `config/site_example.yaml`
  (a fake/test calibration) - the harness will still run, it just won't
  reflect a real camera's geometry.
- `manual_line_counts`: for each counting line name (defined in the site
  config), how many of each vehicle class you counted crossing it by eye.
  Use whatever class names the site config's `vehicle_classes` lists (see
  below if you're not sure which those are).

### Adding a clip from a region with no fine-tuned model yet

Don't just copy the Karachi class list (`motorcycle, car, rickshaw,
qingqi, minibus, bus, pickup, truck, water_tanker, cart, pedestrian`)
into a site config for a different region - a generic detector doesn't
know most of those classes, and a fine-tuned one for THAT region doesn't
exist yet either. Instead, point `site_config` at a config with
`class_source: coco_subset` and a `vehicle_classes` list limited to what
a generic detector can actually see there - usually some of `car, bus,
truck, motorcycle, bicycle, pedestrian`. `config/site_example_coco_only.yaml`
is a template for exactly this. The system will warn (not silently miss)
if the detector and `vehicle_classes` don't line up - see its own header
comment for the "person" vs "pedestrian" naming note.

If you also timed a vehicle's speed, add this:

```json
  "known_speed_checks": [
    {
      "track_hint": "silver sedan entering frame at t=00:14",
      "approx_speed_kmh": 38,
      "tolerance_kmh": 6,
      "source": "paced alongside in another vehicle's speedometer"
    }
  ]
```

- `track_hint`: describe which vehicle, in plain English, and **include
  a timestamp in the clip in the form `t=MM:SS`** (like `t=00:14` for 14
  seconds in) - the harness uses that to find the matching detection.
  Without a parseable `t=MM:SS`, this check will be reported as "not
  measured" rather than guessed.
- `approx_speed_kmh` / `tolerance_kmh`: your best estimate and how
  confident you are in it (±6 km/h in the example above).

Add a `notes` field for anything odd - shaky camera, weird lighting,
whatever you'd want a person reviewing this to know.

## Step 3: run it

From the repository root:

```bash
python -m tests.eval.run_eval
```

This prints a table (one row per clip) to the terminal and writes a
fuller report to `tests/eval/results/report_<date>_<time>.md`. It exits
with a non-zero status if the results are bad enough to fail the quality
gate (see below) - useful for CI, ignorable if you're just looking.

## Reading the output

- **One row per clip.** There is deliberately no single overall score -
  a system that's great by day and bad at night should look like that in
  the report, not average out to "pretty good."
- **count error**: how far the predicted count was from your manual
  count, per vehicle class per line. `!GATE` means it failed the quality
  gate (more than 20% off).
- **speed checks**: how many of your `known_speed_checks` came within
  tolerance. "not measured" means there was nothing to compare against
  (no speed checks provided, or the harness couldn't find a matching
  detection near the time you gave).
- **fragmentation**: a rough diagnostic, not a real measurement - how
  often the tracker seems to be losing and re-finding the same vehicle
  near a counting line (values near 1.0 are good; 3+ suggests real
  tracking trouble, likely from occlusion).
- **worst offenders**: the handful of specific clip/metric combinations
  that were most wrong, so you know exactly which few seconds of footage
  to go re-watch instead of everything.

## No trained model yet? That's fine

If there's no trained detector weights file yet, the harness can still
run against a clip using a **hand-annotated detections file** instead of
a real detector - a human marks where the vehicles are in each frame,
and the harness replays that. This is mostly for testing the harness
itself; if you're just adding real footage, you don't need to do this -
just follow Steps 1-3 above. Ask an engineer if you want to add one of
these for a specific clip.

## The synthetic placeholder clip

`tests/eval/ground_truth/synth_day_normal_01.json` is not real footage -
it's a synthetic clip generated by
`tests/eval/clips/generate_placeholder_clip.py`, with an exact,
known-in-advance answer, used to prove the harness itself works
correctly. If `tests/eval/clips/synth_day_normal_01.mp4` is missing
(it's gitignored, like every clip), regenerate it with:

```bash
python -m tests.eval.clips.generate_placeholder_clip
```

Replace it with real footage as soon as you have some - a synthetic clip
proves the harness works, not that the system works on a real road.
