"""
KT-Radar :: eval scorecard rendering
---------------------------------------
Prints a terminal table and returns markdown for the same content. The
main table is always one row per clip, grouped by condition - never a
single aggregated top-line number, because an 88% aggregate that's 96%
by day and 61% at night is a night-time problem, not an 88% system. Any
aggregate is a separate, clearly-labelled, opt-in section.
"""

from __future__ import annotations

from typing import List

from tests.eval.results import ClipResult


def _count_summary(r: ClipResult) -> str:
    if not r.count_results:
        no_manual = "no manual_line_counts provided" in " ".join(r.skipped_checks)
        return "not measured" if no_manual else "n/a"
    parts = []
    for cr in r.count_results:
        pct = f"{cr.pct_error:.0%}" if cr.pct_error is not None else "n/a"
        flag = " !GATE" if cr.gate_fail else ""
        parts.append(
            f"{cr.line}/{cr.cls_name}: pred={cr.predicted} manual={cr.manual} err={pct}{flag}"
        )
    return "; ".join(parts)


def _speed_summary(r: ClipResult) -> str:
    if not r.speed_results:
        return "not measured"
    passed = sum(1 for s in r.speed_results if s.status == "pass")
    failed = sum(1 for s in r.speed_results if s.status == "fail")
    unmeasured = sum(1 for s in r.speed_results if s.status == "not measured")
    bits = []
    if passed or failed:
        bits.append(f"{passed}/{passed + failed} pass")
    if unmeasured:
        bits.append(f"{unmeasured} not measured")
    mae = f", MAE={r.speed_mae:.1f}km/h" if r.speed_mae is not None else ""
    return (", ".join(bits) or "not measured") + mae


def _fragmentation_summary(r: ClipResult) -> str:
    if not r.fragmentation:
        return "n/a"
    return "; ".join(f"{line}={ratio:.2f}" for line, ratio in sorted(r.fragmentation.items())
                      if ratio is not None)


def _worst_offenders(results: List[ClipResult], n: int = 3) -> List[str]:
    # count errors are scored as a share of the manual count (0-1+), speed
    # errors as a multiple of that check's own tolerance - different units,
    # but both are "how many times over what's acceptable", so ranking
    # them together is a reasonable way to surface the worst few.
    scored = []
    for r in results:
        for cr in r.count_results:
            if cr.pct_error is not None:
                scored.append((
                    cr.pct_error,
                    f"{r.clip_name}: count '{cr.cls_name}' on '{cr.line}' - "
                    f"{cr.pct_error:.0%} error (predicted {cr.predicted}, manual {cr.manual})",
                ))
        for sr in r.speed_results:
            if sr.measured_speed_kmh is not None:
                err = abs(sr.measured_speed_kmh - sr.approx_speed_kmh)
                scored.append((
                    err / max(sr.tolerance_kmh, 1e-6),
                    f"{r.clip_name}: speed check '{sr.description}' - {err:.1f} km/h off "
                    f"(measured {sr.measured_speed_kmh:.1f}, expected {sr.approx_speed_kmh:.1f})",
                ))
    scored.sort(key=lambda t: t[0], reverse=True)
    return [desc for _, desc in scored[:n]]


def _group_by_condition(results: List[ClipResult]) -> List[tuple]:
    conditions = sorted({r.condition for r in results})
    return [(c, [r for r in results if r.condition == c]) for c in conditions]


def _aggregate_lines(results: List[ClipResult]) -> List[str]:
    all_pct = [cr.pct_error for r in results for cr in r.count_results
               if cr.pct_error is not None]
    all_speed_mae = [r.speed_mae for r in results if r.speed_mae is not None]

    lines = ["aggregate (misleading if conditions differ - see per-clip results above)"]
    if all_pct:
        lines.append(f"  mean count error across all measured classes/lines: "
                     f"{sum(all_pct) / len(all_pct):.0%}")
    else:
        lines.append("  count error: not measured anywhere")
    if all_speed_mae:
        lines.append(f"  mean speed MAE across all clips with speed checks: "
                     f"{sum(all_speed_mae) / len(all_speed_mae):.1f} km/h")
    else:
        lines.append("  speed MAE: not measured anywhere")
    return lines


def _detail_lines(r: ClipResult) -> List[str]:
    lines = [f"--- {r.clip_name} ({r.condition}) - detector: {r.detector_mode} ---"]
    if r.notes:
        lines.append(f"  notes: {r.notes}")
    for cr in r.count_results:
        pct = f"{cr.pct_error:.0%}" if cr.pct_error is not None else "n/a (manual=0)"
        flag = "  <-- GATE FAIL" if cr.gate_fail else ""
        lines.append(f"  count  {cr.line}/{cr.cls_name}: predicted={cr.predicted} "
                     f"manual={cr.manual} abs_err={cr.abs_error} pct_err={pct}{flag}")
    for sr in r.speed_results:
        if sr.status == "not measured":
            lines.append(f"  speed  \"{sr.description}\": not measured ({sr.reason})")
        else:
            flag = "  <-- " + sr.status.upper()
            lines.append(
                f"  speed  \"{sr.description}\": measured={sr.measured_speed_kmh:.1f}km/h "
                f"expected={sr.approx_speed_kmh:.1f}km/h tol={sr.tolerance_kmh:.1f}km/h "
                f"at clip_t={sr.matched_event_clip_ts:.2f}s (hint_t={sr.hint_time_s:.2f}s){flag}"
            )
    for line, ratio in sorted(r.fragmentation.items()):
        if ratio is not None:
            lines.append(f"  fragmentation  {line}: {ratio:.2f} (crossings per probable vehicle)")
    for s in r.skipped_checks:
        lines.append(f"  skipped: {s}")
    return lines


def _main_table_rows(results: List[ClipResult]) -> List[List[str]]:
    rows = [["clip", "condition", "detector", "count error", "speed checks", "fragmentation"]]
    for _condition, group in _group_by_condition(results):
        for r in group:
            rows.append([
                r.clip_name, r.condition, r.detector_mode,
                _count_summary(r), _speed_summary(r), _fragmentation_summary(r),
            ])
    return rows


def render_terminal(results: List[ClipResult]) -> str:
    rows = _main_table_rows(results)
    widths = [max(len(row[i]) for row in rows) for i in range(len(rows[0]))]
    lines = []
    for i, row in enumerate(rows):
        lines.append("  ".join(cell.ljust(widths[j]) for j, cell in enumerate(row)))
        if i == 0:
            lines.append("  ".join("-" * w for w in widths))

    worst = _worst_offenders(results)
    if worst:
        lines.append("\nworst offenders:")
        for w in worst:
            lines.append(f"  - {w}")

    lines.append("")
    lines.extend(_aggregate_lines(results))

    lines.append("\nper-clip detail:")
    for r in results:
        lines.extend(_detail_lines(r))

    return "\n".join(lines)


def render_markdown(results: List[ClipResult]) -> str:
    rows = _main_table_rows(results)
    lines = ["# KT-Radar eval report", ""]
    lines.append("| " + " | ".join(rows[0]) + " |")
    lines.append("|" + "|".join(["---"] * len(rows[0])) + "|")
    for row in rows[1:]:
        lines.append("| " + " | ".join(row) + " |")

    worst = _worst_offenders(results)
    if worst:
        lines.append("\n## Worst offenders\n")
        for w in worst:
            lines.append(f"- {w}")

    lines.append("\n## Aggregate\n")
    lines.append("_(misleading if conditions differ - see the per-clip table above)_\n")
    for line in _aggregate_lines(results)[1:]:
        lines.append(f"- {line.strip()}")

    lines.append("\n## Per-clip detail\n")
    for r in results:
        lines.append("```")
        lines.extend(_detail_lines(r))
        lines.append("```")

    from tests.eval.run_eval import all_gate_failures
    failures = all_gate_failures(results)
    lines.append("\n## Gate\n")
    if failures:
        lines.append("**FAIL**\n")
        for f in failures:
            lines.append(f"- {f}")
    else:
        lines.append("**PASS**")

    return "\n".join(lines)
