#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path("/user_data/gemini-live-check/result/2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark")
ORG = ROOT / "organized"
OUT = ROOT / "corrected_activity_metrics"
CONDITIONS = [("no_tick", "native_no_tick"), ("with_tick", "external_single_tick")]
LATENCIES = [3000, 5000, 8000, 12000]
TEXT_EVENT_TYPES = {"text_output", "output_transcription"}
PCM_BYTES_PER_SECOND = 48_000


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    with path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def mean(values: list[float]) -> float | None:
    return round(sum(values) / len(values), 3) if values else None


def med(values: list[float]) -> float | None:
    return round(median(values), 3) if values else None


def first_event_ms(events: list[dict], event_type: str) -> int | None:
    for event in events:
        if event.get("type") == event_type and isinstance(event.get("event_ms"), (int, float)):
            return int(event["event_ms"])
    return None


def actual_tick_times(events: list[dict]) -> list[int]:
    return [
        int(event["event_ms"])
        for event in events
        if event.get("type") == "client_status_tick_sent" and isinstance(event.get("event_ms"), (int, float))
    ]


def count_pre_result_text_events(events: list[dict], start_ms: int, final_ms: int) -> int:
    return sum(
        1
        for event in events
        if event.get("type") in TEXT_EVENT_TYPES
        and isinstance(event.get("event_ms"), (int, float))
        and start_ms <= int(event["event_ms"]) < final_ms
    )


def merged_audio_segments_before_final(events: list[dict], start_ms: int, final_ms: int, merge_gap_ms: int = 200) -> int:
    chunks: list[tuple[int, int]] = []
    for event in events:
        if event.get("type") != "audio_output" or not isinstance(event.get("event_ms"), (int, float)):
            continue
        event_ms = int(event["event_ms"])
        if not (start_ms <= event_ms < final_ms):
            continue
        byte_count = int(event.get("bytes") or 0)
        duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
        chunks.append((event_ms, int(event_ms + duration_ms)))
    segments: list[list[int]] = []
    for start, end in sorted(chunks):
        if not segments or start - segments[-1][1] > merge_gap_ms:
            segments.append([start, end])
        else:
            segments[-1][1] = max(segments[-1][1], end)
    return len(segments)


def audio_intervals(events: list[dict]) -> list[tuple[float, float]]:
    intervals: list[tuple[float, float]] = []
    for event in events:
        if event.get("type") != "audio_output" or not isinstance(event.get("event_ms"), (int, float)):
            continue
        byte_count = int(event.get("bytes") or 0)
        if byte_count <= 0:
            continue
        start = float(event["event_ms"])
        duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
        intervals.append((start, start + duration_ms))
    return intervals


def clipped_union_duration_ms(intervals: list[tuple[float, float]], window_start_ms: float, window_end_ms: float) -> float | None:
    if window_end_ms <= window_start_ms:
        return None
    clipped: list[tuple[float, float]] = []
    for start, end in intervals:
        clipped_start = max(start, window_start_ms)
        clipped_end = min(end, window_end_ms)
        if clipped_end > clipped_start:
            clipped.append((clipped_start, clipped_end))
    if not clipped:
        return 0.0
    clipped.sort()
    merged: list[list[float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return sum(end - start for start, end in merged)


def merged_clipped_intervals(intervals: list[tuple[float, float]], window_start_ms: float, window_end_ms: float) -> list[tuple[float, float]]:
    if window_end_ms <= window_start_ms:
        return []
    clipped: list[tuple[float, float]] = []
    for start, end in intervals:
        clipped_start = max(start, window_start_ms)
        clipped_end = min(end, window_end_ms)
        if clipped_end > clipped_start:
            clipped.append((clipped_start, clipped_end))
    clipped.sort()
    merged: list[list[float]] = []
    for start, end in clipped:
        if not merged or start > merged[-1][1]:
            merged.append([start, end])
        else:
            merged[-1][1] = max(merged[-1][1], end)
    return [(start, end) for start, end in merged]


def max_silent_gap_ms(intervals: list[tuple[float, float]], window_start_ms: float, window_end_ms: float) -> float | None:
    if window_end_ms <= window_start_ms:
        return None
    merged = merged_clipped_intervals(intervals, window_start_ms, window_end_ms)
    cursor = window_start_ms
    max_gap = 0.0
    for start, end in merged:
        max_gap = max(max_gap, start - cursor)
        cursor = max(cursor, end)
    max_gap = max(max_gap, window_end_ms - cursor)
    return max_gap


def occupancy_ratio(audio_ms: float | None, denominator_ms: int | None) -> float | str:
    if audio_ms is None or denominator_ms is None or denominator_ms <= 0:
        return ""
    return round(audio_ms / denominator_ms, 4)


def tick_status(condition_label: str, latency: int, ticks: list[int]) -> str:
    if condition_label == "no_tick":
        return "not_applicable"
    if latency <= 4000:
        return "tick_skipped_final_before_tick"
    return "actual_tick_sent" if ticks else "tick_expected_but_not_sent"


def attempt_dirs(condition_label: str, latency: int) -> list[Path]:
    return sorted((ORG / condition_label / f"{latency}ms").glob("attempt_*"))


def collect() -> tuple[list[dict], list[dict]]:
    attempt_rows: list[dict] = []
    summary_rows: list[dict] = []
    for condition_label, condition_name in CONDITIONS:
        for latency in LATENCIES:
            cell_rows: list[dict] = []
            for attempt_dir in attempt_dirs(condition_label, latency):
                summary_path = attempt_dir / "formal_attempt_summary.json"
                timeline_path = attempt_dir / "timeline" / "events.jsonl"
                if not summary_path.exists() or not timeline_path.exists():
                    continue
                summary = read_json(summary_path)
                if summary.get("close_1008"):
                    continue
                events = read_jsonl(timeline_path)
                start_ms = first_event_ms(events, "user_message_sent") or 0
                final_ms = summary.get("final_tool_response_sent_time_ms") or first_event_ms(events, "final_tool_response_sent")
                if not isinstance(final_ms, (int, float)):
                    continue
                final_ms = int(final_ms)
                window_ms = max(0, final_ms - start_ms)
                event_count = count_pre_result_text_events(events, start_ms, final_ms)
                event_rate = event_count / (window_ms / 1000) if window_ms > 0 else math.nan
                audio_segments = merged_audio_segments_before_final(events, start_ms, final_ms)
                intervals = audio_intervals(events)
                tool_call_ms = summary.get("tool_call_time_ms") or first_event_ms(events, "tool_call_received")
                tool_call_ms = int(tool_call_ms) if isinstance(tool_call_ms, (int, float)) else None
                tool_to_final_window_ms = final_ms - tool_call_ms if tool_call_ms is not None else None
                start_to_final_audio_ms = clipped_union_duration_ms(intervals, start_ms, final_ms)
                tool_to_final_audio_ms = clipped_union_duration_ms(intervals, tool_call_ms, final_ms) if tool_call_ms is not None else None
                max_start_to_final_silent_gap_ms = max_silent_gap_ms(intervals, start_ms, final_ms)
                ticks = actual_tick_times(events)
                row = {
                    "condition": condition_name,
                    "condition_label": condition_label,
                    "latency_ms": latency,
                    "attempt": attempt_dir.name,
                    "session_valid": bool(summary.get("session_valid")),
                    "close_1008": bool(summary.get("close_1008")),
                    "user_prompt_sent_time_ms": start_ms,
                    "tool_call_time_ms": tool_call_ms if tool_call_ms is not None else "",
                    "final_tool_response_sent_time_ms": final_ms,
                    "pre_result_window_ms": window_ms,
                    "pre_result_text_transcription_event_count": event_count,
                    "pre_result_text_transcription_event_rate_per_sec": round(event_rate, 3) if not math.isnan(event_rate) else "",
                    "pre_result_merged_audio_segment_count": audio_segments,
                    "assistant_audio_duration_from_start_to_final_ms": round(start_to_final_audio_ms, 3) if start_to_final_audio_ms is not None else "",
                    "assistant_audio_duration_from_tool_call_to_final_ms": round(tool_to_final_audio_ms, 3) if tool_to_final_audio_ms is not None else "",
                    "tool_call_to_final_window_ms": tool_to_final_window_ms if tool_to_final_window_ms is not None and tool_to_final_window_ms > 0 else "",
                    "audio_occupancy_ratio_from_start_to_final": occupancy_ratio(start_to_final_audio_ms, window_ms),
                    "audio_occupancy_ratio_from_tool_call_to_final": occupancy_ratio(tool_to_final_audio_ms, tool_to_final_window_ms),
                    "max_silent_gap_ms_from_start_to_final": round(max_start_to_final_silent_gap_ms, 3) if max_start_to_final_silent_gap_ms is not None else "",
                    "actual_tick_count": len(ticks),
                    "actual_tick_times_ms": ";".join(str(t) for t in ticks),
                    "tick_status": tick_status(condition_label, latency, ticks),
                    "post_tool_final_answer": bool(summary.get("post_tool_final_answer")),
                }
                attempt_rows.append(row)
                cell_rows.append(row)

            event_counts = [float(row["pre_result_text_transcription_event_count"]) for row in cell_rows]
            rates = [float(row["pre_result_text_transcription_event_rate_per_sec"]) for row in cell_rows if row["pre_result_text_transcription_event_rate_per_sec"] != ""]
            audio_counts = [float(row["pre_result_merged_audio_segment_count"]) for row in cell_rows]
            start_occ = [float(row["audio_occupancy_ratio_from_start_to_final"]) for row in cell_rows if row["audio_occupancy_ratio_from_start_to_final"] != ""]
            tool_occ = [float(row["audio_occupancy_ratio_from_tool_call_to_final"]) for row in cell_rows if row["audio_occupancy_ratio_from_tool_call_to_final"] != ""]
            max_gaps = [float(row["max_silent_gap_ms_from_start_to_final"]) for row in cell_rows if row["max_silent_gap_ms_from_start_to_final"] != ""]
            summary_rows.append(
                {
                    "condition": condition_name,
                    "condition_label": condition_label,
                    "latency_ms": latency,
                    "non_1008_attempts": len(cell_rows),
                    "valid_attempts": sum(1 for row in cell_rows if row["session_valid"]),
                    "valid_runs": sum(1 for row in cell_rows if row["session_valid"]),
                    "tick_status_values": ";".join(sorted({str(row["tick_status"]) for row in cell_rows})),
                    "avg_pre_result_text_transcription_event_count": mean(event_counts),
                    "median_pre_result_text_transcription_event_count": med(event_counts),
                    "avg_pre_result_text_transcription_event_rate_per_sec": mean(rates),
                    "median_pre_result_text_transcription_event_rate_per_sec": med(rates),
                    "avg_pre_result_merged_audio_segment_count": mean(audio_counts),
                    "median_pre_result_merged_audio_segment_count": med(audio_counts),
                    "mean_audio_occupancy_ratio_from_start_to_final": mean(start_occ),
                    "median_audio_occupancy_ratio_from_start_to_final": med(start_occ),
                    "mean_audio_occupancy_ratio_from_tool_call_to_final": mean(tool_occ),
                    "median_audio_occupancy_ratio_from_tool_call_to_final": med(tool_occ),
                    "mean_max_silent_gap_ms_from_start_to_final": mean(max_gaps),
                    "median_max_silent_gap_ms_from_start_to_final": med(max_gaps),
                }
            )
    return attempt_rows, summary_rows


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def grouped_bar(rows: list[dict], key: str, title: str, ylabel: str, out_path: Path) -> None:
    latencies = LATENCIES
    conditions = ["native_no_tick", "external_single_tick"]
    x = np.arange(len(latencies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, condition in enumerate(conditions):
        values = []
        for latency in latencies:
            row = next((item for item in rows if item["condition"] == condition and item["latency_ms"] == latency), None)
            values.append(float(row[key]) if row and row[key] is not None else float("nan"))
        ax.bar(x + (index - 0.5) * width, values, width, label=condition)
    ax.set_title(title)
    ax.set_xlabel("latency_ms")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x)
    ax.set_xticklabels([str(latency) for latency in latencies])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_md(path: Path, summary_rows: list[dict]) -> None:
    lines = [
        "# Corrected Activity Metrics",
        "",
        "`pre_result_output_frequency` from the original formal summary is ambiguous. It counted `text_output` + `output_transcription` events before final tool response and divided by the pre-result window. That is an event-fragment rate, not utterance frequency.",
        "",
        "Corrected definitions used here:",
        "",
        "- `pre_result_window_ms = final_tool_response_sent_time_ms - user_prompt_sent_time_ms`.",
        "- `pre_result_text_transcription_event_count` counts timeline events where type is `text_output` or `output_transcription` before final tool response.",
        "- `pre_result_text_transcription_event_rate_per_sec = event_count / (pre_result_window_ms / 1000)`.",
        "- `pre_result_merged_audio_segment_count` is an approximate spoken-activity count based on merged audio chunks before final; it is closer to utterance activity than transcription fragment count, but still not ASR sentence segmentation.",
        "- `audio_occupancy_ratio_from_start_to_final = union assistant audio duration between user prompt and final tool response / (final_tool_response_sent_time_ms - user_prompt_sent_time_ms)`.",
        "- `audio_occupancy_ratio_from_tool_call_to_final = union assistant audio duration between tool call and final tool response / (final_tool_response_sent_time_ms - tool_call_time_ms)`.",
        "- `max_silent_gap_ms_from_start_to_final` is the longest no-assistant-audio gap between user prompt and final tool response.",
        "- Audio occupancy uses assistant `audio_output` intervals from the timeline. If an audio interval crosses a window boundary, only the overlap is counted. If the denominator is <= 0, the per-attempt value is left blank.",
        "- `external_single_tick` at 3000 ms is marked `tick_skipped_final_before_tick`; no actual tick was sent.",
        "",
        "| condition | latency_ms | non_1008 | valid | tick_status | avg event count | avg event rate/sec | avg merged audio segments | start-to-final occupancy | tool-call-to-final occupancy | mean max silent gap ms |",
        "|---|---:|---:|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['latency_ms']} | {row['non_1008_attempts']} | {row['valid_attempts']} | "
            f"{row['tick_status_values']} | {row['avg_pre_result_text_transcription_event_count']} | "
            f"{row['avg_pre_result_text_transcription_event_rate_per_sec']} | {row['avg_pre_result_merged_audio_segment_count']} | "
            f"{row['mean_audio_occupancy_ratio_from_start_to_final']} | {row['mean_audio_occupancy_ratio_from_tool_call_to_final']} | "
            f"{row['mean_max_silent_gap_ms_from_start_to_final']} |"
        )
    lines.extend(
        [
            "",
            "Charts:",
            "",
            "- `avg_pre_result_text_transcription_event_count.png`",
            "- `avg_pre_result_text_transcription_event_rate_per_sec.png`",
            "- `avg_pre_result_merged_audio_segment_count.png`",
            "- `audio_occupancy_ratio_from_start_to_final.png`",
            "- `audio_occupancy_ratio_from_tool_call_to_final.png`",
            "- `max_silent_gap_ms_from_start_to_final.png`",
            "",
        ]
    )
    path.write_text("\n".join(lines))


def main() -> None:
    OUT.mkdir(exist_ok=True)
    viz_dir = ROOT / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    attempt_rows, summary_rows = collect()
    write_csv(OUT / "attempt_activity_metrics_non_1008.csv", attempt_rows)
    write_csv(OUT / "summary_activity_metrics_non_1008.csv", summary_rows)
    write_csv(ROOT / "waiting_activity_summary.csv", summary_rows)
    write_md(OUT / "README.md", summary_rows)
    grouped_bar(
        summary_rows,
        "avg_pre_result_text_transcription_event_count",
        "Avg Pre-result Text/Transcription Event Count",
        "avg event count",
        OUT / "avg_pre_result_text_transcription_event_count.png",
    )
    grouped_bar(
        summary_rows,
        "avg_pre_result_text_transcription_event_rate_per_sec",
        "Avg Pre-result Text/Transcription Event Rate",
        "avg events / second",
        OUT / "avg_pre_result_text_transcription_event_rate_per_sec.png",
    )
    grouped_bar(
        summary_rows,
        "avg_pre_result_merged_audio_segment_count",
        "Avg Pre-result Merged Audio Segment Count",
        "avg merged audio segments",
        OUT / "avg_pre_result_merged_audio_segment_count.png",
    )
    for target_dir in (OUT, viz_dir):
        grouped_bar(
            summary_rows,
            "mean_audio_occupancy_ratio_from_start_to_final",
            "Audio Occupancy Ratio: User Prompt to Final Tool Response",
            "mean audio occupancy ratio",
            target_dir / "audio_occupancy_ratio_from_start_to_final.png",
        )
        grouped_bar(
            summary_rows,
            "mean_audio_occupancy_ratio_from_tool_call_to_final",
            "Audio Occupancy Ratio: Tool Call to Final Tool Response",
            "mean audio occupancy ratio",
            target_dir / "audio_occupancy_ratio_from_tool_call_to_final.png",
        )
        grouped_bar(
            summary_rows,
            "mean_max_silent_gap_ms_from_start_to_final",
            "Max Silent Gap: User Prompt to Final Tool Response",
            "mean max silent gap (ms)",
            target_dir / "max_silent_gap_ms_from_start_to_final.png",
        )
    print(OUT)


if __name__ == "__main__":
    main()
