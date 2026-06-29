#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


AUDIO_SEGMENT_MERGE_GAP_MS = 250
AUDIO_BAR_HEIGHT = 0.14


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def merge_segments(chunks: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for chunk in sorted(chunks, key=lambda item: item["start_ms"]):
        if not segments or chunk["start_ms"] - segments[-1]["end_ms"] > AUDIO_SEGMENT_MERGE_GAP_MS:
            segments.append(dict(chunk))
            continue
        segments[-1]["end_ms"] = max(segments[-1]["end_ms"], chunk["end_ms"])
    return segments


def collect_attempt(organized_dir: Path, attempt_index: int, latency_ms: int | None, task_slug: str) -> dict:
    latency_pattern = str(latency_ms) if latency_ms is not None else "*"
    summary_matches = sorted((organized_dir / "per_attempt").glob(f"{task_slug}__latency{latency_pattern}__attempt{attempt_index:03d}.summary.json"))
    timeline_matches = sorted((organized_dir / "timelines").glob(f"{task_slug}__latency{latency_pattern}__attempt{attempt_index:03d}.timeline.jsonl"))
    if not summary_matches or not timeline_matches:
        raise FileNotFoundError(f"Missing summary/timeline for attempt {attempt_index:03d} in {organized_dir}")
    summary_path = summary_matches[0]
    timeline_path = timeline_matches[0]
    summary = read_json(summary_path)
    events = read_jsonl(timeline_path)

    audio_chunks = [
        {"start_ms": int(event["start_ms"]), "end_ms": int(event["end_ms"])}
        for event in events
        if event.get("type") == "audio_output"
        and event.get("start_ms") is not None
        and event.get("end_ms") is not None
    ]
    pending_ticks = [
        int(event["event_ms"])
        for event in events
        if event.get("type") == "pending_tick_sent" and event.get("event_ms") is not None
    ]

    return {
        "attempt_index": attempt_index,
        "latency_ms": summary.get("latency_ms"),
        "valid": bool(summary.get("valid_run")),
        "sequential": bool(summary.get("sequential_two_tool_success")),
        "parallel": bool(summary.get("parallel_tool_call_detected")),
        "calendar_call_ms": summary.get("tool1_call_time_ms"),
        "calendar_result_ms": summary.get("tool1_result_sent_time_ms"),
        "route_call_ms": summary.get("tool2_call_time_ms"),
        "route_result_ms": summary.get("tool2_result_sent_time_ms"),
        "pending_tick_times_ms": pending_ticks,
        "audio_segments": merge_segments(audio_chunks),
    }


def max_time(attempts: list[dict]) -> int:
    max_ms = 0
    for attempt in attempts:
        for key in ["calendar_call_ms", "calendar_result_ms", "route_call_ms", "route_result_ms"]:
            if attempt.get(key) is not None:
                max_ms = max(max_ms, int(attempt[key]))
        for segment in attempt["audio_segments"]:
            max_ms = max(max_ms, int(segment["end_ms"]))
        for tick in attempt.get("pending_tick_times_ms", []):
            max_ms = max(max_ms, int(tick))
    return int(math.ceil((max_ms + 1000) / 1000) * 1000)


def draw(attempts: list[dict], output_path: Path, title: str) -> None:
    x_max_ms = max_time(attempts)
    y_positions = list(range(len(attempts)))[::-1]

    fig, ax = plt.subplots(figsize=(14, max(4.2, 1.6 + 0.48 * len(attempts))))
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, x_max_ms / 1000)
    ax.set_ylim(-0.55, len(attempts) - 0.45)
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylabel("attempt")
    ax.set_yticks(y_positions)
    ax.set_yticklabels([f"{attempt['attempt_index']:03d}" for attempt in attempts])
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    for y, attempt in zip(y_positions, attempts):
        ax.hlines(y, 0, x_max_ms / 1000, color="#CBD5E1", linewidth=0.9, zorder=0)
        for segment in attempt["audio_segments"]:
            ax.broken_barh(
                [(segment["start_ms"] / 1000, (segment["end_ms"] - segment["start_ms"]) / 1000)],
                (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
                facecolors="#1E88E5",
                edgecolors="#0D47A1",
                linewidth=0.55,
                zorder=2,
            )

        markers = [
            ("calendar_call_ms", "#7B3FCE", "-", 1.5),
            ("calendar_result_ms", "#16A34A", "-", 1.5),
            ("route_call_ms", "#F57C00", "-", 1.5),
            ("route_result_ms", "#E53935", "-", 1.5),
        ]
        for key, color, linestyle, linewidth in markers:
            if attempt.get(key) is not None:
                ax.vlines(
                    int(attempt[key]) / 1000,
                    y - 0.27,
                    y + 0.27,
                    color=color,
                    linestyle=linestyle,
                    linewidth=linewidth,
                    zorder=3,
                )
        for tick in attempt.get("pending_tick_times_ms", []):
            ax.vlines(
                tick / 1000,
                y - 0.22,
                y + 0.22,
                color="#6B7280",
                linestyle=":",
                linewidth=1.15,
                zorder=3,
            )

        if attempt["parallel"]:
            ax.scatter([x_max_ms / 1000 - 0.4], [y], marker="x", color="#D81B60", s=40, zorder=4)
        elif attempt["sequential"]:
            ax.scatter([x_max_ms / 1000 - 0.4], [y], marker="o", color="#16A34A", s=20, zorder=4)

    handles = [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="assistant audio"),
        plt.Line2D([], [], color="#7B3FCE", label="calendar tool call"),
        plt.Line2D([], [], color="#16A34A", label="calendar result sent"),
        plt.Line2D([], [], color="#F57C00", label="route tool call"),
        plt.Line2D([], [], color="#E53935", label="route result sent"),
        plt.Line2D([], [], color="#6B7280", linestyle=":", label="pending tick"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#16A34A", label="sequential success", markersize=6),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.42), ncol=3, fontsize=9, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(bottom=0.34)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("organized_dir", type=Path)
    parser.add_argument("--first", type=int, default=1)
    parser.add_argument("--count", type=int, default=5)
    parser.add_argument("--latency-ms", type=int, default=None)
    parser.add_argument("--condition", default=None)
    parser.add_argument("--task-slug", default="*", help="Artifact filename prefix before __latency. Default: wildcard.")
    args = parser.parse_args()

    organized_dir = args.organized_dir.resolve()
    attempts = [collect_attempt(organized_dir, index, args.latency_ms, args.task_slug) for index in range(args.first, args.first + args.count)]
    latency_label = f"latency{args.latency_ms}" if args.latency_ms is not None else "all"
    output_prefix = args.task_slug if args.task_slug != "*" else "two_step_task"
    output_path = organized_dir / "visualizations" / f"{output_prefix}_{latency_label}_attempts_{args.first:03d}_{args.first + args.count - 1:03d}_timegraph.png"
    title_latency = f", latency {args.latency_ms}ms" if args.latency_ms is not None else ""
    title_condition = f", {args.condition}" if args.condition else ""
    draw(attempts, output_path, f"Calendar-route native two-tool{title_condition}{title_latency}: attempts {args.first:03d}-{args.first + args.count - 1:03d}")
    print(output_path)


if __name__ == "__main__":
    main()
