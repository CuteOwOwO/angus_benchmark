#!/usr/bin/env python3
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


PCM_BYTES_PER_SECOND = 48_000
AUDIO_SEGMENT_MERGE_GAP_MS = 200
AUDIO_BAR_HEIGHT = 0.12
CONDITIONS = ["periodic_tick_4s", "tick_after_audio_idle_repeat_0s", "tick_after_audio_idle_repeat_1s"]
LATENCIES = [8000, 12000]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def event_times(events: list[dict], *types: str) -> list[int]:
    return [
        int(event["event_ms"])
        for event in events
        if event.get("type") in types and event.get("event_ms") is not None
    ]


def first_event_time(events: list[dict], event_type: str) -> int | None:
    times = event_times(events, event_type)
    return times[0] if times else None


def merge_segments(chunks: list[dict]) -> list[dict]:
    segments = []
    for chunk in sorted(chunks, key=lambda item: item["start_ms"]):
        if not segments or chunk["start_ms"] - segments[-1]["end_ms"] > AUDIO_SEGMENT_MERGE_GAP_MS:
            segments.append({**chunk, "chunk_count": 1})
            continue
        segments[-1]["end_ms"] = max(segments[-1]["end_ms"], chunk["end_ms"])
        segments[-1]["chunk_count"] += 1
    return segments


def collect_attempt(attempt_dir: Path) -> dict | None:
    record_path = attempt_dir / "pilot_attempt_record.json"
    events_path = attempt_dir / "timeline" / "events.jsonl"
    if not record_path.exists() or not events_path.exists():
        return None
    record = read_json(record_path)
    events = read_jsonl(events_path)
    audio_chunks = []
    audio_cursor_ms = 0.0
    text_events = []
    for event in events:
        ms = event.get("event_ms")
        if ms is None:
            continue
        if event.get("type") == "audio_output" and event.get("bytes", 0) > 0:
            duration_ms = int(event["bytes"]) / PCM_BYTES_PER_SECOND * 1000
            start_ms = max(float(ms), audio_cursor_ms)
            end_ms = start_ms + duration_ms
            audio_cursor_ms = end_ms
            audio_chunks.append({"start_ms": int(start_ms), "end_ms": int(end_ms)})
        if event.get("type") in {"text_output", "output_transcription"} and event.get("text"):
            text_events.append({"time_ms": int(ms), "text": " ".join(str(event["text"]).split())})
    return {
        "name": attempt_dir.name,
        "condition": record["condition"],
        "latency_ms": int(record["latency_ms"]),
        "valid": bool(record["valid"]),
        "tool_call_ms": first_event_time(events, "tool_call_received"),
        "final_ms": first_event_time(events, "final_tool_response_sent"),
        "fixed_tick_times_ms": event_times(events, "client_status_tick_sent"),
        "boundary_detected_times_ms": event_times(events, "speech_boundary_detected"),
        "boundary_tick_times_ms": event_times(events, "boundary_client_status_tick_sent"),
        "turn_complete_times_ms": event_times(events, "turn_complete"),
        "close_times_ms": event_times(events, "session_closed"),
        "audio_segments": merge_segments(audio_chunks),
        "text_events": text_events,
    }


def collect(organized_dir: Path) -> list[dict]:
    out = []
    for attempt_dir in sorted((organized_dir / "per_attempt").glob("*")):
        item = collect_attempt(attempt_dir)
        if item:
            out.append(item)
    return out


def max_time(attempts: list[dict], minimum_ms: int) -> int:
    max_ms = minimum_ms
    for attempt in attempts:
        for key in ["tool_call_ms", "final_ms"]:
            if attempt.get(key) is not None:
                max_ms = max(max_ms, int(attempt[key]))
        for key in ["fixed_tick_times_ms", "boundary_detected_times_ms", "boundary_tick_times_ms", "turn_complete_times_ms", "close_times_ms"]:
            for time_ms in attempt.get(key, []):
                max_ms = max(max_ms, int(time_ms))
        for segment in attempt.get("audio_segments", []):
            max_ms = max(max_ms, int(segment["end_ms"]))
        for text in attempt.get("text_events", []):
            max_ms = max(max_ms, int(text["time_ms"]))
    return int(math.ceil((max_ms + 1000) / 1000) * 1000)


def draw_overlay(ax, attempts: list[dict], title: str, x_max_ms: int, show_xlabel: bool = True, show_y: bool = True) -> None:
    attempts = sorted(attempts, key=lambda item: item["name"])
    y_positions = list(range(len(attempts)))[::-1]
    ax.set_title(title, fontsize=12)
    ax.set_xlim(0, x_max_ms / 1000)
    ax.set_ylim(-0.5, max(0.5, len(attempts) - 0.5))
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    if show_xlabel:
        ax.set_xlabel("time since user prompt (seconds)")
    if show_y:
        ax.set_ylabel("attempt")
        ax.set_yticks(y_positions)
        ax.set_yticklabels([attempt["name"].split("__")[-1].replace("run_", "r") for attempt in attempts], fontsize=8)
    else:
        ax.set_yticks(y_positions)
        ax.set_yticklabels([])

    for y, attempt in zip(y_positions, attempts):
        ax.hlines(y, 0, x_max_ms / 1000, color="#C9CDD1", linewidth=0.8, zorder=0)
        if attempt.get("tool_call_ms") is not None:
            ax.vlines(attempt["tool_call_ms"] / 1000, y - 0.24, y + 0.24, color="#7B3FCE", linewidth=1.2, zorder=3)
        if attempt.get("final_ms") is not None:
            ax.vlines(attempt["final_ms"] / 1000, y - 0.24, y + 0.24, color="#E53935", linewidth=1.4, zorder=3)
        for tick in attempt.get("fixed_tick_times_ms", []):
            ax.vlines(tick / 1000, y - 0.21, y + 0.21, color="#777777", linewidth=1.1, linestyles=":", zorder=3)
        for tick in attempt.get("boundary_tick_times_ms", []):
            ax.vlines(tick / 1000, y - 0.23, y + 0.23, color="#16A34A", linewidth=1.3, linestyles="--", zorder=4)
        for boundary in attempt.get("boundary_detected_times_ms", []):
            ax.scatter([boundary / 1000], [y], marker="v", color="#16A34A", edgecolor="white", linewidth=0.4, s=24, zorder=5)
        for segment in attempt.get("audio_segments", []):
            ax.broken_barh(
                [(segment["start_ms"] / 1000, (segment["end_ms"] - segment["start_ms"]) / 1000)],
                (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
                facecolors="#1E88E5",
                edgecolors="#0D47A1",
                linewidth=0.65,
                zorder=2,
            )
        if attempt.get("text_events"):
            ax.scatter(
                [text["time_ms"] / 1000 for text in attempt["text_events"]],
                [y] * len(attempt["text_events"]),
                color="#F57C00",
                edgecolor="white",
                linewidth=0.3,
                s=18,
                zorder=4,
            )
        for close in attempt.get("close_times_ms", []):
            ax.scatter([close / 1000], [y], marker="x", color="#D81B60", s=34, linewidth=1.4, zorder=5)
        if not attempt.get("valid"):
            ax.text(x_max_ms / 1000 - 0.1, y + 0.13, "not valid", ha="right", va="bottom", fontsize=7, color="#9E2A2B")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def legend_handles():
    return [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio output"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F57C00", label="text/transcription", markersize=6),
        plt.Line2D([], [], color="#777777", linestyle=":", label="fixed pending tick"),
        plt.Line2D([], [], color="#16A34A", linestyle="--", label="boundary pending tick"),
        plt.Line2D([], [], marker="v", color="w", markerfacecolor="#16A34A", markeredgecolor="#16A34A", label="boundary detected", markersize=6),
        plt.Line2D([], [], color="#7B3FCE", label="tool call"),
        plt.Line2D([], [], color="#E53935", label="final tool response"),
    ]


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: plot_boundary_tick_policy_pilot.py <organized_dir>")
    organized_dir = Path(sys.argv[1]).resolve()
    out_dir = organized_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)
    attempts = collect(organized_dir)
    paths = []

    for condition in CONDITIONS:
        for latency in LATENCIES:
            cell = [attempt for attempt in attempts if attempt["condition"] == condition and attempt["latency_ms"] == latency]
            if not cell:
                continue
            x_max = max_time(cell, latency + 10_000)
            height = max(4.2, 1.2 + len(cell) * 0.36)
            fig, ax = plt.subplots(figsize=(14, height))
            draw_overlay(ax, cell, f"{condition}, latency {latency}ms", x_max)
            ax.legend(handles=legend_handles(), loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=4, fontsize=8, frameon=False)
            fig.subplots_adjust(bottom=0.2)
            path = out_dir / f"overlay_condition_{condition}_latency_{latency}ms.png"
            fig.savefig(path, dpi=160, bbox_inches="tight")
            plt.close(fig)
            paths.append(path)

    fig, axes = plt.subplots(len(CONDITIONS), len(LATENCIES), figsize=(18, 11), sharex=False)
    for row, condition in enumerate(CONDITIONS):
        for col, latency in enumerate(LATENCIES):
            ax = axes[row][col]
            cell = [attempt for attempt in attempts if attempt["condition"] == condition and attempt["latency_ms"] == latency]
            draw_overlay(
                ax,
                cell,
                f"{condition}\n{latency}ms",
                max_time(cell, latency + 10_000) if cell else latency + 10_000,
                show_xlabel=(row == len(CONDITIONS) - 1),
                show_y=(col == 0),
            )
    fig.legend(handles=legend_handles(), loc="lower center", ncol=7, frameon=False, fontsize=9)
    fig.suptitle("Boundary tick policy pilot timeline overlays", fontsize=16)
    fig.subplots_adjust(top=0.9, bottom=0.08, wspace=0.14, hspace=0.42)
    gallery = out_dir / "boundary_tick_policy_overlay_gallery.png"
    fig.savefig(gallery, dpi=160, bbox_inches="tight")
    plt.close(fig)
    paths.append(gallery)

    print("Boundary tick pilot visualizations:")
    for path in paths:
        print(f"- {path}")


if __name__ == "__main__":
    main()
