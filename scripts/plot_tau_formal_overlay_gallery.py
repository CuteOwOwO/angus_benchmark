#!/usr/bin/env python3
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


ROOT = Path("/user_data/gemini-live-check/result/2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark")
ORG = ROOT / "organized"
OUT = ROOT / "overlay_gallery_non_1008"

PCM_BYTES_PER_SECOND = 48_000
AUDIO_SEGMENT_MERGE_GAP_MS = 200
AUDIO_BAR_HEIGHT = 0.12

CONDITIONS = [
    ("no_tick", "no_tick"),
    ("with_tick", "with_tick"),
]
LATENCIES = [3000, 5000, 8000, 12000]


def read_json(path: Path) -> dict:
    return json.loads(path.read_text())


def read_jsonl(path: Path) -> list[dict]:
    events = []
    with path.open() as f:
        for line in f:
            if line.strip():
                events.append(json.loads(line))
    return events


def event_ms(event: dict):
    return event.get("event_ms", event.get("eventMs"))


def parts_from_server_event(event: dict) -> list[dict]:
    return (
        event.get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )


def merge_segments(chunks: list[dict]) -> list[dict]:
    segments = []
    for chunk in sorted(chunks, key=lambda item: item["start_ms"]):
        if not segments or chunk["start_ms"] - segments[-1]["end_ms"] > AUDIO_SEGMENT_MERGE_GAP_MS:
            segments.append({**chunk, "chunk_count": 1})
            continue
        segments[-1]["end_ms"] = max(segments[-1]["end_ms"], chunk["end_ms"])
        segments[-1]["chunk_count"] += 1
    return segments


def attempt_sort_key(path: Path) -> int:
    try:
        return int(path.name.replace("attempt_", ""))
    except ValueError:
        return 0


def collect_attempt(attempt_dir: Path) -> dict | None:
    summary_path = attempt_dir / "formal_attempt_summary.json"
    if not summary_path.exists():
        summary_path = attempt_dir / "summary.json"
    if not summary_path.exists() or not (attempt_dir / "raw_log.jsonl").exists():
        return None
    summary = read_json(summary_path)
    if summary.get("close_1008"):
        return None

    events = read_jsonl(attempt_dir / "raw_log.jsonl")
    post_path = attempt_dir / "postprocessed_summary.json"
    post = read_json(post_path) if post_path.exists() else {}

    audio_chunks = []
    text_events = []
    tick_times = list(post.get("tick_times_ms") or [])
    final_time = post.get("final_tool_response_ms") or summary.get("final_tool_response_sent_time_ms")
    close_events = list(post.get("close_events") or [])

    for event in events:
        ms = event_ms(event)
        if ms is None:
            continue
        if event.get("type") == "client_status_tick_sent":
            tick_times.append(ms)
        if event.get("type") == "final_tool_response_sent":
            final_time = ms
        if event.get("type") in {"close", "session_closed", "socket_closed"}:
            close_events.append({"time_ms": ms, "code": event.get("code") or event.get("close_code")})

        if event.get("type") != "server_event":
            continue
        server = event.get("message", {}).get("serverContent", {})
        out_tx = server.get("outputTranscription", {})
        if out_tx.get("text"):
            text_events.append({"time_ms": ms, "text": out_tx["text"]})
        for part in parts_from_server_event(event):
            if part.get("text"):
                text_events.append({"time_ms": ms, "text": part["text"]})
            inline = part.get("inlineData", {})
            byte_count = inline.get("bytes", 0)
            if byte_count:
                duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
                audio_chunks.append(
                    {
                        "start_ms": ms,
                        "end_ms": ms + duration_ms,
                    }
                )

    return {
        "name": attempt_dir.name,
        "valid": bool(summary.get("session_valid")),
        "tool_call_ms": summary.get("tool_call_time_ms"),
        "final_ms": final_time,
        "tick_times_ms": sorted(set(round(t) for t in tick_times)),
        "turn_complete_ms": summary.get("turnComplete_time_ms"),
        "post_final_answer": bool(summary.get("post_tool_final_answer")),
        "audio_segments": merge_segments(audio_chunks),
        "text_events": text_events,
        "close_events": close_events,
    }


def collect_cell(condition_dir: Path) -> list[dict]:
    attempts = []
    for attempt_dir in sorted(condition_dir.glob("attempt_*"), key=attempt_sort_key):
        item = collect_attempt(attempt_dir)
        if item is not None:
            attempts.append(item)
    return attempts


def max_time_for_attempts(attempts: list[dict], minimum_ms: int) -> int:
    max_ms = minimum_ms
    for attempt in attempts:
        for key in ["tool_call_ms", "final_ms", "turn_complete_ms"]:
            if attempt.get(key) is not None:
                max_ms = max(max_ms, int(attempt[key]))
        for tick in attempt.get("tick_times_ms", []):
            max_ms = max(max_ms, int(tick))
        for text in attempt.get("text_events", []):
            max_ms = max(max_ms, int(text["time_ms"]))
        for segment in attempt.get("audio_segments", []):
            max_ms = max(max_ms, int(segment["end_ms"]))
        for close in attempt.get("close_events", []):
            if close.get("time_ms") is not None:
                max_ms = max(max_ms, int(close["time_ms"]))
    return int(math.ceil((max_ms + 500) / 1000.0) * 1000)


def draw_overlay(ax, attempts: list[dict], title: str, x_max_ms: int, show_xlabel: bool = True) -> None:
    y_positions = list(range(len(attempts)))[::-1]
    ax.set_title(title, fontsize=14)
    ax.set_xlim(0, x_max_ms / 1000)
    ax.set_ylim(-0.5, max(0.5, len(attempts) - 0.5))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([attempt["name"].replace("attempt_", "run_") for attempt in attempts], fontsize=8)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    if show_xlabel:
        ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylabel("run attempt")

    for y, attempt in zip(y_positions, attempts):
        ax.hlines(y, 0, x_max_ms / 1000, color="#C9CDD1", linewidth=0.8, zorder=0)
        tool_call = attempt.get("tool_call_ms")
        final_ms = attempt.get("final_ms")
        if tool_call is not None:
            ax.vlines(tool_call / 1000, y - 0.24, y + 0.24, color="#7B3FCE", linewidth=1.2, zorder=3)
        if final_ms is not None:
            ax.vlines(final_ms / 1000, y - 0.24, y + 0.24, color="#E53935", linewidth=1.4, zorder=3)
        for tick in attempt.get("tick_times_ms", []):
            ax.vlines(tick / 1000, y - 0.21, y + 0.21, color="#777777", linewidth=1.1, linestyles=":", zorder=3)
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
            xs = [text["time_ms"] / 1000 for text in attempt["text_events"]]
            ys = [y] * len(xs)
            ax.scatter(xs, ys, color="#F57C00", edgecolor="white", linewidth=0.3, s=18, zorder=4)
        for close in attempt.get("close_events", []):
            if close.get("time_ms") is not None:
                ax.scatter([close["time_ms"] / 1000], [y], marker="x", color="#D81B60", s=34, linewidth=1.4, zorder=5)
        if not attempt.get("valid"):
            ax.text(x_max_ms / 1000 - 0.1, y + 0.13, "not valid", ha="right", va="bottom", fontsize=7, color="#9E2A2B")

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)


def save_cell_plot(condition_label: str, latency: int, attempts: list[dict], x_max_ms: int) -> Path:
    height = max(4.6, 1.25 + len(attempts) * 0.36)
    fig, ax = plt.subplots(figsize=(14.875, height))
    draw_overlay(
        ax,
        attempts,
        f"Overlay timeline: condition_{condition_label}, latency {latency}ms (non-1008 runs)",
        x_max_ms,
    )
    legend_handles = [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio output"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F57C00", label="text event", markersize=6),
        plt.Line2D([], [], color="#777777", linestyle=":", label="pending tick"),
        plt.Line2D([], [], color="#7B3FCE", label="tool call"),
        plt.Line2D([], [], color="#E53935", label="final tool response"),
        plt.Line2D([], [], marker="x", color="#D81B60", linestyle="None", label="session error/close", markersize=6),
    ]
    ax.legend(handles=legend_handles, loc="lower center", bbox_to_anchor=(0.5, -0.26), ncol=3, fontsize=9, frameon=False)
    fig.subplots_adjust(bottom=0.18)
    path = OUT / f"overlay_condition_{condition_label}_latency_{latency}ms.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def save_gallery(cell_data: dict[tuple[str, int], tuple[list[dict], int]]) -> Path:
    fig, axes = plt.subplots(2, 4, figsize=(22, 10), sharex=False)
    for row, (folder, label) in enumerate(CONDITIONS):
        for col, latency in enumerate(LATENCIES):
            attempts, x_max_ms = cell_data[(label, latency)]
            ax = axes[row][col]
            draw_overlay(ax, attempts, f"{label}, {latency}ms", x_max_ms, show_xlabel=(row == 1))
            if col != 0:
                ax.set_ylabel("")
                ax.set_yticklabels([])
    legend_handles = [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio output"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F57C00", label="text event", markersize=6),
        plt.Line2D([], [], color="#777777", linestyle=":", label="pending tick"),
        plt.Line2D([], [], color="#7B3FCE", label="tool call"),
        plt.Line2D([], [], color="#E53935", label="final tool response"),
        plt.Line2D([], [], marker="x", color="#D81B60", linestyle="None", label="session error/close", markersize=6),
    ]
    fig.legend(handles=legend_handles, loc="lower center", ncol=6, frameon=False)
    fig.suptitle("Formal benchmark overlay gallery (non-1008 runs)", fontsize=18)
    fig.subplots_adjust(top=0.92, bottom=0.08, wspace=0.18, hspace=0.22)
    path = OUT / "overlay_gallery_2x4_non_1008.png"
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return path


def main() -> None:
    OUT.mkdir(exist_ok=True)
    cell_data = {}
    rows = []
    for folder, label in CONDITIONS:
        for latency in LATENCIES:
            attempts = collect_cell(ORG / folder / f"{latency}ms")
            x_max_ms = max_time_for_attempts(attempts, minimum_ms=latency + 9000)
            path = save_cell_plot(label, latency, attempts, x_max_ms)
            cell_data[(label, latency)] = (attempts, x_max_ms)
            rows.append((label, latency, len(attempts), sum(1 for a in attempts if a["valid"]), path.name))
    gallery = save_gallery(cell_data)

    lines = [
        "# Overlay Gallery Non-1008",
        "",
        "These charts mirror the older external-wait overlay gallery style. Close-1008 attempts are excluded.",
        "",
        "| condition | latency_ms | non_1008_attempts | valid_attempts | image |",
        "|---|---:|---:|---:|---|",
    ]
    for label, latency, total, valid, name in rows:
        lines.append(f"| {label} | {latency} | {total} | {valid} | [{name}]({name}) |")
    lines.extend(["", f"Combined gallery: [{gallery.name}]({gallery.name})", ""])
    (OUT / "README.md").write_text("\n".join(lines))
    print(OUT)


if __name__ == "__main__":
    main()
