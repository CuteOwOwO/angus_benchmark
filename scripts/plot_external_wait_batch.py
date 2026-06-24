#!/usr/bin/env python3
import argparse
import json
import math
from pathlib import Path
from statistics import median

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch


ROOT = Path(__file__).resolve().parents[1]
DPI = 200
PCM_BYTES_PER_SECOND = 48_000
OVERLAY_AUDIO_BAR_HEIGHT = 0.08
CONDITION_LABELS = {
    "condition_no_tick": "no_tick",
    "condition_tick_every_3000ms": "tick_every_3000ms",
}
OUTPUT_NOTE = "Output count/rate is based on audio/text event chunks, not utterance count."


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    if not path.exists():
        return events
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def value(row: dict, key: str):
    item = row.get(key)
    return item if isinstance(item, (int, float)) and math.isfinite(item) else None


def condition_label(condition: str) -> str:
    return CONDITION_LABELS.get(condition, condition.removeprefix("condition_"))


def grouped_bar(ax, rows: list[dict], metric: str, title: str, ylabel: str) -> None:
    latencies = sorted({row.get("latency_ms") for row in rows if row.get("latency_ms") is not None})
    conditions = list(dict.fromkeys(row.get("condition") for row in rows if row.get("condition")))
    if not latencies or not conditions:
        ax.text(0.5, 0.5, "no data", ha="center", va="center")
        ax.set_title(title)
        return

    width = min(0.8 / len(conditions), 0.34)
    base = list(range(len(latencies)))
    for index, condition in enumerate(conditions):
        offsets = [x + (index - (len(conditions) - 1) / 2) * width for x in base]
        values = []
        for latency in latencies:
            row = next((item for item in rows if item.get("latency_ms") == latency and item.get("condition") == condition), {})
            values.append(value(row, metric) or 0)
        ax.bar(offsets, values, width=width, label=condition_label(condition))

    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xticks(base)
    ax.set_xticklabels([f"{latency} ms" for latency in latencies])
    ax.grid(axis="y", linestyle=":", alpha=0.35)


def draw_summary_timing(rows: list[dict], out_path: Path) -> None:
    metrics = [
        ("avg_first_response_latency_ms", "Avg first response", "ms"),
        ("median_first_response_latency_ms", "Median first response", "ms"),
        ("avg_post_external_first_response_latency_ms", "Avg post-result response", "ms"),
        ("median_post_external_first_response_latency_ms", "Median post-result response", "ms"),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7))
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        grouped_bar(ax, rows, metric, title, ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("External-wait batch timing metrics", y=0.98)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def draw_summary_waiting(rows: list[dict], out_path: Path) -> None:
    metrics = [
        ("avg_assistant_output_count_before_external_result", "Avg output events before result", "event/chunk count"),
        ("avg_assistant_output_rate_before_external_result_per_sec", "Avg output event rate before result", "events/chunks per sec"),
    ]
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    for ax, (metric, title, ylabel) in zip(axes.flat, metrics):
        grouped_bar(ax, rows, metric, title, ylabel)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("External-wait output while waiting", y=0.98)
    fig.text(0.5, 0.01, OUTPUT_NOTE, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def draw_summary_stability(rows: list[dict], out_path: Path) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.6))
    grouped_bar(axes[0], rows, "valid_run_rate", "Valid run rate", "rate")
    axes[0].set_ylim(0, 1.05)

    latencies = sorted({row.get("latency_ms") for row in rows if row.get("latency_ms") is not None})
    conditions = list(dict.fromkeys(row.get("condition") for row in rows if row.get("condition")))
    error_metrics = [
        ("server_1008_errors", "1008", "#B00020"),
        ("server_1011_errors", "1011", "#D95F02"),
        ("other_errors", "other", "#7570B3"),
    ]
    width = min(0.8 / max(1, len(conditions)), 0.34)
    base = list(range(len(latencies)))
    for condition_index, condition in enumerate(conditions):
        bottoms = [0.0 for _ in latencies]
        offsets = [x + (condition_index - (len(conditions) - 1) / 2) * width for x in base]
        for metric, label, color in error_metrics:
            values = []
            for latency in latencies:
                row = next((item for item in rows if item.get("latency_ms") == latency and item.get("condition") == condition), {})
                values.append(value(row, metric) or 0)
            axes[1].bar(offsets, values, width=width, bottom=bottoms, color=color, label=label if condition_index == 0 else None)
            bottoms = [bottom + item for bottom, item in zip(bottoms, values)]
        for x, condition_name in zip(offsets, [condition] * len(offsets)):
            axes[1].text(x, -0.05, condition_label(condition_name), rotation=20, ha="right", va="top", fontsize=8)

    axes[1].set_title("Error counts")
    axes[1].set_ylabel("count")
    axes[1].set_xticks(base)
    axes[1].set_xticklabels([f"{latency} ms" for latency in latencies])
    axes[1].grid(axis="y", linestyle=":", alpha=0.35)
    axes[1].legend(frameon=False)

    handles, labels = axes[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=max(1, len(labels)), frameon=False)
    fig.suptitle("External-wait session stability", y=0.98)
    fig.tight_layout(rect=(0, 0.05, 1, 0.9))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def server_event_parts(event: dict) -> list[dict]:
    return (
        event.get("data", {})
        .get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )


def is_assistant_output(event: dict) -> bool:
    if event.get("event_type") != "server_event":
        return False
    server_content = event.get("data", {}).get("message", {}).get("serverContent", {})
    if server_content.get("outputTranscription", {}).get("text"):
        return True
    for part in server_event_parts(event):
        if part.get("text"):
            return True
        inline = part.get("inlineData", {})
        if inline.get("bytes", 0) > 0:
            return True
    return False


def assistant_audio_segments(event: dict) -> list[tuple[int, float]]:
    ms = event_ms(event)
    if ms is None or event.get("event_type") != "server_event":
        return []
    segments: list[tuple[int, float]] = []
    for part in server_event_parts(event):
        byte_count = part.get("inlineData", {}).get("bytes", 0)
        if byte_count > 0:
            segments.append((ms, byte_count / PCM_BYTES_PER_SECOND * 1000))
    return segments


def assistant_text_times(event: dict) -> list[int]:
    ms = event_ms(event)
    if ms is None or event.get("event_type") != "server_event":
        return []
    server_content = event.get("data", {}).get("message", {}).get("serverContent", {})
    times: list[int] = []
    if server_content.get("outputTranscription", {}).get("text"):
        times.append(ms)
    for part in server_event_parts(event):
        if part.get("text"):
            times.append(ms)
    return times


def event_ms(event: dict) -> int | None:
    item = event.get("relative_time_ms")
    return int(item) if isinstance(item, (int, float)) else None


def run_sort_key(path: Path) -> int:
    name = path.name
    try:
        return int(name.split("_")[1])
    except Exception:
        return 999999


def read_run_summary(run_dir: Path) -> dict | None:
    path = run_dir / "summary.json"
    if not path.exists():
        return None
    data = read_json(path)
    scenarios = data.get("scenarios") or []
    if not scenarios:
        return None
    return scenarios[0]


def collect_overlay_runs(batch_dir: Path, condition: str, latency_ms: int) -> list[dict]:
    condition_dir = batch_dir / condition
    runs: list[dict] = []
    if not condition_dir.exists():
        return runs
    run_dirs = [
        path
        for path in condition_dir.rglob("run_*")
        if path.is_dir()
    ]
    for run_dir in sorted(run_dirs, key=lambda path: (str(path.parent), run_sort_key(path))):
        summary = read_run_summary(run_dir)
        if not summary or summary.get("latency_ms") != latency_ms:
            continue
        if not summary.get("session_valid"):
            continue
        events = read_jsonl(run_dir / "raw_log.jsonl")
        external_ms = summary.get("external_result_injected_time_ms")
        audio_segments: list[tuple[int, float]] = []
        text_times: list[int] = []
        pending_ticks: list[int] = []
        closes: list[dict] = []
        for event in events:
            ms = event_ms(event)
            if ms is None:
                continue
            event_type = event.get("event_type")
            if event_type == "external_status_sent":
                pending_ticks.append(ms)
            elif event_type in {"session_closed", "socket_error"}:
                closes.append({"timeMs": ms, "eventType": event_type, "data": event.get("data", {})})
            elif event_type == "server_event":
                audio_segments.extend(assistant_audio_segments(event))
                text_times.extend(assistant_text_times(event))
        runs.append(
            {
                "run": run_dir.name,
                "summary": summary,
                "audioSegments": audio_segments,
                "textTimes": text_times,
                "pendingTicks": pending_ticks,
                "externalMs": external_ms,
                "closes": closes,
            }
        )
    return runs


def draw_overlay(batch_dir: Path, condition: str, latency_ms: int, post_external_wait_ms: int | None, out_path: Path) -> bool:
    runs = collect_overlay_runs(batch_dir, condition, latency_ms)
    if not runs:
        return False
    x_max_ms = latency_ms + (post_external_wait_ms or max((run["summary"].get("post_external_wait_ms") or 6000 for run in runs), default=6000))
    fig_height = max(3.0, 0.28 * len(runs) + 1.45)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_title(f"Overlay timeline: {condition}, latency {latency_ms}ms (valid runs only)")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylabel("run attempt")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    y_positions = list(range(len(runs)))
    ax.set_yticks(y_positions)
    ax.set_yticklabels([run["run"] for run in runs])

    for y, run in zip(y_positions, runs):
        ax.hlines(y, 0, x_max_ms / 1000, color="#C9CDD1", linewidth=0.6, zorder=1)
        for start_ms, duration_ms in run["audioSegments"]:
            ax.broken_barh(
                [(start_ms / 1000, max(duration_ms / 1000, 0.01))],
                (y - OVERLAY_AUDIO_BAR_HEIGHT / 2, OVERLAY_AUDIO_BAR_HEIGHT),
                facecolors="#1E88E5",
                edgecolors="#0D47A1",
                linewidth=0.35,
                alpha=0.85,
                zorder=3,
            )
        if run["textTimes"]:
            ax.scatter([ms / 1000 for ms in run["textTimes"]], [y] * len(run["textTimes"]), s=18, color="#F57C00", edgecolor="white", linewidth=0.25, alpha=0.9, zorder=4)
        for tick_ms in run["pendingTicks"]:
            ax.vlines(tick_ms / 1000, y - 0.12, y + 0.12, color="#616161", linewidth=0.9, linestyle=":", zorder=4)
        external_ms = run["externalMs"]
        if isinstance(external_ms, (int, float)):
            ax.vlines(external_ms / 1000, y - 0.14, y + 0.14, color="#D32F2F", linewidth=1.05, zorder=4)
        for close in run["closes"]:
            data = close.get("data", {})
            color = "#B00020" if data.get("code") in {1008, 1011} else "#7B1FA2"
            ax.scatter([close["timeMs"] / 1000], [y], marker="x", s=45, color=color, linewidth=1.5, zorder=5)
            if data.get("code") in {1008, 1011}:
                ax.text(close["timeMs"] / 1000, y + 0.13, str(data.get("code")), color=color, fontsize=7, ha="center", va="bottom")

    ax.set_xlim(0, max(1, x_max_ms / 1000))
    ax.set_ylim(-0.55, len(runs) - 0.45)
    legend_handles = [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio output"),
        Line2D([], [], marker="o", linestyle="", color="#F57C00", label="text event", markersize=5),
        Line2D([], [], color="#616161", linestyle=":", label="pending tick"),
        Line2D([], [], color="#D32F2F", label="external result injected"),
        Line2D([], [], marker="x", linestyle="", color="#B00020", label="session error/close", markersize=6),
    ]
    ax.legend(handles=legend_handles, loc="upper center", bbox_to_anchor=(0.5, -0.13), ncol=3, fontsize=8, frameon=False)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)
    return True


def enrich_rows_with_medians(summary: dict) -> list[dict]:
    rows = [dict(row) for row in summary.get("rows", [])]
    runs = summary.get("runs", [])
    for row in rows:
        condition = row.get("condition")
        latency = row.get("latency_ms")
        valid_runs = [
            run
            for run in runs
            if run.get("session_valid")
            and run.get("condition") == condition
            and run.get("latency_ms") == latency
        ]
        values = [
            run.get("post_external_first_response_latency_ms")
            for run in valid_runs
            if isinstance(run.get("post_external_first_response_latency_ms"), (int, float))
        ]
        row["median_post_external_first_response_latency_ms"] = median(values) if values else None
    return rows


def generate_visualizations(batch_dir: Path) -> list[Path]:
    summary_path = batch_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")
    summary = read_json(summary_path)
    rows = enrich_rows_with_medians(summary)
    viz_dir = batch_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    generated: list[Path] = []

    charts = [
        (draw_summary_timing, viz_dir / "summary_timing_bars.png"),
        (draw_summary_waiting, viz_dir / "summary_waiting_bars.png"),
        (draw_summary_stability, viz_dir / "summary_stability_bars.png"),
    ]
    for draw, path in charts:
        draw(rows, path)
        generated.append(path)

    post_wait_by_row = {
        (run.get("condition"), run.get("latency_ms")): run.get("post_external_wait_ms")
        for run in summary.get("runs", [])
        if run.get("condition") and run.get("latency_ms")
    }
    for row in rows:
        condition = row.get("condition")
        latency_ms = row.get("latency_ms")
        if not condition or not isinstance(latency_ms, int):
            continue
        out_path = viz_dir / f"overlay_{condition}_latency_{latency_ms}ms.png"
        if draw_overlay(batch_dir, condition, latency_ms, post_wait_by_row.get((condition, latency_ms)), out_path):
            generated.append(out_path)

    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate external-wait batch summary and overlay visualizations.")
    parser.add_argument("--input", required=True, type=Path, help="Batch result folder, e.g. result/external_wait_pilot_3s_...")
    args = parser.parse_args()
    generated = generate_visualizations(args.input.resolve())
    print("Generated visualizations:")
    for path in generated:
        print(f"- {rel(path)}")


if __name__ == "__main__":
    main()
