#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(row: dict, key: str) -> float:
    value = row.get(key, "")
    if value == "":
        return float("nan")
    return float(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def grouped_bar(rows: list[dict], key: str, title: str, ylabel: str, out_path: Path) -> None:
    latencies = sorted({int(row["latency_ms"]) for row in rows})
    conditions = ["native_no_tick", "external_single_tick"]
    x = np.arange(len(latencies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, condition in enumerate(conditions):
        values = []
        for latency in latencies:
            row = next((item for item in rows if item["condition"] == condition and int(item["latency_ms"]) == latency), None)
            values.append(number(row, key) if row else float("nan"))
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


def stability(rows: list[dict], out_path: Path) -> None:
    labels = [f"{row['condition']}\n{row['latency_ms']}ms" for row in rows]
    total = np.array([number(row, "total_attempts") for row in rows])
    valid = np.array([number(row, "valid_attempts") for row in rows])
    close1008 = np.array([number(row, "1008_count") for row in rows])
    retries = np.array([number(row, "retry_count") for row in rows])
    x = np.arange(len(rows))
    width = 0.25
    fig, ax = plt.subplots(figsize=(13, 5))
    ax.bar(x - width, total, width, label="total attempts", color="#9ca3af")
    ax.bar(x, valid, width, label="valid attempts", color="#2563eb")
    ax.bar(x + width, close1008, width, label="1008 count", color="#dc2626")
    for idx, retry in enumerate(retries):
        ax.text(x[idx] - width, total[idx] + 0.15, f"r={int(retry)}", ha="center", va="bottom", fontsize=8)
    ax.set_title("Stability / Retry Overview")
    ax.set_ylabel("count")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30, ha="right")
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: plot_tau_live_tool_formal_benchmark.py <result_dir>")
    result_dir = Path(sys.argv[1]).resolve()
    rows = read_rows(result_dir / "summary.csv")
    out_dir = result_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    grouped_bar(rows, "first_audio_time_ms_avg", "First Audio Time", "avg ms", out_dir / "first_audio_time.png")
    grouped_bar(
        rows,
        "pre_result_text_transcription_event_count_avg",
        "Pre-result Text/Transcription Event Count",
        "avg event count before final tool response",
        out_dir / "pre_result_text_transcription_event_count.png",
    )
    grouped_bar(
        rows,
        "pre_result_text_transcription_event_rate_per_sec_avg",
        "Pre-result Text/Transcription Event Rate",
        "avg text/transcription events per second",
        out_dir / "pre_result_text_transcription_event_rate.png",
    )
    grouped_bar(
        rows,
        "audio_occupancy_ratio_from_start_to_final_mean",
        "Audio Occupancy Ratio: User Prompt to Final Tool Response",
        "mean audio occupancy ratio",
        out_dir / "audio_occupancy_ratio_from_start_to_final.png",
    )
    grouped_bar(
        rows,
        "audio_occupancy_ratio_from_tool_call_to_final_mean",
        "Audio Occupancy Ratio: Tool Call to Final Tool Response",
        "mean audio occupancy ratio",
        out_dir / "audio_occupancy_ratio_from_tool_call_to_final.png",
    )
    grouped_bar(
        rows,
        "max_silent_gap_ms_from_start_to_final_mean",
        "Max Silent Gap: User Prompt to Final Tool Response",
        "mean max silent gap (ms)",
        out_dir / "max_silent_gap_ms_from_start_to_final.png",
    )
    grouped_bar(
        rows,
        "post_final_answer_latency_ms_avg",
        "Post-final Answer Latency",
        "avg ms to first final-answer keyword",
        out_dir / "post_final_answer_latency.png",
    )
    stability(rows, out_dir / "stability_retry_overview.png")

    print("Formal benchmark visualizations:")
    for path in [
        out_dir / "first_audio_time.png",
        out_dir / "pre_result_text_transcription_event_count.png",
        out_dir / "pre_result_text_transcription_event_rate.png",
        out_dir / "audio_occupancy_ratio_from_start_to_final.png",
        out_dir / "audio_occupancy_ratio_from_tool_call_to_final.png",
        out_dir / "max_silent_gap_ms_from_start_to_final.png",
        out_dir / "post_final_answer_latency.png",
        out_dir / "stability_retry_overview.png",
    ]:
        print(f"- {rel(path)}")


if __name__ == "__main__":
    main()
