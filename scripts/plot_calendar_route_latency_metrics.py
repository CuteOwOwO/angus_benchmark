#!/usr/bin/env python3
from __future__ import annotations

import csv
import math
import re
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as file:
        return list(csv.DictReader(file))


def as_float(value: str) -> float | None:
    if value is None or value == "":
        return None
    try:
        parsed = float(value)
    except ValueError:
        return None
    if not math.isfinite(parsed):
        return None
    return parsed


def slug(name: str) -> str:
    return re.sub(r"[^a-zA-Z0-9]+", "_", name).strip("_").lower()


def draw_bar(rows: list[dict[str, str]], column: str, output_path: Path) -> None:
    latencies = sorted({row["latency_ms"] for row in rows}, key=lambda item: float(item))
    conditions = sorted({row.get("condition", "all") or "all" for row in rows})
    values_by_condition = {
        condition: [
            as_float(next((row[column] for row in rows if row["latency_ms"] == latency and (row.get("condition", "all") or "all") == condition), ""))
            for latency in latencies
        ]
        for condition in conditions
    }
    numeric = [value for values in values_by_condition.values() for value in values if value is not None]
    if not numeric:
        return

    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(latencies))
    width = min(0.36, 0.8 / max(1, len(conditions)))
    palette = {
        "no_tick": "#4C78A8",
        "periodic_tick_4s": "#F58518",
        "all": "#4C78A8",
    }
    for condition_index, condition in enumerate(conditions):
        offset = (condition_index - (len(conditions) - 1) / 2) * width
        values = values_by_condition[condition]
        bars = ax.bar(
            x + offset,
            [value if value is not None else 0 for value in values],
            width=width,
            color=palette.get(condition, "#54A24B"),
            edgecolor="#1F2937",
            linewidth=0.6,
            label=condition,
        )
        for bar, value in zip(bars, values):
            if value is None:
                continue
            label = f"{value:.3g}"
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height(), label, ha="center", va="bottom", fontsize=7)
    ax.set_title(column.replace("_", " "), fontsize=13)
    ax.set_xlabel("tool latency (ms)")
    ax.set_ylabel(column)
    ax.set_xticks(x)
    ax.set_xticklabels(latencies)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ymax = max(numeric)
    if 0 <= min(numeric) and ymax <= 1:
        ax.set_ylim(0, 1.05)
    elif min(numeric) >= 0:
        ax.set_ylim(0, ymax * 1.15 if ymax else 1)
    if len(conditions) > 1:
        ax.legend(frameon=False, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=170)
    plt.close(fig)


def write_diff(rows: list[dict[str, str]], output_csv: Path, output_png: Path) -> None:
    diff_rows: list[dict[str, str]] = []
    for row in rows:
        stage1 = as_float(row.get("stage1_max_silence_gap_ms_mean", ""))
        stage2 = as_float(row.get("stage2_max_silence_gap_ms_mean", ""))
        diff = None if stage1 is None or stage2 is None else stage2 - stage1
        diff_rows.append(
            {
                "condition": row.get("condition", "all"),
                "latency_ms": row["latency_ms"],
                "stage1_max_silence_gap_ms_mean": "" if stage1 is None else f"{stage1:.3f}",
                "stage2_max_silence_gap_ms_mean": "" if stage2 is None else f"{stage2:.3f}",
                "stage2_minus_stage1_max_silence_gap_ms": "" if diff is None else f"{diff:.3f}",
            }
        )
    with output_csv.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(diff_rows[0]))
        writer.writeheader()
        writer.writerows(diff_rows)

    latencies = sorted({row["latency_ms"] for row in diff_rows}, key=lambda item: float(item))
    conditions = sorted({row.get("condition", "all") or "all" for row in diff_rows})
    fig, ax = plt.subplots(figsize=(9, 4.8))
    x = np.arange(len(latencies))
    width = min(0.36, 0.8 / max(1, len(conditions)))
    palette = {"no_tick": "#4C78A8", "periodic_tick_4s": "#F58518", "all": "#F58518"}
    for condition_index, condition in enumerate(conditions):
        values = [
            as_float(next((row["stage2_minus_stage1_max_silence_gap_ms"] for row in diff_rows if row["latency_ms"] == latency and (row.get("condition", "all") or "all") == condition), ""))
            for latency in latencies
        ]
        offset = (condition_index - (len(conditions) - 1) / 2) * width
        bars = ax.bar(x + offset, [value if value is not None else 0 for value in values], width=width, color=palette.get(condition, "#54A24B"), edgecolor="#7C2D12", linewidth=0.6, label=condition)
        for bar, value in zip(bars, values):
            if value is None:
                continue
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                value,
                f"{value:.0f}",
                ha="center",
                va="bottom" if value >= 0 else "top",
                fontsize=7,
            )
    ax.axhline(0, color="#111827", linewidth=0.8)
    ax.set_title("stage2 - stage1 max silence gap", fontsize=13)
    ax.set_xlabel("tool latency (ms)")
    ax.set_ylabel("difference (ms)")
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.set_xticks(x)
    ax.set_xticklabels(latencies)
    if len(conditions) > 1:
        ax.legend(frameon=False, fontsize=8)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    output_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_png, dpi=170)
    plt.close(fig)


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: plot_calendar_route_latency_metrics.py <organized_dir>")
    organized_dir = Path(sys.argv[1]).resolve()
    metrics_path = organized_dir / "metrics_by_latency.csv"
    rows = read_rows(metrics_path)
    if not rows:
        raise SystemExit(f"No rows in {metrics_path}")

    out_dir = organized_dir / "visualizations" / "metrics_by_latency_bars"
    columns = [column for column in rows[0].keys() if column not in {"condition", "latency_ms"}]
    for column in columns:
        draw_bar(rows, column, out_dir / f"{slug(column)}.png")

    write_diff(
        rows,
        organized_dir / "stage2_minus_stage1_max_silence_gap_by_latency.csv",
        out_dir / "stage2_minus_stage1_max_silence_gap_ms.png",
    )
    print(f"Wrote {len(columns)} metric bar charts to {out_dir}")
    print(organized_dir / "stage2_minus_stage1_max_silence_gap_by_latency.csv")


if __name__ == "__main__":
    main()
