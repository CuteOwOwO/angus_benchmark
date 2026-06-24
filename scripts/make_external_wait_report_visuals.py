#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
DPI = 200
CONDITIONS = [
    ("condition_no_tick", "no_tick"),
    ("condition_tick_every_3000ms", "tick_every_3000ms"),
]
LATENCIES = [3000, 5000, 8000, 12000]
OUTPUT_RATE_NOTE = "Output rate is based on assistant audio/text event chunks, not utterance-level segmentation."


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def load_rows(folders: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for folder in folders:
        summary_path = folder / "summary.json"
        if not summary_path.exists():
            continue
        data = read_json(summary_path)
        for row in data.get("rows", []):
            rows.append(dict(row, source_folder=rel(folder)))
    rows.sort(key=lambda item: (item.get("latency_ms", 0), item.get("condition", "")))
    return rows


def row_for(rows: list[dict], condition: str, latency_ms: int) -> dict:
    return next((row for row in rows if row.get("condition") == condition and row.get("latency_ms") == latency_ms), {})


def write_csv(path: Path, rows: list[dict]) -> None:
    headers = [
        "condition",
        "latency_ms",
        "target_valid_runs",
        "attempted_runs",
        "valid_runs",
        "invalid_runs",
        "server_1008_errors",
        "server_1011_errors",
        "other_errors",
        "valid_run_rate",
        "avg_first_response_latency_ms",
        "median_first_response_latency_ms",
        "avg_assistant_output_count_before_external_result",
        "avg_assistant_output_rate_before_external_result_per_sec",
        "avg_post_external_first_response_latency_ms",
        "has_output_after_external_result_rate",
        "source_folder",
    ]
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def draw_grouped_metric(rows: list[dict], metric: str, title: str, ylabel: str, out_path: Path, footer: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.34
    x_positions = list(range(len(LATENCIES)))
    colors = ["#4C78A8", "#F58518"]
    for index, (condition, label) in enumerate(CONDITIONS):
        values = []
        for latency in LATENCIES:
            value = row_for(rows, condition, latency).get(metric)
            values.append(value if isinstance(value, (int, float)) else 0)
        offsets = [x + (index - 0.5) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=label, color=colors[index])

    ax.set_title(title, fontsize=14)
    ax.set_xlabel("External wait time")
    ax.set_ylabel(ylabel)
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{latency // 1000}s" for latency in LATENCIES])
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    if footer:
        fig.text(0.5, 0.02, footer, ha="center", fontsize=9, color="#555555")
        fig.tight_layout(rect=(0, 0.06, 1, 1))
    else:
        fig.tight_layout()
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def copy_overlays(out_dir: Path, pilot3s: Path, batch: Path) -> list[str]:
    overlay_dir = out_dir / "overlays"
    overlay_dir.mkdir(parents=True, exist_ok=True)
    copied: list[str] = []
    sources = {
        "overlay_condition_no_tick_latency_3000ms.png": pilot3s / "visualizations" / "overlay_condition_no_tick_latency_3000ms.png",
        "overlay_condition_tick_every_3000ms_latency_3000ms.png": pilot3s / "visualizations" / "overlay_condition_tick_every_3000ms_latency_3000ms.png",
        "overlay_condition_no_tick_latency_5000ms.png": batch / "visualizations" / "overlay_condition_no_tick_latency_5000ms.png",
        "overlay_condition_tick_every_3000ms_latency_5000ms.png": batch / "visualizations" / "overlay_condition_tick_every_3000ms_latency_5000ms.png",
        "overlay_condition_no_tick_latency_8000ms.png": batch / "visualizations" / "overlay_condition_no_tick_latency_8000ms.png",
        "overlay_condition_tick_every_3000ms_latency_8000ms.png": batch / "visualizations" / "overlay_condition_tick_every_3000ms_latency_8000ms.png",
        "overlay_condition_no_tick_latency_12000ms.png": batch / "visualizations" / "overlay_condition_no_tick_latency_12000ms.png",
        "overlay_condition_tick_every_3000ms_latency_12000ms.png": batch / "visualizations" / "overlay_condition_tick_every_3000ms_latency_12000ms.png",
    }
    for name, source in sources.items():
        if source.exists():
            target = overlay_dir / name
            shutil.copy2(source, target)
            copied.append(rel(target))
    return copied


def main() -> None:
    parser = argparse.ArgumentParser(description="Make report-ready external-wait summary visuals from existing result folders.")
    parser.add_argument("--pilot3s", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pilot3s = args.pilot3s.resolve()
    batch = args.batch.resolve()
    out_dir = (args.output or (ROOT / "result" / f"external_wait_report_visuals_{timestamp()}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_rows([pilot3s, batch])
    summary_csv = out_dir / "combined_metrics_summary.csv"
    summary_json = out_dir / "combined_metrics_summary.json"
    write_csv(summary_csv, rows)
    summary_json.write_text(json.dumps({"rows": rows, "sources": [rel(pilot3s), rel(batch)]}, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    first_latency_path = out_dir / "bar_first_response_latency_2x4.png"
    output_rate_path = out_dir / "bar_before_result_output_rate_2x4.png"
    draw_grouped_metric(
        rows,
        "avg_first_response_latency_ms",
        "First response latency by condition and external wait time",
        "First response latency (ms)",
        first_latency_path,
    )
    draw_grouped_metric(
        rows,
        "avg_assistant_output_rate_before_external_result_per_sec",
        "Pre-result assistant output event rate by condition and external wait time",
        "Output event rate before result (events/sec)",
        output_rate_path,
        footer=OUTPUT_RATE_NOTE,
    )
    copied = copy_overlays(out_dir, pilot3s, batch)

    print("Generated report visuals:")
    for path in [first_latency_path, output_rate_path, summary_csv, summary_json]:
        print(f"- {rel(path)}")
    if copied:
        print("Copied overlays:")
        for path in copied:
            print(f"- {path}")


if __name__ == "__main__":
    main()
