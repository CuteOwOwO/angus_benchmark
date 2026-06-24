#!/usr/bin/env python3
from __future__ import annotations

import csv
import sys
from pathlib import Path
from statistics import stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]


def read_rows(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def number(row: dict | None, key: str) -> float:
    if not row:
        return float("nan")
    value = row.get(key, "")
    if value == "" or value.lower() in {"null", "na"}:
        return float("nan")
    return float(value)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def grouped_bar(
    rows: list[dict],
    key: str,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    out_path: Path,
    errors: dict[tuple[str, int], float] | None = None,
    labels: dict[tuple[str, int], str] | None = None,
) -> None:
    latencies = sorted({int(row["latency_ms"]) for row in rows})
    conditions = ["native_no_tick", "external_single_tick"]
    x = np.arange(len(latencies))
    width = 0.36
    fig, ax = plt.subplots(figsize=(10, 5))
    colors = {"native_no_tick": "#2563eb", "external_single_tick": "#f97316"}
    for index, condition in enumerate(conditions):
        values = []
        yerr = []
        for latency in latencies:
            row = next((item for item in rows if item["condition"] == condition and int(item["latency_ms"]) == latency), None)
            values.append(number(row, key))
            yerr.append((errors or {}).get((condition, latency), 0.0))
        bar_positions = x + (index - 0.5) * width
        ax.bar(
            bar_positions,
            values,
            width,
            yerr=yerr if errors else None,
            capsize=4 if errors else 0,
            label=condition,
            color=colors.get(condition),
        )
        if labels:
            for xpos, value, latency in zip(bar_positions, values, latencies):
                label = labels.get((condition, latency), "")
                if label and not np.isnan(value):
                    ax.text(xpos, min(value + (ylim[1] - ylim[0]) * 0.035, ylim[1]), label, ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("latency_ms")
    ax.set_ylabel(ylabel)
    ax.set_ylim(*ylim)
    ax.set_xticks(x)
    ax.set_xticklabels([str(latency) for latency in latencies])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def metric_std_by_group(attempt_rows: list[dict], key: str) -> dict[tuple[str, int], float]:
    out: dict[tuple[str, int], float] = {}
    for condition in ["native_no_tick", "external_single_tick"]:
        for latency in sorted({int(row["latency_ms"]) for row in attempt_rows}):
            values = [
                float(row[key])
                for row in attempt_rows
                if row["condition"] == condition
                and int(row["latency_ms"]) == latency
                and row.get("valid", "").lower() == "true"
                and row.get(key, "") != ""
            ]
            out[(condition, latency)] = stdev(values) if len(values) >= 2 else 0.0
    return out


def diversity_labels(rows: list[dict]) -> dict[tuple[str, int], str]:
    labels: dict[tuple[str, int], str] = {}
    for row in rows:
        condition = row["condition"]
        latency = int(row["latency_ms"])
        n = row.get("diversity_n_nonempty_transcripts", "")
        evaluable = row.get("diversity_evaluable", "")
        labels[(condition, latency)] = f"n={n}" if evaluable == "true" else "NA"
    return labels


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("Usage: plot_llm_judge_formal.py <formal_benchmark_result_dir>")
    result_dir = Path(sys.argv[1]).resolve()
    rows = read_rows(result_dir / "llm_judge_v3_summary.csv")
    attempt_rows = read_rows(result_dir / "llm_judge_v3_attempts.csv")
    out_dir = result_dir / "visualizations"
    out_dir.mkdir(parents=True, exist_ok=True)

    plots = [
        (
            "final_core_answer_correct_rate_valid",
            "LLM Judge V3: Final Core Answer Correct Rate",
            "rate",
            (0, 1),
            out_dir / "llm_v3_final_core_answer_rate.png",
            metric_std_by_group(attempt_rows, "final_core_answer_correct"),
            None,
        ),
        (
            "waiting_speech_coverage_rate_valid",
            "LLM Judge V3: Waiting Speech Coverage",
            "coverage rate",
            (0, 1),
            out_dir / "llm_v3_waiting_speech_coverage.png",
            metric_std_by_group(attempt_rows, "waiting_speech_present"),
            None,
        ),
        (
            "mean_waiting_task_relevance_score_when_spoke_valid",
            "LLM Judge V3: Waiting Task Relevance When Spoke",
            "mean score (0-3)",
            (0, 3),
            out_dir / "llm_v3_waiting_task_relevance_when_spoke.png",
            metric_std_by_group(attempt_rows, "waiting_task_relevance_score"),
            None,
        ),
        (
            "pre_result_hallucination_rate_valid",
            "LLM Judge V3: Pre-result Hallucination Rate",
            "rate",
            (0, 1),
            out_dir / "llm_v3_pre_result_hallucination_rate.png",
            metric_std_by_group(attempt_rows, "pre_result_hallucination"),
            None,
        ),
        (
            "waiting_diversity_score",
            "LLM Judge V3: Waiting Diversity",
            "score (0-3)",
            (0, 3),
            out_dir / "llm_v3_waiting_diversity_score.png",
            None,
            diversity_labels(rows),
        ),
    ]

    print("LLM judge visualizations:")
    for key, title, ylabel, ylim, out_path, errors, labels in plots:
        grouped_bar(rows, key, title, ylabel, ylim, out_path, errors=errors, labels=labels)
        print(f"- {rel(out_path)}")


if __name__ == "__main__":
    main()
