#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import math
import re
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median, stdev

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "result" / "2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark"
CONDITIONS = ["native_no_tick", "external_single_tick"]
COLORS = {"native_no_tick": "#2563eb", "external_single_tick": "#f97316"}


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as file:
        return list(csv.DictReader(file))


def write_csv(path: Path, rows: list[dict[str, object]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def is_true(value: str) -> bool:
    return value.strip().lower() == "true"


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def split_sentences(text: str) -> list[str]:
    # ASR punctuation is imperfect, so also split on a few common waiting-speech phrase boundaries.
    normalized = re.sub(r"\s+", " ", text.strip())
    if not normalized:
        return []
    chunks = re.split(
        r"(?<=[.!?])\s+|(?=\b(?:i'm still|still checking|please wait|just a moment|it might take|i am still)\b)",
        normalized,
        flags=re.IGNORECASE,
    )
    return [chunk.strip(" ,;") for chunk in chunks if len(tokenize(chunk)) >= 3]


def ngrams(tokens: list[str], n: int) -> Counter[tuple[str, ...]]:
    return Counter(tuple(tokens[index : index + n]) for index in range(0, max(0, len(tokens) - n + 1)))


def clipped_precision(candidate: list[str], references: list[list[str]], n: int) -> float:
    candidate_ngrams = ngrams(candidate, n)
    if not candidate_ngrams:
        return 0.0
    max_ref_counts: Counter[tuple[str, ...]] = Counter()
    for ref in references:
        ref_ngrams = ngrams(ref, n)
        for gram, count in ref_ngrams.items():
            max_ref_counts[gram] = max(max_ref_counts[gram], count)
    clipped = sum(min(count, max_ref_counts[gram]) for gram, count in candidate_ngrams.items())
    total = sum(candidate_ngrams.values())
    # Add-one smoothing keeps short waiting utterances from becoming all-or-nothing.
    return (clipped + 1.0) / (total + 1.0)


def brevity_penalty(candidate: list[str], references: list[list[str]]) -> float:
    if not candidate:
        return 0.0
    candidate_len = len(candidate)
    ref_lens = [len(ref) for ref in references if ref]
    if not ref_lens:
        return 0.0
    closest_ref_len = min(ref_lens, key=lambda length: (abs(length - candidate_len), length))
    if candidate_len > closest_ref_len:
        return 1.0
    return math.exp(1.0 - closest_ref_len / candidate_len)


def bleu(candidate: list[str], references: list[list[str]], max_n: int) -> float | None:
    references = [ref for ref in references if ref]
    if not candidate or not references:
        return None
    usable_n = min(max_n, len(candidate), max(len(ref) for ref in references))
    if usable_n <= 0:
        return None
    precisions = [clipped_precision(candidate, references, n) for n in range(1, usable_n + 1)]
    log_precision = sum(math.log(max(value, 1e-12)) for value in precisions) / usable_n
    return brevity_penalty(candidate, references) * math.exp(log_precision)


def fmt(value: float | None) -> str:
    if value is None or math.isnan(value):
        return ""
    return f"{value:.4f}"


def sample_std(values: list[float]) -> float | None:
    return stdev(values) if len(values) >= 2 else None


def group_key(row: dict[str, str]) -> tuple[str, int]:
    return row["condition"], int(row["latency_ms"])


def group_label(key: tuple[str, int]) -> str:
    return f"{key[0]}_{key[1]}ms"


def load_attempts(result_dir: Path) -> list[dict[str, object]]:
    rows = read_rows(result_dir / "asr_attempts.csv")
    attempts: list[dict[str, object]] = []
    for row in rows:
        transcript = row.get("pre_result_transcript", "").strip()
        if not is_true(row.get("valid", "")):
            continue
        attempts.append(
            {
                "condition": row["condition"],
                "latency_ms": int(row["latency_ms"]),
                "attempt_id": row["attempt_id"],
                "pre_result_transcript": transcript,
                "tokens": tokenize(transcript),
            }
        )
    return attempts


def group_bleu(attempts: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_group: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for attempt in attempts:
        by_group[(str(attempt["condition"]), int(attempt["latency_ms"]))].append(attempt)

    attempt_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for key in sorted(by_group, key=lambda item: (item[0], item[1])):
        items = by_group[key]
        nonempty = [item for item in items if item["tokens"]]
        bleu2_values: list[float] = []
        bleu4_values: list[float] = []
        for item in items:
            refs = [other["tokens"] for other in nonempty if other is not item]
            candidate = item["tokens"]
            group_bleu_2 = bleu(candidate, refs, 2)
            group_bleu_4 = bleu(candidate, refs, 4)
            if group_bleu_2 is not None:
                bleu2_values.append(group_bleu_2)
            if group_bleu_4 is not None:
                bleu4_values.append(group_bleu_4)
            attempt_rows.append(
                {
                    "condition": item["condition"],
                    "latency_ms": item["latency_ms"],
                    "attempt_id": item["attempt_id"],
                    "pre_result_transcript": item["pre_result_transcript"],
                    "token_count": len(candidate),
                    "group_bleu_2": fmt(group_bleu_2),
                    "group_lexical_diversity_2": fmt(None if group_bleu_2 is None else 1.0 - group_bleu_2),
                    "group_bleu_4": fmt(group_bleu_4),
                    "group_lexical_diversity_4": fmt(None if group_bleu_4 is None else 1.0 - group_bleu_4),
                }
            )

        mean_bleu2 = mean(bleu2_values) if bleu2_values else None
        mean_bleu4 = mean(bleu4_values) if bleu4_values else None
        summary_rows.append(
            {
                "condition": key[0],
                "latency_ms": key[1],
                "valid_attempts": len(items),
                "nonempty_pre_result_transcripts": len(nonempty),
                "mean_group_bleu_2": fmt(mean_bleu2),
                "std_group_bleu_2": fmt(sample_std(bleu2_values)),
                "median_group_bleu_2": fmt(median(bleu2_values) if bleu2_values else None),
                "mean_group_lexical_diversity_2": fmt(None if mean_bleu2 is None else 1.0 - mean_bleu2),
                "mean_group_bleu_4": fmt(mean_bleu4),
                "std_group_bleu_4": fmt(sample_std(bleu4_values)),
                "median_group_bleu_4": fmt(median(bleu4_values) if bleu4_values else None),
                "mean_group_lexical_diversity_4": fmt(None if mean_bleu4 is None else 1.0 - mean_bleu4),
                "evaluable": len(nonempty) >= 2,
            }
        )
    return attempt_rows, summary_rows


def self_bleu(attempts: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    attempt_rows: list[dict[str, object]] = []
    by_group: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)

    for attempt in attempts:
        sentences = split_sentences(str(attempt["pre_result_transcript"]))
        sentence_tokens = [tokenize(sentence) for sentence in sentences]
        sentence_bleu2_values: list[float] = []
        sentence_bleu4_values: list[float] = []
        for index, candidate in enumerate(sentence_tokens):
            refs = [tokens for ref_index, tokens in enumerate(sentence_tokens) if ref_index != index]
            sentence_bleu_2 = bleu(candidate, refs, 2)
            sentence_bleu_4 = bleu(candidate, refs, 4)
            if sentence_bleu_2 is not None:
                sentence_bleu2_values.append(sentence_bleu_2)
            if sentence_bleu_4 is not None:
                sentence_bleu4_values.append(sentence_bleu_4)

        mean_bleu2 = mean(sentence_bleu2_values) if sentence_bleu2_values else None
        mean_bleu4 = mean(sentence_bleu4_values) if sentence_bleu4_values else None
        row = {
            "condition": attempt["condition"],
            "latency_ms": attempt["latency_ms"],
            "attempt_id": attempt["attempt_id"],
            "sentence_count": len(sentence_tokens),
            "token_count": len(attempt["tokens"]),
            "self_bleu_2": fmt(mean_bleu2),
            "self_repetition_score_2": fmt(mean_bleu2),
            "intra_attempt_lexical_diversity_2": fmt(None if mean_bleu2 is None else 1.0 - mean_bleu2),
            "self_bleu_4": fmt(mean_bleu4),
            "intra_attempt_lexical_diversity_4": fmt(None if mean_bleu4 is None else 1.0 - mean_bleu4),
            "evaluable": len(sentence_tokens) >= 2,
            "sentences": " | ".join(sentences),
            "pre_result_transcript": attempt["pre_result_transcript"],
        }
        attempt_rows.append(row)
        by_group[(str(attempt["condition"]), int(attempt["latency_ms"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(by_group, key=lambda item: (item[0], item[1])):
        items = by_group[key]
        evaluable_items = [item for item in items if item["evaluable"]]
        bleu2_values = [float(item["self_bleu_2"]) for item in evaluable_items if item["self_bleu_2"]]
        bleu4_values = [float(item["self_bleu_4"]) for item in evaluable_items if item["self_bleu_4"]]
        mean_bleu2 = mean(bleu2_values) if bleu2_values else None
        mean_bleu4 = mean(bleu4_values) if bleu4_values else None
        summary_rows.append(
            {
                "condition": key[0],
                "latency_ms": key[1],
                "valid_attempts": len(items),
                "attempts_with_2plus_waiting_sentences": len(evaluable_items),
                "mean_self_bleu_2": fmt(mean_bleu2),
                "std_self_bleu_2": fmt(sample_std(bleu2_values)),
                "median_self_bleu_2": fmt(median(bleu2_values) if bleu2_values else None),
                "mean_intra_attempt_lexical_diversity_2": fmt(None if mean_bleu2 is None else 1.0 - mean_bleu2),
                "mean_self_bleu_4": fmt(mean_bleu4),
                "std_self_bleu_4": fmt(sample_std(bleu4_values)),
                "median_self_bleu_4": fmt(median(bleu4_values) if bleu4_values else None),
                "mean_intra_attempt_lexical_diversity_4": fmt(None if mean_bleu4 is None else 1.0 - mean_bleu4),
                "evaluable": len(evaluable_items) >= 1,
            }
        )
    return attempt_rows, summary_rows


def cross_group(attempts: list[dict[str, object]]) -> list[dict[str, object]]:
    by_group: dict[tuple[str, int], list[list[str]]] = defaultdict(list)
    for attempt in attempts:
        tokens = attempt["tokens"]
        if tokens:
            by_group[(str(attempt["condition"]), int(attempt["latency_ms"]))].append(tokens)

    rows: list[dict[str, object]] = []
    for source_key in sorted(by_group, key=lambda item: (item[0], item[1])):
        source_items = by_group[source_key]
        for reference_key in sorted(by_group, key=lambda item: (item[0], item[1])):
            reference_items = by_group[reference_key]
            scores: list[float] = []
            for candidate in source_items:
                refs = reference_items
                if source_key == reference_key:
                    refs = [ref for ref in reference_items if ref is not candidate]
                score = bleu(candidate, refs, 2)
                if score is not None:
                    scores.append(score)
            rows.append(
                {
                    "source_group": group_label(source_key),
                    "source_condition": source_key[0],
                    "source_latency_ms": source_key[1],
                    "reference_group": group_label(reference_key),
                    "reference_condition": reference_key[0],
                    "reference_latency_ms": reference_key[1],
                    "source_nonempty_count": len(source_items),
                    "reference_nonempty_count": len(reference_items),
                    "mean_bleu_2_to_reference": fmt(mean(scores) if scores else None),
                    "n_scored": len(scores),
                }
            )
    return rows


def plot_group_bleu(summary_rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    latencies = sorted({int(row["latency_ms"]) for row in summary_rows})
    x = np.arange(len(latencies))
    width = 0.36
    paths: list[Path] = []
    for key, title, ylabel, out_name in [
        ("mean_group_bleu_2", "Waiting Speech Group-BLEU-2", "mean group-BLEU-2", "waiting_group_bleu_2.png"),
        (
            "mean_group_lexical_diversity_2",
            "Waiting Speech Group Lexical Diversity",
            "mean 1 - group-BLEU-2",
            "waiting_group_lexical_diversity_2.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for index, condition in enumerate(CONDITIONS):
            values = []
            labels = []
            for latency in latencies:
                row = next(
                    (
                        item
                        for item in summary_rows
                        if item["condition"] == condition and int(item["latency_ms"]) == latency
                    ),
                    None,
                )
                values.append(float(row[key]) if row and row.get(key) not in {"", None} else np.nan)
                labels.append(f"n={row['nonempty_pre_result_transcripts']}" if row else "n=0")
            positions = x + (index - 0.5) * width
            ax.bar(positions, values, width, label=condition, color=COLORS[condition])
            for xpos, value, label in zip(positions, values, labels):
                if not np.isnan(value):
                    ax.text(xpos, min(value + 0.025, 1.0), label, ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("latency_ms")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(latency) for latency in latencies])
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_self_bleu(summary_rows: list[dict[str, object]], out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    latencies = sorted({int(row["latency_ms"]) for row in summary_rows})
    x = np.arange(len(latencies))
    width = 0.36
    paths: list[Path] = []
    for key, title, ylabel, out_name in [
        ("mean_self_bleu_2", "Waiting Speech Intra-attempt Self-BLEU-2", "mean self-BLEU-2", "waiting_self_bleu_2.png"),
        (
            "mean_intra_attempt_lexical_diversity_2",
            "Waiting Speech Intra-attempt Lexical Diversity",
            "mean 1 - self-BLEU-2",
            "waiting_self_lexical_diversity_2.png",
        ),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        for index, condition in enumerate(CONDITIONS):
            values = []
            labels = []
            for latency in latencies:
                row = next(
                    (
                        item
                        for item in summary_rows
                        if item["condition"] == condition and int(item["latency_ms"]) == latency
                    ),
                    None,
                )
                values.append(float(row[key]) if row and row.get(key) not in {"", None} else np.nan)
                labels.append(f"n={row['attempts_with_2plus_waiting_sentences']}" if row else "n=0")
            positions = x + (index - 0.5) * width
            ax.bar(positions, values, width, label=condition, color=COLORS[condition])
            for xpos, value, label in zip(positions, values, labels):
                if not np.isnan(value):
                    ax.text(xpos, min(value + 0.025, 1.0), label, ha="center", va="bottom", fontsize=8)
        ax.set_title(title)
        ax.set_xlabel("latency_ms")
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 1)
        ax.set_xticks(x)
        ax.set_xticklabels([str(latency) for latency in latencies])
        ax.grid(axis="y", alpha=0.25)
        ax.legend()
        fig.tight_layout()
        out_path = out_dir / out_name
        fig.savefig(out_path, dpi=160)
        plt.close(fig)
        paths.append(out_path)
    return paths


def plot_cross_matrix(cross_rows: list[dict[str, object]], out_dir: Path) -> Path | None:
    groups = sorted({str(row["source_group"]) for row in cross_rows})
    if not groups:
        return None
    matrix = np.full((len(groups), len(groups)), np.nan)
    index = {group: pos for pos, group in enumerate(groups)}
    for row in cross_rows:
        value = row.get("mean_bleu_2_to_reference")
        if value == "":
            continue
        matrix[index[str(row["source_group"])]][index[str(row["reference_group"])]] = float(value)

    fig, ax = plt.subplots(figsize=(10, 8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_xticks(np.arange(len(groups)))
    ax.set_yticks(np.arange(len(groups)))
    ax.set_xticklabels(groups, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(groups, fontsize=8)
    ax.set_title("Cross-group Waiting Speech BLEU-2")
    for row in range(len(groups)):
        for col in range(len(groups)):
            value = matrix[row, col]
            if not np.isnan(value):
                ax.text(col, row, f"{value:.2f}", ha="center", va="center", fontsize=7)
    fig.colorbar(image, ax=ax, label="mean BLEU-2 to reference group")
    fig.tight_layout()
    out_path = out_dir / "waiting_cross_group_bleu_2_matrix.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def write_notes(
    result_dir: Path,
    out_dir: Path,
    group_summary_rows: list[dict[str, object]],
    self_summary_rows: list[dict[str, object]],
    plot_paths: list[Path],
) -> Path:
    notes_path = out_dir / "README.md"
    lines = [
        "# Waiting Speech Lexical Diversity",
        "",
        f"Source folder: `{rel(result_dir)}`",
        "",
        "This postprocess uses valid attempts from `asr_attempts.csv` and only the `pre_result_transcript` field.",
        "Runtime tick messages, external status payloads, tool payloads, and final answers are not included in the text being scored.",
        "",
        "Metrics:",
        "",
        "- `group_bleu_2`: for each pre-result transcript, BLEU-2 against the other non-empty transcripts in the same condition x latency group, then averaged. This is the old cross-attempt metric, renamed from self-BLEU to group-BLEU.",
        "- `group_lexical_diversity_2`: `1 - group_bleu_2`; higher means less cross-attempt lexical overlap / more varied wording across runs.",
        "- `self_bleu_2`: within a single attempt, split the pre-result transcript into waiting sentences, score each sentence against the other waiting sentences in the same attempt, then average. Higher means more repetition inside that attempt.",
        "- `intra_attempt_lexical_diversity_2`: `1 - self_bleu_2`; higher means less repetition within the attempt.",
        "- Empty/single-transcript groups are marked non-evaluable for group-BLEU. Attempts with fewer than 2 waiting sentences are non-evaluable for self-BLEU.",
        "- BLEU uses lowercase word tokens and add-one smoothed n-gram precision so short waiting utterances are not all-or-nothing.",
        "",
        "Generated files:",
        "",
        "- `waiting_group_bleu_attempts.csv`",
        "- `waiting_group_bleu_summary.csv`",
        "- `waiting_self_bleu_attempts.csv`",
        "- `waiting_self_bleu_summary.csv`",
        "- `waiting_cross_group_bleu_matrix.csv`",
    ]
    lines.extend(f"- `{rel(path)}`" for path in plot_paths)
    lines.extend(["", "Quick read: group-BLEU across attempts", ""])
    for row in group_summary_rows:
        lines.append(
            "- "
            f"{row['condition']} {row['latency_ms']}ms: "
            f"n={row['nonempty_pre_result_transcripts']}, "
            f"group-BLEU-2={row['mean_group_bleu_2'] or 'NA'}, "
            f"group lexical diversity={row['mean_group_lexical_diversity_2'] or 'NA'}"
        )
    lines.extend(["", "Quick read: self-BLEU within attempts", ""])
    for row in self_summary_rows:
        lines.append(
            "- "
            f"{row['condition']} {row['latency_ms']}ms: "
            f"n={row['attempts_with_2plus_waiting_sentences']}, "
            f"self-BLEU-2={row['mean_self_bleu_2'] or 'NA'}, "
            f"intra-attempt lexical diversity={row['mean_intra_attempt_lexical_diversity_2'] or 'NA'}"
        )
    notes_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return notes_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--result-dir", default=str(DEFAULT_RESULT_DIR))
    args = parser.parse_args()

    result_dir = Path(args.result_dir).resolve()
    out_dir = result_dir / "lexical_diversity"
    viz_dir = result_dir / "visualizations"
    attempts = load_attempts(result_dir)
    group_attempt_rows, group_summary_rows = group_bleu(attempts)
    self_attempt_rows, self_summary_rows = self_bleu(attempts)
    cross_rows = cross_group(attempts)

    write_csv(
        out_dir / "waiting_group_bleu_attempts.csv",
        group_attempt_rows,
        [
            "condition",
            "latency_ms",
            "attempt_id",
            "token_count",
            "group_bleu_2",
            "group_lexical_diversity_2",
            "group_bleu_4",
            "group_lexical_diversity_4",
            "pre_result_transcript",
        ],
    )
    write_csv(
        out_dir / "waiting_group_bleu_summary.csv",
        group_summary_rows,
        [
            "condition",
            "latency_ms",
            "valid_attempts",
            "nonempty_pre_result_transcripts",
            "mean_group_bleu_2",
            "std_group_bleu_2",
            "median_group_bleu_2",
            "mean_group_lexical_diversity_2",
            "mean_group_bleu_4",
            "std_group_bleu_4",
            "median_group_bleu_4",
            "mean_group_lexical_diversity_4",
            "evaluable",
        ],
    )
    write_csv(
        out_dir / "waiting_self_bleu_attempts.csv",
        self_attempt_rows,
        [
            "condition",
            "latency_ms",
            "attempt_id",
            "sentence_count",
            "token_count",
            "self_bleu_2",
            "self_repetition_score_2",
            "intra_attempt_lexical_diversity_2",
            "self_bleu_4",
            "intra_attempt_lexical_diversity_4",
            "evaluable",
            "sentences",
            "pre_result_transcript",
        ],
    )
    write_csv(
        out_dir / "waiting_self_bleu_summary.csv",
        self_summary_rows,
        [
            "condition",
            "latency_ms",
            "valid_attempts",
            "attempts_with_2plus_waiting_sentences",
            "mean_self_bleu_2",
            "std_self_bleu_2",
            "median_self_bleu_2",
            "mean_intra_attempt_lexical_diversity_2",
            "mean_self_bleu_4",
            "std_self_bleu_4",
            "median_self_bleu_4",
            "mean_intra_attempt_lexical_diversity_4",
            "evaluable",
        ],
    )
    write_csv(
        out_dir / "waiting_cross_group_bleu_matrix.csv",
        cross_rows,
        [
            "source_group",
            "source_condition",
            "source_latency_ms",
            "reference_group",
            "reference_condition",
            "reference_latency_ms",
            "source_nonempty_count",
            "reference_nonempty_count",
            "mean_bleu_2_to_reference",
            "n_scored",
        ],
    )
    plot_paths = plot_group_bleu(group_summary_rows, viz_dir)
    plot_paths.extend(plot_self_bleu(self_summary_rows, viz_dir))
    cross_plot = plot_cross_matrix(cross_rows, viz_dir)
    if cross_plot:
        plot_paths.append(cross_plot)
    notes_path = write_notes(result_dir, out_dir, group_summary_rows, self_summary_rows, plot_paths)

    print("Waiting speech lexical diversity outputs:")
    print(f"- {rel(out_dir / 'waiting_group_bleu_attempts.csv')}")
    print(f"- {rel(out_dir / 'waiting_group_bleu_summary.csv')}")
    print(f"- {rel(out_dir / 'waiting_self_bleu_attempts.csv')}")
    print(f"- {rel(out_dir / 'waiting_self_bleu_summary.csv')}")
    print(f"- {rel(out_dir / 'waiting_cross_group_bleu_matrix.csv')}")
    print(f"- {rel(notes_path)}")
    for path in plot_paths:
        print(f"- {rel(path)}")


if __name__ == "__main__":
    main()
