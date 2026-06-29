#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
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
DEFAULT_ASR_DIR = (
    ROOT
    / "result"
    / "2026-06-29_10-15-29-961_calendar_route_two_step_2step_tick_vs_no_tick_latency_sweep(2tool大跑)"
    / "organized"
    / "asr"
)
COLORS = {"no_tick": "#2563eb", "periodic_tick_4s": "#f97316"}

FINAL_CUE_RE = re.compile(
    r"\b("
    r"(?<!when )you should (?:plan to )?leave"
    r"|should leave (?:around|by|at)"
    r"|leave (?:around|by|at) (?:2|two)"
    r"|plan to leave (?:around|by|at)"
    r"|recommend leaving"
    r")\b",
    flags=re.IGNORECASE,
)
FINAL_LEAD_IN_RE = re.compile(
    r"\b(to arrive|considering (?:the|a)|with (?:a )?(?:35|thirty five)[ -]?minute|with a 10 minute buffer)\b",
    flags=re.IGNORECASE,
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def write_csv(path: Path, rows: list[dict[str, object]], headers: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+(?:'[a-z0-9]+)?", text.lower())


def split_sentences(text: str) -> list[str]:
    normalized = compact(text)
    if not normalized:
        return []
    chunks = re.split(
        r"(?<=[.!?])\s+|(?=\b(?:ok|okay|got it|i'm now|now i'm|i'll|i will|your next|getting|checking)\b)",
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
    return (clipped + 1.0) / (total + 1.0)


def brevity_penalty(candidate: list[str], references: list[list[str]]) -> float:
    if not candidate:
        return 0.0
    ref_lens = [len(ref) for ref in references if ref]
    if not ref_lens:
        return 0.0
    closest = min(ref_lens, key=lambda length: (abs(length - len(candidate)), length))
    if len(candidate) > closest:
        return 1.0
    return math.exp(1.0 - closest / len(candidate))


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


def cut_final_answer(row: dict) -> tuple[str, str, str]:
    segments = row.get("segments") or []
    texts = [compact(str(segment.get("text", ""))) for segment in segments]
    texts = [text for text in texts if text]
    if texts:
        cut_index: int | None = None
        cut_reason = "no_final_cue_found"
        for index, text in enumerate(texts):
            lookahead = compact(" ".join(texts[index : index + 3]))
            if FINAL_CUE_RE.search(text):
                cut_index = index
                cut_reason = "segment_final_cue"
                break
            if FINAL_LEAD_IN_RE.search(text) and FINAL_CUE_RE.search(lookahead):
                cut_index = index
                cut_reason = "segment_final_lead_in_plus_cue"
                break
        if cut_index is None:
            return compact(" ".join(texts)), "", cut_reason
        return compact(" ".join(texts[:cut_index])), compact(" ".join(texts[cut_index:])), cut_reason

    transcript = compact(str(row.get("transcript", "")))
    match = FINAL_CUE_RE.search(transcript)
    if not match:
        return transcript, "", "no_final_cue_found"
    return compact(transcript[: match.start()]), compact(transcript[match.start() :]), "transcript_final_cue"


def load_attempts(asr_dir: Path) -> list[dict[str, object]]:
    summary = json.loads((asr_dir / "asr_summary.json").read_text(encoding="utf-8"))
    attempts: list[dict[str, object]] = []
    for row in summary.get("rows") or []:
        if not (row.get("asr_available") and not row.get("asr_error")):
            continue
        waiting, final_answer, cut_reason = cut_final_answer(row)
        attempts.append(
            {
                "condition": row["condition"],
                "latency_ms": int(row["latency_ms"]),
                "latency_s": int(row.get("latency_s") or int(row["latency_ms"]) // 1000),
                "attempt_id": row["attempt_id"],
                "audio_path": row["audio_path"],
                "full_transcript": compact(row.get("transcript", "")),
                "waiting_transcript": waiting,
                "removed_final_transcript": final_answer,
                "final_cut_reason": cut_reason,
                "tokens": tokenize(waiting),
            }
        )
    return attempts


def group_bleu(attempts: list[dict[str, object]]) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    by_group: dict[tuple[str, int], list[dict[str, object]]] = defaultdict(list)
    for attempt in attempts:
        by_group[(str(attempt["condition"]), int(attempt["latency_ms"]))].append(attempt)

    attempt_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for key in sorted(by_group, key=lambda item: (item[1], item[0])):
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
                    "latency_s": item["latency_s"],
                    "attempt_id": item["attempt_id"],
                    "token_count": len(candidate),
                    "group_bleu_2": fmt(group_bleu_2),
                    "group_lexical_diversity_2": fmt(None if group_bleu_2 is None else 1.0 - group_bleu_2),
                    "group_bleu_4": fmt(group_bleu_4),
                    "group_lexical_diversity_4": fmt(None if group_bleu_4 is None else 1.0 - group_bleu_4),
                    "final_cut_reason": item["final_cut_reason"],
                    "waiting_transcript": item["waiting_transcript"],
                    "removed_final_transcript": item["removed_final_transcript"],
                }
            )
        mean_bleu2 = mean(bleu2_values) if bleu2_values else None
        mean_bleu4 = mean(bleu4_values) if bleu4_values else None
        summary_rows.append(
            {
                "condition": key[0],
                "latency_ms": key[1],
                "latency_s": key[1] // 1000,
                "attempts": len(items),
                "nonempty_waiting_transcripts": len(nonempty),
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
        sentences = split_sentences(str(attempt["waiting_transcript"]))
        sentence_tokens = [tokenize(sentence) for sentence in sentences]
        bleu2_values: list[float] = []
        bleu4_values: list[float] = []
        for index, candidate in enumerate(sentence_tokens):
            refs = [tokens for ref_index, tokens in enumerate(sentence_tokens) if ref_index != index]
            score2 = bleu(candidate, refs, 2)
            score4 = bleu(candidate, refs, 4)
            if score2 is not None:
                bleu2_values.append(score2)
            if score4 is not None:
                bleu4_values.append(score4)
        mean_bleu2 = mean(bleu2_values) if bleu2_values else None
        mean_bleu4 = mean(bleu4_values) if bleu4_values else None
        row = {
            "condition": attempt["condition"],
            "latency_ms": attempt["latency_ms"],
            "latency_s": attempt["latency_s"],
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
            "waiting_transcript": attempt["waiting_transcript"],
        }
        attempt_rows.append(row)
        by_group[(str(row["condition"]), int(row["latency_ms"]))].append(row)

    summary_rows: list[dict[str, object]] = []
    for key in sorted(by_group, key=lambda item: (item[1], item[0])):
        items = by_group[key]
        evaluable = [item for item in items if item["evaluable"]]
        bleu2_values = [float(item["self_bleu_2"]) for item in evaluable if item["self_bleu_2"]]
        bleu4_values = [float(item["self_bleu_4"]) for item in evaluable if item["self_bleu_4"]]
        mean_bleu2 = mean(bleu2_values) if bleu2_values else None
        mean_bleu4 = mean(bleu4_values) if bleu4_values else None
        summary_rows.append(
            {
                "condition": key[0],
                "latency_ms": key[1],
                "latency_s": key[1] // 1000,
                "attempts": len(items),
                "attempts_with_2plus_waiting_sentences": len(evaluable),
                "mean_self_bleu_2": fmt(mean_bleu2),
                "std_self_bleu_2": fmt(sample_std(bleu2_values)),
                "median_self_bleu_2": fmt(median(bleu2_values) if bleu2_values else None),
                "mean_intra_attempt_lexical_diversity_2": fmt(None if mean_bleu2 is None else 1.0 - mean_bleu2),
                "mean_self_bleu_4": fmt(mean_bleu4),
                "std_self_bleu_4": fmt(sample_std(bleu4_values)),
                "median_self_bleu_4": fmt(median(bleu4_values) if bleu4_values else None),
                "mean_intra_attempt_lexical_diversity_4": fmt(None if mean_bleu4 is None else 1.0 - mean_bleu4),
                "evaluable": len(evaluable) >= 1,
            }
        )
    return attempt_rows, summary_rows


def plot_grouped_bars(summary_rows: list[dict[str, object]], key: str, title: str, ylabel: str, out_path: Path) -> None:
    latencies = sorted({int(row["latency_ms"]) for row in summary_rows})
    conditions = [condition for condition in ["no_tick", "periodic_tick_4s"] if any(row["condition"] == condition for row in summary_rows)]
    x = np.arange(len(latencies))
    width = 0.34
    fig, ax = plt.subplots(figsize=(10, 5))
    for index, condition in enumerate(conditions):
        offset = (index - (len(conditions) - 1) / 2) * width
        values = []
        labels = []
        for latency in latencies:
            row = next(
                (item for item in summary_rows if item["condition"] == condition and int(item["latency_ms"]) == latency),
                None,
            )
            values.append(float(row[key]) if row and row.get(key) not in {"", None} else np.nan)
            labels.append(f"n={row.get('nonempty_waiting_transcripts', row.get('attempts_with_2plus_waiting_sentences', 0))}" if row else "n=0")
        positions = x + offset
        ax.bar(positions, values, width, label=condition, color=COLORS.get(condition))
        for xpos, value, label in zip(positions, values, labels):
            if not np.isnan(value):
                ax.text(xpos, min(value + 0.025, 1.0), label, ha="center", va="bottom", fontsize=8)
    ax.set_title(title)
    ax.set_xlabel("latency")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{latency // 1000}s" for latency in latencies])
    ax.grid(axis="y", alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def write_notes(out_dir: Path, asr_dir: Path, group_summary: list[dict[str, object]], self_summary: list[dict[str, object]], plots: list[Path]) -> None:
    lines = [
        "# Calendar Route Waiting BLEU",
        "",
        f"ASR source: `{rel(asr_dir)}`",
        "",
        "This pass uses `.assistant.wav` ASR transcripts. It removes the final answer with a simple regex cue such as `you should leave` / `leave around 2:15`, then scores the remaining waiting and bridge speech.",
        "",
        "Why not split by tool1 result report yet: that phrase is part of the behavior we want to measure, so using it as a hard boundary would bake the label into the metric.",
        "",
        "Metrics:",
        "",
        "- `group_bleu_2`: BLEU-2 of one attempt's waiting transcript against the other attempts in the same condition x latency group. Higher means more similar across runs.",
        "- `group_lexical_diversity_2`: `1 - group_bleu_2`. Higher means more varied wording across runs.",
        "- `self_bleu_2`: sentence-level BLEU-2 inside the same attempt's waiting transcript. Higher means more repetition within that attempt.",
        "- `intra_attempt_lexical_diversity_2`: `1 - self_bleu_2`. Higher means less repetition inside an attempt.",
        "",
        "Generated plots:",
    ]
    lines.extend(f"- `{rel(path)}`" for path in plots)
    lines.extend(["", "Group summary:", ""])
    for row in group_summary:
        lines.append(
            f"- {row['condition']} {row['latency_s']}s: group-BLEU-2={row['mean_group_bleu_2']}, "
            f"group diversity={row['mean_group_lexical_diversity_2']}"
        )
    lines.extend(["", "Intra-attempt summary:", ""])
    for row in self_summary:
        lines.append(
            f"- {row['condition']} {row['latency_s']}s: self-BLEU-2={row['mean_self_bleu_2']}, "
            f"intra diversity={row['mean_intra_attempt_lexical_diversity_2']}"
        )
    (out_dir / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--asr-dir", type=Path, default=DEFAULT_ASR_DIR)
    parser.add_argument("--output-dir", type=Path, default=None)
    args = parser.parse_args()

    asr_dir = args.asr_dir.resolve()
    out_dir = (args.output_dir or (asr_dir.parent / "lexical_diversity_waiting_bleu")).resolve()
    attempts = load_attempts(asr_dir)
    group_attempts, group_summary = group_bleu(attempts)
    self_attempts, self_summary = self_bleu(attempts)

    write_csv(
        out_dir / "waiting_group_bleu_attempts.csv",
        group_attempts,
        [
            "condition",
            "latency_ms",
            "latency_s",
            "attempt_id",
            "token_count",
            "group_bleu_2",
            "group_lexical_diversity_2",
            "group_bleu_4",
            "group_lexical_diversity_4",
            "final_cut_reason",
            "waiting_transcript",
            "removed_final_transcript",
        ],
    )
    write_csv(
        out_dir / "waiting_group_bleu_summary.csv",
        group_summary,
        [
            "condition",
            "latency_ms",
            "latency_s",
            "attempts",
            "nonempty_waiting_transcripts",
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
        self_attempts,
        [
            "condition",
            "latency_ms",
            "latency_s",
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
            "waiting_transcript",
        ],
    )
    write_csv(
        out_dir / "waiting_self_bleu_summary.csv",
        self_summary,
        [
            "condition",
            "latency_ms",
            "latency_s",
            "attempts",
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

    viz_dir = out_dir / "visualizations"
    plots = [
        viz_dir / "waiting_group_bleu_2.png",
        viz_dir / "waiting_group_lexical_diversity_2.png",
        viz_dir / "waiting_self_bleu_2.png",
        viz_dir / "waiting_intra_attempt_lexical_diversity_2.png",
    ]
    plot_grouped_bars(group_summary, "mean_group_bleu_2", "Waiting/Bridge Group-BLEU-2", "mean group-BLEU-2", plots[0])
    plot_grouped_bars(
        group_summary,
        "mean_group_lexical_diversity_2",
        "Waiting/Bridge Group Lexical Diversity",
        "mean 1 - group-BLEU-2",
        plots[1],
    )
    plot_grouped_bars(self_summary, "mean_self_bleu_2", "Waiting/Bridge Intra-attempt Self-BLEU-2", "mean self-BLEU-2", plots[2])
    plot_grouped_bars(
        self_summary,
        "mean_intra_attempt_lexical_diversity_2",
        "Waiting/Bridge Intra-attempt Lexical Diversity",
        "mean 1 - self-BLEU-2",
        plots[3],
    )
    write_notes(out_dir, asr_dir, group_summary, self_summary, plots)

    print(f"Attempts loaded: {len(attempts)}")
    print(f"Output dir: {rel(out_dir)}")
    print(f"Group summary: {rel(out_dir / 'waiting_group_bleu_summary.csv')}")
    print(f"Self summary: {rel(out_dir / 'waiting_self_bleu_summary.csv')}")
    print(f"README: {rel(out_dir / 'README.md')}")


if __name__ == "__main__":
    main()
