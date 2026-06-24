#!/usr/bin/env python3
import argparse
import csv
import json
import math
import re
from collections import Counter
from itertools import combinations
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
LATENCIES = [3000, 5000, 8000, 12000]
CONDITIONS = [
    ("condition_no_tick", "no_tick"),
    ("condition_tick_every_3000ms", "tick_every_3000ms"),
]
PUNCT_RE = re.compile(r"[^a-z0-9<>\s_]+")
SPACE_RE = re.compile(r"\s+")
SENTENCE_SPLIT_RE = re.compile(r"(?<=[.!?])\s+")
SINGLE_FINAL_VARIANTS = {
    "angus",
    "angas",
    "anges",
    "angis",
    "aengus",
    "ingus",
    "ingas",
    "inges",
    "ingis",
    "inghis",
    "ingeus",
    "engus",
    "engas",
    "enges",
    "dingus",
    "thinkus",
}
PHRASE_FINAL_VARIANTS = [
    "and guess",
    "an guess",
    "ang guess",
    "thank us",
    "end us",
    "in us",
]
FINAL_TOKEN = "final_answer"
DPI = 200


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def basic_tokens(text: str) -> list[str]:
    cleaned = PUNCT_RE.sub(" ", text.lower())
    return [token for token in SPACE_RE.sub(" ", cleaned).strip().split(" ") if token]


def normalize_final_answer_tokens(text: str) -> list[str]:
    tokens = basic_tokens(text)
    if not tokens:
        return []
    normalized = [FINAL_TOKEN if token in SINGLE_FINAL_VARIANTS else token for token in tokens]
    split_at = max(0, len(normalized) - 12)
    prefix = normalized[:split_at]
    tail_text = " ".join(normalized[split_at:])
    for phrase in PHRASE_FINAL_VARIANTS:
        tail_text = re.sub(rf"\b{re.escape(phrase)}\b", f" {FINAL_TOKEN} ", tail_text)
    tail_tokens = [token for token in SPACE_RE.sub(" ", tail_text).strip().split(" ") if token]
    return prefix + tail_tokens


def multiset_jaccard(tokens_a: list[str], tokens_b: list[str]) -> float | None:
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    if not counts_a and not counts_b:
        return None
    all_tokens = set(counts_a) | set(counts_b)
    intersection = sum(min(counts_a[token], counts_b[token]) for token in all_tokens)
    union = sum(max(counts_a[token], counts_b[token]) for token in all_tokens)
    return intersection / union if union else None


def word_count_cosine(tokens_a: list[str], tokens_b: list[str]) -> float | None:
    counts_a = Counter(tokens_a)
    counts_b = Counter(tokens_b)
    if not counts_a or not counts_b:
        return None
    all_tokens = set(counts_a) | set(counts_b)
    dot = sum(counts_a[token] * counts_b[token] for token in all_tokens)
    norm_a = math.sqrt(sum(value * value for value in counts_a.values()))
    norm_b = math.sqrt(sum(value * value for value in counts_b.values()))
    return dot / (norm_a * norm_b) if norm_a and norm_b else None


def avg(values: list[float | None]) -> float | None:
    real = [value for value in values if isinstance(value, (int, float)) and math.isfinite(value)]
    return sum(real) / len(real) if real else None


def sentence_split(text: str) -> list[str]:
    return [sentence.strip() for sentence in SENTENCE_SPLIT_RE.split(text.strip()) if sentence.strip()]


def transcript_text(tokens: list[str]) -> str:
    return " ".join(tokens)


def load_asr_rows(asr_summary_dir: Path) -> list[dict]:
    corrected = asr_summary_dir / "asr_summary_corrected.csv"
    fallback = asr_summary_dir / "asr_summary.csv"
    source = corrected if corrected.exists() else fallback
    rows: list[dict] = []
    with source.open("r", encoding="utf-8") as file:
        for row in csv.DictReader(file):
            if row.get("asr_available") not in {"True", "true", True}:
                continue
            if row.get("asr_error"):
                continue
            row["latency_ms"] = int(row["latency_ms"])
            row["raw_tokens"] = basic_tokens(row.get("transcript", ""))
            row["normalized_tokens"] = normalize_final_answer_tokens(row.get("transcript", ""))
            row["normalized_transcript"] = transcript_text(row["normalized_tokens"])
            rows.append(row)
    rows.sort(key=lambda item: (item["latency_ms"], item["condition"], item["run_id"]))
    return rows


def latest_visuals_dir() -> Path:
    candidates = [
        path
        for path in (ROOT / "result").iterdir()
        if path.is_dir() and ("external_wait_report_visuals" in path.name)
    ]
    if not candidates:
        target = ROOT / "result" / "external_wait_report_visuals_manual"
        target.mkdir(parents=True, exist_ok=True)
        return target
    return sorted(candidates, key=lambda path: path.name)[-1]


def sklearn_available() -> bool:
    try:
        import sklearn  # noqa: F401

        return True
    except Exception:
        return False


def tfidf_pair_values(docs: list[str]) -> dict[tuple[int, int], float] | None:
    try:
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
    except Exception:
        return None
    if len(docs) < 2:
        return {}
    matrix = TfidfVectorizer().fit_transform(docs)
    sims = cosine_similarity(matrix)
    values: dict[tuple[int, int], float] = {}
    for i, j in combinations(range(len(docs)), 2):
        values[(i, j)] = float(sims[i, j])
    return values


def group_rows(rows: list[dict]) -> dict[tuple[str, int], list[dict]]:
    grouped: dict[tuple[str, int], list[dict]] = {}
    for row in rows:
        grouped.setdefault((row["condition"], row["latency_ms"]), []).append(row)
    return grouped


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def analyze_intra_run(rows: list[dict]) -> tuple[list[dict], dict[tuple[str, int], dict]]:
    details: list[dict] = []
    for row in rows:
        sentences = sentence_split(row["transcript"])
        raw_jaccards: list[float | None] = []
        raw_cosines: list[float | None] = []
        normalized_jaccards: list[float | None] = []
        normalized_cosines: list[float | None] = []
        for sentence_a, sentence_b in combinations(sentences, 2):
            raw_a = basic_tokens(sentence_a)
            raw_b = basic_tokens(sentence_b)
            norm_a = normalize_final_answer_tokens(sentence_a)
            norm_b = normalize_final_answer_tokens(sentence_b)
            raw_jaccards.append(multiset_jaccard(raw_a, raw_b))
            raw_cosines.append(word_count_cosine(raw_a, raw_b))
            normalized_jaccards.append(multiset_jaccard(norm_a, norm_b))
            normalized_cosines.append(word_count_cosine(norm_a, norm_b))
        num_pairs = len(sentences) * (len(sentences) - 1) // 2
        details.append(
            {
                "condition": row["condition"],
                "latency_ms": row["latency_ms"],
                "run_id": row["run_id"],
                "num_sentences": len(sentences),
                "num_sentence_pairs": num_pairs,
                "raw_intra_multiset_jaccard": avg(raw_jaccards),
                "raw_intra_word_count_cosine": avg(raw_cosines),
                "normalized_intra_multiset_jaccard": avg(normalized_jaccards),
                "normalized_intra_word_count_cosine": avg(normalized_cosines),
                "insufficient_sentences": len(sentences) < 2,
                "transcript": row["transcript"],
                "audio_path": row["audio_path"],
            }
        )

    grouped_summary: dict[tuple[str, int], dict] = {}
    for key, items in group_rows(details).items():
        grouped_summary[key] = {
            "condition": key[0],
            "latency_ms": key[1],
            "avg_num_sentences": avg([item["num_sentences"] for item in items]),
            "avg_raw_intra_multiset_jaccard": avg([item["raw_intra_multiset_jaccard"] for item in items]),
            "avg_raw_intra_word_count_cosine": avg([item["raw_intra_word_count_cosine"] for item in items]),
            "avg_normalized_intra_multiset_jaccard": avg([item["normalized_intra_multiset_jaccard"] for item in items]),
            "avg_normalized_intra_word_count_cosine": avg([item["normalized_intra_word_count_cosine"] for item in items]),
        }
    return details, grouped_summary


def analyze_group_pairwise(rows: list[dict], tfidf_available: bool) -> tuple[list[dict], list[dict]]:
    pair_rows: list[dict] = []
    summary_rows: list[dict] = []
    for (condition, latency_ms), items in sorted(group_rows(rows).items(), key=lambda item: (item[0][1], item[0][0])):
        raw_docs = [transcript_text(item["raw_tokens"]) for item in items]
        norm_docs = [transcript_text(item["normalized_tokens"]) for item in items]
        raw_tfidf = tfidf_pair_values(raw_docs) if tfidf_available else None
        norm_tfidf = tfidf_pair_values(norm_docs) if tfidf_available else None
        raw_jaccards: list[float | None] = []
        raw_cosines: list[float | None] = []
        raw_tfidfs: list[float | None] = []
        norm_jaccards: list[float | None] = []
        norm_cosines: list[float | None] = []
        norm_tfidfs: list[float | None] = []
        for i, j in combinations(range(len(items)), 2):
            item_a = items[i]
            item_b = items[j]
            raw_j = multiset_jaccard(item_a["raw_tokens"], item_b["raw_tokens"])
            raw_c = word_count_cosine(item_a["raw_tokens"], item_b["raw_tokens"])
            norm_j = multiset_jaccard(item_a["normalized_tokens"], item_b["normalized_tokens"])
            norm_c = word_count_cosine(item_a["normalized_tokens"], item_b["normalized_tokens"])
            raw_t = raw_tfidf.get((i, j)) if raw_tfidf is not None else None
            norm_t = norm_tfidf.get((i, j)) if norm_tfidf is not None else None
            raw_jaccards.append(raw_j)
            raw_cosines.append(raw_c)
            raw_tfidfs.append(raw_t)
            norm_jaccards.append(norm_j)
            norm_cosines.append(norm_c)
            norm_tfidfs.append(norm_t)
            pair_rows.append(
                {
                    "condition": condition,
                    "latency_ms": latency_ms,
                    "run_id_a": item_a["run_id"],
                    "run_id_b": item_b["run_id"],
                    "raw_multiset_jaccard": raw_j,
                    "raw_word_count_cosine": raw_c,
                    "raw_tfidf_cosine": raw_t,
                    "normalized_multiset_jaccard": norm_j,
                    "normalized_word_count_cosine": norm_c,
                    "normalized_tfidf_cosine": norm_t,
                    "transcript_a": item_a["transcript"],
                    "transcript_b": item_b["transcript"],
                }
            )
        raw_j_avg = avg(raw_jaccards)
        raw_c_avg = avg(raw_cosines)
        raw_t_avg = avg(raw_tfidfs)
        norm_j_avg = avg(norm_jaccards)
        norm_c_avg = avg(norm_cosines)
        norm_t_avg = avg(norm_tfidfs)
        summary_rows.append(
            {
                "condition": condition,
                "latency_ms": latency_ms,
                "num_runs": len(items),
                "avg_words_per_run": avg([len(item["raw_tokens"]) for item in items]),
                "avg_sentences_per_run": avg([len(sentence_split(item["transcript"])) for item in items]),
                "within_group_raw_multiset_jaccard": raw_j_avg,
                "within_group_raw_word_count_cosine": raw_c_avg,
                "within_group_raw_tfidf_cosine": raw_t_avg,
                "within_group_normalized_multiset_jaccard": norm_j_avg,
                "within_group_normalized_word_count_cosine": norm_c_avg,
                "within_group_normalized_tfidf_cosine": norm_t_avg,
                "within_group_raw_multiset_diversity": None if raw_j_avg is None else 1 - raw_j_avg,
                "within_group_raw_word_count_diversity": None if raw_c_avg is None else 1 - raw_c_avg,
                "within_group_raw_tfidf_diversity": None if raw_t_avg is None else 1 - raw_t_avg,
                "within_group_normalized_multiset_diversity": None if norm_j_avg is None else 1 - norm_j_avg,
                "within_group_normalized_word_count_diversity": None if norm_c_avg is None else 1 - norm_c_avg,
                "within_group_normalized_tfidf_diversity": None if norm_t_avg is None else 1 - norm_t_avg,
            }
        )
    return pair_rows, summary_rows


def global_summary(rows: list[dict], tfidf_available: bool) -> dict:
    raw_docs = [transcript_text(item["raw_tokens"]) for item in rows]
    norm_docs = [transcript_text(item["normalized_tokens"]) for item in rows]
    raw_tfidf = tfidf_pair_values(raw_docs) if tfidf_available else None
    norm_tfidf = tfidf_pair_values(norm_docs) if tfidf_available else None
    raw_jaccards: list[float | None] = []
    raw_cosines: list[float | None] = []
    norm_jaccards: list[float | None] = []
    norm_cosines: list[float | None] = []
    raw_tfidfs: list[float | None] = []
    norm_tfidfs: list[float | None] = []
    for i, j in combinations(range(len(rows)), 2):
        raw_jaccards.append(multiset_jaccard(rows[i]["raw_tokens"], rows[j]["raw_tokens"]))
        raw_cosines.append(word_count_cosine(rows[i]["raw_tokens"], rows[j]["raw_tokens"]))
        norm_jaccards.append(multiset_jaccard(rows[i]["normalized_tokens"], rows[j]["normalized_tokens"]))
        norm_cosines.append(word_count_cosine(rows[i]["normalized_tokens"], rows[j]["normalized_tokens"]))
        raw_tfidfs.append(raw_tfidf.get((i, j)) if raw_tfidf is not None else None)
        norm_tfidfs.append(norm_tfidf.get((i, j)) if norm_tfidf is not None else None)
    raw_j = avg(raw_jaccards)
    raw_c = avg(raw_cosines)
    raw_t = avg(raw_tfidfs)
    norm_j = avg(norm_jaccards)
    norm_c = avg(norm_cosines)
    norm_t = avg(norm_tfidfs)
    return {
        "num_runs": len(rows),
        "num_pairs": len(rows) * (len(rows) - 1) // 2,
        "tfidf_available": tfidf_available,
        "global_raw_multiset_jaccard": raw_j,
        "global_raw_word_count_cosine": raw_c,
        "global_raw_tfidf_cosine": raw_t,
        "global_normalized_multiset_jaccard": norm_j,
        "global_normalized_word_count_cosine": norm_c,
        "global_normalized_tfidf_cosine": norm_t,
        "global_raw_diversity": None if raw_c is None else 1 - raw_c,
        "global_normalized_diversity": None if norm_c is None else 1 - norm_c,
    }


def top_repeated(rows: list[dict]) -> list[dict]:
    buckets: dict[str, list[dict]] = {}
    for row in rows:
        buckets.setdefault(row["normalized_transcript"], []).append(row)
    out: list[dict] = []
    for rank, (transcript, items) in enumerate(
        sorted(buckets.items(), key=lambda pair: (-len(pair[1]), pair[0]))[:20],
        start=1,
    ):
        example = items[0]
        out.append(
            {
                "rank": rank,
                "count": len(items),
                "ratio": len(items) / len(rows) if rows else None,
                "condition": example["condition"],
                "latency_ms": example["latency_ms"],
                "normalized_transcript": transcript,
                "example_raw_transcript": example["transcript"],
                "example_audio_path": example["audio_path"],
            }
        )
    return out


def draw_group_bar(summary_rows: list[dict], metric: str, title: str, ylabel: str, footer: str, out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 5.6))
    width = 0.34
    x_positions = list(range(len(LATENCIES)))
    colors = ["#4C78A8", "#F58518"]
    for idx, (condition, label) in enumerate(CONDITIONS):
        values = []
        for latency in LATENCIES:
            row = next((item for item in summary_rows if item["condition"] == condition and item["latency_ms"] == latency), {})
            value = row.get(metric)
            values.append(value if isinstance(value, (int, float)) else 0)
        offsets = [x + (idx - 0.5) * width for x in x_positions]
        ax.bar(offsets, values, width=width, label=label, color=colors[idx])
    ax.set_title(title)
    ax.set_ylabel(ylabel)
    ax.set_xlabel("External wait time")
    ax.set_xticks(x_positions)
    ax.set_xticklabels([f"{latency // 1000}s" for latency in LATENCIES])
    ax.set_ylim(0, 1.05)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.text(0.5, 0.02, footer, ha="center", fontsize=9, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def cosine_matrix(items: list[dict], token_key: str) -> list[list[float]]:
    matrix: list[list[float]] = []
    for item_a in items:
        row = []
        for item_b in items:
            if item_a is item_b:
                row.append(1.0)
            else:
                row.append(word_count_cosine(item_a[token_key], item_b[token_key]) or 0)
        matrix.append(row)
    return matrix


def draw_heatmap(items: list[dict], token_key: str, title: str, out_path: Path) -> None:
    matrix = cosine_matrix(items, token_key)
    fig, ax = plt.subplots(figsize=(6.5, 5.8))
    image = ax.imshow(matrix, vmin=0, vmax=1, cmap="Blues")
    ax.set_title(title)
    ax.set_xticks(range(len(items)))
    ax.set_yticks(range(len(items)))
    ax.set_xticklabels([item["run_id"] for item in items], rotation=45, ha="right", fontsize=7)
    ax.set_yticklabels([item["run_id"] for item in items], fontsize=7)
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, label="word-count cosine similarity")
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def write_outputs(out_dir: Path, rows: list[dict], tfidf_available: bool) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    intra_details, intra_summary_by_group = analyze_intra_run(rows)
    pair_rows, summary_rows = analyze_group_pairwise(rows, tfidf_available)
    for summary in summary_rows:
        intra = intra_summary_by_group.get((summary["condition"], summary["latency_ms"]), {})
        summary.update(intra)
    global_data = global_summary(rows, tfidf_available)
    repeated = top_repeated(rows)

    write_csv(
        out_dir / "asr_intra_run_sentence_similarity.csv",
        intra_details,
        [
            "condition",
            "latency_ms",
            "run_id",
            "num_sentences",
            "num_sentence_pairs",
            "raw_intra_multiset_jaccard",
            "raw_intra_word_count_cosine",
            "normalized_intra_multiset_jaccard",
            "normalized_intra_word_count_cosine",
            "insufficient_sentences",
            "transcript",
            "audio_path",
        ],
    )
    write_csv(
        out_dir / "asr_pairwise_similarity_by_group.csv",
        pair_rows,
        [
            "condition",
            "latency_ms",
            "run_id_a",
            "run_id_b",
            "raw_multiset_jaccard",
            "raw_word_count_cosine",
            "raw_tfidf_cosine",
            "normalized_multiset_jaccard",
            "normalized_word_count_cosine",
            "normalized_tfidf_cosine",
            "transcript_a",
            "transcript_b",
        ],
    )
    summary_headers = [
        "condition",
        "latency_ms",
        "num_runs",
        "avg_words_per_run",
        "avg_sentences_per_run",
        "within_group_raw_multiset_jaccard",
        "within_group_raw_word_count_cosine",
        "within_group_raw_tfidf_cosine",
        "within_group_normalized_multiset_jaccard",
        "within_group_normalized_word_count_cosine",
        "within_group_normalized_tfidf_cosine",
        "within_group_raw_multiset_diversity",
        "within_group_raw_word_count_diversity",
        "within_group_raw_tfidf_diversity",
        "within_group_normalized_multiset_diversity",
        "within_group_normalized_word_count_diversity",
        "within_group_normalized_tfidf_diversity",
        "avg_num_sentences",
        "avg_raw_intra_multiset_jaccard",
        "avg_raw_intra_word_count_cosine",
        "avg_normalized_intra_multiset_jaccard",
        "avg_normalized_intra_word_count_cosine",
    ]
    write_csv(out_dir / "asr_transcript_diversity_summary.csv", summary_rows, summary_headers)
    (out_dir / "asr_transcript_diversity_summary.json").write_text(
        json.dumps(
            {
                "tfidf_available": tfidf_available,
                "normalization": {
                    "final_answer_token": FINAL_TOKEN,
                    "single_variants": sorted(SINGLE_FINAL_VARIANTS),
                    "phrase_variants_tail_only": PHRASE_FINAL_VARIANTS,
                    "tail_token_window": 12,
                },
                "rows": summary_rows,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    (out_dir / "asr_global_similarity_summary.json").write_text(
        json.dumps(global_data, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    write_csv(
        out_dir / "top_repeated_transcripts.csv",
        repeated,
        [
            "rank",
            "count",
            "ratio",
            "condition",
            "latency_ms",
            "normalized_transcript",
            "example_raw_transcript",
            "example_audio_path",
        ],
    )
    draw_group_bar(
        summary_rows,
        "within_group_normalized_word_count_cosine",
        "Within-group transcript similarity",
        "Normalized word-count cosine similarity",
        "Higher similarity means responses are more template-like.",
        out_dir / "bar_within_group_similarity.png",
    )
    draw_group_bar(
        summary_rows,
        "within_group_normalized_word_count_diversity",
        "Within-group transcript diversity",
        "Normalized word-count diversity",
        "Diversity = 1 - normalized word-count cosine similarity.",
        out_dir / "bar_within_group_diversity.png",
    )
    for (condition, latency), items in sorted(group_rows(rows).items(), key=lambda item: (item[0][1], item[0][0])):
        label = "no_tick" if condition == "condition_no_tick" else "tick_every_3000ms"
        draw_heatmap(
            items,
            "raw_tokens",
            f"Raw similarity: {label}, {latency}ms",
            out_dir / f"heatmap_raw_{label}_{latency}ms.png",
        )
        draw_heatmap(
            items,
            "normalized_tokens",
            f"Normalized similarity: {label}, {latency}ms",
            out_dir / f"heatmap_normalized_{label}_{latency}ms.png",
        )


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze ASR transcript lexical diversity for external-wait results.")
    parser.add_argument("--pilot3s", type=Path)
    parser.add_argument("--batch", type=Path)
    parser.add_argument("--asr-summary", required=True, type=Path)
    parser.add_argument("--visuals-dir", type=Path)
    args = parser.parse_args()
    visuals_dir = args.visuals_dir.resolve() if args.visuals_dir else latest_visuals_dir().resolve()
    out_dir = visuals_dir / "asr_diversity"
    asr_summary_dir = args.asr_summary.resolve()
    rows = load_asr_rows(asr_summary_dir)
    tfidf = sklearn_available()
    write_outputs(out_dir, rows, tfidf)
    print(f"ASR diversity directory: {rel(out_dir)}")
    print(f"rows={len(rows)}")
    print(f"tfidf_available={tfidf}")


if __name__ == "__main__":
    main()
