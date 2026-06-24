#!/usr/bin/env python3
import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PHRASE_PATTERNS = [
    ("the_final_answer_is", re.compile(r"\bthe\s+final\s+answer\s+is\b")),
    ("final_answer_is", re.compile(r"\bfinal\s+answer\s+is\b")),
    ("the_answer_is", re.compile(r"\bthe\s+answer\s+is\b")),
    ("answer_is", re.compile(r"\banswer\s+is\b")),
    ("is_angus", re.compile(r"\bis\s+angus\b")),
    ("plain_angus", re.compile(r"^\s*angus[.!?\s]*$")),
    ("ends_with_angus", re.compile(r"\bangus[.!?\s]*$")),
    ("contains_angus", re.compile(r"\bangus\b")),
    ("contains_inghis", re.compile(r"\binghis\b")),
    ("contains_and_guess", re.compile(r"\band\s+guess\b")),
    ("contains_dingus", re.compile(r"\bdingus\b")),
    ("waiting_final_not_available", re.compile(r"\bfinal\s+answer\s+is\s+not\s+available\s+yet\b")),
    ("waiting_im_waiting", re.compile(r"\bi'?m\s+waiting\s+for\s+the\s+final\s+answer\b")),
    ("waiting_notify_ready", re.compile(r"\bnotify\s+you\s+when\s+it\s+is\s+ready\b")),
]


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text.lower()).strip()


def classify(transcript: str) -> list[str]:
    normalized = normalize(transcript)
    labels = [name for name, pattern in PHRASE_PATTERNS if pattern.search(normalized)]
    return labels or ["other"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Bucket ASR transcripts by simple answer/waiting phrase string matches.")
    parser.add_argument("--asr-summary", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    asr_summary = args.asr_summary.resolve()
    data = json.loads(asr_summary.read_text(encoding="utf-8"))
    out_dir = (args.output or (asr_summary.parent / f"answer_phrase_analysis_{timestamp()}")).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    overall = Counter()
    by_group: dict[tuple[str, int], Counter] = defaultdict(Counter)
    examples: dict[str, list[dict]] = defaultdict(list)

    for item in data.get("rows", []):
        transcript = item.get("transcript") or ""
        labels = classify(transcript)
        row = {
            "condition": item.get("condition"),
            "latency_ms": item.get("latency_ms"),
            "run_id": item.get("run_id"),
            "audio_path": item.get("audio_path"),
            "transcript": transcript,
            "phrase_labels": ";".join(labels),
        }
        rows.append(row)
        key = (str(item.get("condition")), int(item.get("latency_ms") or 0))
        for label in labels:
            overall[label] += 1
            by_group[key][label] += 1
            if len(examples[label]) < 5:
                examples[label].append(row)

    detail_csv = out_dir / "answer_phrase_detail.csv"
    with detail_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "latency_ms", "run_id", "audio_path", "phrase_labels", "transcript"])
        writer.writeheader()
        writer.writerows(rows)

    summary_rows = []
    labels = sorted(overall)
    for (condition, latency_ms), counter in sorted(by_group.items(), key=lambda pair: (pair[0][1], pair[0][0])):
        total = sum(1 for row in rows if row["condition"] == condition and row["latency_ms"] == latency_ms)
        for label in labels:
            count = counter[label]
            summary_rows.append(
                {
                    "condition": condition,
                    "latency_ms": latency_ms,
                    "phrase_label": label,
                    "count": count,
                    "rate": count / total if total else None,
                    "group_total": total,
                }
            )

    summary_csv = out_dir / "answer_phrase_summary.csv"
    with summary_csv.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["condition", "latency_ms", "phrase_label", "count", "rate", "group_total"])
        writer.writeheader()
        writer.writerows(summary_rows)

    summary_json = out_dir / "answer_phrase_summary.json"
    summary_json.write_text(
        json.dumps(
            {
                "source": rel(asr_summary),
                "total_rows": len(rows),
                "overall_counts": dict(overall.most_common()),
                "examples": examples,
                "summary_csv": rel(summary_csv),
                "detail_csv": rel(detail_csv),
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )

    print(f"Answer phrase analysis directory: {rel(out_dir)}")
    print("Overall counts:")
    for label, count in overall.most_common():
        print(f"- {label}: {count}")


if __name__ == "__main__":
    main()
