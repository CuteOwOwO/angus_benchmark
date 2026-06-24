#!/usr/bin/env python3
import argparse
import csv
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
FINAL_KEYWORDS = [
    "shipped",
    "tomorrow",
    "ups",
    "1z999aa10123456784",
    "a123",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def clean_text(text: str) -> str:
    return " ".join(text.split())


def normalize_for_match(text: str) -> str:
    return re.sub(r"[^a-z0-9#]+", "", text.lower())


def has_final_content(text: str) -> bool:
    normalized = normalize_for_match(text)
    hits = sum(1 for keyword in FINAL_KEYWORDS if keyword in normalized)
    return hits >= 2 and ("shipped" in normalized or "tomorrow" in normalized)


def first_event_ms(events: list[dict], event_type: str) -> int | None:
    for event in events:
        if event.get("type") == event_type and event.get("event_ms") is not None:
            return int(event["event_ms"])
    return None


def close_code(events: list[dict]) -> int | None:
    for event in events:
        if event.get("type") == "session_closed":
            code = event.get("code")
            return int(code) if code is not None else None
    return None


def text_fragments_after(events: list[dict], final_ms: int, event_type: str) -> list[dict]:
    fragments: list[dict] = []
    for event in events:
        if event.get("type") != event_type:
            continue
        event_ms = event.get("event_ms")
        text = event.get("text")
        if event_ms is None or not text or int(event_ms) < final_ms:
            continue
        fragments.append({"event_ms": int(event_ms), "delta_ms": int(event_ms) - final_ms, "text": clean_text(text)})
    return fragments


def analyze_attempt(attempt_dir: Path, run_dir: Path) -> dict:
    summary = read_json(attempt_dir / "summary.json")
    events = read_jsonl(attempt_dir / "timeline" / "events.jsonl")
    final_ms = first_event_ms(events, "final_tool_response_sent")
    text_after = text_fragments_after(events, final_ms, "text_output") if final_ms is not None else []
    spoken_after = text_fragments_after(events, final_ms, "output_transcription") if final_ms is not None else []
    text_joined = clean_text(" ".join(item["text"] for item in text_after))
    spoken_joined = clean_text(" ".join(item["text"] for item in spoken_after))
    return {
        "condition": summary.get("condition"),
        "attempt_index": summary.get("attempt_index"),
        "relative_attempt_dir": rel(attempt_dir).removeprefix(rel(run_dir) + "/"),
        "close_code": summary.get("close_code") or close_code(events),
        "session_valid": summary.get("session_valid"),
        "final_tool_response_sent": final_ms is not None,
        "final_tool_response_sent_time_ms": final_ms,
        "text_output_after_final_count": len(text_after),
        "first_text_output_after_final_delta_ms": text_after[0]["delta_ms"] if text_after else None,
        "text_output_after_final_has_result": has_final_content(text_joined),
        "output_transcription_after_final_count": len(spoken_after),
        "first_output_transcription_after_final_delta_ms": spoken_after[0]["delta_ms"] if spoken_after else None,
        "output_transcription_after_final_has_result": has_final_content(spoken_joined),
        "text_output_after_final": text_joined,
        "output_transcription_after_final": spoken_joined,
    }


def summarize(rows: list[dict]) -> list[dict]:
    summaries: list[dict] = []
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        summaries.append(
            {
                "condition": condition,
                "attempts": len(condition_rows),
                "final_sent": sum(1 for row in condition_rows if row["final_tool_response_sent"]),
                "any_text_after_final": sum(1 for row in condition_rows if row["text_output_after_final_count"] > 0),
                "text_after_final_has_result": sum(1 for row in condition_rows if row["text_output_after_final_has_result"]),
                "spoken_after_final": sum(1 for row in condition_rows if row["output_transcription_after_final_count"] > 0),
                "spoken_after_final_has_result": sum(1 for row in condition_rows if row["output_transcription_after_final_has_result"]),
                "close_1008": sum(1 for row in condition_rows if row["close_code"] == 1008),
            }
        )
    return summaries


def write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)


def write_markdown(path: Path, run_dir: Path, rows: list[dict], summary_rows: list[dict]) -> None:
    lines = [
        "# Text After Final Tool Response Analysis",
        "",
        f"Run folder: `{rel(run_dir)}`",
        "",
        "This uses Gemini Live `text_output` and `output_transcription` events from `timeline/events.jsonl`; it is not external ASR.",
        "",
        "| condition | attempts | final_sent | any_text_after_final | text_result | spoken_after_final | spoken_result | 1008 |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary_rows:
        lines.append(
            f"| {row['condition']} | {row['attempts']} | {row['final_sent']} | {row['any_text_after_final']} | "
            f"{row['text_after_final_has_result']} | {row['spoken_after_final']} | "
            f"{row['spoken_after_final_has_result']} | {row['close_1008']} |"
        )
    lines.extend(["", "## Per Attempt", ""])
    for row in rows:
        spoken = row["output_transcription_after_final"] or "(none)"
        text = row["text_output_after_final"] or "(none)"
        lines.extend(
            [
                f"### {row['condition']} attempt {int(row['attempt_index']):04d}",
                "",
                f"- final_sent: {row['final_tool_response_sent']} at {row['final_tool_response_sent_time_ms']} ms",
                f"- close_code: {row['close_code']}",
                f"- spoken_after_final_has_result: {row['output_transcription_after_final_has_result']}",
                f"- text_after_final_has_result: {row['text_output_after_final_has_result']}",
                f"- spoken_after_final: {spoken}",
                f"- text_output_after_final: {text}",
                "",
            ]
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze Gemini Live text events after final tool responses.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    attempt_dirs = sorted(run_dir.glob("condition_*/attempt_*"))
    rows = [analyze_attempt(attempt_dir, run_dir) for attempt_dir in attempt_dirs]
    summary_rows = summarize(rows)
    write_csv(run_dir / "text_after_final_analysis.csv", rows)
    write_csv(run_dir / "text_after_final_summary.csv", summary_rows)
    write_markdown(run_dir / "text_after_final_analysis.md", run_dir, rows, summary_rows)
    print(f"Analyzed attempts: {len(rows)}")
    for row in summary_rows:
        print(
            f"{row['condition']}: final={row['final_sent']}/{row['attempts']}, "
            f"spoken_result={row['spoken_after_final_has_result']}/{row['attempts']}, "
            f"text_result={row['text_after_final_has_result']}/{row['attempts']}"
        )


if __name__ == "__main__":
    main()
