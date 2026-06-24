#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "result" / "2026-06-22_11-15-58-195_tau_live_tool_formal_benchmark"
CONDITION_NAMES = {
    "no_tick": "native_no_tick",
    "with_tick": "external_single_tick",
}
ANSWER_KEYWORDS = ["shipped", "ups", "tracking", "tomorrow", "1z999aa10123456784"]
PCM_BYTES_PER_SECOND = 48_000
AUDIO_SEGMENT_MERGE_GAP_MS = 200


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def normalize_text(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def keyword_hit(text: str) -> bool:
    lowered = text.lower()
    return any(keyword in lowered for keyword in ANSWER_KEYWORDS)


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def attempt_sort_key(path: Path) -> tuple[int, str]:
    try:
        return int(path.name.replace("attempt_", "")), path.name
    except ValueError:
        return 999999, path.name


def collect_attempts(result_dir: Path) -> list[dict]:
    rows: list[dict] = []
    organized = result_dir / "organized"
    for condition_label, condition in CONDITION_NAMES.items():
        for latency_dir in sorted((organized / condition_label).glob("*ms")):
            latency_ms = int(latency_dir.name.removesuffix("ms"))
            for attempt_dir in sorted(latency_dir.glob("attempt_*"), key=attempt_sort_key):
                summary_path = attempt_dir / "formal_attempt_summary.json"
                if not summary_path.exists():
                    summary_path = attempt_dir / "summary.json"
                summary = read_json(summary_path) if summary_path.exists() else {}
                audio_path = attempt_dir / "audio" / "assistant_output_compressed.wav"
                timeline_audio_path = attempt_dir / "audio" / "assistant_output_timeline.wav"
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": condition_label,
                        "latency_ms": latency_ms,
                        "attempt_id": attempt_dir.name,
                        "attempt_dir": attempt_dir,
                        "audio_path": audio_path,
                        "timeline_audio_path": timeline_audio_path,
                        "summary": summary,
                        "valid": bool(summary.get("session_valid")),
                        "close_1008": bool(summary.get("close_1008")),
                        "final_tool_response_sent_time_ms": summary.get("final_tool_response_sent_time_ms"),
                    }
                )
    return rows


def unavailable_asr(attempt: dict, model_name: str, device: str, compute_type: str, reason: str) -> dict:
    return {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "audio_file": rel(attempt["audio_path"]),
        "transcript": "",
        "segments": [],
        "asr_available": False,
        "asr_error": reason,
    }


def transcribe_audio(model, attempt: dict, model_name: str, device: str, compute_type: str) -> dict:
    segments_iter, info = model.transcribe(str(attempt["audio_path"]))
    segments: list[dict] = []
    transcript_parts: list[str] = []
    for index, segment in enumerate(segments_iter):
        text = normalize_text(segment.text)
        transcript_parts.append(text)
        segments.append(
            {
                "segment_index": index,
                "raw_start": float(segment.start),
                "raw_end": float(segment.end),
                "start": float(segment.start),
                "end": float(segment.end),
                "text": text,
            }
        )
    return {
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "audio_file": rel(attempt["audio_path"]),
        "transcript": normalize_text(" ".join(part for part in transcript_parts if part)),
        "segments": segments,
        "asr_available": True,
        "asr_error": None,
    }


def write_per_audio_outputs(asr: dict, audio_path: Path) -> tuple[Path, Path]:
    txt_path = audio_path.with_suffix(".asr.txt")
    json_path = audio_path.with_suffix(".asr.json")
    txt_path.write_text(asr.get("transcript") or f"ASR unavailable: {asr.get('asr_error')}\n", encoding="utf-8")
    json_path.write_text(json.dumps(asr, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt_path, json_path


def audio_output_events(attempt_dir: Path) -> list[dict]:
    events_path = attempt_dir / "timeline" / "events.jsonl"
    if not events_path.exists():
        return []
    return [
        event
        for event in read_jsonl(events_path)
        if event.get("type") == "audio_output"
        and isinstance(event.get("event_ms"), (int, float))
        and int(event.get("bytes") or 0) > 0
    ]


def compressed_to_global_map(attempt_dir: Path) -> list[dict]:
    entries: list[dict] = []
    speech_cursor_ms = 0.0
    global_cursor_ms = 0.0
    for event in audio_output_events(attempt_dir):
        byte_count = int(event.get("bytes") or 0)
        duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
        requested_start_ms = float(event["event_ms"])
        global_start_ms = max(requested_start_ms, global_cursor_ms)
        entries.append(
            {
                "speech_start_ms": speech_cursor_ms,
                "speech_end_ms": speech_cursor_ms + duration_ms,
                "global_start_ms": global_start_ms,
                "global_end_ms": global_start_ms + duration_ms,
            }
        )
        speech_cursor_ms += duration_ms
        global_cursor_ms = global_start_ms + duration_ms
    return entries


def map_speech_time_to_global_ms(mapping: list[dict], speech_ms: float) -> float | None:
    if not mapping:
        return None
    if speech_ms <= mapping[0]["speech_start_ms"]:
        return mapping[0]["global_start_ms"]
    for entry in mapping:
        if entry["speech_start_ms"] <= speech_ms <= entry["speech_end_ms"]:
            return entry["global_start_ms"] + (speech_ms - entry["speech_start_ms"])
    last = mapping[-1]
    if speech_ms >= last["speech_end_ms"]:
        return last["global_end_ms"]
    return None


def align_asr_segments_to_global(asr: dict, attempt_dir: Path) -> tuple[list[dict], str]:
    mapping = compressed_to_global_map(attempt_dir)
    aligned = []
    alignment_mode = "compressed_audio_mapped_to_timeline_audio_chunks"
    for segment in asr.get("segments") or []:
        raw_start_ms = float(segment.get("raw_start", segment.get("start", 0.0))) * 1000
        raw_end_ms = float(segment.get("raw_end", segment.get("end", 0.0))) * 1000
        global_start_ms = map_speech_time_to_global_ms(mapping, raw_start_ms)
        global_end_ms = map_speech_time_to_global_ms(mapping, raw_end_ms)
        item = dict(segment)
        item["raw_start"] = float(segment.get("raw_start", segment.get("start", 0.0)))
        item["raw_end"] = float(segment.get("raw_end", segment.get("end", 0.0)))
        if global_start_ms is None or global_end_ms is None:
            item["start"] = item["raw_start"]
            item["end"] = item["raw_end"]
            item["alignment_warning"] = "failed_to_map_raw_asr_time_to_global_time"
            alignment_mode = "raw_compressed_asr_time_unmapped"
        else:
            item["start"] = global_start_ms / 1000
            item["end"] = global_end_ms / 1000
            item["global_start_ms"] = round(global_start_ms, 3)
            item["global_end_ms"] = round(global_end_ms, 3)
        aligned.append(item)
    return aligned, alignment_mode


def relation_to_final(segment: dict, final_ms: object) -> str:
    if not isinstance(final_ms, (int, float)):
        return "unknown"
    start_ms = float(segment["start"]) * 1000
    end_ms = float(segment["end"]) * 1000
    if end_ms <= final_ms:
        return "pre_result"
    if start_ms >= final_ms:
        return "post_final"
    return "overlaps_final"


def split_transcripts(segments: list[dict], final_ms: object) -> dict:
    buckets = {"pre_result": [], "post_final": [], "overlaps_final": [], "unknown": []}
    for segment in segments:
        buckets[relation_to_final(segment, final_ms)].append(segment["text"])
    return {key: normalize_text(" ".join(value)) for key, value in buckets.items()}


def audio_chunks_from_timeline(attempt_dir: Path) -> list[dict]:
    chunks = []
    for event in audio_output_events(attempt_dir):
        byte_count = int(event.get("bytes") or 0)
        start_ms = float(event["event_ms"])
        duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
        chunks.append({"start_ms": start_ms, "end_ms": start_ms + duration_ms})
    return chunks


def merge_audio_chunks(chunks: list[dict]) -> list[dict]:
    merged: list[dict] = []
    for chunk in sorted(chunks, key=lambda item: item["start_ms"]):
        if not merged or chunk["start_ms"] - merged[-1]["end_ms"] > AUDIO_SEGMENT_MERGE_GAP_MS:
            merged.append({"start_ms": chunk["start_ms"], "end_ms": chunk["end_ms"]})
        else:
            merged[-1]["end_ms"] = max(merged[-1]["end_ms"], chunk["end_ms"])
    return merged


def overlap_ms(start_a: float, end_a: float, start_b: float, end_b: float) -> float:
    return max(0.0, min(end_a, end_b) - max(start_a, start_b))


def sanity_check(attempts: list[dict], attempt_results: dict[str, dict], sample_size: int = 8) -> list[dict]:
    samples = []
    candidates = [
        attempt
        for attempt in attempts
        if attempt_results.get(str(attempt["attempt_dir"]), {}).get("asr_available")
        and attempt_results.get(str(attempt["attempt_dir"]), {}).get("segments")
    ]
    for attempt in candidates[:sample_size]:
        asr = attempt_results[str(attempt["attempt_dir"])]
        segments = asr.get("segments", [])
        audio_blocks = merge_audio_chunks(audio_chunks_from_timeline(attempt["attempt_dir"]))
        if not segments or not audio_blocks:
            samples.append(
                {
                    "condition": attempt["condition"],
                    "latency_ms": attempt["latency_ms"],
                    "attempt_id": attempt["attempt_id"],
                    "status": "missing_asr_or_audio_blocks",
                    "segment_count": len(segments),
                    "audio_block_count": len(audio_blocks),
                }
            )
            continue
        aligned = 0
        deltas = []
        for segment in segments:
            start_ms = float(segment["start"]) * 1000
            end_ms = float(segment["end"]) * 1000
            overlaps = [overlap_ms(start_ms, end_ms, block["start_ms"], block["end_ms"]) for block in audio_blocks]
            if max(overlaps or [0]) > 0:
                aligned += 1
            nearest_start_delta = min(abs(start_ms - block["start_ms"]) for block in audio_blocks)
            deltas.append(nearest_start_delta)
        samples.append(
            {
                "condition": attempt["condition"],
                "latency_ms": attempt["latency_ms"],
                "attempt_id": attempt["attempt_id"],
                "status": "ok",
                "segment_count": len(segments),
                "audio_block_count": len(audio_blocks),
                "segments_overlapping_audio_blocks": aligned,
                "overlap_rate": round(aligned / len(segments), 3),
                "mean_nearest_audio_block_start_delta_ms": round(mean(deltas), 1) if deltas else None,
                "first_asr_segment_ms": round(float(segments[0]["start"]) * 1000, 1),
                "first_audio_block_ms": round(audio_blocks[0]["start_ms"], 1),
            }
        )
    return samples


def by_cell_summary(attempt_rows: list[dict]) -> list[dict]:
    aggregate: dict[tuple[str, int], dict] = {}
    for row in attempt_rows:
        key = (row["condition"], row["latency_ms"])
        item = aggregate.setdefault(
            key,
            {
                "condition": row["condition"],
                "latency_ms": row["latency_ms"],
                "attempts": 0,
                "attempts_with_audio": 0,
                "attempts_with_asr": 0,
                "attempts_failed_asr": 0,
                "pre_result_contains_answer_keywords_count": 0,
                "post_final_contains_answer_keywords_count": 0,
            },
        )
        item["attempts"] += 1
        if row["audio_exists"]:
            item["attempts_with_audio"] += 1
        if row["asr_success"]:
            item["attempts_with_asr"] += 1
        else:
            item["attempts_failed_asr"] += 1
        if row["pre_result_contains_answer_keywords"]:
            item["pre_result_contains_answer_keywords_count"] += 1
        if row["post_final_contains_answer_keywords"]:
            item["post_final_contains_answer_keywords_count"] += 1
    return sorted(aggregate.values(), key=lambda item: (item["condition"], item["latency_ms"]))


def write_notes(path: Path, args: argparse.Namespace, summary: dict, sanity: list[dict], failed: list[dict]) -> None:
    lines = [
        "# Formal Benchmark ASR Notes",
        "",
        f"Result folder: `{rel(args.result_dir)}`",
        "",
        "## ASR Settings",
        "",
        f"- ASR engine: `faster-whisper`",
        f"- Model: `{args.model}`",
        f"- Device: `{args.device}`",
        f"- Compute type: `{args.compute_type}`",
        "- Timestamp granularity: segment-level timestamps.",
        "- Word-level timestamps: not requested / not emitted.",
        "",
        "## Audio / Timestamp Assumption",
        "",
        "- ASR source: `audio/assistant_output_compressed.wav` for each attempt.",
        "- Timeline reference: `audio/assistant_output_timeline.wav` plus `timeline/events.jsonl` audio chunks.",
        "- Sanity check found direct faster-whisper timestamps on silence-padded timeline audio can start at 0 even when first audio appears later. To avoid bad final-boundary splits, ASR is run on compressed audio and then mapped back to attempt-global timeline timestamps using the same audio chunk placement used to build timeline audio.",
        "- Segment splitting uses `final_tool_response_sent_time_ms` from each attempt summary.",
        "",
        "## Sanity Check",
        "",
        "A small sample compares aligned ASR segment times against merged assistant audio blocks from `timeline/events.jsonl`.",
        "",
        "| condition | latency_ms | attempt | ASR segments | audio blocks | overlap rate | first ASR ms | first audio ms | mean nearest block start delta ms |",
        "|---|---:|---|---:|---:|---:|---:|---:|---:|",
    ]
    for item in sanity:
        lines.append(
            f"| {item.get('condition')} | {item.get('latency_ms')} | {item.get('attempt_id')} | "
            f"{item.get('segment_count', '')} | {item.get('audio_block_count', '')} | {item.get('overlap_rate', '')} | "
            f"{item.get('first_asr_segment_ms', '')} | {item.get('first_audio_block_ms', '')} | "
            f"{item.get('mean_nearest_audio_block_start_delta_ms', '')} |"
        )
    lines.extend(
        [
            "",
            "Interpretation: overlap rate near 1.0 indicates ASR segments land inside timeline audio blocks. Differences of a few hundred ms are expected because ASR segments are VAD/decoder segments, not raw PCM chunk boundaries.",
            "",
            "## Risks",
            "",
            "- ASR may mishear tracking numbers, short filler words, or synthetic speech artifacts.",
            "- Segment boundaries can be coarse and may straddle the final tool response boundary; such segments are classified as `overlaps_final` and kept separate.",
            "- The emitted `segments` in `*.asr.json`, `asr_segments.csv`, and `asr_attempts.csv` use aligned attempt-global timestamps. Raw compressed-audio ASR timestamps are preserved as `raw_segments` in each per-audio JSON.",
            "- This alignment assumes compressed audio was built by concatenating assistant audio chunks in timeline order, which matches the postprocessing code that generated these artifacts.",
            "",
            "## Coverage",
            "",
            f"- total_attempts: {summary['total_attempts']}",
            f"- attempts_with_audio: {summary['attempts_with_audio']}",
            f"- attempts_with_asr: {summary['attempts_with_asr']}",
            f"- attempts_failed_asr: {summary['attempts_failed_asr']}",
            "",
        ]
    )
    if failed:
        lines.extend(["## ASR Failures", "", "| condition | latency_ms | attempt | audio | error |", "|---|---:|---|---|---|"])
        for row in failed:
            lines.append(f"| {row['condition']} | {row['latency_ms']} | {row['attempt_id']} | `{row['audio_file']}` | {row['asr_error']} |")
    else:
        lines.extend(["## ASR Failures", "", "No ASR failures recorded."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run ASR over formal benchmark timeline audio and split transcripts around final tool response.")
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    args.result_dir = args.result_dir.resolve()

    attempts = collect_attempts(args.result_dir)
    if args.limit:
        attempts = attempts[: args.limit]

    reason = None
    model = None
    if not faster_whisper_available():
        reason = "missing faster-whisper Python package"
    else:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        except Exception as error:
            reason = f"failed to initialize faster-whisper model: {error}"

    segment_rows: list[dict] = []
    attempt_rows: list[dict] = []
    attempt_results: dict[str, dict] = {}

    for index, attempt in enumerate(attempts, start=1):
        audio_path = attempt["audio_path"]
        audio_exists = audio_path.exists()
        if not audio_exists:
            asr = unavailable_asr(attempt, args.model, args.device, args.compute_type, "missing compressed audio")
        elif model is None:
            asr = unavailable_asr(attempt, args.model, args.device, args.compute_type, reason or "ASR unavailable")
        else:
            json_path = audio_path.with_suffix(".asr.json")
            if json_path.exists() and not args.force:
                asr = read_json(json_path)
            else:
                try:
                    asr = transcribe_audio(model, attempt, args.model, args.device, args.compute_type)
                except Exception as error:
                    asr = unavailable_asr(attempt, args.model, args.device, args.compute_type, f"ASR failed: {error}")
                    asr["asr_available"] = True
        if asr.get("asr_available") and not asr.get("asr_error"):
            raw_segments = list(asr.get("segments") or [])
            aligned_segments, alignment_mode = align_asr_segments_to_global(asr, attempt["attempt_dir"])
            asr["raw_segments"] = raw_segments
            asr["segments"] = aligned_segments
            asr["alignment_mode"] = alignment_mode
            asr["timeline_audio_file"] = rel(attempt["timeline_audio_path"])
        if audio_exists:
            txt_path, json_path = write_per_audio_outputs(asr, audio_path)
        else:
            txt_path, json_path = None, None

        attempt_results[str(attempt["attempt_dir"])] = asr
        final_ms = attempt["final_tool_response_sent_time_ms"]
        segments = asr.get("segments") or []
        split = split_transcripts(segments, final_ms)
        full = asr.get("transcript") or normalize_text(" ".join(segment.get("text", "") for segment in segments))

        for segment in segments:
            start_sec = float(segment["start"])
            end_sec = float(segment["end"])
            relation = relation_to_final(segment, final_ms)
            segment_rows.append(
                {
                    "condition": attempt["condition"],
                    "latency_ms": attempt["latency_ms"],
                    "attempt_id": attempt["attempt_id"],
                    "valid": attempt["valid"],
                    "audio_file": rel(audio_path),
                    "timeline_audio_file": rel(attempt["timeline_audio_path"]),
                    "segment_index": segment.get("segment_index"),
                    "raw_segment_start_sec": round(float(segment.get("raw_start", segment["start"])), 3),
                    "raw_segment_end_sec": round(float(segment.get("raw_end", segment["end"])), 3),
                    "segment_start_sec": round(start_sec, 3),
                    "segment_end_sec": round(end_sec, 3),
                    "segment_start_ms": round(start_sec * 1000, 1),
                    "segment_end_ms": round(end_sec * 1000, 1),
                    "final_tool_response_sent_time_ms": final_ms if isinstance(final_ms, (int, float)) else "",
                    "relation_to_final": relation,
                    "text": segment.get("text", ""),
                }
            )

        attempt_rows.append(
            {
                "condition": attempt["condition"],
                "latency_ms": attempt["latency_ms"],
                "attempt_id": attempt["attempt_id"],
                "valid": attempt["valid"],
                "audio_exists": audio_exists,
                "audio_file": rel(audio_path),
                "timeline_audio_file": rel(attempt["timeline_audio_path"]),
                "asr_success": bool(asr.get("asr_available") and not asr.get("asr_error")),
                "asr_error": asr.get("asr_error"),
                "alignment_mode": asr.get("alignment_mode", ""),
                "asr_txt_path": rel(txt_path) if txt_path else "",
                "asr_json_path": rel(json_path) if json_path else "",
                "final_tool_response_sent_time_ms": final_ms if isinstance(final_ms, (int, float)) else "",
                "full_transcript": full,
                "pre_result_transcript": split["pre_result"],
                "post_final_transcript": split["post_final"],
                "overlaps_final_transcript": split["overlaps_final"],
                "unknown_transcript": split["unknown"],
                "post_final_contains_answer_keywords": keyword_hit(split["post_final"]),
                "pre_result_contains_answer_keywords": keyword_hit(split["pre_result"]),
            }
        )
        print(f"[{index}/{len(attempts)}] {attempt['condition']} {attempt['latency_ms']} {attempt['attempt_id']} asr_success={attempt_rows[-1]['asr_success']}")

    failed = [row for row in attempt_rows if not row["asr_success"]]
    by_cell = by_cell_summary(attempt_rows)
    summary = {
        "result_dir": rel(args.result_dir),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "timestamp_granularity": "segment-level",
        "total_attempts": len(attempt_rows),
        "attempts_with_audio": sum(1 for row in attempt_rows if row["audio_exists"]),
        "attempts_with_asr": sum(1 for row in attempt_rows if row["asr_success"]),
        "attempts_failed_asr": len(failed),
        "by_condition_latency": by_cell,
        "failed_attempts": failed,
    }

    write_csv(
        args.result_dir / "asr_segments.csv",
        segment_rows,
        [
            "condition",
            "latency_ms",
            "attempt_id",
            "valid",
            "audio_file",
            "timeline_audio_file",
            "segment_index",
            "raw_segment_start_sec",
            "raw_segment_end_sec",
            "segment_start_sec",
            "segment_end_sec",
            "segment_start_ms",
            "segment_end_ms",
            "final_tool_response_sent_time_ms",
            "relation_to_final",
            "text",
        ],
    )
    write_csv(
        args.result_dir / "asr_attempts.csv",
        attempt_rows,
        [
            "condition",
            "latency_ms",
            "attempt_id",
            "valid",
            "audio_exists",
            "audio_file",
            "timeline_audio_file",
            "asr_success",
            "asr_error",
            "alignment_mode",
            "asr_txt_path",
            "asr_json_path",
            "final_tool_response_sent_time_ms",
            "full_transcript",
            "pre_result_transcript",
            "post_final_transcript",
            "overlaps_final_transcript",
            "unknown_transcript",
            "post_final_contains_answer_keywords",
            "pre_result_contains_answer_keywords",
        ],
    )
    (args.result_dir / "asr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    sanity = sanity_check(attempts, attempt_results)
    write_notes(args.result_dir / "asr_notes.md", args, summary, sanity, failed)
    print(f"ASR attempts: {rel(args.result_dir / 'asr_attempts.csv')}")
    print(f"ASR segments: {rel(args.result_dir / 'asr_segments.csv')}")
    print(f"ASR summary: {rel(args.result_dir / 'asr_summary.json')}")
    print(f"ASR notes: {rel(args.result_dir / 'asr_notes.md')}")


if __name__ == "__main__":
    main()
