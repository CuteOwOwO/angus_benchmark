#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_RESULT_DIR = ROOT / "result" / "2026-06-29_10-15-29-961_calendar_route_two_step_2step_tick_vs_no_tick_latency_sweep(2tool大跑)"

AUDIO_RE = re.compile(
    r"calendar_route_two_step_(?P<condition>no_tick|periodic_tick_4s)_2step__latency(?P<latency_ms>\d+)__attempt(?P<attempt>\d+)\.assistant\.wav$"
)


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip())


def write_csv(path: Path, rows: list[dict], headers: list[str]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def collect_audio(result_dir: Path) -> list[dict]:
    audio_dir = result_dir / "organized" / "audio"
    rows: list[dict] = []
    for audio_path in sorted(audio_dir.glob("*.assistant.wav")):
        match = AUDIO_RE.match(audio_path.name)
        if not match:
            continue
        condition = match.group("condition")
        latency_ms = int(match.group("latency_ms"))
        attempt = int(match.group("attempt"))
        rows.append(
            {
                "condition": condition,
                "latency_ms": latency_ms,
                "latency_s": latency_ms // 1000,
                "attempt_index": attempt,
                "attempt_id": f"attempt{attempt:03d}",
                "audio_path": audio_path,
            }
        )
    return rows


def transcribe_audio(model, item: dict, model_name: str, device: str, compute_type: str) -> dict:
    segments_iter, info = model.transcribe(str(item["audio_path"]))
    segments: list[dict] = []
    parts: list[str] = []
    for index, segment in enumerate(segments_iter):
        text = compact(segment.text)
        if text:
            parts.append(text)
        segments.append(
            {
                "segment_index": index,
                "start_sec": round(float(segment.start), 3),
                "end_sec": round(float(segment.end), 3),
                "text": text,
            }
        )
    return {
        "condition": item["condition"],
        "latency_ms": item["latency_ms"],
        "latency_s": item["latency_s"],
        "attempt_index": item["attempt_index"],
        "attempt_id": item["attempt_id"],
        "audio_path": rel(item["audio_path"]),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "language": getattr(info, "language", None),
        "language_probability": getattr(info, "language_probability", None),
        "asr_available": True,
        "asr_error": None,
        "transcript": compact(" ".join(parts)),
        "segments": segments,
    }


def unavailable(item: dict, model_name: str, device: str, compute_type: str, reason: str) -> dict:
    return {
        "condition": item["condition"],
        "latency_ms": item["latency_ms"],
        "latency_s": item["latency_s"],
        "attempt_index": item["attempt_index"],
        "attempt_id": item["attempt_id"],
        "audio_path": rel(item["audio_path"]),
        "model": model_name,
        "device": device,
        "compute_type": compute_type,
        "asr_available": False,
        "asr_error": reason,
        "transcript": "",
        "segments": [],
    }


def write_per_audio_outputs(out_dir: Path, row: dict) -> tuple[Path, Path]:
    stem = Path(row["audio_path"]).name.removesuffix(".wav")
    txt_path = out_dir / f"{stem}.asr.txt"
    json_path = out_dir / f"{stem}.asr.json"
    txt_path.write_text(row.get("transcript") or f"ASR unavailable: {row.get('asr_error')}\n", encoding="utf-8")
    json_path.write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt_path, json_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", nargs="?", type=Path, default=DEFAULT_RESULT_DIR)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    out_dir = result_dir / "organized" / "asr"
    out_dir.mkdir(parents=True, exist_ok=True)

    audio_items = collect_audio(result_dir)
    if args.limit is not None:
        audio_items = audio_items[: args.limit]

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

    rows: list[dict] = []
    segment_rows: list[dict] = []
    for index, item in enumerate(audio_items, start=1):
        stem = item["audio_path"].name.removesuffix(".wav")
        json_path = out_dir / f"{stem}.asr.json"
        if json_path.exists() and not args.force:
            row = json.loads(json_path.read_text(encoding="utf-8"))
        elif model is None:
            row = unavailable(item, args.model, args.device, args.compute_type, reason or "ASR unavailable")
        else:
            try:
                row = transcribe_audio(model, item, args.model, args.device, args.compute_type)
            except Exception as error:
                row = unavailable(item, args.model, args.device, args.compute_type, f"ASR failed: {error}")
        txt_path, json_path = write_per_audio_outputs(out_dir, row)
        row["asr_txt_path"] = rel(txt_path)
        row["asr_json_path"] = rel(json_path)
        rows.append(row)
        for segment in row.get("segments") or []:
            segment_rows.append(
                {
                    "condition": row["condition"],
                    "latency_ms": row["latency_ms"],
                    "latency_s": row["latency_s"],
                    "attempt_index": row["attempt_index"],
                    "attempt_id": row["attempt_id"],
                    "audio_path": row["audio_path"],
                    "segment_index": segment.get("segment_index"),
                    "start_sec": segment.get("start_sec"),
                    "end_sec": segment.get("end_sec"),
                    "text": segment.get("text", ""),
                }
            )
        print(
            f"[{index}/{len(audio_items)}] {item['condition']} latency={item['latency_ms']} "
            f"{item['attempt_id']} asr_success={row.get('asr_available') and not row.get('asr_error')}"
        )

    summary = {
        "result_dir": rel(result_dir),
        "audio_scope": "assistant_compressed_audio_only",
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "audio_count": len(audio_items),
        "asr_success_count": sum(1 for row in rows if row.get("asr_available") and not row.get("asr_error")),
        "asr_failed_count": sum(1 for row in rows if not (row.get("asr_available") and not row.get("asr_error"))),
        "rows": rows,
    }
    (out_dir / "asr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "asr_summary.csv",
        rows,
        [
            "condition",
            "latency_ms",
            "latency_s",
            "attempt_index",
            "attempt_id",
            "audio_path",
            "asr_available",
            "asr_error",
            "asr_txt_path",
            "asr_json_path",
            "transcript",
        ],
    )
    write_csv(
        out_dir / "asr_segments.csv",
        segment_rows,
        [
            "condition",
            "latency_ms",
            "latency_s",
            "attempt_index",
            "attempt_id",
            "audio_path",
            "segment_index",
            "start_sec",
            "end_sec",
            "text",
        ],
    )
    print(f"ASR summary: {rel(out_dir / 'asr_summary.json')}")
    print(f"ASR CSV: {rel(out_dir / 'asr_summary.csv')}")
    print(f"ASR segments: {rel(out_dir / 'asr_segments.csv')}")


if __name__ == "__main__":
    main()
