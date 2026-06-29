#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import wave
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
AUDIO_BAR_HEIGHT = 0.13
AUDIO_MERGE_GAP_MS = 200


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def compact(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def csv_escape(value: Any) -> Any:
    if isinstance(value, (list, dict)):
        return json.dumps(value, ensure_ascii=False)
    return value


def write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    headers = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: csv_escape(row.get(key)) for key in headers})


def event_times(events: list[dict], event_type: str) -> list[int]:
    return [int(event["event_ms"]) for event in events if event.get("type") == event_type and event.get("event_ms") is not None]


def text_events(events: list[dict]) -> list[dict]:
    out = []
    for event in events:
        if event.get("type") != "text_output" or event.get("event_ms") is None or not event.get("text"):
            continue
        out.append({"time_ms": int(event["event_ms"]), "text": compact(str(event["text"]))})
    return out


def merge_audio(intervals: list[dict]) -> list[dict]:
    merged = []
    for interval in sorted(intervals, key=lambda item: item["start_ms"]):
        start = float(interval["start_ms"])
        end = float(interval["end_ms"])
        if not merged or start - merged[-1]["end_ms"] > AUDIO_MERGE_GAP_MS:
            merged.append({"start_ms": start, "end_ms": end, "chunk_count": 1})
            continue
        merged[-1]["end_ms"] = max(merged[-1]["end_ms"], end)
        merged[-1]["chunk_count"] += 1
    return merged


def collect_attempts(result_dir: Path) -> list[dict]:
    attempts = []
    for attempt_dir in sorted((result_dir / "attempts").glob("attempt_*")):
        summary_path = attempt_dir / "attempt_summary.json"
        stage_path = attempt_dir / "stage_records.json"
        audio_path = attempt_dir / "audio_intervals.json"
        timeline_paths = sorted(attempt_dir.glob("*.timeline.jsonl"))
        if not summary_path.exists() or not stage_path.exists() or not timeline_paths:
            continue
        summary = read_json(summary_path)
        stages = read_json(stage_path)
        audio_intervals = read_json(audio_path) if audio_path.exists() else []
        events = read_jsonl(timeline_paths[0])
        attempts.append(
            {
                "attempt_dir": attempt_dir,
                "name": attempt_dir.name,
                "summary": summary,
                "stages": stages,
                "audio_segments": merge_audio(audio_intervals),
                "text_events": text_events(events),
                "tick_times": event_times(events, "pending_tick_sent"),
                "close_times": event_times(events, "session_closed"),
            }
        )
    return attempts


def max_time_ms(attempts: list[dict]) -> int:
    max_ms = 18_000
    for attempt in attempts:
        for stage in attempt["stages"]:
            max_ms = max(max_ms, int(stage.get("call_ms") or 0), int(stage.get("response_ms") or 0))
            for tick in stage.get("pending_tick_times_ms", []):
                max_ms = max(max_ms, int(tick))
        for segment in attempt["audio_segments"]:
            max_ms = max(max_ms, int(segment["end_ms"]))
        for text in attempt["text_events"]:
            max_ms = max(max_ms, int(text["time_ms"]))
        for close in attempt["close_times"]:
            max_ms = max(max_ms, int(close))
    return int(math.ceil((max_ms + 1000) / 1000) * 1000)


def stage_color(index: int) -> str:
    return "#7B3FCE" if index == 1 else "#0A7B55"


def draw_timeline(attempts: list[dict], out_path: Path) -> None:
    x_max = max_time_ms(attempts)
    height = max(4.8, 1.7 + 0.62 * len(attempts))
    fig, ax = plt.subplots(figsize=(15, height))
    ax.set_title("Two-step airline native tool-wait pilot: tick4s, latency8000")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylabel("attempt")
    ax.set_xlim(0, x_max / 1000)
    ax.set_ylim(-0.6, len(attempts) - 0.4)
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    y_positions = list(range(len(attempts)))[::-1]
    ax.set_yticks(y_positions)
    ax.set_yticklabels([attempt["name"].replace("attempt_", "a") for attempt in attempts], fontsize=9)

    for y, attempt in zip(y_positions, attempts):
        ax.hlines(y, 0, x_max / 1000, color="#C9CDD1", linewidth=0.8, zorder=0)
        for stage in attempt["stages"][:2]:
            index = int(stage.get("index") or 0)
            call = stage.get("call_ms")
            response = stage.get("response_ms")
            color = stage_color(index)
            lane_y = y + (0.12 if index == 1 else -0.12)
            if call is not None and response is not None and response > call:
                ax.add_patch(
                    Rectangle(
                        (call / 1000, lane_y - 0.11),
                        (response - call) / 1000,
                        0.22,
                        facecolor=color,
                        edgecolor="none",
                        alpha=0.09,
                        zorder=0,
                    )
                )
            if call is not None:
                ax.vlines(call / 1000, lane_y - 0.2, lane_y + 0.2, color=color, linewidth=1.7, zorder=4)
            if response is not None:
                ax.vlines(response / 1000, lane_y - 0.2, lane_y + 0.2, color=color, linewidth=1.7, linestyle="--", zorder=4)
            for tick in stage.get("pending_tick_times_ms", []):
                ax.vlines(tick / 1000, lane_y - 0.18, lane_y + 0.18, color="#6B7280", linewidth=1.2, linestyle=":", zorder=5)
                ax.scatter([tick / 1000], [lane_y], marker="o", s=24, color="#F59E0B", edgecolor="white", linewidth=0.4, zorder=6)

        for segment in attempt["audio_segments"]:
            ax.broken_barh(
                [(segment["start_ms"] / 1000, (segment["end_ms"] - segment["start_ms"]) / 1000)],
                (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
                facecolors="#1E88E5",
                edgecolors="#0D47A1",
                linewidth=0.65,
                zorder=2,
            )
        if attempt["text_events"]:
            ax.scatter(
                [text["time_ms"] / 1000 for text in attempt["text_events"]],
                [y] * len(attempt["text_events"]),
                color="#F97316",
                edgecolor="white",
                linewidth=0.3,
                s=18,
                zorder=5,
            )
        for close in attempt["close_times"]:
            ax.scatter([close / 1000], [y], marker="x", color="#D81B60", s=34, linewidth=1.4, zorder=7)

    handles = [
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="assistant audio"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F97316", label="text/transcription event", markersize=6),
        plt.Line2D([], [], color="#7B3FCE", label="stage 1 call/result"),
        plt.Line2D([], [], color="#0A7B55", label="stage 2 call/result"),
        plt.Line2D([], [], color="#6B7280", linestyle=":", label="pending tick"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F59E0B", label="tick event", markersize=6),
    ]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.5, -0.28), ncol=6, fontsize=8, frameon=False)
    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    fig.subplots_adjust(bottom=0.22)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def draw_stage_bars(attempts: list[dict], out_path: Path) -> None:
    labels = [attempt["name"].replace("attempt_", "a") for attempt in attempts]
    stage1_gap = [attempt["summary"].get("stage_1_max_silence_gap_before_tool_result_ms") or 0 for attempt in attempts]
    stage2_gap = [attempt["summary"].get("stage_2_max_silence_gap_before_tool_result_ms") or 0 for attempt in attempts]
    x = list(range(len(labels)))
    fig, ax = plt.subplots(figsize=(10, 4.6))
    ax.bar([i - 0.18 for i in x], stage1_gap, width=0.36, label="stage 1", color="#7B3FCE")
    ax.bar([i + 0.18 for i in x], stage2_gap, width=0.36, label="stage 2", color="#0A7B55")
    ax.set_title("Max silence gap before each tool result")
    ax.set_ylabel("milliseconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    ax.legend(frameon=False)
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def write_stage_transcripts(attempts: list[dict], out_dir: Path) -> None:
    rows = []
    for attempt in attempts:
        summary = attempt["summary"]
        for stage in attempt["stages"][:2]:
            index = int(stage.get("index") or 0)
            text = compact(" ".join(event.get("text", "") for event in stage.get("text_events", [])))
            row = {
                "attempt_index": summary["attempt_index"],
                "stage_index": index,
                "tool_name": stage.get("tool_name"),
                "call_ms": stage.get("call_ms"),
                "response_ms": stage.get("response_ms"),
                "pending_tick_times_ms": stage.get("pending_tick_times_ms", []),
                "transcript": text,
            }
            rows.append(row)
            (out_dir / f"attempt{summary['attempt_index']:03d}_stage{index}_{stage.get('tool_name')}.txt").write_text(text + "\n", encoding="utf-8")
    write_csv(out_dir / "stage_transcripts.csv", rows)
    (out_dir / "stage_transcripts.json").write_text(json.dumps(rows, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def write_timeline_audio_for_attempt(attempt: dict, organized_audio_dir: Path) -> Path | None:
    attempt_dir = attempt["attempt_dir"]
    compressed_candidates = sorted(attempt_dir.glob("*.assistant.wav"))
    if not compressed_candidates:
        return None
    compressed_path = compressed_candidates[0]
    intervals_path = attempt_dir / "audio_intervals.json"
    if not intervals_path.exists():
        return None
    intervals = read_json(intervals_path)
    if not intervals:
        return None

    with wave.open(str(compressed_path), "rb") as src:
        params = src.getparams()
        frames = src.readframes(src.getnframes())

    pieces: list[bytes] = []
    source_cursor = 0
    timeline_cursor = 0
    for interval in intervals:
        chunk_bytes = int(interval.get("bytes") or 0)
        if chunk_bytes <= 0:
            continue
        start_bytes = max(0, round(float(interval["start_ms"]) / 1000 * params.framerate * params.sampwidth * params.nchannels))
        if start_bytes > timeline_cursor:
            pieces.append(bytes(start_bytes - timeline_cursor))
            timeline_cursor = start_bytes
        chunk = frames[source_cursor : source_cursor + chunk_bytes]
        if not chunk:
            break
        pieces.append(chunk)
        source_cursor += len(chunk)
        timeline_cursor += len(chunk)

    if not pieces:
        return None
    out_path = organized_audio_dir / compressed_path.name.replace(".assistant.wav", ".assistant_timeline.wav")
    with wave.open(str(out_path), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(b"".join(pieces))
    local_out = attempt_dir / out_path.name
    with wave.open(str(local_out), "wb") as dst:
        dst.setparams(params)
        dst.writeframes(b"".join(pieces))
    return out_path


def write_timeline_audio(attempts: list[dict], organized_audio_dir: Path) -> list[Path]:
    organized_audio_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for attempt in attempts:
        path = write_timeline_audio_for_attempt(attempt, organized_audio_dir)
        if path:
            outputs.append(path)
    return outputs


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def run_asr(result_dir: Path, out_dir: Path, model_name: str, device: str, compute_type: str) -> None:
    audio_dir = result_dir / "organized" / "audio"
    audio_files = sorted(audio_dir.glob("*.wav"))
    rows = []
    if not faster_whisper_available():
        for audio_path in audio_files:
            rows.append(
                {
                    "audio_path": rel(audio_path),
                    "asr_available": False,
                    "asr_error": "faster_whisper not installed",
                    "transcript": "",
                    "segments": [],
                }
            )
    else:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name, device=device, compute_type=compute_type)
        for audio_path in audio_files:
            segments_iter, _info = model.transcribe(str(audio_path))
            segments = []
            parts = []
            for segment in segments_iter:
                text = segment.text.strip()
                if text:
                    parts.append(text)
                segments.append({"start": float(segment.start), "end": float(segment.end), "text": text})
            transcript = compact(" ".join(parts))
            row = {
                "audio_path": rel(audio_path),
                "asr_available": True,
                "asr_error": None,
                "model": model_name,
                "device": device,
                "compute_type": compute_type,
                "transcript": transcript,
                "segments": segments,
            }
            rows.append(row)
            stem = audio_path.stem
            (out_dir / f"{stem}.asr.txt").write_text(transcript + "\n", encoding="utf-8")
            (out_dir / f"{stem}.asr.json").write_text(json.dumps(row, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    summary = {
        "audio_count": len(audio_files),
        "asr_success_count": sum(1 for row in rows if row.get("asr_available") and not row.get("asr_error")),
        "rows": rows,
    }
    (out_dir / "asr_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_csv(
        out_dir / "asr_summary.csv",
        [
            {
                "audio_path": row["audio_path"],
                "asr_available": row["asr_available"],
                "asr_error": row.get("asr_error"),
                "transcript": row.get("transcript", ""),
            }
            for row in rows
        ],
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("result_dir", type=Path)
    parser.add_argument("--asr", action="store_true")
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    args = parser.parse_args()

    result_dir = args.result_dir.resolve()
    organized = result_dir / "organized"
    viz_dir = organized / "visualizations"
    asr_dir = organized / "asr"
    judge_dir = organized / "judge_outputs"
    viz_dir.mkdir(parents=True, exist_ok=True)
    asr_dir.mkdir(parents=True, exist_ok=True)
    judge_dir.mkdir(parents=True, exist_ok=True)

    attempts = collect_attempts(result_dir)
    if not attempts:
        raise SystemExit(f"No attempts found in {result_dir}")

    timeline_path = viz_dir / "airline_suitcase_2step__tick4s__latency8000_timechart.png"
    gaps_path = viz_dir / "airline_suitcase_2step__tick4s__latency8000_silence_gaps.png"
    draw_timeline(attempts, timeline_path)
    draw_stage_bars(attempts, gaps_path)
    write_stage_transcripts(attempts, judge_dir)
    timeline_audio_paths = write_timeline_audio(attempts, organized / "audio")
    if args.asr:
        run_asr(result_dir, asr_dir, args.model, args.device, args.compute_type)

    print("Two-step airline postprocess outputs:")
    print(f"- {rel(timeline_path)}")
    print(f"- {rel(gaps_path)}")
    print(f"- {rel(judge_dir / 'stage_transcripts.json')}")
    for path in timeline_audio_paths:
        print(f"- {rel(path)}")
    if args.asr:
        print(f"- {rel(asr_dir / 'asr_summary.json')}")


if __name__ == "__main__":
    main()
