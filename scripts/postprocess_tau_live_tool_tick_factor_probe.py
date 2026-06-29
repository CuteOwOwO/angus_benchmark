#!/usr/bin/env python3
import argparse
import csv
import json
import shutil
import wave
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
PCM_BYTES_PER_SECOND = 48_000
AUDIO_SEGMENT_MERGE_GAP_MS = 200
AUDIO_BAR_HEIGHT = 0.12
MARKER_HALF_HEIGHT = 0.18
WAITING_REGION_HEIGHT = 0.24
WAITING_REGION_ALPHA = 0.08
BASELINE_COLOR = "#9AA0A6"
AUDIO_COLOR = "#1E88E5"
AUDIO_EDGE_COLOR = "#0D47A1"
TEXT_COLOR = "#F57C00"
TEXT_EDGE_COLOR = "white"
TICK_TYPES = {
    "client_status_tick_sent",
    "boundary_client_status_tick_sent",
    "interim_function_response_sent",
    "check_status_pending_response_sent",
    "main_pending_tool_response_sent",
}


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: object) -> None:
    path.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def audio_output_events(events: list[dict]) -> list[dict]:
    return [event for event in events if event.get("type") == "audio_output" and event.get("bytes", 0) > 0]


def wav_duration_ms(path: Path) -> int:
    with wave.open(str(path), "rb") as wav:
        return round(wav.getnframes() / wav.getframerate() * 1000)


def build_timeline_audio(compressed_path: Path, timeline_path: Path, events: list[dict]) -> dict:
    outputs = audio_output_events(events)
    if not compressed_path.exists() or not outputs:
        return {
            "compressed_audio_file": rel(compressed_path) if compressed_path.exists() else "none",
            "timeline_audio_file": "none",
            "compressed_duration_ms": wav_duration_ms(compressed_path) if compressed_path.exists() else None,
            "timeline_duration_ms": None,
            "audio_output_chunks": len(outputs),
        }

    with wave.open(str(compressed_path), "rb") as wav:
        params = wav.getparams()
        pcm = wav.readframes(wav.getnframes())

    bytes_per_ms = params.framerate * params.nchannels * params.sampwidth / 1000
    block_align = params.nchannels * params.sampwidth
    parts: list[bytes] = []
    cursor_bytes = 0
    source_cursor = 0

    for event in outputs:
        byte_count = int(event["bytes"])
        chunk = pcm[source_cursor : source_cursor + byte_count]
        source_cursor += byte_count
        requested_start = max(0, round(float(event.get("event_ms") or 0) * bytes_per_ms))
        aligned_start = requested_start - (requested_start % block_align)
        start_bytes = max(aligned_start, cursor_bytes)
        if start_bytes > cursor_bytes:
            parts.append(bytes(start_bytes - cursor_bytes))
            cursor_bytes = start_bytes
        parts.append(chunk)
        cursor_bytes += len(chunk)

    timeline_path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(timeline_path), "wb") as wav:
        wav.setparams(params)
        wav.writeframes(b"".join(parts))

    return {
        "compressed_audio_file": rel(compressed_path),
        "timeline_audio_file": rel(timeline_path),
        "compressed_duration_ms": wav_duration_ms(compressed_path),
        "timeline_duration_ms": wav_duration_ms(timeline_path),
        "audio_output_chunks": len(outputs),
        "source_bytes_used": source_cursor,
        "source_bytes_total": len(pcm),
    }


def first_time(events: list[dict], event_type: str) -> int | None:
    for event in events:
        if event.get("type") == event_type and event.get("event_ms") is not None:
            return int(event["event_ms"])
    return None


def event_times(events: list[dict], event_type: str) -> list[int]:
    return [int(event["event_ms"]) for event in events if event.get("type") == event_type and event.get("event_ms") is not None]


def text_events(events: list[dict]) -> list[dict]:
    texts: list[dict] = []
    for event in events:
        if event.get("type") not in {"text_output", "output_transcription"}:
            continue
        event_ms = event.get("event_ms")
        text = event.get("text")
        if event_ms is None or not text:
            continue
        texts.append({"timeMs": int(event_ms), "text": " ".join(text.split()), "label": f"T{len(texts) + 1}"})
    return texts


def close_events(events: list[dict]) -> list[dict]:
    closes: list[dict] = []
    for event in events:
        if event.get("type") == "session_closed" and event.get("event_ms") is not None:
            closes.append({"timeMs": int(event["event_ms"]), "code": event.get("code"), "reason": event.get("reason")})
    return closes


def audio_chunks(events: list[dict]) -> list[dict]:
    chunks: list[dict] = []
    for event in audio_output_events(events):
        event_ms = int(event.get("event_ms") or 0)
        byte_count = int(event.get("bytes") or 0)
        duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
        chunks.append(
            {
                "startMs": event_ms,
                "endMs": event_ms + duration_ms,
                "durationMs": duration_ms,
                "bytes": byte_count,
                "phase": event.get("phase"),
            }
        )
    return chunks


def merge_audio_segments(chunks: list[dict]) -> list[dict]:
    segments: list[dict] = []
    for chunk in sorted(chunks, key=lambda item: item["startMs"]):
        if not segments or chunk["startMs"] - segments[-1]["endMs"] > AUDIO_SEGMENT_MERGE_GAP_MS:
            segments.append(
                {
                    "startMs": round(chunk["startMs"], 3),
                    "endMs": round(chunk["endMs"], 3),
                    "durationMs": round(chunk["durationMs"], 3),
                    "bytes": chunk["bytes"],
                    "chunkCount": 1,
                    "phase": chunk.get("phase"),
                }
            )
            continue
        segments[-1]["endMs"] = round(max(segments[-1]["endMs"], chunk["endMs"]), 3)
        segments[-1]["bytes"] += chunk["bytes"]
        segments[-1]["chunkCount"] += 1
        segments[-1]["durationMs"] = round(segments[-1]["endMs"] - segments[-1]["startMs"], 3)
    return segments


def max_time_ms(events: list[dict], segments: list[dict], texts: list[dict], closes: list[dict], include_close: bool = False) -> float:
    values = [0]
    values.extend(segment["endMs"] for segment in segments)
    values.extend(text["timeMs"] for text in texts)
    if include_close:
        values.extend(close["timeMs"] for close in closes)
    for event_type in [
        "tool_call_received",
        "final_tool_response_sent",
        "client_status_tick_sent",
        "boundary_client_status_tick_sent",
        "interim_function_response_sent",
        "check_status_pending_response_sent",
        "main_pending_tool_response_sent",
        "turn_complete",
    ]:
        values.extend(event_times(events, event_type))
    return max(values) + 1200


def rounded_axis_limit_ms(value_ms: float) -> int:
    if value_ms <= 0:
        return 1000
    if value_ms <= 15_000:
        step = 1000
    elif value_ms <= 30_000:
        step = 2000
    else:
        step = 5000
    return max(step, int(((value_ms + step - 1) // step) * step))


def run_axis_limit_ms(attempt_dirs: list[Path]) -> int:
    max_ms = 1000.0
    for attempt_dir in attempt_dirs:
        events_path = attempt_dir / "timeline" / "events.jsonl"
        if not events_path.exists():
            continue
        events = read_jsonl(events_path)
        segments = merge_audio_segments(audio_chunks(events))
        max_ms = max(max_ms, max_time_ms(events, segments, text_events(events), close_events(events), include_close=False))
    return rounded_axis_limit_ms(max_ms)


def draw_attempt_timeline(attempt_dir: Path, events: list[dict], out_path: Path, x_limit_ms: int | None = None) -> dict:
    segments = merge_audio_segments(audio_chunks(events))
    texts = text_events(events)
    closes = close_events(events)
    tool_call_times = event_times(events, "tool_call_received")
    final_response_ms = first_time(events, "final_tool_response_sent")
    tick_times = [int(event["event_ms"]) for event in events if event.get("type") in TICK_TYPES and event.get("event_ms") is not None]
    turn_times = event_times(events, "turn_complete")
    limit_ms = x_limit_ms or rounded_axis_limit_ms(max_time_ms(events, segments, texts, closes, include_close=False))
    x_max = max(1, limit_ms / 1000)
    label = f"{attempt_dir.parent.name}/{attempt_dir.name}"

    fig, ax = plt.subplots(figsize=(11, 2.0))
    ax.set_title(label)
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels(["audio"])
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.hlines(0, 0, x_max, color=BASELINE_COLOR, linewidth=0.8, zorder=1)
    ax.vlines(0, -MARKER_HALF_HEIGHT, MARKER_HALF_HEIGHT, color="#333333", linewidth=1.1, zorder=3)
    ax.text(0, MARKER_HALF_HEIGHT + 0.03, "user", rotation=90, va="bottom", ha="center", fontsize=7)

    if tool_call_times and final_response_ms and final_response_ms > tool_call_times[0]:
        ax.add_patch(
            Rectangle(
                (tool_call_times[0] / 1000, -WAITING_REGION_HEIGHT / 2),
                (final_response_ms - tool_call_times[0]) / 1000,
                WAITING_REGION_HEIGHT,
                facecolor="#7B3FCE",
                edgecolor="none",
                alpha=WAITING_REGION_ALPHA,
                zorder=0,
            )
        )

    for segment in segments:
        ax.broken_barh(
            [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
            (-AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
            facecolors=AUDIO_COLOR,
            edgecolors=AUDIO_EDGE_COLOR,
            linewidth=0.6,
            zorder=2,
        )
    if not segments:
        ax.text(0.25, 0.08, "no audio", va="bottom", ha="left", color="#9E2A2B", fontsize=9)

    marker_specs = [
        ("tool call", tool_call_times, "#7B3FCE", "-"),
        ("tick/pending", tick_times, "#546E7A", ":"),
        ("final tool", [final_response_ms] if final_response_ms is not None else [], "#0A7B55", "-"),
        ("turn complete", turn_times, "#795548", "--"),
    ]
    for label_text, times, color, linestyle in marker_specs:
        for ms in times:
            x = ms / 1000
            ax.vlines(x, -MARKER_HALF_HEIGHT, MARKER_HALF_HEIGHT, color=color, linewidth=1.2, linestyle=linestyle, zorder=3)
            ax.text(x, MARKER_HALF_HEIGHT + 0.03, label_text, rotation=90, va="bottom", ha="center", color=color, fontsize=6.5)

    for close in closes:
        x = close["timeMs"] / 1000
        ax.vlines(x, -0.16, 0.16, color="#B00020", linewidth=1.4, zorder=3)
        ax.text(x, -0.2, f"close {close.get('code')}", rotation=90, va="top", ha="center", color="#B00020", fontsize=7)

    for item in texts:
        x = item["timeMs"] / 1000
        ax.scatter([x], [0], color=TEXT_COLOR, edgecolor=TEXT_EDGE_COLOR, linewidth=0.4, s=30, zorder=4)
        ax.text(x, 0.12, item["label"], ha="center", va="bottom", fontsize=6, color="#9C4A00")

    ax.set_xlim(0, x_max)
    ax.legend(
        handles=[
            Patch(facecolor=AUDIO_COLOR, edgecolor=AUDIO_EDGE_COLOR, label="assistant audio"),
            Patch(facecolor="#7B3FCE", alpha=WAITING_REGION_ALPHA, label="tool pending window"),
        ],
        loc="upper right",
        fontsize=7,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return {
        "timeline_image_file": rel(out_path),
        "tool_call_times_ms": tool_call_times,
        "tick_times_ms": tick_times,
        "final_tool_response_ms": final_response_ms,
        "turn_complete_times_ms": turn_times,
        "close_events": closes,
    }


def draw_condition_timeline(run_dir: Path, rows: list[dict], out_path: Path, x_limit_ms: int | None = None, title: str | None = None) -> None:
    rows = sorted(rows, key=lambda row: (str(row["condition"]), row["attempt_index"]))
    if not rows:
        return
    fig, ax = plt.subplots(figsize=(12, max(3.0, 0.55 * len(rows) + 1.0)))
    condition = rows[0]["condition"]
    ax.set_title(title or f"tau Live tick factor: {condition}")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks(range(len(rows)))
    multi_condition = len({row["condition"] for row in rows}) > 1
    ax.set_yticklabels(
        [
            f"{row['condition']} / attempt {row['attempt_index']:04d}" if multi_condition else f"attempt {row['attempt_index']:04d}"
            for row in rows
        ],
        fontsize=8,
    )
    ax.grid(axis="x", linestyle=":", alpha=0.3)
    max_ms = x_limit_ms or 1000
    x_max = (x_limit_ms or max_ms) / 1000

    for y, row in enumerate(rows):
        events = read_jsonl(run_dir / row["relative_attempt_dir"] / "timeline" / "events.jsonl")
        segments = merge_audio_segments(audio_chunks(events))
        texts = text_events(events)
        closes = close_events(events)
        tool_call_times = event_times(events, "tool_call_received")
        final_ms = first_time(events, "final_tool_response_sent")
        tick_times = [
            int(event["event_ms"])
            for event in events
            if event.get("type") in TICK_TYPES and event.get("event_ms") is not None
        ]
        turn_times = event_times(events, "turn_complete")

        ax.hlines(y, 0, x_max, color=BASELINE_COLOR, linewidth=0.7, zorder=1)
        ax.vlines(0, y - 0.18, y + 0.18, color="#333333", linewidth=1.0, zorder=3)

        if tool_call_times and final_ms and final_ms > tool_call_times[0]:
            ax.add_patch(
                Rectangle(
                    (tool_call_times[0] / 1000, y - WAITING_REGION_HEIGHT / 2),
                    (final_ms - tool_call_times[0]) / 1000,
                    WAITING_REGION_HEIGHT,
                    facecolor="#7B3FCE",
                    edgecolor="none",
                    alpha=WAITING_REGION_ALPHA,
                    zorder=0,
                )
            )

        for segment in segments:
            ax.broken_barh(
                [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
                (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
                facecolors=AUDIO_COLOR,
                edgecolors="none",
                zorder=2,
            )
            if x_limit_ms is None:
                max_ms = max(max_ms, segment["endMs"])

        marker_specs = [
            (tool_call_times, "#7B3FCE", "-", 1.0),
            (tick_times, "#546E7A", ":", 1.0),
            ([final_ms] if final_ms is not None else [], "#0A7B55", "-", 1.0),
            (turn_times, "#795548", "--", 0.9),
        ]
        for times, color, linestyle, linewidth in marker_specs:
            for ms in times:
                if x_limit_ms is not None and ms > x_limit_ms:
                    continue
                ax.vlines(ms / 1000, y - MARKER_HALF_HEIGHT, y + MARKER_HALF_HEIGHT, color=color, linewidth=linewidth, linestyle=linestyle, zorder=3)
                if x_limit_ms is None:
                    max_ms = max(max_ms, ms)

        for close in closes:
            ms = close["timeMs"]
            if x_limit_ms is not None and ms > x_limit_ms:
                continue
            ax.vlines(ms / 1000, y - 0.16, y + 0.16, color="#B00020", linewidth=1.0, zorder=3)
            if x_limit_ms is None:
                max_ms = max(max_ms, ms)

        for item in texts:
            ms = item["timeMs"]
            if x_limit_ms is not None and ms > x_limit_ms:
                continue
            x = ms / 1000
            ax.scatter([x], [y], color=TEXT_COLOR, edgecolor=TEXT_EDGE_COLOR, linewidth=0.35, s=22, zorder=4)
            ax.text(x, y + 0.16, item["label"], ha="center", va="bottom", fontsize=5.8, color="#9C4A00")
            if x_limit_ms is None:
                max_ms = max(max_ms, ms)

    final_axis_ms = x_limit_ms or rounded_axis_limit_ms(max_ms)
    ax.set_xlim(0, final_axis_ms / 1000)
    ax.legend(
        handles=[
            Patch(facecolor=AUDIO_COLOR, label="assistant audio"),
            Patch(facecolor=TEXT_COLOR, label="text/output transcription"),
            Patch(facecolor="#7B3FCE", alpha=WAITING_REGION_ALPHA, label="tool pending window"),
            Patch(facecolor="#7B3FCE", label="tool call"),
            Patch(facecolor="#546E7A", label="tick/pending"),
            Patch(facecolor="#0A7B55", label="final tool"),
            Patch(facecolor="#795548", label="turn complete"),
            Patch(facecolor="#B00020", label="close"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def process_attempt(attempt_dir: Path, run_dir: Path, x_limit_ms: int) -> dict:
    summary = read_json(attempt_dir / "summary.json")
    events = read_jsonl(attempt_dir / "timeline" / "events.jsonl")
    audio_dir = attempt_dir / "audio"
    existing = audio_dir / "assistant_output.wav"
    compressed = audio_dir / "assistant_output_compressed.wav"
    timeline_audio = audio_dir / "assistant_output_timeline.wav"
    if existing.exists() and not compressed.exists():
        shutil.copyfile(existing, compressed)
    audio_info = build_timeline_audio(compressed if compressed.exists() else existing, timeline_audio, events)
    image_info = draw_attempt_timeline(attempt_dir, events, attempt_dir / "attempt_timeline.png", x_limit_ms)
    enriched = {**summary, **audio_info, **image_info}
    write_json(attempt_dir / "postprocessed_summary.json", enriched)
    relative_attempt_dir = rel(attempt_dir).removeprefix(rel(run_dir) + "/")
    return {
        "condition": summary["condition"],
        "attempt_index": summary["attempt_index"],
        "relative_attempt_dir": relative_attempt_dir,
        "compressed_audio_file": audio_info["compressed_audio_file"],
        "timeline_audio_file": audio_info["timeline_audio_file"],
        "compressed_duration_ms": audio_info["compressed_duration_ms"],
        "timeline_duration_ms": audio_info["timeline_duration_ms"],
        "timeline_image_file": image_info["timeline_image_file"],
        "timeline_x_max_ms": x_limit_ms,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add compressed/timeline audio and timeline charts to a tau Live tick factor result.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    attempt_dirs = sorted(run_dir.glob("condition_*/attempt_*"))
    x_limit_ms = run_axis_limit_ms(attempt_dirs)
    rows = [process_attempt(attempt_dir, run_dir, x_limit_ms) for attempt_dir in attempt_dirs]
    condition_outputs: list[dict] = []
    for condition in sorted({row["condition"] for row in rows}):
        condition_rows = [row for row in rows if row["condition"] == condition]
        out_path = run_dir / f"condition_{condition}" / f"{condition}_attempts_timeline.png"
        draw_condition_timeline(run_dir, condition_rows, out_path, x_limit_ms)
        condition_outputs.append({"condition": condition, "attempts": len(condition_rows), "timeline_image_file": rel(out_path)})
    all_attempts_path = run_dir / "all_attempts_timeline.png"
    draw_condition_timeline(run_dir, rows, all_attempts_path, x_limit_ms, title="tau Live tick factor: all attempts")

    with (run_dir / "postprocessed_audio_timeline_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        run_dir / "postprocessed_audio_timeline_summary.json",
        {
            "run_dir": rel(run_dir),
            "attempt_count": len(rows),
            "timeline_x_max_ms": x_limit_ms,
            "all_attempts_timeline_image_file": rel(all_attempts_path),
            "condition_timeline_images": condition_outputs,
            "attempts": rows,
        },
    )
    print(f"Postprocessed attempts: {len(rows)}")
    for row in condition_outputs:
        print(f"{row['condition']} timeline: {row['timeline_image_file']}")


if __name__ == "__main__":
    main()
