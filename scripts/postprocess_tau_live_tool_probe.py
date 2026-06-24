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
AUDIO_SEGMENT_MERGE_GAP_MS = 200
AUDIO_BAR_HEIGHT = 0.12
TOOL_MARKER_HALF_HEIGHT = 0.18
WAITING_REGION_HEIGHT = 0.24
WAITING_REGION_ALPHA = 0.08
BASELINE_COLOR = "#9AA0A6"
PCM_BYTES_PER_SECOND = 48_000


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
                }
            )
            continue
        segments[-1]["endMs"] = round(max(segments[-1]["endMs"], chunk["endMs"]), 3)
        segments[-1]["bytes"] += chunk["bytes"]
        segments[-1]["chunkCount"] += 1
        segments[-1]["durationMs"] = round(segments[-1]["endMs"] - segments[-1]["startMs"], 3)
    return segments


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


def first_time(events: list[dict], event_type: str) -> int | None:
    for event in events:
        if event.get("type") == event_type and event.get("event_ms") is not None:
            return int(event["event_ms"])
    return None


def close_events(events: list[dict]) -> list[dict]:
    closes: list[dict] = []
    for event in events:
        if event.get("type") != "session_closed" or event.get("event_ms") is None:
            continue
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


def max_time_ms(events: list[dict], segments: list[dict], texts: list[dict], closes: list[dict]) -> float:
    values = [0]
    values.extend(segment["endMs"] for segment in segments)
    values.extend(text["timeMs"] for text in texts)
    values.extend(close["timeMs"] for close in closes)
    for event_type in ["tool_call_received", "tool_response_sent", "post_tool_wait_elapsed"]:
        value = first_time(events, event_type)
        if value is not None:
            values.append(value)
    return max(values) + 500


def draw_attempt_timeline(attempt_dir: Path, events: list[dict], out_path: Path) -> dict:
    segments = merge_audio_segments(audio_chunks(events))
    texts = text_events(events)
    closes = close_events(events)
    tool_call_ms = first_time(events, "tool_call_received")
    tool_response_ms = first_time(events, "tool_response_sent")
    post_wait_ms = first_time(events, "post_tool_wait_elapsed")
    limit_ms = max_time_ms(events, segments, texts, closes)
    x_max = max(1, limit_ms / 1000)

    label = f"{attempt_dir.parent.name}/{attempt_dir.name}"
    fig, ax = plt.subplots(figsize=(10, 1.9))
    ax.set_title(label)
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels(["audio"])
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.hlines(0, 0, x_max, color=BASELINE_COLOR, linewidth=0.8, zorder=1)
    ax.vlines(0, -TOOL_MARKER_HALF_HEIGHT, TOOL_MARKER_HALF_HEIGHT, color="#333333", linewidth=1.1, zorder=3)
    ax.text(0, TOOL_MARKER_HALF_HEIGHT + 0.03, "user", rotation=90, va="bottom", ha="center", fontsize=7)

    if tool_call_ms is not None and tool_response_ms is not None and tool_response_ms > tool_call_ms:
        ax.add_patch(
            Rectangle(
                (tool_call_ms / 1000, -WAITING_REGION_HEIGHT / 2),
                (tool_response_ms - tool_call_ms) / 1000,
                WAITING_REGION_HEIGHT,
                facecolor="#7B3FCE",
                edgecolor="none",
                alpha=WAITING_REGION_ALPHA,
                zorder=0,
            )
        )

    for segment in segments:
        color = "#1E88E5" if segment.get("phase") != "after_tool_response" else "#00897B"
        edge = "#0D47A1" if segment.get("phase") != "after_tool_response" else "#00574B"
        ax.broken_barh(
            [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
            (-AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
            facecolors=color,
            edgecolors=edge,
            linewidth=0.7,
            zorder=2,
        )
    if not segments:
        ax.text(0.25, 0.08, "no audio", va="bottom", ha="left", color="#9E2A2B", fontsize=9)

    for ms, label_text, color in [
        (tool_call_ms, "tool call", "#7B3FCE"),
        (tool_response_ms, "tool response", "#0A7B55"),
        (post_wait_ms, "post wait end", "#795548"),
    ]:
        if ms is None:
            continue
        x = ms / 1000
        ax.vlines(x, -TOOL_MARKER_HALF_HEIGHT, TOOL_MARKER_HALF_HEIGHT, color=color, linewidth=1.6, zorder=3)
        ax.text(x, TOOL_MARKER_HALF_HEIGHT + 0.03, label_text, rotation=90, va="bottom", ha="center", color=color, fontsize=7)

    for close in closes:
        x = close["timeMs"] / 1000
        ax.vlines(x, -0.16, 0.16, color="#B00020", linewidth=1.4, zorder=3)
        ax.text(x, -0.2, f"close {close.get('code')}", rotation=90, va="top", ha="center", color="#B00020", fontsize=7)

    for item in texts:
        x = item["timeMs"] / 1000
        ax.scatter([x], [0], color="#F57C00", edgecolor="white", linewidth=0.4, s=36, zorder=4)
        ax.text(x, 0.12, item["label"], ha="center", va="bottom", fontsize=6.5, color="#9C4A00")

    ax.set_xlim(0, x_max)
    ax.legend(
        handles=[
            Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="waiting/pre-tool audio"),
            Patch(facecolor="#00897B", edgecolor="#00574B", label="post-tool audio"),
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
        "tool_call_ms": tool_call_ms,
        "tool_response_ms": tool_response_ms,
        "text_event_count": len(texts),
        "audio_segment_count": len(segments),
        "close_events": closes,
    }


def draw_combined(run_dir: Path, attempt_rows: list[dict], out_path: Path, title: str) -> None:
    rows = sorted(attempt_rows, key=lambda row: (row["latency_ms"], row["attempt_index"]))
    if not rows:
        return
    fig_height = max(4, 0.36 * len(rows) + 1.2)
    fig, ax = plt.subplots(figsize=(12, fig_height))
    ax.set_title(title)
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.8, len(rows) - 0.2)
    ax.set_yticks(range(len(rows)))
    ax.set_yticklabels([f"{row['latency_ms']}ms / {row['attempt_index']:04d}" for row in rows], fontsize=7)
    ax.grid(axis="x", linestyle=":", alpha=0.3)

    max_ms = 1000
    for y, row in enumerate(rows):
        events = read_jsonl(run_dir / row["relative_attempt_dir"] / "timeline" / "events.jsonl")
        segments = merge_audio_segments(audio_chunks(events))
        for segment in segments:
            color = "#1E88E5" if segment.get("phase") != "after_tool_response" else "#00897B"
            ax.broken_barh(
                [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
                (y - 0.08, 0.16),
                facecolors=color,
                edgecolors="none",
                zorder=2,
            )
            max_ms = max(max_ms, segment["endMs"])
        call_ms = first_time(events, "tool_call_received")
        response_ms = first_time(events, "tool_response_sent")
        if call_ms is not None:
            ax.vlines(call_ms / 1000, y - 0.18, y + 0.18, color="#7B3FCE", linewidth=1.0)
            max_ms = max(max_ms, call_ms)
        if response_ms is not None:
            ax.vlines(response_ms / 1000, y - 0.18, y + 0.18, color="#0A7B55", linewidth=1.0)
            max_ms = max(max_ms, response_ms)
        for close in close_events(events):
            ax.vlines(close["timeMs"] / 1000, y - 0.16, y + 0.16, color="#B00020", linewidth=1.0)
            max_ms = max(max_ms, close["timeMs"])

    ax.set_xlim(0, max_ms / 1000 + 0.5)
    ax.legend(
        handles=[
            Patch(facecolor="#1E88E5", label="waiting/pre-tool audio"),
            Patch(facecolor="#00897B", label="post-tool audio"),
            Patch(facecolor="#7B3FCE", label="tool call"),
            Patch(facecolor="#0A7B55", label="tool response"),
            Patch(facecolor="#B00020", label="close"),
        ],
        loc="upper right",
        fontsize=8,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=170)
    plt.close(fig)


def draw_latency_timelines(run_dir: Path, rows: list[dict]) -> list[dict]:
    outputs: list[dict] = []
    latencies = sorted({row["latency_ms"] for row in rows})
    for latency_ms in latencies:
        latency_rows = [row for row in rows if row["latency_ms"] == latency_ms]
        out_path = run_dir / f"latency_{latency_ms}ms" / "timeline" / f"latency_{latency_ms}ms_timeline.png"
        draw_combined(
            run_dir,
            latency_rows,
            out_path,
            f"tau Live native tool probe: latency {latency_ms}ms",
        )
        outputs.append(
            {
                "latency_ms": latency_ms,
                "attempts": len(latency_rows),
                "timeline_image_file": rel(out_path),
            }
        )
    return outputs


def process_attempt(attempt_dir: Path, run_dir: Path) -> dict:
    summary = read_json(attempt_dir / "summary.json")
    events = read_jsonl(attempt_dir / "timeline" / "events.jsonl")
    audio_dir = attempt_dir / "audio"
    existing = audio_dir / "assistant_output.wav"
    compressed = audio_dir / "assistant_output_compressed.wav"
    timeline_audio = audio_dir / "assistant_output_timeline.wav"
    if existing.exists() and not compressed.exists():
        shutil.copyfile(existing, compressed)

    audio_info = build_timeline_audio(compressed if compressed.exists() else existing, timeline_audio, events)
    image_info = draw_attempt_timeline(attempt_dir, events, attempt_dir / "timeline" / "timeline.png")
    enriched = {**summary, **audio_info, **image_info}
    write_json(attempt_dir / "postprocessed_summary.json", enriched)
    return {
        "latency_ms": summary["latency_ms"],
        "attempt_index": summary["attempt_index"],
        "relative_attempt_dir": rel(attempt_dir).removeprefix(rel(run_dir) + "/"),
        "compressed_audio_file": audio_info["compressed_audio_file"],
        "timeline_audio_file": audio_info["timeline_audio_file"],
        "compressed_duration_ms": audio_info["compressed_duration_ms"],
        "timeline_duration_ms": audio_info["timeline_duration_ms"],
        "timeline_image_file": image_info["timeline_image_file"],
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Add compressed/timeline audio names and timeline plots to a tau Live tool probe result.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()

    attempt_dirs = sorted(run_dir.glob("latency_*ms/attempt_*"))
    rows = [process_attempt(attempt_dir, run_dir) for attempt_dir in attempt_dirs]
    draw_combined(run_dir, rows, run_dir / "timeline" / "all_attempts_timeline.png", "tau Live native tool probe timelines")
    latency_timeline_rows = draw_latency_timelines(run_dir, rows)

    with (run_dir / "postprocessed_audio_timeline_summary.csv").open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0].keys()) if rows else [])
        writer.writeheader()
        writer.writerows(rows)

    write_json(
        run_dir / "postprocessed_audio_timeline_summary.json",
        {
            "run_dir": rel(run_dir),
            "attempt_count": len(rows),
            "combined_timeline_image_file": rel(run_dir / "timeline" / "all_attempts_timeline.png"),
            "latency_timeline_images": latency_timeline_rows,
            "attempts": rows,
        },
    )
    print(f"Postprocessed attempts: {len(rows)}")
    print(f"Combined timeline: {rel(run_dir / 'timeline' / 'all_attempts_timeline.png')}")
    for row in latency_timeline_rows:
        print(f"Latency {row['latency_ms']}ms timeline: {row['timeline_image_file']}")


if __name__ == "__main__":
    main()
