#!/usr/bin/env python3
import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle


ROOT = Path(__file__).resolve().parents[1]
AUDIO_SEGMENT_MERGE_GAP_MS = 200
PCM_BYTES_PER_SECOND = 48_000
COMBINED_FIGSIZE = (10, 3)
SCENARIO_FIGSIZE = (10, 1.8)
AUDIO_BAR_HEIGHT = 0.12
TEXT_MARKER_SIZE = 40
EXTERNAL_MARKER_HALF_HEIGHT = 0.18
WAITING_REGION_HEIGHT = 0.24
WAITING_REGION_ALPHA = 0.08
BASELINE_COLOR = "#9AA0A6"
SCENARIOS = [
    "slow_correct_3s",
    "slow_correct_5s",
    "slow_correct_8s",
    "slow_correct_12s",
]


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def event_parts(event: dict) -> list[dict]:
    return (
        event.get("data", {})
        .get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )


def audio_chunks(events: list[dict], scenario: str) -> list[dict]:
    chunks: list[dict] = []
    for event in events:
        if event.get("scenario_id") != scenario or event.get("event_type") != "server_event":
            continue
        event_ms = event.get("relative_time_ms")
        if event_ms is None:
            continue
        for part in event_parts(event):
            inline = part.get("inlineData", {})
            byte_count = inline.get("bytes", 0)
            if byte_count <= 0:
                continue
            duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
            chunks.append(
                {
                    "startMs": int(event_ms),
                    "endMs": int(event_ms) + duration_ms,
                    "durationMs": duration_ms,
                    "bytes": byte_count,
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
                }
            )
            continue
        segments[-1]["endMs"] = round(max(segments[-1]["endMs"], chunk["endMs"]), 3)
        segments[-1]["bytes"] += chunk["bytes"]
        segments[-1]["chunkCount"] += 1
        segments[-1]["durationMs"] = round(segments[-1]["endMs"] - segments[-1]["startMs"], 3)
    return segments


def truncate(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "..."


def text_events(events: list[dict], scenario: str) -> list[dict]:
    texts: list[dict] = []
    for event in events:
        if event.get("scenario_id") != scenario:
            continue
        event_ms = event.get("relative_time_ms")
        if event_ms is None:
            continue
        if event.get("event_type") == "output_transcription":
            text = event.get("data", {}).get("text")
            if text:
                texts.append({"label": f"T{len(texts) + 1}", "timeMs": int(event_ms), "text": " ".join(text.split())})
        if event.get("event_type") != "server_event":
            continue
        for part in event_parts(event):
            text = part.get("text")
            if text:
                texts.append({"label": f"T{len(texts) + 1}", "timeMs": int(event_ms), "text": " ".join(text.split())})
    return texts


def status_ticks(events: list[dict], scenario: str) -> list[int]:
    ticks: list[int] = []
    for event in events:
        if event.get("scenario_id") != scenario:
            continue
        if event.get("event_type") != "external_status_sent":
            continue
        event_ms = event.get("relative_time_ms")
        if event_ms is not None:
            ticks.append(int(event_ms))
    return ticks


def close_events(events: list[dict], scenario: str) -> list[dict]:
    closes: list[dict] = []
    for event in events:
        if event.get("scenario_id") != scenario:
            continue
        if event.get("event_type") not in {"session_closed", "socket_error"}:
            continue
        event_ms = event.get("relative_time_ms")
        if event_ms is not None:
            closes.append({"timeMs": int(event_ms), "eventType": event.get("event_type"), "data": event.get("data", {})})
    return closes


def max_time_ms(external_ms: int | None, texts: list[dict], segments: list[dict], ticks: list[int], closes: list[dict]) -> float:
    values = [0]
    if external_ms is not None:
        values.append(external_ms)
    values.extend(text["timeMs"] for text in texts)
    values.extend(segment["endMs"] for segment in segments)
    values.extend(ticks)
    values.extend(close["timeMs"] for close in closes)
    return max(values) + 500


def add_waiting_region(ax, external_ms: int | None, y: float) -> None:
    if external_ms is None or external_ms <= 0:
        return
    ax.add_patch(
        Rectangle(
            (0, y - WAITING_REGION_HEIGHT / 2),
            external_ms / 1000,
            WAITING_REGION_HEIGHT,
            facecolor="#7B3FCE",
            edgecolor="none",
            alpha=WAITING_REGION_ALPHA,
            zorder=0,
        )
    )


def draw_audio_segments(ax, segments: list[dict], y: float) -> None:
    for segment in segments:
        ax.broken_barh(
            [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
            (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
            facecolors="#1E88E5",
            edgecolors="#0D47A1",
            linewidth=0.7,
            zorder=2,
        )


def draw_external_marker(ax, external_ms: int | None, y: float, labels: bool = False) -> None:
    if external_ms is None:
        return
    x = external_ms / 1000
    ax.vlines(x, y - EXTERNAL_MARKER_HALF_HEIGHT, y + EXTERNAL_MARKER_HALF_HEIGHT, color="#0A7B55", linewidth=1.8, zorder=3)
    if labels:
        ax.text(
            x,
            y + EXTERNAL_MARKER_HALF_HEIGHT + 0.03,
            "external result injected",
            rotation=90,
            va="bottom",
            ha="center",
            color="#0A7B55",
            fontsize=8,
        )


def draw_status_ticks(ax, ticks: list[int], y: float) -> None:
    for tick_ms in ticks:
        x = tick_ms / 1000
        ax.vlines(x, y - 0.11, y + 0.11, color="#546E7A", linewidth=1.1, linestyle=":", zorder=3)


def draw_close_markers(ax, closes: list[dict], y: float, labels: bool = False) -> None:
    for close in closes:
        x = close["timeMs"] / 1000
        ax.vlines(x, y - 0.16, y + 0.16, color="#B00020", linewidth=1.4, zorder=3)
        if labels:
            code = close.get("data", {}).get("code")
            label = f"close {code}" if code else close.get("eventType", "close")
            ax.text(x, y - 0.2, label, rotation=90, va="top", ha="center", color="#B00020", fontsize=7)


def draw_scenario_timeline(scenario: str, summary: dict, events: list[dict], out_path: Path) -> dict:
    chunks = audio_chunks(events, scenario)
    segments = merge_audio_segments(chunks)
    texts = text_events(events, scenario)
    ticks = status_ticks(events, scenario)
    closes = close_events(events, scenario)
    external_ms = summary.get("external_result_injected_time_ms")
    limit_ms = max_time_ms(external_ms, texts, segments, ticks, closes)
    x_max = max(1, limit_ms / 1000)

    fig, ax = plt.subplots(figsize=SCENARIO_FIGSIZE)
    ax.set_title(f"{scenario} external wait timeline")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels([scenario])
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    ax.hlines(0, 0, x_max, color=BASELINE_COLOR, linewidth=0.8, zorder=1)
    ax.vlines(0, -EXTERNAL_MARKER_HALF_HEIGHT, EXTERNAL_MARKER_HALF_HEIGHT, color="#333333", linewidth=1.1, zorder=3)
    ax.text(0, EXTERNAL_MARKER_HALF_HEIGHT + 0.03, "user prompt", rotation=90, va="bottom", ha="center", fontsize=7)
    add_waiting_region(ax, external_ms, 0)
    draw_status_ticks(ax, ticks, 0)
    draw_external_marker(ax, external_ms, 0, labels=True)
    draw_close_markers(ax, closes, 0, labels=True)

    if segments:
        draw_audio_segments(ax, segments, 0)
    else:
        ax.text(min(0.5, x_max * 0.75), 0.08, "no audio", va="bottom", ha="left", color="#9E2A2B", fontsize=9)

    for item in texts:
        x = item["timeMs"] / 1000
        ax.scatter([x], [0], color="#F57C00", edgecolor="white", linewidth=0.4, s=TEXT_MARKER_SIZE, zorder=4)
        ax.text(x, 0.12, item["label"], ha="center", va="bottom", fontsize=7, color="#9C4A00")

    text_lines = [f"{item['label']} @ {item['timeMs']} ms: {truncate(item['text'])}" for item in texts[:8]]
    if len(texts) > 8:
        text_lines.append(f"... {len(texts) - 8} more text events")
    if text_lines:
        fig.text(0.08, 0.01, "\n".join(textwrap.wrap("\n".join(text_lines), 150)), fontsize=6.5, va="bottom")
        fig.subplots_adjust(bottom=0.32)
    else:
        fig.subplots_adjust(bottom=0.22)

    ax.set_xlim(-0.1, x_max)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return {
        "audioSegments": segments,
        "textEvents": texts,
        "statusTicks": ticks,
        "closeEvents": closes,
        "timelineImagePath": rel(out_path),
        "externalResultInjectedMs": external_ms,
    }


def draw_all_scenarios(data: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    ax.set_title("Controlled external result timelines")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    row_gap = 0.55
    y_positions = {scenario: (len(SCENARIOS) - 1 - index) * row_gap for index, scenario in enumerate(SCENARIOS)}
    ax.set_yticks([y_positions[scenario] for scenario in SCENARIOS])
    ax.set_yticklabels(SCENARIOS)
    ax.set_ylim(-0.35, max(y_positions.values()) + 0.35)

    max_ms = 1000
    for item in data.values():
        if item.get("externalResultInjectedMs") is not None:
            max_ms = max(max_ms, item["externalResultInjectedMs"])
        for segment in item.get("audioSegments", []):
            max_ms = max(max_ms, segment["endMs"])
        for text in item.get("textEvents", []):
            max_ms = max(max_ms, text["timeMs"])
        for tick_ms in item.get("statusTicks", []):
            max_ms = max(max_ms, tick_ms)
        for close in item.get("closeEvents", []):
            max_ms = max(max_ms, close["timeMs"])
    x_max = (max_ms + 500) / 1000

    for scenario in SCENARIOS:
        item = data.get(scenario)
        if not item:
            continue
        y = y_positions[scenario]
        external_ms = item.get("externalResultInjectedMs")
        ax.hlines(y, 0, x_max, color=BASELINE_COLOR, linewidth=0.75, zorder=1)
        add_waiting_region(ax, external_ms, y)
        draw_status_ticks(ax, item.get("statusTicks", []), y)
        draw_external_marker(ax, external_ms, y)
        draw_close_markers(ax, item.get("closeEvents", []), y)
        draw_audio_segments(ax, item.get("audioSegments", []), y)
        if not item.get("audioSegments"):
            ax.text(x_max * 0.99, y - 0.07, "no audio", va="top", ha="right", color="#9E2A2B", fontsize=7)
        for text in item.get("textEvents", []):
            ax.scatter([text["timeMs"] / 1000], [y], color="#F57C00", edgecolor="white", linewidth=0.35, s=TEXT_MARKER_SIZE, zorder=4)

    legend_handles = [
        plt.Line2D([], [], color="#546E7A", linestyle=":", label="pending status tick"),
        plt.Line2D([], [], color="#0A7B55", label="external result injected"),
        plt.Line2D([], [], color="#B00020", label="close/error"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F57C00", label="text event", markersize=5),
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio segment"),
        Patch(facecolor="#7B3FCE", alpha=WAITING_REGION_ALPHA, edgecolor="none", label="waiting region"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=3, fontsize=6.5, frameon=False)
    ax.set_xlim(-0.1, x_max)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate PNG timelines for controlled external-result runs.")
    parser.add_argument("run_dir", type=Path)
    args = parser.parse_args()
    run_dir = args.run_dir.resolve()
    summary_path = run_dir / "summary.json"
    log_path = run_dir / "raw_log.jsonl"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json: {summary_path}")
    if not log_path.exists():
        raise SystemExit(f"Missing raw_log.jsonl: {log_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    events = read_jsonl(log_path)
    scenarios_by_id = {scenario.get("scenario_id") or scenario.get("scenario"): scenario for scenario in summary.get("scenarios", [])}
    timeline_dir = run_dir / "timeline"
    timeline_data: dict[str, dict] = {}

    for scenario_id in SCENARIOS:
        scenario_summary = scenarios_by_id.get(scenario_id)
        if not scenario_summary:
            continue
        rendered = draw_scenario_timeline(scenario_id, scenario_summary, events, timeline_dir / f"{scenario_id}_timeline.png")
        scenario_summary["audioSegments"] = rendered["audioSegments"]
        scenario_summary["textEvents"] = rendered["textEvents"]
        scenario_summary["statusTicks"] = rendered["statusTicks"]
        scenario_summary["closeEvents"] = rendered["closeEvents"]
        scenario_summary["timelineImagePath"] = rendered["timelineImagePath"]
        timeline_data[scenario_id] = rendered

    all_path = timeline_dir / "all_scenarios_timeline.png"
    draw_all_scenarios(timeline_data, all_path)
    summary["timeline"] = {
        "audioSegmentMergeGapMs": AUDIO_SEGMENT_MERGE_GAP_MS,
        "pcmBytesPerSecond": PCM_BYTES_PER_SECOND,
        "allScenariosTimelineImagePath": rel(all_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Timeline directory: {rel(timeline_dir)}")
    print(f"Combined timeline: {rel(all_path)}")


if __name__ == "__main__":
    main()
