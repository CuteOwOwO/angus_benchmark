#!/usr/bin/env python3
import argparse
import json
import math
import textwrap
import unicodedata
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.patches import Rectangle


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
LOG_DIR = ROOT / "logs"
AUDIO_SEGMENT_MERGE_GAP_MS = 200
PCM_BYTES_PER_SECOND = 48_000
COMBINED_FIGSIZE = (10, 3)
SCENARIO_FIGSIZE = (10, 1.8)
AUDIO_BAR_HEIGHT = 0.12
TEXT_MARKER_SIZE = 40
TOOL_MARKER_HALF_HEIGHT = 0.18
WAITING_REGION_HEIGHT = 0.24
WAITING_REGION_ALPHA = 0.08
BASELINE_COLOR = "#9AA0A6"
SCENARIOS = [
    "slow_correct_3s",
    "slow_correct_5s",
    "slow_correct_8s",
    "slow_correct_12s",
]


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def event_parts(event: dict) -> list[dict]:
    return (
        event.get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )


def audio_chunks(events: list[dict], scenario: str, start_ms: int) -> list[dict]:
    chunks: list[dict] = []
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = event.get("eventMs")
        if event_ms is None:
            continue
        for part in event_parts(event):
            inline = part.get("inlineData", {})
            byte_count = inline.get("bytes", 0)
            if byte_count <= 0:
                continue
            offset_ms = int(event_ms)
            duration_ms = byte_count / PCM_BYTES_PER_SECOND * 1000
            chunks.append(
                {
                    "startMs": offset_ms,
                    "endMs": offset_ms + duration_ms,
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


def text_events(events: list[dict], scenario: str) -> list[dict]:
    texts: list[dict] = []
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = event.get("eventMs")
        if event_ms is None:
            continue
        for part in event_parts(event):
            text = part.get("text")
            if text:
                texts.append(
                    {
                        "label": f"T{len(texts) + 1}",
                        "timeMs": int(event_ms),
                        "text": " ".join(text.split()),
                    }
                )
    return texts


def scenario_tool_times(summary: dict) -> tuple[int | None, int | None]:
    timings = summary.get("timings", {})
    start = timings.get("userMessageSentAt") or timings.get("sessionOpenedAt")
    call_at = timings.get("primaryToolCallAt") or timings.get("toolCallAt")
    response_at = timings.get("primaryToolResponseSentAt") or timings.get("toolResponseSentAt")
    if start is None:
        return None, None
    return (
        call_at - start if call_at is not None else None,
        response_at - start if response_at is not None else None,
    )


def scenario_start(summary: dict) -> int:
    timings = summary.get("timings", {})
    return timings.get("userMessageSentAt") or timings.get("sessionOpenedAt") or 0


def truncate(text: str, limit: int = 90) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "..."


def fmt(value: object) -> str:
    return "n/a" if value is None else str(value)


def display_width(value: str) -> int:
    width = 0
    for char in value:
        width += 2 if unicodedata.east_asian_width(char) in {"F", "W"} else 1
    return width


def pad_display(value: str, width: int) -> str:
    return value + " " * max(0, width - display_width(value))


def table_text(headers: list[str], rows: list[list[str]]) -> str:
    widths = [display_width(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = max(widths[index], display_width(cell))

    lines = [
        " | ".join(pad_display(header, widths[index]) for index, header in enumerate(headers)),
        "-+-".join("-" * width for width in widths),
    ]
    for row in rows:
        lines.append(" | ".join(pad_display(cell, widths[index]) for index, cell in enumerate(row)))
    return "\n".join(lines)


def max_time_ms(call_ms: int | None, response_ms: int | None, texts: list[dict], segments: list[dict]) -> float:
    values = [0]
    if call_ms is not None:
        values.append(call_ms)
    if response_ms is not None:
        values.append(response_ms)
    values.extend(text["timeMs"] for text in texts)
    values.extend(segment["endMs"] for segment in segments)
    return max(values) + 500


def add_waiting_region(ax, tool_call_ms: int | None, tool_response_ms: int | None, y: float) -> None:
    if tool_call_ms is None or tool_response_ms is None or tool_response_ms <= tool_call_ms:
        return
    ax.add_patch(
        Rectangle(
            (tool_call_ms / 1000, y - WAITING_REGION_HEIGHT / 2),
            (tool_response_ms - tool_call_ms) / 1000,
            WAITING_REGION_HEIGHT,
            facecolor="#7B3FCE",
            edgecolor="none",
            alpha=WAITING_REGION_ALPHA,
            zorder=0,
        )
    )


def draw_thin_audio_segments(ax, segments: list[dict], y: float) -> None:
    for segment in segments:
        ax.broken_barh(
            [(segment["startMs"] / 1000, (segment["endMs"] - segment["startMs"]) / 1000)],
            (y - AUDIO_BAR_HEIGHT / 2, AUDIO_BAR_HEIGHT),
            facecolors="#1E88E5",
            edgecolors="#0D47A1",
            linewidth=0.7,
            zorder=2,
        )


def draw_tool_markers(ax, tool_call_ms: int | None, tool_response_ms: int | None, y: float, labels: bool = False) -> None:
    if tool_call_ms is not None:
        x = tool_call_ms / 1000
        ax.vlines(x, y - TOOL_MARKER_HALF_HEIGHT, y + TOOL_MARKER_HALF_HEIGHT, color="#7B3FCE", linewidth=1.6, zorder=3)
        if labels:
            ax.text(x, y + TOOL_MARKER_HALF_HEIGHT + 0.03, "tool call", rotation=90, va="bottom", ha="center", color="#7B3FCE", fontsize=8)
    if tool_response_ms is not None:
        x = tool_response_ms / 1000
        ax.vlines(x, y - TOOL_MARKER_HALF_HEIGHT, y + TOOL_MARKER_HALF_HEIGHT, color="#0A7B55", linewidth=1.6, zorder=3)
        if labels:
            ax.text(
                x,
                y + TOOL_MARKER_HALF_HEIGHT + 0.03,
                "tool response",
                rotation=90,
                va="bottom",
                ha="center",
                color="#0A7B55",
                fontsize=8,
            )


def draw_scenario_timeline(scenario: str, summary: dict, events: list[dict], out_path: Path) -> dict:
    start_ms = scenario_start(summary)
    chunks = audio_chunks(events, scenario, start_ms)
    segments = merge_audio_segments(chunks)
    texts = text_events(events, scenario)
    tool_call_ms, tool_response_ms = scenario_tool_times(summary)
    limit_ms = max_time_ms(tool_call_ms, tool_response_ms, texts, segments)

    fig, ax = plt.subplots(figsize=SCENARIO_FIGSIZE)
    ax.set_title(f"{scenario} timeline")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.set_ylim(-0.45, 0.45)
    ax.set_yticks([0])
    ax.set_yticklabels([scenario])
    ax.grid(axis="x", linestyle=":", alpha=0.35)

    x_max = max(1, limit_ms / 1000)
    ax.hlines(0, 0, x_max, color=BASELINE_COLOR, linewidth=0.8, zorder=1)
    ax.vlines(0, -TOOL_MARKER_HALF_HEIGHT, TOOL_MARKER_HALF_HEIGHT, color="#333333", linewidth=1.1, zorder=3)
    ax.text(0, TOOL_MARKER_HALF_HEIGHT + 0.03, "user prompt", rotation=90, va="bottom", ha="center", fontsize=7)

    add_waiting_region(ax, tool_call_ms, tool_response_ms, 0)
    draw_tool_markers(ax, tool_call_ms, tool_response_ms, 0, labels=True)

    if segments:
        draw_thin_audio_segments(ax, segments, 0)
    else:
        ax.text(min(0.5, x_max * 0.75), 0.08, "no audio", va="bottom", ha="left", color="#9E2A2B", fontsize=9)
    if tool_call_ms is None:
        ax.text(x_max * 0.98, -0.08, "no tool call", va="top", ha="right", color="#7B3FCE", fontsize=8)

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
        "timelineImagePath": rel(out_path),
    }


def draw_all_scenarios(data: dict[str, dict], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=COMBINED_FIGSIZE)
    ax.set_title("All scenarios timeline")
    ax.set_xlabel("time since user prompt (seconds)")
    ax.grid(axis="x", linestyle=":", alpha=0.35)
    row_gap = 0.55
    y_positions = {scenario: (len(SCENARIOS) - 1 - index) * row_gap for index, scenario in enumerate(SCENARIOS)}
    ax.set_yticks([y_positions[scenario] for scenario in SCENARIOS])
    ax.set_yticklabels(SCENARIOS)
    ax.set_ylim(-0.35, max(y_positions.values()) + 0.35)

    max_ms = 1000
    for scenario, y in y_positions.items():
        item = data.get(scenario)
        if not item:
            continue
        call_ms = item.get("toolCallMs")
        response_ms = item.get("toolResponseMs")
        if call_ms is not None:
            max_ms = max(max_ms, call_ms)
        if response_ms is not None:
            max_ms = max(max_ms, response_ms)
        for segment in item.get("audioSegments", []):
            max_ms = max(max_ms, segment["endMs"])
        for text in item.get("textEvents", []):
            max_ms = max(max_ms, text["timeMs"])

    x_max = (max_ms + 500) / 1000
    for index, scenario in enumerate(SCENARIOS):
        item = data.get(scenario)
        if not item:
            continue
        y = y_positions[scenario]
        call_ms = item.get("toolCallMs")
        response_ms = item.get("toolResponseMs")
        ax.hlines(y, 0, x_max, color=BASELINE_COLOR, linewidth=0.75, zorder=1)
        add_waiting_region(ax, call_ms, response_ms, y)
        draw_tool_markers(ax, call_ms, response_ms, y)
        if call_ms is None:
            ax.text(x_max * 0.99, y + 0.07, "no tool call", va="bottom", ha="right", color="#7B3FCE", fontsize=7)

        draw_thin_audio_segments(ax, item.get("audioSegments", []), y)
        if not item.get("audioSegments"):
            ax.text(x_max * 0.99, y - 0.07, "no audio", va="top", ha="right", color="#9E2A2B", fontsize=7)

        for text in item.get("textEvents", []):
            ax.scatter([text["timeMs"] / 1000], [y], color="#F57C00", edgecolor="white", linewidth=0.35, s=TEXT_MARKER_SIZE, zorder=4)

    legend_handles = [
        plt.Line2D([], [], color="#7B3FCE", label="tool call"),
        plt.Line2D([], [], color="#0A7B55", label="tool response"),
        plt.Line2D([], [], marker="o", color="w", markerfacecolor="#F57C00", label="text event", markersize=5),
        Patch(facecolor="#1E88E5", edgecolor="#0D47A1", label="audio segment"),
        Patch(facecolor="#7B3FCE", alpha=WAITING_REGION_ALPHA, edgecolor="none", label="waiting region"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", ncol=5, fontsize=6.5, frameon=False)
    ax.set_xlim(-0.1, x_max)
    for spine in ["left", "right", "top"]:
        ax.spines[spine].set_visible(False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def write_text_summary(data: dict[str, dict], out_path: Path) -> None:
    blocks: list[str] = []
    for scenario in SCENARIOS:
        item = data.get(scenario)
        if not item:
            continue
        lines = [f"Scenario: {scenario}", ""]
        for text in item.get("textEvents", []):
            lines.extend([f"[{text['timeMs']} ms] {text['label']}", truncate(text["text"], 500), ""])
        blocks.append("\n".join(lines).rstrip())
    out_path.write_text(("\n\n" + "-" * 60 + "\n\n").join(blocks) + "\n", encoding="utf-8")


def write_summary_md(summary: dict, out_path: Path) -> None:
    rows = [
        "| scenario | tool call ms | tool response delay ms | first text ms | first audio ms | audio after response ms | timeline |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for scenario in summary.get("scenarios", []):
        timings = scenario.get("timings", {})
        rows.append(
            "| {scenario} | {call} | {delay} | {text} | {audio} | {after} | {timeline} |".format(
                scenario=scenario.get("scenario", "unknown"),
                call=timings.get("timeToPrimaryToolCallMs", "n/a"),
                delay=timings.get("primaryToolResponseDelayMs", "n/a"),
                text=timings.get("timeToFirstTextMs", "n/a"),
                audio=timings.get("timeToFirstAudioMs", "n/a"),
                after=timings.get("timeFromToolResponseToFirstAudioMs", "n/a"),
                timeline=scenario.get("timelineImagePath", "n/a"),
            )
        )
    out_path.write_text("\n".join(["# Tool Bench Summary", "", *rows, ""]), encoding="utf-8")


def write_summary_txt(summary: dict, out_path: Path) -> None:
    rows: list[list[str]] = []
    for scenario in summary.get("scenarios", []):
        timings = scenario.get("timings", {})
        rows.append(
            [
                scenario.get("scenario", "unknown"),
                fmt(timings.get("timeToPrimaryToolCallMs")),
                fmt(timings.get("timeToFirstAudioMs")),
                fmt(timings.get("timeFromToolResponseToFirstAudioMs")),
                fmt(timings.get("timeToFirstTextMs")),
                fmt(timings.get("primaryToolResponseDelayMs")),
                fmt(scenario.get("toolCallCount")),
                fmt(scenario.get("extraToolCallCount")),
                fmt(scenario.get("toolResponseSentCount")),
                fmt(scenario.get("audioBeforeToolResponseMs")),
                fmt(scenario.get("timelineImagePath")),
            ]
        )

    headers = [
        "scenario",
        "多久呼叫",
        "第一次輸出",
        "接收到回答",
        "第一次文字",
        "主要工具回傳延遲",
        "toolCallCount",
        "extraToolCallCount",
        "toolResponseSentCount",
        "audioBeforeToolResponseMs",
        "timelineImagePath",
    ]
    prompts = summary.get("prompts", {})
    report = "\n".join(
        [
            f"Run: {summary.get('runTimestamp') or summary.get('runStamp') or 'n/a'}",
            f"Log: {summary.get('logFile') or 'n/a'}",
            "Note: 多久呼叫 = primaryToolCallAt - userMessageSentAt",
            "Note: 第一次輸出 = firstAudioAt - userMessageSentAt; only audio output counts.",
            "Note: 接收到回答 = first audio after primaryToolResponseSentAt - primaryToolResponseSentAt",
            "Note: 第一次文字 = firstTextAt - userMessageSentAt",
            "Note: 主要工具回傳延遲 = primaryToolResponseSentAt - primaryToolCallAt",
            "Note: audioBeforeToolResponseMs is only set when first audio happened before the tool response.",
            "",
            table_text(headers, rows),
            "",
            "Prompt version:",
            fmt(summary.get("prompt_version") or prompts.get("promptVersion")),
            "",
            "System prompt:",
            fmt(prompts.get("systemInstruction")),
            "",
            "User prompt:",
            fmt(prompts.get("prompt")),
            "",
        ]
    )
    out_path.write_text(report, encoding="utf-8")


def find_log_for_run(run_dir: Path) -> Path:
    raw_log = run_dir / "raw_log.jsonl"
    if raw_log.exists():
        return raw_log
    run_stamp = run_dir.name
    log_path = LOG_DIR / f"tool-bench-{run_stamp}.jsonl"
    if log_path.exists():
        return log_path
    raise SystemExit(f"Could not find raw_log.jsonl or logs/tool-bench-{run_stamp}.jsonl")


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate static PNG timeline visualizations for a tool-bench run.")
    parser.add_argument("run_dir", type=Path, help="Run folder, e.g. result/<runTimestamp>")
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    if not run_dir.exists():
        raise SystemExit(f"Run directory does not exist: {run_dir}")

    log_path = find_log_for_run(run_dir)
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        raise SystemExit(f"Missing summary.json in {run_dir}")

    events = read_jsonl(log_path)
    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    scenarios_by_id = {scenario.get("scenario"): scenario for scenario in summary.get("scenarios", [])}
    timeline_dir = run_dir / "timeline"
    timeline_data: dict[str, dict] = {}

    for scenario in SCENARIOS:
        scenario_summary = scenarios_by_id.get(scenario)
        if not scenario_summary:
            continue
        image_path = timeline_dir / f"{scenario}_timeline.png"
        rendered = draw_scenario_timeline(scenario, scenario_summary, events, image_path)
        tool_call_ms, tool_response_ms = scenario_tool_times(scenario_summary)
        rendered["toolCallMs"] = tool_call_ms
        rendered["toolResponseMs"] = tool_response_ms
        timeline_data[scenario] = rendered
        scenario_summary["primaryToolCallAt"] = scenario_summary.get("timings", {}).get("primaryToolCallAt")
        scenario_summary["primaryToolResponseSentAt"] = scenario_summary.get("timings", {}).get("primaryToolResponseSentAt")
        scenario_summary["firstTextAt"] = scenario_summary.get("timings", {}).get("firstTextAt")
        scenario_summary["firstAudioAt"] = scenario_summary.get("timings", {}).get("firstAudioAt")
        scenario_summary["firstAudioAfterToolResponseAt"] = scenario_summary.get("timings", {}).get("firstAudioAfterToolResponseAt")
        scenario_summary["expected_final_answer"] = scenario_summary.get("expected_final_answer") or summary.get("expected_final_answer")
        scenario_summary["final_answer_exact_match"] = scenario_summary.get("final_answer_exact_match")
        scenario_summary["tool_response_schema"] = scenario_summary.get("tool_response_schema") or summary.get("tool_response_schema")
        scenario_summary["audioBeforeToolResponseMs"] = None
        first_audio_at = scenario_summary.get("timings", {}).get("firstAudioAt")
        response_at = scenario_summary.get("timings", {}).get("primaryToolResponseSentAt")
        if first_audio_at is not None and response_at is not None and first_audio_at < response_at:
            scenario_summary["audioBeforeToolResponseMs"] = response_at - first_audio_at
        scenario_summary["audioSegments"] = rendered["audioSegments"]
        scenario_summary["timelineImagePath"] = rendered["timelineImagePath"]

    all_path = timeline_dir / "all_scenarios_timeline.png"
    draw_all_scenarios(timeline_data, all_path)
    write_text_summary(timeline_data, timeline_dir / "text_event_summary.txt")

    summary["runTimestamp"] = summary.get("runTimestamp") or summary.get("runStamp") or run_dir.name
    prompts = summary.get("prompts", {})
    summary["prompt_version"] = summary.get("prompt_version") or prompts.get("promptVersion")
    summary["timeline"] = {
        "audioSegmentMergeGapMs": AUDIO_SEGMENT_MERGE_GAP_MS,
        "pcmBytesPerSecond": PCM_BYTES_PER_SECOND,
        "allScenariosTimelineImagePath": rel(all_path),
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_summary_md(summary, run_dir / "summary.md")
    write_summary_txt(summary, run_dir / "summary.txt")

    print(f"Timeline directory: {rel(timeline_dir)}")
    print(f"Combined timeline: {rel(all_path)}")


if __name__ == "__main__":
    main()
