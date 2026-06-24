#!/usr/bin/env python3
import argparse
import json
import unicodedata
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
RESULT_DIR = ROOT / "result"


def read_summaries(path: Path) -> list[dict]:
    summaries: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("type") == "scenario_summary":
                summaries.append(event)
    return summaries


def read_events(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def is_successful(summaries: list[dict]) -> bool:
    return (
        len(summaries) >= 4
        and all(summary.get("toolCallReceived") for summary in summaries)
        and all(summary.get("toolResponseSent") for summary in summaries)
    )


def sorted_logs() -> list[Path]:
    return sorted(
        LOG_DIR.glob("tool-bench-*.jsonl"),
        key=lambda path: path.stem.removeprefix("tool-bench-"),
        reverse=True,
    )


def latest_log() -> Path:
    candidates = sorted_logs()
    if not candidates:
        raise SystemExit(f"No tool bench logs found in {LOG_DIR}")
    return candidates[0]


def latest_successful_log() -> Path:
    candidates = sorted_logs()
    for path in candidates:
        if is_successful(read_summaries(path)):
            return path
    raise SystemExit(f"No successful tool bench log found in {LOG_DIR}")


def ms_delta(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return end - start


def fmt(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def iso_to_ms(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def raw_tool_stats_by_scenario(events: list[dict]) -> dict[str, dict]:
    stats: dict[str, dict] = {}
    for event in events:
        scenario = event.get("scenario")
        if not scenario:
            continue
        entry = stats.setdefault(
            scenario,
            {
                "primaryToolCallId": None,
                "primaryToolCallAt": None,
                "primaryToolResponseSentAt": None,
                "lastToolCallAt": None,
                "lastToolResponseSentAt": None,
                "toolCallCount": 0,
                "extraToolCallCount": 0,
                "toolResponseSentCount": 0,
            },
        )
        event_ms = iso_to_ms(event.get("ts"))
        if event.get("type") == "tool_call_received":
            for call in event.get("functionCalls", []):
                entry["toolCallCount"] += 1
                entry["lastToolCallAt"] = event_ms
                if entry["primaryToolCallId"] is None:
                    entry["primaryToolCallId"] = call.get("id")
                    entry["primaryToolCallAt"] = event_ms
                else:
                    entry["extraToolCallCount"] += 1
        elif event.get("type") == "tool_response_sent":
            entry["toolResponseSentCount"] += 1
            entry["lastToolResponseSentAt"] = event_ms
            if event.get("functionCallId") == entry.get("primaryToolCallId"):
                entry["primaryToolResponseSentAt"] = event_ms
    return stats


def first_audio_at_by_scenario(events: list[dict]) -> dict[str, int]:
    first_audio: dict[str, int] = {}
    for event in events:
        scenario = event.get("scenario")
        if not scenario or scenario in first_audio or event.get("type") != "server_event":
            continue

        parts = (
            event.get("message", {})
            .get("serverContent", {})
            .get("modelTurn", {})
            .get("parts", [])
        )
        if any(part.get("inlineData", {}).get("bytes", 0) > 0 for part in parts):
            summaries = [item for item in events if item.get("scenario") == scenario and item.get("type") == "scenario_summary"]
            opened_at = summaries[0].get("timings", {}).get("sessionOpenedAt") if summaries else None
            event_ms = event.get("eventMs")
            if opened_at is not None and event_ms is not None:
                first_audio[scenario] = opened_at + event_ms
    return first_audio


def event_has_audio_output(event: dict) -> bool:
    parts = (
        event.get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )
    return any(part.get("inlineData", {}).get("bytes", 0) > 0 for part in parts)


def event_has_text_output(event: dict) -> bool:
    parts = (
        event.get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )
    return any(part.get("text") for part in parts)


def first_text_at_by_scenario(events: list[dict]) -> dict[str, int]:
    first_text: dict[str, int] = {}
    for event in events:
        scenario = event.get("scenario")
        if not scenario or scenario in first_text or event.get("type") != "server_event":
            continue
        if not event_has_text_output(event):
            continue
        summaries = [item for item in events if item.get("scenario") == scenario and item.get("type") == "scenario_summary"]
        opened_at = summaries[0].get("timings", {}).get("sessionOpenedAt") if summaries else None
        event_ms = event.get("eventMs")
        if opened_at is not None and event_ms is not None:
            first_text[scenario] = opened_at + event_ms
    return first_text


def first_audio_after_time(events: list[dict], scenario: str, start_time: int | None) -> int | None:
    if start_time is None:
        return None
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = iso_to_ms(event.get("ts"))
        if event_ms is not None and event_ms >= start_time and event_has_audio_output(event):
            return event_ms
    return None


def extract_prompts(events: list[dict], log_path: Path) -> tuple[str, str, str]:
    for event in events:
        if event.get("type") == "prompt_metadata":
            return (
                event.get("promptVersion") or "n/a",
                event.get("systemInstruction") or "n/a",
                event.get("prompt") or "n/a",
            )

    run_stamp = log_path.stem.removeprefix("tool-bench-")
    summary_path = RESULT_DIR / run_stamp / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        prompts = summary.get("prompts", {})
        return (
            prompts.get("promptVersion") or "n/a",
            prompts.get("systemInstruction") or "n/a",
            prompts.get("prompt") or "n/a",
        )

    for event in events:
        if event.get("type") == "user_message_sent" and event.get("prompt"):
            return ("n/a", "n/a", event["prompt"])

    return ("n/a", "n/a", "n/a")


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


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze the latest successful Gemini Live tool-bench JSONL log.")
    parser.add_argument("log", nargs="?", type=Path, help="Optional path to a specific tool-bench JSONL log.")
    parser.add_argument(
        "--latest-successful",
        action="store_true",
        help="Analyze the newest log where all scenarios received tool calls and tool responses.",
    )
    args = parser.parse_args()

    log_path = (args.log.resolve() if args.log else (latest_successful_log() if args.latest_successful else latest_log()))
    events = read_events(log_path)
    summaries = [event for event in events if event.get("type") == "scenario_summary"]
    if not summaries:
        raise SystemExit(f"No scenario_summary entries found in {log_path}")

    first_audio_lookup = first_audio_at_by_scenario(events)
    first_text_lookup = first_text_at_by_scenario(events)
    raw_tool_stats = raw_tool_stats_by_scenario(events)
    rows = []
    for summary in summaries:
        scenario = summary.get("scenario", "unknown")
        raw_stats = raw_tool_stats.get(scenario, {})
        timings = summary.get("timings", {})
        session_opened_at = timings.get("sessionOpenedAt")
        start_at = timings.get("userMessageSentAt") or session_opened_at
        tool_call_at = timings.get("primaryToolCallAt") or raw_stats.get("primaryToolCallAt") or timings.get("toolCallAt")
        tool_response_sent_at = (
            timings.get("primaryToolResponseSentAt")
            or raw_stats.get("primaryToolResponseSentAt")
            or timings.get("toolResponseSentAt")
        )
        time_to_tool_call_ms = timings.get("timeToPrimaryToolCallMs")
        if time_to_tool_call_ms is None:
            time_to_tool_call_ms = ms_delta(start_at, tool_call_at)
        response_delay_ms = timings.get("primaryToolResponseDelayMs")
        if response_delay_ms is None:
            response_delay_ms = ms_delta(tool_call_at, tool_response_sent_at)
        first_audio_at = timings.get("firstAudioAt") or first_audio_lookup.get(scenario)
        first_text_at = timings.get("firstTextAt") or first_text_lookup.get(scenario)
        first_audio_after_tool_response_at = timings.get("firstAudioAfterToolResponseAt") or first_audio_after_time(
            events,
            scenario,
            tool_response_sent_at,
        )
        audio_after_tool_response_ms = timings.get("timeFromToolResponseToFirstAudioMs")
        if audio_after_tool_response_ms is None:
            audio_after_tool_response_ms = ms_delta(tool_response_sent_at, first_audio_after_tool_response_at)
        audio_before_tool_response_ms = (
            ms_delta(first_audio_at, tool_response_sent_at)
            if first_audio_at and tool_response_sent_at and first_audio_at < tool_response_sent_at
            else None
        )
        rows.append(
            [
                scenario,
                fmt(time_to_tool_call_ms),
                fmt(ms_delta(start_at, first_audio_at)),
                fmt(audio_after_tool_response_ms),
                fmt(ms_delta(start_at, first_text_at)),
                fmt(response_delay_ms),
                fmt(summary.get("toolCallCount") if summary.get("toolCallCount") is not None else raw_stats.get("toolCallCount")),
                fmt(
                    summary.get("extraToolCallCount")
                    if summary.get("extraToolCallCount") is not None
                    else raw_stats.get("extraToolCallCount")
                ),
                fmt(
                    summary.get("toolResponseSentCount")
                    if summary.get("toolResponseSentCount") is not None
                    else raw_stats.get("toolResponseSentCount")
                ),
                fmt(audio_before_tool_response_ms),
                summary.get("timelineAudioFile") or "n/a",
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
        "timelineAudioFile",
    ]
    prompt_version, system_prompt, user_prompt = extract_prompts(events, log_path)
    report = "\n".join(
        [
            f"Log: {log_path.relative_to(ROOT)}",
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
            prompt_version,
            "",
            "System prompt:",
            system_prompt,
            "",
            "User prompt:",
            user_prompt,
            "",
        ]
    )

    RESULT_DIR.mkdir(parents=True, exist_ok=True)
    result_path = RESULT_DIR / f"tool-bench-analysis-{log_path.stem.removeprefix('tool-bench-')}.txt"
    result_path.write_text(report, encoding="utf-8")

    print(report)
    print(f"Result file: {result_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
