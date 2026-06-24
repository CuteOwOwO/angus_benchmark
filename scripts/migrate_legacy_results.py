#!/usr/bin/env python3
import argparse
import json
import shutil
import subprocess
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOG_DIR = ROOT / "logs"
AUDIO_DIR = LOG_DIR / "audio"
RESULT_DIR = ROOT / "result"
VISUALIZER = ROOT / "scripts" / "visualize_timeline.py"


def read_jsonl(path: Path) -> list[dict]:
    events: list[dict] = []
    with path.open("r", encoding="utf-8") as file:
        for line in file:
            if line.strip():
                events.append(json.loads(line))
    return events


def iso_to_ms(value: str | None) -> int | None:
    if not value:
        return None
    return int(datetime.fromisoformat(value.replace("Z", "+00:00")).timestamp() * 1000)


def ms_delta(start: int | None, end: int | None) -> int | None:
    if start is None or end is None:
        return None
    return end - start


def event_parts(event: dict) -> list[dict]:
    return (
        event.get("message", {})
        .get("serverContent", {})
        .get("modelTurn", {})
        .get("parts", [])
    )


def has_audio(event: dict) -> bool:
    return any(part.get("inlineData", {}).get("bytes", 0) > 0 for part in event_parts(event))


def has_text(event: dict) -> bool:
    return any(part.get("text") for part in event_parts(event))


def first_server_event_at(events: list[dict], scenario: str, opened_at: int | None) -> int | None:
    if opened_at is None:
        return None
    for event in events:
        if event.get("scenario") == scenario and event.get("type") == "server_event" and event.get("eventMs") is not None:
            return opened_at + event["eventMs"]
    return None


def first_audio_at(events: list[dict], scenario: str, opened_at: int | None) -> int | None:
    if opened_at is None:
        return None
    for event in events:
        if event.get("scenario") == scenario and event.get("type") == "server_event" and event.get("eventMs") is not None and has_audio(event):
            return opened_at + event["eventMs"]
    return None


def first_text_at(events: list[dict], scenario: str, opened_at: int | None) -> int | None:
    if opened_at is None:
        return None
    for event in events:
        if event.get("scenario") == scenario and event.get("type") == "server_event" and event.get("eventMs") is not None and has_text(event):
            return opened_at + event["eventMs"]
    return None


def first_audio_after(events: list[dict], scenario: str, at_ms: int | None) -> int | None:
    if at_ms is None:
        return None
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = iso_to_ms(event.get("ts"))
        if event_ms is not None and event_ms >= at_ms and has_audio(event):
            return event_ms
    return None


def first_text_between(events: list[dict], scenario: str, start_ms: int | None, end_ms: int | None) -> int | None:
    if start_ms is None:
        return None
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = iso_to_ms(event.get("ts"))
        if event_ms is None or event_ms < start_ms:
            continue
        if end_ms is not None and event_ms >= end_ms:
            continue
        if has_text(event):
            return event_ms
    return None


def tool_stats(events: list[dict], scenario: str) -> dict:
    stats = {
        "primaryToolCallId": None,
        "primaryToolCallAt": None,
        "primaryToolResponseSentAt": None,
        "lastToolCallAt": None,
        "lastToolResponseSentAt": None,
        "toolCallCount": 0,
        "extraToolCallCount": 0,
        "toolResponseSentCount": 0,
    }
    for event in events:
        if event.get("scenario") != scenario:
            continue
        event_ms = iso_to_ms(event.get("ts"))
        if event.get("type") == "tool_call_received":
            for call in event.get("functionCalls", []):
                stats["toolCallCount"] += 1
                stats["lastToolCallAt"] = event_ms
                if stats["primaryToolCallId"] is None:
                    stats["primaryToolCallId"] = call.get("id")
                    stats["primaryToolCallAt"] = event_ms
                else:
                    stats["extraToolCallCount"] += 1
        elif event.get("type") == "tool_response_sent":
            stats["toolResponseSentCount"] += 1
            stats["lastToolResponseSentAt"] = event_ms
            if event.get("functionCallId") == stats["primaryToolCallId"]:
                stats["primaryToolResponseSentAt"] = event_ms
    return stats


def first_event_ts(events: list[dict], scenario: str, event_type: str) -> int | None:
    for event in events:
        if event.get("scenario") == scenario and event.get("type") == event_type:
            return iso_to_ms(event.get("ts"))
    return None


def event_types_seen(events: list[dict], scenario: str) -> list[str]:
    seen: list[str] = []
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        for event_type in event.get("eventTypes", []):
            if event_type not in seen:
                seen.append(event_type)
    return seen


def audio_bytes(events: list[dict], scenario: str, response_at: int | None) -> tuple[int, int]:
    before = 0
    after = 0
    for event in events:
        if event.get("scenario") != scenario or event.get("type") != "server_event":
            continue
        event_ms = iso_to_ms(event.get("ts"))
        total = sum(part.get("inlineData", {}).get("bytes", 0) for part in event_parts(event))
        if not total:
            continue
        if response_at is not None and event_ms is not None and event_ms >= response_at:
            after += total
        else:
            before += total
    return before, after


def prompt_metadata(events: list[dict]) -> dict:
    for event in events:
        if event.get("type") == "prompt_metadata":
            return {
                "promptVersion": event.get("promptVersion") or "n/a",
                "prompt": event.get("prompt") or "n/a",
                "systemInstruction": event.get("systemInstruction") or "n/a",
            }
    for event in events:
        if event.get("type") == "user_message_sent" and event.get("prompt"):
            return {
                "promptVersion": "n/a",
                "prompt": event["prompt"],
                "systemInstruction": "n/a",
            }
    return {
        "promptVersion": "n/a",
        "prompt": "n/a",
        "systemInstruction": "n/a",
    }


def write_prompts_file(run_dir: Path, prompts: dict) -> None:
    run_dir.joinpath("prompts.txt").write_text(
        "\n".join(
            [
                f"PROMPT_VERSION={prompts.get('promptVersion', 'n/a')}",
                "",
                "PROMPT:",
                prompts.get("prompt") or "n/a",
                "",
                "SYSTEM_INSTRUCTION:",
                prompts.get("systemInstruction") or "n/a",
                "",
            ]
        ),
        encoding="utf-8",
    )


def enrich_summary(summary: dict, events: list[dict]) -> dict:
    scenario = summary.get("scenario", "unknown")
    timings = dict(summary.get("timings", {}))
    stats = tool_stats(events, scenario)
    opened_at = timings.get("sessionOpenedAt") or first_event_ts(events, scenario, "session_opened")
    user_sent_at = timings.get("userMessageSentAt") or first_event_ts(events, scenario, "user_message_sent")
    start_at = user_sent_at or opened_at
    primary_call_at = timings.get("primaryToolCallAt") or stats["primaryToolCallAt"] or timings.get("toolCallAt")
    primary_response_at = (
        timings.get("primaryToolResponseSentAt")
        or stats["primaryToolResponseSentAt"]
        or timings.get("toolResponseSentAt")
    )
    first_audio = timings.get("firstAudioAt") or first_audio_at(events, scenario, opened_at)
    first_text = timings.get("firstTextAt") or first_text_at(events, scenario, opened_at)
    first_audio_after_response = timings.get("firstAudioAfterToolResponseAt") or first_audio_after(events, scenario, primary_response_at)
    first_text_after_call = timings.get("firstTextAfterToolCallAt") or first_text_between(events, scenario, primary_call_at, primary_response_at)
    first_output_after_response = timings.get("firstOutputAfterToolResponseAt")

    timings.update(
        {
            "sessionOpenedAt": opened_at,
            "userMessageSentAt": user_sent_at,
            "firstServerEventAt": timings.get("firstServerEventAt") or first_server_event_at(events, scenario, opened_at),
            "firstOutputAt": timings.get("firstOutputAt") or first_audio,
            "firstAudioAt": first_audio,
            "firstTextAt": first_text,
            "primaryToolCallAt": primary_call_at,
            "primaryToolResponseSentAt": primary_response_at,
            "lastToolCallAt": timings.get("lastToolCallAt") or stats["lastToolCallAt"],
            "lastToolResponseSentAt": timings.get("lastToolResponseSentAt") or stats["lastToolResponseSentAt"],
            "toolCallAt": primary_call_at,
            "toolResponseSentAt": primary_response_at,
            "timeToPrimaryToolCallMs": timings.get("timeToPrimaryToolCallMs") or ms_delta(start_at, primary_call_at),
            "primaryToolResponseDelayMs": timings.get("primaryToolResponseDelayMs") or ms_delta(primary_call_at, primary_response_at),
            "timeToPrimaryToolResponseMs": timings.get("timeToPrimaryToolResponseMs") or ms_delta(start_at, primary_response_at),
            "timeToFirstOutputMs": timings.get("timeToFirstOutputMs") or ms_delta(start_at, first_audio),
            "timeToFirstAudioMs": timings.get("timeToFirstAudioMs") or ms_delta(start_at, first_audio),
            "timeToFirstTextMs": timings.get("timeToFirstTextMs") or ms_delta(start_at, first_text),
            "timeFromToolResponseToFirstAudioMs": timings.get("timeFromToolResponseToFirstAudioMs") or ms_delta(primary_response_at, first_audio_after_response),
            "primaryToolCallToFirstAudioMs": timings.get("primaryToolCallToFirstAudioMs") or ms_delta(primary_call_at, first_audio),
            "firstAudioAfterToolResponseAt": first_audio_after_response,
            "firstTextAfterToolCallAt": first_text_after_call,
            "firstOutputAfterToolResponseAt": first_output_after_response,
        }
    )
    before_bytes, after_bytes = audio_bytes(events, scenario, primary_response_at)
    enriched = dict(summary)
    enriched["scenario"] = scenario
    enriched["timings"] = timings
    enriched["primaryToolCallId"] = enriched.get("primaryToolCallId") or stats["primaryToolCallId"]
    enriched["toolCallCount"] = enriched.get("toolCallCount", stats["toolCallCount"])
    enriched["extraToolCallCount"] = enriched.get("extraToolCallCount", stats["extraToolCallCount"])
    enriched["toolResponseSentCount"] = enriched.get("toolResponseSentCount", stats["toolResponseSentCount"])
    enriched["eventTypesSeen"] = enriched.get("eventTypesSeen") or event_types_seen(events, scenario)
    enriched["audioBytesBeforeToolResponse"] = enriched.get("audioBytesBeforeToolResponse", before_bytes)
    enriched["audioBytesAfterToolResponse"] = enriched.get("audioBytesAfterToolResponse", after_bytes)
    return enriched


def build_summary(log_path: Path, events: list[dict]) -> dict:
    run_stamp = log_path.stem.removeprefix("tool-bench-")
    summaries = [event for event in events if event.get("type") == "scenario_summary"]
    prompts = prompt_metadata(events)
    model = "n/a"
    for event in events:
        if event.get("type") == "session_opened" and event.get("model"):
            model = event["model"]
            break
    return {
        "runTimestamp": run_stamp,
        "runStamp": run_stamp,
        "prompt_version": prompts.get("promptVersion"),
        "model": model,
        "logFile": str(log_path.relative_to(ROOT)),
        "rawLogFile": "raw_log.jsonl",
        "prompts": prompts,
        "scenarios": [enrich_summary(summary, events) for summary in summaries],
    }


def migrate_log(log_path: Path, overwrite: bool) -> Path:
    run_stamp = log_path.stem.removeprefix("tool-bench-")
    run_dir = RESULT_DIR / run_stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    raw_log = run_dir / "raw_log.jsonl"
    if overwrite or not raw_log.exists():
        shutil.copy2(log_path, raw_log)

    legacy_analysis = RESULT_DIR / f"tool-bench-analysis-{run_stamp}.txt"
    if legacy_analysis.exists():
        legacy_target = run_dir / "legacy-analysis.txt"
        if overwrite or not legacy_target.exists():
            shutil.copy2(legacy_analysis, legacy_target)

    audio_source = AUDIO_DIR / run_stamp
    if audio_source.exists():
        audio_target = run_dir / "audio"
        if audio_target.exists() and overwrite:
            shutil.rmtree(audio_target)
        if overwrite or not audio_target.exists():
            shutil.copytree(audio_source, audio_target, dirs_exist_ok=True)

    events = read_jsonl(log_path)
    summary = build_summary(log_path, events)
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_prompts_file(run_dir, summary["prompts"])

    subprocess.run(["python3", str(VISUALIZER), str(run_dir)], cwd=ROOT, check=True)
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser(description="Migrate legacy tool-bench outputs into result/<runTimestamp>/ folders.")
    parser.add_argument("logs", nargs="*", type=Path, help="Optional specific JSONL logs to migrate.")
    parser.add_argument("--overwrite", action="store_true", help="Overwrite generated files in existing run folders.")
    args = parser.parse_args()

    logs = args.logs or sorted(LOG_DIR.glob("tool-bench-*.jsonl"))
    migrated: list[Path] = []
    for log in logs:
        log_path = log.resolve()
        if not log_path.exists():
            print(f"skip missing log: {log}")
            continue
        try:
            run_dir = migrate_log(log_path, args.overwrite)
            migrated.append(run_dir)
            print(f"migrated {log_path.name} -> {run_dir.relative_to(ROOT)}")
        except Exception as exc:
            print(f"failed {log_path.name}: {type(exc).__name__}: {exc}")
            raise

    print(f"Done. Migrated {len(migrated)} run folders.")


if __name__ == "__main__":
    main()
