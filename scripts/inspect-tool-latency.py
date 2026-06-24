#!/usr/bin/env python3
import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def parse_ts(value: str) -> int:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    return int(dt.astimezone(timezone.utc).timestamp() * 1000)


def fmt(value: int | None) -> str:
    return "n/a" if value is None else str(value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect per-tool-call latency for one scenario in a tool-bench JSONL log.")
    parser.add_argument("log", type=Path)
    parser.add_argument("scenario", nargs="?", default="slow_correct_12s")
    args = parser.parse_args()

    log_path = args.log
    if not log_path.is_absolute():
        log_path = ROOT / log_path

    tool_calls: dict[str, dict] = {}
    responses: dict[str, dict] = {}
    skipped_responses: dict[str, dict] = {}
    summary = None

    with log_path.open("r", encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            event = json.loads(line)
            if event.get("scenario") != args.scenario:
                continue

            if event.get("type") == "tool_call_received":
                for call in event.get("functionCalls", []):
                    call_id = call.get("id")
                    if call_id:
                        tool_calls[call_id] = {
                            "ts": event["ts"],
                            "tsMs": parse_ts(event["ts"]),
                            "name": call.get("name"),
                            "args": call.get("args"),
                        }

            if event.get("type") == "tool_response_sent":
                call_id = event.get("functionCallId")
                if call_id:
                    responses[call_id] = {
                        "ts": event["ts"],
                        "tsMs": parse_ts(event["ts"]),
                        "delayMs": event.get("delayMs"),
                        "response": event.get("response"),
                    }

            if event.get("type") == "tool_response_skipped":
                call_id = event.get("functionCallId")
                if call_id:
                    skipped_responses[call_id] = {
                        "ts": event["ts"],
                        "tsMs": parse_ts(event["ts"]),
                        "delayMs": event.get("delayMs"),
                        "reason": event.get("reason"),
                    }

            if event.get("type") == "scenario_summary":
                summary = event

    print(f"Log: {log_path.relative_to(ROOT)}")
    print(f"Scenario: {args.scenario}")
    print(f"Tool calls received: {len(tool_calls)}")
    print(f"Tool responses sent: {len(responses)}")
    print("")

    headers = ["#", "callId", "callTs", "delayMs", "expectedResponseTs", "actualResponseTs", "status", "latenessMs", "args"]
    rows = []
    scenario_delay = summary.get("delayMs") if summary else None
    for index, (call_id, call) in enumerate(tool_calls.items(), start=1):
        response = responses.get(call_id)
        skipped = skipped_responses.get(call_id)
        delay = (
            response.get("delayMs")
            if response
            else skipped.get("delayMs")
            if skipped
            else scenario_delay
            if scenario_delay is not None
            else 0
        )
        expected = call["tsMs"] + delay
        actual = response.get("tsMs") if response else None
        status = "sent" if response else f"skipped: {skipped.get('reason')}" if skipped else "pending/no response"
        rows.append(
            [
                str(index),
                call_id,
                call["ts"],
                fmt(delay),
                datetime.fromtimestamp(expected / 1000, tz=timezone.utc).isoformat().replace("+00:00", "Z"),
                response["ts"] if response else "n/a",
                status,
                fmt(actual - expected if actual is not None else None),
                json.dumps(call.get("args"), ensure_ascii=False),
            ]
        )

    widths = [len(header) for header in headers]
    for row in rows:
        for index, cell in enumerate(row):
            widths[index] = min(max(widths[index], len(cell)), 60)

    print(" | ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("-+-".join("-" * width for width in widths))
    for row in rows:
        print(" | ".join(cell[: widths[index]].ljust(widths[index]) for index, cell in enumerate(row)))

    if summary:
        timings = summary.get("timings", {})
        print("")
        print("Scenario summary:")
        print(f"  toolCallAt: {timings.get('toolCallAt')}")
        print(f"  toolResponseSentAt: {timings.get('toolResponseSentAt')}")
        if timings.get("toolCallAt") and timings.get("toolResponseSentAt"):
            print(f"  summary responseDelayMs: {timings['toolResponseSentAt'] - timings['toolCallAt']}")
        print("")
        print("Important:")
        print("  The summary uses the first toolCallAt but the latest toolResponseSentAt seen by the harness.")
        print("  If multiple tool calls occur, summary responseDelayMs can look much larger than one delay interval.")


if __name__ == "__main__":
    main()
