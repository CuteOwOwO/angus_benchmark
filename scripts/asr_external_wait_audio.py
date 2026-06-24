#!/usr/bin/env python3
import argparse
import csv
import importlib.util
import json
import re
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[1]
CONDITIONS = {
    "condition_no_tick": "no_tick",
    "condition_tick_every_3000ms": "tick_every_3000ms",
}
DPI = 200


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def timestamp() -> str:
    return datetime.utcnow().strftime("%Y-%m-%d_%H-%M-%S")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def normalize(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^a-z0-9\s]", " ", text.lower())).strip()


def run_sort_key(path: Path) -> int:
    try:
        return int(path.name.split("_")[1])
    except Exception:
        return 999999


def read_run_summary(run_dir: Path) -> dict | None:
    summary_path = run_dir / "summary.json"
    if not summary_path.exists():
        return None
    data = read_json(summary_path)
    scenarios = data.get("scenarios") or []
    return scenarios[0] if scenarios else None


def collect_valid_audio(result_dirs: list[Path]) -> list[dict]:
    rows: list[dict] = []
    for result_dir in result_dirs:
        for run_dir in sorted([path for path in result_dir.rglob("run_*") if path.is_dir()], key=lambda path: (str(path.parent), run_sort_key(path))):
            summary = read_run_summary(run_dir)
            if not summary or not summary.get("session_valid"):
                continue
            condition = summary.get("condition")
            latency_ms = summary.get("latency_ms")
            scenario_id = summary.get("scenario_id")
            if not condition or not isinstance(latency_ms, int) or not scenario_id:
                continue
            audio_path = run_dir / "audio" / f"{scenario_id}_compressed.wav"
            if not audio_path.exists():
                audio_file = summary.get("audio_file")
                candidate = ROOT / audio_file if isinstance(audio_file, str) else None
                if candidate and candidate.exists() and not candidate.name.endswith("_timeline.wav"):
                    audio_path = candidate
            if audio_path.exists() and not audio_path.name.endswith("_timeline.wav"):
                rows.append(
                    {
                        "condition": condition,
                        "condition_label": CONDITIONS.get(str(condition), str(condition).removeprefix("condition_")),
                        "latency_ms": latency_ms,
                        "run_id": run_dir.name,
                        "run_dir": run_dir,
                        "audio_path": audio_path,
                    }
                )
    return rows


def faster_whisper_available() -> bool:
    return importlib.util.find_spec("faster_whisper") is not None


def unavailable_result(row: dict, args: argparse.Namespace, reason: str) -> dict:
    return {
        "condition": row["condition"],
        "latency_ms": row["latency_ms"],
        "run_id": row["run_id"],
        "audio_path": rel(row["audio_path"]),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "transcript": "",
        "contains_angus": False,
        "contains_angus_normalized": False,
        "segments": [],
        "asr_available": False,
        "asr_error": reason,
    }


def write_per_audio_outputs(result: dict, audio_path: Path) -> tuple[Path, Path]:
    txt_path = audio_path.with_suffix(".asr.txt")
    json_path = audio_path.with_suffix(".asr.json")
    if result.get("asr_available"):
        txt_path.write_text(str(result.get("transcript") or ""), encoding="utf-8")
    else:
        txt_path.write_text(f"ASR unavailable: {result.get('asr_error')}\n", encoding="utf-8")
    json_path.write_text(json.dumps(result, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return txt_path, json_path


def transcribe_audio(model, row: dict, args: argparse.Namespace) -> dict:
    segments_iter, _info = model.transcribe(str(row["audio_path"]))
    segments = []
    transcript_parts = []
    for segment in segments_iter:
        text = segment.text.strip()
        transcript_parts.append(text)
        segments.append({"start": float(segment.start), "end": float(segment.end), "text": text})
    transcript = " ".join(part for part in transcript_parts if part).strip()
    normalized = normalize(transcript)
    contains = "angus" in transcript
    contains_normalized = "angus" in normalized
    return {
        "condition": row["condition"],
        "latency_ms": row["latency_ms"],
        "run_id": row["run_id"],
        "audio_path": rel(row["audio_path"]),
        "model": args.model,
        "device": args.device,
        "compute_type": args.compute_type,
        "transcript": transcript,
        "contains_angus": contains,
        "contains_angus_normalized": contains_normalized,
        "segments": segments,
        "asr_available": True,
        "asr_error": None,
    }


def write_summary(out_dir: Path, results: list[dict], total_audio: int) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "asr_summary.csv"
    headers = [
        "condition",
        "latency_ms",
        "run_id",
        "audio_path",
        "asr_txt_path",
        "asr_json_path",
        "transcript",
        "contains_angus",
        "contains_angus_normalized",
        "asr_available",
        "asr_error",
    ]
    with csv_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(results)

    aggregate: dict[str, dict] = {}
    for result in results:
        key = f"{result['condition']}|{result['latency_ms']}"
        item = aggregate.setdefault(
            key,
            {
                "condition": result["condition"],
                "latency_ms": result["latency_ms"],
                "audio_count": 0,
                "asr_success_count": 0,
                "contains_angus_count": 0,
                "contains_angus_rate": None,
            },
        )
        item["audio_count"] += 1
        if result.get("asr_available") and not result.get("asr_error"):
            item["asr_success_count"] += 1
            if result.get("contains_angus_normalized"):
                item["contains_angus_count"] += 1
    for item in aggregate.values():
        item["contains_angus_rate"] = (
            item["contains_angus_count"] / item["asr_success_count"] if item["asr_success_count"] else None
        )

    aggregate_rows = sorted(aggregate.values(), key=lambda item: (item["latency_ms"], item["condition"]))
    chart_path = out_dir / "bar_contains_angus_rate_8bars.png"
    draw_contains_angus_chart(aggregate_rows, chart_path)

    json_path = out_dir / "asr_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "total_valid_run_audio_found": total_audio,
                "total_asr_success": sum(1 for result in results if result.get("asr_available") and not result.get("asr_error")),
                "contains_angus_rate_chart": rel(chart_path),
                "aggregate": aggregate_rows,
                "rows": results,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )


def draw_contains_angus_chart(aggregate_rows: list[dict], out_path: Path) -> None:
    order = [
        ("condition_no_tick", 3000),
        ("condition_tick_every_3000ms", 3000),
        ("condition_no_tick", 5000),
        ("condition_tick_every_3000ms", 5000),
        ("condition_no_tick", 8000),
        ("condition_tick_every_3000ms", 8000),
        ("condition_no_tick", 12000),
        ("condition_tick_every_3000ms", 12000),
    ]
    labels: list[str] = []
    values: list[float] = []
    counts: list[str] = []
    colors: list[str] = []
    for condition, latency_ms in order:
        row = next((item for item in aggregate_rows if item["condition"] == condition and item["latency_ms"] == latency_ms), None)
        labels.append(f"{CONDITIONS.get(condition, condition)}\n{latency_ms // 1000}s")
        rate = row.get("contains_angus_rate") if row else None
        values.append(rate if isinstance(rate, (int, float)) else 0)
        counts.append(f"{row.get('contains_angus_count', 0)}/{row.get('asr_success_count', 0)}" if row else "0/0")
        colors.append("#4C78A8" if condition == "condition_no_tick" else "#F58518")

    fig, ax = plt.subplots(figsize=(12, 5.6))
    x_positions = list(range(len(labels)))
    bars = ax.bar(x_positions, values, color=colors, width=0.72)
    ax.set_title("ASR contains 'Angus' rate by condition and external wait time", fontsize=14)
    ax.set_ylabel("Contains Angus rate")
    ax.set_ylim(0, 1.05)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(labels, fontsize=9)
    ax.grid(axis="y", linestyle=":", alpha=0.35)
    for bar, count in zip(bars, counts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.025, count, ha="center", va="bottom", fontsize=9)
    ax.text(
        0.5,
        0.01,
        "ASR model: faster-whisper tiny.en CPU int8. Matching uses normalized substring 'angus'.",
        ha="center",
        transform=fig.transFigure,
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(out_path, dpi=DPI, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Run or stub ASR for valid external-wait compressed audio files.")
    parser.add_argument("--pilot3s", required=True, type=Path)
    parser.add_argument("--batch", required=True, type=Path)
    parser.add_argument("--model", default="tiny.en")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    result_dirs = [args.pilot3s.resolve(), args.batch.resolve()]
    rows = collect_valid_audio(result_dirs)
    out_dir = (args.output or (ROOT / "result" / f"external_wait_asr_summary_{timestamp()}")).resolve()
    reason = None
    model = None
    if not faster_whisper_available():
        reason = "missing faster-whisper Python package. Suggested setup: pip install faster-whisper"
    else:
        try:
            from faster_whisper import WhisperModel

            model = WhisperModel(args.model, device=args.device, compute_type=args.compute_type)
        except Exception as error:
            reason = f"failed to initialize faster-whisper model: {error}"

    results: list[dict] = []
    for row in rows:
        if model is None:
            result = unavailable_result(row, args, reason or "ASR unavailable")
        else:
            try:
                result = transcribe_audio(model, row, args)
            except Exception as error:
                result = unavailable_result(row, args, f"ASR failed: {error}")
                result["asr_available"] = True
        txt_path, json_path = write_per_audio_outputs(result, row["audio_path"])
        result["asr_txt_path"] = rel(txt_path)
        result["asr_json_path"] = rel(json_path)
        results.append(result)

    write_summary(out_dir, results, len(rows))
    print(f"ASR summary directory: {rel(out_dir)}")
    print(f"Valid run audio found: {len(rows)}")
    print(f"ASR success: {sum(1 for result in results if result.get('asr_available') and not result.get('asr_error'))}")
    if reason:
        print(f"ASR unavailable: {reason}")


if __name__ == "__main__":
    main()
