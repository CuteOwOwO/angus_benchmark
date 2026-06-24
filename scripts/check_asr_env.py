#!/usr/bin/env python3
import importlib.util
import json
import shutil
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def command_exists(name: str) -> bool:
    return shutil.which(name) is not None


def run_command(args: list[str]) -> dict:
    try:
        completed = subprocess.run(args, check=False, capture_output=True, text=True, timeout=20)
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "stdout": completed.stdout.strip()[:2000],
            "stderr": completed.stderr.strip()[:2000],
        }
    except Exception as error:
        return {"ok": False, "error": str(error)}


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def latest_audio_file() -> Path | None:
    candidates = sorted(ROOT.glob("result/**/audio/*_compressed.wav"), key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0] if candidates else None


def main() -> None:
    audio = latest_audio_file()
    result = {
        "python": sys.version.split()[0],
        "ffmpeg_available": command_exists("ffmpeg"),
        "ffmpeg_version": run_command(["ffmpeg", "-version"]) if command_exists("ffmpeg") else None,
        "nvidia_smi_available": command_exists("nvidia-smi"),
        "nvidia_smi": run_command(["nvidia-smi"]) if command_exists("nvidia-smi") else None,
        "packages": {
            "faster_whisper": module_available("faster_whisper"),
            "whisper": module_available("whisper"),
            "torch": module_available("torch"),
        },
        "whisper_cpp_available": command_exists("whisper-cli") or command_exists("main"),
        "sample_audio": str(audio.relative_to(ROOT)) if audio else None,
        "asr_available": False,
        "asr_scope": None,
        "post_external_transcript": None,
        "contains_final_answer_angus": None,
        "notes": [],
    }

    if not result["ffmpeg_available"]:
      result["notes"].append("ASR unavailable: missing ffmpeg.")

    if result["packages"]["faster_whisper"]:
        result["asr_available"] = True
        result["notes"].append("faster-whisper is installed. CPU ASR should be possible with tiny/base models; GPU depends on CUDA availability.")
    elif result["packages"]["whisper"]:
        result["asr_available"] = True
        result["notes"].append("openai-whisper is installed. CPU ASR should be possible with tiny/base models.")
    else:
        result["notes"].append("ASR unavailable: missing faster-whisper/openai-whisper Python package.")
        result["notes"].append("Suggested setup: pip install faster-whisper, then test a tiny/base model on a short wav.")

    if result["asr_available"] and audio:
        result["asr_scope"] = "full_assistant_audio"
        result["notes"].append("ASR package exists, but this check does not download models or run transcription automatically.")
        result["notes"].append("To avoid surprise large model downloads, run transcription manually with a tiny/base model.")
    elif not audio:
        result["notes"].append("No sample *_compressed.wav found under result/**/audio.")

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
