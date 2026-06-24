#!/usr/bin/env python3
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
RESULT_DIR = ROOT / "result"
TEXT_SUFFIXES = {".json", ".jsonl", ".csv", ".txt", ".md"}
TIMESTAMP_FIRST = re.compile(r"^\d{4}-\d{2}-\d{2}[T_]")
PATTERNS = [
    re.compile(r"^(?P<prefix>.+)_(?P<ts>\d{4}-\d{2}-\d{2}T\d{2}-\d{2}-\d{2}-\d{3}Z)$"),
    re.compile(r"^(?P<prefix>.+)_(?P<ts>\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:-\d{3})?)$"),
]


def target_name(name: str) -> str | None:
    if TIMESTAMP_FIRST.match(name):
        return None
    for pattern in PATTERNS:
        match = pattern.match(name)
        if match:
            return f"{match.group('ts')}_{match.group('prefix')}"
    return None


def replace_text_references(replacements: dict[str, str]) -> int:
    changed = 0
    for path in RESULT_DIR.rglob("*"):
        if not path.is_file() or path.suffix not in TEXT_SUFFIXES:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        updated = text
        for old, new in replacements.items():
            updated = updated.replace(f"result/{old}", f"result/{new}")
            updated = updated.replace(f"{RESULT_DIR}/{old}", f"{RESULT_DIR}/{new}")
        if updated != text:
            path.write_text(updated, encoding="utf-8")
            changed += 1
    return changed


def main() -> None:
    replacements: dict[str, str] = {}
    for path in sorted(RESULT_DIR.iterdir()):
        if not path.is_dir():
            continue
        new_name = target_name(path.name)
        if not new_name:
            continue
        target = path.with_name(new_name)
        if target.exists():
            raise SystemExit(f"Refusing to overwrite existing directory: {target}")
        path.rename(target)
        replacements[path.name] = new_name
        print(f"{path.name} -> {new_name}")

    changed_files = replace_text_references(replacements) if replacements else 0
    print(f"renamed_dirs={len(replacements)}")
    print(f"updated_text_files={changed_files}")


if __name__ == "__main__":
    main()
