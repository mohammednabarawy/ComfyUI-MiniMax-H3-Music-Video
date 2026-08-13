#!/usr/bin/env python3
"""Fail if a release tree contains common secrets, personal paths, or media."""

from __future__ import annotations

import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
TEXT_SUFFIXES = {".py", ".js", ".json", ".md", ".txt", ".toml", ".yml", ".yaml", ".gitignore"}
FORBIDDEN_MEDIA = {".mp3", ".wav", ".flac", ".mp4", ".mov", ".mkv", ".jpg", ".jpeg", ".png", ".webp"}
CHECKS = {
    "API credential": re.compile(
        r"(?i)(?:AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{20,}|"
        r"nvapi-[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,}|gh[opsu]_[0-9A-Za-z]{20,})"
    ),
    "absolute personal path": re.compile(
        r"(?i)(?<![A-Za-z])[A-Z]:" + r"[\\/](?!/)"
        + "|" + r"C:" + r"\\Users\\"
        + "|/" + r"Users/[^/]+"
        + "|/" + r"home/[^/]+"
    ),
    "private key": re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
}


def main() -> int:
    findings: list[str] = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or ".git" in path.parts:
            continue
        relative = path.relative_to(ROOT)
        if path.suffix.lower() in FORBIDDEN_MEDIA:
            findings.append(f"forbidden media: {relative}")
            continue
        if path.suffix.lower() not in TEXT_SUFFIXES and path.name not in {"LICENSE"}:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            findings.append(f"unexpected binary file: {relative}")
            continue
        for label, pattern in CHECKS.items():
            for match in pattern.finditer(text):
                line = text.count("\n", 0, match.start()) + 1
                findings.append(f"{label}: {relative}:{line}")
    if findings:
        print("Release audit failed:")
        print("\n".join(f"- {item}" for item in findings))
        return 1
    print("Release audit passed: no credential shapes, absolute personal paths, private keys, or media files.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
