#!/usr/bin/env python3
"""Create a share-safe ComfyUI workflow from a local frontend workflow JSON."""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any


MEDIA_LOADERS = {
    "LoadAudio",
    "LoadImage",
    "PixaromaLoadImageMini",
    "VHS_LoadAudio",
    "VHS_LoadVideo",
}
SECRET_PATTERN = re.compile(
    r"(?i)(?:AIza[0-9A-Za-z_-]{20,}|AQ\.[0-9A-Za-z_-]{20,}|"
    r"nvapi-[0-9A-Za-z_-]{20,}|sk-[0-9A-Za-z_-]{20,}|gh[opsu]_[0-9A-Za-z]{20,})"
)
ABSOLUTE_PATH_PATTERN = re.compile(
    r"(?i)(?<![A-Za-z])[A-Z]:" + r"[\\/](?!/)"
    + "|" + r"C:" + r"\\Users\\"
    + "|/" + r"Users/[^/]+"
    + "|/" + r"home/[^/]+"
)

GENERIC_IDENTITY_MASTER = """subject_definitions:
<Subject 1> is the same performer shown in <Picture 1>, <Picture 2>, and <Picture 3>. <Picture 1> is the primary facial-identity reference, <Picture 2> supports angled facial geometry, and <Picture 3> supports natural body scale. The pictures do not prescribe wardrobe, pose, location, framing, lighting, or text.

summary:
[reference generation + audio reuse] Create one continuous identity-preserving music-video shot using the supplied references and current audio segment.

retention_analysis:
<Subject 1>: partially_preserved - preserve stable facial proportions and landmarks, skin tone, apparent age, hairstyle, facial-hair pattern when present, and natural body proportions while allowing the directed wardrobe and scene. <Audio 1>: fully_copy - reuse the supplied audio segment unchanged as the final soundtrack.

detailed_description:
[Shot 1] Follow the connected current-shot direction. Preserve one coherent performer, plausible anatomy, natural movement, and one continuous camera move. Do not duplicate, beautify, age-shift, widen, narrow, inflate, or reinterpret the performer. Do not copy clothing or backgrounds from the identity references. No subtitles, logos, watermarks, or visible written text.

overall_soundscape:
Use restrained location ambience and physical movement sounds without duplicating the music or vocals.

non_diegetic_music:
<Audio 1> is directly reused as the complete audience-only song segment for this clip."""

GENERIC_CURRENT_SHOT = """[Shot 1] Create a five-second continuous music-video shot in a clearly directed location. Give the performer newly designed plain, logo-free wardrobe unrelated to the reference clothing. Use one strong opening composition, one meaningful physical action, one coherent camera move, and an edit-friendly final frame. If the performer is visibly singing, synchronize natural lip and jaw movement to the supplied audio; otherwise keep the mouth natural and do not force lip sync."""


def _clean_nested(value: Any) -> Any:
    if isinstance(value, dict):
        cleaned = {}
        for key, item in value.items():
            if str(key).lower() in {"fullpath", "source_path", "absolute_path", "videopreview"}:
                continue
            cleaned[key] = _clean_nested(item)
        return cleaned
    if isinstance(value, list):
        return [_clean_nested(item) for item in value]
    if isinstance(value, str) and ABSOLUTE_PATH_PATTERN.search(value):
        return ""
    return value


def _set_widget_for_input(node: dict[str, Any], input_name: str, value: Any) -> None:
    inputs = node.get("inputs", [])
    widgets = node.get("widgets_values")
    if not isinstance(widgets, list):
        return
    index = next((i for i, item in enumerate(inputs) if item.get("name") == input_name), None)
    if index is not None and index < len(widgets):
        widgets[index] = value


def sanitize(workflow: dict[str, Any]) -> dict[str, Any]:
    workflow = _clean_nested(workflow)
    for node in workflow.get("nodes", []):
        node_type = node.get("type", "")
        title = str(node.get("title", ""))
        widgets = node.get("widgets_values")

        if node_type in MEDIA_LOADERS and isinstance(widgets, list) and widgets:
            widgets[0] = ""
        if node_type == "MiniMaxMusicVideoController" and isinstance(widgets, list) and widgets:
            widgets[0] = "my_music_video"
        if node_type == "MiniMaxDirectorCloud":
            _set_widget_for_input(node, "custom_base_url", "")
            _set_widget_for_input(node, "credential_file", "")
            _set_widget_for_input(node, "api_key", "")
        if node_type == "PrimitiveStringMultiline" and isinstance(widgets, list) and widgets:
            if "CORRECTED FULL LYRICS" in title.upper():
                widgets[0] = ""
            elif "FULL-REFERENCE MASTER" in title.upper():
                widgets[0] = GENERIC_IDENTITY_MASTER
            elif "CURRENT SHOT" in title.upper():
                widgets[0] = GENERIC_CURRENT_SHOT

    serialized = json.dumps(workflow, ensure_ascii=False)
    if SECRET_PATTERN.search(serialized):
        raise ValueError("A value shaped like an API credential remains in the workflow")
    if ABSOLUTE_PATH_PATTERN.search(serialized):
        raise ValueError("An absolute user or machine path remains in the workflow")
    return workflow


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    args = parser.parse_args()
    workflow = json.loads(args.source.read_text(encoding="utf-8"))
    clean = sanitize(workflow)
    args.destination.parent.mkdir(parents=True, exist_ok=True)
    args.destination.write_text(
        json.dumps(clean, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Sanitized {args.source.name} -> {args.destination}")


if __name__ == "__main__":
    main()
