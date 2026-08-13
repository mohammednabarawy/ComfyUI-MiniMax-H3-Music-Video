"""Static validator for serialized ComfyUI frontend workflows.

Checks graph endpoints, slot indexes, slot names/types, live node contracts,
duplicate target connections, required connections, stale ID counters, cycles,
and the critical V16/V17 cached-preprocessing/batch-Director routing handoffs.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from collections import defaultdict, deque
from pathlib import Path


PRIMITIVES = {
    "PrimitiveFloat", "PrimitiveInt", "PrimitiveBoolean", "PrimitiveStringMultiline",
    "MarkdownNote", "ComfyMathExpression",
}
FRONTEND_EXTRA_INPUTS = {
    "LoadImage": {"upload"},
    "LoadAudio": {"audioUI", "upload"},
    # Allow validation immediately after a node update while a running ComfyUI
    # process still exposes its pre-restart object_info contract.
    "MiniMaxH3ReferenceSelector": {"strict_identity_mode"},
    "MiniMaxH3PromptComposer": {"single_identity_mode"},
}
FRONTEND_OMITTED_REQUIRED_INPUTS = {
    # Pixaroma stores the selected image in widgets_values and materializes the
    # backend image input while converting the frontend graph to an API prompt.
    "PixaromaLoadImageMini": {"image"},
    # NVIDIA's RTX VSR node serializes these dynamic-combo controls in
    # widgets_values.  They are materialized as backend inputs only when the
    # frontend converts the workflow graph to an API prompt.  This is also how
    # NVIDIA's own/reference workflows serialize the node.
    "RTXVideoSuperResolution": {"resize_type", "quality"},
}


def compatible(source, target):
    if source == target or source == "*" or target == "*":
        return True
    source_types = {item.strip() for item in str(source).split(",")}
    target_types = {item.strip() for item in str(target).split(",")}
    return bool(source_types & target_types)


def live_input_names(definition):
    names = set()
    autogrow = []
    for section in ("required", "optional"):
        for name, spec in definition.get("input", {}).get(section, {}).items():
            names.add(name)
            if (isinstance(spec, list) and spec and isinstance(spec[0], str)
                    and spec[0] in {"COMFY_AUTOGROW_V3", "COMFY_DYNAMICCOMBO_V3"}):
                autogrow.append(name + ".")
    return names, autogrow


def validate(path, object_info, enforce_music_video_contracts=True):
    workflow = json.loads(Path(path).read_text(encoding="utf-8"))
    errors = []
    nodes = workflow.get("nodes", [])
    links = workflow.get("links", [])
    by_id = {}
    for node in nodes:
        if node["id"] in by_id:
            errors.append(f"duplicate node id {node['id']}")
        by_id[node["id"]] = node

    if nodes and workflow.get("last_node_id", -1) < max(by_id):
        errors.append("last_node_id is stale")
    if links and workflow.get("last_link_id", -1) < max(link[0] for link in links):
        errors.append("last_link_id is stale")

    link_ids = set()
    target_slots = set()
    incoming = defaultdict(list)
    outgoing = defaultdict(list)
    graph = defaultdict(set)
    indegree = {node_id: 0 for node_id in by_id}

    for link in links:
        if len(link) != 6:
            errors.append(f"malformed link {link!r}")
            continue
        link_id, source_id, source_slot, target_id, target_slot, link_type = link
        if link_id in link_ids:
            errors.append(f"duplicate link id {link_id}")
        link_ids.add(link_id)
        if source_id not in by_id or target_id not in by_id:
            errors.append(f"link {link_id} references missing node")
            continue
        source = by_id[source_id]; target = by_id[target_id]
        if source_slot >= len(source.get("outputs", [])):
            errors.append(f"link {link_id} invalid source slot {source_id}:{source_slot}")
            continue
        if target_slot >= len(target.get("inputs", [])):
            errors.append(f"link {link_id} invalid target slot {target_id}:{target_slot}")
            continue
        target_key = (target_id, target_slot)
        if target_key in target_slots:
            errors.append(f"duplicate connection into {target_id}:{target_slot}")
        target_slots.add(target_key)
        source_type = source["outputs"][source_slot]["type"]
        target_type = target["inputs"][target_slot]["type"]
        if link_type != source_type:
            errors.append(f"link {link_id} declares {link_type}, source emits {source_type}")
        if not compatible(source_type, target_type):
            errors.append(f"link {link_id} type mismatch {source_type} -> {target_type}")
        if target["inputs"][target_slot].get("link") != link_id:
            errors.append(f"target {target_id}:{target_slot} does not point back to link {link_id}")
        if link_id not in (source["outputs"][source_slot].get("links") or []):
            errors.append(f"source {source_id}:{source_slot} does not list link {link_id}")
        incoming[target_id].append((link_id, source_id, source_slot, target_slot))
        outgoing[source_id].append((link_id, target_id, target_slot, source_slot))
        if target_id not in graph[source_id]:
            graph[source_id].add(target_id)
            indegree[target_id] += 1

    # Check node serialization against the exact live server contracts.
    for node in nodes:
        node_type = node["type"]
        if node_type in PRIMITIVES:
            continue
        definition = object_info.get(node_type)
        if not definition:
            errors.append(f"node {node['id']} uses unavailable type {node_type}")
            continue
        allowed, autogrow = live_input_names(definition)
        allowed |= FRONTEND_EXTRA_INPUTS.get(node_type, set())
        for item in node.get("inputs", []):
            name = item["name"]
            if name not in allowed and not any(name.startswith(prefix) for prefix in autogrow):
                errors.append(f"node {node['id']} has stale input {name!r} for {node_type}")
        live_outputs = list(zip(definition.get("output_name") or definition.get("output", []),
                                definition.get("output", [])))
        serialized_outputs = [(item["name"], item["type"]) for item in node.get("outputs", [])]
        if serialized_outputs != live_outputs:
            errors.append(f"node {node['id']} output contract differs from live {node_type}")

        serialized_by_name = {item["name"]: item for item in node.get("inputs", [])}
        for name, spec in definition.get("input", {}).get("required", {}).items():
            if name not in serialized_by_name:
                if name in FRONTEND_OMITTED_REQUIRED_INPUTS.get(node_type, set()):
                    continue
                errors.append(f"node {node['id']} missing required slot {name!r}")
                continue
            item = serialized_by_name[name]
            cfg = spec[1] if len(spec) > 1 and isinstance(spec[1], dict) else {}
            input_type = "COMBO" if isinstance(spec[0], list) else spec[0]
            widget_backed = input_type in {
                "STRING", "INT", "FLOAT", "BOOLEAN", "COMBO", "COMFY_DYNAMICCOMBO_V3"
            } and not cfg.get("forceInput", False)
            if item.get("link") is None and not widget_backed:
                errors.append(f"node {node['id']} required input {name!r} has no connection")

    # Detect graph cycles.
    queue = deque(node_id for node_id, degree in indegree.items() if degree == 0)
    visited = 0
    while queue:
        node_id = queue.popleft(); visited += 1
        for child in graph[node_id]:
            indegree[child] -= 1
            if indegree[child] == 0:
                queue.append(child)
    if visited != len(nodes):
        errors.append("workflow graph contains a dependency cycle")

    if not enforce_music_video_contracts:
        stats = {"nodes": len(nodes), "links": len(links), "output_nodes": [
            node["id"] for node in nodes if object_info.get(node["type"], {}).get("output_node")
        ]}
        return errors, stats

    # Critical V16/V17 handoffs that previously failed. Node IDs are stable
    # across these generated workflow revisions, so retain title aliases for
    # readable diagnostics while allowing descriptive V17 titles.
    titled = {node.get("title", ""): node for node in nodes}
    title_alias_ids = {
        "SONG CONTROLLER — AUTO LENGTH / RESUME": 112,
        "WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE": 120,
        "H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR": 125,
        "H3 REFERENCE-AWARE PROMPT COMPOSER": 126,
        "H3 MULTIREF DIRECT — selected identity pair + audio": 309,
        "H3 MULTIREF MATCH — identity pair + previous + audio": 318,
    }
    for alias, node_id in title_alias_ids.items():
        if node_id in by_id:
            titled.setdefault(alias, by_id[node_id])
    def require_link(source_title, source_name, target_title, target_name):
        source = titled.get(source_title); target = titled.get(target_title)
        if not source or not target:
            errors.append(f"missing critical node: {source_title!r} or {target_title!r}")
            return
        source_slot = next((i for i, item in enumerate(source["outputs"]) if item["name"] == source_name), None)
        target_slot = next((i for i, item in enumerate(target["inputs"]) if item["name"] == target_name), None)
        if source_slot is None or target_slot is None:
            errors.append(f"missing critical slot: {source_title}.{source_name} -> {target_title}.{target_name}")
            return
        if not any(x[1] == source["id"] and x[2] == source_slot and x[3] == target["id"] and x[4] == target_slot for x in links):
            errors.append(f"missing critical link: {source_title}.{source_name} -> {target_title}.{target_name}")

    require_link("SONG CONTROLLER — AUTO LENGTH / RESUME", "previous_clip", "PREVIOUS CLIP END FRAME", "existing_clip")
    require_link("CACHED PREPROCESS — GPU WHISPER LARGE-V3 + LIGHTWEIGHT BEATS", "beat_positions", "SONG CONTROLLER — AUTO LENGTH / RESUME", "beat_positions")
    require_link("CACHED PREPROCESS — GPU WHISPER LARGE-V3 + LIGHTWEIGHT BEATS", "whisper_chunks", "SONG CONTROLLER — AUTO LENGTH / RESUME", "whisper_chunks")
    require_link("SONG CONTROLLER — AUTO LENGTH / RESUME", "plan", "WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE", "plan")
    require_link("SONG CONTROLLER — AUTO LENGTH / RESUME", "clip_number", "WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE", "current_shot_index")
    require_link("WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE", "current_shot_plan_json", "VALIDATED SHOT PLAN", "shot_plan_json")
    require_link("VALIDATED SHOT PLAN", "full_shot_json", "H3 REFERENCE-AWARE PROMPT COMPOSER", "shot_plan_json")
    require_link("VALIDATED SHOT PLAN", "audio_sync_priority", "H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR", "audio_sync_priority")
    selector = titled.get("H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR")
    secondary_links = (selector["outputs"][1].get("links") or []) if selector else []
    strict_identity = not bool(secondary_links)
    require_link("H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR", "primary_identity", "H3 MULTIREF DIRECT — selected identity pair + audio", "ref_images.ref_image_0")
    require_link("H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR", "primary_identity", "H3 MULTIREF MATCH — identity pair + previous + audio", "ref_images.ref_image_0")
    if strict_identity:
        require_link("PREVIOUS CLIP END FRAME", "image", "H3 MULTIREF MATCH — identity pair + previous + audio", "ref_images.ref_image_1")
        if secondary_links:
            errors.append("strict V17 identity mode still sends a secondary face reference")
    else:
        require_link("H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR", "secondary_identity", "H3 MULTIREF DIRECT — selected identity pair + audio", "ref_images.ref_image_1")
        require_link("PREVIOUS CLIP END FRAME", "image", "H3 MULTIREF MATCH — identity pair + previous + audio", "ref_images.ref_image_2")
    require_link("PREVIOUS CLIP END FRAME", "image", "H3 I2V CONTINUITY — previous frame is frame 0", "first_frame")
    require_link("PREVIOUS CLIP END FRAME", "image", "H3 B-ROLL CONTINUE/MATCH — previous + audio, zero identity refs", "ref_images.ref_image_0")
    require_link("H3 REFERENCE-AWARE PROMPT COMPOSER", "broll_audio_prompt", "H3 B-ROLL — audio + scene prompt, zero portraits", "prompt")
    require_link("H3 REFERENCE-AWARE PROMPT COMPOSER", "broll_previous_prompt", "H3 B-ROLL CONTINUE/MATCH — previous + audio, zero identity refs", "prompt")
    require_link("H3 REFERENCE-AWARE PROMPT COMPOSER", "i2v_prompt", "H3 I2V CONTINUITY — previous frame is frame 0", "prompt")
    require_link("H3 REFERENCE-AWARE PROMPT COMPOSER", "multiref_prompt", "H3 MULTIREF DIRECT — selected identity pair + audio", "prompt")
    require_link("H3 REFERENCE-AWARE PROMPT COMPOSER", "multiref_match_prompt", "H3 MULTIREF MATCH — identity pair + previous + audio", "prompt")
    require_link("SAVE CURRENT CLIP", "clip_path", "DIRECTOR STATE — COMMIT AFTER CLIP EXISTS", "completed_clip_path")
    require_link("SONG CONTROLLER — AUTO LENGTH / RESUME", "audio_slice", "DIRECTOR STATE — COMMIT AFTER CLIP EXISTS", "debug_audio")
    require_link("H3 DETERMINISTIC REFERENCE + STRATEGY SELECTOR", "render_strategy", "DIRECTOR STATE — COMMIT AFTER CLIP EXISTS", "render_strategy")
    require_link("SAVE STATE ONLY FOR NEW CLIP", "OUTPUT", "QUEUE NEXT CLIP / ASSEMBLE FINAL VIDEO", "director_state_json")
    require_link("WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE", "storyboard_text", "PREFLIGHT — COMPLETE STORYBOARD", "source")
    require_link("WHOLE-SONG DIRECTOR — ONE QWEN LOAD + DISK CACHE", "all_shot_plans_json", "PREFLIGHT — ALL TIMED SHOT JSON", "source")
    require_link("USE CORRECTED LYRICS, ELSE WHISPER", "lyrics", "PREFLIGHT — FULL SELECTED LYRICS", "source")

    serialized = json.dumps(workflow, ensure_ascii=False).lower()
    if "flux" in serialized:
        errors.append("H3-only workflow still contains a FLUX node, model, title, or setting")
    cloud_directors = [node for node in nodes if node.get("type") == "MiniMaxDirectorCloud"]
    if "gemini" in serialized and not cloud_directors:
        errors.append("local-only workflow still contains a Gemini node, field, model, title, or setting")
    local_directors = [node for node in nodes if node.get("type") == "MiniMaxDirectorLocal"]
    if local_directors:
        errors.append(f"per-clip local Director nodes remain: {len(local_directors)}")
    batch_directors = [node for node in nodes if node.get("type") == "MiniMaxDirectorBatch"]
    if len(batch_directors) + len(cloud_directors) != 1:
        errors.append(
            "expected exactly one whole-song batch or cloud Director node, found "
            f"{len(batch_directors)} batch and {len(cloud_directors)} cloud"
        )
    preprocessors = [node for node in nodes if node.get("type") == "MiniMaxSongPreprocessor"]
    if len(preprocessors) != 1:
        errors.append(f"expected exactly one cached song preprocessor, found {len(preprocessors)}")
    if any(node.get("type") == "MiniMaxDirectorGemini" for node in nodes):
        errors.append("workflow still contains a Gemini Director node")

    broll = titled.get("H3 B-ROLL — audio + scene prompt, zero portraits")
    if broll:
        image_inputs = [item for item in broll.get("inputs", [])
                        if item["name"].startswith("ref_images.") and item.get("link") is not None]
        if image_inputs:
            errors.append("performer-free H3 B-roll branch exposes identity image inputs")

    stats = {"nodes": len(nodes), "links": len(links), "output_nodes": [
        node["id"] for node in nodes if object_info.get(node["type"], {}).get("output_node")
    ]}
    return errors, stats


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workflow", type=Path)
    parser.add_argument("--server", default="http://127.0.0.1:8188")
    parser.add_argument("--generic", action="store_true",
                        help="Run graph and live-node validation without V16/V17 music-video contracts")
    args = parser.parse_args()
    with urllib.request.urlopen(args.server.rstrip("/") + "/object_info") as response:
        object_info = json.load(response)
    errors, stats = validate(args.workflow, object_info,
                             enforce_music_video_contracts=not args.generic)
    if errors:
        for error in errors:
            print("ERROR:", error)
        raise SystemExit(1)
    print(f"VALID: {args.workflow} | {stats['nodes']} nodes | {stats['links']} links | output nodes {stats['output_nodes']}")


if __name__ == "__main__":
    main()
