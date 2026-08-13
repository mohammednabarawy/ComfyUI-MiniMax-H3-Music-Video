"""V19 whole-song cloud Director with provider failover and disk caching."""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import os
import re
import ssl
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any, Dict, Iterable, List, Tuple

from .director_batch import _select_current, _stable_planning_payload, _storyboard
from .director_local import _strip_json_fence
from .schemas import DirectorBundle, ShotPlan, validate_semantic_rules


logger = logging.getLogger("comfyui_minimax_music_video.director_cloud")
_CACHE_VERSION = 19
_PROMPT_VERSION = "v19-creative-trendy-one-call-skill-20260813"

_PROVIDERS = {
    "gemini": {
        "base_url": "https://generativelanguage.googleapis.com/v1beta/openai",
        "model": "gemini-3.6-flash",
        "credential_file": "",
        "environment": ("GEMINI_API_KEY",),
        "preset": {
            "max_output_tokens": 32768,
            "temperature": 0.5,
            "reasoning_effort": "low",
            "timeout_sec": 300,
        },
    },
    "nvidia": {
        "base_url": "https://integrate.api.nvidia.com/v1",
        "model": "z-ai/glm-5.2",
        "credential_file": "",
        "environment": ("NVIDIA_API_KEY", "NGC_API_KEY"),
        "preset": {
            "max_output_tokens": 16384,
            "temperature": 0.6,
            "reasoning_effort": "low",
            "timeout_sec": 300,
        },
    },
    "zen": {
        "base_url": "https://opencode.ai/zen/v1",
        "model": "nemotron-3-ultra-free",
        "credential_file": "",
        "environment": ("ZEN_API_KEY", "OPENCODE_API_KEY"),
        "preset": {
            "max_output_tokens": 16384,
            "temperature": 0.6,
            "reasoning_effort": "low",
            "timeout_sec": 360,
        },
    },
}

_CUSTOM_PRESET = {
    "max_output_tokens": 12000,
    "temperature": 0.5,
    "reasoning_effort": "none",
    "timeout_sec": 300,
}


def _cache_file(run_dir: str) -> Path:
    return Path(run_dir).resolve() / "director_cloud_plan_v19.json"


def _cloud_cache_key(
    plan: Dict[str, Any],
    identity_prompt: str,
    provider: str,
    model: str,
    use_provider_preset: bool,
    planning_scope: str,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str,
) -> str:
    payload = {
        "cache_version": _CACHE_VERSION,
        "prompt_version": _PROMPT_VERSION,
        "plan": _stable_planning_payload(plan),
        "identity_prompt": identity_prompt,
        "provider": provider,
        "model": model,
        "use_provider_preset": bool(use_provider_preset),
        "planning_scope": planning_scope,
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "reasoning_effort": reasoning_effort,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _extract_api_key(path: str) -> str:
    if not path:
        return ""
    key_path = Path(path).expanduser()
    if not key_path.is_file():
        return ""
    text = key_path.read_text(encoding="utf-8-sig", errors="ignore")
    for pattern in (
        r"nvapi-[A-Za-z0-9_-]+",
        r"AIza[A-Za-z0-9_-]+",
        r"sk-[A-Za-z0-9_.-]+",
    ):
        match = re.search(pattern, text)
        if match:
            return match.group(0)
    for line in text.splitlines():
        match = re.match(r"(?i)^\s*(?:api\s*key|key|token)\s*[:=]\s*(\S+)\s*$", line)
        if match:
            return match.group(1)
    return ""


def _provider_order(provider: str, fallback_chain: str) -> List[str]:
    requested = str(provider).strip().lower()
    chain = [item.strip().lower() for item in str(fallback_chain).split(",") if item.strip()]
    ordered = (["gemini"] if requested == "auto" else [requested]) + chain
    result: List[str] = []
    for item in ordered:
        if item in (*_PROVIDERS.keys(), "custom_openai") and item not in result:
            result.append(item)
    return result or ["gemini", "nvidia", "zen"]


def _credentials(provider: str, explicit_key: str, credential_file: str) -> str:
    if explicit_key.strip():
        return explicit_key.strip()
    config = _PROVIDERS.get(provider, {})
    for name in config.get("environment", ()):
        value = os.environ.get(name, "").strip()
        if value:
            return value
    selected_file = credential_file.strip() or str(config.get("credential_file", ""))
    key = _extract_api_key(selected_file)
    if not key:
        raise ValueError(f"No API credential was found for the {provider} Director provider.")
    return key


def _provider_config(
    provider: str,
    requested_provider: str,
    selected_model: str,
    custom_base_url: str,
) -> Tuple[str, str]:
    if provider == "custom_openai":
        if not custom_base_url.strip() or not selected_model.strip():
            raise ValueError("custom_openai requires both custom_base_url and model.")
        return custom_base_url.rstrip("/"), selected_model.strip()
    config = _PROVIDERS[provider]
    primary = "gemini" if requested_provider == "auto" else requested_provider
    model = selected_model.strip() if provider == primary else ""
    known_defaults = {str(item["model"]) for item in _PROVIDERS.values()}
    if model in known_defaults and model != str(config["model"]):
        model = ""
    return str(config["base_url"]), model or str(config["model"])


def _provider_preset(provider: str) -> Dict[str, Any]:
    selected = "gemini" if provider == "auto" else str(provider).strip().lower()
    config = _PROVIDERS.get(selected)
    return dict(config.get("preset", _CUSTOM_PRESET)) if config else dict(_CUSTOM_PRESET)


def _effective_settings(
    provider: str,
    use_provider_preset: bool,
    max_output_tokens: int,
    temperature: float,
    reasoning_effort: str,
    timeout_sec: int,
) -> Dict[str, Any]:
    if use_provider_preset:
        return _provider_preset(provider)
    return {
        "max_output_tokens": int(max_output_tokens),
        "temperature": float(temperature),
        "reasoning_effort": str(reasoning_effort),
        "timeout_sec": int(timeout_sec),
    }


def _fetch_provider_models(
    provider: str,
    explicit_key: str = "",
    credential_file: str = "",
    custom_base_url: str = "",
) -> List[str]:
    """Fetch the current OpenAI-compatible model catalog for the selected provider."""
    selected = "gemini" if provider == "auto" else str(provider).strip().lower()
    if selected == "custom_openai":
        if not custom_base_url.strip():
            raise ValueError("Enter custom_base_url before refreshing custom models.")
        base_url = custom_base_url.rstrip("/")
    elif selected in _PROVIDERS:
        base_url = str(_PROVIDERS[selected]["base_url"])
    else:
        raise ValueError(f"Unsupported Director provider: {provider}")

    key = _credentials(selected, explicit_key, credential_file)
    request = urllib.request.Request(
        f"{base_url}/models",
        headers={
            "Authorization": f"Bearer {key}",
            "Accept": "application/json",
            "User-Agent": "ComfyUI-MiniMax-H3-V19/1.1",
        },
        method="GET",
    )
    with urllib.request.urlopen(
        request, timeout=30, context=ssl.create_default_context()
    ) as response:
        payload = json.loads(response.read().decode("utf-8"))
    entries = payload.get("data") or payload.get("models") or []
    models = []
    for item in entries:
        value = item.get("id") if isinstance(item, dict) else item
        if isinstance(value, str) and value.strip():
            model_id = value.strip()
            if selected == "gemini" and model_id.startswith("models/"):
                model_id = model_id.removeprefix("models/")
            lowered = model_id.lower()
            blocked_kinds = (
                "embedding", "embed-", "rerank", "retrieval", "-tts", "tts-",
                "native-audio", "live-api", "computer-use", "image-generation",
                "flash-image", "pro-image", "robotics", "antigravity", "deep-research",
            )
            if any(token in lowered for token in blocked_kinds):
                continue
            if selected == "gemini" and not lowered.startswith("gemini-"):
                continue
            models.append(model_id)
    models = sorted(set(models), key=str.casefold)
    default = str(_PROVIDERS.get(selected, {}).get("model", ""))
    if default in models:
        models.remove(default)
        models.insert(0, default)
    if not models:
        raise ValueError(f"{selected} returned an empty model catalog.")
    return models


def _strict_schema() -> Dict[str, Any]:
    schema = DirectorBundle.model_json_schema()

    def normalize(value: Any) -> None:
        if isinstance(value, dict):
            if value.get("type") == "object":
                value["additionalProperties"] = False
            for child in value.values():
                normalize(child)
        elif isinstance(value, list):
            for child in value:
                normalize(child)

    normalize(schema)
    return schema


def _chunks_for_scope(plan: Dict[str, Any], planning_scope: str) -> List[Dict[str, Any]]:
    all_chunks = plan.get("director_chunks") or plan.get("chunks") or []
    chunks = all_chunks if planning_scope == "whole_song" else (plan.get("chunks") or all_chunks)
    if not chunks:
        raise ValueError("The music-video controller supplied no chunks to plan.")
    return list(chunks)


def _chunk_brief(chunks: Iterable[Dict[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for item in chunks:
        start = float(item["start_ms"]) / 1000.0
        duration = float(item["duration_ms"]) / 1000.0
        result.append({
            "shot_id": int(item["index"]) + 1,
            "start_sec": round(start, 3),
            "end_sec": round(start + duration, 3),
            "duration_sec": round(duration, 3),
            "lyrics_excerpt": item.get("lyrics") or "[instrumental passage]",
        })
    return result


def _director_messages(
    chunks: List[Dict[str, Any]],
    full_lyrics: str,
    identity_prompt: str,
    song_bpm: float,
    song_duration: float,
    validation_error: str = "",
) -> List[Dict[str, str]]:
    system = (
        "You are the senior creative director and shot planner for a premium contemporary Arabic music video. "
        "Read Arabic lyrics accurately but write all visual directions in precise English. Plan the complete song in "
        "one response. First create one strong visual thesis, emotional arc, recurring motifs, palette progression, "
        "location logic and wardrobe progression. Then plan every supplied clip as one continuous MiniMax H3 shot. "
        "Rotate deliberately among performance, narrative, conceptual/symbolic, environmental and B-roll imagery. "
        "Verses favor narrative or observation, pre-choruses build tension, choruses become iconic and performance-led, "
        "bridges permit the boldest conceptual break, and the ending resolves the visual thesis. Avoid literal lyric "
        "illustration and generic AI cliches such as unjustified neon corridors, endless mirrors, floating particles, "
        "meaningless morphing, random text props or repetitive close-up singing. Consecutive shots must not repeat the "
        "same location + framing + camera movement + performer action combination. Fresh editorial cuts are the default; "
        "use continue or match_cut only when the adjacent scene genuinely benefits from the previous frame. "
        "For each clip use 3-5 meaningful visible beats, never more than one beat per second, and one coherent camera "
        "movement. Camera movement must use one correct H3 term: zoom in/out, push in/pull out, pan left/right, truck "
        "left/right, tilt up/down, pedestal up/down, arc shot, tracking shot, static shot, shake slightly/strongly, POV, "
        "or roll clockwise/counterclockwise. Speed/amplitude may use only: with small amplitude, with large amplitude, "
        "at slow speed, at fast speed. Do not stack conflicting movements. Only performance_lipsync shots may direct "
        "visible singing; narrative, silhouette, rear-view, conceptual, environmental and B-roll shots keep mouths closed "
        "or faces unseen. A visible performance shot must state that lips, jaw and mouth visibly follow the supplied song "
        "segment, without freezing expression or head movement. Reference image A controls facial identity; the selected "
        "B/C/D/E angle or body image supports view and scale only. Clothing, pose, background and lighting come from the "
        "shot plan, not from the identity pictures. Return only one JSON object matching the supplied schema."
    )
    schema_text = json.dumps(_strict_schema(), ensure_ascii=False, separators=(",", ":"))
    user = (
        f"CREATIVE DIRECTOR BIBLE AND IDENTITY RULES:\n{identity_prompt}\n\n"
        f"FULL SELECTED LYRICS:\n{full_lyrics}\n\n"
        f"BPM: {float(song_bpm):.3f}\nSONG DURATION: {float(song_duration):.3f} seconds\n\n"
        "AUTHORITATIVE CLIP TIMELINE (return exactly one shot_plan for every item, in this order; copy each "
        "lyrics_excerpt verbatim and copy duration_sec exactly):\n"
        f"{json.dumps(_chunk_brief(chunks), ensure_ascii=False, indent=2)}\n\n"
        "performance_mode must be one of performance_lipsync, performance_nonsinging, narrative, conceptual, "
        "environmental, or broll. section_role names the song section. beat_sequence contains the 3-5 visible beats. "
        "Use shot_type to reinforce the category. Keep scene_prompt focused on opening composition and motion_prompt "
        "chronological across the clip. Avoid any visible writing.\n\n"
        f"OUTPUT JSON SCHEMA:\n{schema_text}"
    )
    if validation_error:
        user += (
            "\n\nThe previous response failed validation. Return a complete corrected replacement. "
            f"Validation error: {validation_error}"
        )
    return [{"role": "system", "content": system}, {"role": "user", "content": user}]


def _response_text(payload: Dict[str, Any]) -> str:
    choices = payload.get("choices") or []
    if not choices:
        raise ValueError("The provider returned no chat-completion choices.")
    content = (choices[0].get("message") or {}).get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        pieces = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                pieces.append(item["text"])
        if pieces:
            return "".join(pieces)
    raise ValueError("The provider returned an empty or unsupported message body.")


def _post_chat(
    base_url: str,
    api_key: str,
    model: str,
    messages: List[Dict[str, str]],
    max_output_tokens: int,
    temperature: float,
    timeout_sec: int,
    reasoning_effort: str,
) -> Tuple[str, str]:
    schema = _strict_schema()
    formats: List[Tuple[str, Dict[str, Any] | None]] = [
        ("json_schema", {
            "type": "json_schema",
            "json_schema": {"name": "minimax_h3_music_video_plan", "strict": True, "schema": schema},
        }),
        ("json_object", {"type": "json_object"}),
        ("prompt_validated_json", None),
    ]
    last_error = ""
    for format_name, response_format in formats:
        body: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": float(temperature),
            "max_tokens": int(max_output_tokens),
            "stream": False,
        }
        if response_format is not None:
            body["response_format"] = response_format
        if reasoning_effort != "none":
            body["reasoning_effort"] = reasoning_effort
        request = urllib.request.Request(
            f"{base_url.rstrip('/')}/chat/completions",
            data=json.dumps(body, ensure_ascii=False).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "ComfyUI-MiniMax-H3-V19/1.0",
            },
            method="POST",
        )
        try:
            with urllib.request.urlopen(
                request, timeout=max(30, int(timeout_sec)), context=ssl.create_default_context()
            ) as response:
                payload = json.loads(response.read().decode("utf-8"))
            return _response_text(payload), format_name
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")[:1200]
            last_error = f"HTTP {exc.code}: {detail}"
            logger.warning("Director response format %s was rejected: HTTP %s", format_name, exc.code)
            # Some OpenAI-compatible providers reject reasoning_effort before they inspect response_format.
            if "reasoning_effort" in detail and "reasoning_effort" in body:
                reasoning_effort = "none"
            continue
        except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as exc:
            last_error = str(exc)
            break
    raise RuntimeError(last_error or "Cloud Director request failed without a response.")


_CAMERA_PATTERNS = (
    (r"zoom\s+in", "the camera zooms in"),
    (r"zoom\s+out", "the camera zooms out"),
    (r"(?:push|dolly)\s+in", "the camera pushes in"),
    (r"(?:pull|dolly)\s+(?:out|back)", "the camera pulls out"),
    (r"pan\s+left", "the camera pans left"),
    (r"pan\s+right", "the camera pans right"),
    (r"(?:truck|slide)\s+left", "the camera trucks left"),
    (r"(?:truck|slide)\s+right", "the camera trucks right"),
    (r"tilt\s+up", "the camera tilts up"),
    (r"tilt\s+down", "the camera tilts down"),
    (r"pedestal\s+up", "the camera pedestals up"),
    (r"pedestal\s+down", "the camera pedestals down"),
    (r"arc|orbit", "the camera performs an arc shot"),
    (r"track", "the camera performs a tracking shot"),
    (r"shake\s+strong", "the camera shakes strongly"),
    (r"handheld|shake", "the camera shakes slightly"),
    (r"\bpov\b|point.of.view", "the camera uses POV"),
    (r"roll\s+counter", "the camera rolls counterclockwise"),
    (r"roll\s+clock", "the camera rolls clockwise"),
    (r"static|locked", "the camera uses a static shot"),
)


def _canonical_camera(text: str) -> str:
    source = str(text or "static shot").lower()
    movement = "the camera uses a static shot"
    for pattern, replacement in _CAMERA_PATTERNS:
        if re.search(pattern, source):
            movement = replacement
            break
    amplitude = ""
    if re.search(r"small|slight|gentle|subtle", source):
        amplitude = " with small amplitude"
    elif re.search(r"large|wide|strong amplitude", source):
        amplitude = " with large amplitude"
    speed = ""
    if re.search(r"slow|restrained", source):
        speed = " at slow speed"
    elif re.search(r"fast|rapid|quick", source):
        speed = " at fast speed"
    if "static shot" in movement:
        amplitude = speed = ""
    return f"{movement}{amplitude}{speed}"


def _normalize_bundle(text: str, chunks: List[Dict[str, Any]]) -> Dict[str, Any]:
    cleaned = _strip_json_fence(text)
    try:
        data = json.loads(cleaned)
    except json.JSONDecodeError:
        # A few OpenAI-compatible gateways occasionally prefix an otherwise
        # valid object with a short acknowledgement despite strict mode.
        start, end = cleaned.find("{"), cleaned.rfind("}")
        if start < 0 or end <= start:
            raise
        data = json.loads(cleaned[start:end + 1])
    if isinstance(data, dict) and isinstance(data.get("director_bundle"), dict):
        data = data["director_bundle"]
    bundle = DirectorBundle.model_validate(data)
    if len(bundle.shot_plans) != len(chunks):
        raise ValueError(
            f"expected {len(chunks)} shot plans for the supplied timeline, got {len(bundle.shot_plans)}"
        )

    normalized: List[ShotPlan] = []
    previous_signature = None
    for position, (shot, chunk) in enumerate(zip(bundle.shot_plans, chunks), start=1):
        shot.shot_id = position
        shot.duration_sec = float(chunk["duration_ms"]) / 1000.0
        shot.lyrics_excerpt = chunk.get("lyrics") or "[instrumental passage]"
        shot.camera_motion = _canonical_camera(shot.camera_motion)
        shot = validate_semantic_rules(shot)
        if not shot.performer_visible and shot.performance_mode == "performance_lipsync":
            shot.performance_mode = "broll"
        if not shot.beat_sequence:
            shot.beat_sequence = [shot.action, shot.camera_motion, shot.ending_composition]
        max_beats = max(1, int(shot.duration_sec))
        shot.beat_sequence = shot.beat_sequence[: min(5, max_beats)]
        if shot.performer_visible is False:
            forbidden = " ".join((shot.scene_prompt, shot.motion_prompt, shot.action)).lower()
            if re.search(r"\b(?:lead performer|singer|lip.?sync|singing to camera)\b", forbidden):
                raise ValueError(f"shot {position} hides the performer but still directs the performer or lip-sync")
        signature = (
            shot.location.strip().lower(), shot.shot_size.strip().lower(),
            shot.camera_motion.strip().lower(), shot.action.strip().lower(),
        )
        if signature == previous_signature:
            raise ValueError(f"shot {position} repeats the complete previous-shot visual combination")
        previous_signature = signature
        normalized.append(shot)

    return {
        "master_plan": bundle.master_plan.model_dump(),
        "shot_plans": [shot.model_dump() for shot in normalized],
    }


def _payload(bundle: Dict[str, Any], chunks: List[Dict[str, Any]], cache_key: str,
             planning_scope: str, provider: str, model: str, response_format: str) -> Dict[str, Any]:
    shots = []
    for shot_plan, chunk in zip(bundle["shot_plans"], chunks):
        start = float(chunk["start_ms"]) / 1000.0
        duration = float(chunk["duration_ms"]) / 1000.0
        shots.append({
            "start_sec": start,
            "end_sec": start + duration,
            "duration_sec": duration,
            "lyrics": chunk.get("lyrics") or "[instrumental passage]",
            "shot_plan": shot_plan,
        })
    return {
        "cache_version": _CACHE_VERSION,
        "cache_key": cache_key,
        "planning_scope": planning_scope,
        "provider": provider,
        "model": model,
        "response_format": response_format,
        "master_plan": bundle["master_plan"],
        "shots": shots,
    }


class MiniMaxDirectorCloud:
    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "plan_song"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "master_plan_json", "all_shot_plans_json", "current_shot_plan_json",
        "storyboard_text", "planning_status", "provider_used",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("MV_PLAN",),
            "run_dir": ("STRING", {"default": ""}),
            "full_lyrics": ("STRING", {"multiline": True}),
            "identity_prompt": ("STRING", {"multiline": True}),
            "song_bpm": ("FLOAT", {"default": 120.0}),
            "current_shot_index": ("INT", {"default": 1, "min": 1}),
            "provider": (["auto", "gemini", "nvidia", "zen", "custom_openai"], {"default": "auto"}),
            "model": ("STRING", {"default": _PROVIDERS["gemini"]["model"]}),
            "use_provider_preset": ("BOOLEAN", {"default": True}),
            "fallback_chain": ("STRING", {"default": "gemini,nvidia,zen"}),
            "custom_base_url": ("STRING", {"default": ""}),
            "credential_file": ("STRING", {"default": ""}),
            "api_key": ("STRING", {"default": "", "password": True}),
            "planning_scope": (["whole_song", "render_limit"], {"default": "whole_song"}),
            "force_replan": ("BOOLEAN", {"default": False}),
            "max_output_tokens": ("INT", {"default": 32768, "min": 1024, "max": 65536, "step": 256}),
            "temperature": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 2.0, "step": 0.05}),
            "reasoning_effort": (["none", "low", "medium", "high"], {"default": "low"}),
            "timeout_sec": ("INT", {"default": 300, "min": 30, "max": 1800, "step": 30}),
        }}

    def plan_song(
        self, plan, run_dir, full_lyrics, identity_prompt, song_bpm, current_shot_index,
        provider="auto", model="gemini-3.6-flash", use_provider_preset=True,
        fallback_chain="gemini,nvidia,zen", custom_base_url="",
        credential_file="", api_key="", planning_scope="whole_song", force_replan=False,
        max_output_tokens=32768, temperature=0.5, reasoning_effort="low", timeout_sec=300,
    ):
        if not run_dir:
            raise ValueError("run_dir is required for the reusable whole-song Director cache.")
        selected_model = str(model or "").strip()
        cache_key = _cloud_cache_key(
            plan, identity_prompt, provider, selected_model, use_provider_preset, planning_scope,
            max_output_tokens, temperature, reasoning_effort,
        )
        cache_path = _cache_file(run_dir)
        if cache_path.is_file() and not force_replan:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                current = _select_current(cached, current_shot_index)
                used = f"{cached.get('provider', 'unknown')} / {cached.get('model', 'unknown')}"
                status = (
                    f"CACHE HIT | {len(cached['shots'])} whole-song shots ready | {used} | "
                    "no local LLM or Director VRAM is loaded"
                )
                return (
                    json.dumps(cached["master_plan"], ensure_ascii=False, indent=2),
                    json.dumps(cached["shots"], ensure_ascii=False, indent=2),
                    json.dumps(current, ensure_ascii=False, indent=2),
                    _storyboard(cached), status, used,
                )

        chunks = _chunks_for_scope(plan, planning_scope)
        errors = []
        for provider_name in _provider_order(provider, fallback_chain):
            try:
                base_url, provider_model = _provider_config(
                    provider_name, provider, selected_model, custom_base_url
                )
                key = _credentials(provider_name, api_key, credential_file)
                settings = _effective_settings(
                    provider_name, use_provider_preset, max_output_tokens,
                    temperature, reasoning_effort, timeout_sec,
                )
                validation_error = ""
                for attempt in range(2):
                    messages = _director_messages(
                        chunks, full_lyrics, identity_prompt, song_bpm,
                        float(plan.get("song_duration", 0.0)), validation_error,
                    )
                    text, response_format = _post_chat(
                        base_url, key, provider_model, messages,
                        settings["max_output_tokens"], settings["temperature"],
                        settings["timeout_sec"], settings["reasoning_effort"],
                    )
                    try:
                        bundle = _normalize_bundle(text, chunks)
                        result = _payload(
                            bundle, chunks, cache_key, planning_scope,
                            provider_name, provider_model, response_format,
                        )
                        cache_path.parent.mkdir(parents=True, exist_ok=True)
                        temp_path = Path(f"{cache_path}.tmp")
                        temp_path.write_text(
                            json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
                        )
                        os.replace(temp_path, cache_path)
                        current = _select_current(result, current_shot_index)
                        used = f"{provider_name} / {provider_model}"
                        status = (
                            f"PLANNED + CACHED | one cloud call created master treatment + "
                            f"{len(result['shots'])} shots | {used} | {response_format} | "
                            "no local Director model loaded"
                        )
                        return (
                            json.dumps(result["master_plan"], ensure_ascii=False, indent=2),
                            json.dumps(result["shots"], ensure_ascii=False, indent=2),
                            json.dumps(current, ensure_ascii=False, indent=2),
                            _storyboard(result), status, used,
                        )
                    except Exception as exc:
                        validation_error = str(exc)
                        logger.warning(
                            "Cloud Director %s validation attempt %s failed: %s",
                            provider_name, attempt + 1, exc,
                        )
                raise RuntimeError(f"returned invalid plan twice: {validation_error}")
            except Exception as exc:
                errors.append(f"{provider_name}: {exc}")
                logger.warning("Cloud Director provider %s failed: %s", provider_name, exc)
        raise RuntimeError("All configured cloud Director providers failed: " + " | ".join(errors))


_ROUTES_REGISTERED = False


def register_server_routes() -> None:
    """Expose model discovery through ComfyUI so credentials stay server-side."""
    global _ROUTES_REGISTERED
    if _ROUTES_REGISTERED:
        return
    try:
        from aiohttp import web
        from server import PromptServer
    except ImportError:
        return
    prompt_server = getattr(PromptServer, "instance", None)
    if prompt_server is None:
        return

    @prompt_server.routes.post("/minimax_music_video/v19/models")
    async def minimax_v19_models(request):
        try:
            body = await request.json()
            provider = str(body.get("provider", "auto"))
            models = await asyncio.to_thread(
                _fetch_provider_models,
                provider,
                str(body.get("api_key", "")),
                str(body.get("credential_file", "")),
                str(body.get("custom_base_url", "")),
            )
            selected = "gemini" if provider == "auto" else provider
            return web.json_response({
                "provider": selected,
                "models": models,
                "default_model": str(_PROVIDERS.get(selected, {}).get("model", models[0])),
                "preset": _provider_preset(selected),
            })
        except Exception as exc:
            logger.warning("V19 model catalog refresh failed: %s", exc)
            return web.json_response({"error": str(exc)}, status=400)

    _ROUTES_REGISTERED = True
