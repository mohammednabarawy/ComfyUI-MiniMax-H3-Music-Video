from __future__ import annotations

import gc
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Dict, Tuple

import torch

from .schemas import MasterPlan, ShotPlan, validate_semantic_rules


logger = logging.getLogger("comfyui_minimax_music_video.director_local")
_DIRECTOR_LOCK = threading.Lock()
_NO_MODEL = "No GGUF models found"
_DLL_HANDLES = []
_DLL_DIRECTORIES = set()


def _llm_root() -> Path:
    try:
        import folder_paths

        return Path(folder_paths.models_dir) / "LLM"
    except Exception:
        return Path(__file__).resolve().parents[2] / "models" / "LLM"


def _available_models() -> list[str]:
    root = _llm_root()
    if not root.is_dir():
        return [_NO_MODEL]
    models = sorted(
        path.relative_to(root).as_posix()
        for path in root.rglob("*.gguf")
        if path.is_file()
    )
    return models or [_NO_MODEL]


def _resolve_model(model: str) -> Path:
    if not model or model == _NO_MODEL:
        raise FileNotFoundError(f"No local GGUF model is available under {_llm_root()}.")
    candidate = Path(model)
    if not candidate.is_absolute():
        candidate = _llm_root() / candidate
    candidate = candidate.resolve()
    if not candidate.is_file():
        raise FileNotFoundError(f"Local Director model was not found: {candidate}")
    return candidate


def _release_comfy_models() -> None:
    """Free ComfyUI model allocations before or after the local Director runs."""
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache()
    except Exception as exc:
        logger.debug("ComfyUI model cleanup was unavailable: %s", exc)
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _prepare_cuda_runtime() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError(
            "The local Director requires CUDA, but PyTorch cannot access an NVIDIA GPU. "
            "CPU fallback is intentionally disabled."
        )
    torch.cuda.init()
    # llama.cpp's dynamic CUDA backend depends on the CUDA 12.8 DLLs bundled
    # with this portable PyTorch installation. Register both DLL directories
    # before importing llama_cpp.
    for directory in (
        Path(torch.__file__).resolve().parent / "lib",
        Path(torch.__file__).resolve().parents[1] / "llama_cpp" / "lib",
    ):
        directory_text = str(directory)
        if (
            directory.is_dir()
            and hasattr(os, "add_dll_directory")
            and directory_text not in _DLL_DIRECTORIES
        ):
            handle = os.add_dll_directory(directory_text)
            _DLL_HANDLES.append(handle)
            _DLL_DIRECTORIES.add(directory_text)


def _strip_json_fence(text: str) -> str:
    value = (text or "").strip()
    if value.startswith("```"):
        value = value.split("\n", 1)[1] if "\n" in value else value[3:]
        if value.endswith("```"):
            value = value[:-3]
        value = value.strip()
        if value.lower().startswith("json"):
            value = value[4:].lstrip()
    return value


def _validate_response(text: str, mode: str, shot_index: int) -> Dict[str, Any]:
    data = json.loads(_strip_json_fence(text))
    # Smaller local models sometimes add one descriptive root key even when
    # instructed to emit the schema object directly. Accept that harmless
    # wrapper while keeping Pydantic validation strict for the inner payload.
    wrapper = "master_plan" if mode == "master" else "shot_plan"
    if isinstance(data, dict) and isinstance(data.get(wrapper), dict):
        data = data[wrapper]
    if mode == "master":
        return MasterPlan.model_validate(data).model_dump()
    shot = ShotPlan.model_validate(data)
    shot.shot_id = max(1, int(shot_index))
    return validate_semantic_rules(shot).model_dump()


def _previous_frame_summary(previous_end_frame) -> str:
    if previous_end_frame is None:
        return "No previous end frame is available."
    try:
        frame = previous_end_frame.detach().float()
        if frame.dim() == 4:
            frame = frame[0]
        rgb = frame.mean(dim=(0, 1)).cpu().tolist()
        brightness = float(frame.mean().cpu())
        return (
            "A previous frame exists and will be supplied directly to H3 when continuity is selected. "
            f"Its normalized mean RGB is {rgb[:3]} and brightness is {brightness:.3f}."
        )
    except Exception:
        return "A previous frame exists and will be supplied directly to H3 when continuity is selected."


def _motion_library() -> str:
    path = Path(__file__).with_name("motion_library") / "metadata.json"
    try:
        return path.read_text(encoding="utf-8") if path.is_file() else "{}"
    except Exception as exc:
        logger.warning("Could not read motion-library metadata: %s", exc)
        return "{}"


def _prompts(
    mode: str,
    lyrics: str,
    target_duration: float,
    master_plan_json: str,
    recent_shots_json: str,
    global_usage_json: str,
    current_shot_index: int,
    previous_end_frame,
    identity_prompt: str,
    song_bpm: float,
    song_duration: float,
) -> Tuple[str, str, type[MasterPlan] | type[ShotPlan]]:
    if mode == "master":
        system = (
            "You are the master director of a premium cinematic Arabic music video. "
            "Understand the Arabic lyrics, but write every visual direction and renderer prompt in precise English. "
            "Design a coherent whole-song concept, narrative progression, motifs, locations, wardrobe, palettes, "
            "and performer/B-roll rules. Create strong variety: do not repeat a location more than twice, do not "
            "repeat camera motion consecutively, vary shot sizes, and avoid generic mirrors or corridors unless the "
            "lyrics specifically justify them. MiniMax H3 is the only image/video renderer. Return only JSON that "
            "matches the supplied schema."
        )
        user = (
            f"CREATIVE DIRECTION AND EXACT PERFORMER IDENTITY:\n{identity_prompt}\n\n"
            f"COMPLETE TIMED/SELECTED LYRICS:\n{lyrics}\n\n"
            f"SONG BPM: {float(song_bpm):.3f}\n"
            f"SONG DURATION: {float(song_duration):.3f} seconds\n\n"
            "Build the reusable master plan for all later ten-second-or-shorter clips."
        )
        return system, user, MasterPlan

    system = (
        "You are the shot director for one clip in a premium cinematic Arabic music video. Understand the Arabic "
        "lyrics and express all MiniMax H3 directions in detailed English. Follow the master plan while avoiding "
        "repetition found in recent shots and global counters. scene_prompt must describe H3's opening composition, "
        "subject/object placement, framing, lighting, palette, wardrobe and pose. motion_prompt must describe action, "
        "camera movement and environmental motion over the full clip duration. MiniMax H3 is the only renderer. "
        "If performer_visible is false, never mention the performer in either prompt. If transition_mode is continue, "
        "video_mode must be i2v. A visible performer in a fresh or match-cut shot uses multiref. Performer-free fresh "
        "B-roll uses t2v. Use high audio_sync_priority for singing, dancing, beat hits or actions tied to a lyric; use "
        "low only when exact frame-zero continuity matters more. A match cut requires a concrete match_anchor. Return "
        "Do not turn an isolated lyric word into a literal occupation or location unless the surrounding lyric supports it. "
        "Avoid chalkboards, signs, posters, newspapers, labels, screens, books, or other text-bearing props; if one is "
        "essential, explicitly describe its surface as completely blank and without marks. Return only JSON matching "
        "the supplied schema."
    )
    user = (
        f"SHOT NUMBER: {int(current_shot_index)}\n"
        f"TARGET DURATION: {float(target_duration):.3f} seconds\n"
        f"SONG BPM: {float(song_bpm):.3f}\n"
        f"FULL SONG DURATION: {float(song_duration):.3f} seconds\n\n"
        f"LYRICS OVERLAPPING THIS AUDIO SLICE:\n{lyrics}\n\n"
        f"EXACT PERFORMER IDENTITY:\n{identity_prompt}\n\n"
        f"MASTER PLAN:\n{master_plan_json or '{}'}\n\n"
        f"RECENT COMPLETED SHOTS:\n{recent_shots_json or '[]'}\n\n"
        f"GLOBAL USAGE COUNTERS:\n{global_usage_json or '{}'}\n\n"
        f"PREVIOUS FRAME CONTEXT:\n{_previous_frame_summary(previous_end_frame)}\n\n"
        f"AVAILABLE MOTION REFERENCES:\n{_motion_library()}\n\n"
        "Direct exactly one shot whose action naturally fills the target duration and whose visual content expresses "
        "the meaning and emotional energy of the current lyrics."
    )
    return system, user, ShotPlan


class MiniMaxDirectorLocal:
    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "direct"
    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("plan_json", "runtime_status")

    @classmethod
    def INPUT_TYPES(cls):
        models = _available_models()
        preferred = "Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
        default_model = preferred if preferred in models else models[0]
        return {
            "required": {
                "mode": (["master", "shot"],),
                "lyrics": ("STRING", {"multiline": True}),
                "model": (models, {"default": default_model}),
            },
            "optional": {
                "target_duration": ("FLOAT", {"default": 10.0, "min": 0.1, "max": 30.0}),
                "master_plan_json": ("STRING", {"multiline": True, "default": "{}"}),
                "recent_shots_json": ("STRING", {"multiline": True, "default": "[]"}),
                "global_usage_json": ("STRING", {"multiline": True, "default": "{}"}),
                "current_shot_index": ("INT", {"default": 1, "min": 1}),
                "previous_end_frame": ("IMAGE",),
                "identity_prompt": ("STRING", {"multiline": True, "default": ""}),
                "song_bpm": ("FLOAT", {"default": 120.0}),
                "song_duration": ("FLOAT", {"default": 0.0}),
                "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 512}),
                "max_tokens": ("INT", {"default": 1600, "min": 256, "max": 4096, "step": 64}),
                "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
                "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
            },
        }

    def direct(
        self,
        mode: str,
        lyrics: str,
        model: str,
        target_duration: float = 10.0,
        master_plan_json: str = "{}",
        recent_shots_json: str = "[]",
        global_usage_json: str = "{}",
        current_shot_index: int = 1,
        previous_end_frame=None,
        identity_prompt: str = "",
        song_bpm: float = 120.0,
        song_duration: float = 0.0,
        n_ctx: int = 8192,
        max_tokens: int = 1600,
        temperature: float = 0.25,
        n_gpu_layers: int = -1,
    ):
        if int(n_gpu_layers) == 0:
            raise ValueError("CPU mode is disabled for the local Director; use n_gpu_layers=-1.")
        if target_duration > 100:
            target_duration /= 1000.0
        model_path = _resolve_model(model)
        system, user, schema_model = _prompts(
            mode,
            lyrics,
            target_duration,
            master_plan_json,
            recent_shots_json,
            global_usage_json,
            current_shot_index,
            previous_end_frame,
            identity_prompt,
            song_bpm,
            song_duration,
        )
        schema = schema_model.model_json_schema()
        llm = None
        with _DIRECTOR_LOCK:
            try:
                _release_comfy_models()
                _prepare_cuda_runtime()
                from llama_cpp import Llama

                llm = Llama(
                    model_path=str(model_path),
                    n_ctx=int(n_ctx),
                    n_batch=min(512, int(n_ctx)),
                    n_gpu_layers=int(n_gpu_layers),
                    main_gpu=0,
                    verbose=False,
                )
                validation_error = ""
                for attempt in range(2):
                    request = user
                    if validation_error:
                        request += (
                            "\n\nYour previous JSON failed validation. Correct it and return a complete replacement JSON. "
                            f"Validation error: {validation_error}"
                        )
                    response = llm.create_chat_completion(
                        messages=[
                            {"role": "system", "content": system},
                            {"role": "user", "content": request},
                        ],
                        max_tokens=int(max_tokens),
                        temperature=float(temperature),
                        top_p=0.9,
                        repeat_penalty=1.08,
                        response_format={"type": "json_object", "schema": schema},
                    )
                    text = response["choices"][0]["message"]["content"]
                    try:
                        validated = _validate_response(text, mode, current_shot_index)
                        output = json.dumps(validated, ensure_ascii=False, indent=2)
                        status = (
                            f"Local GPU Director completed with {model_path.name}; "
                            "the GGUF model was closed before H3 execution."
                        )
                        return (output, status)
                    except Exception as exc:
                        validation_error = str(exc)
                        logger.warning("Local Director validation attempt %s failed: %s", attempt + 1, exc)
                raise RuntimeError(f"Local Director returned invalid JSON twice: {validation_error}")
            finally:
                if llm is not None:
                    try:
                        llm.close()
                    except Exception as exc:
                        logger.warning("Could not close the local Director model cleanly: %s", exc)
                del llm
                gc.collect()
                _release_comfy_models()
