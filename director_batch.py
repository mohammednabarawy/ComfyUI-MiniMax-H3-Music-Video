"""Whole-song local Director planning with one GGUF load and a reusable disk cache."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
from pathlib import Path
from typing import Any, Dict

from .director_local import (
    _DIRECTOR_LOCK,
    _available_models,
    _prepare_cuda_runtime,
    _prompts,
    _release_comfy_models,
    _resolve_model,
    _validate_response,
)
from .schemas import DirectorState, MasterPlan, ShotPlan


logger = logging.getLogger("comfyui_minimax_music_video.director_batch")
_CACHE_VERSION = 3


def _stable_planning_payload(plan: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "song_duration": plan.get("song_duration"),
        "fps": plan.get("fps"),
        "director_chunks": plan.get("director_chunks") or plan.get("chunks") or [],
    }


def _batch_cache_key(plan, identity_prompt, model_path, n_ctx, max_tokens, temperature, scope):
    stat = model_path.stat()
    payload = {
        "version": _CACHE_VERSION,
        "plan": _stable_planning_payload(plan),
        "identity_prompt": identity_prompt,
        "model": {"path": str(model_path), "size": stat.st_size, "mtime_ns": stat.st_mtime_ns},
        "n_ctx": int(n_ctx),
        "max_tokens": int(max_tokens),
        "temperature": float(temperature),
        "scope": scope,
    }
    return hashlib.sha256(
        json.dumps(payload, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def _cache_file(run_dir: str) -> Path:
    return Path(run_dir).resolve() / "director_batch_plan.json"


def _select_current(payload: Dict[str, Any], current_shot_index: int) -> Dict[str, Any]:
    shots = payload.get("shots") or []
    if not shots:
        raise ValueError("The whole-song Director cache contains no shots.")
    wanted = max(1, int(current_shot_index))
    for item in shots:
        if int(item["shot_plan"].get("shot_id", 0)) == wanted:
            return item["shot_plan"]
    raise IndexError(f"Shot {wanted} is not present in the whole-song Director plan.")


def _storyboard(payload: Dict[str, Any]) -> str:
    master = payload.get("master_plan", {})
    lines = [
        f"CONCEPT: {master.get('concept', '')}",
        f"NARRATIVE ARC: {master.get('narrative_arc', '')}",
        "",
    ]
    for item in payload.get("shots", []):
        shot = item["shot_plan"]
        lines.extend([
            f"SHOT {shot.get('shot_id')} | {item['start_sec']:.2f}-{item['end_sec']:.2f}s | {shot.get('video_mode')} | {shot.get('transition_mode')}",
            f"LYRICS: {item.get('lyrics', '')}",
            f"SCENE: {shot.get('scene_prompt', '')}",
            f"MOTION: {shot.get('motion_prompt', '')}",
            "",
        ])
    return "\n".join(lines).strip()


def _complete(llm, mode, user, system, shot_index, max_tokens, temperature):
    schema_model = MasterPlan if mode == "master" else ShotPlan
    validation_error = ""
    for attempt in range(2):
        request = user
        if validation_error:
            request += (
                "\n\nYour previous JSON failed validation. Return a complete corrected replacement. "
                f"Validation error: {validation_error}"
            )
        response = llm.create_chat_completion(
            messages=[{"role": "system", "content": system}, {"role": "user", "content": request}],
            max_tokens=int(max_tokens),
            temperature=float(temperature),
            top_p=0.9,
            repeat_penalty=1.08,
            response_format={"type": "json_object", "schema": schema_model.model_json_schema()},
        )
        text = response["choices"][0]["message"]["content"]
        try:
            return _validate_response(text, mode, shot_index)
        except Exception as exc:
            validation_error = str(exc)
            logger.warning("Batch Director %s validation attempt %s failed: %s", mode, attempt + 1, exc)
    raise RuntimeError(f"Batch Director returned invalid {mode} JSON twice: {validation_error}")


class MiniMaxDirectorBatch:
    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "plan_song"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "master_plan_json", "all_shot_plans_json", "current_shot_plan_json",
        "storyboard_text", "planning_status",
    )

    @classmethod
    def INPUT_TYPES(cls):
        models = _available_models()
        preferred = "Qwen2.5-7B-Instruct-1M-Q4_K_M.gguf"
        return {"required": {
            "plan": ("MV_PLAN",),
            "run_dir": ("STRING", {"default": ""}),
            "full_lyrics": ("STRING", {"multiline": True}),
            "identity_prompt": ("STRING", {"multiline": True}),
            "song_bpm": ("FLOAT", {"default": 120.0}),
            "current_shot_index": ("INT", {"default": 1, "min": 1}),
            "model": (models, {"default": preferred if preferred in models else models[0]}),
            "planning_scope": (["whole_song", "render_limit"], {"default": "whole_song"}),
            "force_replan": ("BOOLEAN", {"default": False}),
            "n_ctx": ("INT", {"default": 8192, "min": 2048, "max": 32768, "step": 512}),
            "max_tokens_per_plan": ("INT", {"default": 1500, "min": 512, "max": 4096, "step": 64}),
            "temperature": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 1.0, "step": 0.05}),
            "n_gpu_layers": ("INT", {"default": -1, "min": -1, "max": 200}),
        }}

    def plan_song(self, plan, run_dir, full_lyrics, identity_prompt, song_bpm,
                  current_shot_index, model, planning_scope="whole_song", force_replan=False,
                  n_ctx=8192, max_tokens_per_plan=1500, temperature=0.25, n_gpu_layers=-1):
        if int(n_gpu_layers) == 0:
            raise ValueError("CPU mode is disabled for the local Director; use n_gpu_layers=-1.")
        if not run_dir:
            raise ValueError("run_dir is required for the reusable whole-song Director cache.")
        model_path = _resolve_model(model)
        cache_key = _batch_cache_key(
            plan, identity_prompt, model_path, n_ctx, max_tokens_per_plan, temperature, planning_scope
        )
        cache_path = _cache_file(run_dir)
        if cache_path.is_file() and not force_replan:
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            if cached.get("cache_key") == cache_key:
                current = _select_current(cached, current_shot_index)
                status = (
                    f"CACHE HIT | {len(cached['shots'])} whole-song shots ready | "
                    "Qwen not loaded; H3 remains available"
                )
                return (
                    json.dumps(cached["master_plan"], ensure_ascii=False, indent=2),
                    json.dumps(cached["shots"], ensure_ascii=False, indent=2),
                    json.dumps(current, ensure_ascii=False, indent=2),
                    _storyboard(cached),
                    status,
                )

        all_chunks = plan.get("director_chunks") or plan.get("chunks") or []
        chunks = all_chunks if planning_scope == "whole_song" else (plan.get("chunks") or all_chunks)
        if not chunks:
            raise ValueError("The music-video controller supplied no chunks to plan.")

        llm = None
        with _DIRECTOR_LOCK:
            try:
                _release_comfy_models()
                _prepare_cuda_runtime()
                from llama_cpp import Llama

                llm = Llama(
                    model_path=str(model_path), n_ctx=int(n_ctx), n_batch=min(512, int(n_ctx)),
                    n_gpu_layers=int(n_gpu_layers), main_gpu=0, verbose=False,
                )
                system, user, _ = _prompts(
                    "master", full_lyrics, 10.0, "{}", "[]", "{}", 1, None,
                    identity_prompt, song_bpm, float(plan.get("song_duration", 0.0)),
                )
                master = _complete(
                    llm, "master", user, system, 1, max_tokens_per_plan, temperature
                )
                state = DirectorState(master_plan=MasterPlan.model_validate(master))
                shots = []
                for chunk in chunks:
                    shot_index = int(chunk["index"]) + 1
                    start_sec = float(chunk["start_ms"]) / 1000.0
                    duration_sec = float(chunk["duration_ms"]) / 1000.0
                    timed_lyrics = (
                        f"TIMECODE {start_sec:.3f}s to {start_sec + duration_sec:.3f}s\n"
                        f"{chunk.get('lyrics') or '[instrumental passage]'}"
                    )
                    recent = json.dumps([item.model_dump() for item in state.recent_shots], ensure_ascii=False)
                    usage = state.global_usage.model_dump_json()
                    system, user, _ = _prompts(
                        "shot", timed_lyrics, duration_sec,
                        json.dumps(master, ensure_ascii=False), recent, usage, shot_index, None,
                        identity_prompt, song_bpm, float(plan.get("song_duration", 0.0)),
                    )
                    shot_dict = _complete(
                        llm, "shot", user, system, shot_index, max_tokens_per_plan, temperature
                    )
                    shot = ShotPlan.model_validate(shot_dict)
                    state.add_shot(shot)
                    shots.append({
                        "start_sec": start_sec,
                        "end_sec": start_sec + duration_sec,
                        "duration_sec": duration_sec,
                        "lyrics": chunk.get("lyrics") or "[instrumental passage]",
                        "shot_plan": shot.model_dump(),
                    })
                payload = {
                    "cache_version": _CACHE_VERSION,
                    "cache_key": cache_key,
                    "planning_scope": planning_scope,
                    "master_plan": master,
                    "shots": shots,
                }
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                temp_path = Path(f"{cache_path}.tmp")
                temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
                os.replace(temp_path, cache_path)
                current = _select_current(payload, current_shot_index)
                status = (
                    f"PLANNED + CACHED | one Qwen load created master plan + {len(shots)} shots | "
                    "Qwen closed before H3; later clips use cache"
                )
                return (
                    json.dumps(master, ensure_ascii=False, indent=2),
                    json.dumps(shots, ensure_ascii=False, indent=2),
                    json.dumps(current, ensure_ascii=False, indent=2),
                    _storyboard(payload),
                    status,
                )
            finally:
                if llm is not None:
                    try:
                        llm.close()
                    except Exception as exc:
                        logger.warning("Could not close batch Director model cleanly: %s", exc)
                del llm
                gc.collect()
                _release_comfy_models()
