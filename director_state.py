import os
import json
import logging
import wave
from pathlib import Path
import numpy as np
from PIL import Image
from typing import Tuple, Dict, Any
from .schemas import DirectorState, ShotPlan, MasterPlan

logger = logging.getLogger(__name__)


def _save_debug_image(image, path: Path) -> None:
    frame = image.detach().float().cpu()
    if frame.dim() == 4:
        frame = frame[0]
    pixels = (frame.clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    Image.fromarray(pixels).save(path)


def _save_debug_audio(audio, path: Path) -> None:
    waveform = audio["waveform"].detach().float().cpu()
    if waveform.dim() == 3:
        waveform = waveform[0]
    if waveform.dim() == 1:
        waveform = waveform.unsqueeze(0)
    samples = (waveform.clamp(-1, 1).numpy().T * 32767.0).round().astype(np.int16)
    with wave.open(str(path), "wb") as output:
        output.setnchannels(samples.shape[1])
        output.setsampwidth(2)
        output.setframerate(int(audio["sample_rate"]))
        output.writeframes(samples.tobytes())

def load_director_state(run_dir: str) -> DirectorState:
    """Load state from disk, return empty state if not found."""
    state_path = os.path.join(run_dir, "director_state.json")
    try:
        if os.path.exists(state_path):
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
                return DirectorState.model_validate(data)
    except Exception as e:
        logger.warning(f"Failed to load director state from {state_path}: {e}")
    return DirectorState()

def save_director_state(run_dir: str, state: DirectorState) -> None:
    """Atomically save state to disk."""
    state_path = os.path.join(run_dir, "director_state.json")
    tmp_path = f"{state_path}.tmp"

    os.makedirs(run_dir, exist_ok=True)

    try:
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(state.model_dump_json(indent=2))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, state_path)
    except Exception as e:
        logger.error(f"Failed to save director state to {state_path}: {e}")
        if os.path.exists(tmp_path):
            try:
                os.remove(tmp_path)
            except:
                pass
        raise

class MiniMaxDirectorState:
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Any]:
        return {
            "required": {
                "run_dir": ("STRING", {"default": ""}),
                "action": (["load", "save"],),
            },
            "optional": {
                "shot_plan_json": ("STRING", {"default": "", "multiline": True}),
                "master_plan_json": ("STRING", {"default": "", "multiline": True}),
                "completed_clip_path": ("STRING", {"default": "", "forceInput": True}),
                "debug_image": ("IMAGE",),
                "debug_audio": ("AUDIO",),
                "previous_end_frame": ("IMAGE",),
                "previous_clip_path": ("STRING", {"default": ""}),
                "render_strategy": ("STRING", {"default": ""}),
                "start_ms": ("FLOAT", {"default": 0.0}),
                "duration_ms": ("FLOAT", {"default": 0.0}),
                "song_bpm": ("FLOAT", {"default": 0.0}),
                "current_lyrics": ("STRING", {"default": "", "multiline": True}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING", "INT", "STRING", "STRING", "BOOLEAN")
    RETURN_NAMES = ("state_json", "master_plan_json", "current_shot", "recent_shots_json", "global_usage_json", "has_master_plan")
    FUNCTION = "process"
    CATEGORY = "MiniMax/MusicVideo"
    OUTPUT_NODE = False

    def process(
        self,
        run_dir: str,
        action: str,
        shot_plan_json: str = "",
        master_plan_json: str = "",
        completed_clip_path: str = "",
        debug_image=None,
        debug_audio=None,
        previous_end_frame=None,
        previous_clip_path: str = "",
        render_strategy: str = "",
        start_ms: float = 0.0,
        duration_ms: float = 0.0,
        song_bpm: float = 0.0,
        current_lyrics: str = ""
    ) -> Tuple[str, str, int, str, str, bool]:

        # Resolve path
        run_dir = os.path.abspath(run_dir)

        state = load_director_state(run_dir)

        if action == "save":
            if shot_plan_json and (not completed_clip_path or not os.path.isfile(completed_clip_path)):
                raise RuntimeError("Director state was not saved because the rendered clip is missing.")
            modified = False

            if master_plan_json:
                try:
                    mp_data = json.loads(master_plan_json)
                    state.master_plan = MasterPlan.model_validate(mp_data)
                    modified = True
                except Exception as e:
                    logger.warning(f"Failed to parse master_plan_json: {e}")

            if shot_plan_json:
                try:
                    sp_data = json.loads(shot_plan_json)
                    shot = ShotPlan.model_validate(sp_data)
                    state.add_shot(shot)
                    modified = True
                except Exception as e:
                    logger.warning(f"Failed to parse shot_plan_json: {e}")

            if modified:
                save_director_state(run_dir, state)

            try:
                debug_dir = Path(run_dir) / "debug"
                debug_dir.mkdir(parents=True, exist_ok=True)
                if master_plan_json:
                    (debug_dir / "master_plan.json").write_text(
                        json.dumps(json.loads(master_plan_json), ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                if shot_plan_json:
                    shot_data = json.loads(shot_plan_json)
                    shot_id = int(shot_data.get("shot_id", state.current_shot))
                    stem = f"shot_{shot_id:04d}"
                    start_sec = float(start_ms) / 1000.0
                    duration_sec = float(duration_ms) / 1000.0
                    (debug_dir / f"{stem}.json").write_text(
                        json.dumps({
                            "start_sec": start_sec,
                            "end_sec": start_sec + duration_sec,
                            "duration_sec": duration_sec,
                            "bpm": float(song_bpm),
                            "current_lyrics": current_lyrics,
                            "render_strategy": render_strategy,
                            "audio": {"start_sec": start_sec, "end_sec": start_sec + duration_sec},
                            "shot_plan": shot_data,
                        },
                                   ensure_ascii=False, indent=2),
                        encoding="utf-8",
                    )
                    if debug_audio is not None:
                        _save_debug_audio(debug_audio, debug_dir / f"{stem}_audio.wav")
                    if debug_image is not None:
                        image_name = f"{stem}_h3_first_frame.png"
                        _save_debug_image(debug_image, debug_dir / image_name)
                    if previous_end_frame is not None and previous_clip_path and os.path.isfile(previous_clip_path):
                        _save_debug_image(previous_end_frame, debug_dir / f"{stem}_previous_frame.png")
            except Exception as e:
                logger.warning("Could not save music-video debug artifacts: %s", e)

        # Prepare outputs
        state_json = state.model_dump_json()
        mp_json = state.master_plan.model_dump_json() if state.master_plan else "{}"
        curr_shot = state.current_shot
        recent_json = json.dumps([s.model_dump() for s in state.recent_shots])
        global_json = state.global_usage.model_dump_json()
        has_master_plan = state.master_plan is not None
        return (state_json, mp_json, curr_shot, recent_json, global_json, has_master_plan)
