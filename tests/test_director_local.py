import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comfyui_minimax_music_video.director_local import (
    MiniMaxDirectorLocal,
    _prompts,
    _validate_response,
)


def _shot_payload(**overrides):
    payload = {
        "shot_id": 999,
        "scene_id": "chorus_01",
        "shot_type": "performance",
        "performer_visible": True,
        "video_mode": "t2v",
        "audio_sync_priority": "high",
        "transition_mode": "fresh",
        "match_anchor": None,
        "match_preserve": [],
        "match_change": [],
        "location": "rain-lit rooftop",
        "action": "The performer sings directly to camera.",
        "visual_metaphor": None,
        "shot_size": "medium",
        "camera_angle": "eye level",
        "camera_motion": "slow dolly in",
        "lighting": "blue hour with warm edge light",
        "palette": "deep blue and amber",
        "scene_prompt": "A cinematic medium shot of the referenced performer on a rooftop.",
        "motion_prompt": "Slow dolly in while the performer sings and rain moves in the backlight.",
        "ending_composition": "The performer fills the center third.",
        "next_shot_hint": "Cut on the final beat.",
        "motion_reference_id": None,
        "motion_reference_strength": "none",
    }
    payload.update(overrides)
    return payload


def test_local_director_enforces_shot_number_and_h3_route():
    result = _validate_response(json.dumps(_shot_payload()), "shot", 3)
    assert result["shot_id"] == 3
    assert result["video_mode"] == "multiref"


def test_local_director_accepts_harmless_named_root_wrapper():
    result = _validate_response(json.dumps({"shot_plan": _shot_payload()}), "shot", 5)
    assert result["shot_id"] == 5


def test_local_director_removes_performer_from_fresh_broll_route():
    result = _validate_response(
        json.dumps(_shot_payload(performer_visible=False, shot_type="b-roll")),
        "shot",
        4,
    )
    assert result["video_mode"] == "t2v"


def test_local_director_prompt_keeps_arabic_slice_and_uses_english_h3_direction():
    system, user, _ = _prompts(
        "shot",
        "مرحبا بالعالم",
        8.0,
        "{}",
        "[]",
        "{}",
        2,
        None,
        "exact referenced performer",
        120.0,
        180.0,
    )
    assert "مرحبا بالعالم" in user
    assert "MiniMax H3" in system
    assert "detailed English" in system
    assert "text-bearing props" in system


def test_local_director_rejects_cpu_mode_before_loading_model():
    node = MiniMaxDirectorLocal()
    try:
        node.direct("master", "lyrics", "missing.gguf", n_gpu_layers=0)
    except ValueError as exc:
        assert "CPU mode is disabled" in str(exc)
    else:
        raise AssertionError("CPU mode must not be accepted")
