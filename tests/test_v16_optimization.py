import json
import sys
from pathlib import Path

import numpy as np
import torch


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comfyui_minimax_music_video import _append_synchronized_postroll, _build_plan
from comfyui_minimax_music_video.audio_preprocessor import _fill_beat_grid, _timestamp_chunks
from comfyui_minimax_music_video.director_batch import (
    _batch_cache_key,
    _select_current,
    _storyboard,
)
from comfyui_minimax_music_video.director_parser import (
    MiniMaxH3PromptComposer,
    MiniMaxH3ReferenceSelector,
)


def _audio(seconds=21, sample_rate=100):
    return {
        "sample_rate": sample_rate,
        "waveform": torch.zeros((1, 2, seconds * sample_rate), dtype=torch.float32),
    }


def test_postroll_pads_audio_and_video_by_the_same_duration():
    images = torch.ones((5, 2, 2, 3), dtype=torch.float32)
    audio = _audio(seconds=1, sample_rate=100)

    padded_images, padded_audio = _append_synchronized_postroll(images, audio, 0.4, 25)

    assert padded_images.shape[0] == 15
    assert padded_audio["waveform"].shape[-1] == 140
    assert torch.equal(padded_images[-1], images[-1])
    assert torch.count_nonzero(padded_audio["waveform"][..., -40:]) == 0


def test_render_limit_does_not_truncate_whole_song_director_timeline():
    plan = _build_plan(
        _audio(), "test", 10.0, 1, 864, 480, 20, 100, "one two three",
        whisper_chunks=None, song_bpm=120.0,
    )
    assert len(plan["chunks"]) == 1
    assert len(plan["director_chunks"]) == 3
    assert plan["render_duration"] == 10.0


def test_batch_cache_key_survives_max_clip_change(tmp_path):
    model = tmp_path / "director.gguf"
    model.write_bytes(b"test")
    director_chunks = [{"index": 0, "start_ms": 0.0, "duration_ms": 10000.0, "lyrics": "x"}]
    first = {"song_duration": 10.0, "fps": 24, "director_chunks": director_chunks, "chunks": director_chunks}
    later = {"song_duration": 10.0, "fps": 24, "director_chunks": director_chunks, "chunks": []}
    assert _batch_cache_key(first, "identity", model, 8192, 1500, 0.25, "whole_song") == _batch_cache_key(
        later, "identity", model, 8192, 1500, 0.25, "whole_song"
    )


def test_batch_outputs_selectable_current_shot_and_readable_storyboard():
    shot = {
        "shot_id": 2,
        "video_mode": "multiref",
        "transition_mode": "fresh",
        "scene_prompt": "A rain-lit rooftop.",
        "motion_prompt": "Slow dolly in.",
    }
    payload = {
        "master_plan": {"concept": "Recovery", "narrative_arc": "Darkness to light"},
        "shots": [{"start_sec": 8.0, "end_sec": 16.0, "lyrics": "مرحبا بالعالم", "shot_plan": shot}],
    }
    assert _select_current(payload, 2) == shot
    preview = _storyboard(payload)
    assert "مرحبا بالعالم" in preview
    assert "SHOT 2" in preview
    assert "A rain-lit rooftop" in preview


def test_lightweight_beat_grid_fills_song_without_visualization_tensor():
    beats = _fill_beat_grid(np.asarray([0.37, 0.87, 1.37]), duration=3.0, interval=0.5)
    assert beats[0] == 0.37
    assert beats[-1] == 2.87
    assert np.allclose(np.diff(beats), 0.5)


def test_whisper_timestamp_tokens_become_controller_chunks():
    result = _timestamp_chunks(
        ["<|0.00|>", "مرحبا", "بالعالم", "<|2.00|>", "هذا", "اختبار", "<|4.00|>"],
        "مرحبا بالعالم هذا اختبار",
        5.0,
        "ar",
    )
    assert result["language"] == "ar"
    assert result["chunks"][0]["timestamp"] == [0.0, 2.0]
    assert "مرحبا" in result["chunks"][0]["text"]


def test_h3_prompt_blocks_pseudo_text_on_scene_props():
    plan = {
        "scene_prompt": "A classroom with a chalkboard.",
        "action": "The performer sings.",
        "shot_size": "medium",
        "camera_angle": "eye level",
        "camera_motion": "slow dolly",
        "lighting": "soft daylight",
        "palette": "warm neutral",
        "motion_prompt": "Natural movement.",
        "ending_composition": "Centered performer.",
    }
    outputs = MiniMaxH3PromptComposer().compose(json.dumps(plan), "three_quarter", "multiref")
    assert all("completely blank, unmarked surface" in prompt for prompt in outputs[:5])
    for prompt in outputs[:5]:
        headings = [
            "subject_definitions:", "summary:", "retention_analysis:",
            "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
        ]
        assert [prompt.index(heading) for heading in headings] == sorted(prompt.index(heading) for heading in headings)
        assert all(prompt.count(heading) == 1 for heading in headings)


def test_v17_single_identity_prompt_ignores_reference_clothing_and_relabels_match_anchor():
    plan = {
        "scene_prompt": "A quiet studio performance.",
        "action": "The performer sings softly.",
        "shot_size": "medium close-up",
        "camera_angle": "eye level",
        "camera_motion": "slow push-in",
        "lighting": "soft daylight",
        "palette": "neutral",
        "motion_prompt": "Restrained movement.",
        "ending_composition": "Stable frontal face.",
    }
    outputs = MiniMaxH3PromptComposer().compose(
        json.dumps(plan), "single_face", "h3_multiref_match", single_identity_mode=True
    )
    direct, match, debug = outputs[3], outputs[4], json.loads(outputs[5])
    assert "sole authoritative facial identity" in direct
    assert "supplies no wardrobe" in direct
    assert "<Picture 2> is the previous clip's final frame" in match
    assert "<Picture 3>" not in match
    assert debug["single_identity_mode"] is True


def test_v17_selector_reports_single_authoritative_face():
    image = torch.zeros((1, 16, 16, 3), dtype=torch.float32)
    selected = MiniMaxH3ReferenceSelector().select(
        image, image + 1, image + 2, image + 3, image + 4,
        "medium", "eye level", True, "fresh", "high",
        strict_identity_mode=True,
    )
    assert selected[2] == "single_face"
    assert torch.equal(selected[0], image)
    assert torch.equal(selected[1], image)
    assert "one authoritative face reference" in selected[8]
