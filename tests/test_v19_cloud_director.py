import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from comfyui_minimax_music_video.director_cloud import (
    _canonical_camera,
    _cloud_cache_key,
    _effective_settings,
    _normalize_bundle,
    _provider_config,
    _provider_order,
)
from comfyui_minimax_music_video.director_parser import MiniMaxH3PromptComposer


def _master():
    return {
        "concept": "A playful lesson becomes sincere connection.",
        "narrative_arc": "Confidence opens into vulnerability.",
        "visual_motifs": ["moving window light"],
        "locations": ["studio", "night street"],
        "wardrobe_progression": ["black tailoring", "soft cream jacket"],
        "color_progression": ["amber", "blue"],
        "performer_usage_plan": "Performance is concentrated in choruses.",
        "section_plans": [{"section": "verse", "approach": "narrative"}],
        "rules": ["No consecutive visual repetition."],
    }


def _shot(shot_id=1, performer=True):
    return {
        "shot_id": shot_id,
        "scene_id": f"scene_{shot_id}",
        "shot_type": "performance" if performer else "conceptual",
        "performer_visible": performer,
        "video_mode": "multiref" if performer else "t2v",
        "audio_sync_priority": "high",
        "transition_mode": "fresh",
        "match_anchor": None,
        "match_preserve": [],
        "match_change": [],
        "location": "amber studio" if shot_id == 1 else "blue street",
        "action": "The performer mouths the song to camera." if performer else "Curtains cross a pool of light.",
        "visual_metaphor": None,
        "shot_size": "medium" if shot_id == 1 else "wide",
        "camera_angle": "eye level",
        "camera_motion": "slow dolly in" if shot_id == 1 else "pan right fast",
        "lighting": "directional amber light",
        "palette": "amber and black",
        "scene_prompt": "The referenced performer stands in an amber studio." if performer else "Empty curtains move over a blue street.",
        "motion_prompt": "One continuous camera move follows the action.",
        "ending_composition": "A stable centered silhouette.",
        "next_shot_hint": "Cut on the beat.",
        "motion_reference_id": None,
        "motion_reference_strength": "none",
        "duration_sec": 99,
        "lyrics_excerpt": "wrong",
        "section_role": "chorus" if performer else "verse",
        "performance_mode": "performance_lipsync" if performer else "conceptual",
        "wardrobe": "plain black tailored jacket",
        "beat_sequence": ["turn", "mouth lyric", "hold", "end"],
    }


def test_provider_order_is_deduplicated_and_respects_primary():
    assert _provider_order("nvidia", "gemini,nvidia,zen") == ["nvidia", "gemini", "zen"]
    assert _provider_order("auto", "gemini,nvidia,zen") == ["gemini", "nvidia", "zen"]


def test_cloud_cache_key_changes_with_provider_without_using_secret(tmp_path):
    plan = {"song_duration": 5.0, "fps": 24, "director_chunks": []}
    one = _cloud_cache_key(plan, "identity", "gemini", "flash", True, "whole_song", 12000, 0.6, "low")
    two = _cloud_cache_key(plan, "identity", "nvidia", "glm", True, "whole_song", 12000, 0.6, "low")
    assert one != two


def test_provider_presets_and_fallback_models_are_provider_specific():
    gemini = _effective_settings("gemini", True, 1, 2.0, "none", 30)
    zen = _effective_settings("zen", True, 1, 2.0, "none", 30)
    assert gemini["max_output_tokens"] == 32768
    assert zen["timeout_sec"] == 360
    assert _provider_config("nvidia", "nvidia", "z-ai/glm-5.2", "") == (
        "https://integrate.api.nvidia.com/v1", "z-ai/glm-5.2"
    )
    # A fallback does not inherit the primary provider's selected model.
    assert _provider_config("zen", "nvidia", "z-ai/glm-5.2", "")[1] == "nemotron-3-ultra-free"


def test_cloud_bundle_uses_authoritative_timing_and_h3_camera_terms():
    chunks = [
        {"index": 0, "start_ms": 0.0, "duration_ms": 5000.0, "lyrics": "مرحبا بالعالم"},
        {"index": 1, "start_ms": 5000.0, "duration_ms": 4000.0, "lyrics": "والبشر عندي تلامذه"},
    ]
    payload = {"master_plan": _master(), "shot_plans": [_shot(1, True), _shot(2, False)]}
    bundle = _normalize_bundle(json.dumps(payload, ensure_ascii=False), chunks)
    first, second = bundle["shot_plans"]
    assert first["duration_sec"] == 5.0
    assert first["lyrics_excerpt"] == "مرحبا بالعالم"
    assert first["camera_motion"] == "the camera pushes in at slow speed"
    assert second["camera_motion"] == "the camera pans right at fast speed"
    assert len(first["beat_sequence"]) <= 5


def test_exact_h3_prompt_contract_and_audio_markers():
    shot = _shot(1, True)
    shot["duration_sec"] = 5.0
    shot["lyrics_excerpt"] = "مرحبا بالعالم"
    prompts = MiniMaxH3PromptComposer().compose(
        json.dumps(shot, ensure_ascii=False), "three_quarter", "h3_multiref_match", False
    )
    direct, match = prompts[3], prompts[4]
    headings = [
        "subject_definitions:", "summary:", "retention_analysis:",
        "detailed_description:", "overall_soundscape:", "non_diegetic_music:",
    ]
    for prompt in prompts[:5]:
        assert [prompt.index(heading) for heading in headings] == sorted(prompt.index(heading) for heading in headings)
        detail = prompt.split("detailed_description:\n", 1)[1].split("\n\noverall_soundscape:", 1)[0]
        assert "\n\n" not in detail
        assert "<Object " not in prompt
        assert "<Audio 1>: fully_preserved" not in prompt
    assert "<Subject 1>" in direct
    assert "<Picture 1>" in direct and "<Picture 2>" in direct
    assert direct.count("مرحبا بالعالم") == 1
    assert "<d>[Arabic] مرحبا بالعالم.</d>" in direct
    assert "<Picture 3> is the previous clip's final frame" in match


def test_camera_falls_back_to_static_without_unsupported_adverbs():
    assert _canonical_camera("very gentle crane move") == "the camera uses a static shot"
