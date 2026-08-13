"""
MiniMaxDirectorParser – Parses Shot Director JSON into routable outputs.

Converts the ShotPlan JSON string from MiniMaxDirectorGemini into individual
typed outputs for the H3-only renderer and its scene router.
"""

import json
import logging

logger = logging.getLogger("comfyui_minimax_music_video.director_parser")


class MiniMaxDirectorParser:
    """Parse a ShotPlan JSON string into individual named outputs for routing."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "parse"
    RETURN_TYPES = (
        "STRING",   # scene_prompt
        "STRING",   # motion_prompt
        "STRING",   # transition_mode  ('fresh', 'match_cut', 'continue')
        "STRING",   # video_mode       ('t2v', 'i2v', 'multiref')
        "STRING",   # audio_sync_priority ('low', 'high')
        "BOOLEAN",  # performer_visible
        "STRING",   # shot_size
        "STRING",   # camera_angle
        "STRING",   # camera_motion
        "STRING",   # location
        "STRING",   # match_anchor     (empty string if None)
        "STRING",   # ending_composition
        "STRING",   # next_shot_hint
        "STRING",   # scene_id
        "STRING",   # full_shot_json   (passthrough for state saving)
        "INT",      # shot_id
    )
    RETURN_NAMES = (
        "scene_prompt",
        "motion_prompt",
        "transition_mode",
        "video_mode",
        "audio_sync_priority",
        "performer_visible",
        "shot_size",
        "camera_angle",
        "camera_motion",
        "location",
        "match_anchor",
        "ending_composition",
        "next_shot_hint",
        "scene_id",
        "full_shot_json",
        "shot_id",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_plan_json": ("STRING", {"multiline": True}),
            },
        }

    def parse(self, shot_plan_json: str):
        try:
            plan = json.loads(shot_plan_json)
        except json.JSONDecodeError as exc:
            logger.error("Failed to parse shot_plan_json: %s", exc)
            raise ValueError(f"Invalid ShotPlan JSON: {exc}") from exc

        def _get(key: str, default=""):
            val = plan.get(key, default)
            return val if val is not None else default

        performer_visible = bool(plan.get("performer_visible", True))
        scene_prompt = _get("scene_prompt", _get("keyframe_prompt"))

        return (
            scene_prompt,
            _get("motion_prompt"),
            _get("transition_mode", "fresh"),
            _get("video_mode", "i2v"),
            _get("audio_sync_priority", "high"),
            performer_visible,
            _get("shot_size", "medium"),
            _get("camera_angle", "eye level"),
            _get("camera_motion", "static"),
            _get("location", "unspecified"),
            _get("match_anchor"),
            _get("ending_composition"),
            _get("next_shot_hint"),
            _get("scene_id"),
            shot_plan_json,  # passthrough for state saving
            int(plan.get("shot_id", 0)),
        )


class MiniMaxH3ReferenceSelector:
    """Deterministically choose H3 identity references and an enforced strategy."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "select"
    RETURN_TYPES = ("IMAGE", "IMAGE", "STRING", "BOOLEAN", "BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = (
        "primary_identity", "secondary_identity", "identity_reference_profile",
        "use_fl2va", "use_performer_multiref", "use_previous_frame", "use_broll_previous",
        "render_strategy", "route_description",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "face_reference": ("IMAGE",),
                "three_quarter_reference": ("IMAGE",),
                "profile_reference": ("IMAGE",),
                "waist_up_reference": ("IMAGE",),
                "full_body_reference": ("IMAGE",),
                "shot_size": ("STRING",),
                "camera_angle": ("STRING",),
                "performer_visible": ("BOOLEAN",),
                "transition_mode": ("STRING",),
                "audio_sync_priority": ("STRING",),
            },
            "optional": {
                "strict_identity_mode": ("BOOLEAN", {"default": False}),
            },
        }

    def select(self, face_reference, three_quarter_reference, profile_reference,
               waist_up_reference, full_body_reference, shot_size, camera_angle,
               performer_visible, transition_mode, audio_sync_priority,
               strict_identity_mode=False):
        size = str(shot_size).lower().replace("-", "_").strip()
        angle = str(camera_angle).lower().replace("-", "_").strip()
        transition = str(transition_mode).lower().strip()
        audio_priority = str(audio_sync_priority).lower().strip()

        secondary = three_quarter_reference
        profile = "three_quarter"
        if "profile" in angle or "profile" in size:
            secondary, profile = profile_reference, "profile"
        elif any(token in size for token in ("wide", "full", "long_shot")):
            secondary, profile = full_body_reference, "full_body"
        elif any(token in size for token in ("medium", "waist", "cowboy")):
            secondary, profile = waist_up_reference, "waist_up"
        elif any(token in size for token in ("close", "portrait", "head")):
            secondary, profile = three_quarter_reference, "three_quarter"

        if strict_identity_mode:
            # V17 connects only primary_identity to H3. Returning the primary
            # image here too keeps the output safe if a user later reconnects
            # the optional secondary socket by mistake.
            secondary, profile = face_reference, "single_face"

        if transition == "continue":
            if audio_priority == "low":
                return (face_reference, secondary, profile, True, False, True, False,
                        "h3_i2v_strict_continue", "Strict H3 I2V continuity; current audio is muxed after generation")
            if performer_visible:
                return (face_reference, secondary, profile, False, True, True, False,
                        "h3_multiref_audio_continue", "Audio-aware H3 performer continuation: identity + previous frame + audio")
            return (face_reference, secondary, "none", False, False, True, True,
                    "h3_ref2va_broll_audio_continue", "Audio-aware H3 B-roll continuation: previous frame + audio, no identity refs")

        if performer_visible:
            use_previous = transition == "match_cut"
            strategy = "h3_multiref_match" if use_previous else "h3_multiref_fresh"
            if strict_identity_mode:
                description = ("H3 strict performer: one authoritative face reference + audio"
                               + (" + previous-frame composition anchor" if use_previous else ""))
            else:
                description = (f"H3 MultiRef performer: face + {profile} identity + audio"
                               + (" + previous-frame match anchor" if use_previous else ""))
            return (face_reference, secondary, profile, False, True, use_previous, False,
                    strategy, description)

        if transition == "match_cut":
            return (face_reference, secondary, "none", False, False, True, True,
                    "h3_ref2va_broll_match", "Audio-aware H3 B-roll match: previous frame + audio, no identity refs")

        return (face_reference, secondary, "none", False, False, False, False,
                "h3_ref2va_broll_audio", "H3 Reference-to-Video B-roll from audio + prompt; no identity refs")


class MiniMaxH3PromptComposer:
    """Compose exact six-section H3 full-reference prompts for every renderer route."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "compose"
    RETURN_TYPES = ("STRING", "STRING", "STRING", "STRING", "STRING", "STRING")
    RETURN_NAMES = (
        "broll_audio_prompt", "broll_previous_prompt", "i2v_prompt", "multiref_prompt",
        "multiref_match_prompt", "prompt_debug",
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "shot_plan_json": ("STRING", {"multiline": True}),
                "identity_reference_profile": ("STRING",),
                "render_strategy": ("STRING",),
            },
            "optional": {
                "single_identity_mode": ("BOOLEAN", {"default": False}),
            },
        }

    def compose(self, shot_plan_json, identity_reference_profile, render_strategy,
                single_identity_mode=False):
        try:
            plan = json.loads(shot_plan_json)
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid ShotPlan JSON for H3 prompt composition: {exc}") from exc

        def value(key, default=""):
            item = plan.get(key, default)
            return item if item is not None else default

        scene = value("scene_prompt", value("keyframe_prompt"))
        metaphor = value("visual_metaphor")
        duration = float(value("duration_sec", 5.0) or 5.0)
        performance_mode = str(value("performance_mode", "narrative")).lower()
        lyrics_excerpt = str(value("lyrics_excerpt", "")).strip()
        wardrobe = value("wardrobe", "directed plain logo-free wardrobe")
        beats = value("beat_sequence", [])
        if not isinstance(beats, list):
            beats = [str(beats)]
        beats = [str(item).strip() for item in beats if str(item).strip()]
        max_beats = max(1, min(5, int(duration)))
        if not beats:
            beats = [value("action"), value("motion_prompt"), value("ending_composition")]
        beat_text = " Then ".join(beats[:max_beats])
        style = (
            "The target video uses a premium contemporary cinematic music-video style with physically credible live-action "
            f"movement, {value('lighting')}, and a {value('palette')} palette."
        )
        visible_performance = performance_mode == "performance_lipsync"
        if lyrics_excerpt and lyrics_excerpt != "[instrumental passage]" and lyrics_excerpt[-1] not in ".?!؟!":
            lyrics_excerpt += "."
        performance_action = (
            " When <Audio 1> reaches the phrase "
            f"<d>[Arabic] {lyrics_excerpt}</d>, the visible performer begins mouthing it in the first beat; the lips, jaw, "
            "and mouth move visibly through every word, then the lips close naturally when the phrase ends."
            if visible_performance and lyrics_excerpt and lyrics_excerpt != "[instrumental passage]" else
            " No visible character lip-syncs; any visible mouth stays naturally closed or remains outside clear view."
        )
        detail = (
            f"{style}\n[Shot 1] A {value('shot_size')} composition from {value('camera_angle')} frames {scene}. "
            f"The directed wardrobe is {wardrobe}. {value('camera_motion')}. {value('action')}. "
            + (f"The visual metaphor is {metaphor}. " if metaphor else "")
            + f"Across this {duration:.3f}-second continuous shot, the visible beat sequence is: {beat_text}. "
            f"{value('motion_prompt')}. The shot resolves with {value('ending_composition')}."
            + performance_action
            + " Preserve plausible anatomy and physically coherent cause-and-effect motion while allowing expressive performance. "
              "Do not slim, enlarge, beautify, age, caricature, duplicate, or reinterpret a referenced person. No internal "
              "cuts, subtitles, captions, logos, watermarks, signs, or visible written text. Any chalkboard, screen, poster, "
              "label, newspaper, or book is a completely blank, unmarked surface."
        )

        def full_prompt(subjects, retention, summary, description=detail, soundscape=None, music=None):
            return (
                f"subject_definitions:\n{subjects}\n\n"
                f"summary:\n{summary}\n\n"
                f"retention_analysis:\n{retention}\n\n"
                f"detailed_description:\n{description}\n\n"
                "overall_soundscape:\n"
                + (soundscape or "A soft ambience floor appropriate to the directed location continues beneath restrained "
                   "physical movement sounds; do not duplicate the music or vocals.")
                + "\n\nnon_diegetic_music:\n"
                + (music or "<Audio 1> is directly reused as the complete audience-only score and vocal track for this clip.")
            )

        broll = full_prompt(
            "<Audio 1> is the exact current source-song segment reused as the target clip's final soundtrack.",
            "<Audio 1>: fully_copy - the complete current song segment becomes the target clip's final soundtrack.",
            f"[reference generation + audio reuse] One continuous performer-free music-video shot presents {scene} and follows <Audio 1>'s timing.",
            detail + " Create symbolic cinematic B-roll only: no lead performer, recognizable face, invented singer, or adjacent-shot character.",
        )
        broll_previous = full_prompt(
            "<Picture 1> is the preceding clip's final frame and provides only the opening composition and screen-direction anchor.\n"
            "<Audio 1> is the exact current source-song segment reused as the target clip's final soundtrack.",
            "<Picture 1> ([Shot 1] first-frame transition anchor): partially_preserved - retain only the planned composition and screen direction while changing the scene content.\n"
            "<Audio 1>: fully_copy - the complete current song segment becomes the target clip's final soundtrack.",
            f"[keyframe completion + audio reuse] One continuous performer-free music-video shot begins from <Picture 1>, develops into {scene}, and follows <Audio 1>'s timing.",
            detail + " Continue or match into symbolic cinematic B-roll with no lead performer or recognizable human face.",
        )
        i2v = full_prompt(
            "<Picture 1> is the supplied first frame and defines the target clip's exact opening composition.",
            "<Picture 1> ([Shot 1] first frame): fully_preserved - preserve the visible identity, body proportions, wardrobe, location geometry, lighting direction, object positions, and screen direction at frame zero.",
            f"[keyframe completion] One continuous music-video shot begins literally from <Picture 1> and develops into {scene}.",
            detail + " Begin literally from the supplied first frame; evolve the action without restarting, teleporting, duplicating people, or unexpectedly changing wardrobe.",
            "Use restrained natural ambience; the clean current song segment is muxed after generation.",
            "N/A",
        )
        secondary_role = {
            "profile": "profile angle and head structure",
            "waist_up": "upper-body proportions and medium-shot appearance",
            "full_body": "full-body proportions and wide-shot appearance",
            "three_quarter": "three-quarter facial structure and head angle",
        }.get(str(identity_reference_profile), "secondary identity angle and body proportions")
        if single_identity_mode:
            identity_subjects = (
                "<Subject 1> is the performer whose sole authoritative facial identity comes from <Picture 1>; "
                "<Picture 1> supplies no wardrobe, pose, background, framing, or lighting instructions.\n"
                "<Audio 1> is the exact current source-song segment reused as the target clip's final soundtrack."
            )
        else:
            identity_subjects = (
                "<Subject 1> is one performer whose primary facial identity comes from <Picture 1> and whose "
                f"{secondary_role} comes from <Picture 2>; both show the same person and supply no wardrobe, "
                "background, pose, framing, or lighting instructions.\n<Audio 1> is the exact current source-song segment reused as the target clip's final soundtrack."
            )
        identity_retention = (
            "<Subject 1> (appears in [Shot 1]): partially_preserved - preserve exact broad facial structure, skull and jaw width, eye shape and "
            "spacing, nose geometry, resting mouth, stable distinctive features visible across the references, hairline, "
            "hairstyle, facial-hair pattern when present, skin tone, apparent age, and natural body proportions while "
            "replacing reference clothing with the directed plain, "
            "logo-free wardrobe.\n<Audio 1>: fully_copy - the complete current song segment becomes the target clip's final soundtrack."
        )
        multiref = full_prompt(
            identity_subjects,
            identity_retention,
            f"[reference generation + audio reuse] One continuous music-video shot shows <Subject 1> in {scene} while <Audio 1> supplies the final soundtrack.",
            detail + " Maintain the exact same single performer throughout; never make a lookalike, duplicate, or oversized body. Identity is preserved while the directed wardrobe replaces reference clothing.",
        )
        preserve = ", ".join(value("match_preserve", [])) or value("match_anchor", "composition")
        change = ", ".join(value("match_change", [])) or "environment, action, and surrounding context"
        previous_picture = "<Picture 2>" if single_identity_mode else "<Picture 3>"
        multiref_match = full_prompt(
            identity_subjects + f"\n{previous_picture} is the previous clip's final frame and provides only the opening composition and match-cut anchor; it is not identity evidence.",
            identity_retention + f"\n{previous_picture} ([Shot 1] first-frame match anchor): partially_preserved - retain only {preserve}; change {change}.",
            f"[keyframe completion + reference generation + audio reuse] One continuous music-video shot preserves <Subject 1>, begins from the composition anchor in {previous_picture}, and follows <Audio 1>'s timing.",
            detail + f" Create an intentional match cut. Facial identity comes only from the identity reference input or inputs; {previous_picture} controls composition only.",
        )
        debug = json.dumps({
            "render_strategy": render_strategy,
            "identity_reference_profile": identity_reference_profile,
            "single_identity_mode": bool(single_identity_mode),
            "broll_audio_prompt": broll,
            "broll_previous_prompt": broll_previous,
            "i2v_prompt": i2v,
            "multiref_prompt": multiref,
            "multiref_match_prompt": multiref_match,
        }, ensure_ascii=False, indent=2)
        return broll, broll_previous, i2v, multiref, multiref_match, debug


class MiniMaxClipDebugSummary:
    """Build a concise per-clip timing/lyrics/routing diagnostic for PreviewAny."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "summarize"
    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("summary",)

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "clip_number": ("INT", {"forceInput": True}),
            "start_ms": ("FLOAT", {"forceInput": True}),
            "duration_ms": ("FLOAT", {"forceInput": True}),
            "bpm": ("FLOAT", {"forceInput": True}),
            "current_lyrics": ("STRING", {"forceInput": True}),
            "render_strategy": ("STRING", {"forceInput": True}),
        }}

    def summarize(self, clip_number, start_ms, duration_ms, bpm, current_lyrics, render_strategy):
        start = float(start_ms) / 1000.0
        duration = float(duration_ms) / 1000.0
        end = start + duration
        return (
            f"SHOT {int(clip_number)}\n\n"
            f"start_sec: {start:.3f}\n"
            f"end_sec: {end:.3f}\n"
            f"duration_sec: {duration:.3f}\n"
            f"BPM: {float(bpm):.3f}\n\n"
            f"lyrics:\n{current_lyrics}\n\n"
            f"render_strategy:\n{render_strategy}\n\n"
            f"audio:\n{start:.3f} -> {end:.3f}"
        ,)


class MiniMaxTransitionRouter:
    """Choose an H3-only scene path from transition and performer visibility."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "route"
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "BOOLEAN", "STRING", "STRING")
    RETURN_NAMES = ("use_t2v", "use_multiref", "use_prev_frame", "h3_description", "render_strategy")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "transition_mode": ("STRING",),
                "performer_visible": ("BOOLEAN",),
            },
        }

    def route(self, transition_mode: str, performer_visible: bool):
        transition_mode = transition_mode.lower().strip()

        if transition_mode == "continue":
            return (False, False, True, "CONTINUE: H3 I2V from previous end frame", "h3_i2v_previous")

        if performer_visible:
            use_prev = transition_mode == "match_cut"
            route = "h3_multiref_with_previous" if use_prev else "h3_multiref_direct"
            desc = ("PERFORMER: H3 MultiRef receives four identity references + audio"
                    + (" + previous end frame" if use_prev else " directly"))
            return (False, True, use_prev, desc, route)

        use_prev = transition_mode == "match_cut"
        if use_prev:
            return (False, False, True, "B-ROLL MATCH: H3 I2V from previous end frame", "h3_i2v_previous")
        return (True, False, False, "B-ROLL FRESH: native H3 text-to-video; no portrait references", "h3_t2v_broll")


class MiniMaxH3ModeRouter:
    """Select the H3 FL2VA weights or H3 reference-to-video weights."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "route"
    RETURN_TYPES = ("BOOLEAN", "BOOLEAN", "STRING")
    RETURN_NAMES = ("use_fl2va", "use_multiref", "h3_description")

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "video_mode": ("STRING",),
                "performer_visible": ("BOOLEAN",),
                "transition_mode": ("STRING",),
            },
        }

    def route(self, video_mode: str, performer_visible: bool, transition_mode: str):
        video_mode = video_mode.lower().strip()
        transition_mode = transition_mode.lower().strip()

        if transition_mode == "continue":
            return (True, False, "I2V CONTINUE: previous end frame is literal frame zero")

        if performer_visible:
            desc = "MULTIREF: four character references + audio slice, with previous frame only for match cuts"
            return (False, True, desc)

        if transition_mode == "match_cut":
            return (True, False, "I2V B-ROLL MATCH: previous end frame is literal frame zero")
        return (True, False, "T2V B-ROLL: native H3 generation with no portrait references")


# Node registration mapping
NODE_CLASS_MAPPINGS = {
    "MiniMaxDirectorParser": MiniMaxDirectorParser,
    "MiniMaxH3ReferenceSelector": MiniMaxH3ReferenceSelector,
    "MiniMaxH3PromptComposer": MiniMaxH3PromptComposer,
    "MiniMaxClipDebugSummary": MiniMaxClipDebugSummary,
    "MiniMaxTransitionRouter": MiniMaxTransitionRouter,
    "MiniMaxH3ModeRouter": MiniMaxH3ModeRouter,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxDirectorParser": "Director → Shot Plan Parser",
    "MiniMaxH3ReferenceSelector": "H3 Deterministic Reference Selector",
    "MiniMaxH3PromptComposer": "H3 Reference-Aware Prompt Composer",
    "MiniMaxClipDebugSummary": "Music Video Clip Debug Summary",
    "MiniMaxTransitionRouter": "Director → H3 Scene Router",
    "MiniMaxH3ModeRouter": "Director → H3 Mode Router",
}
