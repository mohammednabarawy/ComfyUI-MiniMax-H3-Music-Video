import copy
import difflib
import gc
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import tempfile
import uuid
import unicodedata
import sys
from pathlib import Path
import cv2
import numpy as np

import folder_paths
import torch

# Pytest imports a repository root named with hyphens as bare ``__init__``.
# Restore a stable package context so relative imports remain testable outside
# ComfyUI's custom-node loader. Normal ComfyUI loading already sets a package.
if not __package__:
    __package__ = "comfyui_minimax_music_video"
    __path__ = [str(Path(__file__).resolve().parent)]
    sys.modules.setdefault(__package__, sys.modules[__name__])

from .runtime_compat import ensure_minimax_h3_dialogue_tokens

ensure_minimax_h3_dialogue_tokens()


def _snap_h3_frames(seconds, fps=25):
    frames = max(5, round(seconds * fps))
    return frames + (5 - frames % 17) % 17


def _append_synchronized_postroll(images, audio, seconds, frame_rate):
    """Hold the last frame and append silence so speech cannot end at EOF."""
    seconds = max(0.0, float(seconds))
    if seconds == 0:
        return images, audio
    if len(images) == 0 or int(audio["sample_rate"]) <= 0:
        raise ValueError("Post-roll requires video frames and a valid audio sample rate.")

    extra_frames = max(1, round(seconds * int(frame_rate)))
    repeats = (extra_frames,) + (1,) * (images.ndim - 1)
    padded_images = torch.cat((images, images[-1:].repeat(repeats)), dim=0)

    waveform = audio["waveform"]
    extra_samples = max(1, round(seconds * int(audio["sample_rate"])))
    silence = waveform.new_zeros((*waveform.shape[:-1], extra_samples))
    padded_audio = {**audio, "waveform": torch.cat((waveform, silence), dim=-1)}
    return padded_images, padded_audio


def _split_scenes(text):
    scenes = [x.strip() for x in re.split(r"(?m)^\s*---+\s*$", text) if x.strip()]
    return scenes or ["A restrained cinematic performance shot with subtle camera movement."]


def _chunk_ranges(duration, chunk_seconds, bpm=120.0, beat_positions=""):
    try:
        beat_data = json.loads(beat_positions) if beat_positions else {}
        beat_times = sorted({float(value) for value in beat_data.get("beat_times", [])
                             if 0.0 < float(value) < duration})
    except (TypeError, ValueError, json.JSONDecodeError):
        beat_times = []

    if len(beat_times) >= 4:
        bar_boundaries = beat_times[::4]
        chunks = []
        start = 0.0
        while start < duration - 1e-6:
            if duration - start <= chunk_seconds:
                end = duration
            else:
                candidates = [value for value in bar_boundaries
                              if start + 1.0 < value <= start + chunk_seconds]
                end = max(candidates) if candidates else min(duration, start + chunk_seconds)
            chunks.append((start, end - start))
            start = end
        return chunks

    if bpm <= 0:
        bpm = 120.0
    beat_duration = 60.0 / bpm
    bar_duration = beat_duration * 4

    bars = max(1, math.floor(chunk_seconds / bar_duration))
    aligned_length = min(chunk_seconds, bars * bar_duration)

    chunks = []
    start = 0.0
    while start < duration - 1e-6:
        length = min(aligned_length, duration - start)
        chunks.append((start, length))
        start += length
    return chunks


def _safe_name(value):
    value = re.sub(r"[^\w.-]+", "_", value.strip(), flags=re.UNICODE).strip(" ._")
    return (value or "music_video")[:64]


def _audio_fingerprint(audio):
    waveform = audio["waveform"].detach().cpu()
    samples = waveform.shape[-1]
    h = hashlib.sha256(f"{audio['sample_rate']}:{tuple(waveform.shape)}".encode())
    take = min(4096, samples)
    for start in (0, max(0, samples // 2 - take // 2), max(0, samples - take)):
        h.update(waveform[..., start : start + take].contiguous().numpy().tobytes())
    return h.hexdigest()


def _latest_clip(output_dir, prefix):
    stem = Path(prefix).name
    folder = Path(output_dir, Path(prefix).parent)
    matches = list(folder.glob(f"{stem}_*-audio.mp4")) if folder.exists() else []
    return str(max(matches, key=lambda p: p.stat().st_mtime)) if matches else ""


def _words(text):
    lines = [line.strip() for line in str(text or "").splitlines()
             if line.strip() and not re.fullmatch(r"\s*\[[^]]+\]\s*", line)]
    return re.findall(r"[^\W_]+(?:['’][^\W_]+)?", " ".join(lines), re.UNICODE)


def _normalized(word):
    value = "".join(c for c in unicodedata.normalize("NFKD", word.lower())
                    if not unicodedata.combining(c) and c != "ـ")
    return value.translate(str.maketrans({"أ": "ا", "إ": "ا", "آ": "ا", "ى": "ي"}))


def _timed_lyrics(whisper_output, full_lyrics):
    chunks = (whisper_output or {}).get("chunks", [])
    timed = []
    heard_words = []
    spans = []
    for chunk in chunks:
        timestamp = chunk.get("timestamp", [])
        if len(timestamp) != 2 or timestamp[1] <= timestamp[0]:
            continue
        words = _words(chunk.get("text", ""))
        start = len(heard_words)
        heard_words.extend(words)
        spans.append((float(timestamp[0]), float(timestamp[1]), start, len(heard_words), " ".join(words)))

    lyric_words = _words(full_lyrics)
    if not lyric_words or not heard_words:
        return [(start, end, raw) for start, end, _, _, raw in spans if raw]

    matcher = difflib.SequenceMatcher(
        None,
        [_normalized(x) for x in heard_words],
        [_normalized(x) for x in lyric_words],
        autojunk=False,
    )
    anchors = {}
    for block in matcher.get_matching_blocks():
        for offset in range(block.size):
            anchors[block.a + offset] = block.b + offset
    anchors.setdefault(0, 0)
    anchors.setdefault(len(heard_words), len(lyric_words))
    anchors = sorted(anchors.items())

    def map_position(position):
        for (left_x, left_y), (right_x, right_y) in zip(anchors, anchors[1:]):
            if position <= right_x:
                if right_x == left_x:
                    return left_y
                ratio = (position - left_x) / (right_x - left_x)
                return round(left_y + ratio * (right_y - left_y))
        return len(lyric_words)

    for start, end, word_start, word_end, raw in spans:
        lyric_start = max(0, min(len(lyric_words), map_position(word_start)))
        lyric_end = max(lyric_start, min(len(lyric_words), map_position(word_end)))
        corrected = " ".join(lyric_words[lyric_start:lyric_end]).strip()
        timed.append((start, end, corrected or raw))
    return timed


def _lyrics_for_range(start, length, timed, full_lyrics, song_duration):
    end = start + length
    text = " ".join(item[2] for item in timed if item[0] < end and item[1] > start).strip()
    if text:
        return text
    words = _words(full_lyrics)
    if words and song_duration > 0:
        first = math.floor(len(words) * start / song_duration)
        last = max(first + 1, math.ceil(len(words) * end / song_duration))
        return " ".join(words[first:last])
    return "[instrumental passage]"


def _build_plan(audio, job_name, chunk_seconds, max_chunks, width, height, steps,
                base_seed, full_lyrics, whisper_chunks=None, song_bpm=120.0,
                beat_positions=""):
    sample_rate = int(audio["sample_rate"])
    duration = audio["waveform"].shape[-1] / sample_rate
    all_ranges = _chunk_ranges(duration, chunk_seconds, song_bpm, beat_positions)
    timed = _timed_lyrics(whisper_chunks, full_lyrics)
    signature_data = {
        "audio": _audio_fingerprint(audio),
        "duration": duration,
        "chunk_seconds": chunk_seconds,
        "width": width,
        "height": height,
        "steps": steps,
        "base_seed": base_seed,
        "full_lyrics": full_lyrics,
        "timed_lyrics": timed,
        "song_bpm": song_bpm,
        "beat_positions": beat_positions,
    }
    signature = hashlib.sha256(json.dumps(signature_data, sort_keys=True).encode()).hexdigest()[:10]
    run_rel = Path("music_video", f"{_safe_name(job_name)}_{signature}")
    chunks = []
    for index, (start, length) in enumerate(all_ranges):
        lyrics = _lyrics_for_range(start, length, timed, full_lyrics, duration)
        prefix = str(run_rel / "clips" / f"clip_{index:04d}").replace("\\", "/")

        chunks.append({
            "index": index,
            "start_ms": start * 1000.0,
            "duration_ms": length * 1000.0,
            "model_frames": _snap_h3_frames(length),
            "seed": base_seed + index,
            "lyrics": lyrics,
            "prefix": prefix,
        })
    render_chunks = chunks[:max_chunks] if max_chunks else chunks
    return {
        "run_rel": str(run_rel).replace("\\", "/"),
        "song_duration": duration,
        "render_duration": sum(chunk["duration_ms"] for chunk in render_chunks) / 1000.0,
        "fps": 24,
        "width": width,
        "height": height,
        "steps": steps,
        # Keep the complete song timeline available to the batch Director even
        # when MAX CLIPS limits a safe test render. This makes the plan cache
        # reusable when the user changes 1 -> 3 -> 0 clips.
        "director_chunks": chunks,
        "chunks": render_chunks,
    }


def _queue_same_prompt(prompt):
    import server

    prompt_server = server.PromptServer.instance
    def comparable(value):
        return {node_id: {key: item for key, item in node.items() if key != "is_changed"}
                for node_id, node in value.items()}

    wanted = comparable(prompt)
    matches = [value for value in prompt_server.prompt_queue.currently_running.values()
               if value[2] is prompt or comparable(value[2]) == wanted]
    if len(matches) != 1:
        raise RuntimeError("Could not identify this running ComfyUI prompt for the next clip.")
    value = matches[0]
    run_number, _, _, extra_data, outputs_to_execute = value[:5]
    sensitive = value[5] if len(value) > 5 else {}
    number = -prompt_server.number
    prompt_server.number += 1
    next_prompt = copy.deepcopy(value[2])
    for node in next_prompt.values():
        node.pop("is_changed", None)
        if node.get("class_type") == "MiniMaxMusicVideoController":
            inputs = node.setdefault("inputs", {})
            inputs["iteration"] = inputs.get("iteration", 0) + 1
    queued = (number, str(uuid.uuid4()), next_prompt, copy.deepcopy(extra_data),
              list(outputs_to_execute), copy.deepcopy(sensitive))
    prompt_server.prompt_queue.put(queued)
    return run_number


def _release_clip_resources(stage):
    """Release H3/TE/VAE allocations between self-queued clip prompts.

    The music-video workflow deliberately runs one clip per ComfyUI prompt. A
    long song can otherwise leave native CUDA allocator blocks reserved across
    prompts even though the tensors using them are gone. Cleanup is best
    effort: a successfully saved clip must not be lost because cleanup itself
    encountered a stale CUDA error.
    """
    gc.collect()
    if not torch.cuda.is_available():
        return
    try:
        import comfy.model_management as model_management

        model_management.unload_all_models()
        model_management.soft_empty_cache(force=True)
        torch.cuda.reset_peak_memory_stats()
        allocated = torch.cuda.memory_allocated() / (1024 ** 2)
        reserved = torch.cuda.memory_reserved() / (1024 ** 2)
        print(
            f"[MiniMax music video] CUDA cleanup {stage}: "
            f"allocated={allocated:.0f} MiB reserved={reserved:.0f} MiB"
        )
    except Exception as exc:
        print(f"[MiniMax music video] CUDA cleanup {stage} was skipped: {exc}")


def _resolve_ffmpeg():
    for env_var in ("VHS_FORCE_FFMPEG_PATH", "FFMPEG_PATH", "IMAGEIO_FFMPEG_EXE"):
        value = os.environ.get(env_var)
        if value and Path(value).is_file():
            return str(Path(value).resolve())
    system = shutil.which("ffmpeg")
    if system:
        return str(Path(system).resolve())
    try:
        from imageio_ffmpeg import get_ffmpeg_exe
        bundled = get_ffmpeg_exe()
        if bundled and Path(bundled).is_file():
            return str(Path(bundled).resolve())
    except Exception:
        pass
    for parent in list(Path(__file__).resolve().parents)[:8]:
        for relative in ("runtime/ffmpeg/bin/ffmpeg.exe", "ffmpeg/bin/ffmpeg.exe", "bin/ffmpeg.exe"):
            candidate = parent / relative
            if candidate.is_file():
                return str(candidate.resolve())
    return ""


def _ensure_vhs_ffmpeg():
    ffmpeg_bin = _resolve_ffmpeg()
    if not ffmpeg_bin:
        return
    os.environ["VHS_FORCE_FFMPEG_PATH"] = ffmpeg_bin
    os.environ["FFMPEG_PATH"] = ffmpeg_bin
    ffmpeg_dir = str(Path(ffmpeg_bin).parent)
    if ffmpeg_dir not in os.environ.get("PATH", ""):
        os.environ["PATH"] = f"{ffmpeg_dir}{os.pathsep}{os.environ.get('PATH', '')}"
    for module_name, module in list(sys.modules.items()):
        if "videohelpersuite" in module_name and hasattr(module, "ffmpeg_path"):
            module.ffmpeg_path = ffmpeg_bin


def _assemble(plan, audio):
    output_dir = Path(folder_paths.get_output_directory()).resolve()
    run_dir = (output_dir / plan["run_rel"]).resolve()
    if os.path.commonpath((str(output_dir), str(run_dir))) != str(output_dir):
        raise ValueError("Music-video output escaped the ComfyUI output directory.")
    clips = [_latest_clip(output_dir, chunk["prefix"]) for chunk in plan["chunks"]]
    missing = [i + 1 for i, path in enumerate(clips) if not path]
    if missing:
        raise RuntimeError(f"Cannot assemble; missing clips: {missing}")

    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            pass
    if not ffmpeg:
        raise RuntimeError("FFmpeg was not found on PATH.")
    run_dir.mkdir(parents=True, exist_ok=True)
    final_path = run_dir / "final.mp4"
    with tempfile.NamedTemporaryFile("w", suffix=".ffconcat", encoding="utf-8", delete=False, dir=run_dir) as f:
        concat_path = f.name
        f.write("ffconcat version 1.0\n")
        for clip in clips:
            quoted = Path(clip).resolve().as_posix().replace("'", "'\\''")
            f.write(f"file '{quoted}'\n")

    waveform = audio["waveform"].detach().float().cpu()[0]
    channels = waveform.shape[0]
    audio_bytes = waveform.transpose(0, 1).contiguous().numpy().tobytes()
    command = [
        ffmpeg, "-y", "-v", "error", "-f", "concat", "-safe", "0", "-i", concat_path,
        "-f", "f32le", "-ar", str(audio["sample_rate"]), "-ac", str(channels), "-i", "pipe:0",
        "-map", "0:v:0", "-map", "1:a:0", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
        "-t", f"{plan['render_duration']:.6f}", "-movflags", "+faststart", str(final_path),
    ]
    try:
        subprocess.run(command, input=audio_bytes, capture_output=True, check=True)
    except subprocess.CalledProcessError as exc:
        message = exc.stderr.decode(errors="replace")
        raise RuntimeError(f"FFmpeg could not assemble the music video:\n{message}") from exc
    finally:
        Path(concat_path).unlink(missing_ok=True)
    if not final_path.exists() or final_path.stat().st_size == 0:
        raise RuntimeError("FFmpeg finished without creating the final video.")
    return str(final_path)


class MiniMaxMusicVideoController:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "job_name": ("STRING", {"default": "my_music_video"}),
            "chunk_seconds": ("FLOAT", {"default": 10.0, "min": 2.0, "max": 15.0, "step": 0.5}),
            "song_bpm": ("FLOAT", {"default": 120.0, "min": 10.0, "max": 300.0, "step": 0.5}),
            "max_chunks": ("INT", {"default": 0, "min": 0, "max": 10000, "step": 1}),
            "width": ("INT", {"default": 864, "min": 256, "max": 1920, "step": 32}),
            "height": ("INT", {"default": 480, "min": 256, "max": 1088, "step": 32}),
            "steps": ("INT", {"default": 20, "min": 1, "max": 100, "step": 1}),
            "base_seed": ("INT", {"default": 157368968253448, "min": 0, "max": 0xffffffffffffffff}),
            "full_lyrics": ("STRING", {"multiline": True, "default": ""}),
            "iteration": ("INT", {"default": 0, "min": 0, "max": 1000000, "advanced": True}),
        }, "optional": {
            "whisper_chunks": ("WHISPER_CHUNKS",),
            "beat_positions": ("STRING", {"default": ""}),
        }}

    RETURN_TYPES = ("MV_PLAN", "INT", "INT", "INT", "INT", "INT", "FLOAT", "FLOAT", "STRING", "BOOLEAN", "STRING", "INT", "INT", "STRING", "AUDIO", "FLOAT", "STRING", "STRING", "FLOAT")
    RETURN_NAMES = ("plan", "width", "height", "model_frames", "steps", "seed", "start_ms", "duration_ms", "clip_prefix", "generate_clip", "existing_clip", "clip_number", "total_clips", "current_lyrics", "audio_slice", "song_duration", "run_dir", "previous_clip", "duration_seconds")
    FUNCTION = "next_chunk"
    CATEGORY = "MiniMax H3/Music Video"

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return float("nan")

    def next_chunk(self, audio, job_name, chunk_seconds, song_bpm, max_chunks, width, height, steps,
                   base_seed, full_lyrics, iteration=0, whisper_chunks=None, beat_positions=""):
        # On a requeued clip this executes after the previous prompt and its
        # temporary outputs have left scope, making it the strongest point to
        # release stale CUDA allocator blocks before H3 is staged again.
        if int(iteration) > 0:
            _release_clip_resources("before next clip")
        plan = _build_plan(audio, job_name, chunk_seconds, max_chunks, width, height, steps,
                           base_seed, full_lyrics, whisper_chunks, song_bpm, beat_positions)
        if not plan["chunks"]:
            raise ValueError("The song contains no audio samples.")

        output_dir = folder_paths.get_output_directory()
        selected = None
        existing = ""
        for chunk in plan["chunks"]:
            path = _latest_clip(output_dir, chunk["prefix"])
            if not path:
                selected = chunk
                break

        generate = selected is not None
        if selected is None:
            selected = plan["chunks"][-1]
            existing = _latest_clip(output_dir, selected["prefix"])

        previous_clip = ""
        if selected["index"] > 0:
            previous_clip = _latest_clip(output_dir, plan["chunks"][selected["index"] - 1]["prefix"])

        sample_rate = audio["sample_rate"]
        start_sample = int((selected["start_ms"] / 1000.0) * sample_rate)
        length_sample = int((selected["duration_ms"] / 1000.0) * sample_rate)
        waveform = audio["waveform"]
        if waveform.dim() == 3:
            waveform_slice = waveform[:, :, start_sample:start_sample+length_sample]
        else:
            waveform_slice = waveform[:, start_sample:start_sample+length_sample]
        audio_slice = {"waveform": waveform_slice, "sample_rate": sample_rate}

        run_dir = str(Path(output_dir) / plan["run_rel"])

        return (plan, width, height, selected["model_frames"], steps,
                selected["seed"], selected["start_ms"], selected["duration_ms"], selected["prefix"],
                generate, existing, selected["index"] + 1, len(plan["chunks"]), selected["lyrics"],
                audio_slice, plan["song_duration"], run_dir, previous_clip,
                selected["duration_ms"] / 1000.0)


class MiniMaxLyricsSelector:
    """Prefer user-corrected lyrics, otherwise use Whisper's full transcription."""

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "transcription": ("STRING", {"multiline": True}),
            "corrected_lyrics": ("STRING", {"multiline": True}),
        }}

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("lyrics", "source")
    FUNCTION = "select"
    CATEGORY = "MiniMax H3/Music Video"

    def select(self, transcription, corrected_lyrics):
        corrected = str(corrected_lyrics or "").strip()
        transcript = str(transcription or "").strip()
        return (corrected or transcript, "corrected" if corrected else "whisper")


class MiniMaxMusicVideoAdvance:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "plan": ("MV_PLAN",),
            "clip_path": ("STRING",),
            "audio": ("AUDIO",),
        }, "optional": {
            "director_state_json": ("STRING", {"forceInput": True}),
        }, "hidden": {"prompt": "PROMPT"}}

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("status_or_final_path",)
    FUNCTION = "advance"
    CATEGORY = "MiniMax H3/Music Video"
    OUTPUT_NODE = True

    def advance(self, plan, clip_path, audio, prompt, director_state_json=""):
        output_dir = folder_paths.get_output_directory()
        missing = [chunk for chunk in plan["chunks"] if not _latest_clip(output_dir, chunk["prefix"])]
        if missing:
            # Start model offload immediately after the encoded clip exists.
            # The controller repeats cleanup at the start of the next prompt,
            # after this prompt's tensors have been released.
            _release_clip_resources("after saved clip")
            _queue_same_prompt(prompt)
            done = len(plan["chunks"]) - len(missing)
            return {"ui": {"text": [f"Saved clip {done}/{len(plan['chunks'])}; queued the next clip."]},
                    "result": (clip_path,)}

        final_path = _assemble(plan, audio)
        final = Path(final_path)
        subfolder = str(final.parent.relative_to(Path(output_dir))).replace("\\", "/")
        preview = {"filename": final.name, "subfolder": subfolder, "type": "output",
                   "format": "video/h264-mp4", "frame_rate": plan["fps"], "fullpath": final_path}
        return {"ui": {"gifs": [preview], "text": [f"Complete: {len(plan['chunks'])} clips"]},
                "result": (final_path,)}


class MiniMaxMusicVideoSaveClip:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "images": ("IMAGE",),
                "audio": ("AUDIO",),
                "filename_prefix": ("STRING",),
            },
            "optional": {
                "format": ("STRING", {"default": "video/h265-mp4"}),
                "frame_rate": ("INT", {"default": 25, "min": 1, "max": 120}),
                "postroll_seconds": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 2.0, "step": 0.04}),
            },
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("clip_path",)
    FUNCTION = "save"
    CATEGORY = "MiniMax H3/Music Video"
    OUTPUT_NODE = True

    def save(self, images, audio, filename_prefix, format="video/h265-mp4", frame_rate=25, postroll_seconds=0.0):
        import nodes

        _ensure_vhs_ffmpeg()
        images, audio = _append_synchronized_postroll(images, audio, postroll_seconds, frame_rate)
        format_norm = str(format or "video/h265-mp4").strip()
        crf_val = 20 if "h265" in format_norm or "hevc" in format_norm else 19

        combiner = nodes.NODE_CLASS_MAPPINGS["VHS_VideoCombine"]()
        try:
            saved = combiner.combine_video(
                images=images,
                audio=audio,
                frame_rate=frame_rate,
                loop_count=0,
                filename_prefix=filename_prefix,
                format=format_norm,
                pingpong=False,
                save_output=True,
                pix_fmt="yuv420p",
                crf=crf_val,
                save_metadata=False,
                trim_to_audio=True,
                extra_pnginfo={"workflow": {"extra": {"VHS_MetadataImage": False,
                                                         "VHS_KeepIntermediate": False}}},
            )
            output_files = saved["result"][0][1]
            if not output_files or not Path(output_files[-1]).exists():
                raise RuntimeError("Video Helper Suite did not save the generated clip.")
            return {
                "ui": {"gifs": saved.get("ui", {}).get("gifs", [])},
                "result": (output_files[-1],),
            }
        except Exception as exc:
            # If h265 was requested but failed (e.g. missing hevc encoder), fallback to h264
            if "h265" in format_norm or "hevc" in format_norm:
                print(f"[MiniMax H3 SaveClip] H.265 encoding failed ({exc}), falling back to H.264 (AVC)...")
                saved = combiner.combine_video(
                    images=images,
                    audio=audio,
                    frame_rate=frame_rate,
                    loop_count=0,
                    filename_prefix=filename_prefix,
                    format="video/h264-mp4",
                    pingpong=False,
                    save_output=True,
                    pix_fmt="yuv420p",
                    crf=19,
                    save_metadata=False,
                    trim_to_audio=True,
                    extra_pnginfo={"workflow": {"extra": {"VHS_MetadataImage": False,
                                                             "VHS_KeepIntermediate": False}}},
                )
                output_files = saved["result"][0][1]
                if not output_files or not Path(output_files[-1]).exists():
                    raise RuntimeError("Video Helper Suite did not save the generated clip.")
                return {
                    "ui": {"gifs": saved.get("ui", {}).get("gifs", [])},
                    "result": (output_files[-1],),
                }
            raise


class MiniMaxLoadPreviousClipFrame:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "existing_clip": ("STRING", {"forceInput": True}),
            "frame_offset_from_end": ("INT", {"default": 3, "min": 1, "max": 100}),
        }}

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "load_frame"
    CATEGORY = "MiniMax H3/Music Video"

    def load_frame(self, existing_clip, frame_offset_from_end):
        if not existing_clip or not Path(existing_clip).exists():
            dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (dummy,)

        cap = cv2.VideoCapture(existing_clip)
        if not cap.isOpened():
            dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (dummy,)

        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        target_frame = max(0, total_frames - frame_offset_from_end)

        cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
        ret, frame = cap.read()
        cap.release()

        if not ret:
            dummy = torch.zeros((1, 64, 64, 3), dtype=torch.float32)
            return (dummy,)

        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_tensor = torch.from_numpy(frame).float() / 255.0
        frame_tensor = frame_tensor.unsqueeze(0)

        return (frame_tensor,)


from .audio_preprocessor import MiniMaxSongPreprocessor
from .director_batch import MiniMaxDirectorBatch
from .director_cloud import MiniMaxDirectorCloud, register_server_routes
from .director_local import MiniMaxDirectorLocal
from .director_parser import (
    MiniMaxDirectorParser,
    MiniMaxH3ReferenceSelector,
    MiniMaxH3PromptComposer,
    MiniMaxClipDebugSummary,
    MiniMaxTransitionRouter,
    MiniMaxH3ModeRouter,
)
from .director_state import MiniMaxDirectorState
from .motion_loader import MiniMaxMotionLoader

NODE_CLASS_MAPPINGS = {
    "MiniMaxLoadPreviousClipFrame": MiniMaxLoadPreviousClipFrame,
    "MiniMaxMusicVideoController": MiniMaxMusicVideoController,
    "MiniMaxLyricsSelector": MiniMaxLyricsSelector,
    "MiniMaxMusicVideoSaveClip": MiniMaxMusicVideoSaveClip,
    "MiniMaxMusicVideoAdvance": MiniMaxMusicVideoAdvance,
    "MiniMaxSongPreprocessor": MiniMaxSongPreprocessor,
    "MiniMaxDirectorBatch": MiniMaxDirectorBatch,
    "MiniMaxDirectorCloud": MiniMaxDirectorCloud,
    "MiniMaxDirectorLocal": MiniMaxDirectorLocal,
    "MiniMaxDirectorParser": MiniMaxDirectorParser,
    "MiniMaxH3ReferenceSelector": MiniMaxH3ReferenceSelector,
    "MiniMaxH3PromptComposer": MiniMaxH3PromptComposer,
    "MiniMaxClipDebugSummary": MiniMaxClipDebugSummary,
    "MiniMaxTransitionRouter": MiniMaxTransitionRouter,
    "MiniMaxH3ModeRouter": MiniMaxH3ModeRouter,
    "MiniMaxDirectorState": MiniMaxDirectorState,
    "MiniMaxMotionLoader": MiniMaxMotionLoader,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "MiniMaxLoadPreviousClipFrame": "MiniMax Load Previous Clip Frame",
    "MiniMaxMusicVideoController": "MiniMax H3 Music Video Controller",
    "MiniMaxLyricsSelector": "Lyrics: Corrected or Whisper",
    "MiniMaxMusicVideoSaveClip": "Save Current 10s Clip",
    "MiniMaxMusicVideoAdvance": "Save Next Clip / Assemble Music Video",
    "MiniMaxSongPreprocessor": "Song Preprocessor (Cached GPU Whisper + Beats)",
    "MiniMaxDirectorBatch": "Director -> Whole Song Local Qwen Batch",
    "MiniMaxDirectorCloud": "Director -> V19 Cloud Multi-Provider Whole Song",
    "MiniMaxDirectorLocal": "Director → Local Qwen GPU",
    "MiniMaxDirectorParser": "Director → Shot Plan Parser",
    "MiniMaxH3ReferenceSelector": "H3 Deterministic Reference Selector",
    "MiniMaxH3PromptComposer": "H3 Reference-Aware Prompt Composer",
    "MiniMaxClipDebugSummary": "Music Video Clip Debug Summary",
    "MiniMaxTransitionRouter": "Director → H3 Scene Router",
    "MiniMaxH3ModeRouter": "Director → H3 Mode Router",
    "MiniMaxDirectorState": "Director → State Database",
    "MiniMaxMotionLoader": "Motion Reference Loader",
}

WEB_DIRECTORY = "./web"
register_server_routes()


def _self_test():
    assert _snap_h3_frames(5.0) == 141
    assert _snap_h3_frames(10.0) == 260
    assert _chunk_ranges(21.0, 10.0, 120.0) == [(0.0, 10.0), (10.0, 10.0), (20.0, 1.0)]
    assert max(length for _, length in _chunk_ranges(21.0, 10.0, 70.0)) <= 10.0
    phased = json.dumps({"beat_times": [0.37 + 0.5 * index for index in range(50)]})
    phased_ranges = _chunk_ranges(21.0, 10.0, 120.0, phased)
    assert abs(phased_ranges[0][1] - 8.37) < 1e-6
    assert _safe_name(" ../أغنية تجريبية ") == "أغنية_تجريبية"
    timed = {"chunks": [{"timestamp": [0.0, 1.0], "text": "مرحبا بالعالم"}]}
    assert _timed_lyrics(timed, "مرحبا بالعالم") == [(0.0, 1.0, "مرحبا بالعالم")]
    corrected = {"chunks": [
        {"timestamp": [0.0, 2.0], "text": "مرحبا بالعالم"},
        {"timestamp": [2.0, 4.0], "text": "هذا اختبار"},
    ]}
    assert _timed_lyrics(corrected, "مرحبا بالعالم هذا اختبار") == [
        (0.0, 2.0, "مرحبا بالعالم"),
        (2.0, 4.0, "هذا اختبار"),
    ]
    assert MiniMaxLyricsSelector().select("auto", "")[0] == "auto"
    assert MiniMaxLyricsSelector().select("auto", "corrected")[0] == "corrected"
    assert MiniMaxH3ModeRouter().route("i2v", True, "fresh")[:2] == (False, True)
    assert MiniMaxH3ModeRouter().route("multiref", False, "fresh")[:2] == (True, False)
    assert MiniMaxH3ModeRouter().route("multiref", True, "continue")[:2] == (True, False)
    assert MiniMaxTransitionRouter().route("fresh", True)[:3] == (False, True, False)
    assert MiniMaxTransitionRouter().route("fresh", False)[:3] == (True, False, False)
    dummy = torch.zeros((1, 8, 8, 3), dtype=torch.float32)
    selected = MiniMaxH3ReferenceSelector().select(
        dummy, dummy + 1, dummy + 2, dummy + 3, dummy + 4,
        "wide full body", "eye level", True, "fresh", "high",
    )
    assert selected[2] == "full_body" and selected[3:6] == (False, True, False)
    hidden = MiniMaxH3ReferenceSelector().select(
        dummy, dummy + 1, dummy + 2, dummy + 3, dummy + 4,
        "wide", "eye level", False, "fresh", "high",
    )
    assert hidden[2] == "none" and hidden[3:6] == (False, False, False)
    audio_continue = MiniMaxH3ReferenceSelector().select(
        dummy, dummy + 1, dummy + 2, dummy + 3, dummy + 4,
        "medium", "eye level", True, "continue", "high",
    )
    assert audio_continue[3:7] == (False, True, True, False)
    strict_continue = MiniMaxH3ReferenceSelector().select(
        dummy, dummy + 1, dummy + 2, dummy + 3, dummy + 4,
        "medium", "eye level", True, "continue", "low",
    )
    assert strict_continue[3:7] == (True, False, True, False)


if __name__ == "__main__":
    _self_test()
