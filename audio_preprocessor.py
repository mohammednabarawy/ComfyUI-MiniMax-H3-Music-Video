"""Cached, VRAM-safe song analysis for the MiniMax H3 music-video workflow."""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch


logger = logging.getLogger("comfyui_minimax_music_video.audio_preprocessor")
_CACHE_VERSION = 2


def _audio_fingerprint(audio: Dict[str, Any]) -> str:
    waveform = audio["waveform"].detach().cpu()
    samples = waveform.shape[-1]
    digest = hashlib.sha256(f"{audio['sample_rate']}:{tuple(waveform.shape)}".encode())
    take = min(4096, samples)
    for start in (0, max(0, samples // 2 - take // 2), max(0, samples - take)):
        digest.update(waveform[..., start : start + take].contiguous().numpy().tobytes())
    return digest.hexdigest()


def _mono_numpy(audio: Dict[str, Any]) -> np.ndarray:
    waveform = audio["waveform"].detach().float().cpu().numpy()
    if waveform.ndim == 3:
        waveform = waveform[0]
    if waveform.ndim == 2:
        waveform = waveform.mean(axis=0)
    return np.ascontiguousarray(waveform, dtype=np.float32)


def _fill_beat_grid(beat_times: np.ndarray, duration: float, interval: float) -> np.ndarray:
    if beat_times.size < 2 or interval <= 0:
        return beat_times
    filled = list(float(value) for value in beat_times)
    current = filled[0] - interval
    while current > 0:
        filled.append(current)
        current -= interval
    current = filled[-1] + interval
    while current < duration:
        filled.append(current)
        current += interval
    ordered = sorted(filled)
    result = [ordered[0]]
    for right in ordered[1:]:
        left = result[-1]
        while right - left > interval * 1.5:
            left += interval
            if left < right:
                result.append(left)
        result.append(right)
    return np.asarray(sorted(set(round(value, 9) for value in result)), dtype=np.float64)


def _analyze_beats(audio: Dict[str, Any], half_time: bool, beat_offset_ms: int):
    import librosa

    source_sr = int(audio["sample_rate"])
    waveform = _mono_numpy(audio)
    duration = waveform.shape[-1] / source_sr
    analysis_sr = min(source_sr, 22050)
    if source_sr != analysis_sr:
        analysis_waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=analysis_sr)
    else:
        analysis_waveform = waveform

    onset_env = librosa.onset.onset_strength(y=analysis_waveform, sr=analysis_sr)
    onset_bpm, beat_frames = librosa.beat.beat_track(
        onset_envelope=onset_env,
        sr=analysis_sr,
        units="frames",
    )
    onset_bpm = float(np.asarray(onset_bpm).reshape(-1)[0])
    beat_times = librosa.frames_to_time(beat_frames, sr=analysis_sr)
    if beat_times.size >= 2:
        median_interval = float(np.median(np.diff(beat_times)))
        bpm = 60.0 / median_interval if median_interval > 0 else onset_bpm
    else:
        median_interval = 60.0 / max(onset_bpm, 120.0)
        bpm = onset_bpm or 120.0

    if half_time:
        bpm /= 2.0
        median_interval *= 2.0
        beat_times = beat_times[::2]
    beat_times = _fill_beat_grid(beat_times, duration, median_interval)
    if beat_offset_ms:
        beat_times = np.clip(beat_times + beat_offset_ms / 1000.0, 0.0, duration)

    payload = {
        "bpm": float(bpm),
        "bpm_source": "median_detected_beat_intervals",
        "beat_times": beat_times.tolist(),
        "num_beats": int(beat_times.size),
        "sample_rate": source_sr,
        "analysis_sample_rate": analysis_sr,
        "audio_duration": float(duration),
    }
    return float(bpm), json.dumps(payload, ensure_ascii=False, indent=2)


def _timestamp_chunks(tokens: list[str], transcription: str, duration: float, language: str):
    timestamp_tokens = []
    for index, token in enumerate(tokens):
        if token.startswith("<|") and token.endswith("|>"):
            raw = token[2:-2]
            try:
                value = float(raw)
            except ValueError:
                continue
            if 0 <= value <= duration:
                timestamp_tokens.append((index, value))

    chunks = []
    for (start_pos, start), (end_pos, end) in zip(timestamp_tokens, timestamp_tokens[1:]):
        text = " ".join(
            token for token in tokens[start_pos + 1 : end_pos]
            if not (token.startswith("<|") and token.endswith("|>"))
        ).strip()
        if text and end > start:
            chunks.append({"text": text, "timestamp": [start, end]})
    if timestamp_tokens:
        start_pos, start = timestamp_tokens[-1]
        text = " ".join(
            token for token in tokens[start_pos + 1 :]
            if not (token.startswith("<|") and token.endswith("|>"))
        ).strip()
        if text:
            previous_duration = (
                chunks[-1]["timestamp"][1] - chunks[-1]["timestamp"][0]
                if chunks else duration - start
            )
            end = min(duration, start + max(0.1, previous_duration))
            if end > start:
                chunks.append({"text": text, "timestamp": [start, end]})
    return {"text": transcription, "chunks": chunks, "language": language}


def _transcribe(audio: Dict[str, Any], model_size: str, language: str):
    import folder_paths
    import librosa
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_dir = Path(folder_paths.models_dir) / "whisper" / f"whisper-{model_size}"
    if not model_dir.is_dir():
        raise FileNotFoundError(
            f"Whisper model is missing: {model_dir}. Install the complete {model_size} model first."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("GPU Whisper preprocessing requires CUDA; CPU fallback is disabled.")

    source_sr = int(audio["sample_rate"])
    waveform = _mono_numpy(audio)
    waveform = librosa.resample(waveform, orig_sr=source_sr, target_sr=16000)
    duration = waveform.shape[-1] / 16000.0
    processor = WhisperProcessor.from_pretrained(str(model_dir), local_files_only=True)
    try:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
        ).to("cuda")
    except TypeError:
        model = WhisperForConditionalGeneration.from_pretrained(
            str(model_dir), torch_dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True
        ).to("cuda")
    model.eval()
    model.requires_grad_(False)

    all_tokens: list[str] = []
    all_text: list[str] = []
    chunk_samples = 30 * 16000
    try:
        for chunk_start in range(0, waveform.shape[-1], chunk_samples):
            chunk_end = min(chunk_start + chunk_samples, waveform.shape[-1])
            chunk = waveform[chunk_start:chunk_end]
            offset = chunk_start / 16000.0
            features = processor(chunk, sampling_rate=16000, return_tensors="pt").input_features
            features = features.to(device="cuda", dtype=model.dtype)
            max_length = getattr(model.config, "max_length", None) or 448
            with torch.inference_mode():
                ids = model.generate(
                    features,
                    task="transcribe",
                    language=None if language == "auto" else language,
                    return_timestamps=True,
                    no_repeat_ngram_size=3,
                    num_beams=5,
                    length_penalty=1.0,
                    max_length=max_length,
                )
            tokens = processor.tokenizer.convert_ids_to_tokens(ids[0])
            for token in tokens:
                if token.startswith("<|") and token.endswith("|>"):
                    raw = token[2:-2]
                    try:
                        token = f"<|{float(raw) + offset:.2f}|>"
                    except ValueError:
                        pass
                all_tokens.append(token)
            all_text.append(
                processor.batch_decode(ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
            )
            del ids, features
        transcription = " ".join(all_text).strip()
        detected_language = language if language != "auto" else "unknown"
        return transcription, _timestamp_chunks(all_tokens, transcription, duration, detected_language)
    finally:
        del model
        del processor
        gc.collect()
        torch.cuda.empty_cache()


def _cache_path(fingerprint: str, model_size: str, language: str, half_time: bool, offset: int) -> Path:
    import folder_paths

    token = hashlib.sha256(
        f"{_CACHE_VERSION}:{fingerprint}:{model_size}:{language}:{half_time}:{offset}".encode()
    ).hexdigest()[:16]
    return Path(folder_paths.get_output_directory()) / "music_video" / "_analysis_cache" / f"{token}.json"


class MiniMaxSongPreprocessor:
    """Analyze beats and Arabic lyrics once, then reuse a disk cache on every queued clip."""

    CATEGORY = "MiniMax/MusicVideo"
    FUNCTION = "analyze"
    RETURN_TYPES = ("AUDIO", "FLOAT", "STRING", "STRING", "WHISPER_CHUNKS", "STRING")
    RETURN_NAMES = (
        "audio", "bpm", "beat_positions", "transcription", "whisper_chunks", "analysis_status"
    )

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "audio": ("AUDIO",),
            "whisper_model": (["large-v3", "large-v3-turbo"], {"default": "large-v3"}),
            "language": (["ar", "auto", "en"], {"default": "ar"}),
            "half_time": ("BOOLEAN", {"default": False}),
            "beat_offset_ms": ("INT", {"default": 0, "min": -1000, "max": 1000}),
            "force_reanalyze": ("BOOLEAN", {"default": False}),
        }}

    def analyze(self, audio, whisper_model="large-v3", language="ar", half_time=False,
                beat_offset_ms=0, force_reanalyze=False):
        fingerprint = _audio_fingerprint(audio)
        cache_path = _cache_path(fingerprint, whisper_model, language, half_time, beat_offset_ms)
        if cache_path.is_file() and not force_reanalyze:
            data = json.loads(cache_path.read_text(encoding="utf-8"))
            status = (
                f"CACHE HIT | {data['duration_seconds']:.2f}s song | {data['bpm']:.2f} BPM | "
                f"{len(data['whisper_chunks'].get('chunks', []))} timed lyric chunks | no model loaded"
            )
            return (
                audio, float(data["bpm"]), data["beat_positions"], data["transcription"],
                data["whisper_chunks"], status,
            )

        started = time.perf_counter()
        bpm_started = time.perf_counter()
        bpm, beat_positions = _analyze_beats(audio, half_time, int(beat_offset_ms))
        bpm_seconds = time.perf_counter() - bpm_started
        whisper_started = time.perf_counter()
        transcription, whisper_chunks = _transcribe(audio, whisper_model, language)
        whisper_seconds = time.perf_counter() - whisper_started
        duration = audio["waveform"].shape[-1] / float(audio["sample_rate"])
        payload = {
            "cache_version": _CACHE_VERSION,
            "fingerprint": fingerprint,
            "duration_seconds": duration,
            "bpm": bpm,
            "beat_positions": beat_positions,
            "transcription": transcription,
            "whisper_chunks": whisper_chunks,
            "whisper_model": whisper_model,
            "language": language,
        }
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        temp_path = Path(f"{cache_path}.tmp")
        temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        os.replace(temp_path, cache_path)
        total = time.perf_counter() - started
        status = (
            f"ANALYZED + CACHED | total {total:.1f}s (beats {bpm_seconds:.1f}s, Whisper {whisper_seconds:.1f}s) | "
            f"{duration:.2f}s song | {bpm:.2f} BPM | {len(whisper_chunks.get('chunks', []))} timed lyric chunks | "
            "Whisper GPU model released before Director/H3"
        )
        logger.info(status)
        return audio, bpm, beat_positions, transcription, whisper_chunks, status
