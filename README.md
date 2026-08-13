# ComfyUI MiniMax H3 Music Video

Custom ComfyUI nodes and reusable workflows for generating long-form MiniMax H3 music videos as short, resumable clips.

The main workflow analyzes a song once, creates a whole-song shot plan, renders one short clip per ComfyUI prompt, saves progress after every clip, and concatenates the completed clips with the original clean audio. It is designed for GPUs that cannot render an entire song in one pass.

## Included workflows

| Workflow | Purpose |
| --- | --- |
| `workflows/minimax_h3_any_song_v20.json` | Automatic song analysis, Director planning, identity-aware H3 routing, resumable clip generation, previews, and final assembly |
| `workflows/minimax_h3_reference_identity.json` | Focused reference-to-video workflow with separate face, angled-face, and body-proportion roles |
| `workflows/minimax_h3_image_to_video.json` | Optimized first-frame image-to-video generation |
| `workflows/minimax_h3_text_to_video.json` | Optimized text-to-video generation |

All bundled workflows are templates. Media selections, lyrics, identity descriptions, API credentials, local paths, generated previews, and output history have been removed.

## Main features

- Reads the source-audio duration and splits it into bounded clips.
- Supports detected beat positions and bar-aware boundaries.
- GPU Whisper `large-v3` or `large-v3-turbo`, including Arabic transcription and timestamps.
- Optional corrected-lyrics override.
- One cached whole-song Director plan instead of one LLM load/request per clip.
- Cloud Director presets for Gemini, NVIDIA and OpenCode Zen, plus an optional local GGUF Director.
- Strict JSON schema and semantic validation for shot plans.
- Exact six-section MiniMax H3 full-reference prompt composition.
- Separate performer, B-roll, continuation and match-cut routes.
- Identity references control identity; per-shot direction controls clothing, location, pose and lighting.
- Debug previews for lyrics, treatment, storyboard, shot JSON, final H3 prompt, selected references and decoded frames.
- Disk-backed progress and automatic resume from the first missing clip.
- Explicit CUDA/model cleanup between clips for long 16 GB VRAM runs.
- Optional RTX Video Super Resolution, bypassed by default.

## Requirements

- A current ComfyUI build with native MiniMax H3 support. Development and validation used ComfyUI `0.32.0`.
- Python 3.12 or a Python version supported by your ComfyUI installation.
- FFmpeg available on `PATH`, or `imageio-ffmpeg` installed.
- MiniMax H3 models and LoRA files are not included.

Install missing workflow nodes with ComfyUI Manager. The supplied graphs use:

- [ComfyUI MiniMax H3 Turbo](https://github.com/Larryvrh/ComfyUI-MiniMax-H3-Turbo)
- [ComfyUI Pixaroma](https://github.com/pixaroma/ComfyUI-Pixaroma)
- [ComfyUI VideoHelperSuite](https://github.com/Kosinkadink/ComfyUI-VideoHelperSuite)
- [NVIDIA RTX Nodes](https://github.com/Comfy-Org/Nvidia_RTX_Nodes_ComfyUI), optional

The workflows also use current ComfyUI core H3, EasyCache, scheduler, sampler and VAE nodes.

## Models

The optimized workflows expect these filenames:

| ComfyUI model folder | Filename |
| --- | --- |
| `models/diffusion_models` | `minimax_h3_fl2va_pruned_w4a8_mixed.safetensors` |
| `models/diffusion_models` | `minimax_h3_ref2va_pruned_w4a8_mixed.safetensors` |
| `models/vae` | `minimax_h3_video_vae_int8_convrot.safetensors` |
| `models/vae` | `minimax_h3_audio_vae_fp32.safetensors` |
| `models/text_encoders` | `qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors` |
| `models/loras` | `minimax_h3_turbo_v4_step600_ema.safetensors` |

The experimental H3 model files are available from [Kijai/MiniMax-H3-experimental](https://huggingface.co/Kijai/MiniMax-H3-experimental/tree/main), and the Turbo LoRA is available from [larryvrh/MiniMax-H3-Turbo-Lora](https://huggingface.co/larryvrh/MiniMax-H3-Turbo-Lora/tree/main). Check each model repository’s license and usage terms separately.

## Installation

Clone this repository inside `ComfyUI/custom_nodes`:

```powershell
cd ComfyUI\custom_nodes
git clone https://github.com/mohammednabarawy/ComfyUI-MiniMax-H3-Music-Video.git
..\..\python_embeded\python.exe -m pip install -r .\ComfyUI-MiniMax-H3-Music-Video\requirements.txt
```

For a manual Python environment:

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/mohammednabarawy/ComfyUI-MiniMax-H3-Music-Video.git
python -m pip install -r ComfyUI-MiniMax-H3-Music-Video/requirements.txt
```

Restart ComfyUI, then drag a workflow from `workflows/` onto the canvas.

## Any-song workflow setup

1. Select a song in `SONG`.
2. Select one clear primary face image and optional angled/body references.
3. Optionally paste corrected lyrics.
4. Set `MAX CLIPS` to `1` for the first test.
5. Configure the Cloud Director or use the local Director variant.
6. Keep RTX VSR bypassed until base-resolution generation succeeds.
7. Queue the workflow and inspect the treatment, storyboard, shot plan, generated prompt and decoded-frame previews.
8. Increase `MAX CLIPS` to `3`, then use `0` only after the first clips are satisfactory.

The default music-video graph renders approximately five seconds per clip at 1344×768 and four Turbo steps. Existing clips are detected by their job name and skipped when the workflow resumes.

## Cloud Director credentials and privacy

The repository contains no API credentials or credential-file paths. Configure credentials at runtime using the node’s `credential_file` field, the `api_key` widget, or one of the supported process variables:

- `GEMINI_API_KEY`
- `NVIDIA_API_KEY` or `NGC_API_KEY`
- `ZEN_API_KEY` or `OPENCODE_API_KEY`

Keep credential files outside the repository. The Cloud Director sends the text planning payload—including selected lyrics and the identity description—to the chosen provider. It does not need the raw reference images or full source-audio file. Use the Local Director if the planning text must remain local.

Cached transcripts, storyboards, state and clips are written under the ComfyUI output directory. They are deliberately excluded by this repository’s `.gitignore`.

## 16 GB VRAM launch profile

The tested long-run profile uses native CUDA allocation, no intermediate-node cache and no sampler previews:

```powershell
$env:PYTORCH_ALLOC_CONF='garbage_collection_threshold:0.75'
.\python_embeded\python.exe -s ComfyUI\main.py --windows-standalone-build --enable-manager --use-sage-attention --disable-cuda-malloc --cache-none --preview-method none
```

The custom controller also unloads models and clears inactive CUDA allocations after a saved clip and before the next queued clip. This addresses cumulative memory fragmentation; it does not make a single clip fit if its resolution or duration already exceeds available VRAM.

## Validation

Inside the portable ComfyUI root:

```powershell
.\python_embeded\python.exe -m pytest .\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Music-Video\tests -q
.\python_embeded\python.exe .\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Music-Video\tools\audit_release.py
.\python_embeded\python.exe .\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Music-Video\validate_workflow.py .\ComfyUI\custom_nodes\ComfyUI-MiniMax-H3-Music-Video\workflows\minimax_h3_any_song_v20.json --server http://127.0.0.1:8188
```

Before publishing a modified workflow, run:

```powershell
.\python_embeded\python.exe .\tools\sanitize_workflow.py local_workflow.json workflows\share_safe.json
```

## Verification boundary

The node tests, serialized graph structure, live ComfyUI node contracts, resumable queue logic and per-clip cleanup have been validated. Output identity, cinematography, lip sync and sound quality still depend on models, references, prompts, hardware and provider output; inspect rendered samples before a full-song run.

## License

Code and original workflow composition in this repository are released under the MIT License. ComfyUI, models, LoRAs and third-party custom nodes retain their own licenses.
