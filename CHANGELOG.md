## 1.0.2 - 2026-09-10

- Backport the official MiniMax H3 dialogue special tokens for older ComfyUI runtimes.
- Generate and save H3 clips consistently at 25 fps, including the H.264 fallback.
- Resolve bundled FFmpeg for VideoHelperSuite before saving clips.

# Changelog

## 1.0.1 - 2026-08-13

- Add the Windows H3 launch workaround for DynamicVRAM `HostBuffer.read_file_slice failed` CUDA OOMs.
- Document conservative headroom and legacy-loading fallback steps.

## 1.0.0 - 2026-08-13

- Initial public release.
- Resumable any-song MiniMax H3 music-video controller.
- Cached Whisper/beat preprocessing and whole-song Director planning.
- Cloud and local Director implementations.
- Identity-aware H3 prompt composition and routing.
- Per-clip CUDA/model cleanup for long renders.
- Four sanitized workflows and release-audit tools.
