# Contributing

1. Never commit API keys, credential files, local absolute paths, personal media, lyrics you do not have permission to distribute, generated identity media, model weights or output caches.
2. Sanitize shared workflows with `tools/sanitize_workflow.py`.
3. Run `tools/audit_release.py`, Python compilation, tests inside a current ComfyUI environment, and live workflow validation when graph contracts change.
4. Keep commits atomic and use conventional prefixes such as `feat:`, `fix:`, `docs:` and `test:`.
5. Describe the exact verification boundary. A valid graph is not proof of visual quality or successful long-form rendering.
