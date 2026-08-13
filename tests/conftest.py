"""Load the custom-node package under a stable import name for pytest."""

from __future__ import annotations

import importlib.util
import os
import sys
from pathlib import Path


PACKAGE_ROOT = Path(__file__).resolve().parents[1]
configured_root = os.environ.get("COMFYUI_ROOT")
COMFYUI_ROOT = Path(configured_root).resolve() if configured_root else PACKAGE_ROOT.parent.parent.resolve()
if str(COMFYUI_ROOT) not in sys.path:
    sys.path.insert(0, str(COMFYUI_ROOT))

PACKAGE_NAME = "comfyui_minimax_music_video"
if PACKAGE_NAME not in sys.modules:
    spec = importlib.util.spec_from_file_location(
        PACKAGE_NAME,
        PACKAGE_ROOT / "__init__.py",
        submodule_search_locations=[str(PACKAGE_ROOT)],
    )
    if spec is None or spec.loader is None:
        raise RuntimeError("Could not create the custom-node test module specification")
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE_NAME] = module
    spec.loader.exec_module(module)
