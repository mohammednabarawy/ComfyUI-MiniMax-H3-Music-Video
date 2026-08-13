import importlib.util
import os
from pathlib import Path

import torch


if comfyui_root := os.environ.get("COMFYUI_ROOT"):
    custom_nodes_root = Path(comfyui_root) / "custom_nodes"
else:
    custom_nodes_root = Path(__file__).resolve().parents[2]

PLUGIN = custom_nodes_root / "ComfyUI-MiniMax-H3-Turbo" / "__init__.py"


def _load_plugin():
    spec = importlib.util.spec_from_file_location("minimax_h3_turbo_local", PLUGIN)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_pruned_adaln_grid_includes_reference_audio_timestep():
    plugin = _load_plugin()
    timestep = torch.tensor([1000.0])

    visual_only = plugin._unique_t(timestep, 12.0, 3.0, True, False)
    visual_and_audio = plugin._unique_t(timestep, 12.0, 3.0, True, True)

    assert visual_only == [0.0, 0.999]
    assert visual_and_audio == [0.0, 0.999, 1.0]


def test_pruned_adaln_grid_matches_current_comfyui_order_mid_schedule():
    plugin = _load_plugin()
    timestep = torch.tensor([500.0])
    actual = plugin._unique_t(timestep, 12.0, 3.0, True, True)

    sigma_v = 0.5
    t_v = 1.0 - sigma_v
    t_a = 1.0 - plugin._time_shift_sigma(sigma_v, 12.0, 3.0)
    expected = sorted({t_v, t_a, max(t_v, 0.999), max(t_a, 1.0)})

    assert actual == expected
