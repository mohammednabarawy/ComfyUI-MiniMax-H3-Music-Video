"""Compatibility fixes for MiniMax H3 support in older ComfyUI builds."""

from __future__ import annotations

MINIMAX_EXTRA_TOKENS = {
    "<d>": 151669,
    "</d>": 151670,
    "<|cutoff|>": 151671,
    "<|lyrics_start|>": 151672,
    "<|lyrics_end|>": 151673,
    "<|caption_start|>": 151674,
    "<|caption_end|>": 151675,
}


def patch_minimax_module(module) -> bool:
    """Install the upstream H3 special-token fix when ComfyUI predates it."""
    tokenizer_class = getattr(module, "MiniMaxQwenSDTokenizer", None)
    if tokenizer_class is None:
        return False
    original = tokenizer_class.__init__
    if getattr(original, "_digital_youtuber_h3_tokens", False):
        return False
    names = getattr(getattr(original, "__code__", None), "co_names", ())
    if getattr(module, "MINIMAX_EXTRA_TOKENS", None) == MINIMAX_EXTRA_TOKENS and "MINIMAX_EXTRA_TOKENS" in names:
        return False

    module.MINIMAX_EXTRA_TOKENS = dict(MINIMAX_EXTRA_TOKENS)

    def compatible_init(self, *args, **kwargs):
        original(self, *args, **kwargs)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": list(MINIMAX_EXTRA_TOKENS)}
        )
        self.inv_vocab = {value: key for key, value in self.tokenizer.get_vocab().items()}

    compatible_init._digital_youtuber_h3_tokens = True
    tokenizer_class.__init__ = compatible_init
    return True


def ensure_minimax_h3_dialogue_tokens() -> bool:
    try:
        from comfy.text_encoders import minimax
    except (ImportError, ModuleNotFoundError):
        return False
    return patch_minimax_module(minimax)
