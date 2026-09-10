from types import SimpleNamespace

from comfyui_minimax_music_video.runtime_compat import MINIMAX_EXTRA_TOKENS, patch_minimax_module


def test_old_comfyui_tokenizer_gets_h3_dialogue_tokens():
    class Tokenizer:
        def __init__(self):
            self.added = []

        def add_special_tokens(self, payload):
            self.added.extend(payload["additional_special_tokens"])

        def get_vocab(self):
            return {token: token_id for token, token_id in MINIMAX_EXTRA_TOKENS.items()}

    class OldMiniMaxTokenizer:
        def __init__(self):
            self.tokenizer = Tokenizer()
            self.inv_vocab = {}

    module = SimpleNamespace(MiniMaxQwenSDTokenizer=OldMiniMaxTokenizer)
    assert patch_minimax_module(module) is True
    instance = OldMiniMaxTokenizer()
    assert instance.tokenizer.added == list(MINIMAX_EXTRA_TOKENS)
    assert instance.inv_vocab[151669] == "<d>"
    assert instance.inv_vocab[151670] == "</d>"
    assert patch_minimax_module(module) is False
