###################################################################################
#
# Adapted from
# https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/layers/tortoise/tokenizer.py
#
########################################################################################################

import os
from typing import *

import torch
from numpy.typing import NDArray
from torch import Tensor
from tokenizers import Tokenizer

from .normalizer import TextNormalizer


class BPETokenizer:
    DEFAULT_VOCAB_FILE = \
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../resource/data/text/tokenizer.json")

    SPACE_TOKEN = "[SPACE]"
    STOP_TOKEN = "[STOP]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, vocab_file: str = DEFAULT_VOCAB_FILE):
        self._normalizer = TextNormalizer.create(normalizer="en_tortoise")

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: `{vocab_file}`")
        
        self._tokenizer = Tokenizer.from_file(vocab_file)

    def normalize(self, txt: str) -> str:
        return self._normalizer.normalize(txt)

    def normalize_and_process(self, txt: str) -> str:
        txt = self._normalizer.normalize(txt)
        txt = txt.replace(" ", self.SPACE_TOKEN)
        return self._tokenizer.encode(txt)

    def tokens(self, txt: str) -> Sequence[int]:
        processed = self.normalize_and_process(txt)
        return processed.tokens

    def encode(self, txt: str) -> Sequence[int]:
        processed = self.normalize_and_process(txt)
        return processed.ids

    def decode(self, seq: Union[Tensor, NDArray]) -> str:
        if isinstance(seq, Tensor):
            seq = seq.cpu().numpy()
        txt = self._tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace(self.SPACE_TOKEN, " ")
        txt = txt.replace(self.STOP_TOKEN, "")
        txt = txt.replace(self.UNK_TOKEN, "")
        return txt


__all__ = ["BPETokenizer"]
