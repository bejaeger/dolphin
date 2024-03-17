###################################################################################
#
# Adapted from
# https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/layers/tortoise/tokenizer.py
#
########################################################################################################

import os
from enum import Enum
from typing import *

import tiktoken
from numpy.typing import NDArray
from torch import Tensor
from tokenizers import Tokenizer as HFTokenizer

from .normalizer import TextNormalizer, TextNormalizers


class Tokenizers(Enum):
    BPE = "bpe"
    GPT2 = "gpt2"


class Tokenizer:
    SPACE_TOKEN = "[SPACE]"
    STOP_TOKEN = "[STOP]"
    UNK_TOKEN = "[UNK]"

    def __init__(self, normalizer: Union[TextNormalizers, str]) -> None:
        self._normalizer = TextNormalizer.create(normalizer=normalizer)

    def normalize(self, txt: str) -> str:
        return self._normalizer.normalize(txt)

    def encode(self, txt: str) -> str:
        txt = self.normalize(txt)
        return self._encode(txt)

    def decode(self, seq: Union[Tensor, NDArray]) -> str:
        return self._decode(seq)
    
    def tokens(self, txt: str) -> Sequence[int]:
        txt = self.normalize(txt)
        return self._tokens(txt)

    def _tokens(self, txt: str) -> Sequence[int]:
        raise NotImplementedError("Subclasses must implement this method")

    def _encode(self, txt: str) -> str:
        raise NotImplementedError("Subclasses must implement this method")

    def _decode(self, seq: Union[Tensor, NDArray]) -> str:
        raise NotImplementedError("Subclasses must implement this method")


    @classmethod
    def create(cls, tokenizer: Union[Tokenizers, str], **kwargs: Any) -> 'Tokenizer':
        try:
            tokenizer = Tokenizers(tokenizer)
            tokenizer_class = {
                Tokenizers.BPE: BPETokenizer,
                Tokenizers.GPT2: GPT2Tokenizer,
            }[tokenizer]
        except KeyError or ValueError:
            raise ValueError(f"Tokenizer `{tokenizer}` not supported")
        
        return tokenizer_class(**kwargs)


class GPT2Tokenizer(Tokenizer):
    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        self._encoder = tiktoken.get_encoding("gpt2")

    def _encode(self, txt: str) -> str:
        return self._encoder.encode(txt, allowed_special={"<|endoftext|>"})

    def _decode(self, seq: Union[Tensor, NDArray]) -> str:
        return self._encoder.decode(seq)

    def _tokens(self, txt: str) -> Sequence[int]:
        tokens = []
        for embed_index in self.encode(txt):
            tokens.append(self._encoder.decode([embed_index]))
        return tokens


class BPETokenizer(Tokenizer):
    DEFAULT_VOCAB_FILE = \
        os.path.join(
            os.path.dirname(os.path.realpath(__file__)), "../../resource/data/text/tokenizer.json")

    def __init__(self, vocab_file: str = DEFAULT_VOCAB_FILE, **kwargs: Any) -> None:
        super().__init__(**kwargs)

        if not os.path.exists(vocab_file):
            raise FileNotFoundError(f"Vocab file not found: `{vocab_file}`")

        self._tokenizer = HFTokenizer.from_file(vocab_file)

    def _encode(self, txt: str) -> Sequence[int]:
        txt = txt.replace(" ", self.SPACE_TOKEN)
        tokens = self._tokenizer.encode(txt)
        return tokens.ids

    def _tokens(self, txt: str) -> Sequence[int]:
        txt = txt.replace(" ", self.SPACE_TOKEN)
        tokens = self._tokenizer.encode(txt)
        return tokens.tokens

    def _decode(self, seq: Union[Tensor, NDArray]) -> str:
        if isinstance(seq, Tensor):
            seq = seq.cpu().numpy()
        txt = self._tokenizer.decode(seq, skip_special_tokens=False).replace(" ", "")
        txt = txt.replace(self.SPACE_TOKEN, " ")
        txt = txt.replace(self.STOP_TOKEN, "")
        txt = txt.replace(self.UNK_TOKEN, "")
        return txt


__all__ = [
    "Tokenizer",
    "Tokenizers",
]
