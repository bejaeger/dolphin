from enum import Enum
from typing import *

from data.text.normalizer.cleaners import english_cleaners
from data.text.normalizer.converters import normalize_text


class TextNormalizers(Enum):
    ENGLISH_TORTOISE = "en_tortoise"
    ENGLISH_VERBALIZED = "en_verbalized"


class TextNormalizer:
    def __init__(self, normalize_fn: Callable) -> None:
        self._normalize_fn = normalize_fn

    def normalize(self, text: str) -> str:
        return self._normalize_fn(text).lower()

    @classmethod
    def create(cls, normalizer: str) -> 'TextNormalizer':
        try:
            normalizer = TextNormalizers(normalizer)
            fn = {
                # @see: https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/utils/text/cleaners.py
                TextNormalizers.ENGLISH_TORTOISE: english_cleaners,
                # @see: https://github.com/yl4579/PL-BERT/tree/main
                TextNormalizers.ENGLISH_VERBALIZED: normalize_text,
            }[normalizer]
        except KeyError or ValueError:
            raise ValueError(f"Normalizer `{normalizer}` not supported")

        return cls(normalize_fn=fn)


__all__ = [
    "TextNormalizer",
    "TextNormalizers",
]
