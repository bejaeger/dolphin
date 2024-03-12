#################################################################
#
# Adapted from
# https://github.com/coqui-ai/TTS/blob/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/utils/text/cleaners.py
#
#####################################################################

from enum import Enum
from typing import *

from data.text.normalizer.cleaners import english_cleaners


class TextNormalizers(Enum):
    ENGLISH_TORTOISE = "en_tortoise"
    ENGLISH_VERBALIZED = "en_verbalized"

class TextNormalizer:
    def __init__(self, normalize_fn: Callable) -> None:
        self._normalize_fn = normalize_fn

    def normalize(self, text: str) -> str:
        return self._normalize_fn(text)

    @classmethod
    def create(cls, normalizer: str) -> 'TextNormalizer':
        try:
            normalizer = TextNormalizers(normalizer)
        except ValueError:
            raise ValueError(f"Normalizer `{normalizer}` not supported")
        
        if normalizer is TextNormalizers.ENGLISH_TORTOISE: 
            fn = english_cleaners
        elif normalizer is TextNormalizers.ENGLISH_VERBALIZED:
            # TODO NEXT: Implement this from https://github.com/coqui-ai/TTS/tree/dbf1a08a0d4e47fdad6172e433eeb34bc6b13b4e/TTS/tts/utils/text
            raise NotImplementedError(f"Normalizer `{normalizer}` not supported")
        
        return cls(normalize_fn=fn)


__all__ = ["TextNormalizer"]
