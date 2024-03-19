from enum import Enum
from typing import *

from .text import Tokenizer
from .audio import AudioFeature


class DatasetPreparers(Enum):
    BPE_SNAC = "bpe_snac"


class DatasetPreparer:
    @classmethod
    def create(
            cls,
            preparer: Union[DatasetPreparers, str],
            tokenizer: Tokenizer,
            audio_feature: AudioFeature,
    ) -> 'DatasetPreparer':
        try:
            preparer = DatasetPreparer(preparer)
            dataloader_class = {
                DatasetPreparers.BPE_SNAC: BPESnacDataset,
            }[preparer]
        except KeyError or ValueError:
            raise ValueError(f"DataLoader `{preparer}` not supported")

        return dataloader_class(tokenizer=tokenizer, audio_feature=audio_feature)


class BPESnacDataset(DatasetPreparer):
    def __init__(
            self,
            tokenizer: Tokenizer,
            audio_feature: AudioFeature,
    ) -> None:
        super().__init__()
        self._tokenizer = tokenizer
        self._audio_feature = audio_feature
