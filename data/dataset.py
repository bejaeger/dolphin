from typing import *

from torch.utils.data import Dataset as TorchDataset


class Dataset(TorchDataset):
    def __init__(self, folder: str) -> None:
        super().__init__()

        self._folder = folder

    # TODO: load prepared data from folder with tokenizer and audio feature found in metadata
