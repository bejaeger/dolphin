from enum import Enum
from typing import *

from torch.utils.data import DataLoader as TorchDataLoader

from .text import Tokenizer


class DataLoaders(Enum):
    SIMPLE = "simple"


# TODO NEXT