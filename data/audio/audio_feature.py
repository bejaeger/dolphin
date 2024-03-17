import logging
import os
from enum import Enum
from typing import *

import librosa
import numpy as np
import soundfile
from einops import rearrange
from numpy.typing import NDArray

from model.codec.snac.snac import SNAC


class AudioFeatures(Enum):
    SNAC_CODEC = "snac_codec"


class AudioFeature:
    def __init__(self, out_sample_rate: int, in_sample_rate: Optional[int] = None) -> None:
        super().__init__()
        self.in_sample_rate = in_sample_rate
        self.out_sample_rate = out_sample_rate

        self._give_warning = True

    def _load_audio(self, audio_path: str) -> Tuple[NDArray, int]:
        pcm, sample_rate = soundfile.read(audio_path)

        if self.in_sample_rate is not None and sample_rate != self.in_sample_rate:
            if self._give_warning:
                logging.warn(f"Resample audio from `{sample_rate}` to `{self.in_sample_rate}`. This warning will only be shown once.")
                self._give_warning = False
            pcm = librosa.resample(np.array(pcm), orig_sr=sample_rate, target_sr=self.in_sample_rate)

        return pcm, sample_rate

    def encode(self, audio_path: Optional[str], pcm: Optional[NDArray] = None, sample_rate: Optional[int] = None) -> NDArray:
        if pcm is None and sample_rate is None:
            pcm, sample_rate = self._load_audio(audio_path)
        elif audio_path is None:
            raise ValueError("Either `audio_path` or `pcm` and `sample_rate` must be provided")
        
        return self._encode(pcm=pcm, sample_rate=sample_rate)
    
    def decode(self, **kwargs) -> NDArray:
        return self._decode(**kwargs)
    
    def _encode(self, pcm: NDArray, sample_rate: int) -> NDArray:
        raise NotImplementedError("Subclass must implement this method")

    @classmethod
    def create(cls, feature: Union[AudioFeatures, str], **kwargs: Any) -> 'AudioFeature':
        try:
            feature = AudioFeatures(feature)
            feature_class = {
                AudioFeatures.SNAC_CODEC: SnacCodecAudioFeature,
            }[feature]
        except KeyError or ValueError:
            raise ValueError(f"AudioFeature `{feature}` not supported")

        return feature_class(**kwargs)


class SnacCodecAudioFeature(AudioFeature):
    HIERARCHY_DOWNSAMPLE_RATE = [4, 2, 1]
    NUM_HIERARCHIES = len(HIERARCHY_DOWNSAMPLE_RATE)

    IN_SAMPLE_RATE = None
    OUT_SAMPLE_RATE = 24000

    def __init__(
            self,
            model_path: str,
            device: str = "cpu",
            config_path: Optional[str] = None) -> None:
        super().__init__(in_sample_rate=self.IN_SAMPLE_RATE, out_sample_rate=self.OUT_SAMPLE_RATE)

        if config_path is None:
            config_path = os.path.join(os.path.dirname(model_path), "config.json")

        self._device = device
        self._model = SNAC.from_pretrained_local(model_path=model_path, config_path=config_path)
        self._model.eval()
        self._model.to(self._device)

    def _encode(self, pcm: NDArray, sample_rate: int) -> Sequence[NDArray]:
        """pcm -> audio codecs"""

        if pcm.ndim == 1:
            pcm = rearrange(pcm, "T -> 1 1 T")
        if pcm.ndim == 2:
            pcm = rearrange(pcm, "B T -> B 1 T")

        audio_data = self._model.preprocess(audio_data=pcm, sample_rate=sample_rate)
        z = self._model.encode(audio_data)
        _, codes, _, _ = self._model.quantize(z)
        codes = self.expand_codes(codes)
        codes = [code.numpy(force=True) for code in codes]
        return codes

    def _decode(self, codes: Sequence[NDArray]) -> NDArray:
        """audio codecs -> pcm"""

        if len(codes) != self.NUM_HIERARCHIES:
            raise ValueError(f"codes must have length {len(self.HIERARCHY_DOWNSAMPLE_RATE)}")
        if len(codes[0]) == len(codes[-1]):
            codes = self.collapse_codes(codes)

        z_q_rec = self._model.quantizer.z_from_codes(codes)
        audio_hat = self._model.decode(z_q_rec)
        audio_hat = audio_hat.squeeze().cpu().numpy(force=True)
        return audio_hat

    def expand_codes(self, codes: Sequence[NDArray]) -> Sequence[NDArray]:
        """
        expand codes so different hierarchies have the same length

        example:
            codes = [
                [5]
                [3, 4],
                [0, 1, 2]] ->
            codes = [
                [5, 5, 5, 5]
                [3, 3, 4, 4],
                [0, 1, 2, 2]]
        """
        if len(codes) != self.NUM_HIERARCHIES:
            raise ValueError(f"codes must have length {len(self.HIERARCHY_DOWNSAMPLE_RATE)}")
        if codes[0].ndim == 1:
            codes = [rearrange(code, "T -> 1 T") for code in codes]

        codes = [np.repeat(code, self.HIERARCHY_DOWNSAMPLE_RATE[i], axis=-1) for i, code in enumerate(codes)]
        return codes

    def collapse_codes(self, codes: Sequence[NDArray]) -> Sequence[NDArray]:
        """inverse of self.expand_codes()"""
        if len(codes) != self.NUM_HIERARCHIES:
            raise ValueError(f"codes must have length {len(self.HIERARCHY_DOWNSAMPLE_RATE)}")
        if codes[0].ndim == 1:
            codes = [rearrange(code, "T -> 1 T") for code in codes]
        
        codes = [code[:, ::self.HIERARCHY_DOWNSAMPLE_RATE[i]] for i, code in enumerate(codes)]
        return codes


__all__ = [
    "AudioFeature",
    "AudioFeatures",
]
