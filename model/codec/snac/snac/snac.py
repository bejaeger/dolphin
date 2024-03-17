import json
import logging
import math
from typing import *

import numpy as np
import torch
import torchaudio
from numpy.typing import NDArray
from torch import Tensor
from torch import nn

from .layers import Encoder, Decoder
from .vq import ResidualVectorQuantize


class SNAC(nn.Module):
    def __init__(
        self,
        sampling_rate=44100,
        encoder_dim=64,
        encoder_rates=[3, 3, 7, 7],
        latent_dim=None,
        decoder_dim=1536,
        decoder_rates=[7, 7, 3, 3],
        attn_window_size=32,
        codebook_size=4096,
        codebook_dim=8,
        vq_strides=[8, 4, 2, 1],
        noise=True,
        depthwise=True,
    ):
        super().__init__()
        self.sampling_rate = sampling_rate
        self.encoder_dim = encoder_dim
        self.encoder_rates = encoder_rates
        self.decoder_dim = decoder_dim
        self.decoder_rates = decoder_rates
        if latent_dim is None:
            latent_dim = encoder_dim * (2 ** len(encoder_rates))
        self.latent_dim = latent_dim
        self.hop_length = np.prod(encoder_rates)
        self.encoder = Encoder(
            encoder_dim,
            encoder_rates,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )
        self.n_codebooks = len(vq_strides)
        self.codebook_size = codebook_size
        self.codebook_dim = codebook_dim
        self.vq_strides = vq_strides
        self.attn_window_size = attn_window_size
        self.quantizer = ResidualVectorQuantize(
            input_dim=latent_dim,
            codebook_size=codebook_size,
            codebook_dim=codebook_dim,
            vq_strides=vq_strides,
        )
        self.decoder = Decoder(
            latent_dim,
            decoder_dim,
            decoder_rates,
            noise,
            depthwise=depthwise,
            attn_window_size=attn_window_size,
        )

    @property
    def sample_rate(self) -> int:
        return self.sampling_rate

    def preprocess(self, audio_data: Union[NDArray, Tensor], sample_rate: int):
        if not isinstance(audio_data, Tensor):
            audio_data = torch.from_numpy(audio_data.astype(np.float32))
        if sample_rate != self.sample_rate:
            logging.warn(f"Resample audio from `{sample_rate}` to `{self.sample_rate}`")
            audio_data = \
                torchaudio.functional.resample(audio_data, orig_freq=sample_rate, new_freq=self.sample_rate)
        
        length = audio_data.shape[-1]
        lcm = math.lcm(self.vq_strides[0], self.attn_window_size or 1)
        pad_to = self.hop_length * lcm
        right_pad = math.ceil(length / pad_to) * pad_to - length
        audio_data = nn.functional.pad(audio_data, (0, right_pad))
        return audio_data

    def encode(self, audio_data):
        z = self.encoder(audio_data)
        return z

    def quantize(self, z):
        z, codes, commitment_loss, codebook_loss = self.quantizer(z)
        return z, codes, commitment_loss, codebook_loss

    def decode(self, z):
        x = self.decoder(z)
        return x

    def encode_decode(self, audio_data: Tensor, sample_rate: int):
        return self.forward(audio_data=audio_data, sample_rate=sample_rate)
    
    def forward(self, audio_data: Tensor, sample_rate: int):
        """
        Shape: audio_data: [B, 1, T]
        """
        audio_data = self.preprocess(audio_data=audio_data, sample_rate=sample_rate)
        z = self.encode(audio_data)
        z, codes, commitment_loss, codebook_loss = self.quantize(z)
        x = self.decode(z)
        audio_length = audio_data.shape[-1]        
        return x[..., :audio_length], z, codes, commitment_loss, codebook_loss

    @classmethod
    def from_config(cls, config_path):
        with open(config_path, "r") as f:
            config = json.load(f)
        model = cls(**config)
        return model

    @classmethod
    def from_pretrained_local(cls, model_path, config_path):
        model = cls.from_config(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model

    @classmethod
    def from_pretrained(cls, repo_id, **kwargs):
        from huggingface_hub import hf_hub_download

        config_path = hf_hub_download(repo_id=repo_id, filename="config.json", **kwargs)
        model_path = hf_hub_download(repo_id=repo_id, filename="pytorch_model.bin", **kwargs)
        model = cls.from_config(config_path)
        state_dict = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state_dict)
        model.eval()
        return model
