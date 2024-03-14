---
license: mit
tags:
- audio
---

# SNAC 🍿

Multi-**S**cale **N**eural **A**udio **C**odec (SNAC) compressess audio into discrete codes at a low bitrate.

👉 This model was primarily trained on speech data, and its recommended use case is speech synthesis. See below for other pretrained models.

🔗 GitHub repository: https://github.com/hubertsiuzdak/snac/

## Overview

SNAC encodes audio into hierarchical tokens similarly to SoundStream, EnCodec, and DAC. However, SNAC introduces a simple change where coarse tokens are sampled less frequently,
covering a broader time span.

This model compresses 24 kHz audio into discrete codes at a 0.98 kbps bitrate. It uses 3 RVQ levels with token rates of 12, 23, and
47 Hz.

## Pretrained models

Currently, all models support only single audio channel (mono).

| Model                                                                       | Bitrate   | Sample Rate | Params | Recommended use case     | 
|-----------------------------------------------------------------------------|-----------|-------------|--------|--------------------------|
| hubertsiuzdak/snac_24khz (this model) | 0.98 kbps | 24 kHz      | 19.8 M | 🗣️ Speech               | 
| [hubertsiuzdak/snac_32khz](https://huggingface.co/hubertsiuzdak/snac_32khz) | 1.9 kbps  | 32 kHz      | 54.5 M | 🎸 Music / Sound Effects | 
| [hubertsiuzdak/snac_44khz](https://huggingface.co/hubertsiuzdak/snac_44khz) | 2.6 kbps  | 44 kHz      | 54.5 M | 🎸 Music / Sound Effects |

## Usage

Install it using:

```bash
pip install snac
```
To encode (and reconstruct) audio with SNAC in Python, use the following code:

```python
import torch
from snac import SNAC

model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
audio = torch.randn(1, 1, 24000).cuda()  # B, 1, T

with torch.inference_mode():
    audio_hat, _, codes, _, _ = model(audio)
```

⚠️ Note that `codes` is a list of token sequences of variable lengths, each corresponding to a different temporal
resolution.

```
>>> [code.shape[1] for code in codes]
[12, 24, 48]
```

## Acknowledgements

Module definitions are adapted from the [Descript Audio Codec](https://github.com/descriptinc/descript-audio-codec).