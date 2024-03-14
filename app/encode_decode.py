import argparse
import logging
import os

import torch
import torchaudio
import soundfile

from model.codec.snac.snac import SNAC

logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    model_path = args.model_path
    config_path = args.config_path
    input_path = args.input_path
    output_path = args.output_path
    device = args.device

    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")

    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    model = SNAC.from_pretrained_local(model_path=model_path, config_path=config_path)
    model.eval()
    model.to(device)

    pcm, sample_rate = soundfile.read(input_path)

    with torch.no_grad():
        pcm = torch.tensor(pcm, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(0)
        audio_hat, z, codes, _, _ = model.encode_decode(audio_data=pcm, sample_rate=sample_rate)

    audio_hat = audio_hat.squeeze().cpu().numpy()
    soundfile.write(output_path, audio_hat, model.sample_rate)

    output_path_orig = output_path.replace(".wav", "_original.wav")
    if sample_rate != model.sample_rate:
        pcm = torchaudio.functional.resample(pcm, orig_freq=sample_rate, new_freq=model.sample_rate)
    soundfile.write(output_path_orig, pcm.squeeze().cpu().numpy(), model.sample_rate)

    with torch.no_grad():
        z_q_rec = model.quantizer.z_from_codes(codes)
        audio_hat_hat = model.decode(z_q_rec)

    output_path_rec_codes = output_path.replace(".wav", "_rec_codes.wav")
    soundfile.write(output_path_rec_codes, audio_hat_hat.squeeze().cpu().numpy(), model.sample_rate)

    logging.info(
        f"Saved reconstructed audio to `{output_path}` "
        f"and original audio to `{output_path_orig}` "
        f"and reconstructed audio from codes to `{output_path_rec_codes}`")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument(
        "-c",
        "--config-path",
        help="Path to config (default: config.json in dir of `--model-path`)")
    parser.add_argument("-i", "--input-path", required=True)
    parser.add_argument("-o", "--output-path", required=True)
    parser.add_argument("-d", "--device", default="cpu")

    main(parser.parse_args())
