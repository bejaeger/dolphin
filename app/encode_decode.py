import argparse
import logging
import os
import shutil

import torch
import soundfile

from model.snac.snac import SNAC

logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    model_path = args.model_path
    config_path = args.config_path
    input_path = args.input_path
    output_path = args.output_path
    device = args.device

    if config_path is None:
        config_path = os.path.join(os.path.dirname(model_path), "config.json")

    model = SNAC.from_pretrained_local(model_path=model_path, config_path=config_path)
    model.eval()
    model.to(device)

    pcm, sample_rate = soundfile.read(input_path)
    pcm = torch.tensor(pcm, dtype=torch.float).to(device).unsqueeze(0).unsqueeze(0)

    with torch.no_grad():
        audio_hat, z, codes, _, _ = model.encode_decode(audio_data=pcm, sample_rate=sample_rate)

    audio_hat = audio_hat.squeeze().cpu().numpy()

    output_folder = os.path.dirname(output_path)
    os.makedirs(output_folder, exist_ok=True)

    soundfile.write(output_path, audio_hat, model.sample_rate)
    output_path_orig = output_path.replace(".wav", "_original.wav")
    shutil.copy(input_path, output_path_orig)

    logging.info(f"Saved reconstructed audio to `{output_path}` and original audio to `{output_path_orig}`")


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
