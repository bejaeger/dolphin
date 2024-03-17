import argparse
import logging

import librosa
import soundfile

from data.audio import AudioFeature, AudioFeatures

logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    model_path = args.model_path
    audio_path = args.audio_path
    config_path = args.config_path
    feature = args.feature
    device = args.device
    output_path = args.output_path

    feature = AudioFeature.create(
        feature,
        model_path=model_path,
        config_path=config_path,
        device=device,
    )

    pcm, sample_rate = soundfile.read(audio_path)

    codes = feature.encode(audio_path=audio_path)
    pcm_hat = feature.decode(codes=codes)

    if output_path is not None:
        soundfile.write(audio_path, pcm_hat, feature.out_sample_rate)

        output_path_orig = audio_path.replace(".wav", "_original.wav")

        if sample_rate != feature.out_sample_rate:
            pcm = librosa.resample(pcm, orig_sr=sample_rate, target_sr=feature.out_sample_rate)

        soundfile.write(output_path_orig, pcm, feature.out_sample_rate)

        logging.info(f"Saved reconstructed audio to `{output_path}` and original audio to `{output_path_orig}` ")
    else:
        logging.info(f"Starting interactive session. You can explore, variables `pcm`, `codes`, and `pcm_hat`")

        import code
        code.interact(local=locals())


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-f",
        "--feature",
        default=AudioFeatures.SNAC_CODEC.value,
        choices=[feature.value for feature in AudioFeatures])
    parser.add_argument("-m", "--model-path", required=True)
    parser.add_argument("-i", "--audio-path", required=True)
    parser.add_argument("-o", "--output-path", help="Save orig and reconstructed pcm")
    parser.add_argument(
        "-c",
        "--config-path",
        help="Path to config (default: config.json in dir of `--model-path`)")
    parser.add_argument("-d", "--device", default="cpu")

    main(parser.parse_args())
