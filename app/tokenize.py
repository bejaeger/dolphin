import argparse
import logging

from data.text import Tokenizer, Tokenizers
from data.text.normalizer import TextNormalizers

logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    text = args.text
    normalizer = args.normalizer
    tokenizer = args.tokenizer

    tokenizer = Tokenizer.create(tokenizer=tokenizer, normalizer=normalizer)

    logging.info("ORIGINAL")
    logging.info(text)
    logging.info("NORMED")
    logging.info(tokenizer.normalize(text))
    logging.info("TOKENS")
    logging.info(tokenizer.tokens(text))
    logging.info("ENCODED")
    logging.info(tokenizer.encode(text))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-t", "--text", default="hello (23 Jan 2020, 12:10 AM)")
    parser.add_argument(
        "-k",
        "--tokenizer",
        choices=[t.value for t in Tokenizers],
        default=Tokenizers.BPE.value)
    parser.add_argument(
        "-n",
        "--normalizer",
        choices=[n.value for n in TextNormalizers],
        default=TextNormalizers.ENGLISH_VERBALIZED.value)

    main(parser.parse_args())
