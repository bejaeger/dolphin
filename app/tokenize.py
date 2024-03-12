import argparse
import logging

from data.text.tokenizer import BPETokenizer

logging.basicConfig(level=logging.INFO)


def main(args: argparse.Namespace) -> None:
    text = args.text

    tokenizer = BPETokenizer()

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
    parser.add_argument("-t", "--text", required=True)

    main(parser.parse_args())
