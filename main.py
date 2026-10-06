import argparse
from importlib import import_module


ENCODERS = {
    "bge-m3": "sentence_encoders.eval_bge_m3",
    "bge-small-desc": "sentence_encoders.eval_bge_small_descriptions",
    "e5-small-v2": "sentence_encoders.eval_e5_small_v2",
    "minilm": "sentence_encoders.eval_minilm",
}


def main():
    parser = argparse.ArgumentParser(description="Run a sentence-encoder zero-shot evaluation.")
    parser.add_argument("encoder", choices=ENCODERS, help="sentence encoder to evaluate")
    args = parser.parse_args()
    import_module(ENCODERS[args.encoder]).main()


if __name__ == "__main__":
    main()
