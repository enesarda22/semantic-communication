import argparse
import glob
import os

from tqdm import tqdm

from semantic_communication.data_processing.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--europarl-folder-path", type=str)
    parser.add_argument("--output-data-fp", default="", type=str)
    parser.add_argument("--train-size", default=0.7, type=float)
    parser.add_argument("--test-size", default=0.5, type=float)
    parser.add_argument("--n-samples", default=None, type=int)
    args = parser.parse_args()

    en_fp = os.path.join(args.europarl_folder_path, "en/*.txt")
    txt_filepaths = sorted(glob.glob(en_fp))

    preprocessed_messages = []
    for txt_fp in tqdm(txt_filepaths, "Txt files"):
        with open(txt_fp, "r", encoding="utf-8") as f:
            m = f.read()

        preprocessed_messages.extend(Preprocessor.preprocess(m))
        if args.n_samples and (args.n_samples < len(preprocessed_messages)):
            preprocessed_messages = preprocessed_messages[: args.n_samples]
            break

    print(f"Number of sentences: {len(preprocessed_messages)}")

    train_data, val_data, test_data = Preprocessor.split_data(
        messages=preprocessed_messages,
        train_size=args.train_size,
        test_size=args.test_size,
    )

    train_fp = os.path.join(args.output_data_fp, Preprocessor.train_data_fn)
    Preprocessor.dump_data(train_data, train_fp)

    val_fp = os.path.join(args.output_data_fp, Preprocessor.val_data_fn)
    Preprocessor.dump_data(val_data, val_fp)

    test_fp = os.path.join(args.output_data_fp, Preprocessor.test_data_fn)
    Preprocessor.dump_data(test_data, test_fp)
