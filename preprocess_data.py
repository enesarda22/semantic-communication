import argparse
import glob
import os

from tqdm import tqdm

from semantic_communication.data_processing.data_handler import DataHandler
from semantic_communication.data_processing.preprocessor import Preprocessor


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--europarl-folder-path", type=str)
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
    Preprocessor.dump_data(preprocessed_messages, DataHandler.data_filename)
