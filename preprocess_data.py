from pathlib import Path
import re
from transformers import AutoTokenizer

def add_tokens(tokenizer, dataset_files):
    # holds all the words in the dataset.
    tokens = set()

    for dataset_file in dataset_files:
        with open(dataset_file, "r") as f:
            print(dataset_file)
            for line in f:
                line = line.strip("\n")
                words = re.split(" ", line)
                for i, word in enumerate(words):
                    if word != "<eos>":
                        if i == 0:
                            tokens.add(word)
                        else:
                            tokens.add(f" {word}")

    tokenizer.add_tokens(list(tokens))
    return tokenizer


def save_vocab(tokenizer, save_dir="gpt2-tokenizer"):
    save_dir = Path(save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)

    return tokenizer.save_vocabulary("gpt2-tokenizer")


if __name__ == "__main__":
    tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
    data_files = list(Path("data").rglob("*.txt"))[:2]
    tokenizer = add_tokens(tokenizer, data_files)
    save_vocab(tokenizer)
