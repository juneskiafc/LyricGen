from pathlib import Path
from transformers import AutoTokenizer


def add_tokens(tokenizer, dataset_file):
    # holds all the words in the dataset.
    tokens = set()

    with open(dataset_file, "r") as f:
        for line in f:
            # split words by spaces
            words = line.strip("\n").split(" ")

            # gpt-2 was pretrained to treat spaces as part of the token.
            # This means each word after the first word needs a space in front of it.
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
    tokenizer = add_tokens(tokenizer, "cleaned_dataset.txt")
    save_vocab(tokenizer)
