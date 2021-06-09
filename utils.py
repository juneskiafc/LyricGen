"""
Utility functions that collects and splits the dataset.

Author: Junhwi Kim (junhwi.kim.20@dartmouth.edu)
"""
import random
from pathlib import Path

def concat_dataset(data_files, output_file="data/all_lyrics.txt"):
    if Path(output_file).is_file():
        return
    
    seen = set()

    with open("data/all_lyrics.txt", "w") as f:
        for data_file in data_files:
            with open(data_file, "r") as d:
                for line in d:
                    if line not in seen:
                        f.write(line)
                        seen.add(line)

def split_dataset(data_file="data/all_lyrics.txt", train_ratio=0.8):
    train_data_file = f"{Path(data_file).with_suffix('')}_train.txt"
    val_data_file = f"{Path(data_file).with_suffix('')}_val.txt"

    if Path(train_data_file).is_file() and Path(val_data_file).is_file():
        return

    num_lines = 0
    with open(data_file, "r") as f:
        for line in f:
            num_lines += 1

    num_train_examples = int(num_lines*train_ratio)
    train_idxs = set(random.sample(range(num_lines), num_train_examples))
    
    with open(data_file, "r") as f:

        tf = open(train_data_file, "w")
        vf = open(val_data_file, "w")

        for i, line in enumerate(f):
            if i in train_idxs:
                tf.write(line)
            else:
                vf.write(line)
        
        tf.close()
        vf.close()

if __name__ == "__main__":
    data_files = list(Path("data").rglob("*.txt"))
    concat_dataset(data_files)
    split_dataset()