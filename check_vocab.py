import json

with open("gpt2-tokenizer/vocab.json", "r") as f:
    data = json.load(f)
    i = 0
    n = len(data.keys())

    with open("cleaned_dataset.txt", "r") as df:
        line = df.readline()
        for word in line.split(" "):
            if f" {word}" not in data and word not in data:
                print(f"{word}!")
                i += 1

    print(f"{i}/{n}")
