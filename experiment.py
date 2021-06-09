from typing import Text
from transformers import AutoModelForCausalLM, AutoTokenizer
import random
from pathlib import Path
from datasets import load_metric

CHECKPOINT_DIR = "checkpoint-1000"
N_TO_GENERATE = 6
N_SUBJECTS = 10

def init_model_and_tokenizer():
    print("Initializing model and tokenizer.")
    model = AutoModelForCausalLM.from_pretrained(
            pretrained_model_name_or_path=CHECKPOINT_DIR,
            config=CHECKPOINT_DIR,
            )

    tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path=CHECKPOINT_DIR)

    return model, tokenizer

def read_lyrics(file="data/all_lyrics_val.txt"):
    print("Reading dataset.")
    with open(file, "r") as vf:
        return list(vf.readlines())

def get_random_lyrics(all_lyrics):
    # sample N random examples.
    print("\tGetting random lyrics...")
    lyric_idxs = random.sample(range(len(all_lyrics)), N_TO_GENERATE)
    lyrics = []

    for i in lyric_idxs:
        lyric = all_lyrics[i]

        if lyric == "":
            alternate_idx = i
            while lyric == "" or alternate_idx in lyric_idxs:
                alternate_idx = random.sample(range(len(all_lyrics)), 1)
                lyric = all_lyrics[alternate_idx]

        lyrics.append(lyric)
    
    return lyrics

def get_first_n_lines(lines, n=1, reverse=False):
    split_lines = lines.split(" ")
    output_line = []

    if reverse:
        i = -1
    else:
        i = 0

    line_count = 0
    while line_count < n:
        try:
            word = split_lines[i]
        except IndexError:
            break
        
        if word == "<eos>":
            line_count += 1
            output_line.append("/")
        else:
            output_line.append(word)
        
        if reverse:
            i -= 1
        else:
            i += 1

    if reverse:
        output_line.reverse()
        if output_line[0] == "/":
            output_line = output_line[1:]
    
    else:
        if output_line[-1] == "/":
            output_line = output_line[:-1]
    
    output_line = " ".join(output_line)

    return output_line

def generate_experiment_for_subject(subject_idx,
                                    model,
                                    tokenizer,
                                    lyrics,
                                    experiment_out_dir,
                                    n_lines_to_generate=3,
                                    n_lines_to_show_as_prior=3,
                                    bleurt_metric=None):
    generated_idxs = set(random.sample(range(N_TO_GENERATE), N_TO_GENERATE//2))
    bleurt_out_file = open(f"{experiment_out_dir}/bleurt_{subject_idx}.txt", "w")
    
    with open(f"{experiment_out_dir}/subject_{subject_idx}.txt", "w") as f:
        for n in range(N_TO_GENERATE):
            lyric = lyrics[n]
            
            # split into words
            split_lyric = lyric.split(" ")
            split_idx = len(split_lyric)//2

            split_prior = split_lyric[:split_idx]

            # truncate prior if larger than 512.
            if len(split_prior) > 512:
                split_prior = split_prior[:512]
            
            prior = " ".join(split_prior)

            model_output = model_predict(model, tokenizer, prior)
            model_output = get_first_n_lines(model_output, n=n_lines_to_generate)
            real_output = get_first_n_lines(" ".join(split_lyric[split_idx:]), n=n_lines_to_generate)

            bleurt_results = bleurt_metric.compute(predictions=[model_output], references=[real_output])
            bleurt = bleurt_results["scores"][0]
            bleurt_out_file.write(str(round(bleurt, 2)) + "\n")

            if n in generated_idxs:
                output = model_output
                mode = "GEN"

            else:
                # get the next real line.
                output = real_output
                mode = "REAL"

            # get the last 3 lines preceding split_idx, for vis purposes
            condition_line = get_first_n_lines(prior, n=n_lines_to_show_as_prior, reverse=True)
            output = output.strip("\n")

            print(f"\t[prior]: {condition_line}")
            print(f"\t[model_outputs]: {model_output}")
            print(f"\t[real_outputs]: {real_output}")

            print("\n")

            f.write(f"[{mode}] {condition_line} --> {output}\n")

    bleurt_out_file.close()

def model_predict(model, tokenizer, input_string):
    input_ids = tokenizer.encode(input_string, return_tensors='pt')

    if input_ids.shape[1] > 512:
        input_ids = input_ids[:, :511]

    model_output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=512, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1,
        pad_token_id=50256, # eos token id
    )
    output = tokenizer.decode(model_output[0], skip_special_tokens=True)

    return output

def generate_experiments(file="data/all_lyrics_val.txt", experiment_out_dir="experiments"):
    Path(experiment_out_dir).mkdir(parents=True, exist_ok=True)

    all_lyrics = read_lyrics(file)
    model, tokenizer = init_model_and_tokenizer()
    print("Initializing BLEURT.")
    bleurt_metric = load_metric("bleurt")

    print("Begin experiment generation.")
    print("-----------------")

    for subject_idx in range(N_SUBJECTS):
        print(f"[subject {subject_idx}]")

        lyrics = get_random_lyrics(all_lyrics)

        print(f"\tgenerating experiment for subject {subject_idx}...")
        generate_experiment_for_subject(subject_idx, model, tokenizer, lyrics, experiment_out_dir, bleurt_metric=bleurt_metric)
        print(f"\tdone.")

if __name__ == "__main__":
    generate_experiments()
        


