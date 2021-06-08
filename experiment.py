from transformers import AutoModelForCausalLM, AutoTokenizer
import random

CHECKPOINT_FILE = "outputs/checkpoint.ckpt"
N_TO_GENERATE = 10
N_SUBJECTS = 10
MAX_SEQ_LEN = 10

def init_model_and_tokenizer():
    model = AutoModelForCausalLM.from_pretrained(
            CHECKPOINT_FILE,
            from_tf=False,
            model_type="gpt2",
            )

    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    return model, tokenizer

def get_num_lines(file="data/all_lyrics_val.txt"):
    with open(file, "r") as vf:
        n_lines = 0
        for line in vf:
            n_lines += 1
    
    return n_lines

def get_random_lyrics(n_lines, file="data/all_lyrics_val.txt"):
    with open(file, "r") as vf:
        # sample N random examples.
        lyric_idxs = set(random.sample(range(n_lines), N_TO_GENERATE))
        lyrics = []
        for i, line in enumerate(vf):
            if i in lyric_idxs:
                lyrics.append(line)

    return lyrics

def generate_experiment_for_subject(subject_idx, model, tokenizer, lyrics):
    generated_idxs = set(random.sample(range(len(lyrics)), len(lyrics)//2))

    with open(f"subject_{subject_idx}.txt", "w") as f:
        outputs = []

        for n in range(N_TO_GENERATE):
            split_idx = len(lyric)//2

            lyric = lyrics[n]

            if n in generated_idxs:     
                output = model_predict(model, tokenizer, lyric[:split_idx])

            else:
                output = lyric[split_idx:split_idx+MAX_SEQ_LEN]

            f.write(output)

def model_predict(model, tokenizer, input_string):
    input_ids = tokenizer.encode(input_string, return_tensors='tf')
    model_output = model.generate(
        input_ids,
        do_sample=True, 
        max_length=MAX_SEQ_LEN, 
        top_k=50, 
        top_p=0.95, 
        num_return_sequences=1
    )
    output = tokenizer.decode(model_output, skip_special_tokens=True)

    return output

def generate_experiments(file="data/all_lyrics_val.txt"):
    n_lines = get_num_lines(file)
    model, tokenizer = init_model_and_tokenizer()
    lyrics = get_random_lyrics(n_lines)

    for subject_idx in range(N_SUBJECTS):
        generate_experiment_for_subject(subject_idx, model, tokenizer, lyrics)

if __name__ == "__main__":
    generate_experiments()
        


