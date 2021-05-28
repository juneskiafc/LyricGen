import os

from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import GPT2Tokenizer
from transformers import Trainer
from datasets import load_dataset

from utils import tokenize_function, compute_metrics

# set wandb project
os.environ["WANDB_PROJECT"] = "lyricgen"

# Load dataset
dataset = load_dataset("text", data_files="cleaned_dataset.txt")["train"]

# Init tokenizer
tokenizer = GPT2Tokenizer(vocab_file="gpt2-tokenizer/vocab.json",
                          merges_file="gpt2-tokenizer/merges.txt",
                          eos_token="<eos>")

dataset = dataset.map(function=tokenize_function,
                      batched=True,
                      fn_kwargs={"tokenizer": tokenizer})

# shuffle + split
DATASET_SHUFFLE_SPLIT_SEED = 10
dataset = dataset.train_test_split(test_size=0.1,
                                   shuffle=True,
                                   seed=DATASET_SHUFFLE_SPLIT_SEED)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# init model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# change model embeddings size
embeddings = model.resize_token_embeddings()
print(embeddings.num_embeddings)
embeddings = model.resize_token_embeddings(len(tokenizer))
print(embeddings.num_embeddings)
raise ValueError

# trainer arguments
training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    num_train_epochs=50,
    save_strategy="epoch",
    report_to="wandb",
)

# init trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=test_dataset,
    compute_metrics=compute_metrics,
)

trainer.train()
trainer.evaluate()
