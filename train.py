from transformers import AutoModelForCausalLM
from transformers import TrainingArguments
from transformers import GPT2Tokenizer
from transformers import Trainer
from datasets import load_metric, load_dataset

# import numpy as np

# Load dataset + shuffle + split
DATASET_SHUFFLE_SPLIT_SEED = 10
dataset = load_dataset("text", data_files="cleaned_dataset.txt")["train"]
dataset = dataset.train_test_split(test_size=0.3,
                                   shuffle=True,
                                   seed=DATASET_SHUFFLE_SPLIT_SEED)
train_dataset = dataset["train"]
test_dataset = dataset["test"]

# Init tokenizer
tokenizer = GPT2Tokenizer(vocab_file="gpt2-tokenizer/vocab.json",
                          merges_file="gpt2-tokenizer/merges.txt",
                          eos_token="<eos>")

# init model
model = AutoModelForCausalLM.from_pretrained("distilgpt2")

# change model embeddings size
model.resize_token_embeddings(len(tokenizer))

# trainer arguments
# training_args = TrainingArguments(
#     output_dir="outputs",
#     evaluation_strategy="epoch",
#     num_train_epochs=1,
#     save_strategy="epoch",
# )

# init trainer
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dataset,
#     eval_dataset=test_dataset,
# )

#
# trainer.train()
#
# inputs = tokenizer.encode("Hello world!", return_tensors="pt")
# outputs = model.generate(inputs, max_length=20, do_sample=True, top_p=0.95, top_k=60)
# generated = tokenizer.decode(outputs[0])
# print(generated)
