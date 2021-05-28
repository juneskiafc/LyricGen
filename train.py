# from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import TrainingArguments
# from transformers import Trainer
from datasets import load_metric, load_dataset
# import numpy as np

training_args = TrainingArguments(
    output_dir="outputs",
    evaluation_strategy="epoch",
    num_train_epochs=1,
    save_strategy="epoch",
)

dataset = load_dataset("text", data_files="dataset.txt")
#
# tokenizer = AutoTokenizer.from_pretrained("distilgpt2")
# model = AutoModelForCausalLM.from_pretrained("distilgpt2")
#
# metric = load_metric("accuracy")
#
#
# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)
#
#
# trainer = Trainer(
#     model=model,
#     args=training_args,
#     train_dataset=train_dset,
#     eval_dataset=val_dset,
#     compute_metrics=compute_metrics,
# )
#
# trainer.train()
#
# inputs = tokenizer.encode("Hello world!", return_tensors="pt")
# outputs = model.generate(inputs, max_length=20, do_sample=True, top_p=0.95, top_k=60)
# generated = tokenizer.decode(outputs[0])
# print(generated)
