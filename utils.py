import numpy as np
from datasets import load_metric


def tokenize_function(examples, tokenizer):
    return tokenizer(examples["text"], padding="max_length", truncation=True)


def compute_metrics(preds, labels):
    metric = load_metric("bleurt")

    predictions = np.argmax(preds, axis=-1)
    return metric.compute(predictions=predictions, references=labels)