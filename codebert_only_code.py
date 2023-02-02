import os
import time
from itertools import chain

import numpy as np
import torch
from datasets import load_dataset, load_metric
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, RobertaConfig, RobertaModel,
                          RobertaTokenizer, Trainer, TrainingArguments)

import wandb

CC_ENV = True
DOWNLOAD_ALLOWED = False


device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")


if CC_ENV:
    run = wandb.init(reinit=True, project="annotation-prediction",
                     settings=wandb.Settings(start_method='fork'))
else:
    run = wandb.init(project="annotation-prediction", tags=[
                     "initial-setup"], notes="complete PyG model v0.1", settings=wandb.Settings(start_method='thread'))

if DOWNLOAD_ALLOWED:
    artifact = run.use_artifact(
        'anonymous/annotation-prediction/fuzzed-df:latest', type='dataset')
    artifact_dir = artifact.download()
    # https://huggingface.co/course/chapter5/2?fw=pt
    path_to_csv = f"{artifact_dir}/fuzzed_dataset_cfg_1897.csv"

else:
    path_to_csv = "./artifacts/fuzzed-df:v6/fuzzed_dataset_cfg_1897.csv"

annot_pred_dataset = load_dataset("csv", data_files=path_to_csv)

annot_pred_dataset = annot_pred_dataset["train"].train_test_split(
    train_size=0.8, seed=42)


if DOWNLOAD_ALLOWED:
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    pretrained_model = RobertaModel.from_pretrained("microsoft/codebert-base")
else:
    tokenizer = RobertaTokenizer.from_pretrained(
        "./saved_transformer/saved_tokenizer_model/")
    pretrained_model = RobertaModel.from_pretrained(
        "./saved_transformer/saved_model/")

# tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
# pretrained_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")


def preprocess_function(examples):
    return tokenizer(examples["new_code"], truncation=True)


tokenized_data = annot_pred_dataset.map(preprocess_function, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)


print("Loading metric")

metric = load_metric("./accuracy.py")


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = np.argmax(logits, axis=-1)
    return metric.compute(predictions=predictions, references=labels)


if DOWNLOAD_ALLOWED:
    model = AutoModelForSequenceClassification.from_pretrained(
        "microsoft/codebert-base", num_labels=2)
else:
    model = AutoModelForSequenceClassification.from_pretrained(
        "./saved_transformer/saved_model/", num_labels=2)


logging_steps = len(tokenized_data["train"]) // 16

training_args = TrainingArguments(
    output_dir="./results",
    learning_rate=2e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=16,
    num_train_epochs=20,
    evaluation_strategy="epoch",
    weight_decay=0.01,
    logging_steps=logging_steps,
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data["train"],
    eval_dataset=tokenized_data["test"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
    data_collator=data_collator,
)

trainer.train()
