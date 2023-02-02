import os
import time
from itertools import chain

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import Dataset, load_dataset, load_metric
from torch.nn import Linear
from torch.utils.data import DataLoader
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          DataCollatorWithPadding, RobertaConfig, RobertaModel,
                          RobertaTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput

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


tokenized_data.set_format("pandas")

df = tokenized_data['train'][:][['input_ids', 'attention_mask', 'label']]
print(df.shape)
dataset = Dataset.from_pandas(df)
tokenized_data['train'] = dataset

df = tokenized_data['test'][:][['input_ids', 'attention_mask', 'label']]
print(df.shape)
dataset = Dataset.from_pandas(df)
tokenized_data['test'] = dataset

# replacing the pretrained_model
pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    "./saved_transformer/saved_model/", num_labels=2)
pretrained_model.classifier = nn.Linear(
    in_features=768, out_features=768, bias=True)


class CodeBertCustom(torch.nn.Module):
    def __init__(self, pretrained_model):
        super(CodeBertCustom, self).__init__()
        self.pretrained = pretrained_model
        # for param in self.pretrained.parameters():
        #     param.requires_grad = False
        self.lin_1 = Linear(768, 768)
        self.lin_2 = Linear(768, 2)

    def forward(self, data):
        outputs = self.pretrained(
            input_ids=data.input_ids, attention_mask=data.attention_mask)

        # (batch, sentence_len, 768) -> (batch, 768)
        x = torch.mean(outputs[0], dim=1)
        # RobertaClassificationHead takes only the first token!
        # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over
        # https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_roberta.html

        x = self.lin_1(x)
        x = x.relu()
        x = F.dropout(x, p=0.2, training=self.training)

        logits = self.lin_2(x)

        loss_fct = nn.CrossEntropyLoss()
        loss = loss_fct(logits.view(-1, 2), data.labels.view(-1))

        return SequenceClassifierOutput(
            loss=loss,
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# tokenized_data = tokenized_data.remove_columns(['filename', 'code', 'fuzztag', 'rawfile', 'new_code', 'edges', 'nodes'])
tokenized_data = tokenized_data.rename_column("label", "labels")
tokenized_data.set_format("torch")
print(tokenized_data["train"].column_names)


train_dataloader = DataLoader(
    tokenized_data["train"], shuffle=True, batch_size=8, collate_fn=data_collator
)
eval_dataloader = DataLoader(
    tokenized_data["test"], batch_size=8, collate_fn=data_collator
)

for batch in train_dataloader:
    break

print({k: v.shape for k, v in batch.items()})

device = torch.device(
    "cuda") if torch.cuda.is_available() else torch.device("cpu")

# model = AutoModelForSequenceClassification.from_pretrained("./saved_transformer/saved_model/", num_labels=2)
model = CodeBertCustom(pretrained_model)
model = model.to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
criterion = torch.nn.CrossEntropyLoss()

print("Training started")


def train():
    model.train()
    accumulation_steps = 8
    optimizer.zero_grad()

    for i, data in enumerate(train_dataloader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data)
        # out = model(input_ids=data.input_ids, attention_mask=data.attention_mask)  # Perform a single forward pass.
        # out = model(data.input_ids_0, data.edge_index_0, data.batch)
        loss = criterion(out.logits, data.labels)  # Compute the loss.
        loss = loss/accumulation_steps 
        loss.backward()  # Derive gradients.
        
        if ((i+1) % accumulation_steps == 0) or (i+1 == len(train_dataloader)):
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.


def test(loader):
    model.eval()

    correct = 0
    for data in loader:  # Iterate in batches over the training/test dataset.
        data = data.to(device)
        out = model(data)
        # out = model(input_ids=data.input_ids, attention_mask=data.attention_mask)
        # out = model(data.input_ids_0, data.edge_index_0, data.batch)
        # Use the class with highest probability.
        pred = out.logits.argmax(dim=1)
        # Check against ground-truth labels.
        correct += int((pred == data.labels).sum())
    # Derive ratio of correct predictions.
    return correct / len(loader.dataset)


for epoch in range(1, 20):
    start_time = time.time()
    train()
    train_acc = test(train_dataloader)
    test_acc = test(eval_dataloader)
    print(f'Epoch: {epoch:03d}, Time: {int(time.time()-start_time)}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
