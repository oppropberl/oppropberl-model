import argparse
import ast
from dataclasses import dataclass
import os
import time

from pkg_resources import require
import GPUtil
from itertools import chain
from pprint import pprint
import re
from random import choice
import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from torch.nn import Linear
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from torch_geometric.nn import GCNConv, global_mean_pool
from torchinfo import summary
from transformers import (AutoModel, AutoModelForMaskedLM,
                          AutoModelForSequenceClassification, AutoTokenizer,
                          RobertaConfig, RobertaModel, RobertaTokenizer)
from transformers.modeling_outputs import SequenceClassifierOutput
from sklearn.metrics import confusion_matrix, classification_report
import wandb

CC_ENV = True
DOWNLOAD_ALLOWED = True
MAX_LENGTH = 256 # tokenizer max length
type_annotations_list = ["@NonNull", "@Nullable"] # should be with @ symbol in the beginning for Java
top_n_annotation_pred = 5 # choose from top n annotation prediction set for a given code (applicable for prediction_mode='annot')

# Idea derived from https://www.tensorflow.org/tutorials/structured_data/imbalanced_data#calculate_class_weights
heatmap_pred_pos_per = [0.963, 0.037] # 3.7% distribution of class 1 (Error) in the training set
weights_heatmap = torch.tensor([1/(2*i) for i in heatmap_pred_pos_per])

parser = argparse.ArgumentParser()
parser.add_argument('--prediction_mode', type=str, required=False, choices=['heatmap', 'annot', 'error'], default='heatmap')
parser.add_argument('--n_epochs', type=str, required=False, default="20")
parser.add_argument('--batch_size', type=str, required=True)
parser.add_argument('--accumulation_steps', type=str, required=True)
parser.add_argument('--model_mode', choices=['code_only', 'cfg_only', 'random_cfg', 'frozen_weights', 'random_code', 'normal'], type=str, required=True)
parser.add_argument('--eval_mode', choices=["Y", "N"], required=False, default="N")
parser.add_argument('--model_name', type=str, required=True)
parser.add_argument('--model_tag', type=str, required=True)
parser.add_argument('--model_notes', type=str, required=True)
parser.add_argument('--just_testing', choices=["Y", "N"], required=False, default="N")
# parser.add_argument('--logging', choices=['debug', 'info', 'normal'], type=str, required=True)
args = parser.parse_args()

args.batch_size = int(args.batch_size)
args.accumulation_steps = int(args.accumulation_steps)
args.n_epochs = int(args.n_epochs)

print(f"Batch size: {args.batch_size}, Accumulation steps: {args.accumulation_steps}, Model mode: {args.model_mode}, Epochs: {args.n_epochs}")

device = torch.device(
    "cuda:0") if torch.cuda.is_available() else torch.device("cpu")

if CC_ENV:
    run = wandb.init(reinit=True, project="annotation-prediction", tags=["initial-setup", args.model_name, args.prediction_mode, args.model_mode], notes=args.model_notes, settings=wandb.Settings(start_method='fork'), save_code=True)
else:
    run = wandb.init(project="annotation-prediction", tags=[
                     "initial-setup"], notes="complete PyG model v0.1", settings=wandb.Settings(start_method='thread'))

wandb.config.update(args)

if DOWNLOAD_ALLOWED:
    artifact = run.use_artifact(
        'anonymous/annotation-prediction/fuzzed-df:latest', type='dataset')
    artifact_dir = artifact.download()
    # https://huggingface.co/course/chapter5/2?fw=pt
    path_to_csv = f"{artifact_dir}/fuzzed_dataset_cfg_30906.csv"

else:
    path_to_csv = "./artifacts/fuzzed-df:v7/fuzzed_dataset_cfg_30906.csv"

annot_pred_dataset = load_dataset("csv", data_files=path_to_csv)

print(annot_pred_dataset)

if DOWNLOAD_ALLOWED:
    tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    "microsoft/codebert-base", num_labels=2)
else:
    tokenizer = RobertaTokenizer.from_pretrained(
        "./saved_transformer/saved_tokenizer_model/")
    pretrained_model = AutoModelForSequenceClassification.from_pretrained(
    "./saved_transformer/saved_model/", num_labels=2)

# add the type qualifiers as single tokens so that they don't split into multiple sub-words. Otherwise it will create problem during MLM due to multiple masks.
num_added_toks = tokenizer.add_tokens(type_annotations_list)
print("We have added", num_added_toks, "tokens")

mask_token_id = tokenizer("<mask>", return_tensors='np')['input_ids'][0,1]

# replacing the pretrained_model last layer
pretrained_model.classifier = nn.Linear(
    in_features=768, out_features=768, bias=True)
# Notice: resize_token_embeddings expect to receive the full size of the new vocabulary, i.e., the length of the tokenizer.
vocab_size=len(tokenizer)
pretrained_model.resize_token_embeddings(vocab_size)

# tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")
# pretrained_model = AutoModel.from_pretrained("microsoft/graphcodebert-base")

annot_pred_dataset.set_format("pandas")

df_train_expt = annot_pred_dataset['train'][:]
if args.just_testing == "Y":
    print("Just testing")
    df_train_expt = df_train_expt[:1000]
    args.n_epochs = 2

if args.model_mode == 'code_only':
    df_train_expt['edges'] = '[[]]'
    df_train_expt['nodes'] = '[{}]'

if args.prediction_mode == 'annot':
    df_train_expt = df_train_expt[df_train_expt.label == 0].copy() # for annotation prediction, use the non-error codes

df_train_expt['len'] = df_train_expt['edges'].apply(
    lambda x: len(ast.literal_eval(x))).to_list()

df_train_expt['edges'] = df_train_expt['edges'].apply(
    lambda x: ast.literal_eval(x))

df_train_expt['nodes'] = df_train_expt['nodes'].apply(
    lambda x: ast.literal_eval(x))

df_train_expt['error_line'] = df_train_expt['error_line'].apply(
    lambda x: ast.literal_eval(x))

df_train_expt['new_code'] = df_train_expt['new_code'].str.replace(r"([\n]{2,})", '\n', regex=True) # replace extra spaces with one

max_graphs = df_train_expt['len'].max()

print("max_graphs: ", max_graphs)

def append_empty_list(len, edges):
    curr_list = edges
    for i in range(max_graphs-len):
        curr_list.append([])
    return curr_list


df_train_expt["edges"] = df_train_expt.apply(
    lambda row: append_empty_list(row["len"], row["edges"]), axis=1)

def append_empty_dict(len, nodes):
    curr_list = nodes
    for i in range(max_graphs-len):
        curr_list.append({})
    return curr_list

df_train_expt["nodes"] = df_train_expt.apply(
    lambda row: append_empty_dict(row["len"], row["nodes"]), axis=1)

for i in range(max_graphs):
    df_train_expt[f"edge_{i}"] = df_train_expt["edges"].apply(lambda x: x[i])
    df_train_expt[f"node_{i}"] = df_train_expt["nodes"].apply(lambda x: x[i])

print("Shape before dropping empty graphs: ", df_train_expt.shape)
if args.model_mode != 'code_only':
    df_train_expt = df_train_expt[df_train_expt.len > 1]
print("Shape after dropping empty graphs: ", df_train_expt.shape)

if args.prediction_mode == 'annot':
    print("No. of rows with no annotations present: ", df_train_expt[~df_train_expt['new_code'].str.contains(r'|'.join(type_annotations_list))])
    df_train_expt = df_train_expt[df_train_expt['new_code'].str.contains(r'|'.join(type_annotations_list))].copy() # remove the code rows with no annotations present

df_train_expt.reset_index(drop=True, inplace=True)

# expt_subset = df_train_expt[['node_0', 'edge_0', 'len', 'label']].copy()

# expt_subset = expt_subset[expt_subset.len>1]

# d1 = expt_subset['node_0'][0]

# e1 = expt_subset['edge_0'][0]

# out = pretrained_model(input_ids = torch.tensor(np.array(t1['input_ids'])), attention_mask = torch.tensor(np.array(t1['attention_mask'])))
# out.pooler_output

# d1, e1

# expt_subset.head()


def create_mapping(d1, e1, graph_id):
    new_map = {}
    for new, old in enumerate(d1.keys()):
        new_map[old] = new  # converting to new node index (integer)

    e1 = np.array([[new_map[x] for x in l] for l in e1])

    new_d1 = {}
    for key, val in d1.items():
        new_d1[new_map[key]] = val

    new_d1 = dict(sorted(new_d1.items()))
    val_array = list(new_d1.values())
    if len(val_array) == 0:  # taking care of padding edges and nodes added for shorter graphs
        val_array = [""]
    t1 = tokenizer(val_array, truncation=True, padding='max_length',
                   max_length=MAX_LENGTH, return_tensors='np')
    # return pd.Series([t1['input_ids'], t1['attention_mask'], e1], index=['q1', 'q2', 'q3'])
    return pd.Series([np.array(t1['input_ids'], dtype='int32'), np.array(t1['attention_mask'], dtype='int32'), e1], index=[f'input_ids_{graph_id}', f'attention_mask_{graph_id}', f'edge_index_{graph_id}'])

def find_nth(haystack, needle, n):
    if n==0: 
        return 0
    start = haystack.find(needle)
    while start >= 0 and n > 1:
        start = haystack.find(needle, start+len(needle))
        n -= 1
    return start

def create_heatmap(new_code, line_nos):
    heatmap = np.zeros(MAX_LENGTH)

    for line_no in line_nos:
        ind_prev = find_nth(new_code, "\n", line_no-1)
        ind_next = find_nth(new_code, "\n", line_no)

        start_token = tokenizer(new_code[:ind_prev], return_tensors='np')['input_ids'][:,1:-1].reshape(-1,).shape[0] + 1 # to compensate for <start_token> in input_ids
        end_token = tokenizer(new_code[:ind_next], return_tensors='np')['input_ids'][:,1:-1].reshape(-1,).shape[0] + 1 # to compensate for <start_token> in input_ids
        heatmap[start_token:end_token+1] = 1 # including the end_token position

    return heatmap

def create_code_tokens(raw_code, name): # Note: Beware of t1['input_ids'] vs t1['input_ids'][0] refactoring
    t1 = tokenizer(raw_code, truncation=True, padding='max_length', 
                    max_length=MAX_LENGTH, return_tensors='np')
    return pd.Series([np.array(t1['input_ids'], dtype='int32'), np.array(t1['attention_mask'], dtype='int32')], index=[f'input_ids_{name}', f'attention_mask_{name}'])

def replace_random(src, frm, to): # to randomly replace annotation with mask
    matches = list(re.finditer(frm, src))
    if len(matches) == 0:
        return src
    replace = choice(matches)
    return src[:replace.start()] + to + src[replace.end():]

df_train_expt['mask_code'] = df_train_expt['new_code'].values

df_train_expt_all_mask = df_train_expt.copy() 

for annotation in type_annotations_list: # mask all annotations
    df_train_expt_all_mask['mask_code'] = df_train_expt_all_mask['mask_code'].str.replace(annotation, "<mask>")

if args.prediction_mode == 'annot':
    # create duplicate entries and mask randomly 
    df_train_expt = pd.concat([df_train_expt]*3, ignore_index=True) 
    for annot in type_annotations_list:
        df_train_expt['mask_code'] = df_train_expt['mask_code'].apply(lambda x: replace_random(x, annot, "<mask>"))

    # merge, drop duplicates and shuffle
    df_train_expt = pd.concat([df_train_expt, df_train_expt_all_mask], ignore_index=True)
    df_train_expt.drop_duplicates(subset='mask_code', ignore_index=True, inplace=True)
    df_train_expt = df_train_expt.sample(frac=1).reset_index(drop=True)
    print("After creating all masks for annotation prediction: ", df_train_expt.shape)

else: # just use all mask
    df_train_expt = df_train_expt_all_mask.copy()
    print("After creating all masks: ", df_train_expt.shape)

all_new_cols_dfs = []
for graph_id in range(max_graphs):
    print("Processing for graph_id ", graph_id)
    df_x = df_train_expt.apply(lambda x: create_mapping(
        x[f'node_{graph_id}'], x[f'edge_{graph_id}'], graph_id), axis=1)
    all_new_cols_dfs.append(df_x)

df_train_expt['heatmap'] = df_train_expt.apply(lambda x: create_heatmap(x.new_code, x.error_line), axis=1)

df_x1 = df_train_expt.apply(lambda x: create_code_tokens(x['new_code'], 'code'), axis=1)
all_new_cols_dfs.append(df_x1)

df_x2 = df_train_expt.apply(lambda x: create_code_tokens(x['mask_code'], 'mask_code'), axis=1)
all_new_cols_dfs.append(df_x2)

for df in all_new_cols_dfs:
    print(df.shape, df.columns)

for df in all_new_cols_dfs:
    df_train_expt = df_train_expt.join(df)

df_train_expt['sample_weight'] = df_train_expt['input_ids_mask_code'].apply(lambda x: np.array([1 if i==mask_token_id else 0 for i in x[0]]))

if args.model_mode == "random_cfg":
    print("Random CFG")
    df_train_expt = df_train_expt.rename(columns = {'input_ids_0':'input_ids_2', 'input_ids_2':'input_ids_0', 'input_ids_1':'input_ids_7', 'input_ids_7':'input_ids_1', 
                                        'input_ids_5':'input_ids_3', 'input_ids_3':'input_ids_5', 'input_ids_4':'input_ids_6', 'input_ids_6':'input_ids_4', 
                                        'attention_mask_0':'attention_mask_2', 'attention_mask_2':'attention_mask_0', 'attention_mask_1':'attention_mask_7', 'attention_mask_7':'attention_mask_1', 
                                        'attention_mask_5':'attention_mask_3', 'attention_mask_3':'attention_mask_5', 'attention_mask_4':'attention_mask_6', 'attention_mask_6':'attention_mask_4'})

if args.model_mode == "random_code":
    print("Random Code")
    df_train_expt['input_ids_code'] = df_train_expt['input_ids_code'].sample(frac=1).values
    df_train_expt['attention_mask_code'] = df_train_expt['attention_mask_code'].sample(frac=1).values

df_train_expt = df_train_expt.sample(
    frac=1, random_state=42).reset_index(drop=True)

print("Shape after joining all sub-dfs: ", df_train_expt.shape)

print("Cols in DF: ", df_train_expt.columns.tolist())

train_size = int(0.8*len(df_train_expt))

print("Training set size: ", train_size)

class MultiGraphData(Data):
    def __init__(self, **kwargs):
        super().__init__()
        for key, value in kwargs.items():
            setattr(self, key, value)

    def __inc__(self, key, value, *args, **kwargs):
        if 'edge_index_' in key:
            n = key.split("_")[-1]
            return getattr(self, f"input_ids_{n}").size(0)
        else:
            return super().__inc__(key, value, *args, **kwargs)


all_data_train = []
all_data_test = []

c = 0

exp_cols = ['input_ids', 'attention_mask', 'edge_index']

for index, row in df_train_expt.iterrows():
    all_kwargs = {}
    # all_kwargs['max_graphs'] = max_graphs
    for ind_ in range(max_graphs):
        for col_ in exp_cols:
            # print(f'{col_}_{ind_}', row[f'{col_}_{ind_}'])
            if col_ == 'edge_index':
                if len(row[f'{col_}_{ind_}']) == 0:
                    all_kwargs[f'{col_}_{ind_}'] = torch.LongTensor([[0], [0]])
                else:
                    all_kwargs[f'{col_}_{ind_}'] = torch.from_numpy(
                        np.vstack(row[f'{col_}_{ind_}']).astype('int64')).t().contiguous()
            else:
                all_kwargs[f'{col_}_{ind_}'] = torch.from_numpy(
                    np.vstack(row[f'{col_}_{ind_}']).astype('int64'))

    all_kwargs['input_ids_code'] = torch.from_numpy(
        np.vstack(row['input_ids_code']).astype('int64'))
    all_kwargs['attention_mask_code'] = torch.from_numpy(
        np.vstack(row['attention_mask_code']).astype('int64'))

    all_kwargs['input_ids_mask_code'] = torch.from_numpy(
        np.vstack(row['input_ids_mask_code']).astype('int64'))
    all_kwargs['attention_mask_mask_code'] = torch.from_numpy(
        np.vstack(row['attention_mask_mask_code']).astype('int64'))

    all_kwargs['error_line'] = torch.IntTensor(row['error_line'])
    all_kwargs['heatmap'] = torch.IntTensor(row['heatmap'])
    all_kwargs['sample_weight'] = torch.IntTensor(row['sample_weight'])

    all_kwargs['y'] = row.label
    # data = Data(**all_kwargs)
    data = MultiGraphData(**all_kwargs)
    # data = PairData(y = all_kwargs['y'], input_ids_0 = all_kwargs['input_ids_0'], attention_mask_0 = all_kwargs['attention_mask_0'], edge_index_0 = all_kwargs['edge_index_0'],
    #                 input_ids_1 = all_kwargs['input_ids_1'], attention_mask_1 = all_kwargs['attention_mask_1'], edge_index_1 = all_kwargs['edge_index_1'])
    # data = MultiGraphData(y = all_kwargs['y'], input_ids_0 = all_kwargs['input_ids_0'], attention_mask_0 = all_kwargs['attention_mask_0'], edge_index_0 = all_kwargs['edge_index_0'])
    data.to(device)
    if c < train_size:
        all_data_train.append(data)
    else:
        all_data_test.append(data)
    c += 1

print("Columns not considered for the MultiGraphData class: ", [i for i in df_train_expt.columns.tolist() if i not in all_kwargs])
print("Train and test set length: ", len(all_data_train), len(all_data_test))

follow_batch = [f'input_ids_{i}' for i in range(max_graphs)]

train_loader = DataLoader(all_data_train, batch_size=args.batch_size,
                          shuffle=True, follow_batch=follow_batch)
test_loader = DataLoader(all_data_test, batch_size=args.batch_size,
                         shuffle=False, follow_batch=follow_batch)

for batch in test_loader:
    print(batch)
    break

annotation_tokens = [tokenizer.convert_tokens_to_ids(sp_token) for sp_token in type_annotations_list]
print("Annotation IDs: ", annotation_tokens)

class GCN(torch.nn.Module):
    def __init__(self, pretrained_model, hidden_channels, num_node_features, vocab_size, max_graphs):
        super(GCN, self).__init__()
        torch.manual_seed(42)

        self.pretrained = pretrained_model
        self.max_graphs = max_graphs
        self.concat_mode = None

        for param in self.pretrained.parameters():
            if args.model_mode == 'frozen_weights':
                print("Frozen Weights")
                param.requires_grad = False
            else:
                param.requires_grad = True

        if args.model_mode == 'cfg_only':
            print("CFG Only")
            concat_channels = hidden_channels*max_graphs
        else:
            concat_channels = hidden_channels*(max_graphs+1) # +1 for code
        
        if args.prediction_mode == 'annot':
            print("Task: Annotation prediction")
            num_classes = vocab_size
            self.concat_mode = 2 # the output is of the form batch_size, sentence_len, num_classes
        elif args.prediction_mode == 'heatmap':
            print("Task: Type Check heatmap prediction")
            num_classes = 2
            self.concat_mode = 2 # the output is of the form batch_size, sentence_len, num_classes
        elif args.prediction_mode == 'error':
            print("Task: Type Check error prediction")
            num_classes = 2
            self.concat_mode = 1 # the output is of the form batch_size, num_classes
        else:
            raise ValueError(f"{args.prediction_mode} task does not exist")

        assert self.concat_mode is not None

        # concat_channels = 256
        # self.lin_code = Linear(num_node_features, concat_channels)
        self.lin_code = Linear(num_node_features, hidden_channels)
        self.conv1 = GCNConv(num_node_features, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, hidden_channels)
        self.conv3 = GCNConv(hidden_channels, hidden_channels)
        self.lin_1 = Linear(concat_channels, concat_channels//2)
        self.lin_2 = Linear(concat_channels//2, num_classes)

    def forward(self, data):
        # 1. Obtain node embeddings
        all_graphs = []

        for graph_id in range(self.max_graphs):
            outputs = self.pretrained(input_ids = getattr(data, f"input_ids_{graph_id}"), attention_mask = getattr(data, f"attention_mask_{graph_id}"))
            # print("After loading a graph - pretrained ", graph_id)
            # GPUtil.showUtilization()
            x = torch.mean(outputs[0], dim=1)
            x = self.conv1(x, getattr(data, f"edge_index_{graph_id}"))
            x = x.relu()
            x = self.conv2(x, getattr(data, f"edge_index_{graph_id}"))
            x = x.relu()
            x = self.conv3(x, getattr(data, f"edge_index_{graph_id}"))

            # 2. Readout layer
            # print("Before pool: ", x.shape)
            x = global_mean_pool(x, getattr(data, f"input_ids_{graph_id}_batch"))  # [batch_size, hidden_channels]
            # print("After pool: ", x.shape)

            if self.concat_mode == 2:
                x = torch.unsqueeze(x, 1) # [batch_size, hidden_channels] -> [batch_size, 1, hidden_channels]
                x = x.repeat(1, MAX_LENGTH, 1) # [batch_size, hidden_channels] -> [batch_size, sentence_len, hidden_channels]

            all_graphs.append(x)
            
            # print("After loading a graph - full ", graph_id)
            # GPUtil.showUtilization()

        if args.model_mode != 'cfg_only':
            if args.prediction_mode == 'annot': # use the mask code input
                outputs = self.pretrained(input_ids=data.input_ids_mask_code, attention_mask=data.attention_mask_mask_code)
            else:
                outputs = self.pretrained(input_ids=data.input_ids_code, attention_mask=data.attention_mask_code)

            if self.concat_mode == 1:
                # (batch, sentence_len, 768) -> (batch, 768)
                x_code = torch.mean(outputs[0], dim=1)
                # RobertaClassificationHead takes only the first token! So, instead we take the mean across all tokens in the sentence.
                # https://stackoverflow.com/questions/62705268/why-bert-transformer-uses-cls-token-for-classification-instead-of-average-over
                # https://huggingface.co/transformers/v1.1.0/_modules/pytorch_transformers/modeling_roberta.html

            elif self.concat_mode == 2:
                # (batch, sentence_len, 768)
                x_code = outputs[0]
            
            # print(x_code.shape)

            x_code = self.lin_code(x_code)
            x_code = x_code.relu()
            all_graphs.append(x_code)

        if self.concat_mode == 1:
            x_concat = torch.cat(all_graphs, dim=1) # (batch, concat_dim)
        elif self.concat_mode == 2:
            x_concat = torch.cat(all_graphs, dim=2) # (batch, sentence_len, concat_dim)
        # print("After concat: ", x.shape)
        x = self.lin_1(x_concat)
        x = x.relu()
        # print("After lin_1: ", x.shape)
        x = F.dropout(x, p=0.2, training=self.training)

        # 3. Apply a final classifier
        logits = self.lin_2(x)

        # loss_fct = nn.CrossEntropyLoss()
        # loss = loss_fct(logits.view(-1, 2), data.y.view(-1))

        return SequenceClassifierOutput(
            loss=None, # calculated later on
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


print("Training started")

# from IPython.display import Javascript
# display(Javascript('''google.colab.output.setIframeHeight(0, true, {maxHeight: 300})'''))

model = GCN(pretrained_model=pretrained_model, hidden_channels=32,
            num_node_features=768, vocab_size=vocab_size, max_graphs=max_graphs)
optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)

if args.prediction_mode == 'annot':
    criterion = torch.nn.CrossEntropyLoss(reduction='none')
elif args.prediction_mode == 'heatmap':
    weights_heatmap = weights_heatmap.to(device)
    criterion = torch.nn.CrossEntropyLoss(reduction='none', weight=weights_heatmap)
elif args.prediction_mode == 'error':
    criterion = torch.nn.CrossEntropyLoss()
else:
    raise ValueError(f"{args.prediction_mode} task does not exist")

# if torch.cuda.device_count() > 1:
#     print("Let's use", torch.cuda.device_count(), "GPUs!")
#     # dim = 0 [30, xxx] -> [10, ...], [10, ...], [10, ...] on 3 GPUs
#     model = torch.nn.DataParallel(model)
model = model.to(device)
summary(model)

# print("Model initialized")
# GPUtil.showUtilization()

def numberToBase(n, b):
    if n == 0:
        return [0]
    digits = []
    while n:
        digits.append(int(n % b))
        n //= b
    return digits[::-1]

def train():
    model.train()
    accumulation_steps = args.accumulation_steps
    optimizer.zero_grad()

    for data_idx, data in enumerate(train_loader):  # Iterate in batches over the training dataset.
        data = data.to(device)
        out = model(data).logits

        if args.prediction_mode == 'heatmap':
            heatmap = data.heatmap
            heatmap = torch.reshape(heatmap, (-1,))
            heatmap = heatmap.to(torch.long)
            # ignoring it, using weight=weights_heatmap instead
            # sample_weight = heatmap + 1 # sample weight for heatmap prediction -> more penalty for misclassified errors (0->1, 1->2)
        elif args.prediction_mode == 'annot':
            sample_weight = data.sample_weight
            sample_weight = torch.reshape(sample_weight, (-1,))
        elif args.prediction_mode == 'error':
            pass
        else:
            raise NotImplementedError

        if args.prediction_mode in ['heatmap', 'annot']:
            input_ids_new_code = data.input_ids_code
            # print("Pre-reshape: ", out.shape, input_ids_new_code.shape) # torch.Size([2, 256, 50267]) torch.Size([2, 256])
            out_org = out[0]
            out_org = out_org.softmax(dim=1)
            out = torch.reshape(out, (-1, out.shape[-1]))
            out = out.softmax(dim=1)
            input_ids_new_code_org = input_ids_new_code[0]
            input_ids_new_code_org = input_ids_new_code_org.to(torch.long)
            input_ids_new_code = torch.reshape(input_ids_new_code, (-1,))
            input_ids_new_code = input_ids_new_code.to(torch.long)
            # print("Post-reshape: ", out.shape, input_ids_new_code.shape) # torch.Size([512, 50267]) torch.Size([512])
            # print("Post-reshape org: ", out_org.shape, input_ids_new_code_org.shape) # torch.Size([256, 50267]) torch.Size([256])

        # Compute the loss
        if args.prediction_mode == 'heatmap':
            loss = criterion(out, heatmap) 
            loss = loss.sum() / weights_heatmap[heatmap].sum()
            # loss = loss*sample_weight
            # loss = loss.mean()
        elif args.prediction_mode == 'annot':
            loss = criterion(out, input_ids_new_code) 
            loss = loss*sample_weight
            loss = loss.mean()
        elif args.prediction_mode == 'error':
            loss = criterion(out, data.y)  
        else:
            raise NotImplementedError


        if (data_idx+1) == len(train_loader) and args.prediction_mode == 'annot':
            # print(data.input_ids_mask_code, mask_token_id)

            masked_indices = np.where(data.input_ids_mask_code[0].cpu() == mask_token_id)
            mask_predictions = out_org[masked_indices]
            # print("masked_indices: ", masked_indices) # (array([ 85,  92,  97, 113]),)
            # print("mask_predictions", mask_predictions, mask_predictions.shape) # torch.Size([4, 50267])
            # print("out_org: ", out_org.shape) # torch.Size([256, 50267])
            # print("input_ids_new_code_org: ", input_ids_new_code_org)
            # top_n = 2
            max_choices = 1

            all_masked_dict = {} # for "sparse" beam search decoding -> decoding preference to annotation tokens

            for mask_combination_count in range(len(annotation_tokens)**len(mask_predictions)): # all possible combinations in the mask
                max_len = len(mask_predictions)
                base_eq = numberToBase(mask_combination_count,len(annotation_tokens))
                padded_base_eq = [0]*(max_len - len(base_eq)) + base_eq
                prob = 1
                for ind, annot_choice in enumerate(padded_base_eq):
                    prob *= mask_predictions[ind, annotation_tokens[annot_choice]]
                all_masked_dict[prob.float().item()] = padded_base_eq

            # print(all_masked_dict)

            all_prob_keys = list(all_masked_dict.keys())
            all_prob_keys.sort(reverse=True) # storing the prob in descending order

            all_annot_chosen = []
            for count, prob in enumerate(all_prob_keys):
                if count == max_choices:
                    break
                annot_chosen = []
                choices = all_masked_dict[prob]
                for annot in choices:
                    annot_chosen.append(annotation_tokens[annot])
                all_annot_chosen.append(annot_chosen)
                # print(annot_chosen)

            # top_indices = out_org.argsort()[:,-top_n:]
            # top_indices = torch.flip(top_indices, (1,))

            # all_masked_p = []
            # for i in range(top_n):
            #     all_masked_p.append([p[top_indices[num,i]].int().item() for num, p in enumerate(out_org)])

            for i in range(len(all_annot_chosen)):
                chosen_annotation_i = all_annot_chosen[i] # annotation ids chosen
                tokens = np.copy(input_ids_new_code_org.cpu())
                # print("tokens: ", tokens)
                for pos_id, mask_token_pos in enumerate(masked_indices[0]):
                    # print("mask_token_pos: ", mask_token_pos)
                    # print("Prediction: ", tokenizer.convert_ids_to_tokens([chosen_annotation_i[pos_id]]))
                    # print("Label: ", tokenizer.convert_ids_to_tokens([tokens[mask_token_pos]]))
                    tokens[mask_token_pos] = chosen_annotation_i[pos_id]
                    # print("tokenizer.convert_ids_to_tokens(p): ", pos_id, chosen_annotation_i[pos_id], tokenizer.convert_ids_to_tokens([chosen_annotation_i[pos_id]]))

                # print("tokens: ", tokens)

                result = {
                    "input_text": tokenizer.decode(input_ids_new_code_org.cpu().numpy()),
                    "prediction": tokenizer.decode(tokens),
                    "probability": all_prob_keys[i],
                    "predicted mask token": tokenizer.convert_ids_to_tokens(chosen_annotation_i),
                }

                pprint(result)


        # print("Model forward pass")
        # GPUtil.showUtilization()

        loss = loss/accumulation_steps 
        
        loss.backward()  # Derive gradients.
        
        if ((data_idx+1) % accumulation_steps == 0) or (data_idx+1 == len(train_loader)):
            optimizer.step()  # Update parameters based on gradients.
            optimizer.zero_grad()  # Clear gradients.

def test(loader, eval_flag, limit=None):
    model.eval()

    correct = 0
    total = 0
    y_pred = []
    y_true = []
    out_all = []
    all_detokenized = []

    count = -1

    for data in loader:  # Iterate in batches over the training/test dataset.
        count+=1
        if limit is not None and count == limit:
            print("Breaking out of the dataloader due to a limit of ", limit)
            break
        data = data.to(device)
        out = model(data).logits  # Perform a single forward pass.

        if args.prediction_mode in ['heatmap', 'annot']:
            heatmap = data.heatmap
            heatmap = torch.reshape(heatmap, (-1,))
            heatmap = heatmap.to(torch.long)

            out_org = out[0]
            out_org = out_org.softmax(dim=1)
            out = torch.reshape(out, (-1, out.shape[-1]))
            out = out.softmax(dim=1)

            input_ids_new_code = data.input_ids_code
            input_ids_new_code_org = input_ids_new_code[0]
            input_ids_new_code_org = input_ids_new_code_org.to(torch.long)
            input_ids_new_code = torch.reshape(input_ids_new_code, (-1,))
            input_ids_new_code = input_ids_new_code.to(torch.long)
        
        elif args.prediction_mode == 'error':
            pass # no additional steps required
        else:
            raise NotImplementedError
        
        if args.prediction_mode == 'heatmap':
            target = heatmap  # Compute the loss.
            # print("out: ", out.shape)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            # print("pred: ", pred.shape, pred)
            # print("target: ", target.shape, target)
            correct += int((pred == target).sum())  # Check against ground-truth labels.
            total += pred.shape[0]
            # print("correct/total: ", correct, total)
            y_pred.extend(pred.data.cpu().numpy())
            y_true.extend(target.data.cpu().numpy())
            out_all.extend(out.data.cpu().numpy())

        elif args.prediction_mode == 'annot': # TODO: currently only calculating for first element of batch
            target = input_ids_new_code  # Compute the loss.
            # print(input_ids_mask_code, mask_token_id)

            masked_indices = np.where(data.input_ids_mask_code[0].cpu() == mask_token_id)
            mask_predictions = out_org[masked_indices]
            # print("masked_indices: ", masked_indices)
            # print("mask_predictions", mask_predictions)
            # print("out_org: ", out_org.shape)
            # print("input_ids_new_code_org: ", input_ids_new_code_org.shape)

            all_masked_dict = {} # for "sparse" beam search decoding -> decoding preference to annotation tokens

            for mask_combination_count in range(len(annotation_tokens)**len(mask_predictions)): # all possible combinations in the mask
                max_len = len(mask_predictions)
                base_eq = numberToBase(mask_combination_count,len(annotation_tokens))
                padded_base_eq = [0]*(max_len - len(base_eq)) + base_eq
                prob = 1
                for ind, annot_choice in enumerate(padded_base_eq):
                    prob *= mask_predictions[ind, annotation_tokens[annot_choice]]
                all_masked_dict[prob.float().item()] = padded_base_eq

            print("all_masked_dict: ", all_masked_dict)

            all_prob_keys = list(all_masked_dict.keys())
            all_prob_keys.sort(reverse=True) # storing the prob in descending order

            all_annot_chosen = []
            for count, prob in enumerate(all_prob_keys):
                if count == top_n_annotation_pred:
                    break
                annot_chosen = []
                choices = all_masked_dict[prob]
                for annot in choices:
                    annot_chosen.append(annotation_tokens[annot])
                all_annot_chosen.append(annot_chosen)
                # print(f"Probability {prob} for {annot_chosen}")

            best_match = 0
            best_match_id = 0

            for i in range(len(all_annot_chosen)):
                chosen_annotation_i = all_annot_chosen[i] # annotation ids chosen
                tokens = np.copy(input_ids_new_code_org.cpu())
                # print("tokens: ", tokens)
                label_list_temp = []
                pred_list_temp = []
                for pos_id, mask_token_pos in enumerate(masked_indices[0]):
                    pred_list_temp.append(tokenizer.convert_ids_to_tokens([chosen_annotation_i[pos_id]])[0])
                    label_list_temp.append(tokenizer.convert_ids_to_tokens([tokens[mask_token_pos]])[0])
                current_match_count = int((np.array(pred_list_temp)==np.array(label_list_temp)).sum())
                if current_match_count > best_match: 
                    best_match = current_match_count
                    best_match_id = i

            chosen_annotation_i = all_annot_chosen[best_match_id] # best annotation ids chosen
            tokens = np.copy(input_ids_new_code_org.cpu())
            tokens_replaced = np.copy(input_ids_new_code_org.cpu())
            # print(f"best_match: {best_match} with id {best_match_id}")

            for pos_id, mask_token_pos in enumerate(masked_indices[0]):
                y_pred.append(tokenizer.convert_ids_to_tokens([chosen_annotation_i[pos_id]])[0])
                y_true.append(tokenizer.convert_ids_to_tokens([tokens[mask_token_pos]])[0])
                tokens_replaced[mask_token_pos] = chosen_annotation_i[pos_id]

            correct += best_match  # Check against ground-truth labels.
            total += len(masked_indices[0])
            # print("correct & total: ", correct, total)
            if eval_flag: # storing only for test/eval files
                all_detokenized.append(tokenizer.decode(tokens))

        elif args.prediction_mode == 'error':
            target = data.y  
            print("out: ", out.shape, out)
            pred = out.argmax(dim=1)  # Use the class with highest probability.
            print("pred: ", pred.shape, pred)
            print("target: ", target.shape, target)
            correct += int((pred == target).sum())  # Check against ground-truth labels.
            total += pred.shape[0]
            print("correct/total: ", correct, total)
            y_pred.extend(pred.data.cpu().numpy())
            y_true.extend(target.data.cpu().numpy())
            out_all.extend(out.data.cpu().numpy())
        
        else:
            raise NotImplementedError
    
    if args.prediction_mode == 'heatmap':
        classes_str = ['No error', 'Error']
        classes_arr = [0, 1]
    elif args.prediction_mode == 'annot':
        classes_str = type_annotations_list
        classes_arr = type_annotations_list
    elif args.prediction_mode == 'error':
        classes_str = ['No error', 'Error']
        classes_arr = [0, 1]
        print(total, len(loader.dataset))
        assert total == len(loader.dataset)
    else:
        raise NotImplementedError

    if eval_flag: # storing only for test/eval files
        with open('all_test_detokenized.pkl', 'wb') as f:
            pickle.dump(all_detokenized, f)

        # To retrieve use 
        # with open('parrot.pkl', 'rb') as f: 
        #   mynewlist = pickle.load(f)

    cf_matrix = confusion_matrix(y_true, y_pred, labels=classes_arr)
    df_cm = pd.DataFrame(cf_matrix, index = [i for i in classes_str],
                        columns = [i for i in classes_str])
    print("Confusion matrix and classification report: ")
    print(df_cm)
    print(classification_report(y_true, y_pred, labels=classes_arr, target_names=classes_str))
    report_dict = classification_report(y_true, y_pred, labels=classes_arr, target_names=classes_str, output_dict=True)
    # plt.figure(figsize = (5,5))
    # sn.heatmap(df_cm, annot=True, cmap='Blues', fmt='g')
    # plt.xlabel('Predicted')
    # plt.ylabel('Actual')
    # plt.savefig('output.png')
    return correct / total, df_cm, report_dict, y_true, y_pred, out_all, classes_str

for epoch in range(1, args.n_epochs):
    start_time = time.time()
    train()
    print("After training epoch ", epoch)
    GPUtil.showUtilization()
    all_train_op = test(train_loader, eval_flag=False, limit=10)
    train_acc = all_train_op[0]
    print("After training metric for epoch ", epoch)
    GPUtil.showUtilization()
    test_acc, df_cm, report_dict, y_true, y_pred, out_all, classes_str = test(test_loader, eval_flag=True)
    if args.prediction_mode in ['error', 'heatmap']:
        wandb.log({"pr" : wandb.plot.pr_curve(y_true, out_all,
                        labels=classes_str)})
        wandb.log({"roc" : wandb.plot.roc_curve(y_true, out_all,
                        labels=classes_str)})
        wandb.log({"cm" : wandb.sklearn.plot_confusion_matrix(y_true, y_pred,
                        labels=classes_str)})         
    table = wandb.Table(dataframe=df_cm)
    wandb.log({'test_confusion_matrix': table})
    wandb.log({'train_accuracy': train_acc, 'test_accuracy': test_acc, 'epoch': epoch})
    wandb.log(report_dict)
    print(f'Epoch: {epoch:03d}, Time: {int(time.time()-start_time)}, Train Acc: {train_acc:.4f}, Test Acc: {test_acc:.4f}')
    torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'acc': test_acc,
            }, f"saved_models/{args.model_name}_epc{epoch}_acc{test_acc:.4f}.pt")

wandb.save(f"saved_models/{args.model_name}_epc{epoch}_acc{test_acc:.4f}.pt")

# # t1 = tokenizer(df_train_expt['new_code'].to_list(), truncation=True, padding='max_length', max_length=MAX_LENGTH)

# # dataset = Dataset.from_pandas(df)
# # annot_pred_dataset['train'] = dataset

# from transformers import DataCollatorWithPadding

# def preprocess_function(examples):
#     return tokenizer(examples["new_code"], truncation=True)

# tokenized_data = annot_pred_dataset.map(preprocess_function, batched=True)
# data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# tokenized_data

# import numpy as np
# from datasets import load_metric

# metric = load_metric("accuracy")

# def compute_metrics(eval_pred):
#     logits, labels = eval_pred
#     predictions = np.argmax(logits, axis=-1)
#     return metric.compute(predictions=predictions, references=labels)


# from torchinfo import summary

# summary(model)

# tokenized_data.set_format("pandas")

# dataset = Dataset.from_pandas(df)

# dataset

# tokenized_data['train'] = dataset

# tokenized_data

# model = RobertaModel.from_pretrained("microsoft/codebert-base")

# import torch.nn as nn

# model.pooler.dense.in_features, model.pooler.dense.out_features

# # model.pooler.dense = nn.Linear(in_features=1024, out_features=1000, bias=True)

# # model

# # (output): RobertaOutput(
# #           (dense): Linear(in_features=3072, out_features=768, bias=True)
# #           (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
# #           (dropout): Dropout(p=0.1, inplace=False)
# #         )
# #       )
# #     )
# #   )
# #   (pooler): RobertaPooler(
# #     (dense): Linear(in_features=768, out_features=768, bias=True)
# #     (activation): Tanh()
# #   )
# # )

# # (output): RobertaOutput(
# #             (dense): Linear(in_features=3072, out_features=768, bias=True)
# #             (LayerNorm): LayerNorm((768,), eps=1e-05, elementwise_affine=True)
# #             (dropout): Dropout(p=0.1, inplace=False)
# #           )
# #         )
# #       )
# #     )
# #   )
# #   (classifier): RobertaClassificationHead(
# #     (dense): Linear(in_features=768, out_features=768, bias=True)
# #     (dropout): Dropout(p=0.1, inplace=False)
# #     (out_proj): Linear(in_features=768, out_features=2, bias=True)
# #   )
# # )

# # Old implementation

# tokenized_data = tokenized_data.remove_columns(['filename', 'code', 'fuzztag', 'rawfile', 'new_code', 'edges', 'nodes'])
# tokenized_data = tokenized_data.rename_column("label", "labels")
# tokenized_data.set_format("torch")
# tokenized_data["train"].column_names

# from torch.utils.data import DataLoader

# train_dataloader = DataLoader(
#     tokenized_data["train"], shuffle=True, batch_size=8, collate_fn=data_collator
# )
# eval_dataloader = DataLoader(
#     tokenized_data["test"], batch_size=8, collate_fn=data_collator
# )

# for batch in train_dataloader:
#     break
# {k: v.shape for k, v in batch.items()}

# from transformers import AutoModelForSequenceClassification

# model = AutoModelForSequenceClassification.from_pretrained("microsoft/codebert-base", num_labels=2)

# # model

# outputs = model(**batch)
# print(outputs.loss, outputs.logits.shape)

# from transformers import AdamW

# optimizer = AdamW(model.parameters(), lr=5e-5)

# from transformers import get_scheduler

# num_epochs = 3
# num_training_steps = num_epochs * len(train_dataloader)
# lr_scheduler = get_scheduler(
#     "linear",
#     optimizer=optimizer,
#     num_warmup_steps=0,
#     num_training_steps=num_training_steps,
# )
# print(num_training_steps)

# import torch

# device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# model.to(device)
# device

# from tqdm.auto import tqdm

# progress_bar = tqdm(range(num_training_steps))

# model.train()
# for epoch in range(num_epochs):
#     for batch in train_dataloader:
#         batch = {k: v.to(device) for k, v in batch.items()}
#         outputs = model(**batch)
#         loss = outputs.loss
#         loss.backward()

#         optimizer.step()
#         lr_scheduler.step()
#         optimizer.zero_grad()
#         progress_bar.update(1)

# from datasets import load_metric

# metric = load_metric("glue", "mrpc")
# model.eval()
# for batch in eval_dataloader:
#     batch = {k: v.to(device) for k, v in batch.items()}
#     with torch.no_grad():
#         outputs = model(**batch)

#     logits = outputs.logits
#     predictions = torch.argmax(logits, dim=-1)
#     metric.add_batch(predictions=predictions, references=batch["labels"])

# metric.compute()

# print(annot_pred_dataset["train"][0]["code"])

# new_code = """
# public class B1C3G4J4J1J0_DivideByZero8_n{
#     static String nullness_test() {
#         String z1 = "not null";
#         try {
#             String var_0 = null;
#             double data_w = 50 % 0.0;
#         } catch (Exception e) {
#             z1 = null;
#         }

#         return z1;
#     }
# }
# """

# new_code_tok = tokenizer(new_code, truncation=True, return_tensors="pt")
# encoded_input = {k: v.to(device) for k, v in new_code_tok.items()}

# # tokenizer.tokenize(new_code)

# model_out = model(**encoded_input)

# logits = model_out.logits
# predictions = torch.argmax(logits, dim=-1)
# predictions
