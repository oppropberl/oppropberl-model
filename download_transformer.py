import torch
from transformers import RobertaTokenizer, RobertaConfig, RobertaModel, AutoTokenizer, AutoModelForMaskedLM, AutoModel

tokenizer = RobertaTokenizer.from_pretrained("microsoft/codebert-base")#, cache_dir="./new_cache_dir/", local_files_only=True)
model = RobertaModel.from_pretrained("microsoft/codebert-base")#, cache_dir="./new_cache_dir/", local_files_only=True)

# tokenizer = AutoTokenizer.from_pretrained("microsoft/graphcodebert-base")#, cache_dir="./new_cache_dir/", local_files_only=True)
# model = AutoModel.from_pretrained("microsoft/graphcodebert-base")#, cache_dir="./new_cache_dir/", local_files_only=True)

tokenizer.save_pretrained('./saved_transformer/saved_tokenizer_model/')
model.save_pretrained('./saved_transformer/saved_model/')

from datasets import load_metric

metric = load_metric("accuracy")