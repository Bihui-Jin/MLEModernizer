# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Predict the word or phrase from tweets that exemplifies the labelled sentiment.

## Metric
Word-level Jaccard score.

## Submission Format
For each ID in the test set, you must predict the string that best supports the sentiment for the tweet in question. Note that the selected text _needs_ to be **quoted** and **complete** (include punctuation, etc. - the above code splits ONLY on whitespace) to work correctly. The file should contain a header and have the following format:
```
textID,selected_text
2,"very good"
5,"I don't care"
6,"bad"
8,"it was, yes"
etc.
```

## Dataset
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

- `textID` - unique ID for each piece of text
- `text` - the text of the tweet
- `sentiment` - the general sentiment of the tweet
- `selected_text` - [train only] the text that supports the tweet's sentiment

# 2. Python version

3.11

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
seaborn==0.12.2
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1
transformers==4.53.3

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        input/
            description.md (130 lines)
            sample_submission.csv (2750 lines)
            sample_submission.csv.zip (18.6 kB)
            test.csv (2750 lines)
            test.csv.zip (114.6 kB)
            train.csv (24733 lines)
            train.csv.zip (1.2 MB)
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
        working/
            tweet-sentiment-extraction/
                description.md (130 lines)
                sample_submission.csv (2750 lines)
                ... and 5 other files
                tweet-sentiment-extraction/
```

-> data/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> data/tweet-sentiment-extraction/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> data/tweet-sentiment-extraction/test.csv has 2749 rows and 3 columns.
The columns are: textID, text, sentiment

-> data/tweet-sentiment-extraction/train.csv has 24732 rows and 4 columns.
The columns are: textID, text, selected_text, sentiment

-> input/sample_submission.csv has 2749 rows and 2 columns.
The columns are: textID, selected_text

-> (stopped after 10 files for performance)

# 5. Target score

0.2875462770462036

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import gc
import random
import time
import math
import re
import string
import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torch.optim import Adam, SGD
import torch.nn.functional as F
from torch.cuda.amp import autocast, GradScaler

import numpy as np
import pandas as pd
import scipy as sp
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.utils.class_weight import compute_class_weight
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split, KFold, GroupKFold, StratifiedKFold, StratifiedGroupKFold

from tqdm import tqdm
from transformers import (
    AutoTokenizer, 
    AutoModel, 
    AutoConfig, 
    AdamW, 
    get_linear_schedule_with_warmup, 
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device= torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CFG:
    DEBUG= False
    TRAIN= True
    N_FOLDS= 5
    TRAIN_FOLDS= [i for i in range(N_FOLDS)]
    SEED= 42
    TEST_BATCHSIZE= 100
    MAX_LENGTH = 128
    MODEL_NAME= 'microsoft/deberta-v3-base'
    FC_DROPOUT= [0.1, 0.2, 0.3, 0.4, 0.5]

## === cell 2
def seed_everything(seed= 42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    
seed_everything(seed= CFG.SEED)

## === cell 3
test_df = pd.read_csv('/kaggle/input/tweet-sentiment-extraction/test.csv')
test_df

## === cell 4
tokenizer= AutoTokenizer.from_pretrained('/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/tokenizer/', use_fast= True)
CFG.TOKENIZER= tokenizer

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/tokenizer/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/3340915134.py in <cell line: 0>()
----> 1 tokenizer= AutoTokenizer.from_pretrained('/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/tokenizer/', use_fast= True)
      2 CFG.TOKENIZER= tokenizer

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
    981 
    982         # Next, let's try to use the tokenizer_config file to get the tokenizer class.
--> 983         tokenizer_config = get_tokenizer_config(pretrained_model_name_or_path, **kwargs)
    984         if "_commit_hash" in tokenizer_config:
    985             kwargs["_commit_hash"] = tokenizer_config["_commit_hash"]

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in get_tokenizer_config(pretrained_model_name_or_path, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, **kwargs)
    813 
    814     commit_hash = kwargs.get("_commit_hash", None)
--> 815     resolved_config_file = cached_file(
    816         pretrained_model_name_or_path,
    817         TOKENIZER_CONFIG_FILE,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    520 
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    521         # Now we try to recover if we can find all files correctly in the cache
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]
    525         if all(file is not None for file in resolved_files):

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    138 ):
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:
    142         return resolved_file

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    104         ):
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 
    108             elif arg_name == "token" and arg_value is not None:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    152 
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"
    156             f" '{repo_id}'. Use `repo_type` argument if needed."

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/tokenizer/'. Use `repo_type` argument if needed.

## === cell 5
collate_fn= DataCollatorWithPadding(CFG.TOKENIZER, padding= 'longest', return_tensors= 'pt')

class QADataset:
    def __init__(self, df):
        self.df = df
        
    def __len__(self):
        return len(self.df)
  
    def __getitem__(self, item):
        text= " ".join(str(self.df.text.iloc[item]).split())
    
        inputs= CFG.TOKENIZER(text,
                              add_special_tokens= True,
                              max_length= CFG.MAX_LENGTH,
                              padding = 'max_length',
                              truncation = True,
                              return_offsets_mapping= True)
    
        for k, v in inputs.items():
            inputs[k]= torch.tensor(v, dtype= torch.long)
            
        tok_text_tokens= CFG.TOKENIZER.convert_ids_to_tokens(inputs.input_ids)
        
        sentiment = [1, 0, 0]
        if self.df.sentiment.iloc[item] == "positive":
            sentiment = [0, 0, 1]
        if self.df.sentiment.iloc[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            'input_ids': inputs['input_ids'],
            'mask': inputs['attention_mask'],
            'text_tokens': ' '.join(tok_text_tokens),
            'sentiment': torch.tensor(sentiment, dtype= torch.long),
            'orig_text': self.df.text.iloc[item],
            'orig_sentiment': self.df.sentiment.iloc[item]
        }

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3771996765.py in <cell line: 0>()
----> 1 collate_fn= DataCollatorWithPadding(CFG.TOKENIZER, padding= 'longest', return_tensors= 'pt')
      2 
      3 class QADataset:
      4     def __init__(self, df):
      5         self.df = df

NameError: name 'DataCollatorWithPadding' is not defined

## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path= None, pretrained= False):
        super().__init__()
        if config_path is None:
            self.config= AutoConfig.from_pretrained(CFG.MODEL_NAME, output_hidden_states= True)
        else:
            self.config= torch.load(config_path)

        if pretrained:
            self.backbone= AutoModel.from_pretrained(CFG.MODEL_NAME, config= self.config)
        else:
            self.backbone= AutoModel.from_config(self.config)
            
        self.fc_dropout= nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])
        
        self.attention_head= nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Softmax(dim= 1)
        )
        
        self.fc= nn.Linear(self.config.hidden_size, 2)
        
    def initialize_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean= 0, std= 1)
            
    def forward(self, input_ids, mask):
        embeddings= self.backbone(input_ids= input_ids, attention_mask= mask).last_hidden_state
        
        logits= self.fc(embeddings)
        start_logits, end_logits= logits.split(1, dim= -1)
        start_logits= start_logits.squeeze(-1)
        end_logits= end_logits.squeeze(-1)

        return start_logits, end_logits

## === cell 7
def test_fn(dataloader, model):
    model.eval()
    val_loss= 0

    fin_output_start= []
    fin_output_end= []
    fin_mask= []
    fin_text_tokens= []
    fin_orig_text= []
    fin_orig_selected= []
    fin_orig_sentiment= []

    with torch.no_grad():
        for data in tqdm(dataloader, total= len(dataloader)):
            input_ids= data['input_ids'].to(device)
            mask= data['mask'].to(device)
            text_tokens= data['text_tokens']
            orig_text= data['orig_text']
            orig_sentiment= data['orig_sentiment']
      
            start_logits, end_logits= model(input_ids, mask)
            fin_output_start.append(start_logits.cpu().detach().numpy())
            fin_output_end.append(end_logits.cpu().detach().numpy())
            fin_mask.append(mask.cpu().detach().numpy())
            
            fin_text_tokens.extend(text_tokens)
            fin_orig_text.extend(orig_text)
            fin_orig_sentiment.extend(orig_sentiment)

            
    fin_output_start= np.vstack(fin_output_start)
    fin_output_end= np.vstack(fin_output_end)
    fin_mask = np.vstack(fin_mask)

    return fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text, fin_orig_selected, fin_orig_sentiment

## === cell 8
test_dataset= QADataset(test_df)
test_loader= DataLoader(test_dataset, batch_size= CFG.TEST_BATCHSIZE, shuffle= False, num_workers = 2, pin_memory=True)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3852026347.py in <cell line: 0>()
----> 1 test_dataset= QADataset(test_df)
      2 test_loader= DataLoader(test_dataset, batch_size= CFG.TEST_BATCHSIZE, shuffle= False, num_workers = 2, pin_memory=True)

NameError: name 'QADataset' is not defined

## === cell 9
for idx, i in enumerate(CFG.TRAIN_FOLDS):
    model= QAModel(config_path= '/kaggle/input/feedback-deberta-baseline-train/config.pth').to(device)
    model.load_state_dict(torch.load(f'/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/QAbert{i}.pth'))
    if idx == 0:
        fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text, fin_orig_selected, fin_orig_sentiment = test_fn(test_loader, model)
    else:
        a, b, c, fin_text_tokens, fin_orig_text, fin_orig_selected, fin_orig_sentiment = test_fn(test_loader, model)
        fin_output_start, fin_output_end, fin_mask = np.add(fin_output_start, a), np.add(fin_output_end, b), np.add(fin_mask, c)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/620942964.py in <cell line: 0>()
      1 for idx, i in enumerate(CFG.TRAIN_FOLDS):
----> 2     model= QAModel(config_path= '/kaggle/input/feedback-deberta-baseline-train/config.pth').to(device)
      3     model.load_state_dict(torch.load(f'/kaggle/input/k/sagarikajadon/tweet-sentiment-extraction/QAbert{i}.pth'))
      4     if idx == 0:
      5         fin_output_start, fin_output_end, fin_mask, fin_text_tokens, fin_orig_text, fin_orig_selected, fin_orig_sentiment = test_fn(test_loader, model)

/tmp/ipykernel_11/1058539900.py in __init__(self, config_path, pretrained)
      5             self.config= AutoConfig.from_pretrained(CFG.MODEL_NAME, output_hidden_states= True)
      6         else:
----> 7             self.config= torch.load(config_path)
      8 
      9         if pretrained:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in load(f, map_location, pickle_module, weights_only, mmap, **pickle_load_args)
   1423         pickle_load_args["encoding"] = "utf-8"
   1424 
-> 1425     with _open_file_like(f, "rb") as opened_file:
   1426         if _is_zipfile(opened_file):
   1427             # The zipfile reader is going to advance the current file position.

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in _open_file_like(name_or_buffer, mode)
    749 def _open_file_like(name_or_buffer, mode):
    750     if _is_path(name_or_buffer):
--> 751         return _open_file(name_or_buffer, mode)
    752     else:
    753         if "w" in mode:

/usr/local/lib/python3.11/dist-packages/torch/serialization.py in __init__(self, name, mode)
    730 class _open_file(_opener):
    731     def __init__(self, name, mode):
--> 732         super().__init__(open(name, mode))
    733 
    734     def __exit__(self, *args):

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/input/feedback-deberta-baseline-train/config.pth'

## === cell 10
fin_output_start = np.divide(fin_output_start, 5)
fin_output_end = np.divide(fin_output_end, 5)
fin_mask = np.divide(fin_mask, 5)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2814391878.py in <cell line: 0>()
----> 1 fin_output_start = np.divide(fin_output_start, 5)
      2 fin_output_end = np.divide(fin_output_end, 5)
      3 fin_mask = np.divide(fin_mask, 5)

NameError: name 'fin_output_start' is not defined

## === cell 11
threshold= 0.3
final_outputs = []
for j in range(len(fin_text_tokens)):
    text_token= fin_text_tokens[j]
    mask= fin_mask[j]
    sentiment= fin_orig_sentiment[j]

    mask_start= fin_output_start[j]* mask
    mask_start= mask_start>= threshold

    mask_end= fin_output_end[j]* mask
    mask_end= mask_end>= threshold

    mask= [0] * len(mask)
    idx_start= np.nonzero(mask_start)[0]
    idx_end= np.nonzero(mask_end)[0]

    if len(idx_start)> 0:
        idx_start= idx_start[0]
        if len(idx_end)> 0:
            idx_end= idx_end[0]
        else:
            idx_end= idx_start
    else:
        idx_start= 0
        idx_end= 0

    for mj in range(idx_start, idx_end+ 1):
        mask[mj]= 1

    output_tokens= [x for i, x in enumerate(text_token.split()) if mask[i]== 1]
    output_tokens= [x for x in output_tokens if x not in ('[CLS]', '[SEP]')]

    final_output= output_tokens[0][1:]
    for ot in output_tokens[1:]:
        if ot.startswith('▁'):
            final_output= final_output + ' ' + ot[1:]
        elif len(ot)== 1 and ot in string.punctuation:
            final_output= final_output + ot
        
    final_outputs.append(final_output)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3540772133.py in <cell line: 0>()
      1 threshold= 0.3
      2 final_outputs = []
----> 3 for j in range(len(fin_text_tokens)):
      4     text_token= fin_text_tokens[j]
      5 #         print(text_token)

NameError: name 'fin_text_tokens' is not defined

## === cell 12
test_ids = test_df['textID']
sub = pd.DataFrame({'textID': test_ids, 'selected_text': final_outputs})

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1567859176.py in <cell line: 0>()
      1 test_ids = test_df['textID']
----> 2 sub = pd.DataFrame({'textID': test_ids, 'selected_text': final_outputs})

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 0 does not match index length 2749

## === cell 13
sub

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/35866194.py in <cell line: 0>()
----> 1 sub

NameError: name 'sub' is not defined

## === cell 14
sub.to_csv('submission.csv', index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2634576230.py in <cell line: 0>()
----> 1 sub.to_csv('submission.csv', index=False)

NameError: name 'sub' is not defined
