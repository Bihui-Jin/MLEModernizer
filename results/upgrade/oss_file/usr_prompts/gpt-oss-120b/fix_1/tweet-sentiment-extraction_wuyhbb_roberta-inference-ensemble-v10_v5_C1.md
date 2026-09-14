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

3.8

# 3. Installed packages

geopandas==0.14.4
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

0.727962076663971

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import argparse
import json
import os
import random
import re
import shutil
from collections import OrderedDict, defaultdict
from functools import partial
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
import tqdm


from sklearn.model_selection import GroupKFold
from torch import nn
from torch.nn import functional as F

from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from transformers import RobertaConfig, RobertaModel, RobertaTokenizer, AutoConfig, AutoModel, AutoTokenizer
from transformers.optimization import (AdamW, get_cosine_schedule_with_warmup,
                                       get_linear_schedule_with_warmup,
                                       get_cosine_with_hard_restarts_schedule_with_warmup)

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from shutil import copyfile
copyfile(src = "../input/utils-v10/utilsv10.py", dst = "../working/utilsv10.py")
copyfile(src = "../input/utils-v10/dataset10.py", dst = "../working/dataset10.py")

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2285780932.py in <cell line: 0>()
      1 from shutil import copyfile
----> 2 copyfile(src = "../input/utils-v10/utilsv10.py", dst = "../working/utilsv10.py")
      3 copyfile(src = "../input/utils-v10/dataset10.py", dst = "../working/dataset10.py")

/usr/lib/python3.11/shutil.py in copyfile(src, dst, follow_symlinks)
    254         os.symlink(os.readlink(src), dst)
    255     else:
--> 256         with open(src, 'rb') as fsrc:
    257             try:
    258                 with open(dst, 'wb') as fdst:

FileNotFoundError: [Errno 2] No such file or directory: '../input/utils-v10/utilsv10.py'

## === cell 2
from utilsv10 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words,
                   load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits, map_to_word)

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1915634673.py in <cell line: 0>()
----> 1 from utilsv10 import (binary_focal_loss, get_learning_rate, jaccard_list, get_best_pred, ensemble, ensemble_words,
      2                    load_model, save_model, set_seed, write_event, evaluate, get_predicts_from_token_logits, map_to_word)

ModuleNotFoundError: No module named 'utilsv10'

## === cell 3
from dataset10 import TrainDataset, MyCollator

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/3251614277.py in <cell line: 0>()
----> 1 from dataset10 import TrainDataset, MyCollator

ModuleNotFoundError: No module named 'dataset10'

## === cell 5
test = pd.read_csv('../input/tweet-sentiment-extraction/test.csv')

## === cell 6
tokenizer = AutoTokenizer.from_pretrained('../input/roberta-base/', do_lower_case=False)

## --- ERROR in cell 6, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-base/'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/tmp/ipykernel_11/2941377962.py in <cell line: 0>()
----> 1 tokenizer = AutoTokenizer.from_pretrained('../input/roberta-base/', do_lower_case=False)

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-base/'. Use `repo_type` argument if needed.

## === cell 7
class Args:
    post = False
    tokenizer = tokenizer
    offset = 4
    batch_size = 16
    workers = 1
args = Args()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2162057798.py in <cell line: 0>()
----> 1 class Args:
      2     post = False
      3     tokenizer = tokenizer
      4     offset = 4
      5     batch_size = 16

/tmp/ipykernel_11/2162057798.py in Args()
      1 class Args:
      2     post = False
----> 3     tokenizer = tokenizer
      4     offset = 4
      5     batch_size = 16

NameError: name 'tokenizer' is not defined

## === cell 9
collator = MyCollator()
test_set = TrainDataset(test, None, tokenizer=tokenizer, mode='test', offset=args.offset)
test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
                                 num_workers=args.workers)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/960771275.py in <cell line: 0>()
----> 1 collator = MyCollator()
      2 test_set = TrainDataset(test, None, tokenizer=tokenizer, mode='test', offset=args.offset)
      3 test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, collate_fn=collator,
      4                                  num_workers=args.workers)

NameError: name 'MyCollator' is not defined

## === cell 11
class TweetModel(nn.Module):

    def __init__(self, pretrain_path=None, dropout=0.2, config=None):
        super(TweetModel, self).__init__()
        if config is not None:
            self.bert = AutoModel.from_config(config)
        else:
            config = AutoConfig.from_pretrained(pretrain_path, output_hidden_states=True)
            self.bert = AutoModel.from_pretrained(
                pretrain_path, cache_dir=None, config=config)
        
        self.cnn =  nn.Conv1d(self.bert.config.hidden_size*3, self.bert.config.hidden_size, 3, padding=1)

        self.gelu = nn.GELU()

        self.whole_head = nn.Sequential(OrderedDict([
            ('dropout', nn.Dropout(0.1)),
            ('l1', nn.Linear(self.bert.config.hidden_size*3, 256)),
            ('act1', nn.GELU()),
            ('dropout', nn.Dropout(0.1)),
            ('l2', nn.Linear(256, 2))
        ]))
        self.se_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.inst_head = nn.Linear(self.bert.config.hidden_size, 2)
        self.dropout = nn.Dropout(0.1)

    def forward(self, inputs, masks, token_type_ids=None, input_emb=None):
        _, pooled_output, hs = self.bert(
            inputs, masks, token_type_ids=token_type_ids, inputs_embeds=input_emb)

        seq_output = torch.cat([hs[-1],hs[-2],hs[-3]], dim=-1)


        avg_output = torch.sum(seq_output*masks.unsqueeze(-1), dim=1, keepdim=False)
        avg_output = avg_output/torch.sum(masks, dim=-1, keepdim=True)
        whole_out = self.whole_head(avg_output)

        seq_output = self.gelu(self.cnn(seq_output.permute(0,2,1)).permute(0,2,1))
        
        se_out = self.se_head(self.dropout(seq_output))  #()
        inst_out = self.inst_head(self.dropout(seq_output))
        return whole_out, se_out[:, :, 0], se_out[:, :, 1], inst_out

## === cell 12
def predict(model: nn.Module, valid_df, valid_loader, args, progress=False) -> Dict[str, float]:
    model.eval()
    all_end_pred, all_whole_pred, all_start_pred, all_inst_out = [], [], [], []
    if progress:
        tq = tqdm.tqdm(total=len(valid_df))
    with torch.no_grad():
        for tokens, types, masks, _, _, _, _, _, _, _ in valid_loader:
            if progress:
                batch_size = tokens.size(0)
                tq.update(batch_size)
            masks = masks.cuda()
            tokens = tokens.cuda()
            types = types.cuda()
            whole_out, start_out, end_out, inst_out = model(tokens, masks, types)
            start_out = start_out.masked_fill(~masks.bool(), -1000)
            end_out= end_out.masked_fill(~masks.bool(), -1000)   

            start_out = torch.softmax(start_out, dim=-1)
            end_out = torch.softmax(end_out, dim=-1)

            all_whole_pred.append(torch.softmax(whole_out, dim=-1)[:,1].cpu().numpy())
            inst_out = torch.softmax(inst_out, dim=-1)
            for idx in range(len(start_out)):
                length = torch.sum(masks[idx,:]).item()-1 # -1 for last token
                all_start_pred.append(torch.softmax(start_out[idx, args.offset:length], axis=-1).cpu())
                all_end_pred.append(torch.softmax(end_out[idx, args.offset:length], axis=-1).cpu())
                all_inst_out.append(inst_out[idx,:,1].cpu())
            assert all_start_pred[-1].dim()==1

    all_whole_pred = np.concatenate(all_whole_pred)
    
    if progress:
        tq.close()
    return all_whole_pred, all_start_pred, all_end_pred, all_inst_out

## === cell 14
config = RobertaConfig.from_pretrained('../input/roberta-base', output_hidden_states=True)
model = TweetModel(config=config)





## --- ERROR in cell 14, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-base'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/roberta-base'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2305155619.py in <cell line: 0>()
----> 1 config = RobertaConfig.from_pretrained('../input/roberta-base', output_hidden_states=True)
      2 model = TweetModel(config=config)
      3 
      4 # load_model(model, '../input/large-uncased-wwm-weights/best-model-0.pt')
      5 

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/roberta-base'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/roberta-base' is the correct path to a directory containing a config.json file

## === cell 15
all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = [], [], [], []

    
for fold in range(5):
    load_model(model, '../input/roberta-weights-v10/best-model-%d.pt' % fold)
    model.cuda()
    fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(model, test, test_loader, args, progress=True)

    all_whole_preds.append(fold_whole_preds)
    all_start_preds.append(fold_start_preds)
    all_end_preds.append(fold_end_preds)
    all_inst_preds.append(fold_inst_preds)


all_whole_preds, all_start_preds, all_end_preds, all_inst_preds = ensemble(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test)
word_preds, inst_word_preds, scores = get_predicts_from_token_logits(all_whole_preds, all_start_preds, all_end_preds, all_inst_preds, test, args)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4026074914.py in <cell line: 0>()
      3 
      4 for fold in range(5):
----> 5     load_model(model, '../input/roberta-weights-v10/best-model-%d.pt' % fold)
      6     model.cuda()
      7     fold_whole_preds, fold_start_preds, fold_end_preds, fold_inst_preds = predict(model, test, test_loader, args, progress=True)

NameError: name 'load_model' is not defined

## === cell 16
test['selected_text'] = word_preds
def f(selected):
    return " ".join(set(selected.lower().split()))
test[['textID','selected_text']].to_csv('submission.csv', index=False)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1046239687.py in <cell line: 0>()
----> 1 test['selected_text'] = word_preds
      2 # test.loc[replace_idx, 'selected_text'] = test.loc[replace_idx, 'text']
      3 def f(selected):
      4     return " ".join(set(selected.lower().split()))
      5 # test.selected_text = test.selected_text.map(f)

NameError: name 'word_preds' is not defined

## === cell 17
test.head(20)
