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
Given questions and answers from various StackExchange properties, predict target values of 30 labels for each question-answer pair.

## Metric
Mean column-wise Spearman's correlation coefficient. The Spearman's rank correlation is computed for each target column, and the mean of these values is calculated for the submission score.

## Submission Format
For each qa_id in the test set, you must predict a probability for each target variable. The predictions should be in the range [0,1]. The file should contain a header and have the following format:

```
qa_id,question_asker_intent_understanding,...,answer_well_written
6,0.0,...,0.5
8,0.5,...,0.1
18,1.0,...,0.0
etc.
```

## Dataset
The list of 30 target labels are the same as the column names in the `sample_submission.csv` file. Target labels with the prefix `question_` relate to the `question_title` and/or `question_body` features in the data. Target labels with the prefix `answer_` relate to the `answer` feature.

Target labels are aggregated from multiple raters, and can have continuous values in the range `[0,1]`. Therefore, predictions must also be in that range.

- **train.csv** - the training data (target labels are the last 30 columns)
- **test.csv** - the test set (you must predict 30 labels for each test set row)
- **sample_submission.csv** - a sample submission file in the correct format; column names are the 30 target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        input/
            description.md (83 lines)
            sample_submission.csv (609 lines)
            sample_submission.csv.zip (8.9 kB)
            test.csv (19551 lines)
            test.csv.zip (471.7 kB)
            train.csv (159837 lines)
            train.csv.zip (4.2 MB)
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
        working/
            google-quest-challenge/
                description.md (83 lines)
                sample_submission.csv (609 lines)
                ... and 5 other files
                google-quest-challenge/
```

-> data/google-quest-challenge/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/google-quest-challenge/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/google-quest-challenge/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> data/sample_submission.csv has 608 rows and 31 columns.
The columns are: qa_id, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer, question_fact_seeking, question_has_commonly_accepted_answer, question_interestingness_others, question_interestingness_self, question_multi_intent, question_not_really_a_question, question_opinion_seeking, question_type_choice, question_type_compare, question_type_consequence... and 16 more columns

-> data/test.csv has 19550 rows and 11 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host

-> data/train.csv has 159836 rows and 41 columns.
The columns are: qa_id, question_title, question_body, question_user_name, question_user_page, answer, answer_user_name, answer_user_page, url, category, host, question_asker_intent_understanding, question_body_critical, question_conversational, question_expect_short_answer... and 26 more columns

-> (stopped after 10 files for performance)

# 5. Target score

0.3567914536991772

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.environ["TRANSFORMERS_OFFLINE"] = "1"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

import re
import numpy as np
import pandas as pd

import torch
from torch.autograd import Variable
from torch.utils.data import DataLoader, TensorDataset

from transformers import AutoTokenizer, BertModel



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")



## === cell 3
train.columns.values



## === cell 4
output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]



## === cell 5
question_output_columns = [col for col in output_columns if "question" in col]
answer_output_colmns = [
    col for col in output_columns if col not in question_output_columns
]



## === cell 6
len(question_output_columns)



## === cell 7
tokenizer = AutoTokenizer.from_pretrained(
    "bert-base-uncased", use_fast=True, local_files_only=True
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1903536164.py in <cell line: 0>()
      1 # Fix: old tokenizer path does not exist in this environment; use standard HF model id in offline mode.
      2 # In Kaggle, bert-base-uncased is typically available in the local cache; offline mode prevents network calls.
----> 3 tokenizer = AutoTokenizer.from_pretrained(
      4     "bert-base-uncased", use_fast=True, local_files_only=True
      5 )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/tokenization_auto.py in from_pretrained(cls, pretrained_model_name_or_path, *inputs, **kwargs)
   1001                     config = AutoConfig.for_model(**config_dict)
   1002                 else:
-> 1003                     config = AutoConfig.from_pretrained(
   1004                         pretrained_model_name_or_path, trust_remote_code=trust_remote_code, **kwargs
   1005                     )

/usr/local/lib/python3.11/dist-packages/transformers/models/auto/configuration_auto.py in from_pretrained(cls, pretrained_model_name_or_path, **kwargs)
   1195         code_revision = kwargs.pop("code_revision", None)
   1196 
-> 1197         config_dict, unused_kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
   1198         has_remote_code = "auto_map" in config_dict and "AutoConfig" in config_dict["auto_map"]
   1199         has_local_code = "model_type" in config_dict and config_dict["model_type"] in CONFIG_MAPPING

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 8
max_length_map = {"question_title": 32, "question_body": 512, "answer": 512}




## === cell 9
def txt_re(txt):
    txt = "" if txt is None else str(txt)
    txt = txt.strip()
    txt = re.sub("https?.*$", "", txt)
    txt = re.sub("https?.*\\s", "", txt)
    txt = re.sub("\\n+", " ", txt)
    txt = re.sub("\\r+", " ", txt)
    txt = re.sub("\\t+", " ", txt)
    txt = re.sub("&gt;", ">", txt)
    txt = re.sub("&lt;", "<", txt)
    txt = re.sub("&amp;", "&", txt)
    txt = re.sub("&quot;", '"', txt)
    return txt




## === cell 10
def get_input(txt, tokenizer, max_length):
    txt = txt_re(txt)
    enc = tokenizer.encode_plus(
        txt,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding="max_length",
        return_attention_mask=True,
        return_token_type_ids=True,
    )
    input_ids = enc["input_ids"]
    segment_masks = enc["token_type_ids"]
    input_masks = enc["attention_mask"]
    return input_ids, segment_masks, input_masks




## === cell 11
def computer_input_array(df):
    t_input_ids, t_segment_masks, t_input_masks = [], [], []
    q_input_ids, q_segment_masks, q_input_masks = [], [], []
    a_input_ids, a_segment_masks, a_input_masks = [], [], []
    for _, instance in df[input_columns].iterrows():
        title, question, answer = (
            instance.question_title,
            instance.question_body,
            instance.answer,
        )

        input_ids, segment_masks, input_masks = get_input(
            title, tokenizer, max_length_map["question_title"]
        )
        t_input_ids.append(input_ids)
        t_segment_masks.append(segment_masks)
        t_input_masks.append(input_masks)

        input_ids, segment_masks, input_masks = get_input(
            question, tokenizer, max_length_map["question_body"]
        )
        q_input_ids.append(input_ids)
        q_segment_masks.append(segment_masks)
        q_input_masks.append(input_masks)

        input_ids, segment_masks, input_masks = get_input(
            answer, tokenizer, max_length_map["answer"]
        )
        a_input_ids.append(input_ids)
        a_segment_masks.append(segment_masks)
        a_input_masks.append(input_masks)

    title = [
        [input_id, segment_mask, input_mask]
        for input_id, segment_mask, input_mask in zip(
            t_input_ids, t_segment_masks, t_input_masks
        )
    ]
    question = [
        [input_id, segment_mask, input_mask]
        for input_id, segment_mask, input_mask in zip(
            q_input_ids, q_segment_masks, q_input_masks
        )
    ]
    answer = [
        [input_id, segment_mask, input_mask]
        for input_id, segment_mask, input_mask in zip(
            a_input_ids, a_segment_masks, a_input_masks
        )
    ]

    return title, question, answer




## === cell 12
title_test, question_test, answer_test = computer_input_array(test)



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1413646897.py in <cell line: 0>()
      1 # Note: this can take time due to 512-token sequences; kept as-is to preserve core logic.
----> 2 title_test, question_test, answer_test = computer_input_array(test)
      3 

/tmp/ipykernel_55/2445997620.py in computer_input_array(df)
     11 
     12         input_ids, segment_masks, input_masks = get_input(
---> 13             title, tokenizer, max_length_map["question_title"]
     14         )
     15         t_input_ids.append(input_ids)

NameError: name 'tokenizer' is not defined

## === cell 13
labels = train[output_columns].values.tolist()



## === cell 14
test_dict = {"title": title_test, "question": question_test, "answer": answer_test}



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2543637430.py in <cell line: 0>()
      1 # Fix: the original pipeline tried to save/load .t7 files but failed before creation.
      2 # For inference we only need test arrays; keep labels computed (unused) to preserve cell order/semantics.
----> 3 test_dict = {"title": title_test, "question": question_test, "answer": answer_test}
      4 

NameError: name 'title_test' is not defined

## === cell 15
ls = os.listdir(".")
ls



## === cell 16
import os

if not os.path.exists("./data"):
    os.mkdir("./data")



## === cell 17
if not os.path.exists("./model"):
    os.mkdir("./model")



## === cell 18
import torch

torch.save(test_dict, "./data/test_data.t7")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/739038170.py in <cell line: 0>()
      2 import torch
      3 
----> 4 torch.save(test_dict, "./data/test_data.t7")
      5 

NameError: name 'test_dict' is not defined

## === cell 19
pass



## === cell 20
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr




## === cell 21
class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained(
            "bert-base-uncased", local_files_only=True
        )
        self.dropout = nn.Dropout(0.2)
        self.pool_t = nn.AvgPool2d((32, 1))
        self.pool = nn.AvgPool2d((512, 1))
        self.output = nn.Linear(768 * 3, 30)
        self.output_t = nn.Linear(768, 21)
        self.output_q = nn.Linear(768, 21)
        self.output_a = nn.Linear(768, 9)

    def forward(
        self,
        t_inputs,
        t_input_masks,
        t_segment_masks,
        q_inputs,
        q_input_masks,
        q_segment_masks,
        a_inputs,
        a_input_masks,
        a_segment_masks,
    ):
        t_outputs = self.bert(
            t_inputs, attention_mask=t_input_masks, token_type_ids=t_segment_masks
        )
        t_x = t_outputs[0]
        t_x = self.dropout(t_x)
        t_x = t_x.unsqueeze(1)
        t_x = self.pool_t(t_x)
        t_x = t_x.squeeze(1).squeeze(1)

        q_outputs = self.bert(
            q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks
        )
        q_x = q_outputs[0]
        q_x = self.dropout(q_x)
        q_x = q_x.unsqueeze(1)
        q_x = self.pool(q_x)
        q_x = q_x.squeeze(1).squeeze(1)

        a_outputs = self.bert(
            a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks
        )
        a_x = a_outputs[0]
        a_x = self.dropout(a_x)
        a_x = a_x.unsqueeze(1)
        a_x = self.pool(a_x)
        a_x = a_x.squeeze(1).squeeze(1)

        t_q_a = torch.cat((t_x, q_x, a_x), -1)
        output = self.output(t_q_a)

        output_t = self.output_t(t_x)
        output_q = self.output_q(q_x)
        output_q = 0.5 * output_t + 0.5 * output_q

        output_a = self.output_a(a_x)
        output_qa = torch.cat((output_q, output_a), -1)

        output = 0.5 * output + 0.5 * output_qa

        x = torch.sigmoid(output)
        return x




## === cell 22
pass



## === cell 23
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader, Dataset, TensorDataset
from sklearn.model_selection import train_test_split, GroupKFold



## === cell 24
test_data = torch.load("./data/test_data.t7", map_location="cpu")



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/3847259442.py in <cell line: 0>()
      1 # Fix: ensure test_data exists (saved in cell 19).
----> 2 test_data = torch.load("./data/test_data.t7", map_location="cpu")
      3 

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

FileNotFoundError: [Errno 2] No such file or directory: './data/test_data.t7'

## === cell 25
test_set = TensorDataset(
    torch.LongTensor(np.array(test_data["title"])),
    torch.LongTensor(np.array(test_data["question"])),
    torch.LongTensor(np.array(test_data["answer"])),
)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1810249517.py in <cell line: 0>()
      1 test_set = TensorDataset(
----> 2     torch.LongTensor(np.array(test_data["title"])),
      3     torch.LongTensor(np.array(test_data["question"])),
      4     torch.LongTensor(np.array(test_data["answer"])),
      5 )

NameError: name 'test_data' is not defined

## === cell 26
test_loader = DataLoader(test_set, batch_size=1, shuffle=False)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3498341488.py in <cell line: 0>()
----> 1 test_loader = DataLoader(test_set, batch_size=1, shuffle=False)
      2 

NameError: name 'test_set' is not defined

## === cell 27
criterion = nn.BCELoss()




## === cell 28
def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)




## === cell 29
"""
gkf = GroupKFold(n_splits=10).split(X=train.question_body, groups=train.question_body)
final_predicts = []
for fold, (train_idx, valid_idx) in enumerate(gkf):
    if fold in [3, 4, 5]:
        model = Model()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        train_set = TensorDataset(torch.LongTensor(np.array(data['title'])[train_idx]),\
                                  torch.LongTensor(np.array(data['question'])[train_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[train_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[train_idx]))
        dev_set = TensorDataset(torch.LongTensor(np.array(data['title'])[valid_idx]),\
                                  torch.LongTensor(np.array(data['question'])[valid_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[valid_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[valid_idx]))
        train_loader = DataLoader(
            train_set,
            batch_size=6,
            shuffle=True, drop_last=True)
        dev_loader = DataLoader(
            dev_set,
            batch_size=min(len(dev_set), 1),
            shuffle=False)
        for epoch_idx in range(3):
            for batch_idx, (input_title, input_question, input_answer, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()

                input_title, input_question, input_answer = input_title.cuda(), input_question.cuda(), input_answer.cuda()
                labels = labels.cuda()

                input_title, input_question, input_answer = Variable(input_title, requires_grad=False), \
                Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                scores = model(input_title[:,0],
                               input_title[:,2],
                               input_title[:,1],
                               input_question[:,0],
                               input_question[:,2],
                               input_question[:,1],
                               input_answer[:,0],
                               input_answer[:,2],
                               input_answer[:,1])

                labels = Variable(labels, requires_grad=False)
                labels = labels.transpose(0, 1)
                scores = scores.transpose(0, 1)
                losses = [criterion(score, label) for score, label in zip(scores, labels)]
                loss = sum(losses)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                optimizer.step()
            print("train epoch: {} loss: {}".format(epoch_idx, loss.item()/30))
            torch.save(model.state_dict(), './model/model_{}_{}.t7'.format(fold, epoch_idx))

        torch.cuda.empty_cache()
        model.eval()
        pre_list = []
        tru_list = []
        with torch.no_grad():
            for input_title, input_question, input_answer, labels in dev_loader:
                input_title, input_question, input_answer = input_title.cuda(), input_question.cuda(), input_answer.cuda()

                input_title, input_question, input_answer = Variable(input_title, requires_grad=False), \
                Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)

                scores = model(input_title[:,0],
                               input_title[:,2],
                               input_title[:,1],
                               input_question[:,0],
                               input_question[:,2],
                               input_question[:,1],
                               input_answer[:,0],
                               input_answer[:,2],
                               input_answer[:,1])
                pre_list.append(scores)
                tru_list.append(labels)
        dev_predicts = [pre.squeeze(0).cpu().numpy().tolist() for pre in pre_list]
        truthes = [t.squeeze(0).numpy().tolist() for t in tru_list]
        dev_rho = compute_spearmanr_ignore_nan(dev_predicts, truthes)
        print("dev score: ", dev_rho)
"""



## === cell 30
pass



## === cell 31
models = []
for fold in range(4):
    model_path = f"../input/google-qa-labeling-ensemble-fold-5/model_{fold}_2.t7"
    if os.path.exists(model_path):
        print(f"model available for prediction at {model_path}")
        model = Model()
        state = torch.load(model_path, map_location="cpu")
        model.load_state_dict(state)
        models.append(model)

if len(models) == 0:
    print(
        "No external fold models found; using a single base (untrained) model for a valid submission."
    )
    models = [Model()]



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/788005261.py in <cell line: 0>()
     14         "No external fold models found; using a single base (untrained) model for a valid submission."
     15     )
---> 16     models = [Model()]
     17 

/tmp/ipykernel_55/1982810225.py in __init__(self)
      3         super().__init__()
      4         # Fix: use standard model id offline, instead of a missing local path.
----> 5         self.bert = BertModel.from_pretrained(
      6             "bert-base-uncased", local_files_only=True
      7         )

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in _wrapper(*args, **kwargs)
    309         old_dtype = torch.get_default_dtype()
    310         try:
--> 311             return func(*args, **kwargs)
    312         finally:
    313             torch.set_default_dtype(old_dtype)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
   4581         if not isinstance(config, PretrainedConfig):
   4582             config_path = config if config is not None else pretrained_model_name_or_path
-> 4583             config, model_kwargs = cls.config_class.from_pretrained(
   4584                 config_path,
   4585                 cache_dir=cache_dir,

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
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 32
len(models)



## === cell 33
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

final_predicts = []
for model in models:
    model = model.to(device)
    model.eval()
    test_predicts = []
    with torch.no_grad():
        for input_title, input_question, input_answer in test_loader:
            input_title = input_title.to(device)
            input_question = input_question.to(device)
            input_answer = input_answer.to(device)

            input_title = Variable(input_title, requires_grad=False)
            input_question = Variable(input_question, requires_grad=False)
            input_answer = Variable(input_answer, requires_grad=False)

            scores = model(
                input_title[:, 0],
                input_title[:, 2],
                input_title[:, 1],
                input_question[:, 0],
                input_question[:, 2],
                input_question[:, 1],
                input_answer[:, 0],
                input_answer[:, 2],
                input_answer[:, 1],
            )  # [1,30]
            test_predicts.append(scores.squeeze(0).detach().cpu().numpy())
    final_predicts.append(np.stack(test_predicts, axis=0))



## === cell 34
pres = np.mean(np.stack(final_predicts, axis=0), axis=0)  # [n_test, 30]
pres = np.clip(pres, 0.0, 1.0)
test_output = pres.tolist()



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3387428789.py in <cell line: 0>()
      1 # Fix: average across models -> [n_test,30], then convert to list-of-lists.
----> 2 pres = np.mean(np.stack(final_predicts, axis=0), axis=0)  # [n_test, 30]
      3 pres = np.clip(pres, 0.0, 1.0)
      4 test_output = pres.tolist()
      5 

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 35
output_cols = [
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 36
output = pd.DataFrame(test_output, columns=output_cols)
output.insert(0, "qa_id", test["qa_id"].values)

output = output[sample_submission.columns.tolist()]



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3565677764.py in <cell line: 0>()
      1 # Fix: build submission aligned to sample_submission (qa_id order/length).
----> 2 output = pd.DataFrame(test_output, columns=output_cols)
      3 output.insert(0, "qa_id", test["qa_id"].values)
      4 
      5 # Ensure exact column order as sample_submission

NameError: name 'test_output' is not defined

## === cell 37
order = [
    "qa_id",
    "question_asker_intent_understanding",
    "question_body_critical",
    "question_conversational",
    "question_expect_short_answer",
    "question_fact_seeking",
    "question_has_commonly_accepted_answer",
    "question_interestingness_others",
    "question_interestingness_self",
    "question_multi_intent",
    "question_not_really_a_question",
    "question_opinion_seeking",
    "question_type_choice",
    "question_type_compare",
    "question_type_consequence",
    "question_type_definition",
    "question_type_entity",
    "question_type_instructions",
    "question_type_procedure",
    "question_type_reason_explanation",
    "question_type_spelling",
    "question_well_written",
    "answer_helpful",
    "answer_level_of_information",
    "answer_plausible",
    "answer_relevance",
    "answer_satisfaction",
    "answer_type_instructions",
    "answer_type_procedure",
    "answer_type_reason_explanation",
    "answer_well_written",
]



## === cell 38
output = output[order]



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1610986860.py in <cell line: 0>()
----> 1 output = output[order]
      2 

NameError: name 'output' is not defined

## === cell 39
output.head()



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2010806425.py in <cell line: 0>()
----> 1 output.head()
      2 

NameError: name 'output' is not defined

## === cell 40
output.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", output.shape)
print("Columns OK:", output.columns.tolist() == sample_submission.columns.tolist())

## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2943057933.py in <cell line: 0>()
----> 1 output.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", output.shape)
      3 print("Columns OK:", output.columns.tolist() == sample_submission.columns.tolist())

NameError: name 'output' is not defined
