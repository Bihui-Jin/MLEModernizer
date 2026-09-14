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

0.3697054710738545

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 2
import numpy as np
from transformers import BertTokenizer, BertModel
import re
import pandas as pd

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
sample_submission = pd.read_csv("../input/google-quest-challenge/sample_submission.csv")
test = pd.read_csv("../input/google-quest-challenge/test.csv")
train = pd.read_csv("../input/google-quest-challenge/train.csv")

## === cell 4
train.columns.values

## === cell 5
output_columns = train.columns.values[11:]
input_columns = train.columns.values[[1, 2, 5]]

## === cell 6
question_output_columns = [col for col in output_columns \
                           if 'question' in col]
answer_output_colmns = [col for col in output_columns 
                        if col not in question_output_columns]

## === cell 7
len(question_output_columns)

## === cell 8
tokenizer = BertTokenizer.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer')

## --- ERROR in cell 8, traceback:
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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1911                     try:
-> 1912                         resolved_config_file = cached_file(
   1913                             pretrained_model_name_or_path,

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

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/2464830251.py in <cell line: 0>()
----> 1 tokenizer = BertTokenizer.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer')

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   1930                     except Exception:
   1931                         # For any other exception, we throw a generic error.
-> 1932                         raise OSError(
   1933                             f"Can't load tokenizer for '{pretrained_model_name_or_path}'. If you were trying to load it from "
   1934                             "'https://huggingface.co/models', make sure you don't have a local directory with the same name. "

OSError: Can't load tokenizer for '../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-tokenizer' is the correct path to a directory containing all relevant files for a BertTokenizer tokenizer.

## === cell 9
max_length_map = {'question_title': 32,
                   'question_body': 512,
                   'answer': 512
                   }

## === cell 10
def txt_re(txt):
    txt = txt.strip()
    txt = re.sub('https?.*$', '', txt)
    txt = re.sub('https?.*\s', '', txt)
    txt = re.sub('\n+', ' ', txt)
    txt = re.sub('\r+', ' ', txt)
    txt = re.sub('\t+', ' ', txt)
    txt = re.sub('&gt;', '>', txt)
    txt = re.sub('&lt;', '<', txt)
    txt = re.sub('&amp;', '&', txt)
    txt = re.sub('&quot;', '\"', txt)
    return txt

## === cell 11
def get_input(txt, pair_txt, tokenizer, max_length):
    txt = txt_re(txt)
    txt = tokenizer.encode_plus(txt, pair_txt, add_special_tokens=True,max_length=max_length, \
                          pad_to_max_length='right')
    input_ids = txt['input_ids']
    segment_masks = txt['token_type_ids']
    input_masks = txt['attention_mask']
    
    return input_ids, segment_masks, input_masks

## === cell 12
def computer_input_array(df):
    q_input_ids, q_segment_masks, q_input_masks = [], [], []
    a_input_ids, a_segment_masks, a_input_masks = [], [], []
    for _, instance in df[input_columns].iterrows():
        title, question, answer = instance.question_title, instance.question_body, instance.answer
        """
        input_ids, segment_masks, input_masks = get_input(title, tokenizer, max_length_map['question_title'])
        t_input_ids.append(input_ids)
        t_segment_masks.append(segment_masks)
        t_input_masks.append(input_masks)
        """
    
        input_ids, segment_masks, input_masks = get_input(title, question, tokenizer, max_length_map['question_body'])
        q_input_ids.append(input_ids)
        q_segment_masks.append(segment_masks)
        q_input_masks.append(input_masks)
    
        input_ids, segment_masks, input_masks = get_input(answer, None, tokenizer, max_length_map['answer'])
        a_input_ids.append(input_ids)
        a_segment_masks.append(segment_masks)
        a_input_masks.append(input_masks)
    question = [[input_id, segment_mask, input_mask] for input_id, segment_mask, input_mask in \
              zip(q_input_ids, q_segment_masks, q_input_masks)]
    answer = [[input_id, segment_mask, input_mask] for input_id, segment_mask, input_mask in \
              zip(a_input_ids, a_segment_masks, a_input_masks)]
    
    return question, answer

## === cell 13
question_train, answer_train = computer_input_array(train)
question_test, answer_test = computer_input_array(test)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1610592000.py in <cell line: 0>()
      1 # title_train,question_train, answer_train = computer_input_array(train)
----> 2 question_train, answer_train = computer_input_array(train)
      3 # title_test, question_test, answer_test = computer_input_array(test)
      4 question_test, answer_test = computer_input_array(test)

/tmp/ipykernel_11/147532249.py in computer_input_array(df)
     12         """
     13 
---> 14         input_ids, segment_masks, input_masks = get_input(title, question, tokenizer, max_length_map['question_body'])
     15         q_input_ids.append(input_ids)
     16         q_segment_masks.append(segment_masks)

NameError: name 'tokenizer' is not defined

## === cell 14
answer_test

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3375019157.py in <cell line: 0>()
----> 1 answer_test

NameError: name 'answer_test' is not defined

## === cell 15
labels = train[output_columns].values.tolist()

## === cell 16
train_dict = {'question': question_train, 'answer': answer_train, 'label': labels}
test_dict = {'question': question_test, 'answer': answer_test}

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2224014182.py in <cell line: 0>()
      1 # train_dict = {'title': title_train, 'question': question_train, 'answer': answer_train, 'label': labels}
----> 2 train_dict = {'question': question_train, 'answer': answer_train, 'label': labels}
      3 # test_dict = {'title': title_test, 'question': question_test, 'answer': answer_test}
      4 test_dict = {'question': question_test, 'answer': answer_test}

NameError: name 'question_train' is not defined

## === cell 17
ls

## === cell 18
import os
os.mkdir('./data')


## === cell 19
os.mkdir('./model')

## === cell 20
import torch
torch.save(train_dict, './data/train_data.t7')
torch.save(test_dict, './data/test_data.t7')

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1092652549.py in <cell line: 0>()
      1 import torch
----> 2 torch.save(train_dict, './data/train_data.t7')
      3 torch.save(test_dict, './data/test_data.t7')

NameError: name 'train_dict' is not defined

## === cell 22
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import BertTokenizer, BertModel
from scipy.stats import spearmanr

## === cell 23
class Model_v1(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-pytorch-model')
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AvgPool2d((512, 1))

        self.output = nn.Linear(768 * 2, 30)
        
    def forward(self, q_inputs, q_input_masks, q_segment_masks, \
                a_inputs, a_input_masks, a_segment_masks):

        q_outputs = self.bert(q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks)
        q_x = q_outputs[0]
        q_x = self.dropout(q_x)
        q_x = q_x.unsqueeze(1)
        q_x = self.pool(q_x)
        q_x = q_x.squeeze(1).squeeze(1)
        
        a_outputs = self.bert(a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks)
        a_x = a_outputs[0]
        a_x = self.dropout(a_x)
        a_x = a_x.unsqueeze(1)
        a_x = self.pool(a_x)
        a_x = a_x.squeeze(1).squeeze(1)
        
        t_q_a = torch.cat((q_x, a_x), -1)
        output = self.output(t_q_a)
        
        x = torch.sigmoid(output)
        return x

## === cell 24
class Model_v2(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = BertModel.from_pretrained('../input/huggingfacetransformermodels/model_classes/bert/bert-base-uncased-pytorch-model')
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AvgPool2d((512, 1))

        self.output_q = nn.Linear(768, 21)
        self.output_a = nn.Linear(768, 9)    
        
    def forward(self, q_inputs, q_input_masks, q_segment_masks, \
                a_inputs, a_input_masks, a_segment_masks):
        """
        t_outputs = self.bert(t_inputs, attention_mask=t_input_masks, token_type_ids=t_segment_masks)
        t_x = t_outputs[0]
        t_x = self.dropout(t_x)
        t_x = t_x.unsqueeze(1)
        t_x = self.pool_t(t_x)
        t_x = t_x.squeeze(1).squeeze(1)
        """
        q_outputs = self.bert(q_inputs, attention_mask=q_input_masks, token_type_ids=q_segment_masks)
        q_x = q_outputs[0]
        q_x = self.dropout(q_x)
        q_x = q_x.unsqueeze(1)
        q_x = self.pool(q_x)
        q_x = q_x.squeeze(1).squeeze(1)
        
        a_outputs = self.bert(a_inputs, attention_mask=a_input_masks, token_type_ids=a_segment_masks)
        a_x = a_outputs[0]
        a_x = self.dropout(a_x)
        a_x = a_x.unsqueeze(1)
        a_x = self.pool(a_x)
        a_x = a_x.squeeze(1).squeeze(1)
        
        output_q = self.output_q(q_x)
        output_a = self.output_a(a_x)
        
        output = torch.cat((output_q, output_a), -1)
        
        x = torch.sigmoid(output)
        return x

## === cell 26
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.utils.data import DataLoader,Dataset, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GroupKFold
from transformers import AdamW, get_linear_schedule_with_warmup

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/86067917.py in <cell line: 0>()
      6 from sklearn.model_selection import train_test_split
      7 from sklearn.model_selection import GroupKFold
----> 8 from transformers import AdamW, get_linear_schedule_with_warmup

ImportError: cannot import name 'AdamW' from 'transformers' (/usr/local/lib/python3.11/dist-packages/transformers/__init__.py)

## === cell 27
data = torch.load('./data/train_data.t7')
test_data = torch.load('./data/test_data.t7')

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3999650267.py in <cell line: 0>()
----> 1 data = torch.load('./data/train_data.t7')
      2 test_data = torch.load('./data/test_data.t7')

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

FileNotFoundError: [Errno 2] No such file or directory: './data/train_data.t7'

## === cell 28
test_set = TensorDataset(torch.LongTensor(np.array(test_data['question'])),
                        torch.LongTensor(np.array(test_data['answer'])))

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/193293088.py in <cell line: 0>()
----> 1 test_set = TensorDataset(torch.LongTensor(np.array(test_data['question'])),
      2                         torch.LongTensor(np.array(test_data['answer'])))

NameError: name 'test_data' is not defined

## === cell 29
test_loader = DataLoader(
        test_set,
        batch_size=1,
        shuffle=False)

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3132555045.py in <cell line: 0>()
      1 test_loader = DataLoader(
----> 2         test_set,
      3         batch_size=1,
      4         shuffle=False)

NameError: name 'test_set' is not defined

## === cell 30
criterion = nn.BCELoss()

## === cell 31
def compute_spearmanr_ignore_nan(trues, preds):
    rhos = []
    for tcol, pcol in zip(np.transpose(trues), np.transpose(preds)):
        rhos.append(spearmanr(tcol, pcol).correlation)
    return np.nanmean(rhos)

## === cell 32
"""
gkf = GroupKFold(n_splits=5).split(X=train.question_body, groups=train.question_body)
final_predicts = []
for fold, (train_idx, valid_idx) in enumerate(gkf):   
    if fold in [1]:
        model = Model()
        model.cuda()
        optimizer = torch.optim.Adam(model.parameters(), lr=2e-5)
        train_set = TensorDataset(torch.LongTensor(np.array(data['question'])[train_idx]), \
                                  torch.LongTensor(np.array(data['answer'])[train_idx]), \
                                  torch.FloatTensor(np.array(data['label'])[train_idx])) 
        dev_set = TensorDataset(torch.LongTensor(np.array(data['question'])[valid_idx]), \
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
            for batch_idx, (input_question, input_answer, labels) in enumerate(train_loader):
                model.train()
                optimizer.zero_grad()
                
                input_question, input_answer = input_question.cuda(), input_answer.cuda()
                labels = labels.cuda()
                
                input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                scores = model(input_question[:,0],
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
                # scheduler.step()
            print("train epoch: {} loss: {}".format(epoch_idx, loss.item()/30))
            torch.save(model.state_dict(), './model/model_{}_{}.t7'.format(fold, epoch_idx))
        
        torch.cuda.empty_cache()
        model.eval()
        pre_list = []
        tru_list = []
        with torch.no_grad():
            for input_question, input_answer, labels in dev_loader:
                input_question, input_answer = input_question.cuda(), input_answer.cuda()
                
                input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                
                scores = model(input_question[:,0],
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

## === cell 34
models = []
for fold in range(5):
    model_path = f'../input/google-qa-labeling-pretrained-v3/model_{fold}_2.t7'
    if os.path.exists(model_path):
        print(f'model available for prediction at {model_path}')
        model = Model_v1()
        model.load_state_dict(torch.load(model_path))
        models.append(model)
for fold in range(5):
    model_path = f'../input/google-qa-labeling-pretrained-v4/model_{fold}_2.t7'
    if os.path.exists(model_path):
        print(f'model available for prediction at {model_path}')
        model = Model_v2()
        model.load_state_dict(torch.load(model_path))
        models.append(model)

## === cell 35
len(models)

## === cell 36
final_predicts = []
for model in models:
    model = model.cuda()
    model.eval()
    test_predicts = []
    with torch.no_grad():
        for input_question, input_answer in test_loader:
            print(input_answer.shape)
            input_question, input_answer = input_question.cuda(), input_answer.cuda()
                
            input_question, input_answer = Variable(input_question, requires_grad=False), Variable(input_answer, requires_grad=False)
                
            scores = model(input_question[:,0],
                           input_question[:,2], 
                           input_question[:,1],
                           input_answer[:,0], 
                           input_answer[:,2], 
                           input_answer[:,1])
            test_predicts.append(scores.reshape(scores.shape[-1]))
    final_predicts.append(test_predicts)

## === cell 37
pres = np.average(final_predicts, axis=0)
test_output = [[p.item() for p in pre] for pre in pres]

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/334521078.py in <cell line: 0>()
      1 pres = np.average(final_predicts, axis=0)
----> 2 test_output = [[p.item() for p in pre] for pre in pres]

TypeError: 'numpy.float64' object is not iterable

## === cell 38
 output_cols = ['question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible',
       'answer_relevance', 'answer_satisfaction',
       'answer_type_instructions', 'answer_type_procedure',
       'answer_type_reason_explanation', 'answer_well_written']

## === cell 39
output_values = np.transpose(test_output).tolist()
output_dict = {k: v for k, v in zip(output_cols, output_values)}
output_dict['qa_id'] = sample_submission['qa_id'].values.tolist()
output = pd.DataFrame.from_dict(output_dict)

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2110423645.py in <cell line: 0>()
----> 1 output_values = np.transpose(test_output).tolist()
      2 output_dict = {k: v for k, v in zip(output_cols, output_values)}
      3 output_dict['qa_id'] = sample_submission['qa_id'].values.tolist()
      4 output = pd.DataFrame.from_dict(output_dict)

NameError: name 'test_output' is not defined

## === cell 40
order = ['qa_id', 'question_asker_intent_understanding',
       'question_body_critical', 'question_conversational',
       'question_expect_short_answer', 'question_fact_seeking',
       'question_has_commonly_accepted_answer',
       'question_interestingness_others', 'question_interestingness_self',
       'question_multi_intent', 'question_not_really_a_question',
       'question_opinion_seeking', 'question_type_choice',
       'question_type_compare', 'question_type_consequence',
       'question_type_definition', 'question_type_entity',
       'question_type_instructions', 'question_type_procedure',
       'question_type_reason_explanation', 'question_type_spelling',
       'question_well_written', 'answer_helpful',
       'answer_level_of_information', 'answer_plausible',
       'answer_relevance', 'answer_satisfaction',
       'answer_type_instructions', 'answer_type_procedure',
       'answer_type_reason_explanation', 'answer_well_written']

## === cell 41
output = output[order]

## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2836234006.py in <cell line: 0>()
----> 1 output = output[order]

NameError: name 'output' is not defined

## === cell 42
output.head()

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2985005345.py in <cell line: 0>()
----> 1 output.head()

NameError: name 'output' is not defined

## === cell 43
output.to_csv('submission.csv', index=False)

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1122403687.py in <cell line: 0>()
----> 1 output.to_csv('submission.csv', index=False)

NameError: name 'output' is not defined
