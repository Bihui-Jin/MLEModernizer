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
from sklearn.model_selection import (
    train_test_split,
    KFold,
    GroupKFold,
    StratifiedKFold,
    StratifiedGroupKFold,
)

from tqdm import tqdm
from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    AdamW,
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup,
    DataCollatorWithPadding,
)

os.environ["TOKENIZERS_PARALLELISM"] = "false"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class CFG:
    DEBUG = False
    TRAIN = True
    N_FOLDS = 5
    TRAIN_FOLDS = [i for i in range(N_FOLDS)]
    SEED = 42
    TEST_BATCHSIZE = 100
    MAX_LENGTH = 128
    MODEL_NAME = "microsoft/deberta-v3-base"
    FC_DROPOUT = [0.1, 0.2, 0.3, 0.4, 0.5]




## === cell 2
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.SEED)



## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()



## === cell 4
CFG.TOKENIZER = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)



## === cell 5
collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset(Dataset):
    def __init__(self, df):
        self.df = df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        text = " ".join(str(self.df.text.iloc[item]).split())
        inputs = CFG.TOKENIZER(
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(0) for k, v in inputs.items()}

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(
            inputs["input_ids"].tolist()
        )

        sentiment = [1, 0, 0]  # default neutral
        if self.df.sentiment.iloc[item] == "positive":
            sentiment = [0, 0, 1]
        elif self.df.sentiment.iloc[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": inputs["input_ids"],
            "mask": inputs["attention_mask"],
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1974823703.py in <cell line: 0>()
      1 # Collate function using the correctly loaded tokenizer
----> 2 collate_fn = DataCollatorWithPadding(
      3     CFG.TOKENIZER, padding="longest", return_tensors="pt"
      4 )
      5 

NameError: name 'DataCollatorWithPadding' is not defined

## === cell 6
class QAModel(nn.Module):
    def __init__(self, config_path=None, pretrained=False):
        super().__init__()
        if config_path is None:
            self.config = AutoConfig.from_pretrained(
                CFG.MODEL_NAME, output_hidden_states=True
            )
        else:
            self.config = torch.load(config_path)

        if pretrained:
            self.backbone = AutoModel.from_pretrained(
                CFG.MODEL_NAME, config=self.config
            )
        else:
            self.backbone = AutoModel.from_config(self.config)

        self.fc_dropout = nn.ModuleList([nn.Dropout(val) for val in CFG.FC_DROPOUT])

        self.attention_head = nn.Sequential(
            nn.Linear(self.config.hidden_size, 512),
            nn.GELU(),
            nn.Linear(512, 1),
            nn.Softmax(dim=1),
        )

        self.fc = nn.Linear(self.config.hidden_size, 2)

    def initialize_parameters(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight.data, mean=0, std=1)

    def forward(self, input_ids, mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = self.fc(embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 7
def test_fn(dataloader, model):
    model.eval()
    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_sentiment = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["mask"].to(device)
            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]

            start_logits, end_logits = model(input_ids, mask)
            fin_output_start.append(start_logits.cpu().numpy())
            fin_output_end.append(end_logits.cpu().numpy())
            fin_mask.append(mask.cpu().numpy())

            fin_text_tokens.extend(text_tokens)
            fin_orig_text.extend(orig_text)
            fin_orig_sentiment.extend(orig_sentiment)

    fin_output_start = np.vstack(fin_output_start)
    fin_output_end = np.vstack(fin_output_end)
    fin_mask = np.vstack(fin_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_sentiment,
    )




## === cell 8
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=True,
    collate_fn=collate_fn,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2383159302.py in <cell line: 0>()
----> 1 test_dataset = QADataset(test_df)
      2 test_loader = DataLoader(
      3     test_dataset,
      4     batch_size=CFG.TEST_BATCHSIZE,
      5     shuffle=False,

NameError: name 'QADataset' is not defined

## === cell 9
for idx, fold_id in enumerate(CFG.TRAIN_FOLDS):
    model = QAModel(pretrained=True).to(device)
    if idx == 0:
        (
            fin_output_start,
            fin_output_end,
            fin_mask,
            fin_text_tokens,
            fin_orig_text,
            fin_orig_sentiment,
        ) = test_fn(test_loader, model)
    else:
        a, b, c, _, _, _ = test_fn(test_loader, model)
        fin_output_start = np.add(fin_output_start, a)
        fin_output_end = np.add(fin_output_end, b)
        fin_mask = np.add(fin_mask, c)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/488449852.py in <cell line: 0>()
      1 # Perform inference over the defined folds using the same pretrained model each time.
      2 for idx, fold_id in enumerate(CFG.TRAIN_FOLDS):
----> 3     model = QAModel(pretrained=True).to(device)
      4     # No external checkpoint is available; use the randomly initialized (pretrained) model.
      5     if idx == 0:

NameError: name 'device' is not defined

## === cell 10
num_folds = len(CFG.TRAIN_FOLDS)
fin_output_start = fin_output_start / num_folds
fin_output_end = fin_output_end / num_folds
fin_mask = fin_mask / num_folds  # mask is 0/1, averaging keeps it as float



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/993333933.py in <cell line: 0>()
      1 # Average the predictions from all folds
      2 num_folds = len(CFG.TRAIN_FOLDS)
----> 3 fin_output_start = fin_output_start / num_folds
      4 fin_output_end = fin_output_end / num_folds
      5 fin_mask = fin_mask / num_folds  # mask is 0/1, averaging keeps it as float

NameError: name 'fin_output_start' is not defined

## === cell 11
threshold = 0.3
final_outputs = []
for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask_row = fin_mask[j]

    mask_start = (fin_output_start[j] * mask_row) >= threshold
    mask_end = (fin_output_end[j] * mask_row) >= threshold

    idx_start_candidates = np.nonzero(mask_start)[0]
    idx_end_candidates = np.nonzero(mask_end)[0]

    if len(idx_start_candidates) > 0:
        idx_start = idx_start_candidates[0]
        idx_end = idx_end_candidates[0] if len(idx_end_candidates) > 0 else idx_start
    else:
        idx_start = 0
        idx_end = 0

    token_mask = [0] * len(mask_row)
    for mj in range(idx_start, idx_end + 1):
        token_mask[mj] = 1

    tokens = [tok for i, tok in enumerate(text_token.split()) if token_mask[i] == 1]
    tokens = [tok for tok in tokens if tok not in ("[CLS]", "[SEP]")]

    if not tokens:
        final_outputs.append("")  # fallback empty string
        continue

    final_output = tokens[0][1:] if tokens[0].startswith("▁") else tokens[0]
    for ot in tokens[1:]:
        if ot.startswith("▁"):
            final_output += " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output += ot
        else:
            final_output += ot
    final_outputs.append(final_output)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2741432173.py in <cell line: 0>()
      1 threshold = 0.3
      2 final_outputs = []
----> 3 for j in range(len(fin_text_tokens)):
      4     text_token = fin_text_tokens[j]
      5     mask_row = fin_mask[j]

NameError: name 'fin_text_tokens' is not defined

## === cell 12
test_ids = test_df["textID"].values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/538339077.py in <cell line: 0>()
      1 test_ids = test_df["textID"].values
----> 2 sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
      3 sub.head()
      4 

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
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length

## === cell 13
sub



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/270188417.py in <cell line: 0>()
----> 1 sub
      2 

NameError: name 'sub' is not defined

## === cell 14
sub.to_csv("submission.csv", index=False)

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/352017882.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)

NameError: name 'sub' is not defined
