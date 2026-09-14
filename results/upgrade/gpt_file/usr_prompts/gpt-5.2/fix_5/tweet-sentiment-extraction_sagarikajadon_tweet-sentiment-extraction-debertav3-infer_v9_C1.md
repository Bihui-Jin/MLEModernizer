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

0.5516448616981506

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14602) has done: 'I fix the environment/runtime failures by (1) removing the protobuf-triggering import stack that breaks at startup, (2) ensuring `DataCollatorWithPadding` is imported, and (3) replacing the invalid private Kaggle input paths for tokenizer/config/weights with public Hugging Face model loading so the notebook can run without missing files. To preserve the core inference logic (start/end span logits over tokens), I keep the same `QAModel` architecture and decoding approach, but load a compatible backbone and run a single deterministic forward pass. Finally, I guarantee we always generate exactly 2749 predictions and write a valid `submission.csv` with the required columns and quoting handled by pandas.'
- What this solution (achieved 0.14602) has done: 'I fix the startup crash caused by an incompatible protobuf/transformers import stack by forcing the pure-Python protobuf implementation before importing `transformers`. Then I correct a key decoding logic bug: applying `Softmax(dim=1)` to a 1D logit vector is wrong (it normalizes across samples when batched and errors/behaves badly per-row); instead I use `dim=-1` so probabilities are computed across token positions. These two changes keep your model/inference approach the same (start/end span logits over tokens) but make it run end-to-end and materially improve span selection toward the target Jaccard score. Finally, I keep the submission writing as-is to ensure a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver

        major = int(pb_ver.split(".")[0])
        if major >= 5:
            raise RuntimeError(f"Incompatible protobuf version: {pb_ver}")
    except Exception:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf>=4.21.0,<5"]
        )
        import importlib

        importlib.invalidate_caches()


_ensure_protobuf_compat()

import gc
import random
import math
import re
import string

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from transformers import (
    AutoTokenizer,
    AutoModel,
    AutoConfig,
    DataCollatorWithPadding,
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




## === cell 1
class CFG:
    DEBUG = False
    TRAIN = False  # inference-only
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
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed=CFG.SEED)




## === cell 3
test_df = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
test_df.head()




## === cell 4
tokenizer = AutoTokenizer.from_pretrained(CFG.MODEL_NAME, use_fast=True)
CFG.TOKENIZER = tokenizer

collate_fn = DataCollatorWithPadding(
    CFG.TOKENIZER, padding="longest", return_tensors="pt"
)


class QADataset:
    """
    Bug fix: DataCollatorWithPadding can only pad/tokenize numeric fields.
    Previously we returned 'text_tokens' as a string in each sample, causing a crash
    when the collator tried to convert/pad it as a tensor.

    Minimal change: return only tokenizer outputs + simple meta fields that we will
    collate manually (orig_text, orig_sentiment). Keep offsets for span -> substring.
    """

    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, item):
        orig_text = str(self.df.text.iloc[item])
        text = " ".join(orig_text.split())
        sentiment_str = self.df.sentiment.iloc[item]

        inputs = CFG.TOKENIZER(
            text,
            add_special_tokens=True,
            max_length=CFG.MAX_LENGTH,
            padding=False,  # let collator pad dynamically
            truncation=True,
            return_offsets_mapping=True,
        )

        sentiment = [1, 0, 0]
        if sentiment_str == "positive":
            sentiment = [0, 0, 1]
        if sentiment_str == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "offset_mapping": inputs["offset_mapping"],
            "orig_text": orig_text,
            "orig_sentiment": sentiment_str,
            "sentiment": sentiment,  # unused, but kept (list of ints)
        }


def qa_collate(batch):
    """
    Bug fix: use DataCollatorWithPadding only on numeric token fields; collate
    offset_mapping and strings separately.
    """
    token_features = [
        {"input_ids": x["input_ids"], "attention_mask": x["attention_mask"]}
        for x in batch
    ]
    padded = collate_fn(token_features)

    padded["offset_mapping"] = [x["offset_mapping"] for x in batch]
    padded["orig_text"] = [x["orig_text"] for x in batch]
    padded["orig_sentiment"] = [x["orig_sentiment"] for x in batch]
    padded["sentiment"] = torch.tensor(
        [x["sentiment"] for x in batch], dtype=torch.long
    )
    return padded




## === cell 5
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

    def forward(self, input_ids, mask):
        embeddings = self.backbone(
            input_ids=input_ids, attention_mask=mask
        ).last_hidden_state
        logits = self.fc(embeddings)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 6
def test_fn(dataloader, model):
    model.eval()

    fin_output_start = []
    fin_output_end = []
    fin_mask = []
    fin_offsets = []
    fin_orig_text = []
    fin_orig_sentiment = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)

            start_logits, end_logits = model(input_ids, mask)

            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())

            fin_offsets.extend(data["offset_mapping"])
            fin_orig_text.extend(data["orig_text"])
            fin_orig_sentiment.extend(data["orig_sentiment"])

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_offsets,
        fin_orig_text,
        fin_orig_sentiment,
    )




## === cell 7
test_dataset = QADataset(test_df)
test_loader = DataLoader(
    test_dataset,
    batch_size=CFG.TEST_BATCHSIZE,
    shuffle=False,
    num_workers=2,
    pin_memory=torch.cuda.is_available(),
    collate_fn=qa_collate,  # important: avoids collator trying to tensorize strings
)




## === cell 8
model = QAModel(config_path=None, pretrained=True).to(device)

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_offsets,
    fin_orig_text,
    fin_orig_sentiment,
) = test_fn(test_loader, model)

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    len(fin_offsets),
    len(fin_orig_text),
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/3218671665.py in <cell line: 0>()
      8     fin_orig_text,
      9     fin_orig_sentiment,
---> 10 ) = test_fn(test_loader, model)
     11 
     12 (

/tmp/ipykernel_55/3339493617.py in test_fn(dataloader, model)
     24             fin_orig_sentiment.extend(data["orig_sentiment"])
     25 
---> 26     fin_output_start = torch.vstack(fin_output_start)
     27     fin_output_end = torch.vstack(fin_output_end)
     28     fin_mask = torch.vstack(fin_mask)

RuntimeError: Sizes of tensors must match except in dimension 0. Expected size 43 but got size 45 for tensor number 1 in the list.

## === cell 9
s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)
fin_mask = fin_mask.float()

final_outputs = []

for j in range(len(fin_orig_text)):
    orig_text = str(fin_orig_text[j])
    text = " ".join(orig_text.split())
    offsets = fin_offsets[j]  # list of (start_char, end_char) for the normalized `text`
    mask = fin_mask[j]

    mask_start = fin_output_start[j] * mask
    mask_end = fin_output_end[j] * mask

    idx_start = int(torch.argmax(mask_start).item())
    idx_end = int(torch.argmax(mask_end).item())
    if idx_end < idx_start:
        idx_end = idx_start

    n_tok = min(len(offsets), int(mask.numel()))
    idx_start = min(idx_start, n_tok - 1)
    idx_end = min(idx_end, n_tok - 1)

    def _find_next_valid_left(i):
        while i > 0 and (offsets[i] is None or offsets[i][1] <= offsets[i][0]):
            i -= 1
        return i

    def _find_next_valid_right(i):
        while i < n_tok - 1 and (offsets[i] is None or offsets[i][1] <= offsets[i][0]):
            i += 1
        return i

    idx_start = _find_next_valid_right(idx_start)
    idx_end = _find_next_valid_left(idx_end)
    if idx_end < idx_start:
        idx_end = idx_start

    start_char = offsets[idx_start][0]
    end_char = offsets[idx_end][1]

    if end_char <= start_char:
        pred = text
    else:
        pred = text[start_char:end_char].strip()
        if pred == "":
            pred = text

    final_outputs.append(pred)

len(final_outputs), final_outputs[0]




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3712114245.py in <cell line: 0>()
      1 # Keep the same probability normalization semantics (token-wise softmax)
      2 s = torch.nn.Softmax(dim=-1)
----> 3 fin_output_start = s(fin_output_start)
      4 fin_output_end = s(fin_output_end)
      5 fin_mask = fin_mask.float()

NameError: name 'fin_output_start' is not defined

## === cell 10
assert len(final_outputs) == len(
    test_df
), f"Pred length {len(final_outputs)} != test length {len(test_df)}"

test_ids = test_df["textID"].values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1881837104.py in <cell line: 0>()
----> 1 assert len(final_outputs) == len(
      2     test_df
      3 ), f"Pred length {len(final_outputs)} != test length {len(test_df)}"
      4 
      5 test_ids = test_df["textID"].values

NameError: name 'final_outputs' is not defined

## === cell 11
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())
print(sub.head(3).to_string(index=False))

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3116703084.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.isna().sum())
      4 print(sub.head(3).to_string(index=False))

NameError: name 'sub' is not defined
