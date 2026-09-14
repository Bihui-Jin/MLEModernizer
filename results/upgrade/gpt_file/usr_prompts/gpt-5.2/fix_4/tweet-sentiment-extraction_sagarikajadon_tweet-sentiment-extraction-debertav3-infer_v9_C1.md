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
    TRAIN = False  # inference-only for this notebook
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
    def __init__(self, df: pd.DataFrame):
        self.df = df.reset_index(drop=True)

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
        )

        for k, v in inputs.items():
            inputs[k] = torch.tensor(v, dtype=torch.long)

        tok_text_tokens = CFG.TOKENIZER.convert_ids_to_tokens(inputs["input_ids"])

        sentiment = [1, 0, 0]
        if self.df.sentiment.iloc[item] == "positive":
            sentiment = [0, 0, 1]
        if self.df.sentiment.iloc[item] == "negative":
            sentiment = [0, 1, 0]

        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "text_tokens": " ".join(tok_text_tokens),
            "sentiment": torch.tensor(sentiment, dtype=torch.long),
            "orig_text": self.df.text.iloc[item],
            "orig_sentiment": self.df.sentiment.iloc[item],
        }




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
    fin_text_tokens = []
    fin_orig_text = []
    fin_orig_selected = []
    fin_orig_sentiment = []

    with torch.no_grad():
        for data in tqdm(dataloader, total=len(dataloader)):
            input_ids = data["input_ids"].to(device)
            mask = data["attention_mask"].to(device)

            text_tokens = data["text_tokens"]
            orig_text = data["orig_text"]
            orig_sentiment = data["orig_sentiment"]

            start_logits, end_logits = model(input_ids, mask)
            fin_output_start.append(start_logits.cpu())
            fin_output_end.append(end_logits.cpu())
            fin_mask.append(mask.cpu())

            fin_text_tokens.extend(text_tokens)
            fin_orig_text.extend(orig_text)
            fin_orig_sentiment.extend(orig_sentiment)

    fin_output_start = torch.vstack(fin_output_start)
    fin_output_end = torch.vstack(fin_output_end)
    fin_mask = torch.vstack(fin_mask)

    return (
        fin_output_start,
        fin_output_end,
        fin_mask,
        fin_text_tokens,
        fin_orig_text,
        fin_orig_selected,
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
    collate_fn=collate_fn,  # use dynamic padding collator (stable keys/types)
)




## === cell 8
model = QAModel(config_path=None, pretrained=True).to(device)

(
    fin_output_start,
    fin_output_end,
    fin_mask,
    fin_text_tokens,
    fin_orig_text,
    fin_orig_selected,
    fin_orig_sentiment,
) = test_fn(test_loader, model)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_54/1061034117.py in <cell line: 0>()
      9     fin_orig_selected,
     10     fin_orig_sentiment,
---> 11 ) = test_fn(test_loader, model)
     12 
     13 

/tmp/ipykernel_54/3511490009.py in test_fn(dataloader, model)
     11 
     12     with torch.no_grad():
---> 13         for data in tqdm(dataloader, total=len(dataloader)):
     14             input_ids = data["input_ids"].to(device)
     15             # Fix: dataset provides attention_mask; keep variable name "mask" for model call.

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

ValueError: Caught ValueError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 767, in convert_to_tensors
    tensor = as_tensor(value)
             ^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 729, in as_tensor
    return torch.tensor(value)
           ^^^^^^^^^^^^^^^^^^^
ValueError: too many dimensions 'str'

The above exception was the direct cause of the following exception:

Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 55, in fetch
    return self.collate_fn(data)
           ^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 272, in __call__
    batch = pad_without_fast_tokenizer_warning(
            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/data/data_collator.py", line 67, in pad_without_fast_tokenizer_warning
    padded = tokenizer.pad(*pad_args, **pad_kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3374, in pad
    return BatchEncoding(batch_outputs, tensor_type=return_tensors)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 240, in __init__
    self.convert_to_tensors(tensor_type=tensor_type, prepend_batch_axis=prepend_batch_axis)
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 783, in convert_to_tensors
    raise ValueError(
ValueError: Unable to create tensor, you should probably activate truncation and/or padding with 'padding=True' 'truncation=True' to have batched tensors with the same length. Perhaps your features (`text_tokens` in this case) have excessive nesting (inputs type `list` where type `int` is expected).


## === cell 9
s = torch.nn.Softmax(dim=-1)
fin_output_start = s(fin_output_start)
fin_output_end = s(fin_output_end)

fin_mask = fin_mask.float()

(
    fin_output_start.shape,
    fin_output_end.shape,
    fin_mask.shape,
    len(fin_text_tokens),
    len(fin_orig_text),
)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3725278300.py in <cell line: 0>()
      1 s = torch.nn.Softmax(dim=-1)
----> 2 fin_output_start = s(fin_output_start)
      3 fin_output_end = s(fin_output_end)
      4 
      5 fin_mask = fin_mask.float()

NameError: name 'fin_output_start' is not defined

## === cell 10
threshold = 0
final_outputs = []

for j in range(len(fin_text_tokens)):
    text_token = fin_text_tokens[j]
    mask = fin_mask[j]
    sentiment = fin_orig_sentiment[j]  # kept for compatibility; not used in decoding

    mask_start = fin_output_start[j] * mask
    mask_end = fin_output_end[j] * mask

    token_keep = [0] * int(mask.numel())

    idx_start = int(torch.argmax(mask_start).item())
    idx_end = int(torch.argmax(mask_end).item())
    if idx_end < idx_start:
        idx_end = idx_start

    for mj in range(idx_start, idx_end + 1):
        token_keep[mj] = 1

    output_tokens = [
        x
        for i, x in enumerate(text_token.split())
        if i < len(token_keep) and token_keep[i] == 1
    ]
    output_tokens = [x for x in output_tokens if x not in ("[CLS]", "[SEP]")]

    final_output = ""
    for ot in output_tokens:
        if ot.startswith("▁"):
            final_output = final_output + " " + ot[1:]
        elif len(ot) == 1 and ot in string.punctuation:
            final_output = final_output + ot
        else:
            final_output = final_output + " " + ot

    final_output = final_output.strip()
    if final_output == "":
        final_output = " ".join(str(test_df.text.iloc[j]).split())

    final_outputs.append(final_output)

len(final_outputs), final_outputs[0]




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/3040316002.py in <cell line: 0>()
      2 final_outputs = []
      3 
----> 4 for j in range(len(fin_text_tokens)):
      5     text_token = fin_text_tokens[j]
      6     mask = fin_mask[j]

NameError: name 'fin_text_tokens' is not defined

## === cell 11
assert len(final_outputs) == len(
    test_df
), f"Pred length {len(final_outputs)} != test length {len(test_df)}"

test_ids = test_df["textID"].values
sub = pd.DataFrame({"textID": test_ids, "selected_text": final_outputs})
sub.head()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_54/1881837104.py in <cell line: 0>()
----> 1 assert len(final_outputs) == len(
      2     test_df
      3 ), f"Pred length {len(final_outputs)} != test length {len(test_df)}"
      4 
      5 test_ids = test_df["textID"].values

AssertionError: Pred length 0 != test length 2749

## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.isna().sum())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_54/2842601452.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.isna().sum())

NameError: name 'sub' is not defined
