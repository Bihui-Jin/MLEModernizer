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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
nltk==3.9.2
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
tokenizers==0.21.2
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

0.7162611484527588

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49835) has done: 'The fix updates the tokenization to use HuggingFace’s RobertaTokenizerFast (removing the incompatible tokenizers call), loads the Roberta model directly from the pretrained hub (so missing local files no longer cause errors), and adjusts data handling to work with the new tokenizer’s output. It also removes the unsupported %time magic and ensures the prediction list is correctly built before writing the required submission.csv file.'
- What this solution (achieved 0.64124) has done: 'The fix switches to the fast RoBERTa tokenizer (which supports offset mappings), updates the import accordingly, and keeps the rest of the pipeline unchanged so training and inference run without errors and a correctly‑sized submission file is produced.'
- What this solution (achieved 0.63342) has done: 'I remove the problematic config loading (which triggers a protobuf error) and avoid the faulty lower‑casing of texts that harms the Jaccard metric. The model be created directly with `output_hidden_states=True`, and the dataset keep the original tweet casing while still adding the leading space required by RoBERTa. These minimal fixes eliminate the runtime error and improve prediction quality, moving the score toward the target.'
- What this solution (achieved 0.62412) has done: 'Implemented fixes to resolve the protobuf import error by forcing the pure‑Python implementation, extended the token length to capture full tweets, and trained the model for more epochs to improve the Jaccard score. These minimal changes keep the original architecture and training logic intact while addressing runtime failures and nudging the validation metric toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import random
import warnings

import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from transformers import RobertaTokenizerFast, RobertaModel

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int = 42):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed_everything(42)

batch_size = 32
N_FOLDS = 10
NUM_WORKERS = 2
MAX_LEN = 256  # increased to capture full tweet context
LINEAR_DROPOUT = 0.2
LR = 3e-5
EPOCHS = 4  # more training epochs for better performance

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
train_file = "/kaggle/input/tweet-sentiment-extraction/train.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

tokenizer = RobertaTokenizerFast.from_pretrained("roberta-base")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/7222132.py in <cell line: 0>()
     13 from torch.utils.data import Dataset, DataLoader
     14 from sklearn.model_selection import train_test_split
---> 15 from transformers import RobertaTokenizerFast, RobertaModel
     16 
     17 warnings.filterwarnings("ignore")

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2152         elif name in self._class_to_module.keys():
   2153             try:
-> 2154                 module = self._get_module(self._class_to_module[name])
   2155                 value = getattr(module, name)
   2156             except (ModuleNotFoundError, RuntimeError) as e:

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
-> 2184             raise e
   2185 
   2186     def __reduce__(self):

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in _get_module(self, module_name)
   2180     def _get_module(self, module_name: str):
   2181         try:
-> 2182             return importlib.import_module("." + module_name, self.__name__)
   2183         except Exception as e:
   2184             raise e

/usr/lib/python3.11/importlib/__init__.py in import_module(name, package)
    124                 break
    125             level += 1
--> 126     return _bootstrap._gcd_import(name[level:], package, level)
    127 
    128 

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/modeling_roberta.py in <module>
     39     TokenClassifierOutput,
     40 )
---> 41 from ...modeling_utils import PreTrainedModel
     42 from ...pytorch_utils import apply_chunking_to_forward, find_pruneable_heads_and_indices, prune_linear_layer
     43 from ...utils import auto_docstring, get_torch_version, logging

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in <module>
     71     verify_tp_plan,
     72 )
---> 73 from .loss.loss_utils import LOSS_MAPPING
     74 from .pytorch_utils import (  # noqa: F401
     75     Conv1D,

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_utils.py in <module>
     19 from torch.nn import BCEWithLogitsLoss, MSELoss
     20 
---> 21 from .loss_d_fine import DFineForObjectDetectionLoss
     22 from .loss_deformable_detr import DeformableDetrForObjectDetectionLoss, DeformableDetrForSegmentationLoss
     23 from .loss_for_object_detection import ForObjectDetectionLoss, ForSegmentationLoss

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_d_fine.py in <module>
     19 
     20 from ..utils import is_vision_available
---> 21 from .loss_for_object_detection import (
     22     box_iou,
     23 )

/usr/local/lib/python3.11/dist-packages/transformers/loss/loss_for_object_detection.py in <module>
     30 
     31 if is_vision_available():
---> 32     from transformers.image_transforms import center_to_corners_format
     33 
     34 

/usr/local/lib/python3.11/dist-packages/transformers/image_transforms.py in <module>
     46 
     47 if is_tf_available():
---> 48     import tensorflow as tf
     49 
     50 if is_flax_available():

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
class TweetDataset(Dataset):
    def __init__(self, df: pd.DataFrame, max_len: int = MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        text = " " + " ".join(str(row["text"]).split())
        encoding = tokenizer(
            text,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"].squeeze(0)  # (max_len)
        masks = encoding["attention_mask"].squeeze(0)  # (max_len)
        offsets = encoding["offset_mapping"].squeeze(0)  # (max_len, 2)

        item = {"ids": ids, "masks": masks, "tweet": text, "offsets": offsets}

        if self.labeled:
            start_idx, end_idx = self._get_target_idx(row, text, offsets)
            item["start_idx"] = start_idx
            item["end_idx"] = end_idx

        return item

    def _get_target_idx(self, row, tweet, offsets):
        selected = " " + " ".join(str(row["selected_text"]).split())
        len_sel = len(selected) - 1  # exclude leading space

        start_char = tweet.find(selected[1:])
        if start_char == -1:
            return 0, len(offsets) - 1

        end_char = start_char + len_sel

        char_targets = [0] * len(tweet)
        for i in range(start_char, end_char):
            char_targets[i] = 1

        target_idxs = []
        for idx, (s, e) in enumerate(offsets.tolist()):
            if sum(char_targets[s:e]) > 0:
                target_idxs.append(idx)

        if not target_idxs:
            return 0, 0
        return target_idxs[0], target_idxs[-1]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1510285177.py in <cell line: 0>()
----> 1 class TweetDataset(Dataset):
      2     def __init__(self, df: pd.DataFrame, max_len: int = MAX_LEN):
      3         self.df = df.reset_index(drop=True)
      4         self.max_len = max_len
      5         self.labeled = "selected_text" in df.columns

/tmp/ipykernel_55/1510285177.py in TweetDataset()
      1 class TweetDataset(Dataset):
----> 2     def __init__(self, df: pd.DataFrame, max_len: int = MAX_LEN):
      3         self.df = df.reset_index(drop=True)
      4         self.max_len = max_len
      5         self.labeled = "selected_text" in df.columns

NameError: name 'MAX_LEN' is not defined

## === cell 2
def get_loader(df: pd.DataFrame, shuffle: bool = True):
    dataset = TweetDataset(df)
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            "roberta-base", output_hidden_states=True
        )
        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        hidden_size = self.roberta.config.hidden_size
        self.fc = nn.Linear(hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        hidden_states = outputs.hidden_states
        x = torch.stack(
            [hidden_states[-1], hidden_states[-2], hidden_states[-3]], dim=0
        )
        x = torch.mean(x, dim=0)  # (batch, seq_len, hidden)
        x = self.dropout(x)
        logits = self.fc(x)  # (batch, seq_len, 2)
        start_logits, end_logits = logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected = ""
    for ix in range(start_idx, end_idx + 1):
        s, e = offsets[ix]
        selected += text[s:e]
        if ix + 1 < len(offsets) and e < offsets[ix + 1][0]:
            selected += " "
    return selected.strip()




## === cell 5
train_df = pd.read_csv(train_file)
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)

train_split, val_split = train_test_split(
    train_df,
    test_size=0.1,
    random_state=42,
    stratify=train_df["sentiment"],
)

train_loader = get_loader(train_split, shuffle=True)
val_loader = get_loader(val_split, shuffle=False)

model = TweetModel()
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(model.parameters(), lr=LR)

for epoch in range(EPOCHS):
    model.train()
    total_loss = 0
    for batch in train_loader:
        ids = batch["ids"].to(device)
        masks = batch["masks"].to(device)
        start_labels = batch["start_idx"].to(device)
        end_labels = batch["end_idx"].to(device)

        optimizer.zero_grad()
        start_logits, end_logits = model(ids, masks)

        loss_start = criterion(start_logits, start_labels)
        loss_end = criterion(end_logits, end_labels)
        loss = (loss_start + loss_end) / 2
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    avg_train_loss = total_loss / len(train_loader)

    model.eval()
    val_loss = 0
    with torch.no_grad():
        for batch in val_loader:
            ids = batch["ids"].to(device)
            masks = batch["masks"].to(device)
            start_labels = batch["start_idx"].to(device)
            end_labels = batch["end_idx"].to(device)

            start_logits, end_logits = model(ids, masks)
            loss_start = criterion(start_logits, start_labels)
            loss_end = criterion(end_logits, end_labels)
            loss = (loss_start + loss_end) / 2
            val_loss += loss.item()
    avg_val_loss = val_loss / len(val_loader)
    print(
        f"Epoch {epoch+1}/{EPOCHS} - Train loss: {avg_train_loss:.4f} - Val loss: {avg_val_loss:.4f}"
    )




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2149728560.py in <cell line: 0>()
----> 1 train_df = pd.read_csv(train_file)
      2 train_df["text"] = train_df["text"].astype(str)
      3 train_df["selected_text"] = train_df["selected_text"].astype(str)
      4 
      5 train_split, val_split = train_test_split(

NameError: name 'train_file' is not defined

## === cell 6
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_loader(test_df, shuffle=False)

model.eval()
predictions = []

with torch.no_grad():
    for batch in test_loader:
        ids = batch["ids"].to(device)
        masks = batch["masks"].to(device)
        tweets = batch["tweet"]
        offsets = batch["offsets"]  # (batch, max_len, 2)

        start_logits, end_logits = model(ids, masks)

        start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

        start_preds = np.argmax(start_probs, axis=1)
        end_preds = np.argmax(end_probs, axis=1)

        for i in range(ids.size(0)):
            if start_preds[i] > end_preds[i]:
                pred = tweets[i]
            else:
                pred = get_selected_text(
                    tweets[i],
                    start_preds[i],
                    end_preds[i],
                    offsets[i].cpu().numpy(),
                )
            predictions.append(pred)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/454184354.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(test_file)
      2 test_df["text"] = test_df["text"].astype(str)
      3 
      4 test_loader = get_loader(test_df, shuffle=False)
      5 

NameError: name 'test_file' is not defined

## === cell 7
sub_df = pd.read_csv(submission_template)
sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(str(x).split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(str(x).split()) == 1 else x
)

sub_df.to_csv("submission.csv", index=False)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2700518372.py in <cell line: 0>()
----> 1 sub_df = pd.read_csv(submission_template)
      2 sub_df["selected_text"] = predictions
      3 sub_df["selected_text"] = sub_df["selected_text"].apply(
      4     lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
      5 )

NameError: name 'submission_template' is not defined
