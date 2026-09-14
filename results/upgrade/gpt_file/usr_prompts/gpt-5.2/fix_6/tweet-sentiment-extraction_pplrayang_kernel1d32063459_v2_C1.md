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

0.7155378460884094

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the environment/runtime blockers so the notebook can run end-to-end and actually write `submission.csv`. The main issues are (1) a `protobuf`/Transformers import-time crash and (2) hardcoded RoBERTa/tokenizer paths that don’t exist in your provided dataset tree, causing the tokenizer files to be missing and preventing inference (which also causes `predictions` to be undefined). I switch to loading `roberta-base` and its fast tokenizer directly from the installed `transformers` package (offline, no internet), and if the external fold checkpoints aren’t present, I fall back to a safe rule-based submission so you always get a valid CSV. This preserves your core model architecture and inference semantics when checkpoints are available, while ensuring a submission file is produced in all cases.'
- What this solution (achieved 0.59324) has done: 'I fix the import-time crash by forcing the pure-Python protobuf implementation *before* importing `transformers`, and I make the RoBERTa/tokenizer loading robust to Kaggle’s offline environment by first trying local Kaggle model caches and then falling back to `roberta-base` only if available locally. This should restore the intended checkpoint-based inference (which is what gets you closer to the target score) instead of silently falling back to the heuristic. I also fix a determinism/perf setting bug (`cudnn.deterministic` + `benchmark` conflict) without changing training/inference semantics, and I keep the submission writing/format exactly as required.'
- What this solution (achieved 0.57486) has done: 'I fix the import-time protobuf crash that prevents `transformers` from loading by forcing a compatible protobuf runtime setting *and* avoiding the code paths that trigger the `MessageFactory.GetPrototype` issue. Then I make the RoBERTa/tokenizer/model loading robust in Kaggle’s offline environment by trying common local cache/dataset locations first and only using `from_pretrained(..., local_files_only=True)` on those resolved paths. Finally, to move score up toward the target, I prevent the low-scoring “echo the whole tweet” fallback unless no local roberta weights can be found at all; if roberta weights exist but fold checkpoints are missing, we still run the base model (random head) would be worse, so we keep the heuristic, but we improve the heuristic slightly using a minimal, sentiment-aware word-span selection (still rule-based, no architecture/training changes) to better align with Jaccard.'

# 9. Code solution

## === cell 0
import os
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.setdefault("TRANSFORMERS_NO_PROTOBUF", "1")

import numpy as np
import pandas as pd
import random
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from tqdm.auto import tqdm

from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast

warnings.filterwarnings("ignore")


def seed_everything(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    os.environ["PYTHONHASHSEED"] = str(seed_value)

    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

PRETRAINED_NAME = "roberta-base"
outdir = "/kaggle/input/robertalineardropout/"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _path_has_roberta_files(path: str) -> bool:
    if not isinstance(path, str) or not path:
        return False
    if not os.path.isdir(path):
        return False
    has_cfg = os.path.isfile(os.path.join(path, "config.json"))
    has_vocab = os.path.isfile(os.path.join(path, "vocab.json"))
    has_merges = os.path.isfile(os.path.join(path, "merges.txt"))
    has_sp = os.path.isfile(os.path.join(path, "sentencepiece.bpe.model"))
    return has_cfg and ((has_vocab and has_merges) or has_sp)


def _try_resolve_pretrained_dir_candidates():
    candidates = [
        "/kaggle/input/roberta-base",
        "/kaggle/input/roberta",
        "/kaggle/input/tweet-sentiment-extraction/roberta-base",
        "/kaggle/input/tweet-sentiment-extraction/roberta",
        os.path.expanduser("~/.cache/huggingface/hub/models--roberta-base/snapshots"),
        "/kaggle/working/.cache/huggingface/hub/models--roberta-base/snapshots",
        PRETRAINED_NAME,  # may resolve via local HF cache if present
    ]

    expanded = []
    for c in candidates:
        if isinstance(c, str) and c.endswith("snapshots") and os.path.isdir(c):
            try:
                for child in sorted(os.listdir(c)):
                    p = os.path.join(c, child)
                    if os.path.isdir(p):
                        expanded.append(p)
            except Exception:
                pass
        else:
            expanded.append(c)

    preferred = []
    for c in expanded:
        if isinstance(c, str) and _path_has_roberta_files(c):
            preferred.append(c)
    rest = [c for c in expanded if c not in preferred]
    return preferred + rest


def _try_load_tokenizer_and_config():
    candidates = _try_resolve_pretrained_dir_candidates()
    last_err = None
    for name in candidates:
        try:
            tok = RobertaTokenizerFast.from_pretrained(name, local_files_only=True)
            cfg = RobertaConfig.from_pretrained(
                name, output_hidden_states=True, local_files_only=True
            )
            print(f"Loaded tokenizer/config from: {name}")
            return tok, cfg, name
        except Exception as e:
            last_err = e
            continue
    raise RuntimeError(
        "Could not load RoBERTa tokenizer/config from local files (offline). "
        "Please ensure roberta-base exists in the Kaggle environment (either as a dataset input or in the HF cache)."
    ) from last_err


TOKENIZER, _ROBERTA_CONFIG, PRETRAINED_RESOLVED = _try_load_tokenizer_and_config()




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_55/1931400908.py in <cell line: 0>()
     17 from tqdm.auto import tqdm
     18 
---> 19 from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
     20 
     21 warnings.filterwarnings("ignore")

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
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(str(row.text).lower().split())
        sentiment = str(row.sentiment).lower()

        enc = TOKENIZER(
            sentiment,
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_offsets_mapping=True,
        )

        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        masks = torch.tensor(enc["attention_mask"], dtype=torch.long)
        offsets = torch.tensor(enc["offset_mapping"], dtype=torch.long)

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (o1, o2) in enumerate(offsets.tolist()):
            if o1 == o2 == 0:
                continue
            if sum(char_targets[o1:o2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1049126818.py in <cell line: 0>()
----> 1 class TweetDataset(torch.utils.data.Dataset):
      2     def __init__(self, df, max_len=MAX_LEN):
      3         self.df = df.reset_index(drop=True)
      4         self.max_len = max_len
      5         self.labeled = "selected_text" in df.columns

/tmp/ipykernel_55/1049126818.py in TweetDataset()
      1 class TweetDataset(torch.utils.data.Dataset):
----> 2     def __init__(self, df, max_len=MAX_LEN):
      3         self.df = df.reset_index(drop=True)
      4         self.max_len = max_len
      5         self.labeled = "selected_text" in df.columns

NameError: name 'MAX_LEN' is not defined

## === cell 2
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        config = _ROBERTA_CONFIG
        self.roberta = RobertaModel.from_pretrained(
            PRETRAINED_RESOLVED, config=config, local_files_only=True
        )

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states

        x = torch.stack([hs[-1], hs[-2], hs[-3]])
        x = torch.mean(x, 0)
        x = self.dropout(x)
        x = self.fc(x)

        start_logits, end_logits = x.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)
        return start_logits, end_logits




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        o1, o2 = int(offsets[ix][0]), int(offsets[ix][1])
        if o1 == o2 == 0:
            continue
        selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and int(offsets[ix][1]) < int(offsets[ix + 1][0]):
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = int(np.argmax(start_logits))
    end_pred = int(np.argmax(end_logits))
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)


_POS_WORDS = {
    "good",
    "great",
    "love",
    "awesome",
    "best",
    "amazing",
    "nice",
    "happy",
    "fantastic",
    "excellent",
    "perfect",
    "wonderful",
    "thanks",
    "thank",
}
_NEG_WORDS = {
    "bad",
    "hate",
    "worst",
    "awful",
    "sad",
    "terrible",
    "horrible",
    "annoying",
    "angry",
    "disappointed",
    "sucks",
    "suck",
    "sorry",
}


def heuristic_select(text: str, sentiment: str) -> str:
    t = str(text)
    s = str(sentiment).lower()
    t_stripped = t.strip()
    if s == "neutral" or len(t_stripped.split()) <= 2:
        return t_stripped

    words = t_stripped.split()
    low_words = [w.strip(".,!?;:\"'()[]{}").lower() for w in words]

    lex = _POS_WORDS if s == "positive" else _NEG_WORDS
    idxs = [i for i, w in enumerate(low_words) if w in lex]
    if not idxs:
        if s == "negative":
            return " ".join(words[max(0, len(words) - 3) :])
        return " ".join(words[: min(3, len(words))])

    i0, i1 = min(idxs), max(idxs)
    i0 = max(0, i0 - 1)
    i1 = min(len(words) - 1, i1 + 1)
    return " ".join(words[i0 : i1 + 1]).strip()




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df, batch_size=batch_size)

predictions = []
models = []


def _checkpoint_exists(path: str) -> bool:
    try:
        return os.path.isfile(path)
    except Exception:
        return False


print("loading models..")
available_ckpts = []
for fold in range(skf.n_splits):
    state_path = f"{outdir}roberta_fold{fold+1}.pth"
    if _checkpoint_exists(state_path):
        available_ckpts.append((fold, state_path))

if len(available_ckpts) > 0:
    for fold, state_path in tqdm(available_ckpts):
        model = TweetModel().to(device)
        state = torch.load(state_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        models.append(model)

    for data in tqdm(test_loader):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data["tweet"]
        offsets = data["offsets"].cpu().numpy()

        start_logits_list = []
        end_logits_list = []
        for model in models:
            with torch.no_grad():
                out_start, out_end = model(ids, masks)
                start_logits_list.append(torch.softmax(out_start, dim=1).cpu().numpy())
                end_logits_list.append(torch.softmax(out_end, dim=1).cpu().numpy())

        start_logits = np.mean(start_logits_list, axis=0)
        end_logits = np.mean(end_logits_list, axis=0)

        for i in range(ids.size(0)):
            start_pred = int(np.argmax(start_logits[i]))
            end_pred = int(np.argmax(end_logits[i]))
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)
else:
    print(f"No fold checkpoints found under {outdir}. Using heuristic predictions.")
    for _, row in test_df.iterrows():
        predictions.append(heuristic_select(row["text"], row["sentiment"]))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3809641892.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(test_file)
      2 test_df["text"] = test_df["text"].astype(str)
      3 test_loader = get_test_loader(test_df, batch_size=batch_size)
      4 
      5 predictions = []

NameError: name 'test_file' is not defined

## === cell 6
sub_df = pd.read_csv(submission_template)

if len(predictions) != len(sub_df):
    raise ValueError(
        f"Prediction length {len(predictions)} != submission rows {len(sub_df)}"
    )

sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].astype(str)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
)

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print(f"Wrote {sub_path} with shape {sub_df.shape}")
print(sub_df.head(10))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3840356560.py in <cell line: 0>()
----> 1 sub_df = pd.read_csv(submission_template)
      2 
      3 if len(predictions) != len(sub_df):
      4     raise ValueError(
      5         f"Prediction length {len(predictions)} != submission rows {len(sub_df)}"

NameError: name 'submission_template' is not defined
