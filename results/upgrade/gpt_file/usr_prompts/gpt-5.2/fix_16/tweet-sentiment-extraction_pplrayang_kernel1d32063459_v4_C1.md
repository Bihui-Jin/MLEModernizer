# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7140237092971802

# 6. Current score

0.49696

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.53017) has done: 'I fix the runtime blockers so the notebook can run end-to-end and actually write `submission.csv`. The main issues are (1) a protobuf/tokenizers import-time crash and (2) missing RoBERTa vocab/merges/model files at the hardcoded paths, which prevents tokenizer/model initialization. I add a small, safe fallback to load `roberta-base` config/model/tokenizer from the installed `transformers` package cache when the Kaggle input paths aren’t present, without changing the model architecture or inference logic. Finally, I ensure `predictions` is always defined (so cell 6 can run) and the submission columns/length match the template.'
- What this solution (achieved 0.53017) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by forcing a compatible protobuf runtime and importing transformers/tokenizers only after setting the env vars, which unblocks the whole pipeline. I also correct the RoBERTa weight-loading path when using a single `.bin` file by loading from the containing directory (not the file path) so local weights (if present) load correctly without changing the model. Finally, I keep the existing inference/ensembling logic intact, but make the model-loading loop robust and deterministic so it always produces a valid `submission.csv` with the right length and columns.'
- What this solution (achieved 0.53017) has done: 'I fix the import-time protobuf/tokenizers crash by setting the required environment variables before importing `tokenizers`/`transformers`, and by ensuring the pure-Python protobuf implementation is used in this Kaggle runtime. I also make model/tokenizer loading robust: if the hardcoded local RoBERTa files aren’t present, it consistently fall back to `roberta-base` from the transformers cache without changing the architecture or inference logic. These changes are execution/stability focused and should also improve the score versus the current run, because they prevent the “randomly initialized head” fallback that severely hurts performance when fold weights can’t be loaded. The script still write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.53017) has done: 'I fix the immediate runtime blocker caused by an incompatible protobuf C++ implementation used by `tokenizers/transformers` in this Kaggle image by forcing the pure-Python protobuf before those imports (and removing the earlier `torch/transformers` imports that currently happen too soon). I keep the model/inference logic identical, but make model-loading robust by allowing fold weights to load even if they were saved with `DataParallel` (strip `module.`) so you actually use the trained heads (this should raise score toward the target). Finally, I keep submission formatting the same and ensure `submission.csv` is always written with the correct columns and row count.'
- What this solution (achieved 0.53017) has done: 'I fix the import-time protobuf/tokenizers crash by ensuring the environment variables are set before any `tokenizers`/`transformers` import, and by forcing a compatible protobuf runtime mode early. Then I make model/tokenizer loading robust without changing the model: if the hardcoded Kaggle input paths don’t exist, it deterministically fall back to `roberta-base` from the local transformers cache. Finally, I keep your inference/ensembling logic intact but ensure fold weight loading is resilient (common `state_dict` nesting / `module.` prefix), so trained weights actually get used—this should increase the score toward your target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.53017) has done: 'I fix the import-time protobuf/tokenizers crash by setting the protobuf env vars before *any* tokenizers/transformers import and by avoiding the standalone `tokenizers` import (it’s not needed when we can use `AutoTokenizer` fast tokenizer). Then I make local-model loading consistent by using `RobertaModel.from_pretrained(ROBERTA_PATH)` only when the directory actually contains a valid HF model layout; otherwise it fall back to `roberta-base` from cache (same architecture). Finally, I keep your inference/ensembling logic intact but ensure we try to load all available fold weights (including fold1/2) so we don’t accidentally ensemble fewer models than intended, which should move score upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.53017) has done: 'I fix the import-time crash coming from an incompatible protobuf/tokenizers stack by forcing the pure-Python protobuf implementation *before* any `transformers` import and by avoiding eager `transformers` imports until after env vars are set. Then I make the “local HF directory” detection robust for your provided `ROBERTA_PATH` layout (your code only checks `config.json`, but your files are named `roberta-base-config.json`, etc.), so the model/tokenizer can reliably load the intended local artifacts when present; otherwise it fall back to `roberta-base` from cache (same architecture). These changes are execution-critical and should also improve score toward the target because they prevent running with a randomly initialized head due to failed tokenizer/model initialization or missed local files. Finally, I keep the inference and submission formatting logic identical and ensure `submission.csv` is always written with the correct columns/row count.'
- What this solution (achieved 0.53017) has done: 'I fix the immediate runtime blocker (`MessageFactory.GetPrototype`), which comes from an incompatible protobuf runtime being picked up when importing `transformers`/`tokenizers`. The minimal reliable fix in this Kaggle image is to force the pure-Python protobuf implementation *before* any transformers import and to ensure `protobuf`’s C++ implementation is not used. I keep your model/dataset/inference logic intact, but reorder imports and add a small defensive fallback so tokenizers/models always load (local files if valid, otherwise `roberta-base` from cache). This should both make the notebook run end-to-end and improve score versus the current run by avoiding broken initialization paths.'
- What this solution (achieved 0.53028) has done: 'I fix the `protobuf`/`transformers` import crash by setting the protobuf environment variables before *any* torch/transformers-related imports and by forcing the pure-Python protobuf implementation early. Then I make tokenizer/model loading more robust without changing your architecture: if the custom `/kaggle/input/robertamodel0524/` folder is not a valid HF directory, we fall back to `roberta-base` (so the notebook always runs), while still using local fold weights when available (to improve score toward the target). Finally, I add a small safe guard so `selected_text` is never empty/whitespace and ensure the submission file is written with the correct columns and row count.'
- What this solution (achieved 0.53028) has done: 'I fix the `protobuf`/`transformers` import crash by ensuring we don’t import `transformers` until after the environment variables are set and by explicitly importing `google.protobuf` once the env is configured. Then I make the tokenizer construction compatible with current `transformers` (don’t pass `vocab_file`/`merges_file` to `AutoTokenizer.from_pretrained`, which can break depending on version), while still preferring the provided local RoBERTa files if they form a valid HF directory. These are execution-critical fixes and score-positive because they prevent the pipeline from falling back to a randomly initialized model/tokenizer path. The rest of the model/inference logic and submission formatting remains unchanged.'
- What this solution (achieved 0.53028) has done: 'I fix the import-time crash caused by the protobuf runtime/API mismatch that breaks `transformers/tokenizers` loading in this Kaggle image. The minimal stable fix is to force the pure-Python protobuf implementation early and (critically) pin the protobuf Python package to a compatible version at runtime before importing `transformers`, then restart imports cleanly. This is execution-critical and score-positive because it allows the script to actually load RoBERTa and (when present) your fold weights instead of dying before inference. All model/inference logic and submission formatting remain the same, and the script still write `submission.csv` with the required columns and row count.'
- What this solution (achieved 0.38764) has done: 'Your current score (0.53028) is far below the target (0.7140), and the most likely cause is that you are either not loading the intended trained fold weights (so you’re effectively using a mostly random span head) or you’re allowing predictions to select tokens from the sentiment-prefix portion instead of the tweet text itself. I make two minimal, score-positive fixes that preserve your model and inference approach: (1) make checkpoint loading tolerant to common key mismatches (`fc.` vs `linear.`, `roberta.` vs `bert.`) while still requiring that weights actually load (so we use trained heads when available), and (2) constrain decoding so start/end indices are chosen only from tokens that map to real tweet character offsets (not the sentiment tokens or special tokens), which better matches the competition’s span-extraction evaluation. These changes keep architecture/loss/training untouched and only adjust robustness + post-processing to align with the metric. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.49675) has done: 'Your score gap to the target is large (0.38764 → 0.714), and the most likely cause is still span decoding picking indices that don’t correspond to real tweet characters (or picking an end far before/after start) even when weights load. I keep your model and ensembling identical, but change decoding to (a) restrict start/end to tokens with valid tweet offsets and (b) choose the best (start,end) pair under a maximum span length using the sum of start/end probabilities—this aligns with the word-level Jaccard metric without changing the architecture/training. I also add the standard competition heuristic: for `neutral` sentiment predict the full tweet, which is a minimal post-processing change widely used for this task. These are small, inference-only changes and should move score upward toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.49696) has done: 'Your current score (0.49675) is far below the target (0.7140), so we should make small, inference-only fixes that better match the Jaccard metric without changing the model/training core. I keep your model, tokenization scheme, and ensembling intact, but improve decoding by (1) restricting valid tokens to only the tweet portion (excluding sentiment/special tokens) and (2) selecting the best (start,end) pair by maximizing start_logit+end_logit (instead of softmax probabilities), which is a standard, minimal alignment fix for span extraction. I also add a tiny post-processing cleanup to better preserve exact tweet substrings (strip only, and fall back to original tweet casing when needed), keeping submission formatting unchanged. These changes are narrow, deterministic, and should move score upward toward the target band.'

# 9. Code solution

## === cell 0
import os
import re
import sys
import random
import warnings

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import subprocess

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver
except Exception:
    _pb_ver = "0"


def _version_tuple(v):
    m = re.match(r"^(\d+)\.(\d+)\.(\d+)", str(v))
    if not m:
        return (0, 0, 0)
    return tuple(int(x) for x in m.groups())


if _version_tuple(_pb_ver) >= (4, 21, 0) or _version_tuple(_pb_ver) == (0, 0, 0):
    subprocess.run(
        [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"],
        check=False,
    )
    import importlib
    import google.protobuf as _gp  # noqa: F401

    importlib.reload(_gp)

import numpy as np
import pandas as pd

import torch
from torch import nn

from sklearn.model_selection import StratifiedKFold

from transformers import RobertaModel, RobertaConfig, AutoTokenizer
from tqdm.auto import tqdm

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
        torch.backends.cudnn.benchmark = True


seed = 42
seed_everything(seed)

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)
NUM_WORKERS = 2

ROBERTA_PATH = "/kaggle/input/robertamodel0524/"
MODEL_CONFIG_PATH = ROBERTA_PATH + "roberta-base-config.json"
MODEL_PATH = ROBERTA_PATH + "roberta-base-pytorch_model.bin"
MODEL_VOCAB_PATH = ROBERTA_PATH + "roberta-base-vocab.json"
MODEL_VOCAB_MERGES_PATH = ROBERTA_PATH + "roberta-base-merges.txt"
outdir = "/kaggle/input/roberta714kernel/"

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _local_roberta_available() -> bool:
    if not os.path.isdir(ROBERTA_PATH):
        return False
    if os.path.exists(os.path.join(ROBERTA_PATH, "config.json")):
        return True
    if (
        os.path.exists(MODEL_CONFIG_PATH)
        and os.path.exists(MODEL_VOCAB_PATH)
        and os.path.exists(MODEL_VOCAB_MERGES_PATH)
    ):
        return True
    return False


USE_LOCAL_HF_DIR = _local_roberta_available()

if USE_LOCAL_HF_DIR and os.path.exists(os.path.join(ROBERTA_PATH, "config.json")):
    _hf_tokenizer = AutoTokenizer.from_pretrained(ROBERTA_PATH, use_fast=True)
else:
    _hf_tokenizer = AutoTokenizer.from_pretrained("roberta-base", use_fast=True)




## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        self.tokenizer = _hf_tokenizer

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet_norm, offsets, tweet_orig = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet_norm
        data["offsets"] = offsets
        data["tweet_orig"] = tweet_orig
        if "sentiment" in row:
            data["sentiment"] = str(row.sentiment)

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet_norm, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet_norm = " " + " ".join(str(row.text).lower().split())
        tweet_orig = " " + " ".join(str(row.text).split())

        cls_id = int(self.tokenizer.cls_token_id)
        sep_id = int(self.tokenizer.sep_token_id)
        pad_id = int(self.tokenizer.pad_token_id)

        sent_enc = self.tokenizer(
            str(row.sentiment), add_special_tokens=False, return_offsets_mapping=True
        )
        tweet_enc = self.tokenizer(
            tweet_norm, add_special_tokens=False, return_offsets_mapping=True
        )

        ids = (
            [cls_id]
            + list(sent_enc["input_ids"])
            + [sep_id, sep_id]
            + list(tweet_enc["input_ids"])
            + [sep_id]
        )
        offsets = (
            [(0, 0)] * (1 + len(sent_enc["input_ids"]) + 2)
            + list(tweet_enc["offset_mapping"])
            + [(0, 0)]
        )

        global PAD_TOK
        PAD_TOK = pad_id

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != PAD_TOK).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet_norm, offsets, tweet_orig

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
        for j, (offset1, offset2) in enumerate(offsets.tolist()):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




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

        if USE_LOCAL_HF_DIR:
            if os.path.exists(os.path.join(ROBERTA_PATH, "config.json")):
                config = RobertaConfig.from_pretrained(
                    ROBERTA_PATH, output_hidden_states=True
                )
                self.roberta = RobertaModel.from_pretrained(ROBERTA_PATH, config=config)
            elif os.path.exists(MODEL_CONFIG_PATH):
                config = RobertaConfig.from_pretrained(
                    MODEL_CONFIG_PATH, output_hidden_states=True
                )
                if os.path.exists(MODEL_PATH):
                    self.roberta = RobertaModel.from_pretrained(
                        "roberta-base",
                        config=config,
                        state_dict=torch.load(MODEL_PATH, map_location="cpu"),
                    )
                else:
                    self.roberta = RobertaModel.from_pretrained(
                        "roberta-base", config=config
                    )
            else:
                config = RobertaConfig.from_pretrained(
                    "roberta-base", output_hidden_states=True
                )
                self.roberta = RobertaModel.from_pretrained(
                    "roberta-base", config=config
                )
        else:
            config = RobertaConfig.from_pretrained(
                "roberta-base", output_hidden_states=True
            )
            self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)

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
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)


def _best_span_from_logits(
    start_logit_1d, end_logit_1d, valid_mask_1d, max_span_len=30
):
    s = np.where(valid_mask_1d, start_logit_1d, -1e9).astype(np.float32)
    e = np.where(valid_mask_1d, end_logit_1d, -1e9).astype(np.float32)

    valid_idx = np.where(valid_mask_1d)[0]
    if valid_idx.size == 0:
        return 0, 0

    best_score = -1e18
    best_i = int(valid_idx[0])
    best_j = int(valid_idx[0])

    for i in valid_idx:
        j_max = min(int(i) + int(max_span_len), len(s) - 1)
        score_ij = s[i] + e[i : j_max + 1]
        j_rel = int(score_ij.argmax())
        score = float(score_ij[j_rel])
        j = int(i + j_rel)
        if score > best_score:
            best_score = score
            best_i, best_j = int(i), int(j)

    return best_i, best_j


def _tweet_token_valid_mask(offsets_2d: np.ndarray) -> np.ndarray:
    return ((offsets_2d[:, 0] != 0) | (offsets_2d[:, 1] != 0)).astype(bool)


def _recover_from_original(tweet_norm: str, tweet_orig: str, pred_norm: str) -> str:
    tn = str(tweet_norm)
    to = str(tweet_orig)
    pn = str(pred_norm).strip()
    if pn == "":
        return to.strip()

    pos = tn.find(pn)
    if pos >= 0:
        try:
            return to[pos : pos + len(pn)].strip()
        except Exception:
            return pn
    return pn




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df, batch_size=batch_size)

predictions = []
models = []

print("loading models..")


def _strip_module_prefix(state_dict):
    if not isinstance(state_dict, dict):
        return state_dict
    if len(state_dict) > 0 and all(
        isinstance(k, str) and k.startswith("module.") for k in state_dict.keys()
    ):
        return {k[len("module.") :]: v for k, v in state_dict.items()}
    return state_dict


def _extract_state_dict(obj):
    if isinstance(obj, dict):
        for key in ("state_dict", "model_state_dict", "model", "net"):
            if key in obj and isinstance(obj[key], dict):
                return obj[key]
    return obj


def _remap_state_dict_keys_for_compat(state_dict: dict) -> dict:
    """
    Minimal score-positive robustness: many public TSE checkpoints use different attribute names
    (e.g., bert/roberta backbone prefix, linear vs fc head). Remapping helps us actually load
    trained weights rather than silently falling back to a random head.
    """
    if not isinstance(state_dict, dict):
        return state_dict

    new_sd = {}
    for k, v in state_dict.items():
        nk = k

        if nk.startswith("bert."):
            nk = "roberta." + nk[len("bert.") :]
        if nk.startswith("roberta_model."):
            nk = "roberta." + nk[len("roberta_model.") :]

        if nk.startswith("linear."):
            nk = "fc." + nk[len("linear.") :]
        if nk.startswith("classifier."):
            nk = "fc." + nk[len("classifier.") :]

        new_sd[nk] = v
    return new_sd


def _load_weights_strict_enough(model, state):
    """
    Keep semantics: we still use the same model; this only improves likelihood that fold weights
    load correctly. We require that all fc.* params load; otherwise the head is random and score tanks.
    """
    missing, unexpected = model.load_state_dict(state, strict=False)

    head_missing = [k for k in missing if k.startswith("fc.")]
    if len(head_missing) > 0:
        raise RuntimeError(f"Checkpoint did not load model head params: {head_missing}")

    return missing, unexpected


for fold in range(skf.n_splits):
    weight_path = f"{outdir}roberta_fold{fold+1}.pth"
    if not os.path.exists(weight_path):
        continue

    model = TweetModel().to(device)
    ckpt = torch.load(weight_path, map_location="cpu")
    state = _extract_state_dict(ckpt)
    state = _strip_module_prefix(state)
    state = _remap_state_dict_keys_for_compat(state)

    try:
        _load_weights_strict_enough(model, state)
    except Exception as e:
        print(f"skip {weight_path} (incompatible): {e}")
        continue

    model.eval()
    print(f"load {weight_path}")
    models.append(model)

if len(models) == 0:
    print(
        "WARNING: No compatible fold weights found; using a roberta-base backbone with randomly initialized head for inference."
    )
    model = TweetModel().to(device)
    model.eval()
    models = [model]


for data in tqdm(test_loader, total=len(test_loader)):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweet = data["tweet"]  # normalized, lowercased (model-space)
    tweet_orig = data["tweet_orig"]  # original-ish whitespace, for nicer output
    offsets = data["offsets"].numpy()  # (bs, max_len, 2)
    sentiments = data.get("sentiment", [""] * len(tweet))

    start_logits_ens = []
    end_logits_ens = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits_ens.append(output[0].detach().cpu().numpy())
            end_logits_ens.append(output[1].detach().cpu().numpy())

    start_logits = np.mean(start_logits_ens, axis=0)
    end_logits = np.mean(end_logits_ens, axis=0)

    for i in range(len(ids)):
        valid = _tweet_token_valid_mask(offsets[i])

        if str(sentiments[i]).lower() == "neutral":
            pred = str(tweet_orig[i]).strip()
            if pred.strip() == "":
                pred = str(tweet[i]).strip()
            predictions.append(pred)
            continue

        start_pred, end_pred = _best_span_from_logits(
            start_logits[i], end_logits[i], valid_mask_1d=valid, max_span_len=30
        )

        if start_pred > end_pred:
            pred_norm = str(tweet[i]).strip()
        else:
            pred_norm = get_selected_text(
                tweet[i], start_pred, end_pred, offsets[i]
            ).strip()

        pred = _recover_from_original(tweet[i], tweet_orig[i], pred_norm)

        if str(pred).strip() == "":
            pred = str(tweet_orig[i]).strip()
            if pred.strip() == "":
                pred = str(tweet[i]).strip()

        predictions.append(pred)

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"Prediction length mismatch: {len(predictions)} vs {len(test_df)}"
    )



## === cell 6
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

sub_df = sub_df[["textID", "selected_text"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head(20))
print("Wrote submission.csv with shape:", sub_df.shape)
