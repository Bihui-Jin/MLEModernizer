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

0.7075039744377136

# 6. Current score

0.53017

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.53017) has done: 'I remove the hard dependency on locally provided RoBERTa asset files (which are missing in this Kaggle environment) and instead load the tokenizer/config/model via `from_pretrained("roberta-base")`, using local files if present and otherwise downloading/caching as Kaggle allows. This fixes the immediate `NoneType` path crash in the dataset and the earlier `FileNotFoundError` in cell 1, while preserving the same RoBERTa-based architecture and inference logic. I also make the weight-loading step robust: if fold `.pth` weights aren’t available, the code still run end-to-end by using the base model (score be lower than the target, but you at least get a valid submission). Finally, I ensure the submission is aligned to `textID` and that the output CSV is correctly written as `submission.csv`.'
- What this solution (achieved 0.53017) has done: 'This crash is coming from an environment-level protobuf incompatibility that Transformers/Tokenizers can hit, not from your model code itself. I fix it by forcing the pure-Python protobuf implementation (via environment variables) *before* importing `transformers`, which avoids the `MessageFactory.GetPrototype` call path. I also make the DataLoader stable in Kaggle by setting `num_workers=0` (this is score-neutral but prevents multiprocessing-related protobuf/tokenizer issues). Everything else (RoBERTa architecture, logits averaging, decoding, and submission writing) is kept the same so the score behavior remains aligned with your current approach.'
- What this solution (achieved 0.53017) has done: 'I fix the protobuf/transformers crash by enforcing the pure-Python protobuf implementation *before any* `transformers`/`tokenizers` import, and by making that setting “hard” (not `setdefault`) so it can’t be overridden by the environment. I also add a safe fallback to disable tokenizer parallelism and keep `num_workers=0`, which is score-neutral but removes a common Kaggle runtime instability. No model architecture, decoding, or inference logic be changed, so the score impact should come only from making the run succeed end-to-end and actually use your fold weights if present. Finally, I keep the output format identical and ensure `submission.csv` is always written.'
- What this solution (achieved 0.53017) has done: 'The crash is due to an incompatibility between installed `protobuf` and `transformers/tokenizers`, where the C++ protobuf backend is used and lacks `MessageFactory.GetPrototype`. I force Transformers to use the pure-Python protobuf backend *before any transformers/tokenizers import* and also disable tokenizers parallelism to avoid multiprocessing-related protobuf issues. I keep your RoBERTa model, logits averaging, decoding, and submission formatting unchanged, only adjusting import order/env vars so the notebook runs end-to-end. The output still be written as `submission.csv` with the required `textID,selected_text` columns.'
- What this solution (achieved 0.53017) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf backend **before** any `transformers/tokenizers` import and by explicitly setting `TRANSFORMERS_NO_PROTOBUF=1`, which prevents Transformers from touching protobuf internals. I also add a safe fallback to set `HF_HUB_DISABLE_TELEMETRY=1` and keep `num_workers=0` to avoid multiprocessing/tokenizer startup issues in Kaggle. No model architecture, decoding, ensembling, or post-processing logic be changed, so any score change should come only from successfully running end-to-end and correctly loading fold weights when available. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.53017) has done: 'We fix the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation and disabling protobuf usage in Transformers *before any* `tokenizers`/`transformers` import occurs (your current cell order still allows an early import path to trigger the crash). We also set a couple of stability env vars (`TOKENIZERS_PARALLELISM`, `num_workers=0` already) and keep the rest of your model/inference logic identical so score changes only come from actually running and (if present) loading the fold weights. Finally, we keep the same I/O paths and ensure `submission.csv` is written with the required columns.'
- What this solution (achieved 0.53017) has done: 'I fix the protobuf/transformers incompatibility that is still triggering `MessageFactory.GetPrototype` by forcing Transformers to avoid protobuf entirely *before* any potential import side-effects and by importing `transformers` only after those env vars are set. I also make the model/tokenizer loading prefer local fold assets (if present) via `local_files_only=True` first, then fall back to cached/downloaded `roberta-base`, which is score-positive when your Kaggle dataset with weights is attached. Finally, I keep your architecture/inference logic unchanged but make the submission merge deterministic and 1:1 aligned to `test_df` ordering so the written `submission.csv` is always valid.'
- What this solution (achieved 0.53017) has done: 'The runtime crash is still coming from protobuf being imported/initialized before your environment variables take effect (directly or indirectly via `tokenizers/transformers`). I harden this by setting the protobuf/transformers env vars at the very top, then proactively importing `google.protobuf` first (so it uses the pure-Python backend), and only then importing `tokenizers/transformers`. I also add a safe fallback that forces `use_fast=False` if the fast tokenizer path triggers protobuf internals in this environment (this keeps the same model/decoding logic; it only changes the tokenizer implementation for stability). No changes are made to the model architecture, ensembling, or decoding; the goal is to run end-to-end and produce `submission.csv` reliably so you can get back to the expected (higher) score when fold weights are present.'
- What this solution (achieved 0.53017) has done: 'The protobuf `MessageFactory.GetPrototype` crash is still being triggered by the C++ protobuf backend getting imported before your environment variables fully take effect in this Kaggle runtime. I harden the top-of-script protobuf suppression by (1) forcing the pure-Python protobuf implementation early and (2) explicitly blocking the `google.protobuf.pyext._message` C++ module via `sys.modules` before importing `tokenizers/transformers`. This is a stability fix (score-neutral) that should let the notebook run end-to-end and actually load your fold weights when present (which is the main path to improving from 0.53 toward your 0.707 target). I keep the model, inference, and submission formatting logic the same, only adjusting import order/guards to eliminate the runtime error.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TRANSFORMERS_NO_PROTOBUF"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"
os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"

import sys

sys.modules["google.protobuf.pyext._message"] = None

import re
import string
import random
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim  # kept to preserve original environment/logic footprint
from sklearn.model_selection import StratifiedKFold

try:
    import google.protobuf  # noqa: F401
except Exception:
    pass

import tokenizers  # noqa: F401
from transformers import AutoModel, AutoTokenizer

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

batch_size = 32
N = 10
skf = StratifiedKFold(n_splits=N, shuffle=True, random_state=seed)

NUM_WORKERS = 0

BASE_INPUT = Path("/kaggle/input")

MODEL_NAME = "roberta-base"

outdir = None
for p in [
    "/kaggle/input/roberta714kernel/",
    "/kaggle/input/roberta714/",
    "/kaggle/input/roberta/",
]:
    if Path(p).exists():
        outdir = p
        break

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2


def find_first_by_name(root: Path, filename: str):
    if not root.exists():
        return None
    hits = list(root.rglob(filename))
    return str(hits[0]) if len(hits) > 0 else None


def _resolve_pretrained_source():
    candidates = []
    if outdir is not None and Path(outdir).exists():
        candidates.append(str(Path(outdir)))
    for d in [
        "/kaggle/input/roberta-base/",
        "/kaggle/input/roberta/",
        "/kaggle/input/roberta714/",
        "/kaggle/input/roberta714kernel/",
    ]:
        if Path(d).exists():
            candidates.append(d)
    return candidates


_PRETRAINED_DIRS = _resolve_pretrained_source()


def load_tokenizer():
    for src in _PRETRAINED_DIRS:
        for use_fast in (True, False):
            try:
                return AutoTokenizer.from_pretrained(
                    src, local_files_only=True, use_fast=use_fast
                )
            except Exception:
                pass

    for use_fast in (True, False):
        try:
            return AutoTokenizer.from_pretrained(MODEL_NAME, use_fast=use_fast)
        except Exception:
            pass

    raise RuntimeError("Failed to load tokenizer from any source.")


_tok_for_ids = load_tokenizer()
CLS_TOK = int(_tok_for_ids.cls_token_id)
PAD_TOK = int(_tok_for_ids.pad_token_id)
SEP_TOK = int(_tok_for_ids.sep_token_id)
del _tok_for_ids




## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        self.tokenizer = load_tokenizer()

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
        sentiment = str(row.sentiment)

        sent_enc = self.tokenizer.encode_plus(
            sentiment,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tweet_enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )

        ids = (
            [CLS_TOK]
            + sent_enc["input_ids"]
            + [SEP_TOK, SEP_TOK]
            + tweet_enc["input_ids"]
            + [SEP_TOK]
        )

        offsets = (
            [(0, 0)] * (1 + len(sent_enc["offset_mapping"]) + 2)
            + list(tweet_enc["offset_mapping"])
            + [(0, 0)]
        )

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]
        else:
            pad_len = self.max_len - len(ids)
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != PAD_TOK).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

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

        roberta_model = None
        for src in _PRETRAINED_DIRS:
            try:
                roberta_model = AutoModel.from_pretrained(
                    src, local_files_only=True, output_hidden_states=True
                )
                break
            except Exception:
                pass
        if roberta_model is None:
            roberta_model = AutoModel.from_pretrained(
                MODEL_NAME, output_hidden_states=True
            )

        self.roberta = roberta_model
        config = self.roberta.config

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = outputs.hidden_states  # tuple of hidden states

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
        o1, o2 = offsets[ix]
        selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom > 0 else 0.0


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)

    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df, batch_size=batch_size)

models = []
print("loading models..")

weight_paths = []
if outdir is not None and Path(outdir).exists():
    for fold in range(skf.n_splits):
        weight_paths.append(str(Path(outdir) / f"roberta_fold{fold+1}.pth"))
else:
    for fold in range(skf.n_splits):
        hit = find_first_by_name(BASE_INPUT, f"roberta_fold{fold+1}.pth")
        weight_paths.append(hit)

for fold, wpath in enumerate(weight_paths):
    if wpath is not None and os.path.exists(wpath):
        model = TweetModel().to(device)
        sd = torch.load(wpath, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        model.eval()
        models.append(model)
        print(f"loaded {wpath}")

if len(models) == 0:
    print(
        "WARNING: No fold weights found; using a single base roberta model. "
        "This will produce a valid submission but likely lower score."
    )
    model = TweetModel().to(device)
    model.eval()
    models = [model]

predictions = []
for data in tqdm(test_loader, total=len(test_loader)):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweet = data["tweet"]
    offsets = data["offsets"].cpu().numpy()

    start_logits_list = []
    end_logits_list = []
    for model in models:
        with torch.no_grad():
            start_logits, end_logits = model(ids, masks)
            start_logits_list.append(torch.softmax(start_logits, dim=1).cpu().numpy())
            end_logits_list.append(torch.softmax(end_logits, dim=1).cpu().numpy())

    start_logits_mean = np.mean(start_logits_list, axis=0)
    end_logits_mean = np.mean(end_logits_list, axis=0)

    for i in range(ids.size(0)):
        start_pred = int(np.argmax(start_logits_mean[i]))
        end_pred = int(np.argmax(end_logits_mean[i]))
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(
                tweet[i], start_pred, end_pred, offsets[i].tolist()
            )
        predictions.append(pred)

if len(predictions) != len(test_df):
    raise RuntimeError(
        f"predictions length {len(predictions)} != test_df length {len(test_df)}"
    )

sub_df = pd.read_csv(submission_template)
sub_df = sub_df[["textID"]].merge(
    test_df[["textID"]], on="textID", how="right", sort=False, validate="1:1"
)
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

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'
