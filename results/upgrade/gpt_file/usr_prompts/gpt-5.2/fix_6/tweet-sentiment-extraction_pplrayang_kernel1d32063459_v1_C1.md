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

0.7150368094444275

# 6. Current score

0.44901

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48895) has done: 'I fix the two blockers preventing an end-to-end run: (1) the early `MessageFactory.GetPrototype` crash (a known protobuf/version issue triggered by `transformers` imports in some Kaggle images) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing transformers, and (2) the missing RoBERTa vocab/merges/config/model files by adding a robust fallback that uses the standard `roberta-base` tokenizer/model via `from_pretrained` when the custom `/kaggle/input/robertamodel0524/` directory is absent. These changes preserve the same architecture (RoBERTa + hidden-state averaging + linear head) and the same inference logic, but ensure the code can actually load a model/tokenizer and generate predictions. Finally, I ensure `predictions` is always defined (so cell 6 doesn’t crash) and that `submission.csv` is written with the required columns and row count.'
- What this solution (achieved 0.48895) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation **and** unsetting the C++ implementation env var before any `transformers` import, plus ensuring the runtime uses a compatible protobuf module without changing your model/inference logic. I also correct the custom RoBERTa weight loading: `RobertaModel.from_pretrained` must receive a directory or model name, not a `.bin` file path; this was silently breaking the intended model loading and hurting score. Finally, I keep your existing architecture and prediction logic unchanged, but make the model/tokenizer loading paths robust so the notebook runs end-to-end and always writes a valid `submission.csv`.'
- What this solution (achieved 0.48895) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before importing `transformers`, which is the actual trigger for this error in many Kaggle images. I also make the fold-weight detection robust: if the provided `/kaggle/input/robertalineardropout/` directory is missing (common in your file listing), the code cleanly fall back to a single `roberta-base` model without crashing, while keeping the same model architecture and inference logic. Finally, I keep your submission writing unchanged but ensure the pipeline always reaches it and produces a valid `submission.csv`. These changes should restore correct execution and, when fold weights are available, improve score substantially toward the target (your current 0.48895 is far below 0.715).'
- What this solution (achieved 0.44901) has done: 'I fix the immediate crash coming from the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation *and* setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before importing `transformers` (this is the most reliable way to avoid the `MessageFactory.GetPrototype` error in Kaggle images). I also make the fallback path score-healthy by (when fold weights are missing) loading a QA-style RoBERTa head (`RobertaForQuestionAnswering`) instead of an untrained randomly-initialized linear head; this preserves the same “RoBERTa → start/end logits → argmax span” inference semantics but avoids catastrophic performance from random weights. Finally, I keep your submission formatting the same, ensuring the script always runs end-to-end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_CPP", None)

import sys
import re
import string
import random
import warnings

import numpy as np
import pandas as pd

import torch
from torch import nn
import tokenizers

from transformers import (
    RobertaModel,
    RobertaConfig,
    RobertaTokenizerFast,
    RobertaForQuestionAnswering,
)
from sklearn.model_selection import StratifiedKFold
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
outdir = "/kaggle/input/robertalineardropout/"

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

MAX_LEN = 96
LINEAR_DROPOUT = 0.2

CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

USE_CUSTOM_ROBERTA_FILES = all(
    os.path.exists(p)
    for p in [MODEL_CONFIG_PATH, MODEL_PATH, MODEL_VOCAB_PATH, MODEL_VOCAB_MERGES_PATH]
)

FOLD_WEIGHT_PATHS = [
    f"{outdir}roberta_fold{fold+1}.pth" for fold in range(skf.n_splits)
]
HAS_FOLD_WEIGHTS = os.path.isdir(outdir) and all(
    os.path.exists(p) for p in FOLD_WEIGHT_PATHS
)

print("USE_CUSTOM_ROBERTA_FILES:", USE_CUSTOM_ROBERTA_FILES)
print("HAS_FOLD_WEIGHTS:", HAS_FOLD_WEIGHTS)
print("device:", device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        self.df = df.reset_index(drop=True)
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns

        self.use_hf_fast = not USE_CUSTOM_ROBERTA_FILES
        if self.use_hf_fast:
            self.hf_tokenizer = RobertaTokenizerFast.from_pretrained(
                "roberta-base", add_prefix_space=True
            )
        else:
            self.tokenizer = tokenizers.ByteLevelBPETokenizer(
                vocab=MODEL_VOCAB_PATH,
                merges=MODEL_VOCAB_MERGES_PATH,
                add_prefix_space=True,
                lowercase=True,
            )

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

        if self.use_hf_fast:
            sentiment_ids = self.hf_tokenizer.encode(
                str(row.sentiment),
                add_special_tokens=False,
            )
            enc = self.hf_tokenizer.encode_plus(
                tweet,
                add_special_tokens=False,
                return_offsets_mapping=True,
            )
            tweet_ids = enc["input_ids"]
            tweet_offsets = enc["offset_mapping"]

            ids = [CLS_TOK] + sentiment_ids + [SEP_TOK, SEP_TOK] + tweet_ids + [SEP_TOK]
            offsets = [(0, 0)] * 4 + list(tweet_offsets) + [(0, 0)]
        else:
            encoding = self.tokenizer.encode(tweet)
            sentiment_id = self.tokenizer.encode(str(row.sentiment)).ids

            ids = (
                [CLS_TOK] + sentiment_id + [SEP_TOK, SEP_TOK] + encoding.ids + [SEP_TOK]
            )
            offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

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

        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        if len(selected_text) > 1:
            for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
                if " " + tweet[ind : ind + len_st] == selected_text:
                    idx0 = ind
                    idx1 = ind + len_st - 1
                    break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                if 0 <= ct < len(char_targets):
                    char_targets[ct] = 1

        target_idx = []
        for j, (offset1, offset2) in enumerate(offsets.tolist()):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        if len(target_idx) == 0:
            return 0, 0

        return target_idx[0], target_idx[-1]




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

        if USE_CUSTOM_ROBERTA_FILES:
            config = RobertaConfig.from_pretrained(
                MODEL_CONFIG_PATH, output_hidden_states=True
            )
            self.roberta = RobertaModel.from_pretrained(ROBERTA_PATH, config=config)
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


class TweetQAModel(nn.Module):
    def __init__(self):
        super().__init__()
        if USE_CUSTOM_ROBERTA_FILES:
            config = RobertaConfig.from_pretrained(MODEL_CONFIG_PATH)
            self.qa = RobertaForQuestionAnswering.from_pretrained(
                ROBERTA_PATH, config=config
            )
        else:
            self.qa = RobertaForQuestionAnswering.from_pretrained("roberta-base")

    def forward(self, input_ids, attention_mask):
        out = self.qa(input_ids=input_ids, attention_mask=attention_mask)
        return out.start_logits, out.end_logits




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




## === cell 5
test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df)

predictions = []
models = []

print("loading models..")
if HAS_FOLD_WEIGHTS:
    for fold in tqdm(range(skf.n_splits)):
        model = TweetModel().to(device)
        state = torch.load(f"{outdir}roberta_fold{fold+1}.pth", map_location=device)
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)
else:
    model = TweetQAModel().to(device)
    model.eval()
    models = [model]
    print(
        "Warning: fold weights not found; using a pretrained QA head for inference fallback."
    )

for data in tqdm(test_loader):
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweet = data["tweet"]
    offsets = data["offsets"].cpu().numpy()

    start_logits = []
    end_logits = []
    for model in models:
        with torch.no_grad():
            out_start, out_end = model(ids, masks)
            start_logits.append(torch.softmax(out_start, dim=1).cpu().numpy())
            end_logits.append(torch.softmax(out_end, dim=1).cpu().numpy())

    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)

    for i in range(len(ids)):
        start_pred = int(np.argmax(start_logits[i]))
        end_pred = int(np.argmax(end_logits[i]))
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)

print("Generated predictions:", len(predictions))



## === cell 6
sub_df = pd.read_csv(submission_template)

if len(predictions) != len(sub_df):
    raise RuntimeError(
        f"Predictions length {len(predictions)} != submission rows {len(sub_df)}"
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

sub_df = sub_df[["textID", "selected_text"]]
sub_df.to_csv("submission.csv", index=False)

print(sub_df.head(20).to_string(index=False))
print("\nWrote submission.csv with shape:", sub_df.shape)
