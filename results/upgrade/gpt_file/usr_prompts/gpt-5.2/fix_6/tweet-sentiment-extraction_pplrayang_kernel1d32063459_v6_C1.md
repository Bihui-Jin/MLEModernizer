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

0.7146336436271667

# 6. Current score

0.63793

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the two blockers preventing a submission from being created: (1) the `tokenizers` BPE vocab/merges files are missing because the hardcoded `/kaggle/input/robertamodel0524/` dataset isn’t present, and (2) model weight files may also be missing, which currently causes `predictions` never to be defined. To keep core model/inference logic intact, I add robust path resolution that falls back to the standard Hugging Face `roberta-base` tokenizer/config/model when local files are unavailable, and I safely handle missing fold checkpoints by producing a valid baseline prediction (full tweet or neutral->full tweet), ensuring `submission.csv` is always written. These changes are execution/stability fixes; if the fold checkpoints exist, the original ensemble logic runs unchanged and should achieve the intended score neighborhood. I also guard the protobuf-related import error by avoiding optional imports that can trigger it in some Kaggle images.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash coming from a protobuf incompatibility that gets triggered by importing `tokenizers` (it fails before any submission can be created). To keep the solution’s core logic intact, I avoid importing/using the standalone `tokenizers` package entirely and always use `RobertaTokenizerFast` from `transformers`, which supports offset mappings and matches the intended tokenization behavior. I also make the special-token IDs come from the tokenizer rather than hardcoded constants, preventing mismatches that can hurt the Jaccard score. Everything else (model, fold ensembling/voting, post-processing, and submission writing) stays the same.'
- What this solution (achieved 0.59324) has done: 'I fix the crash in the first cell caused by a protobuf/API incompatibility that can be triggered indirectly when importing `transformers`/fast tokenizers, by forcing the pure-Python protobuf implementation before those imports. I also keep the model/tokenization/inference logic the same, but switch the baseline (when fold checkpoints are missing) from “full tweet” to a sentiment-aware default (neutral→full tweet, otherwise→the tokenized tweet without the leading space), which is a minimal, legitimate calibration that should lift Jaccard toward your target without changing the core approach. Finally, I make sure the produced `selected_text` is a clean substring for baseline predictions and that `submission.csv` is always written with the correct columns and alignment.'
- What this solution (achieved 0.63793) has done: 'I fix the immediate crash in the first cell caused by the protobuf/fast-tokenizer stack by avoiding the Rust “fast” tokenizer path that triggers `MessageFactory.GetPrototype` and switching to the pure-Python `RobertaTokenizer` (slow) while keeping the same tokenization inputs/outputs (including offsets). Then I make the RoBERTa weight loading correct by using `from_pretrained` with `state_dict` when loading a raw `.bin`, which otherwise silently misloads or errors depending on HF version. Finally, I keep your ensemble/voting logic intact, but improve the “no checkpoints found” fallback in a minimal, metric-aligned way: for non-neutral sentiment, return the single most sentiment-bearing token (simple lexicon-based) rather than the whole tweet, which should move Jaccard upward toward the target without changing the core model pipeline when checkpoints exist. The script still always produce `submission.csv` with correct columns and ordering.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import warnings
import random

import numpy as np
import pandas as pd

import torch
from torch import nn

from transformers import RobertaModel, RobertaConfig, RobertaTokenizer
from tqdm import tqdm  # avoid tqdm.notebook which can break in non-notebook runners

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def _exists(path: str) -> bool:
    try:
        return path is not None and os.path.exists(path)
    except Exception:
        return False


def _available_fold_paths():
    paths = []
    for fold in range(2, N):
        mp = f"{outdir}roberta_fold{fold+1}.pth"
        if _exists(mp):
            paths.append(mp)
    return paths


AVAILABLE_FOLD_PATHS = _available_fold_paths()

HF_TOKENIZER_NAME = "roberta-base"
HF_TOKENIZER = RobertaTokenizer.from_pretrained(
    HF_TOKENIZER_NAME, add_prefix_space=True
)

CLS_TOK = int(HF_TOKENIZER.cls_token_id)
PAD_TOK = int(HF_TOKENIZER.pad_token_id)
SEP_TOK = int(HF_TOKENIZER.sep_token_id)

print("Device:", device)
print("Using HF tokenizer (slow):", HF_TOKENIZER_NAME)
print("Special token ids:", {"CLS": CLS_TOK, "PAD": PAD_TOK, "SEP": SEP_TOK})
print("Available fold checkpoints:", len(AVAILABLE_FOLD_PATHS))




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
        self.hf_tok = (
            HF_TOKENIZER  # always use HF tokenizer to avoid tokenizers/protobuf issues
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

        enc = self.hf_tok.encode_plus(
            str(row.sentiment),
            tweet,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
        )
        ids = enc["input_ids"]

        offsets = [(0, 0)] * len(ids)
        try:
            backend = self.hf_tok.backend_tokenizer
            be = backend.encode(str(row.sentiment), tweet)
            offsets_be = list(be.offsets)
            if len(offsets_be) < self.max_len:
                offsets_be = offsets_be + [(0, 0)] * (self.max_len - len(offsets_be))
            offsets = offsets_be[: self.max_len]
        except Exception:
            offsets = [(0, 0)] * self.max_len

        ids = ids[: self.max_len]
        if len(ids) < self.max_len:
            ids = ids + [PAD_TOK] * (self.max_len - len(ids))

        ids = torch.tensor(ids, dtype=torch.long)
        masks = torch.where(
            ids != PAD_TOK,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )
        offsets = torch.tensor(offsets, dtype=torch.long)
        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(str(row.selected_text).lower().split())

        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (
            i
            for i, e in enumerate(tweet)
            if len(selected_text) > 1 and e == selected_text[1]
        ):
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
            if offset1 < offset2 and sum(char_targets[offset1:offset2]) > 0:
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
    )
    return loader




## === cell 3
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()

        if _exists(MODEL_CONFIG_PATH):
            config = RobertaConfig.from_pretrained(
                MODEL_CONFIG_PATH, output_hidden_states=True
            )
        else:
            config = RobertaConfig.from_pretrained(
                "roberta-base", output_hidden_states=True
            )

        if _exists(MODEL_PATH):
            try:
                state_dict = torch.load(MODEL_PATH, map_location="cpu")
                self.roberta = RobertaModel.from_pretrained(
                    "roberta-base", config=config, state_dict=state_dict
                )
            except Exception:
                self.roberta = RobertaModel.from_pretrained(MODEL_PATH, config=config)
        else:
            self.roberta = RobertaModel.from_pretrained("roberta-base", config=config)

        self.dropout = nn.Dropout(LINEAR_DROPOUT)
        self.fc = nn.Linear(config.hidden_size, 2)
        nn.init.normal_(self.fc.weight, std=0.02)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        output = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hs = output.hidden_states

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
        if o1 < o2:
            selected_text += text[o1:o2]
        if (ix + 1) < len(offsets) and int(offsets[ix][1]) < int(offsets[ix + 1][0]):
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return 0.0 if denom == 0 else float(len(c)) / denom


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
predictions = []
max_votes = []
models = []

print("loading models..")

if len(AVAILABLE_FOLD_PATHS) > 0:
    for model_path in AVAILABLE_FOLD_PATHS:
        model = TweetModel().to(device)
        state = torch.load(model_path, map_location=device)
        model.load_state_dict(state)
        model.eval()
        print(f"load {model_path}")
        models.append(model)
else:
    print(
        "WARNING: No fold checkpoints found in outdir; will generate sentiment-aware baseline predictions."
    )

if len(models) > 0:
    for data in tqdm(test_loader, total=len(test_loader)):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data["tweet"]
        offsets = data["offsets"].cpu().numpy()

        start_logits = []
        end_logits = []
        for model in models:
            with torch.no_grad():
                s_log, e_log = model(ids, masks)
                start_logits.append(torch.softmax(s_log, dim=1).cpu().numpy())
                end_logits.append(torch.softmax(e_log, dim=1).cpu().numpy())

        mean_start_logits = np.mean(start_logits, axis=0)
        mean_end_logits = np.mean(end_logits, axis=0)

        for i in range(ids.size(0)):
            prediction = []
            mean_start_pred = int(np.argmax(mean_start_logits[i]))
            mean_end_pred = int(np.argmax(mean_end_logits[i]))
            if mean_start_pred > mean_end_pred:
                mean_pred = tweet[i]
            else:
                mean_pred = get_selected_text(
                    tweet[i], mean_start_pred, mean_end_pred, offsets[i]
                )

            for k in range(len(models)):
                start = int(np.argmax(start_logits[k][i]))
                end = int(np.argmax(end_logits[k][i]))
                if start > end:
                    pred = tweet[i]
                else:
                    pred = get_selected_text(tweet[i], start, end, offsets[i])
                prediction.append(pred)

            votes = {mean_pred: 1}
            for p in prediction:
                votes[p] = votes.get(p, 0) + 1

            max_vote = max(votes.values())
            chosen = None
            for v in votes:
                if votes[v] == max_vote:
                    chosen = v
                    break

            predictions.append(chosen if chosen is not None else tweet[i])
            max_votes.append(max_vote)
else:
    POS_WORDS = {
        "good",
        "great",
        "love",
        "loved",
        "awesome",
        "amazing",
        "best",
        "nice",
        "happy",
        "wonderful",
        "fantastic",
        "excellent",
        "perfect",
        "thanks",
        "thank",
        "yay",
        "cool",
    }
    NEG_WORDS = {
        "bad",
        "worst",
        "hate",
        "hated",
        "awful",
        "terrible",
        "sad",
        "angry",
        "upset",
        "disappointed",
        "sucks",
        "suck",
        "poor",
        "annoying",
        "sorry",
        "pain",
        "mad",
    }

    for _, row in test_df.iterrows():
        raw = str(row.text)
        sent = str(row.sentiment).lower().strip()

        if sent == "neutral":
            pred = raw
        else:
            tokens = raw.split()
            tokens_l = [t.strip() for t in tokens]
            best = None
            if sent == "positive":
                for t in tokens_l:
                    w = "".join(
                        [ch for ch in t.lower() if ch.isalnum() or ch in ("'",)]
                    )
                    if w in POS_WORDS:
                        best = t
                        break
            elif sent == "negative":
                for t in tokens_l:
                    w = "".join(
                        [ch for ch in t.lower() if ch.isalnum() or ch in ("'",)]
                    )
                    if w in NEG_WORDS:
                        best = t
                        break

            pred = best if best is not None else raw

        predictions.append(pred)
        max_votes.append(0)

assert len(predictions) == len(test_df), (len(predictions), len(test_df))



## === cell 6
sub_df = pd.read_csv(submission_template)

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

if "textID" in test_df.columns and "textID" in sub_df.columns:
    if not sub_df["textID"].equals(test_df["textID"]):
        tmp = pd.DataFrame(
            {"textID": test_df["textID"].values, "selected_text": predictions}
        )
        sub_df = sub_df.drop(columns=["selected_text"]).merge(
            tmp, on="textID", how="left"
        )

sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)



## === cell 7
sub_df.head(batch_size)



## === cell 8
try:
    test_df_out = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
    test_df_out["selected_text"] = predictions
    test_df_out["selected_text"] = test_df_out["selected_text"].apply(
        lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
    )
    test_df_out["selected_text"] = test_df_out["selected_text"].apply(
        lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
    )
    test_df_out["selected_text"] = test_df_out["selected_text"].apply(
        lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
    )
    test_df_out["max_votes"] = max_votes
    test_df_out.to_csv("test_pred.csv", index=False)
    test_df_out.head(batch_size)
except Exception as e:
    print("Skipping test_pred.csv debug write due to:", repr(e))
