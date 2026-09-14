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

0.7140237092971802

# 6. Current score

0.59329

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fixed the tokenizer/model loading path so that it falls back to the public `roberta‑base` model when the local directory does not contain a valid checkpoint, and I adjusted the code to use this resolved path throughout. This eliminates the HFValidationError, ensures that the dataset loader runs, and guarantees that the `predictions` list is always defined so the submission file is written correctly. No core modeling logic was changed.'
- What this solution (achieved 0.59324) has done: 'The fix forces the model to always use the public `roberta‑base` checkpoint (avoiding the protobuf error from a broken local directory) and adds safe truncation of token sequences to the defined ``MAX_LEN`` (so long tweets no longer overflow). The maximum length is also increased to 128 to capture more context, which should raise the Jaccard score toward the target while preserving the original architecture and training logic.'
- What this solution (achieved 0.59324) has done: 'The changes fix the prediction logic: all saved folds are now used (range 0‑N‑1) and raw logits are averaged before applying soft‑max, which yields more accurate start/end positions and moves the Jaccard score toward the target while keeping the original model architecture untouched. Additionally, the code now safely handles the case of missing model files.'
- What this solution (achieved 0.59357) has done: 'The fix adds a protobuf environment flag before importing `transformers` to stop the `MessageFactory` import error, and introduces a simple rule‑based fallback that selects the sentiment word (or the whole tweet for neutral sentiment) when no trained models are available. This ensures a valid `.csv` submission is always written and raises the Jaccard score toward the target while keeping the original architecture unchanged.'
- What this solution (achieved 0.59357) has done: 'Implemented a minimal fix to actually load the trained model checkpoints by correcting the checkpoint directory path. The code now points to the typical working directory (`/kaggle/working/`) where model files are stored, and falls back to the original path if needed. This enables the ensemble of fine‑tuned models to be used during inference, which should raise the Jaccard score toward the target while preserving all original logic.'
- What this solution (achieved 0.59357) has done: 'Implemented robust imports, avoided transformer crashes, added an average‑length‑based heuristic to improve fallback predictions, and restructured the inference flow so model loading is optional. The script now reliably writes a `submission.csv` and nudges the Jaccard score toward the target.'
- What this solution (achieved 0.59329) has done: 'I keep the existing pipeline intact and only improve the rule‑based fallback used when no transformer model can be loaded. The new heuristic expands the selected span around the sentiment word to match the average selected‑text length per sentiment, working on word tokens instead of raw characters, which yields selections that better align with the ground‑truth spans and should raise the Jaccard score toward the target. No other logic is changed.'
- What this solution (achieved 0.59329) has done: 'I added robust handling for the Transformer model so that any protobuf‑related import or loading error no longer crashes the notebook – the code now falls back to a dummy model that returns zero logits. I also improved the rule‑based fallback: it now expands the selected span on both sides until the average selected‑text length for the sentiment is reached, giving more realistic predictions when no model checkpoints are available. These changes keep the original architecture untouched, guarantee a valid `submission.csv` file, and nudge the Jaccard score closer to the target.'
- What this solution (achieved 0.59329) has done: 'Implemented robust fallback handling and improved rule‑based selector.  
- Wrapped the Transformers import in a safe try/except; on failure the pipeline skips model loading entirely, avoiding the protobuf crash.  
- Replaced the heuristic with a more accurate version: it locates the first and last tokens containing the sentiment word, then expands outward token‑wise until reaching the average selected‑text length for that sentiment (or tweet bounds). This yields selections that better match the ground‑truth spans, nudging the Jaccard score toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.59329) has done: 'The fix updates the rule‑based fallback: it now centers a window of the average selected‑text length around the first token that contains the sentiment word, handling tweet boundaries gracefully. This improves the selected‑text predictions when no model checkpoints are available, moving the Jaccard score closer to the target while keeping the overall pipeline untouched.'
- What this solution (achieved 0.59328) has done: 'I updated the fallback heuristic to select a more natural span: it now finds the exact sentiment word in the tweet and returns the substring from that word up to the next punctuation mark (or the tweet end), which better matches the expected selected text and should raise the Jaccard score toward the target. I also added a small safety guard around the Transformers import to ensure the script continues even if the library cannot be loaded.'
- What this solution (achieved 0.59329) has done: 'Implemented robust handling to avoid transformer import crashes and enhanced the rule‑based fallback. The script now forces `TRANSFORMERS_AVAILABLE` to **False** when any import error occurs, bypassing model loading and preventing runtime failures. The heuristic selector was upgraded: it still returns the whole tweet for neutral sentiment, but for other sentiments it finds the sentiment word, expands to the next punctuation, and then trims or pads the span to match the average selected‑text length for that sentiment (computed from the training data). This modest improvement is expected to raise the Jaccard score toward the target while keeping the core architecture untouched.'
- What this solution (achieved 0.59329) has done: 'Implemented an enhanced rule‑based fallback that selects a token window around the sentiment word sized to the average selected‑text length for that sentiment. This replaces the previous heuristic, improving alignment with expected spans and moving the Jaccard score nearer the target while keeping all core modeling logic untouched. Added a small safeguard to ensure the transformers import flag is correctly set even if import fails.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import warnings
import random
import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold
from tqdm.notebook import tqdm
import re
import string
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import CountVectorizer

warnings.filterwarnings("ignore")


def seed_everything(seed_value):
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

MODEL_PATH = "roberta-base"
outdir_primary = "/kaggle/working/"
outdir_fallback = "/kaggle/input/roberta714kernel/"
outdir = outdir_primary if os.path.isdir(outdir_primary) else outdir_fallback

test_file = "/kaggle/input/tweet-sentiment-extraction/test.csv"
submission_template = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
train_file = "/kaggle/input/tweet-sentiment-extraction/train.csv"

MAX_LEN = 128
LINEAR_DROPOUT = 0.2
CLS_TOK = 0
PAD_TOK = 1
SEP_TOK = 2

try:
    from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast

    TRANSFORMERS_AVAILABLE = True
except Exception as e:
    print("Transformers import failed (fallback to rule‑based):", e)
    TRANSFORMERS_AVAILABLE = False

train_df = pd.read_csv(train_file)
train_df["selected_text"] = train_df["selected_text"].astype(str)
train_df["sel_len"] = train_df["selected_text"].apply(lambda x: len(str(x).split()))
avg_len_per_sentiment = (
    train_df.groupby("sentiment")["sel_len"].mean().round().astype(int).to_dict()
)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=MAX_LEN):
        if not TRANSFORMERS_AVAILABLE:
            raise RuntimeError("Transformers not available – cannot create dataset.")
        self.df = df
        self.max_len = max_len
        self.labeled = "selected_text" in df.columns
        self.tokenizer = RobertaTokenizerFast.from_pretrained(
            MODEL_PATH, add_prefix_space=True, use_fast=True
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
        tweet = " " + " ".join(row.text.lower().split())
        tweet_enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
        )
        tweet_ids = tweet_enc["input_ids"]
        tweet_offsets = tweet_enc["offset_mapping"]

        sentiment_enc = self.tokenizer.encode(row.sentiment, add_special_tokens=False)
        ids = [CLS_TOK] + sentiment_enc + [SEP_TOK, SEP_TOK] + tweet_ids + [SEP_TOK]
        offsets = [(0, 0)] * 4 + list(tweet_offsets) + [(0, 0)]

        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [PAD_TOK] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = torch.where(
            ids != PAD_TOK,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )
        offsets = torch.tensor(offsets, dtype=torch.long)
        return ids, masks, tweet, offsets

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())

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
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx




## === cell 2
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=NUM_WORKERS
    )
    return loader




## === cell 3
if TRANSFORMERS_AVAILABLE:

    class TweetModel(nn.Module):
        def __init__(self):
            super(TweetModel, self).__init__()
            self.valid = True
            try:
                config = RobertaConfig.from_pretrained(
                    MODEL_PATH, output_hidden_states=True
                )
                self.roberta = RobertaModel.from_pretrained(MODEL_PATH, config=config)
                self.dropout = nn.Dropout(LINEAR_DROPOUT)
                self.fc = nn.Linear(config.hidden_size, 2)
                nn.init.normal_(self.fc.weight, std=0.02)
                nn.init.normal_(self.fc.bias, std=0.0)
            except Exception as e:
                print("Failed to load Roberta model:", e)
                self.valid = False
                self.roberta = None
                self.dropout = nn.Identity()
                self.fc = nn.Linear(768, 2)

        def forward(self, input_ids, attention_mask):
            if not self.valid:
                batch_sz = input_ids.shape[0]
                return (
                    torch.zeros(batch_sz, device=input_ids.device),
                    torch.zeros(batch_sz, device=input_ids.device),
                )
            _, _, hs = self.roberta(input_ids, attention_mask)
            x = torch.stack([hs[-1], hs[-2], hs[-3]])
            x = torch.mean(x, 0)
            x = self.dropout(x)
            x = self.fc(x)
            start_logits, end_logits = x.split(1, dim=-1)
            start_logits = start_logits.squeeze(-1)
            end_logits = end_logits.squeeze(-1)
            return start_logits, end_logits

else:

    class TweetModel:
        pass  # placeholder when transformers are unavailable




## === cell 4
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


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
def heuristic_selected_text(row):
    """
    Improved rule‑based fallback:
    • Neutral sentiment → whole tweet.
    • Locate the token that exactly matches the sentiment word (punctuation stripped).
    • Return a contiguous window of tokens centered on that word whose length
      matches the average selected‑text length for the sentiment (computed from training data).
    """
    sentiment = row["sentiment"].lower()
    tweet = row["text"]
    if sentiment == "neutral":
        return tweet

    tokens = tweet.split()
    target_idx = None
    for i, tok in enumerate(tokens):
        cleaned = tok.lower().strip(string.punctuation)
        if cleaned == sentiment:
            target_idx = i
            break

    if target_idx is None:
        return tweet

    target_len = avg_len_per_sentiment.get(sentiment, 3)
    target_len = min(target_len, len(tokens))

    half = target_len // 2
    start = max(0, target_idx - half)
    end = start + target_len
    if end > len(tokens):
        end = len(tokens)
        start = max(0, end - target_len)

    selected = " ".join(tokens[start:end])
    return selected




## === cell 6
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(test_file)
test_df["text"] = test_df["text"].astype(str)

predictions = []
models = []

print("loading models..")
if TRANSFORMERS_AVAILABLE:
    for fold in range(skf.n_splits):
        model = TweetModel()
        model.to(device)
        model_path = f"{outdir}roberta_fold{fold+1}.pth"
        if os.path.exists(model_path):
            try:
                model.load_state_dict(torch.load(model_path, map_location=device))
                model.eval()
                print(f"loaded {model_path}")
                models.append(model)
                continue
            except Exception as e:
                print(f"Failed to load {model_path}: {e}")

        fallback_path = f"{outdir_fallback}roberta_fold{fold+1}.pth"
        if os.path.exists(fallback_path):
            try:
                model.load_state_dict(torch.load(fallback_path, map_location=device))
                model.eval()
                print(f"loaded {fallback_path}")
                models.append(model)
                continue
            except Exception as e:
                print(f"Failed to load {fallback_path}: {e}")

        print(f"Warning: model file {model_path} not found – skipping this fold.")
else:
    print("Transformers unavailable – skipping model loading.")

if models:
    test_loader = get_test_loader(test_df)
    for data in tqdm(test_loader, desc="Predicting"):
        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweet = data["tweet"]
        offsets = data["offsets"].numpy()

        start_logits_fold = []
        end_logits_fold = []
        for model in models:
            with torch.no_grad():
                start_logits, end_logits = model(ids, masks)
                start_logits_fold.append(start_logits.cpu().numpy())
                end_logits_fold.append(end_logits.cpu().numpy())

        start_logits_mean = np.mean(start_logits_fold, axis=0)
        end_logits_mean = np.mean(end_logits_fold, axis=0)

        start_probs = torch.nn.functional.softmax(
            torch.from_numpy(start_logits_mean), dim=1
        ).numpy()
        end_probs = torch.nn.functional.softmax(
            torch.from_numpy(end_logits_mean), dim=1
        ).numpy()

        for i in range(len(ids)):
            start_pred = np.argmax(start_probs[i])
            end_pred = np.argmax(end_probs[i])
            if start_pred > end_pred:
                pred = tweet[i]
            else:
                pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
            predictions.append(pred)
else:
    print("No model checkpoints found – using rule‑based fallback.")
    predictions = [heuristic_selected_text(row) for _, row in test_df.iterrows()]

sub_df = pd.read_csv(submission_template)
if len(predictions) != len(sub_df):
    print(
        f"Length mismatch: predictions {len(predictions)} vs submission rows {len(sub_df)}"
    )
    predictions = predictions[: len(sub_df)]
    if len(predictions) < len(sub_df):
        predictions += [""] * (len(sub_df) - len(predictions))

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
print("submission.csv written")
