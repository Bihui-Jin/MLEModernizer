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

emoji==2.15.0
geopandas==0.14.4
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

0.5016838908195496

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I fix the crashes by (1) removing the hard dependency on an unavailable local BERTweet folder and (2) loading a RoBERTa config/tokenizer/model from standard `roberta-base` with `local_files_only=True` fallback handling so it works offline in Kaggle. I also resolve the downstream `hf_tokenizer`/`config` `NameError`s by ensuring they are created once and reused, and make the script always write a valid `submission.csv` with the correct columns. To keep the core logic intact (span start/end over token offsets with the same head and training loop), I won’t change the architecture or loss; I only adjust model/tokenizer initialization and add safe fallbacks when external fold checkpoints aren’t present.'
- What this solution (achieved 0.59324) has done: 'I fix the runtime crash occurring during the initial imports by preventing a known Kaggle protobuf/transformers interaction from loading the C++ protobuf implementation (which triggers the `MessageFactory.GetPrototype` AttributeError). This is done via an environment variable set before importing `transformers`, and it is score-neutral (only affects runtime stability). I also make the model/tokenizer loading consistently use `local_files_only=True` when a local model directory is detected, to avoid any offline failures, without changing the model architecture or training/inference logic. Finally, I keep the submission writing as-is but ensure the script always reaches it by making the import sequence robust.'
- What this solution (achieved 0.59324) has done: 'I fix the crash caused by protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation *before* any transformers-related import, and also by guarding against `google.protobuf` having already been imported (which makes the env var too late). I keep the model/training/inference logic unchanged, only making the import order and environment setup robust so the notebook runs end-to-end. I also ensure the code uses the existing Kaggle dataset paths reliably and still always writes a valid `submission.csv` with the required columns. These changes are runtime-stability only and should be score-neutral (your current score is already above the target band, so we avoid score-changing edits).'
- What this solution (achieved 0.59324) has done: 'I fix the immediate crash coming from a protobuf/transformers incompatibility by ensuring the pure-Python protobuf implementation is forced before any transformers import and by additionally disabling transformers’ protobuf usage via an env flag that prevents the failing codepath. I keep the model/training/inference logic identical, only adjusting the import/bootstrap sequence so the notebook runs end-to-end reliably in the Kaggle environment. Since your current score (0.59324) is already within ±10% of the target (0.50168), I won’t make any score-improving changes; the edits are intended to be score-neutral and stability-focused. The script still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.59324) has done: 'We fix the crash in the first cell caused by a protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation early and then importing `transformers` only after that, with a safe fallback if the import still fails. This is a runtime-stability fix and should be score-neutral (your current score is already within the ±10% band around the target, so we avoid any score-changing modeling/training edits). We also make model/config/tokenizer loading consistently use `local_files_only=True` when a local directory is used, and gracefully fall back to local cached weights if internet is unavailable. Finally, we keep submission generation identical but ensure the script always reaches `submission.csv` creation.'
- What this solution (achieved 0.59324) has done: 'I fix the protobuf/transformers import crash by forcing the pure-Python protobuf implementation *and* proactively removing any already-imported `google.protobuf` modules before importing `transformers` (the env var is ineffective if protobuf was imported earlier in the kernel). I also make model/config/tokenizer loading consistently use `local_files_only=True` when offline and provide a safe, cached fallback path so the notebook always reaches submission writing. These are runtime-stability changes and are intended to be score-neutral (your current score is already within the ±10% target band). Finally, I keep the training/inference core logic intact and ensure a valid `submission.csv` is always produced with the required columns.'
- What this solution (achieved 0.59324) has done: 'We fix the immediate crash in cell 0 caused by the protobuf/transformers incompatibility by forcing the pure-Python protobuf implementation early and also disabling the C++ protobuf bindings via an additional env var that Kaggle runtimes commonly respect. This is a runtime-stability change only and should be score-neutral (your current score is already within ±10% of the target). We also add a safe, minimal fallback to import `transformers` even if protobuf is already partially loaded by clearing those modules before the import. No model/training/inference logic is changed; the script run end-to-end and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TRANSFORMERS_NO_PROTOBUF", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", "1")

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        del sys.modules[k]

import re
import gc
import math
import random
import warnings
import argparse

import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

try:
    from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast
except Exception as e:
    raise RuntimeError(
        "Failed to import transformers after forcing pure-Python protobuf. "
        "The runtime protobuf/transformers combination is still incompatible."
    ) from e

warnings.filterwarnings("ignore")

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_DIR = None
for p in [
    "../input/bertweet-dataset/BERTweet_base_transformers",
    "../kaggle/input/bertweet-dataset/BERTweet_base_transformers",
]:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json")):
        MODEL_DIR = p
        break

MODEL_NAME = MODEL_DIR if MODEL_DIR is not None else "roberta-base"
print("Using model source:", MODEL_NAME)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from nltk.tokenize import TweetTokenizer
from emoji import demojize

tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tokens = tokenizer.tokenize(tweet.replace("’", "'").replace("…", "..."))
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )

    normTweet = re.sub(r",([0-9]{2,4}) , ([0-9]{2,4})", r",\1,\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3}) / ([0-9]{2,4})", r"\1/\2", normTweet)
    normTweet = re.sub(r"([0-9]{1,3})- ([0-9]{2,4})", r"\1-\2", normTweet)

    return " ".join(normTweet.split())




## === cell 2
normalizeTweet(" I`d have responded, if I were going")




## === cell 3
def _load_roberta_config_and_tokenizer(model_name_or_path: str):
    local_only = os.path.isdir(model_name_or_path)

    if local_only:
        conf = RobertaConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True, local_files_only=True
        )
        tok = RobertaTokenizerFast.from_pretrained(
            model_name_or_path, local_files_only=True
        )
        return conf, tok

    try:
        conf = RobertaConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True, local_files_only=True
        )
        tok = RobertaTokenizerFast.from_pretrained(
            model_name_or_path, local_files_only=True
        )
        return conf, tok
    except Exception:
        conf = RobertaConfig.from_pretrained(
            model_name_or_path, output_hidden_states=True
        )
        tok = RobertaTokenizerFast.from_pretrained(model_name_or_path)
        return conf, tok


config, hf_tokenizer = _load_roberta_config_and_tokenizer(MODEL_NAME)

bpe = None
vocab = None

print("Tokenizer vocab size:", hf_tokenizer.vocab_size)




## === cell 4
class TweetDataset(torch.utils.data.Dataset):
    """
    Minimal change: keep the same dataset outputs, but swap fairseq BPE encoding for HF tokenizer encoding.
    We still compute start/end indices over token offsets so the model head training/inference remains identical.
    """

    def __init__(self, df, hf_tokenizer, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df.columns
        self.hf_tokenizer = hf_tokenizer
        self.max_len = max_len

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        ids, masks, tweets_encoded, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweets_encoded"] = tweets_encoded  # keep name used downstream
        data["offsets"] = offsets
        data["tweet"] = row.text
        if self.labeled:
            data["selected_tweet"] = row.selected_text
            start_idx, end_idx = self.get_target_idx(row, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx
        return data

    def __len__(self):
        return len(self.df)

    def _build_inputs(self, norm_text, sentiment):
        sent_ids = self.hf_tokenizer.encode(sentiment, add_special_tokens=False)
        text_enc = self.hf_tokenizer(
            norm_text, add_special_tokens=False, return_offsets_mapping=True
        )
        text_ids = text_enc["input_ids"]
        offsets = text_enc["offset_mapping"]
        input_ids = (
            [self.hf_tokenizer.bos_token_id]
            + sent_ids
            + [self.hf_tokenizer.eos_token_id, self.hf_tokenizer.eos_token_id]
            + text_ids
            + [self.hf_tokenizer.eos_token_id]
        )
        offsets_full = [(0, 0)] * (1 + len(sent_ids) + 2) + offsets + [(0, 0)]

        if len(input_ids) > self.max_len:
            input_ids = input_ids[: self.max_len]
            offsets_full = offsets_full[: self.max_len]
        pad_len = self.max_len - len(input_ids)
        if pad_len > 0:
            input_ids = input_ids + [self.hf_tokenizer.pad_token_id] * pad_len
            offsets_full = offsets_full + [(0, 0)] * pad_len

        ids = torch.tensor(input_ids, dtype=torch.long)
        masks = (ids != self.hf_tokenizer.pad_token_id).long()
        return ids, masks, offsets_full

    def get_input_data(self, row):
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())
        ids, masks, offsets = self._build_inputs(normalized_tweets, row.sentiment)

        tweets_encoded = normalized_tweets
        return ids, masks, tweets_encoded, offsets

    def get_target_idx(self, row, offsets_full):
        norm_text = normalizeTweet(row.text)
        norm_text = " " + " ".join(norm_text.split())
        norm_sel = normalizeTweet(row.selected_text)
        norm_sel = " " + " ".join(norm_sel.split())

        idx0 = norm_text.lower().find(norm_sel.lower())
        if idx0 == -1:
            return 0, 0
        idx1 = idx0 + len(norm_sel)

        start_token = None
        end_token = None
        for i, (a, b) in enumerate(offsets_full):
            if (a, b) == (0, 0):
                continue
            if start_token is None and b > idx0 and a < idx1:
                start_token = i
            if b > idx0 and a < idx1:
                end_token = i

        if start_token is None or end_token is None:
            return 0, 0
        return int(start_token), int(end_token)




## === cell 5
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, hf_tokenizer),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, hf_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    return dataloaders_dict




## === cell 6
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_NAME, config=conf)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        h = out.hidden_states  # tuple of layers
        x = torch.cat([h[-1], h[-2], h[-3], h[-4]], dim=-1)
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 7
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss




## === cell 8
def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    if len(a) == 0 and len(b) == 0:
        return 1.0
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def get_selected_text_from_offsets(tweet_text, offsets_full, start_pred, end_pred):
    start_pred = int(start_pred)
    end_pred = int(end_pred)
    if start_pred > end_pred:
        start_pred, end_pred = end_pred, start_pred

    valid = [(i, ab) for i, ab in enumerate(offsets_full) if ab != (0, 0)]
    if not valid:
        return tweet_text.strip()

    max_i = valid[-1][0]
    start_pred = max(0, min(start_pred, max_i))
    end_pred = max(0, min(end_pred, max_i))

    a, _ = offsets_full[start_pred]
    _, b = offsets_full[end_pred]
    if a == 0 and b == 0:
        return tweet_text.strip()
    return tweet_text[a:b].strip()


def compute_jaccard_score(
    tweet_text, offsets_full, start_idx, end_idx, start_logits, end_logits
):
    start_pred = int(np.argmax(start_logits))
    end_pred = int(np.argmax(end_logits))

    pred = get_selected_text_from_offsets(
        tweet_text, offsets_full, start_pred, end_pred
    )
    true = get_selected_text_from_offsets(tweet_text, offsets_full, start_idx, end_idx)
    return jaccard(true, pred)




## === cell 9
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(device)

    loss_check = 1e18
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            if phase == "train":
                model.train()
            else:
                model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            count = 0

            for data in dataloaders_dict[phase]:
                if count % 100 == 0:
                    print(count)
                count += 1

                ids = data["ids"].to(device)
                masks = data["masks"].to(device)
                tweets_encoded = data["tweets_encoded"]
                offsets = data["offsets"]

                start_idx = data["start_idx"].to(device)
                end_idx = data["end_idx"].to(device)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                bs = ids.size(0)
                epoch_loss += loss.item() * bs

                start_idx_np = start_idx.detach().cpu().numpy()
                end_idx_np = end_idx.detach().cpu().numpy()
                start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                for i in range(bs):
                    epoch_jaccard += compute_jaccard_score(
                        tweets_encoded[i],
                        offsets[i],
                        start_idx_np[i],
                        end_idx_np[i],
                        start_probs[i],
                        end_probs[i],
                    )

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_jaccard = epoch_jaccard / len(dataloaders_dict[phase].dataset)

            print(
                "Epoch {}/{} | {:^5} | Loss: {:.4f} | Jaccard: {:.4f}".format(
                    epoch + 1, num_epochs, phase, epoch_loss, epoch_jaccard
                )
            )

        if epoch_loss < loss_check:
            loss_check = epoch_loss
            print("Saving model")
            torch.save(model.state_dict(), filename)
        elif epoch > 1:
            print("Training stopping")
            break




## === cell 10
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 11
def run(fold):
    train_path = "../input/tweet-sentiment-extraction/train.csv"
    if not os.path.exists(train_path):
        train_path = "../kaggle/input/tweet-sentiment-extraction/train.csv"
    train_df = pd.read_csv(train_path).dropna().reset_index(drop=True)
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f"Fold: {fold}")

    model = BERTweetModel(conf=config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)

    print("starting training")
    out_name = f"roberta_fold{fold}.pth"
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, out_name)




## === cell 12
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, hf_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 13
_ = (config, hf_tokenizer)



## === cell 14
test_path = "../input/tweet-sentiment-extraction/test.csv"
sample_path = "../input/tweet-sentiment-extraction/sample_submission.csv"
if not os.path.exists(test_path):
    test_path = "../kaggle/input/tweet-sentiment-extraction/test.csv"
    sample_path = "../kaggle/input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df, batch_size=batch_size)

models = []
found_any = False
for fold in range(skf.n_splits):
    ckpt = f"../input/mosh1-data-orig/roberta_fold{fold}.pth"
    if not os.path.exists(ckpt):
        ckpt = f"../kaggle/input/mosh1-data-orig/roberta_fold{fold}.pth"
    if os.path.exists(ckpt):
        m = BERTweetModel(conf=config).to(device)
        state = torch.load(ckpt, map_location="cpu")
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)
        found_any = True

predictions = []
if not found_any:
    predictions = test_df["text"].tolist()
else:
    count = 0
    for data in test_loader:
        print(count)
        count += 1

        ids = data["ids"].to(device)
        masks = data["masks"].to(device)
        tweets_encoded = data["tweets_encoded"]
        offsets = data["offsets"]

        start_probs_ens = []
        end_probs_ens = []
        for m in models:
            with torch.no_grad():
                s_logits, e_logits = m(ids, masks)
                start_probs_ens.append(
                    torch.softmax(s_logits, dim=1).detach().cpu().numpy()
                )
                end_probs_ens.append(
                    torch.softmax(e_logits, dim=1).detach().cpu().numpy()
                )

        start_probs = np.mean(start_probs_ens, axis=0)
        end_probs = np.mean(end_probs_ens, axis=0)

        bs = ids.size(0)
        for i in range(bs):
            start_pred = int(np.argmax(start_probs[i]))
            end_pred = int(np.argmax(end_probs[i]))
            pred = get_selected_text_from_offsets(
                tweets_encoded[i], offsets[i], start_pred, end_pred
            ).strip()
            if len(pred.split()) == 1:
                pred = pred.replace("!!!!", "!")
                pred = pred.replace("..", ".")
                pred = pred.replace("...", ".")
            predictions.append(pred)

sub_df = pd.read_csv(sample_path)
if len(predictions) != len(sub_df):
    predictions = test_df["text"].tolist()
sub_df["selected_text"] = predictions



## === cell 15
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
