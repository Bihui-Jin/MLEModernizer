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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
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

from transformers import RobertaModel, RobertaConfig, RobertaTokenizerFast

warnings.filterwarnings("ignore")

seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if os.path.isdir("../input/bertweet-dataset/BERTweet_base_transformers"):
    base_path = "../input/bertweet-dataset"
elif os.path.isdir("../kaggle/input/bertweet-dataset/BERTweet_base_transformers"):
    base_path = "../kaggle/input/bertweet-dataset"
else:
    base_path = "../input/bertweet-dataset"

MODEL_DIR = os.path.join(base_path, "BERTweet_base_transformers")
if not os.path.isdir(MODEL_DIR):
    raise FileNotFoundError(f"Expected local BERTweet folder not found: {MODEL_DIR}")



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
config = RobertaConfig.from_pretrained(MODEL_DIR, output_hidden_states=True)

hf_tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1385946811.py in <cell line: 0>()
      1 # Fix: Transformers expects a directory (with config.json), not a direct config.json file path.
      2 # Keep output_hidden_states=True as in original logic.
----> 3 config = RobertaConfig.from_pretrained(MODEL_DIR, output_hidden_states=True)
      4 
      5 # Fix: replace missing fairseq+fastBPE pipeline with a local HF tokenizer from the same folder.

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/bertweet-dataset/BERTweet_base_transformers'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertweet-dataset/BERTweet_base_transformers' is the correct path to a directory containing a config.json file

## === cell 4
bpe = None
vocab = None




## === cell 5
class TweetDataset(torch.utils.data.Dataset):
    """
    Minimal change: keep the same dataset outputs, but swap fairseq BPE encoding for HF tokenizer encoding.
    We still compute start/end indices over token positions so the model head training/inference remains identical.
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




## === cell 6
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




## === cell 7
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(MODEL_DIR, config=conf)
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




## === cell 8
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    total_loss = start_loss + end_loss
    return total_loss




## === cell 9
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




## === cell 10
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




## === cell 11
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 12
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




## === cell 13
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




## === cell 14
config = RobertaConfig.from_pretrained(MODEL_DIR, output_hidden_states=True)
hf_tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

HFValidationError                         Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    521         # Now we try to recover if we can find all files correctly in the cache
--> 522         resolved_files = [
    523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in <listcomp>(.0)
    522         resolved_files = [
--> 523             _get_cache_file_to_return(path_or_repo_id, filename, cache_dir, revision) for filename in full_filenames
    524         ]

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in _get_cache_file_to_return(path_or_repo_id, full_filename, cache_dir, revision)
    139     # We try to see if we have a cached version (not up to date):
--> 140     resolved_file = try_to_load_from_cache(path_or_repo_id, full_filename, cache_dir=cache_dir, revision=revision)
    141     if resolved_file is not None and resolved_file != _CACHED_NO_EXIST:

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    105             if arg_name in ["repo_id", "from_id", "to_id"]:
--> 106                 validate_repo_id(arg_value)
    107 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in validate_repo_id(repo_id)
    153     if repo_id.count("/") > 1:
--> 154         raise HFValidationError(
    155             "Repo id must be in the form 'repo_name' or 'namespace/repo_name':"

HFValidationError: Repo id must be in the form 'repo_name' or 'namespace/repo_name': '../input/bertweet-dataset/BERTweet_base_transformers'. Use `repo_type` argument if needed.

During handling of the above exception, another exception occurred:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/1119169300.py in <cell line: 0>()
      1 # Keep config loading once (already loaded in cell 4); this cell existed and errored previously due to file path.
      2 # Re-assert config/tokenizer are available.
----> 3 config = RobertaConfig.from_pretrained(MODEL_DIR, output_hidden_states=True)
      4 hf_tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_DIR)
      5 

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, **kwargs)
    566         cls._set_token_in_kwargs(kwargs, token)
    567 
--> 568         config_dict, kwargs = cls.get_config_dict(pretrained_model_name_or_path, **kwargs)
    569         if cls.base_config_key and cls.base_config_key in config_dict:
    570             config_dict = config_dict[cls.base_config_key]

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    606         original_kwargs = copy.deepcopy(kwargs)
    607         # Get config dict associated with the base config file
--> 608         config_dict, kwargs = cls._get_config_dict(pretrained_model_name_or_path, **kwargs)
    609         if config_dict is None:
    610             return {}, kwargs

/usr/local/lib/python3.11/dist-packages/transformers/configuration_utils.py in _get_config_dict(cls, pretrained_model_name_or_path, **kwargs)
    688             except Exception:
    689                 # For any other exception, we throw a generic error.
--> 690                 raise OSError(
    691                     f"Can't load the configuration of '{pretrained_model_name_or_path}'. If you were trying to load it"
    692                     " from 'https://huggingface.co/models', make sure you don't have a local directory with the same"

OSError: Can't load the configuration of '../input/bertweet-dataset/BERTweet_base_transformers'. If you were trying to load it from 'https://huggingface.co/models', make sure you don't have a local directory with the same name. Otherwise, make sure '../input/bertweet-dataset/BERTweet_base_transformers' is the correct path to a directory containing a config.json file

## === cell 15
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



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2466774571.py in <cell line: 0>()
     10 test_df["text"] = test_df["text"].astype(str)
     11 
---> 12 test_loader = get_test_loader(test_df, batch_size=batch_size)
     13 
     14 # Attempt to load external pretrained fold models if available in the environment

/tmp/ipykernel_55/259429400.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2     loader = torch.utils.data.DataLoader(
----> 3         TweetDataset(df, hf_tokenizer),
      4         batch_size=batch_size,
      5         shuffle=False,

NameError: name 'hf_tokenizer' is not defined

## === cell 16
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2105150184.py in <cell line: 0>()
      1 # Write valid Kaggle submission
----> 2 sub_df.to_csv("submission.csv", index=False)
      3 print(sub_df.head())
      4 print("Wrote submission.csv with shape:", sub_df.shape)

NameError: name 'sub_df' is not defined
