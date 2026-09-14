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

0.7090124487876892

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import argparse
import warnings
import random
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



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re

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
base_path = "../input/bertweet-dataset"
bertweet_dir = os.path.join(base_path, "BERTweet_base_transformers")
config = RobertaConfig.from_pretrained(bertweet_dir, output_hidden_states=True)
bertweet_tokenizer = RobertaTokenizerFast.from_pretrained(
    bertweet_dir, add_prefix_space=True
)




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
/tmp/ipykernel_55/157992333.py in <cell line: 0>()
      2 base_path = "../input/bertweet-dataset"
      3 bertweet_dir = os.path.join(base_path, "BERTweet_base_transformers")
----> 4 config = RobertaConfig.from_pretrained(bertweet_dir, output_hidden_states=True)
      5 bertweet_tokenizer = RobertaTokenizerFast.from_pretrained(
      6     bertweet_dir, add_prefix_space=True

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
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, tokenizer, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df.columns
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text = row["text"]
        sentiment = row["sentiment"]

        norm_text = " " + " ".join(normalizeTweet(text).split())
        norm_sent = normalizeTweet(sentiment)

        tweet_ids = self.tokenizer.encode(norm_text, add_special_tokens=False)
        sent_ids = self.tokenizer.encode(norm_sent, add_special_tokens=False)

        ids = (
            [self.tokenizer.bos_token_id]
            + sent_ids
            + [self.tokenizer.eos_token_id] * 2
            + tweet_ids
            + [self.tokenizer.eos_token_id]
        )

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids = ids + [self.tokenizer.pad_token_id] * pad_len
        else:
            ids = ids[: self.max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != self.tokenizer.pad_token_id).long()

        item = {
            "ids": ids,
            "masks": masks,
            "tweets_encoded": self.tokenizer.decode(
                tweet_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
            ),
            "tweet": text,
            "sentiment": sentiment,
        }
        if self.labeled:
            item["selected_tweet"] = row["selected_text"]
            start_idx, end_idx = self.get_target_idx(text, row["selected_text"])
            item["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
            item["end_idx"] = torch.tensor(end_idx, dtype=torch.long)
        return item

    def get_target_idx(self, text, selected):
        norm_text = " " + " ".join(normalizeTweet(text).split())
        norm_sel = " " + " ".join(normalizeTweet(selected).split())

        encoding = self.tokenizer(
            norm_text, return_offsets_mapping=True, add_special_tokens=False
        )
        offsets = encoding.offset_mapping

        start_char = norm_text.find(norm_sel.strip())
        if start_char == -1:
            return 4, 4
        end_char = start_char + len(norm_sel.strip())

        start_token = None
        end_token = None
        for i, (s, e) in enumerate(offsets):
            if start_token is None and s <= start_char < e:
                start_token = i
            if s < end_char <= e:
                end_token = i
                break
        if start_token is None:
            start_token = 0
        if end_token is None:
            end_token = len(offsets) - 1
        return start_token + 4, end_token + 4




## === cell 5
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, bertweet_tokenizer),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, bertweet_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )

    return {"train": train_loader, "val": val_loader}




## === cell 6
import torch.nn as nn


class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            os.path.join(base_path, "BERTweet_base_transformers", "model.bin"),
            config=conf,
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        a, b, h = self.roberta(input_ids, attention_mask)
        x = torch.cat([h[-1], h[-2], h[-3], h[-4]], dim=-1)
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 7
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce = nn.CrossEntropyLoss()
    start_loss = ce(start_logits, start_positions)
    end_loss = ce(end_logits, end_positions)
    return start_loss + end_loss




## === cell 8
def get_selected_text(tweets_encoded, start_idx, end_idx):
    selected = ""
    for token in tweets_encoded.split()[start_idx - 4 : end_idx - 3]:
        selected += " " + token
    selected = selected.replace("@@ ", "").replace("@@", "")
    return selected.strip()


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    length = len(tweets_encoded.split())
    start_pred = max(start_pred, 4)
    end_pred = min(end_pred, 3 + length)
    if start_pred > end_pred:
        start_pred, end_pred = 4, 3 + length
    pred = get_selected_text(tweets_encoded, start_pred, end_pred)
    true = get_selected_text(tweets_encoded, start_idx, end_idx)
    return jaccard(true, pred)




## === cell 9
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    if torch.cuda.is_available():
        model.cuda()
    best_loss = float("inf")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            for batch in dataloaders_dict[phase]:
                ids = batch["ids"]
                masks = batch["masks"]
                tweets_enc = batch["tweets_encoded"]
                start_idx = batch["start_idx"]
                end_idx = batch["end_idx"]
                if torch.cuda.is_available():
                    ids = ids.cuda()
                    masks = masks.cuda()
                    start_idx = start_idx.cuda()
                    end_idx = end_idx.cuda()
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                epoch_loss += loss.item() * ids.size(0)
                start_idx_cpu = start_idx.cpu().numpy()
                end_idx_cpu = end_idx.cpu().numpy()
                start_soft = torch.softmax(start_logits, dim=1).cpu().numpy()
                end_soft = torch.softmax(end_logits, dim=1).cpu().numpy()
                for i in range(len(ids)):
                    epoch_jaccard += compute_jaccard_score(
                        tweets_enc[i],
                        start_idx_cpu[i],
                        end_idx_cpu[i],
                        start_soft[i],
                        end_soft[i],
                    )
            epoch_loss /= len(dataloaders_dict[phase].dataset)
            epoch_jaccard /= len(dataloaders_dict[phase].dataset)
            print(
                f"Epoch {epoch+1}/{num_epochs} | {phase.upper():5} | Loss: {epoch_loss:.4f} | Jaccard: {epoch_jaccard:.4f}"
            )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), filename)
        else:
            print("Early stopping")
            break




## === cell 10
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 11
def run(fold):
    train_df = (
        pd.read_csv("tweet-sentiment-extraction/train.csv")
        .dropna()
        .reset_index(drop=True)
    )
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)
    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f"Fold: {fold}")
    model = BERTweetModel(conf=config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    dataloaders = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
    train_model(
        model, dataloaders, loss_fn, optimizer, num_epochs, f"roberta_fold{fold}.pth"
    )




## === cell 12
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, bertweet_tokenizer),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return loader




## === cell 13
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df)
predictions = []
models = []
for fold in range(skf.n_splits):
    model = BERTweetModel(conf=config)
    if torch.cuda.is_available():
        model.cuda()
    ckpt_path = f"roberta_fold{fold}.pth"
    if os.path.exists(ckpt_path):
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    else:
        ckpt_path = f"../input/mosh1-data-orig/roberta_fold{fold}.pth"
        model.load_state_dict(torch.load(ckpt_path, map_location="cpu"))
    model.eval()
    models.append(model)

for batch in test_loader:
    ids = batch["ids"]
    masks = batch["masks"]
    tweets_enc = batch["tweets_encoded"]
    raw_tweet = batch["tweet"]
    if torch.cuda.is_available():
        ids = ids.cuda()
        masks = masks.cuda()
    start_logits = []
    end_logits = []
    for m in models:
        with torch.no_grad():
            s, e = m(ids, masks)
            start_logits.append(torch.softmax(s, dim=1).cpu().numpy())
            end_logits.append(torch.softmax(e, dim=1).cpu().numpy())
    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)
    for i in range(len(ids)):
        length = len(tweets_enc[i].split())
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        start_pred = max(start_pred, 4)
        end_pred = min(end_pred, 3 + length)
        if start_pred > end_pred:
            start_pred, end_pred = 4, 3 + length
        pred = get_selected_text(tweets_enc[i], start_pred, end_pred)
        pred = pred.strip()
        predictions.append(pred)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1630638891.py in <cell line: 0>()
      2 test_df["text"] = test_df["text"].astype(str)
      3 
----> 4 test_loader = get_test_loader(test_df)
      5 predictions = []
      6 models = []

/tmp/ipykernel_55/13238142.py in get_test_loader(df, batch_size)
      1 def get_test_loader(df, batch_size=32):
      2     loader = torch.utils.data.DataLoader(
----> 3         TweetDataset(df, bertweet_tokenizer),
      4         batch_size=batch_size,
      5         shuffle=False,

NameError: name 'bertweet_tokenizer' is not defined

## === cell 14
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sub_df["selected_text"] = predictions
sub_df.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2688697133.py in <cell line: 0>()
      1 sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
----> 2 sub_df["selected_text"] = predictions
      3 sub_df.to_csv("submission.csv", index=False)
      4 print("Submission saved to submission.csv")

NameError: name 'predictions' is not defined
