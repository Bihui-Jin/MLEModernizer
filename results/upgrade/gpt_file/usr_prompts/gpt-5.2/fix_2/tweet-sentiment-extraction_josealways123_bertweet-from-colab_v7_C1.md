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

0.7062389850616455

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
import warnings
import random
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

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
base_path = "../input/bertweet-dataset"
local_model_dir = os.path.join(base_path, "BERTweet_base_transformers")

if os.path.isdir(local_model_dir):
    MODEL_SRC = local_model_dir
    local_only = True
else:
    MODEL_SRC = "vinai/bertweet-base"
    local_only = False

config = RobertaConfig.from_pretrained(
    MODEL_SRC, output_hidden_states=True, local_files_only=local_only
)
hf_tokenizer = RobertaTokenizerFast.from_pretrained(
    MODEL_SRC, local_files_only=local_only
)



## === cell 4


class HF_BPE:
    def __init__(self, tokenizer: RobertaTokenizerFast):
        self.tokenizer = tokenizer

    def encode(self, text: str) -> str:
        toks = self.tokenizer.tokenize(text, add_special_tokens=False)
        return " ".join(toks)


class HF_Vocab:
    def __init__(self, tokenizer: RobertaTokenizerFast):
        self.tokenizer = tokenizer
        self.unk_id = tokenizer.unk_token_id

    def encode_line(
        self, token_str: str, append_eos: bool = False, add_if_not_exist: bool = False
    ):
        toks = (
            token_str.split() if isinstance(token_str, str) and len(token_str) else []
        )
        ids = self.tokenizer.convert_tokens_to_ids(toks)
        ids = [int(i) if i is not None else int(self.unk_id) for i in ids]
        if append_eos:
            ids.append(int(self.tokenizer.eos_token_id))
        return torch.tensor(ids, dtype=torch.long)


bpe = HF_BPE(hf_tokenizer)
vocab = HF_Vocab(hf_tokenizer)




## === cell 5
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, bpe, vocab, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df.columns
        self.bpe = bpe
        self.vocab = vocab
        self.max_len = max_len

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]
        ids, masks, tweets_encoded = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweets_encoded"] = tweets_encoded
        data["tweet"] = row.text
        if self.labeled:
            data["selected_tweet"] = row.selected_text
            start_idx, end_idx = self.get_target_idx(row, tweets_encoded)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx
        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())
        tweets_encoded = self.bpe.encode(normalized_tweets)
        encoding_ids = (
            self.vocab.encode_line(
                tweets_encoded, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )
        sentiment_id = (
            self.vocab.encode_line(
                self.bpe.encode(row.sentiment), append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )

        ids = [0] + sentiment_id + [2, 2] + encoding_ids + [2]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
        else:
            ids = ids[: self.max_len]
        ids = torch.tensor(ids, dtype=torch.long)
        masks = torch.where(
            ids != 1,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )

        return ids, masks, tweets_encoded

    def get_target_idx(self, row, tweets_encoded):
        normalized_selected_tweets = normalizeTweet(row.selected_text)
        normalized_selected_tweets = " " + " ".join(normalized_selected_tweets.split())
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())

        len_st = len(normalized_selected_tweets) - 1
        idx0 = None
        idx1 = None
        for ind in (
            i
            for i, e in enumerate(normalized_tweets)
            if e == normalized_selected_tweets[1]
        ):
            if (
                " " + normalized_tweets[ind : ind + len_st]
                == normalized_selected_tweets
            ):
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        if idx0 is None and len(normalized_selected_tweets.split()) > 1:
            normalized_selected_tweets_1 = " " + " ".join(
                normalized_selected_tweets.split()[1:]
            )
            len_st_1 = len(normalized_selected_tweets_1) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_1[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_1]
                    == normalized_selected_tweets_1
                ):
                    idx0 = ind
                    idx1 = ind + len_st_1 - 1
                    break
        if idx0 is None and len(normalized_selected_tweets.split()) > 1:
            normalized_selected_tweets_2 = " " + " ".join(
                normalized_selected_tweets.split()[:-1]
            )
            len_st_2 = len(normalized_selected_tweets_2) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_2[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_2]
                    == normalized_selected_tweets_2
                ):
                    idx0 = ind
                    idx1 = ind + len_st_2 - 1
                    break
        if idx0 is None and len(normalized_selected_tweets.split()) > 1:
            normalized_selected_tweets_3 = " " + " ".join(
                normalized_selected_tweets_2.split()[:-1]
            )
            len_st_3 = len(normalized_selected_tweets_3) - 1
            for ind in (
                i
                for i, e in enumerate(normalized_tweets)
                if e == normalized_selected_tweets_3[1]
            ):
                if (
                    " " + normalized_tweets[ind : ind + len_st_3]
                    == normalized_selected_tweets_3
                ):
                    idx0 = ind
                    idx1 = ind + len_st_3 - 1
                    break

        sum_tot = -1
        flag = 0
        if idx0 is not None and idx1 is not None:
            for i, token in enumerate(tweets_encoded.split()):
                if "@@" not in token:
                    sum_tot += len(token) + 1
                else:
                    sum_tot += len(token) - 2
                if sum_tot >= idx0 and flag == 0:
                    start_idx = i
                    flag = 1
                if sum_tot >= idx1:
                    end_idx = i
                    break
        if idx0 is None or idx1 is None:
            start_idx = 0
            end_idx = 0
        return start_idx + 4, end_idx + 4




## === cell 6
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    dataloaders_dict = {"train": train_loader, "val": val_loader}
    return dataloaders_dict




## === cell 7
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained(
            MODEL_SRC, config=conf, local_files_only=local_only
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        )
        h = out.hidden_states
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
def get_selected_text(tweets_encoded, start_idx, end_idx):
    selected_text = ""
    for _, token in enumerate(tweets_encoded.split()[start_idx - 4 : end_idx - 3]):
        token = " " + token
        selected_text += token
    selected_text = re.sub("@@ ", "", selected_text)
    selected_text = re.sub("@@", "", selected_text)
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c) + 1e-12)


def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    length = len(tweets_encoded.split())
    if start_pred < 4:
        start_pred = 4
    if end_pred > 3 + length:
        end_pred = 3 + length
    if start_pred > end_pred:
        start_pred = 4
        end_pred = 3 + length
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    else:
        pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()
    return jaccard(true, pred)




## === cell 10
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(DEVICE)

    loss_check = 1000
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

                ids = data["ids"].to(DEVICE)
                masks = data["masks"].to(DEVICE)
                tweets_encoded = data["tweets_encoded"]

                start_idx = data["start_idx"].to(DEVICE)
                end_idx = data["end_idx"].to(DEVICE)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * ids.size(0)

                start_idx_np = start_idx.detach().cpu().numpy()
                end_idx_np = end_idx.detach().cpu().numpy()
                start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                for i in range(ids.size(0)):
                    jaccard_score = compute_jaccard_score(
                        tweets_encoded[i],
                        start_idx_np[i],
                        end_idx_np[i],
                        start_probs[i],
                        end_probs[i],
                    )
                    epoch_jaccard += jaccard_score

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
    train_df = (
        pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
        .dropna()
        .reset_index(drop=True)
    )
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f"Fold: {fold}")
    model = BERTweetModel(conf=config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = loss_fn
    dataloaders_dict = get_train_val_loaders(train_df, train_idx, val_idx, batch_size)
    print("starting training")
    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        f"roberta_fold{fold}.pth",
    )




## === cell 13
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 14
config = RobertaConfig.from_pretrained(
    MODEL_SRC, output_hidden_states=True, local_files_only=local_only
)




## === cell 15
def postprocessing(pred, tweet):
    pred_wo_spaces = "".join(pred.split())
    if len(pred_wo_spaces) == 0:
        return tweet
    length = len(pred_wo_spaces)
    flag = 0
    if len(tweet) > 0 and tweet[-1] == "@":
        return tweet
    else:
        for index, value in enumerate(tweet):
            count = 0
            letter = pred_wo_spaces[count]
            if value == letter:
                start_idx = index
                end_idx = index
                count += 1
                end_idx += 1
                while end_idx < len(tweet):
                    if tweet[end_idx] == " ":
                        end_idx += 1
                    elif tweet[end_idx] == "!" and pred_wo_spaces[count] != "!":
                        end_idx += 1
                    elif tweet[end_idx] == "." and pred_wo_spaces[count] != ".":
                        end_idx += 1
                    elif tweet[end_idx] == "*" and pred_wo_spaces[count] != "*":
                        end_idx += 1
                    elif tweet[end_idx] == "-" and pred_wo_spaces[count] != "-":
                        end_idx += 1
                    elif tweet[end_idx] == "?" and pred_wo_spaces[count] != "?":
                        end_idx += 1
                    elif count < length and tweet[end_idx] == pred_wo_spaces[count]:
                        end_idx += 1
                        count += 1
                    else:
                        break
                    if count == length:
                        flag = 1
                        break
                if flag == 1:
                    break
        if flag == 1:
            return tweet[start_idx:end_idx]
        elif "HTTPURL" in pred.split():
            return tweet
        elif "@USER" in pred.split():
            return tweet
        else:
            return tweet




## === cell 16
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df)
predictions = []
models = []

for fold in range(skf.n_splits):
    model = BERTweetModel(conf=config).to(DEVICE)
    weight_path = f"../input/mosh1-data-orig/roberta_fold{fold}.pth"
    if os.path.isfile(weight_path):
        state = torch.load(weight_path, map_location="cpu")
        model.load_state_dict(state)
        model.eval()
        models.append(model)

if len(models) == 0:
    model = BERTweetModel(conf=config).to(DEVICE)
    model.eval()
    models = [model]

count = 0
for data in test_loader:
    print(count)
    count += 1
    ids = data["ids"].to(DEVICE)
    masks = data["masks"].to(DEVICE)
    tweets_encoded = data["tweets_encoded"]
    tweet = data["tweet"]

    start_logits = []
    end_logits = []
    for model in models:
        with torch.no_grad():
            out_start, out_end = model(ids, masks)
            start_logits.append(torch.softmax(out_start, dim=1).detach().cpu().numpy())
            end_logits.append(torch.softmax(out_end, dim=1).detach().cpu().numpy())

    start_logits = np.mean(start_logits, axis=0)
    end_logits = np.mean(end_logits, axis=0)

    for i in range(ids.size(0)):
        start_pred = int(np.argmax(start_logits[i]))
        end_pred = int(np.argmax(end_logits[i]))
        length = len(tweets_encoded[i].split())
        if start_pred < 4:
            start_pred = 4
        if end_pred > 3 + length:
            end_pred = 3 + length
        if start_pred > end_pred:
            start_pred = 4
            end_pred = 3 + length
            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()
        else:
            pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()

        pred = postprocessing(pred, tweet[i].strip())
        predictions.append(pred)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4017775694.py in <cell line: 0>()
     36     for model in models:
     37         with torch.no_grad():
---> 38             out_start, out_end = model(ids, masks)
     39             start_logits.append(torch.softmax(out_start, dim=1).detach().cpu().numpy())
     40             end_logits.append(torch.softmax(out_end, dim=1).detach().cpu().numpy())

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/557214434.py in forward(self, input_ids, attention_mask)
     13     def forward(self, input_ids, attention_mask):
     14         # transformers>=4 returns BaseModelOutputWithPoolingAndCrossAttentions when return_dict=True (default)
---> 15         out = self.roberta(
     16             input_ids=input_ids, attention_mask=attention_mask, return_dict=True
     17         )

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/modeling_roberta.py in forward(self, input_ids, attention_mask, token_type_ids, position_ids, head_mask, inputs_embeds, encoder_hidden_states, encoder_attention_mask, past_key_values, use_cache, output_attentions, output_hidden_states, return_dict)
    822                 )
    823             else:
--> 824                 extended_attention_mask = _prepare_4d_attention_mask_for_sdpa(
    825                     attention_mask, embedding_output.dtype, tgt_len=seq_length
    826                 )

/usr/local/lib/python3.11/dist-packages/transformers/modeling_attn_mask_utils.py in _prepare_4d_attention_mask_for_sdpa(mask, dtype, tgt_len)
    452 
    453     # torch.jit.trace, symbolic_trace and torchdynamo with fullgraph=True are unable to capture data-dependent controlflows.
--> 454     if not is_tracing and torch.all(mask == 1):
    455         return None
    456     else:

RuntimeError: CUDA error: device-side assert triggered
CUDA kernel errors might be asynchronously reported at some other API call, so the stacktrace below might be incorrect.
For debugging consider passing CUDA_LAUNCH_BLOCKING=1
Compile with `TORCH_USE_CUDA_DSA` to enable device-side assertions.


## === cell 17
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
if len(predictions) != len(sub_df):
    predictions = predictions[: len(sub_df)]
sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if len(x.split()) == 1 else x
)
sub_df.to_csv("submission.csv", index=False)
sub_df.head()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2981566631.py in <cell line: 0>()
      3     # Safety: align lengths if something odd happened (shouldn't).
      4     predictions = predictions[: len(sub_df)]
----> 5 sub_df["selected_text"] = predictions
      6 sub_df["selected_text"] = sub_df["selected_text"].apply(
      7     lambda x: x.replace("!!!!", "!") if len(x.split()) == 1 else x

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (416) does not match length of index (2749)
