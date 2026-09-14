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
import os
import re
import random
import warnings

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(seed)

DATA_DIR = "/kaggle/input/tweet-sentiment-extraction"
WORK_DIR = "/kaggle/working"



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
def _find_any_local_roberta_dir():
    """
    FIX: Original code required a separately added roberta-base dataset.
    In Kaggle/no-internet we should first try local HF cache; if not present,
    also try common Kaggle input locations. We keep architecture unchanged (RobertaModel).
    """
    candidates = [
        "/kaggle/input/roberta-base",
        "/kaggle/input/bert-roberta/roberta-base",
        "/kaggle/input/transformers/roberta-base",
        "/kaggle/input/huggingface-roberta/roberta-base",
        "/kaggle/input/roberta-base-squad2",
        "/kaggle/input/roberta-base-pytorch/roberta-base",
        "/kaggle/input/cardiffnlp-twitter-roberta-base-sentiment",
        "/kaggle/input/twitter-roberta-base-sentiment",
    ]

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    candidates += [
        os.path.join(hf_home, "hub"),
        os.path.expanduser("~/.cache/huggingface/hub"),
        os.path.expanduser("~/.cache/huggingface/transformers"),
    ]

    def looks_like_model_dir(p):
        return os.path.isdir(p) and os.path.isfile(os.path.join(p, "config.json"))

    for p in candidates:
        if looks_like_model_dir(p):
            return p

    for base in [os.path.expanduser("~/.cache/huggingface/hub"), hf_home]:
        if not os.path.isdir(base):
            continue
        for root, _, files in os.walk(base):
            if "config.json" in files:
                if ("vocab.json" in files and "merges.txt" in files) or (
                    "tokenizer.json" in files
                ):
                    return root
    return None


MODEL_DIR = _find_any_local_roberta_dir()
MODEL_ID_TRIES = [
    MODEL_DIR,  # may be None
    "roberta-base",
    "cardiffnlp/twitter-roberta-base-sentiment",
]

config = None
hf_tokenizer = None
MODEL_DIR_USED = None

last_err = None
for model_id in MODEL_ID_TRIES:
    if not model_id:
        continue
    try:
        config = RobertaConfig.from_pretrained(
            model_id, output_hidden_states=True, local_files_only=True
        )
        hf_tokenizer = RobertaTokenizerFast.from_pretrained(
            model_id, local_files_only=True
        )
        MODEL_DIR_USED = model_id
        break
    except Exception as e:
        last_err = e
        continue

if config is None or hf_tokenizer is None or MODEL_DIR_USED is None:
    raise RuntimeError(
        "Could not load a local RoBERTa tokenizer/config. This notebook requires roberta-base "
        "or a compatible RoBERTa checkpoint to be available in the local HuggingFace cache or Kaggle inputs. "
        f"Last error: {repr(last_err)}"
    )

MODEL_DIR = MODEL_DIR_USED
print("Using model from:", MODEL_DIR)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/515889360.py in <cell line: 0>()
     76 
     77 if config is None or hf_tokenizer is None or MODEL_DIR_USED is None:
---> 78     raise RuntimeError(
     79         "Could not load a local RoBERTa tokenizer/config. This notebook requires roberta-base "
     80         "or a compatible RoBERTa checkpoint to be available in the local HuggingFace cache or Kaggle inputs. "

RuntimeError: Could not load a local RoBERTa tokenizer/config. This notebook requires roberta-base or a compatible RoBERTa checkpoint to be available in the local HuggingFace cache or Kaggle inputs. Last error: OSError("We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.\nCheck your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.")

## === cell 4
class _HFLikeBPE:
    def __init__(self, tok):
        self.tok = tok

    def encode(self, text: str) -> str:
        toks = self.tok.tokenize(text, add_prefix_space=False)
        return " ".join(toks)


class _HFLikeVocab:
    def __init__(self, tok):
        self.tok = tok

    def encode_line(
        self, token_str: str, append_eos=False, add_if_not_exist=False
    ) -> torch.LongTensor:
        toks = token_str.split() if isinstance(token_str, str) else list(token_str)
        ids = self.tok.convert_tokens_to_ids(toks)
        return torch.tensor(ids, dtype=torch.long)


bpe = _HFLikeBPE(hf_tokenizer)
vocab = _HFLikeVocab(hf_tokenizer)




## === cell 5
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, bpe, vocab, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df
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
        data["sentiment"] = row.sentiment
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
        masks = (ids != 1).long()
        return ids, masks, tweets_encoded

    def get_target_idx(self, row, tweets_encoded):
        normalized_selected_tweets = normalizeTweet(row.selected_text)
        normalized_selected_tweets = " " + " ".join(normalized_selected_tweets.split())
        normalized_tweets = normalizeTweet(row.text)
        normalized_tweets = " " + " ".join(normalized_tweets.split())

        len_st = len(normalized_selected_tweets) - 1
        idx0 = None
        idx1 = None

        if len(normalized_selected_tweets) > 1:
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

        if idx0 is None and len(normalized_selected_tweets.split()) > 2:
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
        else:
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
            MODEL_DIR, config=conf, local_files_only=True
        )
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
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
    for token in tweets_encoded.split()[start_idx - 4 : end_idx - 3]:
        selected_text += " " + token
    selected_text = re.sub("@@ ", "", selected_text)
    selected_text = re.sub("@@", "", selected_text)
    selected_text = selected_text.replace("Ġ", " ")
    return " ".join(selected_text.split())


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


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
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    loss_check = 1000
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0
            for count, data in enumerate(dataloaders_dict[phase]):
                if count % 100 == 0:
                    print(count)

                ids = data["ids"].to(device)
                masks = data["masks"].to(device)
                tweets_encoded = data["tweets_encoded"]

                start_idx = data["start_idx"].to(device)
                end_idx = data["end_idx"].to(device)

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
                start_logits_np = (
                    torch.softmax(start_logits, dim=1).detach().cpu().numpy()
                )
                end_logits_np = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

                for i in range(ids.size(0)):
                    epoch_jaccard += compute_jaccard_score(
                        tweets_encoded[i],
                        start_idx_np[i],
                        end_idx_np[i],
                        start_logits_np[i],
                        end_logits_np[i],
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
    train_df = (
        pd.read_csv(os.path.join(DATA_DIR, "train.csv")).dropna().reset_index(drop=True)
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
        os.path.join(WORK_DIR, f"roberta_fold{fold}.pth"),
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
config = config




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
                while end_idx < len(tweet) and count < length:
                    if tweet[end_idx] == " ":
                        end_idx += 1
                    elif (
                        tweet[end_idx] in ["!", ".", "*", "-", "?"]
                        and pred_wo_spaces[count] != tweet[end_idx]
                    ):
                        end_idx += 1
                    elif tweet[end_idx] == pred_wo_spaces[count]:
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
            while start_idx > 0:
                if tweet[start_idx - 1] == " ":
                    break
                else:
                    start_idx = start_idx - 1
            while end_idx < len(tweet) - 1:
                if tweet[end_idx] == " ":
                    break
                else:
                    end_idx = end_idx + 1
            return tweet[start_idx:end_idx]
        elif "HTTPURL" in pred.split():
            return tweet
        elif "@USER" in pred.split():
            return tweet
        else:
            return tweet




## === cell 16
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

test_df = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df, batch_size=batch_size)

models = []
for fold in range(skf.n_splits):
    ckpt_candidates = [
        f"/kaggle/input/mosh1-data-orig/roberta_fold{fold}.pth",
        os.path.join(WORK_DIR, f"roberta_fold{fold}.pth"),
    ]
    ckpt_path = None
    for c in ckpt_candidates:
        if os.path.isfile(c):
            ckpt_path = c
            break

    if ckpt_path is not None:
        m = BERTweetModel(conf=config).to(device)
        state = torch.load(ckpt_path, map_location="cpu")
        m.load_state_dict(state, strict=True)
        m.eval()
        models.append(m)

if len(models) == 0:
    run(0)
    ckpt_path = os.path.join(WORK_DIR, "roberta_fold0.pth")
    m = BERTweetModel(conf=config).to(device)
    state = torch.load(ckpt_path, map_location="cpu")
    m.load_state_dict(state, strict=True)
    m.eval()
    models.append(m)

predictions = []
for data in test_loader:
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweets_encoded = data["tweets_encoded"]
    tweet = data["tweet"]

    start_logits_list = []
    end_logits_list = []
    for model in models:
        with torch.no_grad():
            output = model(ids, masks)
            start_logits_list.append(
                torch.softmax(output[0], dim=1).detach().cpu().numpy()
            )
            end_logits_list.append(
                torch.softmax(output[1], dim=1).detach().cpu().numpy()
            )

    start_logits = np.mean(start_logits_list, axis=0)
    end_logits = np.mean(end_logits_list, axis=0)

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
        try:
            pred = postprocessing(pred, tweet[i].strip())
        except Exception:
            pred = tweet[i]
        predictions.append(pred)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
LocalEntryNotFoundError                   Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    469             # This is slightly better for only 1 file
--> 470             hf_hub_download(
    471                 path_or_repo_id,

/usr/local/lib/python3.11/dist-packages/huggingface_hub/utils/_validators.py in _inner_fn(*args, **kwargs)
    113 
--> 114         return fn(*args, **kwargs)
    115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in hf_hub_download(repo_id, filename, subfolder, repo_type, revision, library_name, library_version, cache_dir, local_dir, user_agent, force_download, proxies, etag_timeout, token, local_files_only, headers, endpoint, resume_download, force_filename, local_dir_use_symlinks)
   1006     else:
-> 1007         return _hf_hub_download_to_cache_dir(
   1008             # Destination

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _hf_hub_download_to_cache_dir(cache_dir, repo_id, filename, repo_type, revision, endpoint, etag_timeout, headers, proxies, token, local_files_only, force_download)
   1113         # Otherwise, raise appropriate error
-> 1114         _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1115 

/usr/local/lib/python3.11/dist-packages/huggingface_hub/file_download.py in _raise_on_head_call_error(head_call_error, force_download, local_files_only)
   1645     if local_files_only:
-> 1646         raise LocalEntryNotFoundError(
   1647             "Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable"

LocalEntryNotFoundError: Cannot find the requested files in the disk cache and outgoing traffic has been disabled. To enable hf.co look-ups and downloads online, set 'local_files_only' to False.

The above exception was the direct cause of the following exception:

OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/202355145.py in <cell line: 0>()
     25 
     26 if len(models) == 0:
---> 27     run(0)
     28     ckpt_path = os.path.join(WORK_DIR, "roberta_fold0.pth")
     29     m = BERTweetModel(conf=config).to(device)

/tmp/ipykernel_55/3028305768.py in run(fold)
      8     (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
      9     print(f"Fold: {fold}")
---> 10     model = BERTweetModel(conf=config)
     11 
     12     optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))

/tmp/ipykernel_55/3575677816.py in __init__(self, conf)
      2     def __init__(self, conf):
      3         super(BERTweetModel, self).__init__()
----> 4         self.roberta = RobertaModel.from_pretrained(
      5             MODEL_DIR, config=conf, local_files_only=True
      6         )

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in _wrapper(*args, **kwargs)
    309         old_dtype = torch.get_default_dtype()
    310         try:
--> 311             return func(*args, **kwargs)
    312         finally:
    313             torch.set_default_dtype(old_dtype)

/usr/local/lib/python3.11/dist-packages/transformers/modeling_utils.py in from_pretrained(cls, pretrained_model_name_or_path, config, cache_dir, ignore_mismatched_sizes, force_download, local_files_only, token, revision, use_safetensors, weights_only, *model_args, **kwargs)
   4581         if not isinstance(config, PretrainedConfig):
   4582             config_path = config if config is not None else pretrained_model_name_or_path
-> 4583             config, model_kwargs = cls.config_class.from_pretrained(
   4584                 config_path,
   4585                 cache_dir=cache_dir,

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
    665             try:
    666                 # Load from local folder or from cache or download from model Hub and cache
--> 667                 resolved_config_file = cached_file(
    668                     pretrained_model_name_or_path,
    669                     configuration_file,

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_file(path_or_repo_id, filename, **kwargs)
    310     ```
    311     """
--> 312     file = cached_files(path_or_repo_id=path_or_repo_id, filenames=[filename], **kwargs)
    313     file = file[0] if file is not None else file
    314     return file

/usr/local/lib/python3.11/dist-packages/transformers/utils/hub.py in cached_files(path_or_repo_id, filenames, cache_dir, force_download, resume_download, proxies, token, revision, local_files_only, subfolder, repo_type, user_agent, _raise_exceptions_for_gated_repo, _raise_exceptions_for_missing_entries, _raise_exceptions_for_connection_errors, _commit_hash, **deprecated_kwargs)
    541             # even when `local_files_only` is True, in which case raising for connections errors only would not make sense)
    542             elif _raise_exceptions_for_missing_entries:
--> 543                 raise OSError(
    544                     f"We couldn't connect to '{HUGGINGFACE_CO_RESOLVE_ENDPOINT}' to load the files, and couldn't find them in the"
    545                     f" cached files.\nCheck your internet connection or see how to run the library in offline mode at"

OSError: We couldn't connect to 'https://huggingface.co' to load the files, and couldn't find them in the cached files.
Check your internet connection or see how to run the library in offline mode at 'https://huggingface.co/docs/transformers/installation#offline-mode'.

## === cell 17
sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
sub_df = sub_df.iloc[: len(predictions)].copy()
sub_df["selected_text"] = predictions
out_path = os.path.join(WORK_DIR, "submission.csv")
sub_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub_df.head())

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1325877123.py in <cell line: 0>()
      1 sub_df = pd.read_csv(os.path.join(DATA_DIR, "sample_submission.csv"))
----> 2 sub_df = sub_df.iloc[: len(predictions)].copy()
      3 sub_df["selected_text"] = predictions
      4 out_path = os.path.join(WORK_DIR, "submission.csv")
      5 sub_df.to_csv(out_path, index=False)

NameError: name 'predictions' is not defined
