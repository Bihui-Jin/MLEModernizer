# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import re
import warnings
import random
import sys

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

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

DEVICE = torch.device("cpu")
print("DEVICE:", DEVICE)



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

MODEL_SRC = None
local_only = False

if os.path.isdir(local_model_dir):
    MODEL_SRC = local_model_dir
    local_only = True
else:
    candidate_sources = ["vinai/bertweet-base", "roberta-base"]
    for src in candidate_sources:
        try:
            _ = RobertaConfig.from_pretrained(src, local_files_only=False)
            MODEL_SRC = src
            local_only = False
            break
        except Exception:
            continue

if MODEL_SRC is None:
    MODEL_SRC = "roberta-base"
    local_only = True

config = RobertaConfig.from_pretrained(
    MODEL_SRC,
    output_hidden_states=True,
    local_files_only=local_only,
)
config.attn_implementation = "eager"

hf_tokenizer = RobertaTokenizerFast.from_pretrained(
    MODEL_SRC, local_files_only=local_only
)

TOKENIZER_LEN = int(len(hf_tokenizer))
print(
    "MODEL_SRC:",
    MODEL_SRC,
    "| local_only:",
    local_only,
    "| len(tokenizer):",
    TOKENIZER_LEN,
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
        self.unk_id = int(tokenizer.unk_token_id)

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
        self.df = df.reset_index(drop=True)
        self.labeled = "selected_text" in df.columns
        self.bpe = bpe
        self.vocab = vocab
        self.max_len = max_len

        self.bos_id = int(self.vocab.tokenizer.bos_token_id)
        self.eos_id = int(self.vocab.tokenizer.eos_token_id)
        self.pad_id = int(self.vocab.tokenizer.pad_token_id)
        self.sep_id = (
            int(self.vocab.tokenizer.sep_token_id)
            if self.vocab.tokenizer.sep_token_id is not None
            else int(self.eos_id)
        )
        self.unk_id = int(self.vocab.tokenizer.unk_token_id)
        self.vocab_size = int(len(self.vocab.tokenizer))

        self._cache = self._build_cache()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        return self._cache[index]

    def _sanitize_ids(self, ids_list):
        out = []
        vsz = self.vocab_size
        unk = self.unk_id
        for x in ids_list:
            try:
                xi = int(x)
            except Exception:
                xi = unk
            if xi < 0 or xi >= vsz:
                xi = unk
            out.append(xi)
        return out

    def _get_input_data_from_norm(self, norm_text: str, sentiment: str):
        tweets_encoded = self.bpe.encode(norm_text)

        encoding_ids = (
            self.vocab.encode_line(
                tweets_encoded, append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )
        encoding_ids = self._sanitize_ids(encoding_ids)

        sentiment_id = (
            self.vocab.encode_line(
                self.bpe.encode(sentiment), append_eos=False, add_if_not_exist=False
            )
            .long()
            .tolist()
        )
        sentiment_id = self._sanitize_ids(sentiment_id)

        ids = (
            [self.bos_id]
            + sentiment_id
            + [self.eos_id, self.eos_id]
            + encoding_ids
            + [self.eos_id]
        )

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.pad_id] * pad_len
        else:
            ids = ids[: self.max_len]

        ids = self._sanitize_ids(ids)
        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != self.pad_id).to(dtype=torch.long)

        tweet_token_len = len(tweets_encoded.split())
        return ids, masks, tweets_encoded, tweet_token_len

    def get_target_idx(self, norm_selected: str, norm_text: str, tweets_encoded: str):
        len_st = len(norm_selected) - 1
        idx0 = None
        idx1 = None
        for ind in (i for i, e in enumerate(norm_text) if e == norm_selected[1]):
            if " " + norm_text[ind : ind + len_st] == norm_selected:
                idx0 = ind
                idx1 = ind + len_st - 1
                break
        if idx0 is None and len(norm_selected.split()) > 1:
            norm_selected_1 = " " + " ".join(norm_selected.split()[1:])
            len_st_1 = len(norm_selected_1) - 1
            for ind in (i for i, e in enumerate(norm_text) if e == norm_selected_1[1]):
                if " " + norm_text[ind : ind + len_st_1] == norm_selected_1:
                    idx0 = ind
                    idx1 = ind + len_st_1 - 1
                    break
        if idx0 is None and len(norm_selected.split()) > 1:
            norm_selected_2 = " " + " ".join(norm_selected.split()[:-1])
            len_st_2 = len(norm_selected_2) - 1
            for ind in (i for i, e in enumerate(norm_text) if e == norm_selected_2[1]):
                if " " + norm_text[ind : ind + len_st_2] == norm_selected_2:
                    idx0 = ind
                    idx1 = ind + len_st_2 - 1
                    break
        if idx0 is None and len(norm_selected.split()) > 1:
            norm_selected_3 = " " + " ".join(norm_selected_2.split()[:-1])
            len_st_3 = len(norm_selected_3) - 1
            for ind in (i for i, e in enumerate(norm_text) if e == norm_selected_3[1]):
                if " " + norm_text[ind : ind + len_st_3] == norm_selected_3:
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

    def _build_cache(self):
        cache = []
        texts = self.df["text"].astype(str).tolist()
        sentiments = self.df["sentiment"].astype(str).tolist()
        selected_texts = (
            self.df["selected_text"].astype(str).tolist() if self.labeled else None
        )

        for i in range(len(self.df)):
            row_text = texts[i]
            row_sent = sentiments[i]

            norm_text = normalizeTweet(row_text)
            norm_text = " " + " ".join(norm_text.split())

            ids, masks, tweets_encoded, tweet_token_len = (
                self._get_input_data_from_norm(norm_text, row_sent)
            )

            data = {
                "ids": ids,
                "masks": masks,
                "tweets_encoded": tweets_encoded,
                "tweet": row_text,
                "tweet_token_len": torch.tensor(tweet_token_len, dtype=torch.long),
            }

            if self.labeled:
                row_sel = selected_texts[i]
                norm_sel = normalizeTweet(row_sel)
                norm_sel = " " + " ".join(norm_sel.split())
                start_idx, end_idx = self.get_target_idx(
                    norm_sel, norm_text, tweets_encoded
                )
                data["selected_tweet"] = row_sel
                data["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
                data["end_idx"] = torch.tensor(end_idx, dtype=torch.long)

            cache.append(data)
        return cache




## === cell 6
def get_train_val_loaders(
    df, train_idx, val_idx, batch_size=32, num_workers=2, pin_memory=None
):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    if pin_memory is None:
        pin_memory = torch.cuda.is_available() and (DEVICE.type == "cuda")

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
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

        try:
            tok_len = int(len(hf_tokenizer))
            self.roberta.resize_token_embeddings(tok_len)
        except Exception:
            pass

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
    toks = tweets_encoded.split()
    start = max(0, start_idx - 4)
    end = max(0, end_idx - 3)
    selected_text = " " + " ".join(toks[start:end]) if end > start else ""
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
def mask_invalid_positions(start_logits, end_logits, tweet_token_len):
    """
    Bugfix: avoid in-place modification on tensors that may be views (PyTorch forbids it for autograd).
    Semantics are identical: only allow positions [4, 3+tweet_token_len] per row.
    """
    bs, seq_len = start_logits.shape
    device = start_logits.device

    pos = torch.arange(seq_len, device=device).unsqueeze(0).expand(bs, -1)  # (bs, L)
    L = torch.as_tensor(tweet_token_len, device=device, dtype=torch.long).view(-1, 1)
    lo = 4
    hi = (3 + L).clamp(min=lo, max=seq_len - 1)  # (bs, 1)
    valid = (pos >= lo) & (pos <= hi)

    neg = torch.tensor(-1e9, device=device, dtype=start_logits.dtype)
    start_logits = torch.where(valid, start_logits, neg)
    end_logits = torch.where(valid, end_logits, neg)
    return start_logits, end_logits




## === cell 11
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    global DEVICE
    use_device = DEVICE
    model.to(use_device)

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

                ids = data["ids"].to(use_device, dtype=torch.long)
                masks = data["masks"].to(use_device, dtype=torch.long)
                tweets_encoded = data["tweets_encoded"]
                tweet_token_len = data["tweet_token_len"].detach().cpu().numpy()

                start_idx = data["start_idx"].to(use_device, dtype=torch.long)
                end_idx = data["end_idx"].to(use_device, dtype=torch.long)

                optimizer.zero_grad(set_to_none=True)

                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    start_logits, end_logits = mask_invalid_positions(
                        start_logits, end_logits, tweet_token_len
                    )

                    seq_len = start_logits.size(1)
                    start_idx_clamped = start_idx.clamp(min=0, max=seq_len - 1)
                    end_idx_clamped = end_idx.clamp(min=0, max=seq_len - 1)

                    loss = criterion(
                        start_logits, end_logits, start_idx_clamped, end_idx_clamped
                    )

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                epoch_loss += loss.item() * ids.size(0)

                start_idx_np = start_idx_clamped.detach().cpu().numpy()
                end_idx_np = end_idx_clamped.detach().cpu().numpy()
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




## === cell 12
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 13
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

    nw = 0
    pm = False
    dataloaders_dict = get_train_val_loaders(
        train_df, train_idx, val_idx, batch_size, num_workers=nw, pin_memory=pm
    )

    print("starting training")
    train_model(
        model,
        dataloaders_dict,
        criterion,
        optimizer,
        num_epochs,
        f"roberta_fold{fold}.pth",
    )




## === cell 14
def get_test_loader(df, batch_size=32, num_workers=2, pin_memory=None):
    if pin_memory is None:
        pin_memory = torch.cuda.is_available()
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, bpe, vocab),
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader




## === cell 15
external_weights_dir = "../input/mosh1-data-orig"


def _find_weight_path(fold: int) -> str:
    fname = f"roberta_fold{fold}.pth"
    cand1 = os.path.join(external_weights_dir, fname)
    cand2 = os.path.join(".", fname)
    if os.path.isfile(cand1):
        return cand1
    if os.path.isfile(cand2):
        return cand2
    return ""


missing_folds = [f for f in range(skf.n_splits) if _find_weight_path(f) == ""]
print("Missing folds:", missing_folds)

if len(missing_folds) > 0 and not os.path.isdir(external_weights_dir):
    for f in missing_folds:
        run(f)




## === cell 16
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




## === cell 17
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)

predictions = []
models = []

use_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_device.type == "cuda":
    try:
        torch.cuda.empty_cache()
    except Exception:
        pass

for fold in range(skf.n_splits):
    wp = _find_weight_path(fold)
    if wp:
        try:
            model = BERTweetModel(conf=config).to(use_device)
        except RuntimeError as e:
            if "device-side assert" in str(e).lower() and use_device.type == "cuda":
                print(
                    "CUDA error while moving model for inference; switching inference to CPU."
                )
                try:
                    torch.cuda.empty_cache()
                except Exception:
                    pass
                use_device = torch.device("cpu")
                model = BERTweetModel(conf=config).to(use_device)
            else:
                raise

        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state, strict=True)
        model.eval()
        models.append(model)

if len(models) == 0:
    print("WARNING: No fold weights found; using a single randomly-initialized model.")
    model = BERTweetModel(conf=config).to(use_device)
    model.eval()
    models = [model]

nw = 0 if use_device.type == "cpu" else 2
pm = False if use_device.type == "cpu" else torch.cuda.is_available()
test_loader = get_test_loader(
    test_df, batch_size=batch_size, num_workers=nw, pin_memory=pm
)

model_vocab_size = int(models[0].roberta.embeddings.word_embeddings.weight.size(0))

count = 0
for data in test_loader:
    if count % 50 == 0:
        print(count)
    count += 1

    ids = data["ids"].to(dtype=torch.long)
    masks = data["masks"].to(dtype=torch.long)

    ids = ids.clamp(min=0, max=model_vocab_size - 1)
    masks = (masks != 0).to(dtype=torch.long)

    ids = ids.to(use_device, non_blocking=False)
    masks = masks.to(use_device, non_blocking=False)

    tweets_encoded = data["tweets_encoded"]
    tweet = data["tweet"]
    tweet_token_len = data["tweet_token_len"].detach().cpu().numpy()

    start_logits = []
    end_logits = []

    try:
        with torch.inference_mode():
            for model in models:
                out_start, out_end = model(ids, masks)
                out_start, out_end = mask_invalid_positions(
                    out_start, out_end, tweet_token_len
                )
                start_logits.append(torch.softmax(out_start, dim=1).cpu().numpy())
                end_logits.append(torch.softmax(out_end, dim=1).cpu().numpy())
    except RuntimeError as e:
        if "device-side assert" in str(e).lower() and use_device.type == "cuda":
            print(
                "CUDA device-side assert encountered; switching inference to CPU for stability."
            )
            try:
                torch.cuda.empty_cache()
            except Exception:
                pass
            use_device = torch.device("cpu")
            models = [m.to(use_device).eval() for m in models]

            test_loader_cpu = get_test_loader(
                test_df, batch_size=batch_size, num_workers=0, pin_memory=False
            )

            predictions = []
            for data_cpu in test_loader_cpu:
                ids_cpu = data_cpu["ids"].to(dtype=torch.long)
                masks_cpu = data_cpu["masks"].to(dtype=torch.long)
                ids_cpu = ids_cpu.clamp(min=0, max=model_vocab_size - 1)
                masks_cpu = (masks_cpu != 0).to(dtype=torch.long)

                ids_cpu = ids_cpu.to(use_device)
                masks_cpu = masks_cpu.to(use_device)

                tweets_encoded_cpu = data_cpu["tweets_encoded"]
                tweet_cpu = data_cpu["tweet"]
                tweet_token_len_cpu = data_cpu["tweet_token_len"].detach().cpu().numpy()

                sl = []
                el = []
                with torch.inference_mode():
                    for model in models:
                        os_, oe_ = model(ids_cpu, masks_cpu)
                        os_, oe_ = mask_invalid_positions(os_, oe_, tweet_token_len_cpu)
                        sl.append(torch.softmax(os_, dim=1).cpu().numpy())
                        el.append(torch.softmax(oe_, dim=1).cpu().numpy())
                sl = np.mean(sl, axis=0)
                el = np.mean(el, axis=0)

                for i in range(ids_cpu.size(0)):
                    start_pred = int(np.argmax(sl[i]))
                    end_pred = int(np.argmax(el[i]))
                    length = len(tweets_encoded_cpu[i].split())
                    if start_pred < 4:
                        start_pred = 4
                    if end_pred > 3 + length:
                        end_pred = 3 + length
                    if start_pred > end_pred:
                        start_pred = 4
                        end_pred = 3 + length
                        pred = get_selected_text(
                            tweets_encoded_cpu[i], start_pred, end_pred
                        ).strip()
                    else:
                        pred = get_selected_text(
                            tweets_encoded_cpu[i], start_pred, end_pred
                        ).strip()

                    pred = postprocessing(pred, tweet_cpu[i].strip())
                    predictions.append(pred)

            break
        else:
            raise

    if len(predictions) == 0:
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
                pred = get_selected_text(
                    tweets_encoded[i], start_pred, end_pred
                ).strip()
            else:
                pred = get_selected_text(
                    tweets_encoded[i], start_pred, end_pred
                ).strip()

            pred = postprocessing(pred, tweet[i].strip())
            predictions.append(pred)

sub_df = test_df[["textID"]].copy()
if len(predictions) != len(sub_df):
    if len(predictions) < len(sub_df):
        missing = len(sub_df) - len(predictions)
        predictions = predictions + test_df["text"].tolist()[-missing:]
    else:
        predictions = predictions[: len(sub_df)]

sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: (
        x.replace("!!!!", "!") if isinstance(x, str) and len(x.split()) == 1 else x
    )
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: (
        x.replace("..", ".") if isinstance(x, str) and len(x.split()) == 1 else x
    )
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: (
        x.replace("...", ".") if isinstance(x, str) and len(x.split()) == 1 else x
    )
)

sub_df = sub_df[["textID", "selected_text"]]
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
print(
    "Models used:",
    len(models),
    "| External weights dir exists:",
    os.path.isdir(external_weights_dir),
    "| Inference device:",
    use_device,
)
