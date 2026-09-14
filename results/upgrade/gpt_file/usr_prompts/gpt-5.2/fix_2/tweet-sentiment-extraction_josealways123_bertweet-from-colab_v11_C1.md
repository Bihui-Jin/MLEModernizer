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

0.7031523585319519

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
import numpy as np
import pandas as pd

import torch
from torch import nn
import torch.optim as optim
from sklearn.model_selection import StratifiedKFold

from transformers import AutoConfig, AutoModel, AutoTokenizer

warnings.filterwarnings("ignore")

seed = 18


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 1
from nltk.tokenize import TweetTokenizer
from emoji import demojize

tokenizer_nltk = TweetTokenizer()


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
    tokens = tokenizer_nltk.tokenize(tweet.replace("’", "'").replace("…", "..."))
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
MODEL_NAME = "vinai/bertweet-base"

try:
    hf_config = AutoConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
    hf_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, normalization=True
    )
    hf_model_backbone = AutoModel.from_pretrained(MODEL_NAME, config=hf_config)
except Exception:
    hf_config = AutoConfig.from_pretrained(
        MODEL_NAME, output_hidden_states=True, local_files_only=True
    )
    hf_tokenizer = AutoTokenizer.from_pretrained(
        MODEL_NAME, use_fast=True, local_files_only=True
    )
    hf_model_backbone = AutoModel.from_pretrained(
        MODEL_NAME, config=hf_config, local_files_only=True
    )

_ = hf_tokenizer.cls_token_id, hf_tokenizer.sep_token_id, hf_tokenizer.pad_token_id




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, hf_tokenizer, max_len=96):
        self.df = df.reset_index(drop=True)
        self.labeled = "selected_text" in df.columns
        self.hf_tokenizer = hf_tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def _tokenize(self, sentiment, text):
        text_norm = normalizeTweet(str(text))
        sent_norm = normalizeTweet(str(sentiment))
        enc = self.hf_tokenizer(
            sent_norm,
            text_norm,
            add_special_tokens=True,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_attention_mask=True,
            return_offsets_mapping=True,
        )
        return enc, text_norm

    def __getitem__(self, index):
        row = self.df.iloc[index]
        enc, text_norm = self._tokenize(row.sentiment, row.text)

        ids = torch.tensor(enc["input_ids"], dtype=torch.long)
        masks = torch.tensor(enc["attention_mask"], dtype=torch.long)

        token_type_ids = enc.get("token_type_ids", None)
        toks = self.hf_tokenizer.convert_ids_to_tokens(enc["input_ids"])
        tweets_encoded = " ".join(toks)

        data = {
            "ids": ids,
            "masks": masks,
            "tweets_encoded": tweets_encoded,
            "tweet": str(row.text),
        }

        if self.labeled:
            sel = str(row.selected_text)
            data["selected_tweet"] = sel
            start_idx, end_idx = self.get_target_idx(row, enc, text_norm)
            data["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
            data["end_idx"] = torch.tensor(end_idx, dtype=torch.long)

        return data

    def get_target_idx(self, row, enc, text_norm):
        selected_norm = normalizeTweet(str(row.selected_text)).strip()
        if not selected_norm:
            return 0, 0

        start_char = text_norm.lower().find(selected_norm.lower())
        if start_char < 0:
            parts = selected_norm.split()
            if len(parts) > 1:
                start_char = text_norm.lower().find(" ".join(parts[1:]).lower())
            if start_char < 0 and len(parts) > 1:
                start_char = text_norm.lower().find(" ".join(parts[:-1]).lower())
        if start_char < 0:
            return 0, 0
        end_char = start_char + len(selected_norm)

        offsets = enc["offset_mapping"]
        token_type_ids = enc.get("token_type_ids", None)

        start_tok = None
        end_tok = None
        for i, (off, tid) in enumerate(
            zip(
                offsets,
                token_type_ids if token_type_ids is not None else [1] * len(offsets),
            )
        ):
            o0, o1 = off
            if o0 == 0 and o1 == 0:
                continue
            if token_type_ids is not None and tid != 1:
                continue
            if start_tok is None and o0 <= start_char < o1:
                start_tok = i
            if o0 < end_char <= o1:
                end_tok = i
                break
            if o0 >= start_char and o1 <= end_char:
                end_tok = i

        if start_tok is None or end_tok is None:
            return 0, 0
        return start_tok, end_tok




## === cell 5
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32, max_len=96):
    train_df = df.iloc[train_idx].reset_index(drop=True)
    val_df = df.iloc[val_idx].reset_index(drop=True)

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df, hf_tokenizer, max_len=max_len),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df, hf_tokenizer, max_len=max_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )

    return {"train": train_loader, "val": val_loader}




## === cell 6
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = hf_model_backbone
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.normal_(self.fc.bias, 0)

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        hidden_states = outputs.hidden_states  # tuple: embeddings + layers
        x = torch.cat(
            [
                hidden_states[-1],
                hidden_states[-2],
                hidden_states[-3],
                hidden_states[-4],
            ],
            dim=-1,
        )
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 7
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    return start_loss + end_loss




## === cell 8
def get_selected_text(tweets_encoded, start_idx, end_idx):
    toks = tweets_encoded.split()
    start_idx = int(start_idx)
    end_idx = int(end_idx)
    start_idx = max(0, min(start_idx, len(toks) - 1))
    end_idx = max(0, min(end_idx, len(toks) - 1))
    if end_idx < start_idx:
        start_idx, end_idx = 0, len(toks) - 1
    sel_toks = toks[start_idx : end_idx + 1]
    return hf_tokenizer.convert_tokens_to_string(sel_toks)


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = int(np.argmax(start_logits))
    end_pred = int(np.argmax(end_logits))
    length = len(tweets_encoded.split())

    start_pred = max(0, min(start_pred, length - 1))
    end_pred = max(0, min(end_pred, length - 1))

    if start_pred > end_pred:
        start_pred, end_pred = 0, length - 1

    pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()
    return jaccard(true, pred)




## === cell 9
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(device)

    loss_check = 1e18
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()

            epoch_loss = 0.0
            epoch_jaccard = 0.0

            for step, data in enumerate(dataloaders_dict[phase]):
                if step % 100 == 0:
                    print(step)

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
                f"Epoch {epoch+1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Jaccard: {epoch_jaccard:.4f}"
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
max_len = 96
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 11
def run(fold):
    train_path = "/kaggle/input/tweet-sentiment-extraction/train.csv"
    if not os.path.exists(train_path):
        train_path = "../input/tweet-sentiment-extraction/train.csv"

    train_df = pd.read_csv(train_path).dropna().reset_index(drop=True)
    train_df["text"] = train_df["text"].astype(str)
    train_df["selected_text"] = train_df["selected_text"].astype(str)

    (train_idx, val_idx) = list(skf.split(train_df, train_df.sentiment))[fold]
    print(f"Fold: {fold}")

    model = BERTweetModel(conf=hf_config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5, betas=(0.9, 0.999))
    criterion = loss_fn

    dataloaders_dict = get_train_val_loaders(
        train_df, train_idx, val_idx, batch_size=batch_size, max_len=max_len
    )
    print("starting training")
    out_name = f"roberta_fold{fold}.pth"
    train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, out_name)
    return out_name




## === cell 12
def get_test_loader(df, batch_size=32, max_len=96):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df, hf_tokenizer, max_len=max_len),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False,
        num_workers=2,
        pin_memory=torch.cuda.is_available(),
    )
    return loader




## === cell 13
def postprocessing(pred, tweet):
    pred_wo_spaces = "".join(str(pred).split())
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
            return tweet[start_idx:end_idx]
        elif "HTTPURL" in str(pred).split():
            return tweet
        elif "@USER" in str(pred).split():
            return tweet
        else:
            return tweet




## === cell 14
ckpt_name = "roberta_fold3.pth"
if not os.path.exists(ckpt_name):
    ckpt_name = run(fold=3)

test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"
sample_path = "/kaggle/input/tweet-sentiment-extraction/sample_submission.csv"
if not os.path.exists(test_path):
    test_path = "../input/tweet-sentiment-extraction/test.csv"
    sample_path = "../input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(test_path)
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df, batch_size=batch_size, max_len=max_len)

model = BERTweetModel(conf=hf_config).to(device)
state = torch.load(ckpt_name, map_location="cpu")
model.load_state_dict(state, strict=True)
model.eval()

predictions = []
for step, data in enumerate(test_loader):
    print(step)
    ids = data["ids"].to(device)
    masks = data["masks"].to(device)
    tweets_encoded = data["tweets_encoded"]
    tweet = data["tweet"]

    with torch.no_grad():
        start_logits, end_logits = model(ids, masks)
        start_probs = torch.softmax(start_logits, dim=1).detach().cpu().numpy()
        end_probs = torch.softmax(end_logits, dim=1).detach().cpu().numpy()

    for i in range(ids.size(0)):
        start_pred = int(np.argmax(start_probs[i]))
        end_pred = int(np.argmax(end_probs[i]))
        length = len(tweets_encoded[i].split())
        start_pred = max(0, min(start_pred, length - 1))
        end_pred = max(0, min(end_pred, length - 1))
        if start_pred > end_pred:
            start_pred, end_pred = 0, length - 1

        pred = get_selected_text(tweets_encoded[i], start_pred, end_pred).strip()
        try:
            pred = postprocessing(pred, tweet[i].strip())
        except Exception:
            pred = tweet[i]
        predictions.append(pred)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_55/3437958567.py in <cell line: 0>()
      5 ckpt_name = "roberta_fold3.pth"
      6 if not os.path.exists(ckpt_name):
----> 7     ckpt_name = run(fold=3)
      8 
      9 test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"

/tmp/ipykernel_55/2699468622.py in run(fold)
     21     print("starting training")
     22     out_name = f"roberta_fold{fold}.pth"
---> 23     train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, out_name)
     24     return out_name
     25 

/tmp/ipykernel_55/2335231647.py in train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename)
     10             epoch_jaccard = 0.0
     11 
---> 12             for step, data in enumerate(dataloaders_dict[phase]):
     13                 if step % 100 == 0:
     14                     print(step)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

NotImplementedError: Caught NotImplementedError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/412628907.py", line 32, in __getitem__
    enc, text_norm = self._tokenize(row.sentiment, row.text)
                     ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/412628907.py", line 18, in _tokenize
    enc = self.hf_tokenizer(
          ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 2855, in __call__
    encodings = self._call_one(text=text, text_pair=text_pair, **all_kwargs)
                ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 2965, in _call_one
    return self.encode_plus(
           ^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py", line 3040, in encode_plus
    return self._encode_plus(
           ^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils.py", line 792, in _encode_plus
    raise NotImplementedError(
NotImplementedError: return_offset_mapping is not available when using Python tokenizers. To use this feature, change your tokenizer to one deriving from transformers.PreTrainedTokenizerFast. More information on available tokenizers at https://github.com/huggingface/transformers/pull/2674


## === cell 15
sub_df = pd.read_csv(sample_path)
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
sub_df.head()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4150727502.py in <cell line: 0>()
----> 1 sub_df = pd.read_csv(sample_path)
      2 sub_df["selected_text"] = predictions
      3 
      4 # keep original single-word cleanup
      5 sub_df["selected_text"] = sub_df["selected_text"].apply(

NameError: name 'sample_path' is not defined
