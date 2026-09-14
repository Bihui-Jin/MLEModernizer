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

warnings.filterwarnings("ignore")
seed = 18
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

from transformers import RobertaModel, RobertaConfig, RobertaTokenizer

try:
    from fairseq.data.encoders.fastbpe import fastBPE
    from fairseq.data import Dictionary

    fastbpe_available = True
except Exception:
    fastbpe_available = False
    fastBPE = None
    Dictionary = None



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
_ = normalizeTweet(" I`d have responded, if I were going")



## === cell 3
config = RobertaConfig.from_pretrained("vinai/bertweet-base", output_hidden_states=True)



## === cell 4
bertweet_tokenizer = RobertaTokenizer.from_pretrained(
    "vinai/bertweet-base", add_prefix_space=True, use_fast=True
)

if fastbpe_available:
    base_path = "../input/bertweet-dataset"
    bpe = fastBPE(
        argparse.Namespace(
            bpe_codes=os.path.join(base_path, "BERTweet_base_transformers/bpe.codes")
        )
    )
    vocab = Dictionary()
    vocab.add_from_file(os.path.join(base_path, "BERTweet_base_transformers/dict.txt"))
else:
    bpe = None
    vocab = None




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2485421369.py in <cell line: 0>()
      1 # Initialise tokenizer (adds a leading space for consistency with RoBERTa)
----> 2 bertweet_tokenizer = RobertaTokenizer.from_pretrained(
      3     "vinai/bertweet-base", add_prefix_space=True, use_fast=True
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in from_pretrained(cls, pretrained_model_name_or_path, cache_dir, force_download, local_files_only, token, revision, trust_remote_code, *init_inputs, **kwargs)
   2012                 logger.info(f"loading file {file_path} from cache at {resolved_vocab_files[file_id]}")
   2013 
-> 2014         return cls._from_pretrained(
   2015             resolved_vocab_files,
   2016             pretrained_model_name_or_path,

/usr/local/lib/python3.11/dist-packages/transformers/tokenization_utils_base.py in _from_pretrained(cls, resolved_vocab_files, pretrained_model_name_or_path, init_configuration, token, cache_dir, local_files_only, _commit_hash, _is_local, trust_remote_code, *init_inputs, **kwargs)
   2258         # Instantiate the tokenizer.
   2259         try:
-> 2260             tokenizer = cls(*init_inputs, **init_kwargs)
   2261         except import_protobuf_decode_error():
   2262             logger.info(

/usr/local/lib/python3.11/dist-packages/transformers/models/roberta/tokenization_roberta.py in __init__(self, vocab_file, merges_file, errors, bos_token, eos_token, sep_token, cls_token, unk_token, pad_token, mask_token, add_prefix_space, **kwargs)
    185         # these special tokens are not part of the vocab.json, let's add them in the correct order
    186 
--> 187         with open(vocab_file, encoding="utf-8") as vocab_handle:
    188             self.encoder = json.load(vocab_handle)
    189         self.decoder = {v: k for k, v in self.encoder.items()}

TypeError: expected str, bytes or os.PathLike object, not NoneType

## === cell 5
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df, max_len=96):
        self.df = df
        self.labeled = "selected_text" in df.columns
        self.max_len = max_len

    def __len__(self):
        return len(self.df)

    def __getitem__(self, index):
        row = self.df.iloc[index]
        encoding = bertweet_tokenizer.encode_plus(
            row.text,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        ids = encoding["input_ids"].squeeze(0)  # shape: (seq_len,)
        masks = encoding["attention_mask"].squeeze(0)  # shape: (seq_len,)
        tweets_encoded = bertweet_tokenizer.convert_ids_to_tokens(ids.tolist())
        tweets_encoded_str = " ".join(tweets_encoded)

        item = {
            "ids": ids,
            "masks": masks,
            "tweets_encoded": tweets_encoded_str,
            "tweet": row.text,
        }
        if self.labeled:
            start_idx = 0
            end_idx = len(tweets_encoded) - 1
            item["selected_tweet"] = row.selected_text
            item["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
            item["end_idx"] = torch.tensor(end_idx, dtype=torch.long)
        return item




## === cell 6
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df), batch_size=batch_size, shuffle=True, drop_last=False
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2
    )

    return {"train": train_loader, "val": val_loader}




## === cell 7
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super(BERTweetModel, self).__init__()
        self.roberta = RobertaModel.from_pretrained("vinai/bertweet-base", config=conf)
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(conf.hidden_size * 4, 2)
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, input_ids, attention_mask):
        a, b, h = self.roberta(input_ids, attention_mask)
        x = torch.cat([h[-1], h[-2], h[-3], h[-4]], dim=-1)
        x = self.fc(self.dropout(x))
        start_logits, end_logits = x.split(1, -1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 8
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce = nn.CrossEntropyLoss()
    start_loss = ce(start_logits, start_positions)
    end_loss = ce(end_logits, end_positions)
    return start_loss + end_loss




## === cell 9
def get_selected_text(tweets_encoded, start_idx, end_idx):
    tokens = tweets_encoded.split()
    selected = tokens[start_idx : end_idx + 1]
    text = " ".join(selected)
    text = text.replace("@@ ", "").replace("@@", "")
    return text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c) + 1e-8)


def compute_jaccard_score(tweets_encoded, start_idx, end_idx, start_logits, end_logits):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    length = len(tweets_encoded.split())
    start_pred = max(start_pred, 0)
    end_pred = min(end_pred, length - 1)
    if start_pred > end_pred:
        start_pred, end_pred = 0, length - 1
    pred = get_selected_text(tweets_encoded, start_pred, end_pred).strip()
    true = get_selected_text(tweets_encoded, start_idx, end_idx).strip()
    return jaccard(true, pred)




## === cell 10
def train_model(model, dataloaders, criterion, optimizer, num_epochs, filename):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_loss = float("inf")
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            epoch_loss = 0.0
            epoch_jacc = 0.0
            for batch in dataloaders[phase]:
                ids = batch["ids"].to(device)
                masks = batch["masks"].to(device)
                start_idx = batch["start_idx"].to(device)
                end_idx = batch["end_idx"].to(device)
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                epoch_loss += loss.item() * ids.size(0)
                start_np = start_logits.detach().cpu().numpy()
                end_np = end_logits.detach().cpu().numpy()
                for i in range(ids.size(0)):
                    j = compute_jaccard_score(
                        batch["tweets_encoded"][i],
                        start_idx[i].item(),
                        end_idx[i].item(),
                        start_np[i],
                        end_np[i],
                    )
                    epoch_jacc += j
            epoch_loss /= len(dataloaders[phase].dataset)
            epoch_jacc /= len(dataloaders[phase].dataset)
            print(
                f"Epoch {epoch+1}/{num_epochs} | {phase} | loss: {epoch_loss:.4f} | Jacc: {epoch_jacc:.4f}"
            )
        if epoch_loss < best_loss:
            best_loss = epoch_loss
            torch.save(model.state_dict(), filename)
    print("Training complete.")




## === cell 11
num_epochs = 2  # short training to keep runtime low
batch_size = 32
skf = StratifiedKFold(n_splits=4, shuffle=True, random_state=seed)




## === cell 12
def run_training(fold):
    df = (
        pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
        .dropna()
        .reset_index(drop=True)
    )
    (train_idx, val_idx) = list(skf.split(df, df.sentiment))[fold]
    print(f"Running fold {fold}")
    model = BERTweetModel(conf=config)
    optimizer = optim.AdamW(model.parameters(), lr=1e-5)
    loaders = get_train_val_loaders(df, train_idx, val_idx, batch_size)
    train_model(
        model, loaders, loss_fn, optimizer, num_epochs, f"roberta_fold{fold}.pth"
    )
    return model




## === cell 13
def get_test_loader(df, batch_size=32):
    return torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
    )




## === cell 14
model = BERTweetModel(conf=config)
model.eval()
if torch.cuda.is_available():
    model.cuda()




## === cell 15
def postprocessing(pred, tweet):
    return pred if pred else tweet




## === cell 16
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df)

predictions = []
with torch.no_grad():
    for batch in test_loader:
        ids = batch["ids"]
        masks = batch["masks"]
        tweets_enc = batch["tweets_encoded"]
        tweets_raw = batch["tweet"]
        if torch.cuda.is_available():
            ids = ids.cuda()
            masks = masks.cuda()
        start_logits, end_logits = model(ids, masks)
        start_logits = torch.softmax(start_logits, dim=1).cpu().numpy()
        end_logits = torch.softmax(end_logits, dim=1).cpu().numpy()
        for i in range(ids.size(0)):
            start_pred = np.argmax(start_logits[i])
            end_pred = np.argmax(end_logits[i])
            length = len(tweets_enc[i].split())
            start_pred = max(start_pred, 0)
            end_pred = min(end_pred, length - 1)
            if start_pred > end_pred:
                start_pred, end_pred = 0, length - 1
            pred = get_selected_text(tweets_enc[i], start_pred, end_pred).strip()
            pred = postprocessing(pred, tweets_raw[i].strip())
            predictions.append(pred)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2576190432.py in <cell line: 0>()
      5 predictions = []
      6 with torch.no_grad():
----> 7     for batch in test_loader:
      8         ids = batch["ids"]
      9         masks = batch["masks"]

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

NameError: Caught NameError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/3410765103.py", line 12, in __getitem__
    encoding = bertweet_tokenizer.encode_plus(
               ^^^^^^^^^^^^^^^^^^
NameError: name 'bertweet_tokenizer' is not defined


## === cell 17
sub = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sub["selected_text"] = predictions
sub.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv")

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/477555028.py in <cell line: 0>()
      1 sub = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
----> 2 sub["selected_text"] = predictions
      3 sub.to_csv("submission.csv", index=False)
      4 print("Submission file saved as submission.csv")

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

ValueError: Length of values (0) does not match length of index (2749)
