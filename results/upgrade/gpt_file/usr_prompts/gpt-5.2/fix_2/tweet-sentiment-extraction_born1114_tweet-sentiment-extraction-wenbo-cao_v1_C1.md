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

0.7109977006912231

# 6. Current score

0.46149

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46149) has done: 'I fix the missing model/tokenizer paths by switching from the non-existent `/kaggle/input/robertabase/` to loading `roberta-base` via `transformers` (which is available in Kaggle) while keeping your span-prediction architecture and training/eval logic intact. I also correct a few runtime-breaking issues: attention mask was all zeros (so the model can’t attend), offsets were taken from a CUDA tensor via `.numpy()`, and the `RobertaModel` forward output unpacking is incompatible with current `transformers`. Finally, because your provided pretrained checkpoint path doesn’t exist, I make training run end-to-end (with a short single epoch for time safety) and ensure a valid `submission.csv` with the required columns is written.'

# 9. Code solution

## === cell 0
import os
import re
import string
import random

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
import torch.nn.functional as F

from tqdm.autonotebook import tqdm
from sklearn.model_selection import train_test_split

import tokenizers
import transformers


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))

print("torch:", torch.__version__)
print("transformers:", transformers.__version__)



## === cell 1
data_path = r"/kaggle/input/tweet-sentiment-extraction/"
train_data = pd.read_csv(os.path.join(data_path, "train.csv"))
train_data.dropna(inplace=True)
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

train_data["text"] = train_data["text"].astype(str).apply(lambda x: x.strip())
train_data["selected_text"] = train_data["selected_text"].astype(str)
test_data["text"] = test_data["text"].astype(str).apply(lambda x: x.strip())

print(train_data.shape, test_data.shape)
print(train_data.head(2))



## === cell 2
MODEL_NAME = "roberta-base"

MAX_LEN = 192
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8

EPOCHS = 1

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

cf = transformers.RobertaConfig.from_pretrained(MODEL_NAME)
HF_TOKENIZER = transformers.RobertaTokenizerFast.from_pretrained(MODEL_NAME)

tok_backend = HF_TOKENIZER._tokenizer  # tokenizers.Tokenizer
TOKENIZER = tok_backend

special_added = ["[POS]", "[NEG]", "[NEU]"]
HF_TOKENIZER.add_tokens(special_added, special_tokens=True)
sentiment_token_map = {
    "positive": HF_TOKENIZER.convert_tokens_to_ids("[POS]"),
    "negative": HF_TOKENIZER.convert_tokens_to_ids("[NEG]"),
    "neutral": HF_TOKENIZER.convert_tokens_to_ids("[NEU]"),
}

print("Sentiment token ids:", sentiment_token_map)




## === cell 3
class Tweet_Dataset:
    def __init__(self, raw_text, sentiment, selected_text):
        self.raw_text = raw_text
        self.sentiment = sentiment
        self.selected_text = selected_text

        self.tokenizer = TOKENIZER
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        dataset = self.preprocess_text(
            self.raw_text[idx], self.sentiment[idx], self.selected_text[idx]
        )
        return dataset

    def preprocess_text(self, tweet_text, sent, s_text):
        tweet_text = " ".join(str(tweet_text).split())
        s_text = " ".join(str(s_text).split())
        s_len = len(s_text)

        idx_start, idx_end = None, None
        if len(s_text) > 0:
            for idx in (i for i, e in enumerate(tweet_text) if e == s_text[0]):
                if tweet_text[idx : idx + s_len] == s_text:
                    idx_start = idx
                    idx_end = idx + s_len
                    break

        target_position_list = [0] * len(tweet_text)
        if idx_start is not None and idx_end is not None:
            for char_idx in range(idx_start, idx_end):
                target_position_list[char_idx] = 1

        encode_tweet = self.tokenizer.encode(tweet_text)
        input_ids_ori, input_offsets = encode_tweet.ids, encode_tweet.offsets

        target_ids = []
        for e, (o1, o2) in enumerate(input_offsets):
            if sum(target_position_list[o1:o2]) > 0:
                target_ids.append(e)

        if len(target_ids) == 0:
            tar_st = 0
            tar_end = 0
        else:
            tar_st = target_ids[0]
            tar_end = target_ids[-1]

        s_id = sentiment_token_map.get(sent, sentiment_token_map["neutral"])
        input_ids = (
            [HF_TOKENIZER.cls_token_id]
            + [s_id]
            + [HF_TOKENIZER.sep_token_id]
            + [HF_TOKENIZER.sep_token_id]
            + input_ids_ori
            + [HF_TOKENIZER.sep_token_id]
        )

        input_mask = [1] * len(input_ids)

        input_type_ids = [0] * len(input_ids)  # kept for API compatibility
        input_offsets = [(0, 0)] * 4 + input_offsets + [(0, 0)]

        tar_st += 4
        tar_end += 4

        padding_len = self.max_len - len(input_ids)
        if padding_len > 0:
            pad_id = HF_TOKENIZER.pad_token_id
            input_ids = input_ids + [pad_id] * padding_len
            input_mask = input_mask + [0] * padding_len
            input_type_ids = input_type_ids + [0] * padding_len
            input_offsets = input_offsets + [(0, 0)] * padding_len
        else:
            input_ids = input_ids[: self.max_len]
            input_mask = input_mask[: self.max_len]
            input_type_ids = input_type_ids[: self.max_len]
            input_offsets = input_offsets[: self.max_len]
            tar_st = min(tar_st, self.max_len - 1)
            tar_end = min(tar_end, self.max_len - 1)

        return {
            "ids": torch.tensor(input_ids, dtype=torch.long),
            "mask": torch.tensor(input_mask, dtype=torch.long),
            "token_type_ids": torch.tensor(input_type_ids, dtype=torch.long),
            "target_start": torch.tensor(tar_st, dtype=torch.long),
            "target_end": torch.tensor(tar_end, dtype=torch.long),
            "tweet": tweet_text,
            "sentiment": sent,
            "selected_text": s_text,
            "offsets": torch.tensor(input_offsets, dtype=torch.long),
        }


def create_data_loader(dataset, use_gpu=True):
    tr_df, val_df = train_test_split(
        dataset, test_size=0.1, stratify=dataset["sentiment"], random_state=42
    )

    tr_data = Tweet_Dataset(
        raw_text=tr_df["text"].values,
        sentiment=tr_df["sentiment"].values,
        selected_text=tr_df["selected_text"].values,
    )
    val_data = Tweet_Dataset(
        raw_text=val_df["text"].values,
        sentiment=val_df["sentiment"].values,
        selected_text=val_df["selected_text"].values,
    )

    tr_ = torch.utils.data.DataLoader(
        tr_data,
        batch_size=TRAIN_BATCH_SIZE,
        pin_memory=use_gpu,
        shuffle=True,
        num_workers=2,
    )
    val_ = torch.utils.data.DataLoader(
        val_data,
        batch_size=VALID_BATCH_SIZE,
        pin_memory=use_gpu,
        shuffle=False,
        num_workers=2,
    )
    return tr_, val_


tr_loader, val_loader = create_data_loader(
    train_data, use_gpu=torch.cuda.is_available()
)
print("train batches:", len(tr_loader), "valid batches:", len(val_loader))




## === cell 4
class HighWay_Model(nn.Module):
    def __init__(self, input_size, gate_bias=-1):
        super().__init__()
        self.normal_layer = nn.Linear(input_size, input_size)
        self.gate_layer = nn.Linear(input_size, input_size)
        self.gate_layer.bias.data.fill_(gate_bias)

    def forward(self, x):
        norm_x = F.relu(self.normal_layer(x))
        gate_x = torch.softmax(self.gate_layer(x), dim=0)
        gate_norm = torch.mul(norm_x, gate_x)
        gate_input = torch.mul((1 - gate_x), x)
        return torch.add(gate_norm, gate_input)


class roBerta_Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = transformers.RobertaModel.from_pretrained(MODEL_NAME, config=cf)
        self.bert.resize_token_embeddings(len(HF_TOKENIZER))

        self.dropout = nn.Dropout(0.3)
        self.fc1 = nn.Linear(cf.hidden_size, 2)
        torch.nn.init.xavier_normal_(self.fc1.weight)

    def forward(self, ids, mask, token_type_ids):
        out = self.bert(input_ids=ids, attention_mask=mask)
        seq_output = out.last_hidden_state  # [bs, seq, hid]
        h_vec = self.dropout(self.fc1(seq_output))
        st_logits, end_logits = h_vec.split(1, dim=-1)
        return st_logits.squeeze(-1), end_logits.squeeze(-1)


model = roBerta_Model().to(device)
print("Model ready on", device)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
def loss_fn(o1, o2, t1, t2):
    loss_fct = nn.CrossEntropyLoss()
    l1 = loss_fct(o1, t1)
    l2 = loss_fct(o2, t2)
    return l1 + l2


def jaccard(str1, str2):
    a = set(str(str1).lower().split())
    b = set(str(str2).lower().split())
    c = a.intersection(b)
    denom = len(a) + len(b) - len(c)
    return float(len(c)) / denom if denom != 0 else 0.0


class AverageMeter(object):
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def train_fn(data_loader, model, opt, device):
    model.train()
    losses = AverageMeter()
    tk0 = tqdm(data_loader, total=len(data_loader))
    for _, data in enumerate(tk0):
        ids = data["ids"].to(device)
        token_type_ids = data["token_type_ids"].to(device)
        mask = data["mask"].to(device)
        target_start = data["target_start"].to(device)
        target_end = data["target_end"].to(device)

        opt.zero_grad(set_to_none=True)
        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)

        loss = loss_fn(o1, o2, target_start, target_end)
        loss.backward()
        opt.step()

        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg)


@torch.no_grad()
def eval_fn(data_loader, model, device, test=False):
    model.eval()
    losses = AverageMeter()
    jac_vals = AverageMeter()
    fin_selected_text = []
    tk0 = tqdm(data_loader, total=len(data_loader))

    for _, data in enumerate(tk0):
        ids = data["ids"].to(device, dtype=torch.long)
        token_type_ids = data["token_type_ids"].to(device, dtype=torch.long)
        mask = data["mask"].to(device, dtype=torch.long)

        target_start = data["target_start"].to(device, dtype=torch.long)
        target_end = data["target_end"].to(device, dtype=torch.long)

        tweet = data["tweet"]
        sentiment = data["sentiment"]
        selected_text = data["selected_text"]

        offsets = data["offsets"].cpu().numpy()

        o1, o2 = model(ids=ids, mask=mask, token_type_ids=token_type_ids)
        loss = loss_fn(o1, o2, target_start, target_end)

        o_start = torch.softmax(o1, dim=1).cpu().numpy()
        o_end = torch.softmax(o2, dim=1).cpu().numpy()

        jacc_scores = []
        batch_target_outputs = []

        for idx, batch_tweet in enumerate(tweet):
            batch_selected_text = selected_text[idx]
            sent_val = sentiment[idx]
            offset_var = offsets[idx]

            idx_st = int(np.argmax(o_start[idx, :]))
            idx_end = int(np.argmax(o_end[idx, :]))
            if idx_st > idx_end:
                idx_end = idx_st

            filter_output = ""
            if sent_val == "neutral" or len(batch_tweet.split()) < 2:
                filter_output = batch_tweet
            else:
                for ix in range(idx_st, idx_end + 1):
                    o1_, o2_ = offset_var[ix]
                    filter_output += batch_tweet[o1_:o2_]
                    if (ix + 1) < len(offset_var) and offset_var[ix][1] < offset_var[
                        ix + 1
                    ][0]:
                        filter_output += " "

            jac = jaccard(batch_selected_text.strip(), filter_output.strip())
            jacc_scores.append(jac)
            batch_target_outputs.append(filter_output.strip())

        jac_vals.update(float(np.mean(jacc_scores)), ids.size(0))
        losses.update(loss.item(), ids.size(0))
        tk0.set_postfix(loss=losses.avg, jaccard=jac_vals.avg)

        fin_selected_text += batch_target_outputs

    return jac_vals.avg, fin_selected_text




## === cell 6
no_decay = ["bias", "LayerNorm.bias", "LayerNorm.weight"]
opt_params = [
    {
        "params": [
            p
            for n, p in model.named_parameters()
            if not any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
    {
        "params": [
            p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)
        ],
        "weight_decay": 0.0,
    },
]

optimizer = transformers.AdamW(opt_params, lr=3e-5)
print("Optimizer ready")



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/4052260572.py in <cell line: 0>()
     17 ]
     18 
---> 19 optimizer = transformers.AdamW(opt_params, lr=3e-5)
     20 print("Optimizer ready")
     21 

/usr/local/lib/python3.11/dist-packages/transformers/utils/import_utils.py in __getattr__(self, name)
   2173 
   2174             if value is None:
-> 2175                 raise AttributeError(f"module {self.__name__} has no attribute {name}")
   2176 
   2177         setattr(self, name, value)

AttributeError: module transformers has no attribute AdamW

## === cell 7
best_j = -1.0
for epoch in range(EPOCHS):
    print(f"\nEPOCH {epoch+1}/{EPOCHS}")
    train_fn(tr_loader, model, optimizer, device)
    j, _ = eval_fn(val_loader, model, device)
    print("valid jaccard:", j)
    if j > best_j:
        best_j = j
        torch.save(model.state_dict(), "best_model.pt")

print("Best valid jaccard:", best_j)

model.load_state_dict(torch.load("best_model.pt", map_location=device))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1776587977.py in <cell line: 0>()
      3 for epoch in range(EPOCHS):
      4     print(f"\nEPOCH {epoch+1}/{EPOCHS}")
----> 5     train_fn(tr_loader, model, optimizer, device)
      6     j, _ = eval_fn(val_loader, model, device)
      7     print("valid jaccard:", j)

NameError: name 'optimizer' is not defined

## === cell 8
ts_data = Tweet_Dataset(
    raw_text=test_data["text"].values,
    sentiment=test_data["sentiment"].values,
    selected_text=test_data["text"].values,  # dummy for test; offsets still computed
)
ts_dataloader = torch.utils.data.DataLoader(
    ts_data, batch_size=16, pin_memory=False, shuffle=False, num_workers=2
)

_, fin_output = eval_fn(ts_dataloader, model, device, test=True)
print("Preds:", len(fin_output))




## === cell 9
def post_process(selected):
    return " ".join(str(selected).split())


sub = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
sub["selected_text"] = fin_output
sub["selected_text"] = sub["selected_text"].apply(post_process)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)

print("Wrote", out_path)
print(sub.head())
