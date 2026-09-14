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

0.20444

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.21458) has done: 'I fixed the missing model files by loading a public `roberta‑base` model and tokenizer from HuggingFace, rewrote the dataset to use the tokenizer’s offset mapping for accurate start/end label creation, updated the model class to use the same architecture (Roberta backbone + linear head) while keeping the original forward signature, and added a short fine‑tuning loop (1 epoch) so the script runs end‑to‑end and writes a valid `submission.csv`. All other logic (loss, Jaccard calculation, post‑processing) is unchanged.'
- What this solution (achieved 0.183) has done: 'I fix the import error for AdamW, improve the prediction logic to choose a valid end index after the start index, simplify the post‑processing (remove the set‑based de‑duplication which hurts Jaccard), and extend training to three epochs to boost performance while keeping the original model architecture unchanged.'
- What this solution (achieved 0.20444) has done: 'I fixed the import error that prevented `AdamW` from being defined by setting the protobuf implementation flag before loading transformers, and I corrected the prediction logic so the end index is chosen independently of the start index (and forced to be ≥ start). These changes unblock training and improve the extraction quality, moving the Jaccard score toward the target while keeping the original model architecture intact.'

# 9. Code solution

## === cell 0
import os, re, string

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
from tqdm.autonotebook import tqdm
import numpy as np, pandas as pd
from sklearn.model_selection import train_test_split

import torch, torch.nn as nn, torch.nn.functional as F
import transformers
from transformers import (
    RobertaTokenizerFast,
    RobertaConfig,
    RobertaModel,
    AdamW,
)  # AdamW now correctly imported

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_path = r"/kaggle/input/tweet-sentiment-extraction/"
train_data = pd.read_csv(os.path.join(data_path, "train.csv")).dropna()
test_data = pd.read_csv(os.path.join(data_path, "test.csv"))

train_data["text"] = train_data["text"].apply(lambda x: x.strip())
test_data["text"] = test_data["text"].apply(lambda x: x.strip())




## === cell 2
MODEL_NAME = "roberta-base"
MAX_LEN = 192
TRAIN_BATCH_SIZE = 64
VALID_BATCH_SIZE = 8
EPOCHS = 3  # a few more epochs for better performance
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME)
config = RobertaConfig.from_pretrained(MODEL_NAME)




## === cell 3
class Tweet_Dataset(torch.utils.data.Dataset):
    def __init__(self, raw_text, sentiment, selected_text):
        self.raw_text = raw_text
        self.sentiment = sentiment
        self.selected_text = selected_text
        self.tokenizer = tokenizer
        self.max_len = MAX_LEN

    def __len__(self):
        return len(self.raw_text)

    def __getitem__(self, idx):
        tweet = " ".join(str(self.raw_text[idx]).split())
        s_text = " ".join(str(self.selected_text[idx]).split())
        sent = self.sentiment[idx]

        enc = self.tokenizer.encode_plus(
            tweet,
            add_special_tokens=True,
            max_length=self.max_len,
            padding="max_length",
            truncation=True,
            return_offsets_mapping=True,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze()
        attention_mask = enc["attention_mask"].squeeze()
        offsets = enc["offset_mapping"].squeeze().tolist()  # list of (start,end)

        if s_text == "":
            start_idx, end_idx = 0, 0
        else:
            char_start = tweet.find(s_text)
            char_end = char_start + len(s_text)
            token_indices = [
                i
                for i, (s, e) in enumerate(offsets)
                if not (e <= char_start or s >= char_end)
            ]
            if token_indices:
                start_idx, end_idx = token_indices[0], token_indices[-1]
            else:
                start_idx, end_idx = 0, 0

        token_type_ids = torch.zeros_like(input_ids)

        return {
            "ids": input_ids.long(),
            "mask": attention_mask.long(),
            "token_type_ids": token_type_ids.long(),
            "target_start": torch.tensor(start_idx, dtype=torch.long),
            "target_end": torch.tensor(end_idx, dtype=torch.long),
            "tweet": tweet,
            "sentiment": sent,
            "selected_text": s_text,
            "offsets": torch.tensor(offsets, dtype=torch.long),
        }


def create_data_loader(df, batch_size, shuffle=True):
    ds = Tweet_Dataset(
        raw_text=df["text"].values,
        sentiment=df["sentiment"].values,
        selected_text=df["selected_text"].values,
    )
    return torch.utils.data.DataLoader(
        ds, batch_size=batch_size, shuffle=shuffle, pin_memory=True
    )




## === cell 4
class RobertaSpanModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.bert = RobertaModel.from_pretrained(MODEL_NAME, config=config)
        self.dropout = nn.Dropout(0.3)
        self.fc = nn.Linear(self.bert.config.hidden_size, 2)  # start & end logits

    def forward(self, ids, mask, token_type_ids):
        seq_output = self.bert(input_ids=ids, attention_mask=mask)[0]
        logits = self.fc(self.dropout(seq_output))
        start_logits, end_logits = logits.split(1, dim=-1)
        return start_logits.squeeze(-1), end_logits.squeeze(-1)




## === cell 5
def loss_fn(start_logits, end_logits, start_labels, end_labels):
    loss_fct = nn.CrossEntropyLoss()
    l1 = loss_fct(start_logits, start_labels)
    l2 = loss_fct(end_logits, end_labels)
    return l1 + l2


def jaccard(str1, str2):
    a, b = set(str1.lower().split()), set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


class AverageMeter:
    def __init__(self):
        self.reset()

    def reset(self):
        self.sum = 0.0
        self.count = 0
        self.avg = 0.0

    def update(self, val, n=1):
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count




## === cell 6
def predict(loader, model, device):
    model.eval()
    predictions = []
    tk0 = tqdm(loader, total=len(loader))
    with torch.no_grad():
        for data in tk0:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            offsets = data["offsets"].cpu().numpy()
            tweets = data["tweet"]
            start_logits, end_logits = model(ids, mask, token_type_ids)

            start_probs = torch.softmax(start_logits, dim=1).cpu().numpy()
            end_probs = torch.softmax(end_logits, dim=1).cpu().numpy()

            for i, tweet in enumerate(tweets):
                start_idx = np.argmax(start_probs[i])
                end_idx = np.argmax(end_probs[i])
                if end_idx < start_idx:
                    end_idx = start_idx
                selected = ""
                for idx in range(start_idx, end_idx + 1):
                    s, e = offsets[i][idx]
                    selected += tweet[s:e]
                    if (
                        idx + 1 < len(offsets[i])
                        and offsets[i][idx][1] < offsets[i][idx + 1][0]
                    ):
                        selected += " "
                predictions.append(selected.strip())
    return predictions




## === cell 7
train_df, val_df = train_test_split(
    train_data, test_size=0.1, stratify=train_data["sentiment"], random_state=42
)

train_loader = create_data_loader(train_df, TRAIN_BATCH_SIZE, shuffle=True)
val_loader = create_data_loader(val_df, VALID_BATCH_SIZE, shuffle=False)

model = RobertaSpanModel().to(device)

no_decay = ["bias", "LayerNorm.weight"]
optimizer_grouped_parameters = [
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
optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, patience=1, verbose=False
)

for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch+1}/{EPOCHS}")
    train_one_epoch(train_loader, model, optimizer, device)
    model.eval()
    val_losses = AverageMeter()
    with torch.no_grad():
        for data in val_loader:
            ids = data["ids"].to(device)
            mask = data["mask"].to(device)
            token_type_ids = data["token_type_ids"].to(device)
            start_labels = data["target_start"].to(device)
            end_labels = data["target_end"].to(device)
            start_logits, end_logits = model(ids, mask, token_type_ids)
            loss = loss_fn(start_logits, end_logits, start_labels, end_labels)
            val_losses.update(loss.item(), ids.size(0))
    print(f"Validation loss: {val_losses.avg:.4f}")
    scheduler.step(val_losses.avg)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3542673697.py in <cell line: 0>()
     25     },
     26 ]
---> 27 optimizer = AdamW(optimizer_grouped_parameters, lr=3e-5)
     28 scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
     29     optimizer, patience=1, verbose=False

NameError: name 'AdamW' is not defined

## === cell 8
test_dataset = Tweet_Dataset(
    raw_text=test_data["text"].values,
    sentiment=test_data["sentiment"].values,
    selected_text=test_data["text"].values,  # placeholder, not used for inference
)
test_loader = torch.utils.data.DataLoader(
    test_dataset, batch_size=16, shuffle=False, pin_memory=True
)

test_predictions = predict(test_loader, model, device)


def post_process(selected):
    return selected.strip()


sub = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
sub["selected_text"] = [post_process(p) for p in test_predictions]
sub.to_csv("submission.csv", index=False)
print("Submission written to submission.csv")
