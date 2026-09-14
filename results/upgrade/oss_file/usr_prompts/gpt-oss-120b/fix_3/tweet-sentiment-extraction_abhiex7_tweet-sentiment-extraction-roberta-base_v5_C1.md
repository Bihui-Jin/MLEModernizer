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

0.690662145614624

# 6. Current score

0.33375

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.33375) has done: 'I replace the failing custom tokenizer and model loading with a standard HuggingFace Roberta tokenizer/model, define the random‑seed constant before it is used, and adjust the dataset to work with the new tokenizer. These fixes resolve the import errors, missing variables, and path issues, allowing the script to run end‑to‑end and produce a proper `submission.csv` while keeping the original model architecture and training logic.'

# 9. Code solution

## === cell 0
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm.autonotebook import tqdm
from sklearn.model_selection import StratifiedKFold
import os

from transformers import (
    RobertaTokenizerFast,
    RobertaModel,
    RobertaConfig,
    get_linear_schedule_with_warmup,
)

SEED = 42

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        pass  # suppress long prints in notebook output




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
train_df["text"] = train_df["text"].astype(str)
train_df["selected_text"] = train_df["selected_text"].astype(str)

MAX_LEN = 192
EPOCHS = 3
BATCH_SIZE = 32
DROPOUT = 0.1
LEARNING_RATE = 2e-5
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

MODEL_NAME = "roberta-base"
tokenizer = RobertaTokenizerFast.from_pretrained(MODEL_NAME, do_lower_case=False)
rob_conf = RobertaConfig.from_pretrained(MODEL_NAME, output_hidden_states=True)
MODEL = RobertaModel.from_pretrained(MODEL_NAME, config=rob_conf)




## === cell 2
def seedall(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


seedall(SEED)




## === cell 3
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.max_len = MAX_LEN
        self.labeled = "selected_text" in df.columns
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        tweet = " " + " ".join(row.text.lower().split())
        enc = self.tokenizer(
            tweet,
            add_special_tokens=False,
            return_offsets_mapping=True,
            truncation=True,
            max_length=self.max_len,
        )
        tweet_ids = enc["input_ids"]
        offsets = enc["offset_mapping"]

        sentiment_ids = self.tokenizer.encode(row.sentiment, add_special_tokens=False)

        ids = (
            [self.tokenizer.cls_token_id]
            + sentiment_ids
            + [
                self.tokenizer.sep_token_id,
                self.tokenizer.sep_token_id,
            ]
            + tweet_ids
            + [self.tokenizer.sep_token_id]
        )

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [self.tokenizer.pad_token_id] * pad_len
            offsets += [(0, 0)] * pad_len
        else:
            ids = ids[: self.max_len]
            offsets = offsets[: self.max_len]

        ids = torch.tensor(ids, dtype=torch.long)
        masks = (ids != self.tokenizer.pad_token_id).long()
        offsets = torch.tensor(offsets, dtype=torch.long)

        out = {
            "ids": ids,
            "masks": masks,
            "tweet": tweet,
            "offsets": offsets,
        }

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            out["start_idx"] = torch.tensor(start_idx, dtype=torch.long)
            out["end_idx"] = torch.tensor(end_idx, dtype=torch.long)

        return out

    def get_target_idx(self, row, tweet, offsets):
        selected_text = " " + " ".join(row.selected_text.lower().split())
        len_st = len(selected_text) - 1
        idx0 = None
        idx1 = None

        for ind in (i for i, e in enumerate(tweet) if e == selected_text[1]):
            if " " + tweet[ind : ind + len_st] == selected_text:
                idx0 = ind
                idx1 = ind + len_st - 1
                break

        char_targets = [0] * len(tweet)
        if idx0 is not None and idx1 is not None:
            for ct in range(idx0, idx1 + 1):
                char_targets[ct] = 1

        target_idx = []
        for j, (off1, off2) in enumerate(offsets.tolist()):
            if sum(char_targets[off1:off2]) > 0:
                target_idx.append(j)

        if not target_idx:
            target_idx = [0]

        return target_idx[0], target_idx[-1]


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_loader = DataLoader(
        TweetDataset(df.iloc[train_idx]),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )
    val_loader = DataLoader(
        TweetDataset(df.iloc[val_idx]),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )
    return {"train": train_loader, "val": val_loader}


def get_test_loader(df, batch_size=32):
    return DataLoader(
        TweetDataset(df),
        batch_size=batch_size,
        shuffle=False,
        num_workers=2,
    )




## === cell 4
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = MODEL
        self.dropout = nn.Dropout(DROPOUT)
        m_size = 128
        self.qa_outputs1c = nn.Conv1d(
            RobertaConfig.from_pretrained(MODEL_NAME).hidden_size, m_size, 2
        )
        self.qa_outputs2c = nn.Conv1d(
            RobertaConfig.from_pretrained(MODEL_NAME).hidden_size, m_size, 2
        )
        self.qa_outputs1 = nn.Linear(m_size, 1)
        self.qa_outputs2 = nn.Linear(m_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask)
        s_out = self.dropout(out.last_hidden_state)
        s_out = torch.nn.functional.pad(s_out.transpose(1, 2), (1, 0))
        out1 = self.qa_outputs1c(s_out).transpose(1, 2)
        out2 = self.qa_outputs2c(s_out).transpose(1, 2)
        start_logits = self.qa_outputs1(self.dropout(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.dropout(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 5
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce = nn.CrossEntropyLoss()
    loss_start = ce(start_logits, start_positions)
    loss_end = ce(end_logits, end_positions)
    return loss_start + loss_end




## === cell 6
def get_selected_text(text, start_idx, end_idx, offsets):
    sel = ""
    for ix in range(start_idx, end_idx + 1):
        sel += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            sel += " "
    return sel


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return (
        float(len(c)) / (len(a) + len(b) - len(c))
        if (len(a) + len(b) - len(c)) > 0
        else 0.0
    )


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 7
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    model.to(DEVICE)
    for epoch in range(num_epochs):
        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            epoch_loss = 0.0
            epoch_jaccard = 0.0
            for data in dataloaders_dict[phase]:
                ids = data["ids"].to(DEVICE)
                masks = data["masks"].to(DEVICE)
                tweet = data["tweet"]
                offsets = data["offsets"].cpu().numpy()
                start_idx = data["start_idx"].to(DEVICE)
                end_idx = data["end_idx"].to(DEVICE)

                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == "train"):
                    start_logits, end_logits = model(ids, masks)
                    loss = criterion(start_logits, end_logits, start_idx, end_idx)
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                    epoch_loss += loss.item() * ids.size(0)

                    start_idx_np = start_idx.cpu().numpy()
                    end_idx_np = end_idx.cpu().numpy()
                    start_logits_np = torch.softmax(start_logits, dim=1).cpu().numpy()
                    end_logits_np = torch.softmax(end_logits, dim=1).cpu().numpy()

                    for i in range(len(ids)):
                        epoch_jaccard += compute_jaccard_score(
                            tweet[i],
                            start_idx_np[i],
                            end_idx_np[i],
                            start_logits_np[i],
                            end_logits_np[i],
                            offsets[i],
                        )

            epoch_loss /= len(dataloaders_dict[phase].dataset)
            epoch_jaccard /= len(dataloaders_dict[phase].dataset)
            print(
                f"Epoch {epoch+1}/{num_epochs} | {phase:^5} | Loss: {epoch_loss:.4f} | Jaccard: {epoch_jaccard:.4f}"
            )
    torch.save(model.state_dict(), filename)




## === cell 8
skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
for fold, (train_idx, val_idx) in enumerate(
    skf.split(train_df, train_df.sentiment), start=1
):
    print(f"Training fold {fold}")
    model = TweetModel()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, betas=(0.9, 0.999)
    )
    dataloaders = get_train_val_loaders(train_df, train_idx, val_idx, BATCH_SIZE)
    train_model(
        model,
        dataloaders,
        loss_fn,
        optimizer,
        EPOCHS,
        f"roberta_fold{fold}.pth",
    )
    break




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/1029150554.py in <cell line: 0>()
      9     )
     10     dataloaders = get_train_val_loaders(train_df, train_idx, val_idx, BATCH_SIZE)
---> 11     train_model(
     12         model,
     13         dataloaders,

/tmp/ipykernel_55/2141704296.py in train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename)
     25                     start_idx_np = start_idx.cpu().numpy()
     26                     end_idx_np = end_idx.cpu().numpy()
---> 27                     start_logits_np = torch.softmax(start_logits, dim=1).cpu().numpy()
     28                     end_logits_np = torch.softmax(end_logits, dim=1).cpu().numpy()
     29 

RuntimeError: Can't call numpy() on Tensor that requires grad. Use tensor.detach().numpy() instead.

## === cell 9
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df)

predictions = []

model = TweetModel()
model_path = "roberta_fold1.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print("Warning: trained model file not found, using untrained model.")
model.to(DEVICE)
model.eval()

for data in test_loader:
    ids = data["ids"].to(DEVICE)
    masks = data["masks"].to(DEVICE)
    tweet = data["tweet"]
    offsets = data["offsets"].cpu().numpy()

    with torch.no_grad():
        start_logits, end_logits = model(ids, masks)
        start_logits = torch.softmax(start_logits, dim=1).cpu().numpy()
        end_logits = torch.softmax(end_logits, dim=1).cpu().numpy()

    for i in range(len(ids)):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)




## === cell 10
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
sub_df["selected_text"] = predictions
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: (
        x.replace("!!!!", "!") if isinstance(x, str) and len(x.split()) == 1 else x
    )
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("..", ".") if isinstance(x, str) and len(x.split()) == 1 else x
)
sub_df["selected_text"] = sub_df["selected_text"].apply(
    lambda x: x.replace("...", ".") if isinstance(x, str) and len(x.split()) == 1 else x
)
sub_df.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")
