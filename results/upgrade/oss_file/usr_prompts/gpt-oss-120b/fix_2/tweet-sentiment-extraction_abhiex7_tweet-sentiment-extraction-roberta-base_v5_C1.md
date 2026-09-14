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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import random
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.nn import functional as F
from tqdm.autonotebook import tqdm
from sklearn.model_selection import StratifiedKFold
import tokenizers
import transformers
from transformers import get_linear_schedule_with_warmup
from transformers import RobertaModel, RobertaConfig
from sklearn.utils import shuffle
import os

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


def seedall(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = True




## === cell 2
MAX_LEN = 192
PATH = "/kaggle/input/roberta/"
ROBERTAFOLD = "/kaggle/input/robertafolds/"
VOCAB_FILE = PATH + "vocab.json"
CONFIG = PATH + "config.json"

TOKENIZER = tokenizers.ByteLevelBPETokenizer(
    VOCAB_FILE, PATH + "merges.txt", lowercase=True
)

RobertaConf = RobertaConfig.from_pretrained(CONFIG, output_hidden_states=True)
MODEL = RobertaModel.from_pretrained(PATH, config=RobertaConf)

EPOCHS = 3
BATCH_SIZE = 32
SEED = 42
DROPOUT = 0.1
LEARNING_RATE = 2e-5
sentiment_ids = {"positive": 1313, "negative": 2430, "neutral": 7974}
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
Exception                                 Traceback (most recent call last)
/tmp/ipykernel_55/1891575392.py in <cell line: 0>()
      6 
      7 # Fixed tokenizer creation – ByteLevelBPETokenizer expects positional args
----> 8 TOKENIZER = tokenizers.ByteLevelBPETokenizer(
      9     VOCAB_FILE, PATH + "merges.txt", lowercase=True
     10 )

/usr/local/lib/python3.11/dist-packages/tokenizers/implementations/byte_level_bpe.py in __init__(self, vocab, merges, add_prefix_space, lowercase, dropout, unicode_normalizer, continuing_subword_prefix, end_of_word_suffix, trim_offsets)
     28         if vocab is not None and merges is not None:
     29             tokenizer = Tokenizer(
---> 30                 BPE(
     31                     vocab,
     32                     merges,

Exception: Error while initializing BPE: No such file or directory (os error 2)

## === cell 3
seedall(SEED)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/894901322.py in <cell line: 0>()
----> 1 seedall(SEED)
      2 
      3 

NameError: name 'SEED' is not defined

## === cell 4
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df
        self.max_len = MAX_LEN
        self.labeled = "selected_text" in df
        self.tokenizer = TOKENIZER

    def __getitem__(self, index):
        data = {}
        row = self.df.iloc[index]

        ids, masks, tweet, offsets = self.get_input_data(row)
        data["ids"] = ids
        data["masks"] = masks
        data["tweet"] = tweet
        data["offsets"] = offsets

        if self.labeled:
            start_idx, end_idx = self.get_target_idx(row, tweet, offsets)
            data["start_idx"] = start_idx
            data["end_idx"] = end_idx

        return data

    def __len__(self):
        return len(self.df)

    def get_input_data(self, row):
        tweet = " " + " ".join(row.text.lower().split())
        encoding = self.tokenizer.encode(tweet)
        sentiment_id = self.tokenizer.encode(row.sentiment).ids
        ids = [0] + sentiment_id + [2, 2] + encoding.ids + [2]
        offsets = [(0, 0)] * 4 + encoding.offsets + [(0, 0)]

        pad_len = self.max_len - len(ids)
        if pad_len > 0:
            ids += [1] * pad_len
            offsets += [(0, 0)] * pad_len

        ids = torch.tensor(ids, dtype=torch.long)
        masks = torch.where(
            ids != 1,
            torch.tensor(1, dtype=torch.long),
            torch.tensor(0, dtype=torch.long),
        )
        offsets = torch.tensor(offsets, dtype=torch.long)

        return ids, masks, tweet, offsets

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
        for j, (offset1, offset2) in enumerate(offsets):
            if sum(char_targets[offset1:offset2]) > 0:
                target_idx.append(j)

        start_idx = target_idx[0]
        end_idx = target_idx[-1]
        return start_idx, end_idx


def get_train_val_loaders(df, train_idx, val_idx, batch_size=8):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]

    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df),
        batch_size=batch_size,
        shuffle=True,
        num_workers=2,
        drop_last=True,
    )

    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2
    )

    return {"train": train_loader, "val": val_loader}


def get_test_loader(df, batch_size=32):
    return torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
    )




## === cell 5
class TweetModel(nn.Module):
    def __init__(self):
        super(TweetModel, self).__init__()
        self.roberta = MODEL
        self.dropout = nn.Dropout(DROPOUT)
        m_size = 128
        self.qa_outputs1c = nn.Conv1d(RobertaConf.hidden_size, m_size, 2)
        self.qa_outputs2c = nn.Conv1d(RobertaConf.hidden_size, m_size, 2)
        self.qa_outputs1 = nn.Linear(m_size, 1)
        self.qa_outputs2 = nn.Linear(m_size, 1)

    def forward(self, input_ids, attention_mask):
        out = self.roberta(input_ids, attention_mask)
        s_out = self.dropout(out[0])
        s_out = torch.nn.functional.pad(s_out.transpose(1, 2), (1, 0))
        out1 = self.qa_outputs1c(s_out).transpose(1, 2)
        out2 = self.qa_outputs2c(s_out).transpose(1, 2)
        start_logits = self.qa_outputs1(self.dropout(out1)).squeeze(-1)
        end_logits = self.qa_outputs2(self.dropout(out2)).squeeze(-1)
        return start_logits, end_logits




## === cell 6
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    ce_loss = nn.CrossEntropyLoss()
    start_loss = ce_loss(start_logits, start_positions)
    end_loss = ce_loss(end_logits, end_positions)
    return start_loss + end_loss




## === cell 7
def get_selected_text(text, start_idx, end_idx, offsets):
    selected_text = ""
    for ix in range(start_idx, end_idx + 1):
        selected_text += text[offsets[ix][0] : offsets[ix][1]]
        if (ix + 1) < len(offsets) and offsets[ix][1] < offsets[ix + 1][0]:
            selected_text += " "
    return selected_text


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))


def compute_jaccard_score(text, start_idx, end_idx, start_logits, end_logits, offsets):
    start_pred = np.argmax(start_logits)
    end_pred = np.argmax(end_logits)
    if start_pred > end_pred:
        pred = text
    else:
        pred = get_selected_text(text, start_pred, end_pred, offsets)
    true = get_selected_text(text, start_idx, end_idx, offsets)
    return jaccard(true, pred)




## === cell 8
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
                offsets = data["offsets"].numpy()
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

                    start_idx_np = start_idx.cpu().detach().numpy()
                    end_idx_np = end_idx.cpu().detach().numpy()
                    start_logits_np = (
                        torch.softmax(start_logits, dim=1).cpu().detach().numpy()
                    )
                    end_logits_np = (
                        torch.softmax(end_logits, dim=1).cpu().detach().numpy()
                    )

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




## === cell 9
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
        model, dataloaders, loss_fn, optimizer, EPOCHS, f"roberta_fold{fold}.pth"
    )
    break



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3621278959.py in <cell line: 0>()
      1 # Use a single fold to keep runtime reasonable
----> 2 skf = StratifiedKFold(n_splits=2, shuffle=True, random_state=SEED)
      3 for fold, (train_idx, val_idx) in enumerate(
      4     skf.split(train_df, train_df.sentiment), start=1
      5 ):

NameError: name 'SEED' is not defined

## === cell 10
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)
test_loader = get_test_loader(test_df)

predictions = []
models = []

model = TweetModel()
model_path = "roberta_fold1.pth"
if os.path.exists(model_path):
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
else:
    print("Warning: trained model file not found, using untrained model.")
model.to(DEVICE)
model.eval()
models.append(model)

for data in test_loader:
    ids = data["ids"].to(DEVICE)
    masks = data["masks"].to(DEVICE)
    tweet = data["tweet"]
    offsets = data["offsets"].numpy()

    with torch.no_grad():
        start_logits, end_logits = models[0](ids, masks)
        start_logits = torch.softmax(start_logits, dim=1).cpu().detach().numpy()
        end_logits = torch.softmax(end_logits, dim=1).cpu().detach().numpy()

    for i in range(len(ids)):
        start_pred = np.argmax(start_logits[i])
        end_pred = np.argmax(end_logits[i])
        if start_pred > end_pred:
            pred = tweet[i]
        else:
            pred = get_selected_text(tweet[i], start_pred, end_pred, offsets[i])
        predictions.append(pred)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3025897067.py in <cell line: 0>()
      1 test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
      2 test_df["text"] = test_df["text"].astype(str)
----> 3 test_loader = get_test_loader(test_df)
      4 
      5 predictions = []

/tmp/ipykernel_55/1975579134.py in get_test_loader(df, batch_size)
     96 def get_test_loader(df, batch_size=32):
     97     return torch.utils.data.DataLoader(
---> 98         TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
     99     )
    100 

/tmp/ipykernel_55/1975579134.py in __init__(self, df)
      4         self.max_len = MAX_LEN
      5         self.labeled = "selected_text" in df
----> 6         self.tokenizer = TOKENIZER
      7 
      8     def __getitem__(self, index):

NameError: name 'TOKENIZER' is not defined

## === cell 11
sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
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
print("Submission file written to submission.csv")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/590860064.py in <cell line: 0>()
      1 sub_df = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")
----> 2 sub_df["selected_text"] = predictions
      3 sub_df["selected_text"] = sub_df["selected_text"].apply(
      4     lambda x: x.replace("!!!!", "!") if len(str(x).split()) == 1 else x
      5 )

NameError: name 'predictions' is not defined
