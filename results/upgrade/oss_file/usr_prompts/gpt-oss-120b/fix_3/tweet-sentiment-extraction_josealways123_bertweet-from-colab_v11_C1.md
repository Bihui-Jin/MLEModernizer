# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5914

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'I remove the failing fairseq/BERTweet file loading and replace the inference pipeline with a simple fallback that returns the whole tweet as the selected text. This fixes the import errors, undefined variables, and ensures a CSV submission is written without changing the overall structure of the notebook.'
- What this solution (achieved 0.5914) has done: 'I add the tweet sentiment to the dataset loader and implement a lightweight rule‑based post‑processing that extracts a short phrase containing a sentiment‑related word (e.g., “good”, “bad”). This keeps the original model untouched while giving predictions that are more focused than the whole tweet, which should raise the Jaccard score toward the target.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import warnings
import torch
from torch import nn
from sklearn.model_selection import StratifiedKFold
from nltk.tokenize import TweetTokenizer
from emoji import demojize
import re

warnings.filterwarnings("ignore")
seed = 18




## === cell 1
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
from types import SimpleNamespace

config = SimpleNamespace(hidden_size=768)




## === cell 4
class TweetDataset(torch.utils.data.Dataset):
    def __init__(self, df):
        self.df = df.reset_index(drop=True)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        return {
            "tweet": row["text"],
            "ids": torch.tensor([0]),  # placeholder
            "masks": torch.tensor([1]),  # placeholder
            "tweets_encoded": row["text"],  # raw text used only for fallback
            "sentiment": (
                row["sentiment"] if "sentiment" in row else ""
            ),  # keep sentiment for post‑processing
        }




## === cell 5
def get_train_val_loaders(df, train_idx, val_idx, batch_size=32):
    train_df = df.iloc[train_idx]
    val_df = df.iloc[val_idx]
    train_loader = torch.utils.data.DataLoader(
        TweetDataset(train_df), batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        TweetDataset(val_df), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return {"train": train_loader, "val": val_loader}




## === cell 6
class BERTweetModel(nn.Module):
    def __init__(self, conf):
        super().__init__()
        self.dummy = nn.Linear(conf.hidden_size, 2)

    def forward(self, input_ids, attention_mask):
        batch = input_ids.shape[0]
        start_logits = torch.zeros(batch, 1)
        end_logits = torch.zeros(batch, 1)
        return start_logits, end_logits




## === cell 7
def loss_fn(start_logits, end_logits, start_positions, end_positions):
    return torch.tensor(0.0)  # placeholder




## === cell 8
def get_selected_text(tweets_encoded, start_idx, end_idx):
    selected_text = ""
    for i, token in enumerate(tweets_encoded.split()[start_idx - 4 : end_idx - 3]):
        selected_text += " " + token
    selected_text = selected_text.replace("@@ ", "").replace("@@", "")
    return selected_text.strip()


def jaccard(str1, str2):
    a = set(str1.lower().split())
    b = set(str2.lower().split())
    c = a.intersection(b)
    return float(len(c)) / (len(a) + len(b) - len(c))




## === cell 9
def train_model(model, dataloaders_dict, criterion, optimizer, num_epochs, filename):
    pass




## === cell 10
num_epochs = 10
batch_size = 32
skf = StratifiedKFold(n_splits=8, shuffle=True, random_state=seed)




## === cell 11
def run(fold):
    pass




## === cell 12
def get_test_loader(df, batch_size=32):
    loader = torch.utils.data.DataLoader(
        TweetDataset(df), batch_size=batch_size, shuffle=False, num_workers=2
    )
    return loader




## === cell 13
def postprocessing(pred, sentiment):
    """
    Simple rule‑based extraction:
    - For positive sentiment, look for a word from a small positive list.
    - For negative sentiment, look for a word from a small negative list.
    - Return a short phrase containing that word (one token before and after if possible).
    - If no cue word is found, fall back to the whole tweet (pred).
    """
    positive_words = {
        "good",
        "great",
        "nice",
        "love",
        "awesome",
        "happy",
        "best",
        "fantastic",
        "excellent",
        "amazing",
        "wonderful",
        "positive",
    }
    negative_words = {
        "bad",
        "worst",
        "hate",
        "terrible",
        "awful",
        "sad",
        "negative",
        "poor",
        "horrible",
        "disappointed",
        "sucks",
    }

    tokens = normalizeTweet(pred).split()
    selected = pred

    if sentiment == "positive":
        cue_set = positive_words
    elif sentiment == "negative":
        cue_set = negative_words
    else:
        cue_set = set()

    if cue_set:
        for idx, tok in enumerate(tokens):
            if tok.lower() in cue_set:
                start = max(idx - 1, 0)
                end = min(idx + 1, len(tokens) - 1)
                selected = " ".join(tokens[start : end + 1])
                break

    return selected




## === cell 14
test_df = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
test_df["text"] = test_df["text"].astype(str)

test_loader = get_test_loader(test_df, batch_size=32)

predictions = []
for data in test_loader:
    tweets = data["tweets_encoded"]
    sentiments = data["sentiment"]
    for txt, snt in zip(tweets, sentiments):
        pred = txt  # fallback: select the whole tweet
        pred = postprocessing(pred, snt)
        predictions.append(pred)




## === cell 15
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
sub_df.head()
