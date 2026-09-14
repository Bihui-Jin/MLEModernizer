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

0.5016838908195496

# 6. Current score

0.42769

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.39562) has done: 'I removed the unavailable Fairseq/BPE dependencies and the faulty local‑path model loading, and replaced them with a standard HuggingFace Roberta tokenizer (the public BERTweet model) just to keep the preprocessing pipeline consistent.  
The script now simply uses the full tweet text as the predicted selected_text, which guarantees a correctly‑formatted CSV submission and avoids the previous import and path errors. This minimal change restores end‑to‑end execution while keeping the original data handling logic.'
- What this solution (achieved 0.42769) has done: 'I guard the tokenizer loading to avoid the TypeError, build a small lookup of frequent selected text snippets per sentiment from the training data, and use this lookup in the prediction function to return a matching phrase when possible (falling back to the whole tweet). This fixes the runtime error and adds a lightweight heuristic that should improve the Jaccard score toward the target while keeping the original logic intact.'

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import warnings
import torch
from nltk.tokenize import TweetTokenizer
from transformers import RobertaTokenizer

warnings.filterwarnings("ignore")
seed = 18
torch.manual_seed(seed)



## === cell 1
tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return "@USER"
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return "HTTPURL"
    elif len(token) == 1:
        return token
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
print(_)



## === cell 3
try:
    roberta_tokenizer = RobertaTokenizer.from_pretrained(
        "vinai/bertweet-base", add_prefix_space=True
    )
except Exception as e:
    print(f"Tokenizer load failed ({e}); proceeding without it.")
    roberta_tokenizer = None



## === cell 4
train_path = "../input/tweet-sentiment-extraction/train.csv"
train_df = pd.read_csv(train_path).astype(str)

train_df["norm_text"] = train_df["text"].apply(normalizeTweet)
train_df["norm_selected"] = train_df["selected_text"].apply(normalizeTweet)

from collections import Counter, defaultdict

sentiment_lookup = defaultdict(list)  # sentiment -> list of (selected_text, count)

for sentiment in train_df["sentiment"].unique():
    subset = train_df[train_df["sentiment"] == sentiment]
    counter = Counter(subset["norm_selected"])
    most_common = [txt for txt, _ in counter.most_common(20)]
    sentiment_lookup[sentiment] = most_common




## === cell 5
def heuristic_predict(row):
    """
    Return a substring that appears in the tweet and is a frequent
    selected_text for the given sentiment. If none matches, fall back
    to the whole (normalised) tweet.
    """
    sentiment = row["sentiment"]
    tweet = row["text"]
    norm_tweet = normalizeTweet(str(tweet))

    for cand in sentiment_lookup.get(sentiment, []):
        if cand in norm_tweet:
            return cand.strip()
    return norm_tweet.strip()




## === cell 6
test_path = "../input/tweet-sentiment-extraction/test.csv"
sample_sub_path = "../input/tweet-sentiment-extraction/sample_submission.csv"

test_df = pd.read_csv(test_path).astype(str)

predictions = test_df.apply(heuristic_predict, axis=1).tolist()

sub_df = pd.read_csv(sample_sub_path)
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

submission_path = "submission.csv"
sub_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
