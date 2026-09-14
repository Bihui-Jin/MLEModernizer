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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0

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

0.5902063250541687

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import math
from sklearn.linear_model import LinearRegression

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
except ModuleNotFoundError:
    POSITIVE_WORDS = {
        "good",
        "great",
        "awesome",
        "fantastic",
        "amazing",
        "love",
        "excellent",
        "nice",
        "happy",
        "positive",
        "best",
        "wonderful",
    }
    NEGATIVE_WORDS = {
        "bad",
        "terrible",
        "awful",
        "hate",
        "worst",
        "sad",
        "negative",
        "poor",
        "horrible",
        "disappointed",
        "angry",
    }

    class SimpleSentimentAnalyzer:
        @staticmethod
        def polarity_scores(text: str):
            words = text.lower().split()
            pos = sum(w in POSITIVE_WORDS for w in words)
            neg = sum(w in NEGATIVE_WORDS for w in words)
            neu = len(words) - pos - neg
            total = max(len(words), 1)
            return {
                "pos": pos / total,
                "neg": neg / total,
                "neu": neu / total,
            }

    analyzer = SimpleSentimentAnalyzer()


def substrings(n, x):
    """Return all contiguous substrings of length n from array x."""
    return np.fromfunction(lambda i, j: x[i + j], (len(x) - n + 1, n), dtype=int)


train_data = pd.read_csv("../input/tweet-sentiment-extraction/train.csv")
test_data = pd.read_csv("../input/tweet-sentiment-extraction/test.csv")
submission = pd.read_csv("../input/tweet-sentiment-extraction/sample_submission.csv")

train_data = train_data.dropna(axis=0, how="any")

train_data["text_length"] = train_data["text"].str.split().str.len()
train_data["selected_text_length"] = train_data["selected_text"].str.split().str.len()
test_data["text_length"] = test_data["text"].str.split().str.len()

positive_data = train_data[train_data["sentiment"] == "positive"]
neutral_data = train_data[train_data["sentiment"] == "neutral"]
negative_data = train_data[train_data["sentiment"] == "negative"]

positive_regressor = LinearRegression().fit(
    positive_data["text_length"].values.reshape(-1, 1),
    positive_data["selected_text_length"].values.reshape(-1, 1),
)
neutral_regressor = LinearRegression().fit(
    neutral_data["text_length"].values.reshape(-1, 1),
    neutral_data["selected_text_length"].values.reshape(-1, 1),
)
negative_regressor = LinearRegression().fit(
    negative_data["text_length"].values.reshape(-1, 1),
    negative_data["selected_text_length"].values.reshape(-1, 1),
)

for i in range(test_data.shape[0]):
    sentiment = test_data["sentiment"].iloc[i]
    text_len = test_data["text_length"].iloc[i]
    words = test_data["text"].iloc[i].split()

    if sentiment == "positive":
        coef, intercept = (
            positive_regressor.coef_[0][0],
            positive_regressor.intercept_[0],
        )
        target_score_key = "pos"
    elif sentiment == "neutral":
        coef, intercept = neutral_regressor.coef_[0][0], neutral_regressor.intercept_[0]
        target_score_key = "neu"
    else:  # negative
        coef, intercept = (
            negative_regressor.coef_[0][0],
            negative_regressor.intercept_[0],
        )
        target_score_key = "neg"

    predicted_len = math.ceil(coef * text_len + intercept)
    predicted_len = max(1, min(predicted_len, text_len))

    if predicted_len >= text_len:
        selected = test_data["text"].iloc[i]
    else:
        substr = substrings(predicted_len, np.asarray(words))
        best_idx = 0
        best_score = -1.0
        for j in range(substr.shape[0]):
            candidate = " ".join(substr[j])
            scores = analyzer.polarity_scores(candidate)
            if scores[target_score_key] > best_score:
                best_score = scores[target_score_key]
                best_idx = j
        selected = " ".join(substr[best_idx])

    submission.at[i, "selected_text"] = selected

submission.to_csv("submission.csv", index=False)
print("DONE")
