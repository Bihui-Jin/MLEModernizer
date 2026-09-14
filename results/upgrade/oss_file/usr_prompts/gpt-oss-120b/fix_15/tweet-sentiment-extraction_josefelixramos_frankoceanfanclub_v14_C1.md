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

0.52757

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52757) has done: 'I slightly adjust the length‑capping logic inside `best_substring` so that the chosen span can be a bit longer (up to the original 0.85 limit) by adding a small offset to the median length ratio. This keeps the overall algorithm unchanged while giving it a better chance to capture more of the true selected text, which should raise the Jaccard score toward the target. No other parts of the pipeline are modified.'
- What this solution (achieved 0.52757) has done: 'I slightly relax the length‑capping and expand the fallback windows so the selected span can include a few more surrounding words. This keeps the original exhaustive search but allows longer candidate spans (offset +0.15 instead of +0.10, still capped at 0.85) and makes the fallback sentiment span grab up to two words before and three after the first/last sentiment word, which should modestly boost the Jaccard score toward the target without changing the core algorithm.'

# 9. Code solution

## === cell 0
import os
import csv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from sklearn.linear_model import LinearRegression  # kept for compatibility, not used

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

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    analyzer = SentimentIntensityAnalyzer()
except ModuleNotFoundError:

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
                "compound": (pos - neg) / total,
            }

    analyzer = SimpleSentimentAnalyzer()

print("Files in /kaggle/input:")
for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))




## === cell 1
def substrings(n, words):
    """Return a list of all contiguous substrings of length n from a list of words."""
    if n <= 0 or n > len(words):
        return []
    return [words[i : i + n] for i in range(len(words) - n + 1)]


def median_ratio(df):
    """Median of selected_text length / text length, capped at 0.85."""
    ratios = df["selected_text_length"] / df["text_length"].replace(0, np.nan)
    if ratios.empty or ratios.isna().all():
        return 0.5
    median = np.nanmedian(ratios)
    return min(median, 0.85)


def best_substring(words, sentiment):
    """Find the contiguous word span that best matches the given sentiment."""
    if not words:
        return ""

    best_score = -float("inf")
    best_candidate = " ".join(words)  # fallback to whole text

    for start in range(len(words)):
        for end in range(start + 1, len(words) + 1):
            cand_words = words[start:end]
            candidate = " ".join(cand_words)
            scores = analyzer.polarity_scores(candidate)
            compound = scores.get("compound", 0)
            diff = compound if sentiment == "positive" else -compound
            if diff > best_score:
                best_score = diff
                best_candidate = candidate

    base_ratio = ratio_dict.get(sentiment, 0.5)
    adjusted_ratio = min(base_ratio + 0.15, 0.85)
    max_len = int(adjusted_ratio * len(words))
    max_len = max(max_len, 1)  # ensure at least one word

    cand_len = len(best_candidate.split())
    if cand_len > max_len:
        best_candidate = " ".join(best_candidate.split()[:max_len])

    if best_candidate == " ".join(words) and sentiment != "neutral":
        sentiment_words = POSITIVE_WORDS if sentiment == "positive" else NEGATIVE_WORDS
        lower_words = [w.lower() for w in words]
        indices = [
            i
            for i, w in enumerate(lower_words)
            if w.strip(".,!?\"'") in sentiment_words
        ]
        if indices:
            start_idx = max(0, indices[0] - 2)
            end_idx = min(len(words), indices[-1] + 3)
            best_candidate = " ".join(words[start_idx:end_idx])

    return best_candidate


def fallback_sentiment_span(text, sentiment):
    """Return a short span containing a sentiment word when the exhaustive
    search yields the whole tweet."""
    words = text.split()
    sentiment_words = POSITIVE_WORDS if sentiment == "positive" else NEGATIVE_WORDS
    lower_words = [w.lower().strip(".,!?\"'") for w in words]
    matches = [i for i, w in enumerate(lower_words) if w in sentiment_words]
    if not matches:
        return text
    start = max(0, matches[0] - 2)
    end = min(len(words), matches[-1] + 3)
    return " ".join(words[start:end])




## === cell 2
train_path = "/kaggle/input/tweet-sentiment-extraction/train.csv"
test_path = "/kaggle/input/tweet-sentiment-extraction/test.csv"

train_data = pd.read_csv(train_path)
test_data = pd.read_csv(test_path)

train_data = train_data.dropna(axis=0, how="any")
train_data["text_length"] = train_data["text"].str.split().str.len()
train_data["selected_text_length"] = train_data["selected_text"].str.split().str.len()
test_data["text_length"] = test_data["text"].str.split().str.len()

positive_data = train_data[train_data["sentiment"] == "positive"]
neutral_data = train_data[train_data["sentiment"] == "neutral"]
negative_data = train_data[train_data["sentiment"] == "negative"]

ratio_dict = {
    "positive": median_ratio(positive_data),
    "neutral": median_ratio(neutral_data),
    "negative": median_ratio(negative_data),
}



## === cell 3
predictions = []

for i in range(test_data.shape[0]):
    sentiment = test_data["sentiment"].iloc[i]

    if sentiment == "neutral":
        selected = test_data["text"].iloc[i]
    else:
        words = test_data["text"].iloc[i].split()
        selected = best_substring(words, sentiment)

        if selected == test_data["text"].iloc[i]:
            selected = fallback_sentiment_span(test_data["text"].iloc[i], sentiment)

    selected = selected.strip()
    predictions.append(selected)

submission = pd.DataFrame({"textID": test_data["textID"], "selected_text": predictions})

submission.to_csv("/kaggle/working/submission.csv", index=False, quoting=csv.QUOTE_ALL)
print("DONE")
