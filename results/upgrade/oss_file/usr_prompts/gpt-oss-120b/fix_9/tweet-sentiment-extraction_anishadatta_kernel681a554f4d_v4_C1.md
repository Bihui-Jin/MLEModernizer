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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0
spacy==3.8.7
spacy-legacy==3.0.12
spacy-loggers==1.0.5

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

0.3950402438640594

# 6. Current score

0.59324

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.59324) has done: 'The fix removes the missing word‑list files, loads the CSVs with their proper headers, and builds the submission using pandas rather than manual list handling. If the word‑lists are unavailable we fall back to empty lists, and the prediction simply returns the whole tweet (which works for all sentiments and avoids the previous indexing errors). The script now writes a correctly‑formatted `submission.csv` with the required headers.'
- What this solution (achieved 0.13035) has done: 'I modify the prediction step to use only the first token of each tweet instead of the whole text. Returning a shorter snippet typically lowers the Jaccard overlap with the true selected text, moving the score down toward the target value while keeping the overall workflow unchanged.'
- What this solution (achieved 0.59324) has done: 'I replace the naïve “first‑token” prediction with a simple rule‑based approach that looks for sentiment‑specific words from the provided positive/negative word lists. For each tweet we scan its tokens; if a token appears in the corresponding sentiment list we return that token (otherwise we fall back to the whole tweet). This modest heuristic should raise the Jaccard score toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.32844) has done: 'I lower the Jaccard score by returning only the first half of each tweet (instead of the whole tweet or a single token). This shortens the predicted snippet, reducing overlap with the true selected text and moving the metric closer to the target while keeping the overall workflow unchanged.'
- What this solution (achieved 0.32844) has done: 'I keep the existing workflow but improve the fallback prediction: when a sentiment‑specific word list is available but none of its words appear in the tweet, returning the whole tweet gives a higher Jaccard overlap than the previous “first half” heuristic. This small change should raise the score from 0.328 toward the target 0.395 while preserving all core logic.'
- What this solution (achieved 0.59324) has done: 'I keep the overall workflow unchanged but modify the prediction logic so that for sentiments other than “positive” or “negative” (e.g., “neutral”) the model returns the whole tweet instead of only the first half. Returning the full text usually increases the Jaccard overlap with the true selected text, moving the score upward toward the target of 0.395 while preserving all existing components.'
- What this solution (achieved 0.32844) has done: 'I adjust the fallback prediction so that when no sentiment‑specific word is found the code returns the first half of the tweet instead of the whole tweet. This makes the predicted snippet shorter, lowering the Jaccard overlap and moving the score from the current 0.593 down toward the target 0.395 while keeping the overall workflow and word‑list logic unchanged.'
- What this solution (achieved 0.59324) has done: 'I keep the overall workflow unchanged but make the fallback prediction return the full tweet instead of only its first half when no sentiment‑specific word is found. This modest change should increase the Jaccard overlap and move the score upward toward the target 0.395 while preserving the core logic and avoiding any new data files.'

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
import csv
from pathlib import Path


def read_wordlist(path, cols=None):
    try:
        if cols:
            return (
                pd.read_csv(path, usecols=cols, header=None)
                .squeeze()
                .astype(str)
                .tolist()
            )
        else:
            return pd.read_csv(path, header=None).squeeze().astype(str).tolist()
    except FileNotFoundError:
        return []


data_train = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/train.csv")
data_test = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/test.csv")
data_sub = pd.read_csv("/kaggle/input/tweet-sentiment-extraction/sample_submission.csv")

positive = set(read_wordlist("/kaggle/input/sentiment-wordlist/eng_positivewords.csv"))
negative = set(read_wordlist("/kaggle/input/sentiment-wordlist/english_slang.csv"))


def predict_selected(row):
    text = str(row["text"])
    sentiment = str(row["sentiment"]).lower()
    tokens = text.split()
    word_set = None
    if sentiment == "positive":
        word_set = positive
    elif sentiment == "negative":
        word_set = negative

    if word_set:
        for tok in tokens:
            if tok.lower() in word_set:
                return tok
        return text

    return text


pred_selected = data_test.apply(predict_selected, axis=1).tolist()

submission = pd.DataFrame(
    {"textID": data_test["textID"], "selected_text": pred_selected}
)

output_path = Path("/kaggle/working/submission.csv")
submission.to_csv(output_path, index=False, quoting=csv.QUOTE_ALL, encoding="utf-8")
