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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sentence-transformers==4.1.0
sklearn-pandas==2.2.0
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

0.7258532643318176

# 6. Current score

0.38352

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.38352) has done: 'I fixed the path handling (removing the undefined `__file__`) and added a lightweight heuristic that chooses, for each tweet, the longest known `selected_text` from the training set that actually appears in the tweet (grouped by sentiment). If no known phrase is found it falls back to using the whole tweet, which matches the required CSV format and improves the Jaccard baseline toward the target score. The script now runs end‑to‑end and writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path
import pandas as pd

base_dir = Path.cwd() / "input" / "tweet-sentiment-extraction"
if not (base_dir / "test.csv").exists():
    base_dir = Path("../input/tweet-sentiment-extraction")
if not (base_dir / "test.csv").exists():
    raise FileNotFoundError(f"Test file not found in expected locations.")

test_path = base_dir / "test.csv"
train_path = base_dir / "train.csv"

test_df = pd.read_csv(test_path)
train_df = pd.read_csv(train_path)



## === cell 1
sentiment_to_phrases = {}
for sentiment, group in train_df.groupby("sentiment"):
    sentiment_to_phrases[sentiment] = list(group["selected_text"].dropna().unique())




## === cell 2
def pick_selected_text(row):
    """
    Return the longest known selected_text (from training) that is a substring of the tweet.
    If none match, return the whole tweet as a safe fallback.
    """
    text = str(row["text"])
    sentiment = row["sentiment"]
    candidates = sentiment_to_phrases.get(sentiment, [])
    best_match = ""
    for cand in candidates:
        if cand and cand in text and len(cand) > len(best_match):
            best_match = cand
    return best_match if best_match else text


test_df["selected_text"] = test_df.apply(pick_selected_text, axis=1)



## === cell 3
submission_path = Path("submission.csv")
test_df[["textID", "selected_text"]].to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
