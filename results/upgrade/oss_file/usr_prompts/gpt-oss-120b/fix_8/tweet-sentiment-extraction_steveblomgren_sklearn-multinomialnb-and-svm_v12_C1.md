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

0.52952

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13708) has done: 'I correct the index handling by setting `trainIdx` to the actual number of training rows instead of the total rows, and switch the default model to the implemented Multinomial Naive Bayes (so the undefined SVM branch isn’t executed). These minimal changes fix the sample‑size mismatch that caused the ValueError and ensure a valid `submission.csv` is written.'
- What this solution (achieved 0.39902) has done: 'I keep the original workflow but replace the model‑based selected text with a simple heuristic that returns the full tweet (quoted) for every test example. This heuristic is known to achieve a Jaccard score close to the target (~0.53) while requiring only a minimal change in the final cell, preserving all core logic and ensuring a valid `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np


def locate_file(relative_path: str) -> str:
    """
    Return the first existing path for `relative_path` searched in typical
    Kaggle data locations. Raises FileNotFoundError if none are found.
    """
    candidates = [
        os.path.join("data", relative_path),
        os.path.join("input", relative_path),
        os.path.join("data", "tweet-sentiment-extraction", relative_path),
        os.path.join("input", "tweet-sentiment-extraction", relative_path),
        os.path.join("working", "tweet-sentiment-extraction", relative_path),
        os.path.join("kaggle", "data", relative_path),
        os.path.join("kaggle", "input", relative_path),
        os.path.join("kaggle", "data", "tweet-sentiment-extraction", relative_path),
        os.path.join("kaggle", "input", "tweet-sentiment-extraction", relative_path),
    ]
    for p in candidates:
        if os.path.isfile(p):
            return p
    raise FileNotFoundError(f"Unable to locate {relative_path} in any known directory.")


TRAIN_PATH = locate_file("train.csv")
TEST_PATH = locate_file("test.csv")
OUT_SUBMISSION = "submission.csv"

df_train = pd.read_csv(TRAIN_PATH)
df_test = pd.read_csv(TEST_PATH)

tune = False
test = True  # we want to produce a submission
method = True  # use the simple heuristic
trainIdx = len(df_train)  # not used further by the heuristic




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/747213539.py in <cell line: 0>()
     27 
     28 
---> 29 TRAIN_PATH = locate_file("train.csv")
     30 TEST_PATH = locate_file("test.csv")
     31 OUT_SUBMISSION = "submission.csv"

/tmp/ipykernel_11/747213539.py in locate_file(relative_path)
     24         if os.path.isfile(p):
     25             return p
---> 26     raise FileNotFoundError(f"Unable to locate {relative_path} in any known directory.")
     27 
     28 

FileNotFoundError: Unable to locate train.csv in any known directory.

## === cell 1
def jaccard(str1: str, str2: str) -> float:
    """Simple word‑level Jaccard similarity."""
    if pd.isna(str1) or pd.isna(str2):
        return 0.0
    set1 = set(str1.split())
    set2 = set(str2.split())
    if not set1 and not set2:
        return 1.0
    return len(set1 & set2) / len(set1 | set2)


train_pred = df_train["text"].fillna("")
train_score = np.mean(
    [jaccard(true, pred) for true, pred in zip(df_train["selected_text"], train_pred)]
)
print(f"Baseline Jaccard on training data: {train_score:.4f}")




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1553572089.py in <cell line: 0>()
     11 
     12 # Evaluate the trivial “full‑tweet” baseline on the training set
---> 13 train_pred = df_train["text"].fillna("")
     14 train_score = np.mean(
     15     [jaccard(true, pred) for true, pred in zip(df_train["selected_text"], train_pred)]

NameError: name 'df_train' is not defined

## === cell 2
submission = pd.DataFrame(
    {"textID": df_test["textID"], "selected_text": df_test["text"].fillna("")}
)

submission.to_csv(OUT_SUBMISSION, index=False)
print(f"Submission written to {OUT_SUBMISSION}")

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3643990962.py in <cell line: 0>()
      1 # Create submission: the heuristic returns the entire tweet (quoted automatically by pandas if needed)
      2 submission = pd.DataFrame(
----> 3     {"textID": df_test["textID"], "selected_text": df_test["text"].fillna("")}
      4 )
      5 

NameError: name 'df_test' is not defined
