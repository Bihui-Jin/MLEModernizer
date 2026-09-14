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
Predict the sentiment of phrases.

## Metric
Classification accuracy.

## Submission Format
For each phrase in the test set, predict a label for the sentiment. Your submission should have a header and look like the following:

```
PhraseId,Sentiment
156061,2
156062,2
156063,2
...
```

## Dataset
The dataset is comprised of tab-separated files with phrases. Each phrase has a PhraseId. Each sentence has a SentenceId.

The sentiment labels are:

0 - negative

1 - somewhat negative

2 - neutral

3 - somewhat positive

4 - positive

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        input/
            description.md (72 lines)
            sampleSubmission.csv (46819 lines)
            sampleSubmission.csv.zip (146.0 kB)
            test.tsv (46819 lines)
            test.tsv.zip (1.1 MB)
            train.tsv (109243 lines)
            train.tsv.zip (2.7 MB)
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
        working/
            movie-review-sentiment-analysis-kernels-only/
                description.md (72 lines)
                sampleSubmission.csv (46819 lines)
                ... and 5 other files
                movie-review-sentiment-analysis-kernels-only/
```

-> data/movie-review-sentiment-analysis-kernels-only/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> data/sampleSubmission.csv has 46818 rows and 2 columns.
Here is some information about the columns:
PhraseId (int64) has range: 29.00 - 156030.00, 0 nan values
Sentiment (int64) has 1 unique values: [2]

-> (stopped after 10 files for performance)

# 5. Target score

0.65072

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.metrics import accuracy_score



## === cell 1
possible_paths = [
    "./input",  # typical Kaggle notebook root
    "./kaggle/input/movie-review-sentiment-analysis-kernels-only",
    "./kaggle/data/movie-review-sentiment-analysis-kernels-only",
    "./data/movie-review-sentiment-analysis-kernels-only",
    "./data",  # fallback
]
base_path = None
for p in possible_paths:
    if os.path.isdir(p):
        base_path = p
        break
if base_path is None:
    raise FileNotFoundError("Could not locate the dataset directory.")

train_path = os.path.join(base_path, "train.tsv")
test_path = os.path.join(base_path, "test.tsv")

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2158199050.py in <cell line: 0>()
     13         break
     14 if base_path is None:
---> 15     raise FileNotFoundError("Could not locate the dataset directory.")
     16 
     17 train_path = os.path.join(base_path, "train.tsv")

FileNotFoundError: Could not locate the dataset directory.

## === cell 2
train_rows, val_rows = train_test_split(
    train_df,
    test_size=0.01,
    random_state=0,
    stratify=train_df["Sentiment"],
)

print(
    f"Split the data: {len(train_rows)} training rows, {len(val_rows)} validation rows."
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/327273706.py in <cell line: 0>()
      1 train_rows, val_rows = train_test_split(
----> 2     train_df,
      3     test_size=0.01,
      4     random_state=0,
      5     stratify=train_df["Sentiment"],

NameError: name 'train_df' is not defined

## === cell 3
vectorizer = TfidfVectorizer(
    lowercase=True,
    stop_words="english",
    ngram_range=(1, 2),
    max_features=50000,
)

X_train = vectorizer.fit_transform(train_rows["Phrase"])
y_train = train_rows["Sentiment"]

clf = LinearSVC(random_state=0)
clf.fit(X_train, y_train)

X_val = vectorizer.transform(val_rows["Phrase"])
y_val = val_rows["Sentiment"]
val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.5f}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2744200218.py in <cell line: 0>()
      6 )
      7 
----> 8 X_train = vectorizer.fit_transform(train_rows["Phrase"])
      9 y_train = train_rows["Sentiment"]
     10 

NameError: name 'train_rows' is not defined

## === cell 4
X_test = vectorizer.transform(test_df["Phrase"])
test_pred = clf.predict(X_test)

submission = pd.DataFrame(
    {
        "PhraseId": test_df["PhraseId"],
        "Sentiment": test_pred.astype(int),
    }
)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/472969523.py in <cell line: 0>()
----> 1 X_test = vectorizer.transform(test_df["Phrase"])
      2 test_pred = clf.predict(X_test)
      3 
      4 submission = pd.DataFrame(
      5     {

NameError: name 'test_df' is not defined
