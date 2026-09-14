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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
sklearn-pandas==2.2.0

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

0.5092469679599348

# 6. Current score

0.62397

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.62397) has done: 'I replace the broken file‑reading code with a full end‑to‑end pipeline: load the TSV data, split the training set, vectorize the text using TF‑IDF, train a multinomial Logistic Regression classifier, report validation accuracy, predict the test set, and finally write a correctly‑formatted `submission.csv`. This fixes the FileNotFoundError, creates a valid submission file, and adds a simple model expected to reach the target accuracy.'

# 9. Code solution

## === cell 0
import os, glob
import numpy as np
import pandas as pd

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
train_path = "/kaggle/input/train.tsv"
test_path = "/kaggle/input/test.tsv"

train_df = pd.read_csv(train_path, sep="\t")
test_df = pd.read_csv(test_path, sep="\t")

print("Train shape:", train_df.shape)
print("Test shape :", test_df.shape)



## === cell 2
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

X = train_df["Phrase"].astype(str)
y = train_df["Sentiment"].astype(int)

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

vectorizer = TfidfVectorizer(
    stop_words="english", ngram_range=(1, 2), max_features=20000
)

X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)



## === cell 3
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

clf = LogisticRegression(
    multi_class="multinomial", solver="lbfgs", max_iter=1000, n_jobs=-1, random_state=42
)

clf.fit(X_train_vec, y_train)

val_pred = clf.predict(X_val_vec)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation Accuracy: {val_acc:.5f}")



## === cell 4
X_test = test_df["Phrase"].astype(str)
X_test_vec = vectorizer.transform(X_test)

test_pred = clf.predict(X_test_vec)

submission = pd.DataFrame({"PhraseId": test_df["PhraseId"], "Sentiment": test_pred})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")
