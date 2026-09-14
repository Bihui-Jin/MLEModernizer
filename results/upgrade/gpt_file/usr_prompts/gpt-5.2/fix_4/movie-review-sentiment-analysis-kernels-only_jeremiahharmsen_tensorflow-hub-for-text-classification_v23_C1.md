# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import random

import numpy as np
import pandas as pd

from sklearn import model_selection
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.neural_network import MLPClassifier

np.random.seed(0)
random.seed(0)

SENTIMENT_LABELS = [
    "negative",
    "somewhat negative",
    "neutral",
    "somewhat positive",
    "positive",
]

print("Environment ready (sklearn-based fallback: no TF/Hub required).")




## === cell 1
def add_readable_labels_column(df, sentiment_value_column):
    df["SentimentLabel"] = df[sentiment_value_column].replace(
        range(5), SENTIMENT_LABELS
    )


def get_data(validation_set_ratio=0.1):
    train_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/train.tsv"
    test_path = "/kaggle/input/movie-review-sentiment-analysis-kernels-only/test.tsv"

    if not os.path.exists(train_path):
        train_path = "/kaggle/input/train.tsv"
    if not os.path.exists(test_path):
        test_path = "/kaggle/input/test.tsv"

    train_df = pd.read_csv(train_path, sep="\t")
    test_df = pd.read_csv(test_path, sep="\t")

    add_readable_labels_column(train_df, "Sentiment")

    train_indices, validation_indices = model_selection.train_test_split(
        np.unique(train_df["SentenceId"]),
        test_size=validation_set_ratio,
        random_state=0,
    )

    validation_df = train_df[train_df["SentenceId"].isin(validation_indices)].copy()
    train_df = train_df[train_df["SentenceId"].isin(train_indices)].copy()

    print(
        "Split the training data into %d training and %d validation examples."
        % (len(train_df), len(validation_df))
    )

    return train_df, validation_df, test_df


train_df, validation_df, test_df = get_data()
train_df.head()



## === cell 2

X_train = train_df["Phrase"].astype(str).fillna("")
y_train = train_df["Sentiment"].astype(int).values

X_val = validation_df["Phrase"].astype(str).fillna("")
y_val = validation_df["Sentiment"].astype(int).values

X_test = test_df["Phrase"].astype(str).fillna("")

model = Pipeline(
    steps=[
        (
            "tfidf",
            TfidfVectorizer(
                ngram_range=(1, 2),
                max_features=50000,
                min_df=2,
                strip_accents="unicode",
                lowercase=True,
            ),
        ),
        (
            "clf",
            MLPClassifier(
                hidden_layer_sizes=(250, 50),
                activation="relu",
                solver="adam",  # stable + fast; analogous to adaptive optimizers
                alpha=1e-4,  # mild regularization (dropout analog)
                batch_size=256,
                learning_rate_init=0.003,
                max_iter=20,  # roughly corresponds to a fixed training budget; no early stopping
                shuffle=True,
                random_state=0,
                verbose=False,
                early_stopping=False,
            ),
        ),
    ]
)

model.fit(X_train, y_train)



## === cell 3
train_acc = float(model.score(X_train, y_train))
val_acc = float(model.score(X_val, y_val))

print(f"Training set accuracy: {train_acc:.6f}")
print(f"Validation set accuracy: {val_acc:.6f}")



## === cell 4
test_pred = model.predict(X_test).astype(int)
test_df["Sentiment"] = test_pred

submission = test_df[["PhraseId", "Sentiment"]].copy()
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Submission columns:", submission.columns.tolist())
print("Sentiment value counts:\n", submission["Sentiment"].value_counts().sort_index())
