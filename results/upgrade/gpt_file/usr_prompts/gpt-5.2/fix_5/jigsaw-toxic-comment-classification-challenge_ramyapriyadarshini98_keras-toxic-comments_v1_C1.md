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
Given a dataset of comments from Wikipedia's talk page edits, predict the probability of each comment being toxic.

## Metric
Mean column-wise ROC AUC; the average of the individual AUCs of each predicted column.

## Submission Format
For each `id` in the test set, you must predict a probability for each of the six possible types of comment toxicity (toxic, severe_toxic, obscene, threat, insult, identity_hate). The columns must be in the same order as shown below. The file should contain a header and have the following format:

```
id,toxic,severe_toxic,obscene,threat,insult,identity_hate
00001cee341fdb12,0.5,0.5,0.5,0.5,0.5,0.5
0000247867823ef7,0.5,0.5,0.5,0.5,0.5,0.5
etc.
```

## Dataset 
- **train.csv** - the training set, contains comments with their binary labels
- **test.csv** - the test set, you must predict the toxicity probabilities for these comments.
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        input/
            description.md (69 lines)
            sample_submission.csv (153165 lines)
            sample_submission.csv.zip (1.5 MB)
            test.csv (552889 lines)
            test.csv.zip (24.6 MB)
            train.csv (561809 lines)
            train.csv.zip (27.7 MB)
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
        working/
            jigsaw-toxic-comment-classification-challenge/
                description.md (69 lines)
                sample_submission.csv (153165 lines)
                ... and 5 other files
                jigsaw-toxic-comment-classification-challenge/
```

-> data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/jigsaw-toxic-comment-classification-challenge/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/jigsaw-toxic-comment-classification-challenge/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/sample_submission.csv has 153164 rows and 7 columns.
The columns are: id, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> data/test.csv has 552888 rows and 2 columns.
The columns are: id, comment_text

-> data/train.csv has 561808 rows and 8 columns.
The columns are: id, comment_text, toxic, severe_toxic, obscene, threat, insult, identity_hate

-> (stopped after 10 files for performance)

# 5. Target score

0.778875022066981

# 6. Current score

0.95954

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.97331) has done: 'I fix the text-cleaning pipeline so it doesn’t try to create a NumPy array from variable-length token lists (the cause of your first crash), while keeping the same cleaning logic and downstream TF‑IDF features. I also remove the hard dependency on NLTK corpora downloads (stopwords/wordnet) by providing safe fallbacks when they’re unavailable, so the notebook runs in Kaggle’s offline environment. Finally, I replace the broken Keras import (protobuf incompatibility) with a simple one-vs-rest `LogisticRegression` classifier on the same TF‑IDF features, which matches the competition’s ROC-AUC objective and produces properly-shaped probability predictions for a valid submission CSV.'
- What this solution (achieved 0.97194) has done: 'Your current score (0.97331) is much higher than the target (0.778875), so to move closer we should intentionally reduce model strength with the smallest, safest change that preserves the same pipeline and semantics. I keep the exact cleaning and LogisticRegression OvR approach, but make the TF‑IDF representation less expressive by restricting it to unigrams, lowering `max_features`, and increasing `min_df` so rare/phrase signals are removed. This reliably decreases ROC-AUC for this competition while still producing valid probabilities and a correct submission file. I also keep paths and submission formatting identical.'
- What this solution (achieved 0.96776) has done: 'Your current score (0.97194) is far above the target (0.778875), so to move closer we should intentionally weaken the representation/model with the smallest safe changes while keeping the same cleaning + TF‑IDF + OneVsRest(LogisticRegression) core logic. I make TF‑IDF substantially less informative by using a much smaller vocabulary and aggressively dropping rare terms, and I also increase regularization (smaller C) to reduce separability. These changes reliably reduce ROC-AUC on this competition without breaking evaluation semantics or submission formatting. The pipeline still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.95954) has done: 'Your current score (0.96776) is well above the target (0.778875), so we should intentionally weaken the model slightly to move closer while keeping the exact same cleaning + TF‑IDF + OneVsRest(LogisticRegression) core approach. The smallest reliable lever is to make the TF‑IDF representation even less informative (smaller vocabulary and dropping more terms) and increase regularization a bit, which typically reduces mean ROC-AUC on this task without breaking semantics. I also ensure we keep the same submission format and column order. No changes to the overall pipeline or training loop, just these parameter adjustments.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import re
import string
from string import digits

print(os.listdir("../input"))



## === cell 1
import warnings

warnings.filterwarnings("ignore")

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

try:
    from nltk.corpus import stopwords

    _stopwords = set(stopwords.words("english"))
except Exception:
    _stopwords = {
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "while",
        "with",
        "for",
        "to",
        "of",
        "in",
        "on",
        "at",
        "by",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "being",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "as",
        "from",
        "into",
        "about",
        "over",
        "under",
        "again",
        "further",
        "then",
        "once",
        "i",
        "me",
        "my",
        "myself",
        "we",
        "our",
        "ours",
        "you",
        "your",
        "yours",
        "he",
        "him",
        "his",
        "she",
        "her",
        "hers",
        "they",
        "them",
        "their",
        "theirs",
        "do",
        "does",
        "did",
        "doing",
        "have",
        "has",
        "had",
        "having",
        "not",
        "no",
        "nor",
        "only",
        "own",
        "same",
        "so",
        "than",
        "too",
        "very",
        "can",
        "will",
        "just",
    }

try:
    from nltk.stem import WordNetLemmatizer

    _lemmatizer = WordNetLemmatizer()
    _use_lemmatizer = True
except Exception:
    _lemmatizer = None
    _use_lemmatizer = False



## === cell 2
train = pd.read_csv("../input/train.csv")
test = pd.read_csv("../input/test.csv")

print("\nTrain data:\n", train.head())
print("\nTest data:\n", test.head())



## === cell 3
train_data = train.drop(train.columns[0], axis=1)
test_data = test

train_comments = train_data.iloc[:, 0]  # comment_text
test_comments = test_data.iloc[:, 1]  # comment_text

train_comments_index = train_comments.index
test_comments_index = test_comments.index

comments = pd.concat([train_comments, test_comments], ignore_index=True)

labels = train_data.iloc[:, 1:]  # the 6 target columns

print("Train Comments Shape: ", train_comments.shape)
print("Test Comments Shape: ", test_comments.shape)
print("Comments Shape after Merge: ", comments.shape)
print("Labels shape: ", labels.shape)




## === cell 4
def _clean_one(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text)
    text = text.translate(str.maketrans(" ", " ", string.punctuation))
    text = text.translate(str.maketrans(" ", " ", "\n"))
    text = text.translate(str.maketrans(" ", " ", digits))
    text = re.sub(r"([a-z])([A-Z])", r"\1 \2", text)
    text = text.lower()

    tokens = text.split()
    tokens = [t for t in tokens if t not in _stopwords]

    if _use_lemmatizer:
        new_tokens = []
        for t in tokens:
            try:
                z = _lemmatizer.lemmatize(t)
                z = _lemmatizer.lemmatize(z, "v")
                new_tokens.append(z)
            except Exception:
                new_tokens.append(t)
        tokens = new_tokens

    return " ".join(tokens)


from tqdm import tqdm

tqdm.pandas()

clean_comments = comments.progress_apply(_clean_one)
clean_data = pd.DataFrame({"comment_text": clean_comments})

train_clean_data = clean_data.loc[train_comments_index].reset_index(drop=True)
test_clean_data = clean_data.drop(train_comments_index, axis=0).reset_index(drop=True)

train_result = pd.concat([train_clean_data, labels.reset_index(drop=True)], axis=1)
test_result = pd.concat(
    [test.iloc[:, 0].reset_index(drop=True), test_clean_data], axis=1
)

print(train_result.head())
print(test_result.head())



## === cell 5
tf_idf = TfidfVectorizer(ngram_range=(1, 1), max_features=1200, min_df=200)

tfidf_train = tf_idf.fit_transform(train_result["comment_text"])
tfidf_test = tf_idf.transform(test_result["comment_text"])

print("tfidf_train:", tfidf_train.shape)
print("tfidf_test:", tfidf_test.shape)



## === cell 6
target_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
X_train = tfidf_train
Y_train = train_result[target_cols].values

base_clf = LogisticRegression(solver="liblinear", max_iter=100, C=0.10)
clf = OneVsRestClassifier(base_clf, n_jobs=-1)

clf.fit(X_train, Y_train)



## === cell 7
y_pred = clf.predict_proba(tfidf_test)

if y_pred.ndim != 2 or y_pred.shape[1] != 6:
    raise ValueError(
        f"Unexpected prediction shape: {y_pred.shape}, expected (n_test, 6)"
    )

print("y_pred:", y_pred.shape, "min/max:", float(np.min(y_pred)), float(np.max(y_pred)))



## === cell 8
submission = pd.DataFrame(
    {
        "id": test_result["id"].values,
        "toxic": y_pred[:, 0],
        "severe_toxic": y_pred[:, 1],
        "obscene": y_pred[:, 2],
        "threat": y_pred[:, 3],
        "insult": y_pred[:, 4],
        "identity_hate": y_pred[:, 5],
    }
)

sample_sub = pd.read_csv("../input/sample_submission.csv")
submission = submission[sample_sub.columns]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
