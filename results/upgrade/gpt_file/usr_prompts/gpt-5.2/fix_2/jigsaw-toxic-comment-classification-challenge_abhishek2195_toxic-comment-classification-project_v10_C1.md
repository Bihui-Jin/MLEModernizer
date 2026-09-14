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

3.8

# 3. Installed packages

geopandas==0.14.4
imbalanced-learn==0.13.0
joblib==1.5.2
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

0.0508228306890561

# 6. Current score

0.94469

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.94469) has done: 'I remove the incompatible `imblearn`/SMOTE dependency (it crashes due to a scikit-learn version mismatch) and replace it with scikit-learn’s built-in `resample` logic to preserve the intended oversampling behavior. I also stop relying on missing external `df.pkl` inputs and instead load `train.csv`/`test.csv` from the provided Kaggle dataset paths, creating the required `lemmatized` text column in a minimal, deterministic way. I fix the Naive Bayes probability assignment bug (needs `predict_proba(... )[:,1]` for the positive class) and ensure the submission matches `sample_submission.csv` columns/order and is written to a `.csv` file. These changes are necessary for the notebook to run end-to-end and produce a valid submission while keeping the same overall TF-IDF + (over)sampling + MultinomialNB core approach.'

# 9. Code solution

## === cell 0
import os
import re
import string
import warnings

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.utils import resample
import joblib

warnings.filterwarnings("ignore")

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
DATA_DIR = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(train_path), f"Missing: {train_path}"
assert os.path.exists(test_path), f"Missing: {test_path}"
assert os.path.exists(sample_path), f"Missing: {sample_path}"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

print(train_df.shape, test_df.shape, sample_sub.shape)
train_df.head()




## === cell 2
def basic_normalize(text: str) -> str:
    if pd.isna(text):
        return ""
    text = str(text).lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text)  # remove urls
    text = re.sub(r"\d+", " ", text)  # remove digits
    text = text.translate(str.maketrans({c: " " for c in string.punctuation}))
    text = re.sub(r"\s+", " ", text).strip()
    return text


train_df["lemmatized"] = train_df["comment_text"].map(basic_normalize)
test_df["lemmatized"] = test_df["comment_text"].map(basic_normalize)

train_df[["id", "comment_text", "lemmatized"]].head()




## === cell 3
def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float16).min
                    and c_max < np.finfo(np.float16).max
                ):
                    df[col] = df[col].astype(np.float16)
                elif (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose:
        print(
            "Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(
                end_mem, 100 * (start_mem - end_mem) / max(start_mem, 1e-9)
            )
        )
    return df


train_df = reduce_mem_usage(train_df, verbose=True)
test_df = reduce_mem_usage(test_df, verbose=True)



## === cell 4
label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

fig, axes = plt.subplots(3, 2, figsize=(15, 15))
for ax, class_name in zip(axes.flatten(), label_cols):
    pd.value_counts(train_df[class_name], sort=True).plot(kind="bar", rot=0, ax=ax)
    ax.set_title(f"{class_name} Distribution")
    ax.set_xticks([0, 1], [0, 1])
    ax.set_xlabel("Labels")
    ax.set_ylabel("Frequency")
plt.tight_layout()
plt.show()



## === cell 5
vec = TfidfVectorizer(max_features=1000, stop_words="english", analyzer="word")

tfidf = vec.fit_transform(train_df["lemmatized"])
tfidf_test = vec.transform(test_df["lemmatized"])

print("TFIDF train shape:", tfidf.shape)
print("TFIDF test shape:", tfidf_test.shape)



## === cell 6
X = tfidf.toarray()
X_test = tfidf_test.toarray()
target = train_df[label_cols].values.astype(np.int8)

print(X.shape, X_test.shape, target.shape)



## === cell 7
prob = sample_sub.copy()
prob["id"] = test_df["id"].values

for c in label_cols:
    if c not in prob.columns:
        prob[c] = 0.5

prob.head()




## === cell 8
def random_oversample_binary(X_in, y_in, random_state=42):
    y_in = np.asarray(y_in).astype(int)
    pos_idx = np.where(y_in == 1)[0]
    neg_idx = np.where(y_in == 0)[0]

    if len(pos_idx) == 0 or len(neg_idx) == 0:
        return X_in, y_in

    if len(pos_idx) < len(neg_idx):
        pos_up = resample(
            pos_idx, replace=True, n_samples=len(neg_idx), random_state=random_state
        )
        idx = np.concatenate([neg_idx, pos_up])
    else:
        neg_up = resample(
            neg_idx, replace=True, n_samples=len(pos_idx), random_state=random_state
        )
        idx = np.concatenate([pos_idx, neg_up])

    rng = np.random.RandomState(random_state)
    rng.shuffle(idx)

    return X_in[idx], y_in[idx]


models = []

for index, value in enumerate(label_cols):
    print(f"{value} - Model:\n")

    y = target[:, index]
    print(f"Before Over-Sampling [{value}] dataset shape=>({X.shape},{y.shape})")

    X_temp, y_temp = random_oversample_binary(X, y, random_state=RANDOM_STATE)
    print(
        f"After Over-Sampling [{value}] dataset shape=>({X_temp.shape},{y_temp.shape})"
    )

    x_train, x_val, y_train, y_val = train_test_split(
        X_temp, y_temp, stratify=y_temp, test_size=0.2, random_state=RANDOM_STATE
    )

    test_model = MultinomialNB()
    test_model = test_model.fit(x_train, y_train)

    train_pred = test_model.predict(x_train)
    print(
        "In-sample Evaluation:\n", classification_report(y_train, train_pred, digits=4)
    )

    val_pred = test_model.predict(x_val)
    print("Out-sample Evaluation\n", classification_report(y_val, val_pred, digits=4))

    final_model = MultinomialNB()
    final_model = final_model.fit(X_temp, y_temp)
    models.append(final_model)

    prob[value] = final_model.predict_proba(X_test)[:, 1].astype(np.float32)

    joblib.dump(final_model, f"{value}-model.pkl")

prob.head()



## === cell 9
prob = prob[["id"] + label_cols]
assert list(prob.columns) == list(
    sample_sub.columns
), "Submission columns/order mismatch."

for c in label_cols:
    prob[c] = prob[c].clip(0.0, 1.0)

out_path = "submission-NB-tfidf-oversample.csv"
prob.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", prob.shape)
prob.describe(include="all").T
