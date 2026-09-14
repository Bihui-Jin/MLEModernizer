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

3.9

# 3. Installed packages

fasttext==0.9.3
geopandas==0.14.4
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
tqdm==4.67.1

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

0.79019

# 6. Current score

0.93526

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.93526) has done: 'Your notebook likely failed to yield a Kaggle score because it doesn’t run end-to-end in Kaggle’s script/console setting: `tqdm.notebook` can error outside Jupyter, and cells start at `0` (your runner may skip it), so imports may not execute. I make the smallest execution-stability fixes: switch to `tqdm.auto`, ensure everything is in sequential cells starting at 1, and add a lightweight check that the submission file is actually written and non-empty. To nudge ROC AUC upward without changing the core model/training approach, I only adjust fastText prediction to request `k=2` consistently but fall back to `k=1` safely, and I ensure labels in the training CSV are exactly in the `__class__{0,1}` format with no accidental whitespace issues. The model architecture/training loop (one fastText model per label, `loss="ova"`, `epoch=1`, `wordNgrams=2`, `dim=100`) remains the same.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np

from fasttext import train_supervised

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

from tqdm.auto import tqdm

from statistics import mean
import unicodedata



## === cell 1
BASE_PATH = "/kaggle/input/jigsaw-toxic-comment-classification-challenge"

train_path = os.path.join(BASE_PATH, "train.csv")
test_path = os.path.join(BASE_PATH, "test.csv")
sample_path = os.path.join(BASE_PATH, "sample_submission.csv")

train_text = pd.read_csv(train_path)
test_text = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

print(train_text.shape, test_text.shape, sample_submission.shape)
train_text.head()



## === cell 2
y_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]




## === cell 3
def clean_it(text, normalize=True):
    s = (
        str(text)
        .replace(",", " ")
        .replace('"', "")
        .replace("'", " ' ")
        .replace(".", " . ")
        .replace("(", " ( ")
        .replace(")", " ) ")
        .replace("!", " ! ")
        .replace("?", " ? ")
        .replace(":", " ")
        .replace(";", " ")
        .lower()
    )
    s = s.replace("\n", " ").replace("\r", " ").replace("\t", " ")
    if normalize:
        s = (
            unicodedata.normalize("NFKD", s)
            .encode("ascii", "ignore")
            .decode("utf-8", "ignore")
        )
    return s


def clean_df(
    data, cleanit=False, shuffleit=False, encodeit=False, label_prefix="__class__"
):
    df = data[["comment_text"]].copy(deep=True)

    for col in y_cols:
        df[col] = (label_prefix + data[col].astype(int).astype(str)) + " "

    if cleanit:
        df["comment_text"] = df["comment_text"].apply(
            lambda x: clean_it(x, normalize=encodeit)
        )

    if shuffleit:
        df = df.sample(frac=1, random_state=123).reset_index(drop=True)

    return df




## === cell 4
X = train_text.comment_text
y = train_text[y_cols]

train, val = train_test_split(train_text, test_size=0.2, shuffle=True, random_state=123)

df_train_cleaned = clean_df(train, cleanit=True, shuffleit=True, encodeit=True)

df_val_cleaned = clean_df(
    val, cleanit=True, shuffleit=False, encodeit=True, label_prefix=""
)



## === cell 5
df_train_cleaned.head()



## === cell 6
df_val_cleaned.head()




## === cell 7
def fasttext_predict_proba_batched(
    model, texts, batch_size=100000, pos_label="__class__1"
):
    """
    Robustly returns probability for pos_label for each text in texts.
    Output shape: (len(texts),)
    """
    texts = list(texts)
    out = np.empty(len(texts), dtype=np.float32)
    idx = 0

    for start in tqdm(
        range(0, len(texts), batch_size), desc="Batched predict", leave=False
    ):
        batch = texts[start : start + batch_size]

        try:
            labels, probs = model.predict(batch, k=2)
        except Exception:
            labels, probs = model.predict(batch, k=1)

        for j in range(len(batch)):
            lab_j = labels[j]
            prob_j = probs[j]
            d = {lab_j[t]: float(prob_j[t]) for t in range(len(lab_j))}
            if pos_label in d:
                p = d[pos_label]
            else:
                other = "__class__0" if pos_label == "__class__1" else "__class__1"
                if other in d:
                    p = 1.0 - d[other]
                else:
                    p = 0.0
            out[idx] = np.float32(min(1.0, max(0.0, p)))
            idx += 1
    return out


model_dict = {}
all_preds = []

train_file = "/kaggle/working/final_train.csv"
val_texts = df_val_cleaned["comment_text"].values

for col in y_cols:
    df_train_cleaned[[col, "comment_text"]].to_csv(
        train_file, header=None, index=False, columns=[col, "comment_text"]
    )

    model = train_supervised(
        input=train_file,
        label="__class__",
        lr=1.0,
        epoch=1,  # keep core logic unchanged
        loss="ova",
        wordNgrams=2,
        dim=100,  # keep core logic unchanged
        thread=2,
        verbose=0,
        seed=123,
    )
    model_dict[col] = model

    toxic_preds = fasttext_predict_proba_batched(
        model, val_texts, batch_size=100000, pos_label="__class__1"
    )
    all_preds.append(toxic_preds)



## === cell 8
all_preds_array = np.transpose(np.array(all_preds))




## === cell 9
def accuracy(y_test, y_pred):
    aucs = []
    for col in range(y_test.shape[1]):
        aucs.append(roc_auc_score(y_test[:, col], y_pred[:, col]))
    return aucs




## === cell 10
y_val_actuals = val[y_cols].astype("int").to_numpy()
mean_auc = mean(accuracy(y_val_actuals, all_preds_array))
print("Validation mean AUC:", mean_auc)



## === cell 11
df_test = test_text[["id", "comment_text"]].copy()
df_test["comment_text"] = df_test["comment_text"].apply(
    lambda x: clean_it(x, normalize=True)
)

df_submit_base = sample_submission[["id"]].merge(
    df_test, on="id", how="left", validate="one_to_one"
)

missing = df_submit_base["comment_text"].isna().sum()
if missing > 0:
    df_submit_base["comment_text"] = df_submit_base["comment_text"].fillna("")
    print(
        f"Warning: {missing} submission ids missing from test.csv; filled with empty text."
    )

test_texts = df_submit_base["comment_text"].values

all_test_preds = []
for col in tqdm(y_cols, desc="Test predict columns"):
    model = model_dict[col]
    toxic_preds = fasttext_predict_proba_batched(
        model, test_texts, batch_size=100000, pos_label="__class__1"
    )
    all_test_preds.append(toxic_preds)

all_test_preds_array = np.transpose(np.array(all_test_preds))
df_submit_base[y_cols] = all_test_preds_array

submission = df_submit_base[["id"] + y_cols].copy()

if list(submission.columns) != ["id"] + y_cols:
    raise ValueError("Submission columns are not in the required order.")
if submission.shape[0] != sample_submission.shape[0]:
    raise ValueError(
        f"Submission row count {submission.shape[0]} != sample_submission row count {sample_submission.shape[0]}"
    )
if not submission["id"].equals(sample_submission["id"]):
    raise ValueError("Submission id order does not match sample_submission id order.")

sub_path = "/kaggle/working/submission.csv"
submission.to_csv(sub_path, index=False)

if (not os.path.exists(sub_path)) or (os.path.getsize(sub_path) < 1000):
    raise RuntimeError("Submission file was not written correctly.")

print(
    "Wrote:", sub_path, "shape:", submission.shape, "bytes:", os.path.getsize(sub_path)
)
submission.head()
