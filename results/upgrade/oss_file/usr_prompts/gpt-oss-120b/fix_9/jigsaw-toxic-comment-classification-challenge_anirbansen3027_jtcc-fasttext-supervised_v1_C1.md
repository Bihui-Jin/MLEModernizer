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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the FastText prediction handling so it correctly extracts the probability of the positive class (labels ending with “1”), preventing empty‑array errors that halted validation and test inference. I also adjust the temporary training file path to a reliable local location. These minimal changes allow the script to run end‑to‑end, compute a valid mean AUC, and generate a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I train each FastText model on the full training set (instead of only the split) and increase the number of epochs to give the models more data and learning time, which should raise the validation AUC toward the target while keeping the overall pipeline unchanged. The rest of the code, including validation, test inference, and submission writing, remains the same.'
- What this solution (achieved 0.5) has done: 'Implemented two key fixes so the pipeline runs end‑to‑end and yields a valid submission: (1) replaced the unsupported loss name `"binary"` with FastText’s `"ova"` (one‑vs‑all) in both training calls; (2) added a comment explaining the change. These corrections allow model training, validation AUC computation, and test inference to complete, producing a correct `submission.csv`. No other logic was altered, preserving the original approach while enabling the score to be evaluated.'
- What this solution (achieved 0.5) has done: 'I increase the FastText training epochs from 15 to 30 (and keep the same learning rate) for both the validation‑split models and the full‑data models. Longer training should raise the validation AUC, moving the score closer to the target without altering any core logic or the submission format.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import unicodedata
from fasttext import train_supervised
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from tqdm import tqdm
from statistics import mean



## === cell 1
possible_dirs = [
    os.path.join("data", "jigsaw-toxic-comment-classification-challenge"),
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/working/jigsaw-toxic-comment-classification-challenge",
]
DATA_DIR = next((d for d in possible_dirs if os.path.isdir(d)), None)
if DATA_DIR is None:
    raise FileNotFoundError("Dataset directory not found in expected locations.")

train_path = os.path.join(DATA_DIR, "train.csv")
test_path = os.path.join(DATA_DIR, "test.csv")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

train_text = pd.read_csv(train_path)
test_text = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_sub_path)
print("Loaded:", train_text.shape, test_text.shape, sample_submission.shape)



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
    s = s.replace("\n", " ")
    if normalize:
        s = unicodedata.normalize("NFKD", s).encode("ascii", "ignore").decode("utf-8")
    return s


def clean_df(
    data, cleanit=False, shuffleit=False, encodeit=False, label_prefix="__class__"
):
    df = data[["comment_text"]].copy(deep=True)
    for col in y_cols:
        df[col] = label_prefix + data[col].astype(str) + " "
    if cleanit:
        df["comment_text"] = df["comment_text"].apply(lambda x: clean_it(x, encodeit))
    if shuffleit:
        df = df.sample(frac=1).reset_index(drop=True)
    return df




## === cell 4
train_split, val_split = train_test_split(train_text, shuffle=True, random_state=123)

df_train_cleaned = clean_df(train_split, cleanit=True, shuffleit=True)
df_val_cleaned = clean_df(val_split, cleanit=True, shuffleit=True, label_prefix="")




## === cell 5
def get_positive_prob(model, text):
    labels, probs = model.predict(text, k=2)
    prob_dict = dict(zip(labels, probs))
    for lbl in prob_dict:
        if lbl.endswith("1"):
            return prob_dict[lbl]
    return probs[0] if probs else 0.0


model_dict = {}
all_val_preds = []

for col in y_cols:
    train_file = os.path.join(os.getcwd(), f"train_{col}.txt")
    df_train_cleaned[[col, "comment_text"]].to_csv(
        train_file,
        header=False,
        index=False,
        sep=" ",
        columns=[col, "comment_text"],
    )

    model = train_supervised(
        input=train_file,
        label="__class__",
        lr=0.5,
        epoch=30,
        loss="ova",
        wordNgrams=2,
        dim=300,
        thread=2,
        verbose=0,
    )
    model_dict[col] = model

    val_preds = []
    for text in tqdm(df_val_cleaned["comment_text"].values, desc=f"Validating {col}"):
        prob_positive = get_positive_prob(model, text)
        val_preds.append(prob_positive)
    all_val_preds.append(val_preds)



## === cell 6
val_pred_array = np.transpose(np.array(all_val_preds))
y_val_actuals = df_val_cleaned[y_cols].astype(int).to_numpy()
mean_auc = mean(
    [
        roc_auc_score(y_val_actuals[:, i], val_pred_array[:, i])
        for i in range(len(y_cols))
    ]
)
print("Mean validation AUC:", mean_auc)



## === cell 7
df_full_cleaned = clean_df(train_text, cleanit=True, shuffleit=True)

for col in y_cols:
    train_file = os.path.join(os.getcwd(), f"full_train_{col}.txt")
    df_full_cleaned[[col, "comment_text"]].to_csv(
        train_file,
        header=False,
        index=False,
        sep=" ",
        columns=[col, "comment_text"],
    )
    model = train_supervised(
        input=train_file,
        label="__class__",
        lr=0.5,
        epoch=30,
        loss="ova",
        wordNgrams=2,
        dim=300,
        thread=2,
        verbose=0,
    )
    model_dict[col] = model  # overwrite with full‑data model for testing

df_test = pd.merge(test_text, sample_submission, on="id")
df_test_cleaned = clean_df(df_test, cleanit=True, shuffleit=False, label_prefix="")

all_test_preds = []
for col in tqdm(y_cols, desc="Testing"):
    model = model_dict[col]
    test_preds = []
    for text in df_test_cleaned["comment_text"].values:
        prob_positive = get_positive_prob(model, text)
        test_preds.append(prob_positive)
    all_test_preds.append(test_preds)

test_pred_array = np.transpose(np.array(all_test_preds))
df_test[y_cols] = test_pred_array
df_test.drop(columns=["comment_text"], inplace=True)



## === cell 8
submission_path = "submission.csv"
df_test.to_csv(submission_path, index=False)
print("Submission file written:", submission_path, "rows:", df_test.shape[0])
