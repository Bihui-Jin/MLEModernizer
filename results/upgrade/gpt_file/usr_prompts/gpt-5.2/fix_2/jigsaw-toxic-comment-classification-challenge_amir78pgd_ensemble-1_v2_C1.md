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

0.9854415183218128

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble several external submissions that are not present in this Kaggle environment (`../input/capsule-net-with-gru/...` etc.), so the first `read_csv` crashes and nothing downstream is defined. To keep the core “average submissions” logic intact while making it runnable end-to-end, I switch to using the provided `sample_submission.csv` as a safe base and (optionally) average any of those ensemble files only if they actually exist. This guarantees a valid `submission.csv` with the correct columns/order and no runtime errors; since no current score was produced, this at least yields a valid submission (though score likely be low unless some ensemble files are available).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

LABEL_COLS = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None




## === cell 1
import os

os.listdir("../input")[:20]



## === cell 2
import numpy as np, pandas as pd

f_caps_gru = "../input/capsule-net-with-gru/submission.csv"
f_dual_embed_pl = "../input/submission-dual-embed-pl/submission_dual_embed (2).csv"
f_dual_embed_mish = "../input/bi-gru-lstm-dual-embedding-with-mish/submission.csv"
f_lstm_glove_tta = (
    "../input/improved-lstm-baseline-glove-dropout-trainta/submission.csv"
)
f_dual_embed_dehyp = (
    "../input/improved-lstm-baseline-bi-lstm-dual-embed-dehyp/submission.csv"
)
f_lstm_fast = "../input/improved-lstm-baseline-fasttext-dropout/submission.csv"
f_nbsvm = "../input/nb-svm-strong-linear-baseline/submission.csv"
f_bi_post = "../input/bi-post/10fold_lstmpp_am.csv"

sample_sub_path = find_first_existing(
    [
        "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "../input/sample_submission.csv",
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
    ]
)
test_path = find_first_existing(
    [
        "../input/jigsaw-toxic-comment-classification-challenge/test.csv",
        "../input/test.csv",
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/input/test.csv",
    ]
)

if sample_sub_path is None or test_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and/or test.csv in the provided input paths."
    )



## === cell 3
pred_files = [
    f_caps_gru,
    f_dual_embed_pl,
    f_dual_embed_mish,
    f_lstm_glove_tta,
    f_dual_embed_dehyp,
    f_lstm_fast,
    f_nbsvm,
    f_bi_post,
]

available = [p for p in pred_files if os.path.exists(p)]
print(f"Found {len(available)} / {len(pred_files)} external prediction files.")
for p in available:
    print(" -", p)

sample_sub = pd.read_csv(sample_sub_path)

test_df = pd.read_csv(test_path, usecols=["id"])
if len(sample_sub) != len(test_df):
    sample_sub = sample_sub.merge(test_df, on="id", how="right", validate="one_to_one")

pred_dfs = []
for p in available:
    df = pd.read_csv(p)
    if "id" not in df.columns:
        raise ValueError(f"Prediction file missing 'id' column: {p}")
    missing = [c for c in LABEL_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"Prediction file missing label columns {missing}: {p}")

    df = df[["id"] + LABEL_COLS].copy()
    df = test_df.merge(df, on="id", how="left")
    df[LABEL_COLS] = df[LABEL_COLS].astype(float).fillna(0.5).clip(0.0, 1.0)
    pred_dfs.append(df)



## === cell 4
p_res = test_df.copy()

if len(pred_dfs) == 0:
    aligned_sample = test_df.merge(sample_sub[["id"] + LABEL_COLS], on="id", how="left")
    aligned_sample[LABEL_COLS] = (
        aligned_sample[LABEL_COLS].astype(float).fillna(0.5).clip(0.0, 1.0)
    )
    p_res = aligned_sample
else:
    avg = np.zeros((len(test_df), len(LABEL_COLS)), dtype=np.float64)
    for df in pred_dfs:
        avg += df[LABEL_COLS].to_numpy(dtype=np.float64)
    avg /= float(len(pred_dfs))
    p_res[LABEL_COLS] = avg

p_res = p_res[["id"] + LABEL_COLS].copy()
p_res[LABEL_COLS] = p_res[LABEL_COLS].astype(float).fillna(0.5).clip(0.0, 1.0)

print(p_res.head())
print(p_res.shape)



## === cell 5
label_cols = LABEL_COLS



## === cell 6
assert list(p_res.columns) == ["id"] + label_cols
assert p_res["id"].isna().sum() == 0
assert p_res[label_cols].isna().sum().sum() == 0
assert len(p_res) == len(test_df)



## === cell 7
p_res.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", p_res.shape)
print(pd.read_csv("submission.csv", nrows=3))
