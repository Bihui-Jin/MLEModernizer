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

0.9862835491331184

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The errors come from trying to read ensemble submission files from `../input/...` that do not exist in this Kaggle environment, so nothing downstream is defined. To keep the ensemble “core logic” (averaging predictions) but make it runnable, I load the provided `sample_submission.csv` as a valid base and replace each missing model file with that base (neutral 0.5 predictions), so the pipeline completes and writes a correctly formatted `submission.csv`. I also add strict column/order alignment on `id` to avoid silent misalignment bugs if any file is present. This yield a valid submission (score likely near 0.5 AUC average, but it run end-to-end).'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

INPUT_ROOT = "../input"
if os.path.isdir(INPUT_ROOT):
    print("Contents of ../input:", os.listdir(INPUT_ROOT)[:50])
else:
    print(
        "No ../input directory found in this environment. Using /kaggle/data paths instead."
    )



## === cell 1
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
f_dpcnn = "../input/dpcnn-wordcloud/10fold_dpcnn_test.csv"
f_dmcnn = "../input/dmcnn-demoji/10fold_dmcnn_am.csv"
f_rcn = "../input/rcn-capsule/10fold_capsule_am.csv"

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

CANDIDATE_SAMPLE = [
    "/kaggle/data/sample_submission.csv",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
    "../input/sample_submission.csv",
]
sample_path = next((p for p in CANDIDATE_SAMPLE if os.path.exists(p)), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations: "
        + ", ".join(CANDIDATE_SAMPLE)
    )

base_sub = pd.read_csv(sample_path)
expected_cols = ["id"] + label_cols
missing = [c for c in expected_cols if c not in base_sub.columns]
if missing:
    raise ValueError(f"sample_submission is missing columns: {missing}")

base_sub = base_sub[expected_cols].copy()


def safe_read_pred(path: str, base_df: pd.DataFrame) -> pd.DataFrame:
    """
    Read a prediction CSV if it exists; otherwise fall back to base_df.
    Also enforce id alignment and required columns.
    """
    if path is None or (not os.path.exists(path)):
        return base_df.copy()

    df = pd.read_csv(path)

    if "id" not in df.columns:
        raise ValueError(f"Prediction file {path} does not contain 'id' column.")

    missing_lbl = [c for c in label_cols if c not in df.columns]
    if missing_lbl:
        raise ValueError(f"Prediction file {path} missing label columns: {missing_lbl}")

    df = df[["id"] + label_cols].copy()

    df = df.set_index("id").reindex(base_df["id"].values)
    if df.isnull().any().any():
        df = df.fillna(base_df.set_index("id").loc[df.index, label_cols])

    df = df.reset_index()
    return df




## === cell 2
p_caps_gru = safe_read_pred(f_caps_gru, base_sub)
p_dual_embed_pl = safe_read_pred(f_dual_embed_pl, base_sub)
p_dual_embed_mish = safe_read_pred(f_dual_embed_mish, base_sub)
p_lstm_glove_tta = safe_read_pred(f_lstm_glove_tta, base_sub)
p_dual_embed_dehyp = safe_read_pred(f_dual_embed_dehyp, base_sub)
p_lstm_fast = safe_read_pred(f_lstm_fast, base_sub)
p_nbsvm = safe_read_pred(f_nbsvm, base_sub)
p_bi_post = safe_read_pred(f_bi_post, base_sub)
p_dpcnn = safe_read_pred(f_dpcnn, base_sub)
p_dmcnn = safe_read_pred(f_dmcnn, base_sub)
p_rcn = safe_read_pred(f_rcn, base_sub)

for name, df in [
    ("p_caps_gru", p_caps_gru),
    ("p_dual_embed_pl", p_dual_embed_pl),
    ("p_dual_embed_mish", p_dual_embed_mish),
    ("p_lstm_glove_tta", p_lstm_glove_tta),
    ("p_dual_embed_dehyp", p_dual_embed_dehyp),
    ("p_lstm_fast", p_lstm_fast),
    ("p_nbsvm", p_nbsvm),
    ("p_bi_post", p_bi_post),
    ("p_dpcnn", p_dpcnn),
    ("p_dmcnn", p_dmcnn),
    ("p_rcn", p_rcn),
]:
    if list(df.columns) != ["id"] + label_cols:
        raise ValueError(
            f"{name} has unexpected columns/order: {df.columns.tolist()[:20]}"
        )
    if df.shape[0] != base_sub.shape[0]:
        raise ValueError(f"{name} row count {df.shape[0]} != base {base_sub.shape[0]}")



## === cell 3
p_res = p_caps_gru.copy()
p_res[label_cols] = (
    p_caps_gru[label_cols]
    + p_dual_embed_pl[label_cols]
    + p_dual_embed_mish[label_cols]
    + p_lstm_glove_tta[label_cols]
    + p_dual_embed_dehyp[label_cols]
    + p_lstm_fast[label_cols]
    + p_nbsvm[label_cols]
    + p_bi_post[label_cols] * 5
    + p_dpcnn[label_cols] * 5
    + p_dmcnn[label_cols] * 5
    + p_rcn[label_cols] * 5
) / 27

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)



## === cell 4
p_res = p_res[["id"] + label_cols].copy()

if p_res["id"].isnull().any():
    raise ValueError("Submission has missing ids.")
if p_res[label_cols].isnull().any().any():
    raise ValueError("Submission has NaNs in prediction columns.")

print(p_res.head())
print("Submission shape:", p_res.shape)



## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("File size (bytes):", os.path.getsize(out_path))
