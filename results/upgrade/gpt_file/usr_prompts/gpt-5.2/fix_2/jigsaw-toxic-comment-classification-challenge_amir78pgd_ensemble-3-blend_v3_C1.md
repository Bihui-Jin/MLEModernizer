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

0.9863197044121592

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble many external submission files from `../input/...` that don’t exist in this environment, so no predictions are ever created. I keep the “ensemble submissions” core idea but make it robust: automatically discover any available submission-like CSVs under the provided `/kaggle/input` and `/kaggle/data` trees, validate they contain the required columns, align them to the sample submission ids, and then average them. If none are found (likely here), I fall back to producing a valid submission by copying `sample_submission.csv` (all 0.5s), ensuring a `.csv` is always written end-to-end without errors.'

# 9. Code solution

## === cell 0
import os

CANDIDATE_ROOTS = [
    "../input",  # typical Kaggle notebooks
    "/kaggle/input",  # typical Kaggle notebooks (absolute)
    "/kaggle/data",  # provided in this environment
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
    "/kaggle/input/jigsaw-toxic-comment-classification-challenge",
]

existing_roots = [p for p in CANDIDATE_ROOTS if os.path.exists(p)]
existing_roots



## === cell 1
import numpy as np
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]


def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

test_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/test.csv",
        "/kaggle/input/test.csv",
        "/kaggle/data/test.csv",
    ]
)

train_path = find_first_existing(
    [
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/train.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/train.csv",
        "/kaggle/input/train.csv",
        "/kaggle/data/train.csv",
    ]
)

sample_path, test_path, train_path



## === cell 2
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(sample_path)
required_cols = ["id"] + label_cols

missing = [c for c in required_cols if c not in sample_sub.columns]
if missing:
    raise ValueError(f"sample_submission.csv missing columns: {missing}")

sample_sub = sample_sub[required_cols].copy()
sample_sub.head()




## === cell 3
def iter_csv_files(root, max_files=2000):
    n = 0
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                yield os.path.join(dirpath, fn)
                n += 1
                if n >= max_files:
                    return


def try_load_submission_csv(path, sample_ids):
    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    if not all(c in df.columns for c in required_cols):
        return None

    if len(df) != len(sample_ids):
        return None

    try:
        df = df[required_cols].copy()
        df = df.set_index("id").reindex(sample_ids)
    except Exception:
        return None

    if df.isna().any().any():
        return None

    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    if df[label_cols].isna().any().any():
        return None

    df[label_cols] = df[label_cols].clip(0.0, 1.0)
    df = df.reset_index()
    return df


sample_ids = sample_sub["id"].tolist()

found_paths = []
models = []

for root in existing_roots:
    for csv_path in iter_csv_files(root):
        df_sub = try_load_submission_csv(csv_path, sample_ids)
        if df_sub is not None:
            models.append(df_sub)
            found_paths.append(csv_path)

len(models), (found_paths[:5] if found_paths else found_paths)



## === cell 4
if models:
    preds = np.stack([m[label_cols].to_numpy(dtype=np.float64) for m in models], axis=0)
    mean_preds = preds.mean(axis=0)

    p_res = sample_sub.copy()
    p_res[label_cols] = mean_preds
    p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)

    p_res = p_res[required_cols]
else:
    p_res = sample_sub.copy()

p_res.head(), p_res.shape



## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)

chk = pd.read_csv(out_path, nrows=5)
chk.columns.tolist(), out_path
