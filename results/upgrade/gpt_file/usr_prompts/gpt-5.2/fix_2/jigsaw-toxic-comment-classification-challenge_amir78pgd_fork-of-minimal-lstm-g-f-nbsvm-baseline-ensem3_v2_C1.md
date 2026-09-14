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

0.983552631199519

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble four external Kaggle Dataset submissions that are not present in your environment (`../input/improved-lstm-baseline-...`). I keep the same “blend multiple submissions” core logic, but make it robust by (1) auto-discovering any available `submission.csv` files under the provided `/kaggle/input` and `/kaggle/data` trees, and (2) falling back to a safe baseline using `sample_submission.csv` (0.5s) if none are found so a valid CSV is always produced. I also enforce correct column order, numeric types, id alignment, and fill any missing label columns to avoid silent format bugs. This run end-to-end and always write `submission.csv`.'

# 9. Code solution

## === cell 0

import os
import glob
import numpy as np
import pandas as pd

RANDOM_SEED = 42

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sub_cols = ["id"] + label_cols

SEARCH_DIRS = [
    "/kaggle/input",
    "/kaggle/data",
    "../input",  # keep original relative pattern, in case it exists
    "../data",
]


def find_existing_file(candidates):
    for p in candidates:
        if p and os.path.exists(p):
            return p
    return None


def list_submission_candidates(max_files=50):
    patterns = []
    for d in SEARCH_DIRS:
        patterns.extend(
            [
                os.path.join(d, "**", "submission.csv"),
                os.path.join(d, "**", "*submission*.csv"),
            ]
        )
    files = []
    for pat in patterns:
        files.extend(glob.glob(pat, recursive=True))
    seen = set()
    uniq = []
    for f in files:
        if f not in seen and os.path.isfile(f):
            seen.add(f)
            uniq.append(f)
    return uniq[:max_files]


def load_submission(path):
    df = pd.read_csv(path)
    if "id" not in df.columns:
        raise ValueError(f"{path} has no 'id' column")
    for c in label_cols:
        if c not in df.columns:
            df[c] = 0.5
    df = df[sub_cols].copy()
    for c in label_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float64")
        df[c] = df[c].fillna(0.5).clip(0.0, 1.0)
    df["id"] = df["id"].astype(str)
    return df




## === cell 1
sample_submission_path = find_existing_file(
    [
        "/kaggle/data/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "/kaggle/input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "../input/jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "../input/sample_submission.csv",
    ]
)

if sample_submission_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv in expected Kaggle directories."
    )

sample = pd.read_csv(sample_submission_path)
if "id" not in sample.columns:
    raise ValueError("sample_submission.csv missing 'id' column")

sample["id"] = sample["id"].astype(str)
for c in label_cols:
    if c not in sample.columns:
        sample[c] = 0.5
sample = sample[sub_cols].copy()

print("Using sample_submission:", sample_submission_path)
print("Sample shape:", sample.shape)



## === cell 2
candidates = list_submission_candidates()

candidates = [
    p
    for p in candidates
    if os.path.abspath(p) != os.path.abspath(sample_submission_path)
]

print(f"Found {len(candidates)} submission candidate file(s).")
for p in candidates[:10]:
    print(" -", p)



## === cell 3
loaded = []
for p in candidates:
    try:
        df = load_submission(p)
        loaded.append((p, df))
    except Exception as e:
        continue

print(f"Loaded {len(loaded)} valid submission(s).")



## === cell 4
if len(loaded) == 0:
    p_res = sample.copy()
    for c in label_cols:
        p_res[c] = 0.5
    print("No valid external submissions found; using 0.5 baseline.")
else:
    aligned = []
    for path, df in loaded:
        merged = sample[["id"]].merge(df, on="id", how="left", sort=False)
        missing = merged[label_cols].isna().mean().mean()
        if missing > 0.05:
            continue
        for c in label_cols:
            merged[c] = merged[c].fillna(0.5).clip(0.0, 1.0).astype("float64")
        aligned.append(merged[sub_cols])

    if len(aligned) == 0:
        p_res = sample.copy()
        for c in label_cols:
            p_res[c] = 0.5
        print(
            "All loaded submissions were incompatible with sample ids; using 0.5 baseline."
        )
    else:
        p_res = sample[["id"]].copy()
        stack = np.stack(
            [a[label_cols].to_numpy(dtype=np.float64) for a in aligned], axis=0
        )
        avg = stack.mean(axis=0)
        p_res[label_cols] = avg
        print(f"Averaged {len(aligned)} submission(s).")



## === cell 5
p_res = p_res[sub_cols].copy()
p_res["id"] = p_res["id"].astype(str)

for c in label_cols:
    p_res[c] = pd.to_numeric(p_res[c], errors="coerce").astype("float64")
    p_res[c] = p_res[c].fillna(0.5).clip(0.0, 1.0)

if len(p_res) != len(sample):
    raise ValueError(
        f"Submission row count mismatch: got {len(p_res)} expected {len(sample)}"
    )

if not p_res["id"].equals(sample["id"]):
    p_res = sample[["id"]].merge(p_res, on="id", how="left", sort=False)
    for c in label_cols:
        p_res[c] = p_res[c].fillna(0.5).clip(0.0, 1.0)
    p_res = p_res[sub_cols]

print("Submission ready. Shape:", p_res.shape)
print(p_res.head())



## === cell 6
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Columns:", list(p_res.columns))
