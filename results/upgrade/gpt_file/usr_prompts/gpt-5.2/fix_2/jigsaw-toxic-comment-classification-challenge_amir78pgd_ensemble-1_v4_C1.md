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

0.9860176495051112

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The failure is because this notebook tries to ensemble multiple external Kaggle dataset submissions from `../input/...`, but those datasets are not present in your environment, so the first `read_csv` crashes and everything downstream is undefined. To keep the “core logic” (simple averaging ensemble) while making it run end-to-end, I change the code to ensemble whatever `.csv` prediction files are actually available under your provided `/kaggle/data/...` tree; if none are found, it fall back to a safe baseline using `sample_submission.csv` so a valid submission is always produced. I also harden the merge/alignment by `id`, enforce the required column order, and clip predictions to `[0, 1]` (score-neutral for proper probabilities, but prevents invalid values). The output always be written as `submission.csv` with the correct header/columns.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

for p in [
    "../input",
    "/kaggle/input",
    "/kaggle/data",
    "/kaggle/data/jigsaw-toxic-comment-classification-challenge",
]:
    try:
        if os.path.exists(p):
            print(p, "->", len(os.listdir(p)), "items")
    except Exception as e:
        print("Could not list", p, ":", repr(e))



## === cell 1
import numpy as np
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
sub_cols = ["id"] + label_cols

DATA_ROOTS = [
    Path("/kaggle/data"),
    Path("/kaggle/input"),  # may or may not exist
]


def find_first_existing(relative_candidates):
    """Return the first existing path among candidates (relative to DATA_ROOTS or absolute)."""
    for cand in relative_candidates:
        cand_path = Path(cand)
        if cand_path.is_absolute() and cand_path.exists():
            return cand_path
        for root in DATA_ROOTS:
            p = root / cand
            if p.exists():
                return p
    return None


sample_path = find_first_existing(
    [
        "jigsaw-toxic-comment-classification-challenge/sample_submission.csv",
        "sample_submission.csv",
    ]
)
test_path = find_first_existing(
    [
        "jigsaw-toxic-comment-classification-challenge/test.csv",
        "test.csv",
    ]
)

if sample_path is None or test_path is None:
    raise FileNotFoundError(
        f"Could not locate sample_submission.csv or test.csv under roots {DATA_ROOTS}. "
        f"sample_path={sample_path}, test_path={test_path}"
    )

sample_submission = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path, usecols=["id"])

for c in sub_cols:
    if c not in sample_submission.columns:
        if c == "id":
            raise ValueError("sample_submission.csv is missing 'id' column.")
        sample_submission[c] = 0.5

sample_submission = sample_submission[sub_cols]

print("Loaded sample_submission:", sample_path, sample_submission.shape)
print("Loaded test:", test_path, test_df.shape)



## === cell 2


def is_submission_like(df: pd.DataFrame) -> bool:
    cols = set(df.columns)
    return ("id" in cols) and all(c in cols for c in label_cols)


def load_submission_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    if "id" in df.columns:
        keep = ["id"] + [c for c in label_cols if c in df.columns]
        df = df[keep]
    return df


candidates = []
for root in DATA_ROOTS:
    if not root.exists():
        continue
    for p in root.rglob("*.csv"):
        if p.name.lower() == "sample_submission.csv":
            continue
        candidates.append(p)

loaded = []
used_paths = []

for p in sorted(candidates):
    try:
        df = load_submission_csv(p)
        if is_submission_like(df):
            loaded.append(df)
            used_paths.append(str(p))
    except Exception:
        continue

print(f"Found {len(loaded)} submission-like CSV(s).")
if len(loaded) > 0:
    print("Using the following files for averaging ensemble:")
    for up in used_paths[:25]:
        print(" -", up)
    if len(used_paths) > 25:
        print(f" - ... ({len(used_paths) - 25} more)")



## === cell 3


def align_to_test(df: pd.DataFrame, test_ids: pd.Series) -> pd.DataFrame:
    df = df.copy()
    df = df.drop_duplicates(subset=["id"], keep="last")
    for c in label_cols:
        if c not in df.columns:
            df[c] = np.nan
    df = df[["id"] + label_cols]
    df = test_ids.to_frame().merge(df, on="id", how="left")
    return df


test_ids = test_df["id"]

if len(loaded) == 0:
    p_res = align_to_test(sample_submission, test_ids)
    p_res[label_cols] = p_res[label_cols].fillna(0.5)
else:
    aligned = [align_to_test(df, test_ids) for df in loaded]
    pred_stack = np.stack(
        [a[label_cols].to_numpy(dtype=np.float32) for a in aligned], axis=0
    )
    preds_mean = np.nanmean(pred_stack, axis=0)
    p_res = pd.DataFrame({"id": test_ids})
    p_res[label_cols] = preds_mean

p_res[label_cols] = p_res[label_cols].clip(0.0, 1.0)

p_res = p_res[sub_cols]

print("Result shape:", p_res.shape)
print(p_res.head())



## === cell 4
assert list(p_res.columns) == sub_cols, "Submission columns/order mismatch."
assert len(p_res) == len(test_df), "Row count mismatch with test."
assert p_res["id"].isna().sum() == 0, "Found missing ids."
for c in label_cols:
    p_res[c] = pd.to_numeric(p_res[c], errors="coerce").fillna(0.5).clip(0.0, 1.0)



## === cell 5
out_path = "submission.csv"
p_res.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", p_res.shape)
print("Columns:", list(p_res.columns))
