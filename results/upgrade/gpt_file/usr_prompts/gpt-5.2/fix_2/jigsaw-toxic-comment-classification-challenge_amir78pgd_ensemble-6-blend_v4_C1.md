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

0.9849812848215592

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The failure is because the code expects out-of-notebook Kaggle “../input/…” model submission files that don’t exist in your current environment; so nothing gets loaded and the ensemble variables are undefined. I keep the same “blend multiple submissions” core idea, but make it robust by (1) auto-discovering which candidate prediction files actually exist under the provided `/kaggle/input/...` tree, (2) validating/aligning them to `sample_submission.csv` by `id`, and (3) averaging only the available ones. If none of those external submissions are present, the script fall back to producing a valid submission using the competition’s baseline (all 0.5), ensuring you always get a proper `submission.csv` with the required columns and order.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
DATA_ROOT = Path("/kaggle/data")

print("Exists /kaggle/input:", INPUT_ROOT.exists())
print("Top-level /kaggle/input entries (first 20):")
if INPUT_ROOT.exists():
    print(sorted([p.name for p in INPUT_ROOT.iterdir()])[:20])



## === cell 1
import numpy as np
import pandas as pd

label_cols = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]

sample_path_candidates = [
    DATA_ROOT / "sample_submission.csv",
    DATA_ROOT
    / "jigsaw-toxic-comment-classification-challenge"
    / "sample_submission.csv",
    INPUT_ROOT / "sample_submission.csv",
    INPUT_ROOT
    / "jigsaw-toxic-comment-classification-challenge"
    / "sample_submission.csv",
]
sample_path = next((p for p in sample_path_candidates if p.exists()), None)
if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv in expected locations."
    )

sample_sub = pd.read_csv(sample_path)
expected_cols = ["id"] + label_cols
if list(sample_sub.columns) != expected_cols:
    missing = set(expected_cols) - set(sample_sub.columns)
    if missing:
        raise ValueError(f"sample_submission.csv missing required columns: {missing}")
    sample_sub = sample_sub[expected_cols]

print("Loaded sample_submission:", sample_path)
print("Rows:", len(sample_sub), "Cols:", list(sample_sub.columns))



## === cell 2

candidate_basenames = [
    "submission.csv",
    "submission_001.csv",
    "10fold_lstmpp_am.csv",
    "10fold_dpcnn_test.csv",
    "10fold_dmcnn_am.csv",
    "10fold_capsule_am.csv",
    "10fold_attn_post_am.csv",
    "results_preds_cat.csv",
]


def find_candidate_files(root: Path, basenames):
    found = []
    if not root.exists():
        return found
    for bn in basenames:
        found.extend(root.rglob(bn))
    uniq = []
    seen = set()
    for p in found:
        sp = str(p)
        if sp not in seen:
            uniq.append(p)
            seen.add(sp)
    return uniq


candidate_files = find_candidate_files(INPUT_ROOT, candidate_basenames)

print("Found candidate prediction files:", len(candidate_files))
for p in candidate_files[:30]:
    print(" -", p)




## === cell 3
def load_and_align_pred(path: Path, sample_df: pd.DataFrame, label_cols):
    """
    Load a prediction CSV and align it to sample_submission ids and required columns.
    Returns aligned predictions as a float32 numpy array (n_rows, n_labels), or None if invalid.
    """
    try:
        df = pd.read_csv(path)
    except Exception as e:
        print(f"Skipping {path} (read error): {e}")
        return None

    if "id" not in df.columns:
        print(f"Skipping {path} (missing id column)")
        return None

    missing = set(label_cols) - set(df.columns)
    if missing:
        print(f"Skipping {path} (missing label columns: {sorted(missing)})")
        return None

    df = df[["id"] + label_cols].copy()

    merged = sample_df[["id"]].merge(df, on="id", how="left", validate="1:1")
    if merged[label_cols].isna().any().any():
        n_miss = int(merged[label_cols].isna().any(axis=1).sum())
        print(f"Skipping {path} (missing predictions for {n_miss} ids after alignment)")
        return None

    arr = merged[label_cols].to_numpy(dtype=np.float32)
    if (arr < -1e-3).any() or (arr > 1 + 1e-3).any():
        print(f"Skipping {path} (values out of expected [0,1] range)")
        return None

    arr = np.clip(arr, 0.0, 1.0)
    return arr


pred_arrays = []
used_files = []

for p in candidate_files:
    arr = load_and_align_pred(p, sample_sub, label_cols)
    if arr is not None:
        pred_arrays.append(arr)
        used_files.append(p)

print(f"Usable prediction files for blending: {len(used_files)}")
for p in used_files:
    print(" -", p)



## === cell 4

if len(pred_arrays) > 0:
    blend = np.mean(np.stack(pred_arrays, axis=0), axis=0)
else:
    print(
        "No external prediction files found/usable. Falling back to 0.5 baseline submission."
    )
    blend = np.full((len(sample_sub), len(label_cols)), 0.5, dtype=np.float32)

p_res = sample_sub.copy()
p_res[label_cols] = blend

p_res = p_res[["id"] + label_cols]
p_res[label_cols] = p_res[label_cols].astype(np.float32).clip(0.0, 1.0)

print("Submission preview:")
print(p_res.head())



## === cell 5
out_path = Path("submission.csv")
p_res.to_csv(out_path, index=False)

print("Wrote:", out_path.resolve())
print("Rows:", len(p_res), "Cols:", list(p_res.columns))
print("Any nulls:", p_res.isna().any().any())
print("Min/max per label:")
print(p_res[label_cols].agg(["min", "max"]))
