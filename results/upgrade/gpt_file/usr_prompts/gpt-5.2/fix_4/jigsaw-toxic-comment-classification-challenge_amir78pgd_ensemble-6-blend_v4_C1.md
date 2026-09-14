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

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The failure is because the code expects out-of-notebook Kaggle “../input/…” model submission files that don’t exist in your current environment; so nothing gets loaded and the ensemble variables are undefined. I keep the same “blend multiple submissions” core idea, but make it robust by (1) auto-discovering which candidate prediction files actually exist under the provided `/kaggle/input/...` tree, (2) validating/aligning them to `sample_submission.csv` by `id`, and (3) averaging only the available ones. If none of those external submissions are present, the script fall back to producing a valid submission using the competition’s baseline (all 0.5), ensuring you always get a proper `submission.csv` with the required columns and order.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score happens because the code usually finds no external prediction files to blend, so it falls back to the constant 0.5 baseline. To move toward the 0.98498 target without changing the “blend existing submissions” core idea, I (1) correctly discover and use Kaggle’s provided model prediction CSVs (`*.csv`) under the competition input directory (not just a fixed basename list), (2) validate that each candidate looks like a real submission (has `id` + all 6 label columns and enough rows), and (3) average only the valid ones aligned by `id`. If none are found, it still produces a valid `submission.csv` as before.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score indicates the script is almost always falling back to the constant-0.5 baseline because it finds no usable external prediction CSVs to blend under `/kaggle/input`. To move toward the 0.98498 target without changing the core “blend existing submissions” approach, I broaden discovery to also search `/kaggle/data` (where the competition files actually are in your environment), and I explicitly exclude the competition’s own `train.csv/test.csv/sample_submission.csv` so they don’t get mistaken as prediction files. I also add a lightweight validity check to reject “degenerate” predictions (nearly constant columns), which often slip through and drag AUC down toward 0.5. If no valid prediction files still exist, it continue to produce a valid `submission.csv` baseline.'

# 9. Code solution

## === cell 0
import os
from pathlib import Path

INPUT_ROOT = Path("/kaggle/input")
DATA_ROOT = Path("/kaggle/data")

print("Exists /kaggle/input:", INPUT_ROOT.exists())
print("Top-level /kaggle/input entries (first 50):")
if INPUT_ROOT.exists():
    print(sorted([p.name for p in INPUT_ROOT.iterdir()])[:50])

print("Exists /kaggle/data:", DATA_ROOT.exists())
print("Top-level /kaggle/data entries (first 50):")
if DATA_ROOT.exists():
    print(sorted([p.name for p in DATA_ROOT.iterdir()])[:50])



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
def discover_candidate_csvs(roots):
    candidates = []
    seen = set()
    for root in roots:
        if root is None or not Path(root).exists():
            continue

        root = Path(root)

        search_roots = []
        comp_root = root / "jigsaw-toxic-comment-classification-challenge"
        if comp_root.exists():
            search_roots.append(comp_root)
        search_roots.append(root)

        for r in search_roots:
            for p in r.rglob("*.csv"):
                sp = str(p.resolve())
                if sp in seen:
                    continue
                seen.add(sp)
                candidates.append(p)

    return candidates


candidate_files = discover_candidate_csvs([INPUT_ROOT, DATA_ROOT])

print(
    "Discovered CSV files under /kaggle/input and /kaggle/data:", len(candidate_files)
)
for p in candidate_files[:40]:
    print(" -", p)




## === cell 3
def is_known_dataset_csv(path: Path):
    name = path.name.lower()
    if name in {"train.csv", "test.csv", "sample_submission.csv"}:
        return True
    if "train" in name and name.endswith(".csv") and ("jigsaw" in str(path).lower()):
        return False
    return False


def load_and_align_pred(path: Path, sample_df: pd.DataFrame, label_cols):
    """
    Load a prediction CSV and align it to sample_submission ids and required columns.
    Returns aligned predictions as a float32 numpy array (n_rows, n_labels), or None if invalid.
    """
    if is_known_dataset_csv(path):
        return None

    try:
        df = pd.read_csv(path)
    except Exception:
        return None

    cols = list(df.columns)
    if "id" not in cols:
        return None

    missing = set(label_cols) - set(cols)
    if missing:
        return None

    df = df[["id"] + label_cols].copy()

    if len(df) < int(0.95 * len(sample_df)):
        return None

    try:
        merged = sample_df[["id"]].merge(df, on="id", how="left", validate="1:1")
    except Exception:
        return None

    if merged[label_cols].isna().any().any():
        miss_rows = int(merged[label_cols].isna().any(axis=1).sum())
        if miss_rows > 0:
            return None

    arr = merged[label_cols].to_numpy(dtype=np.float32)
    if not np.isfinite(arr).all():
        return None
    if (arr < -1e-3).any() or (arr > 1 + 1e-3).any():
        return None

    arr = np.clip(arr, 0.0, 1.0)

    col_std = arr.std(axis=0)
    if float(col_std.max()) < 1e-4:
        return None

    return arr


pred_arrays = []
used_files = []

for p in candidate_files:
    arr = load_and_align_pred(p, sample_sub, label_cols)
    if arr is not None:
        pred_arrays.append(arr)
        used_files.append(p)

print(f"Usable prediction files for blending: {len(used_files)}")
for p in used_files[:50]:
    print(" -", p)



## === cell 4
if len(pred_arrays) > 0:
    blend = np.mean(np.stack(pred_arrays, axis=0), axis=0)
else:
    print("No usable prediction CSVs found. Falling back to 0.5 baseline submission.")
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
print("Blended from files:" if used_files else "No blend files used.")
for p in used_files[:20]:
    print(" -", p)
