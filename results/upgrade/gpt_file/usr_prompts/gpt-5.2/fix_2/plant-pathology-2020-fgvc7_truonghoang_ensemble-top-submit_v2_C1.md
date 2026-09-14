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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

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
scipy==1.15.3
sklearn-pandas==2.2.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.959855568552444

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to read eight external “plantpathology” submission CSVs that are not present in your Kaggle filesystem, so `sub1`…`sub8` are never created. To make it run end-to-end and still follow the same “use existing submissions / entropy selection” core idea, I load whatever prediction CSVs actually exist in the provided `../input` tree, validate/align them to `sample_submission.csv` by `image_id`, and then apply your entropy-based per-row selection across the available files. If none are found, it safely falls back to the sample submission probabilities (uniform baseline) and still writes a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from scipy.stats import entropy



## === cell 1
INPUT_ROOT = "../input"
COMP_DIR_CANDIDATES = [
    "../input/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7/plant-pathology-2020-fgvc7",
    "../input",
]


def find_first_existing(path_list, filename):
    for d in path_list:
        p = os.path.join(d, filename)
        if os.path.exists(p):
            return p
    return None


sample_path = find_first_existing(COMP_DIR_CANDIDATES, "sample_submission.csv")
test_path = find_first_existing(COMP_DIR_CANDIDATES, "test.csv")

if sample_path is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv under expected ../input paths."
    )
if test_path is None:
    raise FileNotFoundError("Could not locate test.csv under expected ../input paths.")

sample_sub = pd.read_csv(sample_path)
test_df = pd.read_csv(test_path)

required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
for c in required_cols:
    if c not in sample_sub.columns:
        raise ValueError(f"sample_submission.csv missing required column: {c}")

target_cols = required_cols[1:]


def discover_csvs(root="../input"):
    csvs = []
    for dirpath, _, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith(".csv"):
                csvs.append(os.path.join(dirpath, fn))
    return csvs


all_csvs = discover_csvs(INPUT_ROOT)


def is_candidate_prediction_csv(df):
    cols = list(df.columns)
    if "image_id" not in cols:
        return False
    if not all(c in cols for c in target_cols):
        return False
    return True


candidates = []
for p in all_csvs:
    base = os.path.basename(p)
    if base in {"train.csv", "test.csv", "sample_submission.csv"}:
        continue
    try:
        df = pd.read_csv(p)
    except Exception:
        continue
    if is_candidate_prediction_csv(df):
        candidates.append(p)

print(f"Found {len(candidates)} candidate prediction CSV(s).")
for p in candidates[:20]:
    print("  ", p)

aligned_subs = []
used_paths = []

for p in candidates:
    df = pd.read_csv(p)

    df = df[["image_id"] + target_cols].copy()

    df = df.drop_duplicates(subset=["image_id"], keep="first")

    merged = sample_sub[["image_id"]].merge(df, on="image_id", how="left")

    if merged[target_cols].isna().any().any():
        continue

    preds = merged[target_cols].astype(float).clip(0.0, 1.0)

    aligned_subs.append(preds.to_numpy())
    used_paths.append(p)

print(f"Usable aligned prediction CSV(s): {len(aligned_subs)}")
for p in used_paths[:20]:
    print("  ", p)



## === cell 2
sub = sample_sub.copy()

if len(aligned_subs) >= 4:
    avg = (aligned_subs[0] + aligned_subs[1] + aligned_subs[2] + aligned_subs[3]) / 4.0
elif len(aligned_subs) > 0:
    avg = np.mean(np.stack(aligned_subs, axis=0), axis=0)
else:
    avg = sub[target_cols].to_numpy()

sub.loc[:, target_cols] = avg
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv (average blend). Shape:", sub.shape)



## === cell 3
sub = sample_sub.copy()

if len(aligned_subs) == 0:
    sub.to_csv("submission.csv", index=False)
    print(
        "No usable prediction CSVs found; wrote baseline submission.csv. Shape:",
        sub.shape,
    )
else:
    ent_list = []
    for arr in aligned_subs:
        ent_list.append(entropy(arr, base=2, axis=1))
    entropies = np.vstack(ent_list).T  # (n_rows, n_models)
    selected = np.argmin(entropies, axis=1)

    stacked = np.stack(aligned_subs, axis=0)  # (n_models, n_rows, n_targets)
    chosen = stacked[selected, np.arange(stacked.shape[1]), :]  # (n_rows, n_targets)

    sub.loc[:, target_cols] = chosen
    sub.to_csv("submission.csv", index=False)
    print("Wrote submission.csv (entropy selection). Shape:", sub.shape)
    print("Used models:", len(aligned_subs))
    if used_paths:
        print("First few used paths:")
        for p in used_paths[:10]:
            print("  ", p)
