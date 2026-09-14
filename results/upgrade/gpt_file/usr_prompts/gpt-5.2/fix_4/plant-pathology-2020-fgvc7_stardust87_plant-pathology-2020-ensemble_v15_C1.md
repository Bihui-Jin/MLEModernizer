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

0.9700013841179632

# 6. Current score

0.45767

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because your notebook expects an external folder of prior submission files (`/kaggle/input/submissions/submissions/`) that doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,1]` fails. I keep your ensembling core logic, but add a safe fallback that generates two simple baseline “submissions” from the provided `sample_submission.csv` so the pipeline always runs end-to-end. I also add small validations (weights length, index bounds, row alignment) and ensure the output `submission.csv` has the required columns and correct row order for `test.csv`. This is primarily a correctness fix (your current score is “Not yielded”); the baseline is only to produce a valid submission file.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from producing near-constant class probabilities (fallback submissions), which yields weak ROC AUC. To move toward the 0.97 target with minimal logic change, I keep your “ensemble CSVs” approach but generate the fallback submissions from a simple, legitimate signal: the class priors from `train.csv`, which is a standard baseline and usually scores substantially better than uniform guesses. I also ensure column order and row alignment strictly follow `test.csv` and keep your weighting/ensembling intact (still ensembling two CSVs). This should improve score while preserving your pipeline semantics and producing a valid `submission.csv`.'
- What this solution (achieved 0.45767) has done: 'Your current 0.5 score is consistent with producing almost-constant probabilities (class priors and lightly smoothed priors), which yields weak mean ROC AUC. To move toward the 0.97 target while keeping your “ensemble CSVs” core logic intact, I keep the ensembling pipeline but make the fallback submissions use a legitimate per-image signal: simple image brightness statistics from the provided JPGs via PIL (no model/architecture changes, still just producing CSVs to ensemble). This create two complementary, non-constant prediction files (different calibrations) and then ensemble them with your existing weights, which should substantially improve ROC AUC versus constants. I also add strict alignment to `test.csv` order inside the fallback creation so the generated CSVs can’t silently misorder rows.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]
BASE_PATH = None
for p in BASE_PATH_CANDIDATES:
    if os.path.exists(os.path.join(p, "sample_submission.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError(
        "Could not locate sample_submission.csv and test.csv in expected Kaggle paths."
    )

SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(BASE_PATH, "test.csv")
TRAIN_CSV_PATH = os.path.join(BASE_PATH, "train.csv")

IMAGES_DIR_CANDIDATES = [
    os.path.join(BASE_PATH, "images"),
    os.path.join(BASE_PATH, "plant-pathology-2020-fgvc7", "images"),
]
IMAGES_DIR = None
for d in IMAGES_DIR_CANDIDATES:
    if os.path.isdir(d):
        IMAGES_DIR = d
        break
if IMAGES_DIR is None:
    raise FileNotFoundError(
        "Could not locate images directory under expected Kaggle paths."
    )

print("Using BASE_PATH:", BASE_PATH)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("TEST_CSV_PATH:", TEST_CSV_PATH)
print("TRAIN_CSV_PATH:", TRAIN_CSV_PATH)
print("IMAGES_DIR:", IMAGES_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"

submissions_all = []
if os.path.exists(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def _make_fallback_submissions(
    sample_sub_path: str, test_csv_path: str, train_csv_path: str, images_dir: str
) -> list:
    """
    Change (score-improving, minimal semantic impact):
    - Instead of constant priors (which tend to score ~0.5 AUC), generate per-image probabilities from a simple,
      legitimate signal available at inference: image brightness statistics from the provided JPGs.
    - Keep the same "generate two CSVs then ensemble" core logic.
    """
    from PIL import (
        Image,
    )  # PIL is available in Kaggle; using it avoids adding new external deps.

    sample = pd.read_csv(sample_sub_path)
    test = pd.read_csv(test_csv_path)
    train = pd.read_csv(train_csv_path)

    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in sample.columns]
    if missing:
        raise ValueError(f"sample_submission.csv missing columns: {missing}")

    test_ids = test["image_id"].tolist()
    sample = sample.set_index("image_id").reindex(test_ids).reset_index()
    if sample["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    for c in target_cols:
        if c not in train.columns:
            raise ValueError(f"train.csv missing required target column: {c}")

    priors = train[target_cols].mean().to_numpy(dtype=np.float64)
    priors = np.clip(priors, 1e-6, 1 - 1e-6)

    n = len(test_ids)
    brightness = np.zeros(n, dtype=np.float64)
    contrast = np.zeros(n, dtype=np.float64)

    for i, img_id in enumerate(test_ids):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        if not os.path.exists(img_path):
            brightness[i] = 0.5
            contrast[i] = 0.5
            continue

        with Image.open(img_path) as im:
            im = im.convert("L")  # grayscale
            arr = np.asarray(im, dtype=np.float32) / 255.0
            m = float(arr.mean())
            s = float(arr.std())
            brightness[i] = np.clip(m, 0.0, 1.0)
            contrast[i] = np.clip(s / 0.25, 0.0, 1.0)

    eps = 1e-9
    b = brightness - np.median(brightness)
    c = contrast - np.median(contrast)

    def logits_to_probs(L):
        L = L - L.max(axis=1, keepdims=True)
        E = np.exp(L)
        P = E / (E.sum(axis=1, keepdims=True) + eps)
        return np.clip(P, 1e-6, 1 - 1e-6)

    prior_logits = np.log(priors / (1.0 - priors))

    L1 = np.stack(
        [
            0.9 * b + 0.2 * c + 0.15 * prior_logits[0],  # healthy
            0.2 * b + 0.6 * c + 0.15 * prior_logits[1],  # multiple_diseases
            -0.3 * b + 0.3 * c + 0.15 * prior_logits[2],  # rust
            -0.4 * b + 0.5 * c + 0.15 * prior_logits[3],  # scab
        ],
        axis=1,
    )
    P1 = logits_to_probs(L1)

    L2 = np.stack(
        [
            0.6 * b + 0.1 * c + 0.20 * prior_logits[0],
            0.1 * b + 0.9 * c + 0.20 * prior_logits[1],
            -0.2 * b + 0.4 * c + 0.20 * prior_logits[2],
            -0.3 * b + 0.7 * c + 0.20 * prior_logits[3],
        ],
        axis=1,
    )
    P2 = logits_to_probs(L2)

    sub1 = sample.copy()
    sub1.loc[:, target_cols] = P1

    sub2 = sample.copy()
    sub2.loc[:, target_cols] = P2

    path1 = "fallback_submission_1.csv"
    path2 = "fallback_submission_2.csv"
    sub1.to_csv(path1, index=False)
    sub2.to_csv(path2, index=False)
    return [os.path.abspath(path1), os.path.abspath(path2)]


if len(submissions_all) == 0:
    submissions_all = _make_fallback_submissions(
        SAMPLE_SUB_PATH, TEST_CSV_PATH, TRAIN_CSV_PATH, IMAGES_DIR
    )
    print("Created fallback submissions:", submissions_all)




## === cell 4
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx must contain at least one index.")
    if len(weights) == 0:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )

    submission_with_weight = []
    base_len = None

    for i in range(len(sub_idx)):
        idx = sub_idx[i]
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={idx} out of range for submissions_all (len={len(submissions_all)})."
            )

        print(f"I'm taking submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])

        cols = ["healthy", "multiple_diseases", "rust", "scab"]
        missing_cols = [c for c in cols if c not in submission.columns]
        if missing_cols:
            raise ValueError(
                f"Submission {submissions_all[idx]} missing columns: {missing_cols}"
            )

        arr = submission.loc[:, cols].to_numpy(dtype=np.float64)

        if base_len is None:
            base_len = arr.shape[0]
        elif arr.shape[0] != base_len:
            raise ValueError(
                f"Row count mismatch across submissions: expected {base_len}, got {arr.shape[0]}"
            )

        submission_with_weight.append(arr * float(weights[i]))

    submission_avg = np.sum(submission_with_weight, axis=0)
    return submission_avg




## === cell 5
def make_submission_file(submission_avg, submissions_all):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    test_ids = test_df["image_id"].tolist()
    submission_df = submission_df.set_index("image_id").reindex(test_ids).reset_index()
    if submission_df["image_id"].isna().any():
        raise ValueError(
            "sample_submission is missing some image_id entries present in test.csv."
        )

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if submission_avg.shape != (len(submission_df), len(cols)):
        raise ValueError(
            f"submission_avg has shape {submission_avg.shape}, expected {(len(submission_df), len(cols))}"
        )

    submission_df.loc[:, cols] = submission_avg
    submission_df[cols] = submission_df[cols].clip(0.0, 1.0)

    submission_df.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("Columns:", list(submission_df.columns))
    print(submission_df.head())




## === cell 6
submission_avg = ensemble(submissions_all, [0, 1], [0.24, 0.76])
make_submission_file(submission_avg, submissions_all)
