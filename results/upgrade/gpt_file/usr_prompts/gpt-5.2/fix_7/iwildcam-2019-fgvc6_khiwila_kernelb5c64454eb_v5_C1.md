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
Label images of animals with their species.

## Metric
Macro F1 score

## Submission Format
```
Id,Predicted
58857ccf-23d2-11e8-a6a3-ec086b02610b,1
591e4006-23d2-11e8-a6a3-ec086b02610b,5
```

The `Id` column corresponds to the test image id. The `Category` is an integer value that indicates the class of the animal, or `0` to represent the absence of an animal.

## Dataset
The training set contains 196,157 images from 138 different locations in Southern California. 

The test set contains 153,730 images from 100 locations in Idaho.

The task is to label each image with one of the following label ids:

```
name, id
empty, 0
deer, 1
moose, 2
squirrel, 3
rodent, 4
small_mammal, 5
elk, 6
pronghorn_antelope, 7
rabbit, 8
bighorn_sheep, 9
fox, 10
coyote, 11
black_bear, 12
raccoon, 13
skunk, 14
wolf, 15
bobcat, 16
cat, 17
dog, 18
opossum, 19
bison, 20
mountain_goat, 21
mountain_lion, 22
```

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
            train/
                train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
        input/
            description.md (109 lines)
            sample_submission.csv (16878 lines)
            sample_submission.csv.zip (133.7 kB)
            test.csv (16878 lines)
            test.csv.zip (415.9 kB)
            test.zip (160 Bytes)
            test_images.zip (1.8 GB)
            train.csv (179423 lines)
            train.csv.zip (4.7 MB)
            train.zip (162 Bytes)
            train_images.zip (26.1 GB)
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
            test/
                test/
                    test/
            test_images/
                59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                ... and 16860 other files
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                ... and 179222 other files
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
        working/
            iwildcam-2019-fgvc6/
                description.md (109 lines)
                sample_submission.csv (16878 lines)
                ... and 9 other files
                iwildcam-2019-fgvc6/
                test/
                    test/
                test_images/
                    59df5d60-23d2-11e8-a6a3-ec086b02610b.jpg (140.1 kB)
                    598de852-23d2-11e8-a6a3-ec086b02610b.jpg (25.0 kB)
                    ... and 16860 other files
                    test_images/
                train/
                    train/
                train_images/
                    598624b9-23d2-11e8-a6a3-ec086b02610b.jpg (186.6 kB)
                    5a27d8f2-23d2-11e8-a6a3-ec086b02610b.jpg (202.1 kB)
                    ... and 179222 other files
                    train_images/
```

-> data/iwildcam-2019-fgvc6/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/iwildcam-2019-fgvc6/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/iwildcam-2019-fgvc6/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/sample_submission.csv has 16877 rows and 3 columns.
The columns are: Unnamed: 0, Id, Category

-> data/test.csv has 16877 rows and 10 columns.
The columns are: date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> data/train.csv has 179422 rows and 11 columns.
The columns are: category_id, date_captured, file_name, frame_num, id, location, rights_holder, seq_id, seq_num_frames, width, height

-> (stopped after 10 files for performance)

# 5. Target score

0.0861696924103301

# 6. Current score

0.05944

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.04581) has done: 'Your script didn’t yield a score because it likely produced an invalid submission for this competition: the required prediction column in the provided sample is `Category`, while your code outputs `Predicted`. I make the smallest change to align the submission schema to the competition’s expected format (`Id,Category`) while keeping your existing “sample from train label distribution” core logic intact. I also robustly drop the extra `Unnamed: 0` column from the sample submission if present and ensure we use the correct `id` field from `test.csv` to preserve row alignment with the competition’s test set.'
- What this solution (achieved 0.04763) has done: 'Your current approach samples labels from the overall training distribution, which tends to overpredict common classes like “empty” and hurts Macro F1 (because rare classes get near-zero recall). To move the score up toward your target without changing the core “distributional sampling” logic, I make the sampling conditional on `location`, using the per-location label distribution from training when available (and falling back to global otherwise). This is still the same basic stochastic label-sampling method, just better calibrated to a strong metadata shift driver in this dataset, and it remains fast and within constraints. I also keep the submission schema exactly `Id,Category` and ensure `id` alignment matches `test.csv`.'
- What this solution (achieved 0.04951) has done: 'Your current approach is still “sample a class label from a distribution,” but Macro F1 is being hurt by extreme class imbalance: rare classes get near-zero recall. To move the score upward toward your target with minimal risk and without changing the core logic, I keep the per-location distribution sampling but apply a very small amount of probability smoothing (Dirichlet/Laplace) so rare classes can be sampled sometimes. I also fix a subtle bug where the per-location probability vector can contain NaNs (when a class never appears at that location), which can break or degrade sampling; we safely fill NaNs with zeros and renormalize. Submission schema and id alignment remain exactly `Id,Category`, and runtime stays very fast.'
- What this solution (achieved 0.0534) has done: 'To move Macro F1 up toward your target while preserving your core “sample a class from (location-conditioned) label distributions” logic, I only adjust the sampling distribution so it’s less dominated by the frequent “empty” class and gives rare classes more chance, which tends to improve Macro F1 (recall across classes). Concretely: (1) add a small probability floor (epsilon) per class before normalization so every class can be sampled at each location, and (2) apply a mild temperature (<1) to flatten probabilities (reducing extreme imbalance) without changing the method. I keep the same inputs/outputs, deterministic RNG seed, per-location fallback to global, and the exact `Id,Category` submission schema. Runtime stays very fast and within the 600s constraint.'
- What this solution (achieved 0.05944) has done: 'We keep your core “sample labels from a (location-conditioned) distribution” logic unchanged, but adjust the distribution shaping slightly to push Macro F1 upward toward your target. Specifically, we reduce the dominance of the “empty” class by applying a small down-weight to class 0 in both global and per-location probability vectors, then renormalize; this tends to improve macro recall for non-empty classes with minimal code change. We also slightly increase the flattening (lower temperature) and smoothing (alpha/epsilon) just a bit to further increase rare-class sampling frequency, which usually helps Macro F1 without changing the method. Submission schema, id alignment, determinism, and runtime remain the same.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

CANDIDATE_INPUT_ROOTS = [
    "../input/iwildcam-2019-fgvc6",
    "/kaggle/input/iwildcam-2019-fgvc6",
    "../input",
    "/kaggle/input",
]


def _find_file(rel_path):
    for root in CANDIDATE_INPUT_ROOTS:
        p = (
            os.path.join(root, rel_path)
            if root.endswith("iwildcam-2019-fgvc6")
            else os.path.join(root, "iwildcam-2019-fgvc6", rel_path)
        )
        if os.path.exists(p):
            return p
    if os.path.exists(rel_path):
        return rel_path
    raise FileNotFoundError(
        f"Could not find {rel_path} under candidate roots: {CANDIDATE_INPUT_ROOTS}"
    )


for root in ["../input", "/kaggle/input"]:
    if os.path.exists(root):
        print(root, "->", os.listdir(root)[:20])



## === cell 1
train_path = _find_file("train.csv")
test_path = _find_file("test.csv")
sample_path = _find_file("sample_submission.csv")

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_submission = pd.read_csv(sample_path)

if "Unnamed: 0" in sample_submission.columns:
    sample_submission = sample_submission.drop(columns=["Unnamed: 0"])

print("train.shape:", train.shape)
print("test.shape:", test.shape)
print("sample_submission.shape:", sample_submission.shape)
print("train columns:", list(train.columns))
print("test columns:", list(test.columns))
print("sample_submission columns:", list(sample_submission.columns))



## === cell 2
from collections import Counter



## === cell 3
train_id = train["file_name"] if "file_name" in train.columns else None
labels = train["category_id"] if "category_id" in train.columns else None

if "id" in test.columns:
    test_ids = test["id"].astype(str).values
elif "Id" in sample_submission.columns:
    test_ids = sample_submission["Id"].astype(str).values
else:
    raise KeyError(
        "Could not find test ids in either test.csv ('id') or sample_submission.csv ('Id')."
    )

print("Example train file_name:", train_id.iloc[0] if train_id is not None else "N/A")
print(
    "Example train category_id:", int(labels.iloc[0]) if labels is not None else "N/A"
)
print("Example test id:", test_ids[0])



## === cell 4
label_counts_global = labels.value_counts().sort_index()
all_classes = label_counts_global.index.to_numpy()
counts_global = label_counts_global.to_numpy(dtype=np.float64)

ALPHA_GLOBAL = 0.08
TEMP_GLOBAL = 0.80  # more flattening than 0.85 -> more rare-class sampling
EPS_GLOBAL = 2e-6

probs_global = counts_global + ALPHA_GLOBAL
probs_global = probs_global + EPS_GLOBAL
probs_global = probs_global / probs_global.sum()
probs_global = np.power(probs_global, TEMP_GLOBAL)

EMPTY_CLASS_ID = 0
EMPTY_WEIGHT = 0.70
if EMPTY_CLASS_ID in set(all_classes.tolist()):
    empty_idx = int(np.where(all_classes == EMPTY_CLASS_ID)[0][0])
    probs_global[empty_idx] *= EMPTY_WEIGHT

probs_global = probs_global / probs_global.sum()

print("Number of classes in train:", len(all_classes))
print(
    "Top-10 class counts (global):\n",
    label_counts_global.sort_values(ascending=False).head(10),
)

rng = np.random.RandomState(42)

n_test = len(test_ids)

have_location = ("location" in train.columns) and ("location" in test.columns)
if have_location:
    counts_by_loc = (
        train.groupby("location")["category_id"]
        .value_counts()
        .unstack(fill_value=0)
        .reindex(columns=all_classes, fill_value=0)
    )

    ALPHA_LOC = 0.30
    TEMP_LOC = 0.80
    EPS_LOC = 8e-6

    probs_by_loc = counts_by_loc.astype(np.float64) + ALPHA_LOC
    probs_by_loc = probs_by_loc + EPS_LOC
    probs_by_loc = probs_by_loc.div(probs_by_loc.sum(axis=1), axis=0)
    probs_by_loc = probs_by_loc.fillna(0.0)

    probs_by_loc = probs_by_loc.pow(TEMP_LOC)

    if EMPTY_CLASS_ID in probs_by_loc.columns:
        probs_by_loc[EMPTY_CLASS_ID] = probs_by_loc[EMPTY_CLASS_ID] * EMPTY_WEIGHT

    row_sums = probs_by_loc.sum(axis=1).replace(0.0, np.nan)
    probs_by_loc = probs_by_loc.div(row_sums, axis=0).fillna(0.0)

    print(
        "Built per-location distributions for",
        probs_by_loc.shape[0],
        "train locations.",
    )
else:
    probs_by_loc = None
    print(
        "No 'location' in both train and test; falling back to global distribution sampling."
    )

preds = np.empty(n_test, dtype=np.int64)

if probs_by_loc is None:
    preds[:] = rng.choice(
        all_classes, size=n_test, replace=True, p=probs_global
    ).astype(int)
else:
    test_locs = test["location"].values
    counts_test = pd.Series(test_locs).value_counts()
    for loc, cnt in counts_test.items():
        if loc in probs_by_loc.index:
            p = probs_by_loc.loc[loc].to_numpy(dtype=np.float64)
            s = float(np.nansum(p))
            if (not np.isfinite(s)) or s <= 0:
                p = probs_global
            else:
                p = np.nan_to_num(p, nan=0.0, posinf=0.0, neginf=0.0)
                ss = float(p.sum())
                p = probs_global if (not np.isfinite(ss) or ss <= 0) else (p / ss)
        else:
            p = probs_global
        preds[test_locs == loc] = rng.choice(
            all_classes, size=int(cnt), replace=True, p=p
        ).astype(int)

preds = np.clip(preds, 0, 22)

print("Pred distribution (top-10):")
print(pd.Series(preds).value_counts().head(10))



## === cell 5
submission = pd.DataFrame({"Id": test_ids, "Category": preds})

assert submission.shape[0] == n_test, "Submission row count mismatch"
assert list(submission.columns) == ["Id", "Category"], "Submission columns mismatch"

out_path = "submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Category value counts (head):")
print(submission["Category"].value_counts().head(15))



## === cell 6
if submission["Id"].isna().any():
    raise ValueError("Submission contains missing Id values")

if not np.issubdtype(submission["Category"].dtype, np.integer):
    submission["Category"] = submission["Category"].astype(int)

print("Submission OK. File size (bytes):", os.path.getsize(out_path))



## === cell 7
workdir = "../working"
if os.path.exists(workdir):
    print("Contents of ../working:", os.listdir(workdir)[:50])
print("Current dir contents:", os.listdir(".")[:50])
