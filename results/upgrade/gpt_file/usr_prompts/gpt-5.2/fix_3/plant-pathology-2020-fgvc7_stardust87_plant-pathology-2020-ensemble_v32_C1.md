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

0.9646650712510072

# 6. Current score

0.70048

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'You’re erroring because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,5]` crashes. I make the path detection robust by falling back to the competition’s `sample_submission.csv` and, when no external submissions are available, generate a valid baseline submission by predicting class priors from `train.csv` (score-improving vs. uniform 0.25 while keeping the “no model training” core approach). I also harden the ensembling code to skip non-CSV files, validate indices/weights, align rows by `image_id`, and always write a correct `submission.csv` with the required columns. This ensures the notebook runs end-to-end and produces a valid `.csv` submission deterministically.'
- What this solution (achieved 0.70048) has done: 'Your current 0.5 score is consistent with predicting near-uninformative constants (class priors), which can’t rank images and therefore yields ~0.5 AUC. To move toward the 0.9647 target with minimal core-logic change, I keep your “no training loop / no deep model” approach but replace constant priors with a lightweight, deterministic image-feature baseline: compute simple per-image color statistics from the provided JPGs and fit one-vs-rest logistic regression for each label. This produces per-image varying probabilities (so ROC AUC can increase substantially) while staying fast and within the installed package set (pandas + scikit-learn). I also keep your ensembling path intact; if external submissions exist, it behaves as before, otherwise it trains this baseline and writes a valid `submission.csv` in the required column order.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.exists(os.path.join(d, "sample_submission.csv")) and os.path.exists(
        os.path.join(d, "test.csv")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find competition files (sample_submission.csv, test.csv). "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")

CANDIDATE_IMAGE_DIRS = [
    os.path.join(DATA_DIR, "images"),
    os.path.join(DATA_DIR, "plant-pathology-2020-fgvc7", "images"),
]
IMAGE_DIR = None
for d in CANDIDATE_IMAGE_DIRS:
    if os.path.isdir(d):
        IMAGE_DIR = d
        break

print("Using DATA_DIR:", DATA_DIR)
print("SAMPLE_SUB_PATH:", SAMPLE_SUB_PATH)
print("IMAGE_DIR:", IMAGE_DIR)



## === cell 2
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()

print("Found submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=None, required_cols=None):
    """
    Weighted average ensemble of submission files.
    Fixes:
      - Validates indices/weights lengths to avoid IndexError.
      - Aligns by image_id to avoid row-order mismatches.
      - Filters to required_cols and returns a DataFrame with image_id + required_cols.
    """
    if required_cols is None:
        required_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files provided to ensemble().")

    ref = pd.read_csv(submissions_all[sub_idx[0]])
    if "image_id" not in ref.columns:
        raise ValueError(f"Missing image_id in {submissions_all[sub_idx[0]]}")
    ref = ref[["image_id"]].copy()

    acc = pd.DataFrame({"image_id": ref["image_id"]})
    for c in required_cols:
        acc[c] = 0.0

    for i, idx in enumerate(sub_idx):
        if idx < 0 or idx >= len(submissions_all):
            raise IndexError(
                f"Requested sub_idx={idx} but only {len(submissions_all)} files exist."
            )
        path = submissions_all[idx]
        w = float(weights[i])
        print(f"I'm taking submission {path} with weight {w}")

        sub = pd.read_csv(path)
        missing = [c for c in (["image_id"] + required_cols) if c not in sub.columns]
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")

        sub = sub[["image_id"] + required_cols].copy()
        sub = ref.merge(sub, on="image_id", how="left", validate="one_to_one")
        if sub[required_cols].isna().any().any():
            raise ValueError(
                f"{path} has missing predictions for some image_id after alignment."
            )

        for c in required_cols:
            acc[c] += sub[c].astype(float) * w

    return acc




## === cell 4
def make_submission_file(submission_df, out_path="submission.csv"):
    """
    Writes a valid submission CSV with correct columns/order.
    Fix: ensure required header and exact columns.
    """
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    missing = [c for c in required_cols if c not in submission_df.columns]
    if missing:
        raise ValueError(f"submission_df missing columns: {missing}")

    submission_df = submission_df[required_cols].copy()
    submission_df.to_csv(out_path, index=False)
    print("Wrote:", out_path, "shape=", submission_df.shape)




## === cell 5
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler


def _img_path_from_id(image_id, image_dir):
    return os.path.join(image_dir, f"{image_id}.jpg")


def extract_basic_rgb_stats(image_ids, image_dir, size=(128, 128)):
    """
    Deterministic, fast features:
      - per-channel mean, std, min, max (R,G,B): 12
      - grayscale mean/std: 2
      - simple vegetation-like index mean/std (G - R) / (G + R + eps): 2
    Total: 16 features per image.
    """
    feats = np.zeros((len(image_ids), 16), dtype=np.float32)
    eps = 1e-6

    for i, image_id in enumerate(image_ids):
        p = _img_path_from_id(image_id, image_dir)
        if not os.path.exists(p):
            raise FileNotFoundError(f"Image not found: {p}")

        im = Image.open(p).convert("RGB")
        if size is not None:
            im = im.resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3

        r = arr[..., 0]
        g = arr[..., 1]
        b = arr[..., 2]

        stats = []
        for ch in (r, g, b):
            stats.extend([ch.mean(), ch.std(), ch.min(), ch.max()])

        gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
        stats.extend([gray.mean(), gray.std()])

        idx = (g - r) / (g + r + eps)
        stats.extend([idx.mean(), idx.std()])

        feats[i, :] = np.array(stats, dtype=np.float32)

    return feats


def fit_ovr_logreg_predict_proba(X_train, y_train_df, X_test, class_names):
    """
    Fits separate binary LogisticRegression models per class (OvR),
    returning test probabilities in the required column order.
    """
    scaler = StandardScaler()
    Xtr = scaler.fit_transform(X_train)
    Xte = scaler.transform(X_test)

    preds = {}
    for c in class_names:
        y = y_train_df[c].astype(int).values

        clf = LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=1.0,
            n_jobs=None,
            random_state=0,
        )
        clf.fit(Xtr, y)
        p = clf.predict_proba(Xte)[:, 1]
        preds[c] = np.clip(p.astype(np.float64), 1e-6, 1 - 1e-6)

    return preds




## === cell 6
required_cols = ["healthy", "multiple_diseases", "rust", "scab"]

if len(submissions_all) >= 1:
    desired_idx = [0, 2, 5]
    if len(submissions_all) > max(desired_idx, default=0):
        sub_idx = desired_idx
        weights = [0.1, 0.8, 0.1]
    else:
        sub_idx = list(range(len(submissions_all)))
        weights = [1.0 / len(submissions_all)] * len(submissions_all)

    submission_df = ensemble(
        submissions_all, sub_idx=sub_idx, weights=weights, required_cols=required_cols
    )
    make_submission_file(submission_df, out_path="submission.csv")
else:
    if IMAGE_DIR is None:
        raise FileNotFoundError(
            "No external submissions found and could not locate images/ directory for feature extraction baseline."
        )

    train = pd.read_csv(TRAIN_CSV_PATH)
    test = pd.read_csv(TEST_CSV_PATH)

    train_ids = train["image_id"].astype(str).tolist()
    test_ids = test["image_id"].astype(str).tolist()

    X_train = extract_basic_rgb_stats(train_ids, IMAGE_DIR, size=(128, 128))
    X_test = extract_basic_rgb_stats(test_ids, IMAGE_DIR, size=(128, 128))

    preds = fit_ovr_logreg_predict_proba(
        X_train, train[required_cols], X_test, required_cols
    )

    submission_df = pd.DataFrame({"image_id": test_ids})
    for c in required_cols:
        submission_df[c] = preds[c]

    make_submission_file(submission_df, out_path="submission.csv")
