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

0.9663173272786793

# 6. Current score

0.65822

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The current notebook fails because it tries to ensemble several external submission CSVs that don’t exist in this Kaggle environment, so `dsub` is never created and the later averaging cell crashes. I keep the same “average multiple submissions” core logic, but make it robust by (1) loading any available candidate prediction files if present and (2) otherwise falling back to a valid baseline submission built from `sample_submission.csv` (uniform 0.25s), guaranteeing a correct `.csv` output. I also fix the input pathing to use the provided `/kaggle/data/...` structure (and optionally `/kaggle/input/...` if present) and ensure columns align to `healthy, multiple_diseases, rust, scab`. This run end-to-end and always write `submission.csv`.'
- What this solution (achieved 0.65822) has done: 'Your current score (0.5) comes from the uniform 0.25 fallback, so we need a real model to move toward the target 0.9663. Since your environment only has numpy/pandas and no deep learning libraries, the smallest legitimate improvement is to replace the missing external-CSV ensemble with an in-notebook classical image pipeline: read images, extract simple color/texture features, and train one-vs-rest logistic regression models for the four labels. This preserves the “predict per-class probabilities” semantics and reliably outputs a valid `submission.csv` aligned to `sample_submission.csv`. I also keep your robust path detection and only change the part that currently can’t possibly improve without the unavailable external files.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
DATA_ROOT_CANDIDATES = [
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
]
DATA_ROOT = None
for p in DATA_ROOT_CANDIDATES:
    if os.path.exists(p):
        if os.path.exists(os.path.join(p, "sample_submission.csv")):
            DATA_ROOT = p
            break
        if os.path.exists(
            os.path.join(p, "plant-pathology-2020-fgvc7", "sample_submission.csv")
        ):
            DATA_ROOT = os.path.join(p, "plant-pathology-2020-fgvc7")
            break
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate competition data folder in expected Kaggle paths."
    )

print("Using DATA_ROOT:", DATA_ROOT)
print("Files in DATA_ROOT (first 20):", sorted(os.listdir(DATA_ROOT))[:20])



## === cell 2
sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))
test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))
train = pd.read_csv(os.path.join(DATA_ROOT, "train.csv"))

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

assert all(c in sub.columns for c in ["image_id"] + TARGET_COLS)
assert (
    list(sub.columns) == ["image_id"] + TARGET_COLS
), "Submission column order mismatch vs sample_submission.csv"

print(
    "train shape:",
    train.shape,
    "test shape:",
    test.shape,
    "sample_submission shape:",
    sub.shape,
)
train.head()



## === cell 3
from PIL import Image
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

IMG_DIR_CANDIDATES = [
    os.path.join(DATA_ROOT, "images"),
    os.path.join(os.path.dirname(DATA_ROOT), "images"),
    "/kaggle/data/images",
    "/kaggle/input/images",
]
IMG_DIR = None
for d in IMG_DIR_CANDIDATES:
    if os.path.exists(d):
        IMG_DIR = d
        break
if IMG_DIR is None:
    raise FileNotFoundError("Could not locate images/ folder in expected paths.")
print("Using IMG_DIR:", IMG_DIR)


def _safe_open_rgb(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def extract_features(image_id, size=96):
    """
    Lightweight feature extractor:
    - resize to small fixed size
    - per-channel mean/std
    - HSV mean/std
    - simple gradient magnitude mean/std (on grayscale)
    These features are fast and usually beat uniform probabilities substantially.
    """
    path = os.path.join(IMG_DIR, f"{image_id}.jpg")
    im = _safe_open_rgb(path).resize((size, size), resample=Image.BILINEAR)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)

    rgb_mean = arr.reshape(-1, 3).mean(axis=0)
    rgb_std = arr.reshape(-1, 3).std(axis=0)

    hsv = np.asarray(im.convert("HSV"), dtype=np.float32) / 255.0
    hsv_mean = hsv.reshape(-1, 3).mean(axis=0)
    hsv_std = hsv.reshape(-1, 3).std(axis=0)

    gray = (
        0.2989 * arr[:, :, 0] + 0.5870 * arr[:, :, 1] + 0.1140 * arr[:, :, 2]
    ).astype(np.float32)
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)
    gx = np.pad(gx, ((0, 0), (0, 1)), mode="edge")
    gy = np.pad(gy, ((0, 1), (0, 0)), mode="edge")
    gmag = np.sqrt(gx * gx + gy * gy)
    g_mean = np.array([gmag.mean()], dtype=np.float32)
    g_std = np.array([gmag.std()], dtype=np.float32)

    return np.concatenate([rgb_mean, rgb_std, hsv_mean, hsv_std, g_mean, g_std], axis=0)


train_ids = train["image_id"].tolist()
test_ids = test["image_id"].tolist()

X_train = np.vstack([extract_features(i) for i in train_ids])
X_test = np.vstack([extract_features(i) for i in test_ids])

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 4
y_mat = train[TARGET_COLS].astype(int).values
y_strat = y_mat.argmax(axis=1)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

test_pred = np.zeros((len(test_ids), len(TARGET_COLS)), dtype=np.float64)

for t, col in enumerate(TARGET_COLS):
    y = y_mat[:, t]
    col_test_pred = np.zeros(len(test_ids), dtype=np.float64)

    for fold, (tr_idx, va_idx) in enumerate(skf.split(X_train, y_strat), 1):
        X_tr, X_va = X_train[tr_idx], X_train[va_idx]
        y_tr = y[tr_idx]

        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_va_s = scaler.transform(X_va)
        X_te_s = scaler.transform(X_test)

        clf = LogisticRegression(
            C=2.0,
            solver="lbfgs",
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )
        clf.fit(X_tr_s, y_tr)

        col_test_pred += clf.predict_proba(X_te_s)[:, 1] / skf.n_splits

    test_pred[:, t] = col_test_pred

test_pred = np.clip(test_pred, 0.0, 1.0)



## === cell 5
out = sub[["image_id"]].merge(test[["image_id"]], on="image_id", how="right")
for j, col in enumerate(TARGET_COLS):
    out[col] = test_pred[:, j].astype(float)

out = out[["image_id"] + TARGET_COLS]

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Per-column mean preds:", {c: float(out[c].mean()) for c in TARGET_COLS})
