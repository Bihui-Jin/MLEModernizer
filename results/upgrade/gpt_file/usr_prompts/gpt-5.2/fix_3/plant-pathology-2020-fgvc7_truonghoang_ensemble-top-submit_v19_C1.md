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

0.97113

# 6. Current score

0.69751

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.52238) has done: 'I remove the dependency on missing external submission files under `../input/plantpathology/` (the cause of the FileNotFoundError) and instead generate predictions from the provided train/test/images in this environment. Because only NumPy/Pandas/OS are available, I implement a simple, deterministic image-feature + one-vs-rest logistic regression (trained via batch gradient descent) to produce valid probabilities for the 4 required columns. I also fix the dataset pathing to use the existing `../input/plant-pathology-2020-fgvc7/` (and fall back to `../input/`), and ensure the submission columns exactly match `sample_submission.csv`. The script run end-to-end and write `submission.csv` with the correct header and 183 rows.'
- What this solution (achieved 0.69751) has done: 'Your current 0.522 AUC is far below the 0.971 target, so we should improve score with minimal core-logic changes by making the existing logistic-regression-on-handcrafted-features actually learn from image content rather than filename/size metadata. I keep the same one-vs-rest logistic regression, gradient-descent training loop, sigmoid, and submission semantics, but replace/extend `build_features()` to extract simple, deterministic pixel statistics from each JPEG (downsampled) using only NumPy (no new packages). This adds real signal for disease patterns while staying lightweight and within time, and it should move AUC substantially upward toward the target band. I also make feature extraction deterministic and cached in-memory to avoid redundant file reads.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import zipfile
from pathlib import Path



## === cell 1
CANDIDATE_ROOTS = [
    Path("../input/plant-pathology-2020-fgvc7"),
    Path("../input"),
    Path("/kaggle/input/plant-pathology-2020-fgvc7"),
    Path("/kaggle/input"),
    Path("../data/plant-pathology-2020-fgvc7"),
    Path("../data"),
    Path("/kaggle/data/plant-pathology-2020-fgvc7"),
    Path("/kaggle/data"),
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if (r / "train.csv").exists() and (r / "test.csv").exists():
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not locate train.csv/test.csv under expected ../input or /kaggle paths."
    )

TRAIN_CSV = DATA_ROOT / "train.csv"
TEST_CSV = DATA_ROOT / "test.csv"
SAMPLE_SUB = DATA_ROOT / "sample_submission.csv"
IMAGES_DIR = DATA_ROOT / "images"
IMAGES_ZIP = DATA_ROOT / "images.zip"

print("Using DATA_ROOT:", DATA_ROOT)
print("Has images dir:", IMAGES_DIR.exists(), "Has images.zip:", IMAGES_ZIP.exists())

if not IMAGES_DIR.exists():
    if IMAGES_ZIP.exists():
        IMAGES_DIR.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(IMAGES_ZIP, "r") as zf:
            zf.extractall(DATA_ROOT)
    else:
        raise FileNotFoundError("Neither images/ directory nor images.zip found.")

print("Images dir:", IMAGES_DIR, "num files:", len(list(IMAGES_DIR.glob("*.jpg"))))



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

TARGETS = [c for c in sample_sub.columns if c != "image_id"]
expected_targets = ["healthy", "multiple_diseases", "rust", "scab"]
if TARGETS != expected_targets:
    if all(t in sample_sub.columns for t in expected_targets):
        TARGETS = expected_targets
    else:
        raise ValueError(
            f"Unexpected target columns in sample_submission: {sample_sub.columns.tolist()}"
        )

assert "image_id" in train_df.columns and "image_id" in test_df.columns
for t in TARGETS:
    if t not in train_df.columns:
        raise ValueError(f"Train is missing target column: {t}")

train_df.head(), test_df.head(), sample_sub.head()




## === cell 3
def jpeg_size(path: Path):
    """
    Return (width, height) for a JPEG file by parsing markers.
    Fallback to (np.nan, np.nan) if parsing fails.
    """
    try:
        with open(path, "rb") as f:
            data = f.read(2048)  # usually enough to hit SOF marker
        if len(data) < 4 or data[0:2] != b"\xff\xd8":
            return (np.nan, np.nan)
        i = 2
        while i + 9 < len(data):
            if data[i] != 0xFF:
                i += 1
                continue
            while i < len(data) and data[i] == 0xFF:
                i += 1
            if i >= len(data):
                break
            marker = data[i]
            i += 1
            if marker in (0xD8, 0xD9):
                continue
            if i + 2 > len(data):
                break
            seglen = int.from_bytes(data[i : i + 2], "big")
            if seglen < 2:
                break
            if marker in (
                0xC0,
                0xC1,
                0xC2,
                0xC3,
                0xC5,
                0xC6,
                0xC7,
                0xC9,
                0xCA,
                0xCB,
                0xCD,
                0xCE,
                0xCF,
            ):
                start = i + 2  # after seglen
                if start + 5 <= len(data):
                    height = int.from_bytes(data[start + 1 : start + 3], "big")
                    width = int.from_bytes(data[start + 3 : start + 5], "big")
                    return (width, height)
                break
            i += seglen
        return (np.nan, np.nan)
    except Exception:
        return (np.nan, np.nan)


def _read_jpeg_bytes(path: Path):
    try:
        with open(path, "rb") as f:
            return f.read()
    except Exception:
        return None


def _byte_hist_feats(b: bytes, n_bins=16, max_bytes=200_000):
    """
    Deterministic, fast "pseudo-image" features from raw JPEG bytes:
    - histogram over byte values (downsampled to n_bins)
    - mean/std of bytes
    - a few segment-wise means to capture coarse structure
    """
    if b is None or len(b) == 0:
        return np.full(n_bins + 2 + 4, np.nan, dtype=float)
    arr = np.frombuffer(b[:max_bytes], dtype=np.uint8).astype(np.float32)
    bins = np.linspace(0, 256, n_bins + 1, dtype=np.float32)
    hist, _ = np.histogram(arr, bins=bins)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-6)
    m = float(arr.mean())
    s = float(arr.std())
    L = arr.shape[0]
    q1 = float(arr[: max(1, L // 4)].mean())
    q2 = float(arr[L // 4 : max(L // 2, L // 4 + 1)].mean())
    q3 = float(arr[L // 2 : max(3 * L // 4, L // 2 + 1)].mean())
    q4 = float(arr[3 * L // 4 :].mean())
    return np.concatenate([hist, np.array([m, s, q1, q2, q3, q4], dtype=np.float32)])


def build_features(df: pd.DataFrame):
    widths = []
    heights = []
    sizes = []
    name_feats = []
    content_feats = []

    for img_id in df["image_id"].astype(str).tolist():
        p = IMAGES_DIR / f"{img_id}.jpg"
        if not p.exists():
            matches = list(IMAGES_DIR.glob(f"{img_id}.*"))
            p = matches[0] if matches else p

        w, h = jpeg_size(p) if p.exists() else (np.nan, np.nan)
        widths.append(w)
        heights.append(h)
        sizes.append(p.stat().st_size if p.exists() else np.nan)

        s = sum(ord(ch) for ch in img_id)
        name_feats.append([len(img_id), s % 997, (s % 97) / 97.0])

        b = _read_jpeg_bytes(p) if p.exists() else None
        content_feats.append(_byte_hist_feats(b, n_bins=16))

    X = pd.DataFrame({"w": widths, "h": heights, "filesize": sizes})
    name_feats = np.array(name_feats, dtype=float)
    X["name_len"] = name_feats[:, 0]
    X["name_mod997"] = name_feats[:, 1]
    X["name_mod97"] = name_feats[:, 2]
    X["aspect"] = X["w"] / (X["h"] + 1e-6)

    content_feats = np.vstack(content_feats).astype(float)
    for i in range(16):
        X[f"bytehist_{i}"] = content_feats[:, i]
    X["byte_mean"] = content_feats[:, 16]
    X["byte_std"] = content_feats[:, 17]
    X["byte_q1"] = content_feats[:, 18]
    X["byte_q2"] = content_feats[:, 19]
    X["byte_q3"] = content_feats[:, 20]
    X["byte_q4"] = content_feats[:, 21]
    return X


X_train_df = build_features(train_df)
X_test_df = build_features(test_df)

med = X_train_df.median(numeric_only=True)
X_train_df = X_train_df.fillna(med)
X_test_df = X_test_df.fillna(med)

X_train_df.head(), X_test_df.head()




## === cell 4
def sigmoid(z):
    z = np.clip(z, -30, 30)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma = np.where(sigma < 1e-12, 1.0, sigma)
    return mu, sigma


def standardize_transform(X, mu, sigma):
    return (X - mu) / sigma


def train_logreg_ovr(X, Y, lr=0.15, n_iter=1200, l2=1e-2):
    """
    Train independent logistic regressions for each column in Y.
    X: (n, d)
    Y: (n, k) binary
    Returns W: (d+1, k) including bias.
    """
    n, d = X.shape
    k = Y.shape[1]
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)  # bias
    W = np.zeros((d + 1, k), dtype=float)
    for j in range(k):
        w = np.zeros(d + 1, dtype=float)
        y = Y[:, j].astype(float)
        for _ in range(n_iter):
            p = sigmoid(Xb @ w)
            grad = (Xb.T @ (p - y)) / n
            grad[1:] += l2 * w[1:]
            w -= lr * grad
        W[:, j] = w
    return W


def predict_logreg_ovr(X, W):
    n = X.shape[0]
    Xb = np.concatenate([np.ones((n, 1)), X], axis=1)
    P = sigmoid(Xb @ W)
    return np.clip(P, 1e-6, 1 - 1e-6)


X_train = X_train_df.to_numpy(dtype=float)
X_test = X_test_df.to_numpy(dtype=float)
Y_train = train_df[TARGETS].to_numpy(dtype=float)

mu, sigma = standardize_fit(X_train)
X_train_s = standardize_transform(X_train, mu, sigma)
X_test_s = standardize_transform(X_test, mu, sigma)

W = train_logreg_ovr(X_train_s, Y_train, lr=0.15, n_iter=1200, l2=1e-2)
P_test = predict_logreg_ovr(X_test_s, W)

P_test.shape



## === cell 5
sub = sample_sub.copy()
sub = sub[["image_id"] + TARGETS].copy()

pred_df = pd.DataFrame(P_test, columns=TARGETS)
sub.loc[:, TARGETS] = pred_df[TARGETS].values

for c in TARGETS:
    sub[c] = pd.to_numeric(sub[c], errors="coerce").fillna(0.25).clip(0.0, 1.0)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
sub.head()
