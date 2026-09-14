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

0.9692618738507616

# 6. Current score

0.5503

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.55365) has done: 'I remove the dependency on missing external Kaggle datasets (the `../input/plantpathology/*.csv` files) that cause the `FileNotFoundError`, and instead generate predictions directly from the provided competition data. To keep the core approach simple and stable in this limited package environment (no deep learning libraries installed), I build a lightweight, deterministic image-feature + multi-label logistic regression pipeline using scikit-learn (available in Kaggle by default) and PIL for image reading. This fixes the runtime errors, ensures predictions align with `test.csv` ordering, and writes a valid `submission.csv` with the exact required columns. The model should yield a non-trivial AUC (typically much better than all-0.25 baselines) without changing any evaluation semantics.'
- What this solution (achieved 0.47194) has done: 'Your current score is far below the target, so the safest way to move toward it (without changing the overall “image features + multi-label logistic regression” core approach) is to fix two performance bottlenecks: overly high-dimensional raw pixels (which makes the linear model unstable/underfit at this sample size) and too-strong default regularization. I keep the same pipeline structure, but extract a compact, more robust feature vector (downsampled grayscale + simple gradient and color stats) and increase `C` to reduce underfitting. I also ensure the class order from `predict_proba` is mapped back to the correct target columns (avoids silent column misalignment). These are minimal, metric-aligned changes that should substantially improve ROC AUC while preserving the same training approach.'
- What this solution (achieved 0.47194) has done: 'Your current score is far below the target, so we should cautiously improve performance while keeping the same “handcrafted image features + OneVsRest logistic regression” approach. The biggest likely issue here is that the `classes_` mapping logic is incorrect for `OneVsRestClassifier` in multilabel mode (it is not a per-target index list), which can silently permute/misalign columns and crush mean ROC AUC. I remove that remapping and instead directly use the returned probability columns in the same order as `target_cols` (the order used to build `Y_train`). I also keep everything else the same to minimize risk and preserve evaluation semantics.'
- What this solution (achieved 0.5503) has done: 'Your current score is far below the target, so we should improve discriminative signal while keeping the same “handcrafted image features + StandardScaler + OneVsRest LogisticRegression” core pipeline intact. The most likely underperformance is that raw flattened grayscale+gradient pixels are too high-dimensional and noisy for a linear model at this dataset size, leading to weak AUC. I keep the same model/training semantics but make the features more robust by (1) using a slightly larger downsample size, (2) adding low-cost global color/texture histograms, and (3) adding a simple “green dominance” vegetation index that helps separate healthy vs diseased leaves. These are minimal, deterministic feature additions that typically boost ROC AUC for linear models without changing the learning algorithm, and the script still write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
    "../input",
]

BASE = None
for c in BASE_CANDIDATES:
    if os.path.exists(c):
        if os.path.isfile(os.path.join(c, "train.csv")) and os.path.isdir(
            os.path.join(c, "images")
        ):
            BASE = c
            break

if BASE is None:
    roots = ["/kaggle/input", "/kaggle/data", "../input", "../kaggle/data", "/"]
    found = []
    for r in roots:
        if os.path.exists(r):
            for dirpath, dirnames, filenames in os.walk(r):
                if (
                    "train.csv" in filenames
                    and "test.csv" in filenames
                    and "images" in dirnames
                ):
                    found.append(dirpath)
            if found:
                break
    if found:
        BASE = found[0]

if BASE is None:
    raise FileNotFoundError(
        "Could not locate competition dataset folder containing train.csv/test.csv/images."
    )

print("Using BASE:", BASE)
print(
    "Files:",
    [
        f
        for f in ["train.csv", "test.csv", "sample_submission.csv"]
        if os.path.exists(os.path.join(BASE, f))
    ],
)

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE, "images")



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert "image_id" in sample_sub.columns
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train CSV missing expected target columns."
assert all(
    c in sample_sub.columns for c in ["image_id"] + target_cols
), "Sample submission missing expected columns."

print("Train shape:", train_df.shape)
print("Test shape:", test_df.shape)
print("Targets:", target_cols)



## === cell 3
try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL (Pillow) is required to read images but is not available in this environment."
    ) from e


def image_path(image_id: str) -> str:
    return (
        os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        if not image_id.lower().endswith(".jpg")
        else os.path.join(IMAGES_DIR, image_id)
    )


def extract_features_one(img_fp: str, size=(48, 48), hist_bins=16) -> np.ndarray:
    with Image.open(img_fp) as im:
        im = im.convert("RGB").resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H, W, 3)

    r = arr[..., 0]
    g = arr[..., 1]
    b = arr[..., 2]

    gray = (0.2989 * r + 0.5870 * g + 0.1140 * b).astype(np.float32)

    gx = np.diff(gray, axis=1, append=gray[:, -1:])
    gy = np.diff(gray, axis=0, append=gray[-1:, :])
    gmag = np.sqrt(gx * gx + gy * gy).astype(np.float32)

    ch_mean = arr.mean(axis=(0, 1)).astype(np.float32)
    ch_std = arr.std(axis=(0, 1)).astype(np.float32)
    gray_stats = np.array([gray.mean(), gray.std()], dtype=np.float32)
    grad_stats = np.array(
        [gmag.mean(), gmag.std(), np.percentile(gmag, 90)], dtype=np.float32
    )

    eps = 1e-6
    exg = (2.0 * g - r - b).astype(np.float32)  # excess green
    gr_ratio = (g + eps) / (r + eps)
    gb_ratio = (g + eps) / (b + eps)
    rg_diff = (r - g).astype(np.float32)
    idx_stats = np.array(
        [
            exg.mean(),
            exg.std(),
            np.percentile(exg, 10),
            np.percentile(exg, 90),
            gr_ratio.mean(),
            gb_ratio.mean(),
            rg_diff.mean(),
            rg_diff.std(),
        ],
        dtype=np.float32,
    )

    def _hist(x01: np.ndarray, bins: int) -> np.ndarray:
        h, _ = np.histogram(x01.reshape(-1), bins=bins, range=(0.0, 1.0), density=True)
        return h.astype(np.float32)

    hist_gray = _hist(gray, hist_bins)
    hist_r = _hist(r, hist_bins)
    hist_g = _hist(g, hist_bins)
    hist_b = _hist(b, hist_bins)
    hist_gmag = _hist(np.clip(gmag, 0.0, 1.0), hist_bins)

    flat_gray = gray.reshape(-1).astype(np.float32)
    flat_gmag = gmag.reshape(-1).astype(np.float32)

    feat = np.concatenate(
        [
            flat_gray,
            flat_gmag,
            ch_mean,
            ch_std,
            gray_stats,
            grad_stats,
            idx_stats,
            hist_gray,
            hist_r,
            hist_g,
            hist_b,
            hist_gmag,
        ],
        axis=0,
    )
    return feat.astype(np.float32)


def extract_features(df: pd.DataFrame) -> np.ndarray:
    feats = []
    missing = 0
    for image_id in df["image_id"].astype(str).tolist():
        fp = image_path(image_id)
        if not os.path.exists(fp):
            alt = os.path.join(IMAGES_DIR, image_id)
            if os.path.exists(alt):
                fp = alt
            else:
                missing += 1
                feats.append(None)
                continue
        feats.append(extract_features_one(fp))
    if missing:
        print(f"Warning: missing {missing} images; using zero features for them.")
    first = next((f for f in feats if f is not None), None)
    if first is None:
        raise RuntimeError(
            "No images were found/read successfully; cannot build features."
        )
    dim = int(first.shape[0])
    X = np.zeros((len(feats), dim), dtype=np.float32)
    for i, f in enumerate(feats):
        if f is not None:
            X[i] = f
    return X


X_train = extract_features(train_df)
X_test = extract_features(test_df)

Y_train = train_df[target_cols].astype(np.float32).values

print("X_train:", X_train.shape, "Y_train:", Y_train.shape, "X_test:", X_test.shape)



## === cell 4
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier

base_clf = LogisticRegression(
    solver="liblinear",
    max_iter=4000,
    C=3.0,
    random_state=RANDOM_STATE,
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", OneVsRestClassifier(base_clf, n_jobs=None)),
    ]
)

model.fit(X_train, Y_train)

proba = model.predict_proba(X_test).astype(np.float32)  # shape: (n_samples, n_targets)
if proba.ndim != 2 or proba.shape[1] != len(target_cols):
    raise RuntimeError(
        f"Unexpected predict_proba shape {proba.shape}; expected (n_samples, {len(target_cols)})."
    )

proba = np.clip(proba, 0.0, 1.0)
print("Pred proba shape:", proba.shape)



## === cell 5
sub = pd.DataFrame({"image_id": test_df["image_id"].astype(str).values})
for j, c in enumerate(target_cols):
    sub[c] = proba[:, j].astype(np.float32)

sub = sub[["image_id"] + target_cols]

assert sub.shape[0] == test_df.shape[0]
assert list(sub.columns) == list(sample_sub.columns)

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(sub.head())
