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

0.96796

# 6. Current score

0.57899

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.70556) has done: 'I remove the dependency on missing external ensemble CSVs (the cause of the FileNotFoundError) and replace it with a self-contained baseline that uses only the provided competition data and images. To keep the approach lightweight and within the installed packages, this generate predictions via simple image-statistics features and per-class logistic regression in a one-vs-rest setup, then write a properly formatted `submission.csv`. I also make the data paths robust to both `/kaggle/input/...` and the provided `/kaggle/data/...` layout so it runs in your environment. This should yield a valid submission and a non-trivial AUC score versus a constant guess, moving toward the target.'
- What this solution (achieved 0.57899) has done: 'Your current pipeline is a lightweight image-bytes + one-vs-rest logistic regression baseline; to move the score up toward 0.96796 without changing the core approach, the highest-leverage minimal change is to use actual image content statistics (pixels) instead of compressed JPEG byte patterns. I keep the same training loop, model type, and loss semantics, but swap the feature extractor to use PIL (available via sklearn dependencies) to compute simple, robust RGB/HSV-like channel stats and a tiny downsampled grayscale thumbnail vector. I also add a very small amount of L2-regularization tuning (still LogisticRegression, same solver) and class_weight balancing to better handle label imbalance, which usually improves mean ROC AUC for this dataset. The script still run end-to-end within constraints and write a valid `submission.csv` with the required columns and test row order.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

CANDIDATE_ROOTS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]


def find_competition_root():
    for r in CANDIDATE_ROOTS:
        if os.path.isdir(r):
            if os.path.isfile(os.path.join(r, "train.csv")) and os.path.isdir(
                os.path.join(r, "images")
            ):
                return r
            pp = os.path.join(r, "plant-pathology-2020-fgvc7")
            if os.path.isfile(os.path.join(pp, "train.csv")) and os.path.isdir(
                os.path.join(pp, "images")
            ):
                return pp
    raise FileNotFoundError(
        "Could not locate plant-pathology-2020-fgvc7 dataset folder in known locations."
    )


DATA_ROOT = find_competition_root()
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
IMG_DIR = os.path.join(DATA_ROOT, "images")

print("DATA_ROOT:", DATA_ROOT)
print(
    "Files:",
    [
        os.path.basename(TRAIN_CSV),
        os.path.basename(TEST_CSV),
        os.path.basename(SAMPLE_SUB_CSV),
    ],
)
print(
    "Images dir exists:",
    os.path.isdir(IMG_DIR),
    "n_files:",
    len(os.listdir(IMG_DIR)) if os.path.isdir(IMG_DIR) else 0,
)

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

TARGETS = [c for c in sample_sub.columns if c != "image_id"]
assert TARGETS == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {TARGETS}"

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)
print("Targets:", TARGETS)



## === cell 1
from PIL import Image


def _safe_open_image(path: str):
    try:
        img = Image.open(path)
        img.load()
        return img
    except Exception:
        return None


def image_pixel_features(image_path: str) -> np.ndarray:
    """
    Simple, robust features from actual pixels (not compressed bytes):
      - width, height, aspect ratio
      - per-channel mean/std for RGB (6)
      - per-channel mean/std for HSV (6)  [PIL conversion, cheap]
      - grayscale thumbnail (16x16) flattened and normalized (256)
    Total dims: 3 + 6 + 6 + 256 = 271
    """
    img = _safe_open_image(image_path)
    if img is None:
        return np.zeros(271, dtype=np.float32)

    w, h = img.size
    ar = float(w) / float(h + 1e-6)

    rgb = img.convert("RGB")
    arr = np.asarray(rgb, dtype=np.float32) / 255.0  # HxWx3
    ch_mean = arr.reshape(-1, 3).mean(axis=0)
    ch_std = arr.reshape(-1, 3).std(axis=0)

    hsv = rgb.convert("HSV")
    harr = np.asarray(hsv, dtype=np.float32) / 255.0
    hsv_mean = harr.reshape(-1, 3).mean(axis=0)
    hsv_std = harr.reshape(-1, 3).std(axis=0)

    thumb = rgb.convert("L").resize((16, 16), resample=Image.BILINEAR)
    t = (np.asarray(thumb, dtype=np.float32) / 255.0).reshape(-1)

    meta = np.array([float(w), float(h), ar], dtype=np.float32)
    feats = np.concatenate(
        [
            meta,
            ch_mean.astype(np.float32),
            ch_std.astype(np.float32),
            hsv_mean.astype(np.float32),
            hsv_std.astype(np.float32),
            t.astype(np.float32),
        ],
        axis=0,
    )
    return feats.astype(np.float32)


def build_feature_matrix(image_ids: pd.Series) -> np.ndarray:
    feats = []
    missing = 0
    for iid in image_ids.tolist():
        p = os.path.join(IMG_DIR, f"{iid}.jpg")
        if not os.path.isfile(p):
            p2 = os.path.join(IMG_DIR, f"{iid}.JPG")
            if os.path.isfile(p2):
                p = p2
            else:
                missing += 1
                feats.append(np.zeros(271, dtype=np.float32))
                continue
        feats.append(image_pixel_features(p))
    if missing:
        print(f"Warning: missing {missing} images; filled features with zeros.")
    X = np.vstack(feats).astype(np.float32)
    return X


X_train = build_feature_matrix(train_df["image_id"])
X_test = build_feature_matrix(test_df["image_id"])

print("X_train:", X_train.shape, "X_test:", X_test.shape)



## === cell 2
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

Y_train = train_df[TARGETS].values.astype(np.int32)

proba_test = np.zeros((len(test_df), len(TARGETS)), dtype=np.float32)

for j, tgt in enumerate(TARGETS):
    y = Y_train[:, j]
    if y.min() == y.max():
        const = float(y.mean())
        proba_test[:, j] = const
        print(f"{tgt}: degenerate labels; using constant {const:.6f}")
        continue

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler(with_mean=True, with_std=True)),
            (
                "lr",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=3000,
                    C=2.0,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    clf.fit(X_train, y)
    proba_test[:, j] = clf.predict_proba(X_test)[:, 1].astype(np.float32)
    print(
        f"{tgt}: trained, test proba range [{proba_test[:, j].min():.4f}, {proba_test[:, j].max():.4f}]"
    )



## === cell 3
sub = sample_sub.copy()
sub = sub.merge(
    test_df[["image_id"]], on="image_id", how="right"
)  # enforce test ordering if needed
sub[TARGETS] = np.clip(proba_test, 0.0, 1.0)

assert (
    sub.shape[0] == test_df.shape[0]
), "Submission row count must match test row count."
assert (
    list(sub.columns) == ["image_id"] + TARGETS
), f"Submission columns mismatch: {sub.columns.tolist()}"

out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", sub.shape)
print(sub.head())
