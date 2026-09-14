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

0.9697366773573868

# 6. Current score

0.56359

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.49496) has done: 'I fix the runtime error by removing the hardcoded `../input/plantpathology/*.csv` dependencies (those files don’t exist in your environment) and instead generate predictions directly from the provided train/test images. To keep changes minimal while ensuring a solid score, I use a simple, deterministic scikit-learn pipeline: load images, extract small grayscale features, train a multi-output logistic regression, and predict probabilities for each label. I also ensure the submission columns and `image_id` ordering exactly match `sample_submission.csv`, and write `submission.csv` with the required suffix.'
- What this solution (achieved 0.53083) has done: 'Your current score is far below the target, so we should improve feature quality and model capacity while keeping the same core approach (deterministic image features → scikit-learn multi-output logistic regression → probability submission). The biggest issue is that 64×64 grayscale flattening discards most discriminative color/texture cues, so we minimally extend the feature extractor to include (1) color information, (2) simple edge/gradient cues, and (3) a few global statistics; this remains lightweight and deterministic. We also apply class-balanced training per label (via `class_weight="balanced"` inside each logistic regression) to handle label imbalance, which typically improves mean ROC AUC without changing evaluation semantics. Submission alignment still strictly follow `sample_submission.csv` and `test.csv` ordering and write `submission.csv`.'
- What this solution (achieved 0.56359) has done: 'Your current score (0.53083) is far below the target (0.9697), so we should improve discriminative power while keeping the same “handcrafted image features → standardize → multi-output logistic regression” core pipeline. The minimal high-impact change is to make the features more informative without changing the model family: add compact color histograms + simple color indices (helps separate rust/scab patterns) while keeping your existing downsampled RGB/gray/edge features. I also switch the logistic regression solver to a multinomial-capable one with the same linear model (still logistic regression) to fit probabilities more stably on high-dimensional standardized features. Submission writing/alignment stays identical and still produces `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_PATH = "/kaggle/data/plant-pathology-2020-fgvc7"
if not os.path.exists(BASE_PATH):
    BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_PATH, "sample_submission.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "images")

print("BASE_PATH:", BASE_PATH)
print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print("Exists IMAGES_DIR:", os.path.exists(IMAGES_DIR))



## === cell 2
from PIL import Image

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

target_cols = [c for c in sample_sub.columns if c != "image_id"]
assert target_cols == [
    "healthy",
    "multiple_diseases",
    "rust",
    "scab",
], f"Unexpected target columns: {target_cols}"


def _resolve_image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def _hist_channel(ch_0_1: np.ndarray, bins: int = 16) -> np.ndarray:
    h, _ = np.histogram(ch_0_1, bins=bins, range=(0.0, 1.0), density=True)
    return h.astype(np.float32)


def extract_features(image_ids, size=(96, 96), hist_bins=16):
    """
    Change (score ↑ toward target): Improve features while preserving the same core pipeline
    (deterministic handcrafted image features -> multi-output logistic regression).

    Keep existing:
      - downsampled RGB
      - downsampled grayscale
      - simple edge magnitude
      - global per-channel stats

    Add minimally:
      - per-channel color histograms (16 bins each) + gray histogram
      - simple vegetation/color indices (ExG/ExR) summary stats

    Still PIL+NumPy only, deterministic, and fast enough for this dataset size.
    """
    n = len(image_ids)
    H, W = size

    d_rgb = 3 * H * W
    d_gray = H * W
    d_edge = H * W
    d_stats = 12

    d_hist = (3 + 1) * hist_bins  # R,G,B,Gray histograms
    d_idx_stats = 8  # mean/std/min/max for ExG and ExR

    d = d_rgb + d_gray + d_edge + d_stats + d_hist + d_idx_stats
    X = np.empty((n, d), dtype=np.float32)

    for i, img_id in enumerate(image_ids):
        p = _resolve_image_path(img_id)
        with Image.open(p) as im:
            im_rgb = im.convert("RGB").resize((W, H), Image.BILINEAR)
            arr_rgb = np.asarray(im_rgb, dtype=np.float32) / 255.0  # (H,W,3)

        rgb_flat = arr_rgb.reshape(-1)

        gray = (
            0.299 * arr_rgb[..., 0] + 0.587 * arr_rgb[..., 1] + 0.114 * arr_rgb[..., 2]
        ).astype(np.float32)
        gray_flat = gray.reshape(-1)

        gx = np.zeros_like(gray, dtype=np.float32)
        gy = np.zeros_like(gray, dtype=np.float32)
        gx[:, 1:] = gray[:, 1:] - gray[:, :-1]
        gy[1:, :] = gray[1:, :] - gray[:-1, :]
        edge = np.sqrt(gx * gx + gy * gy).astype(np.float32)
        edge_flat = edge.reshape(-1)

        stats = []
        for c in range(3):
            ch = arr_rgb[..., c]
            stats.extend(
                [float(ch.mean()), float(ch.std()), float(ch.min()), float(ch.max())]
            )
        stats = np.asarray(stats, dtype=np.float32)

        h_r = _hist_channel(arr_rgb[..., 0], bins=hist_bins)
        h_g = _hist_channel(arr_rgb[..., 1], bins=hist_bins)
        h_b = _hist_channel(arr_rgb[..., 2], bins=hist_bins)
        h_gray = _hist_channel(gray, bins=hist_bins)
        hist = np.concatenate([h_r, h_g, h_b, h_gray], axis=0)

        R, G, B = arr_rgb[..., 0], arr_rgb[..., 1], arr_rgb[..., 2]
        exg = (2.0 * G - R - B).astype(np.float32)
        exr = (1.4 * R - G).astype(np.float32)
        idx_stats = np.asarray(
            [
                exg.mean(),
                exg.std(),
                exg.min(),
                exg.max(),
                exr.mean(),
                exr.std(),
                exr.min(),
                exr.max(),
            ],
            dtype=np.float32,
        )

        X[i] = np.concatenate(
            [rgb_flat, gray_flat, edge_flat, stats, hist, idx_stats], axis=0
        )

    return X


X_train = extract_features(train_df["image_id"].values)
y_train = train_df[target_cols].values.astype(int)
X_test = extract_features(test_df["image_id"].values)

print("X_train:", X_train.shape, "y_train:", y_train.shape, "X_test:", X_test.shape)



## === cell 3
base_clf = LogisticRegression(
    solver="lbfgs",
    C=2.0,
    max_iter=6000,
    random_state=42,
    class_weight="balanced",
)

model = Pipeline(
    steps=[
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("clf", MultiOutputClassifier(base_clf, n_jobs=None)),
    ]
)

model.fit(X_train, y_train)

probas = model.predict_proba(X_test)
pred = np.column_stack([p[:, 1] for p in probas]).astype(np.float32)

eps = 1e-6
pred = np.clip(pred, eps, 1.0 - eps)

print("pred:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## === cell 4
sub = test_df[["image_id"]].merge(sample_sub, on="image_id", how="left")
sub[target_cols] = pred
sub = sub[["image_id"] + target_cols]

sub_path = "submission.csv"
sub.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub.head())
