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

0.9693194356375324

# 6. Current score

0.72571

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist here, so `submissions_all` is empty and indexing `[1,2]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, build a valid baseline submission by copying `sample_submission.csv` and filling equal probabilities. I also make `ensemble()` validate indices/weights and use the competition’s available `sample_submission.csv` path so the notebook always produces a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.68841) has done: 'Your current 0.5 score comes from the equal-probability fallback (0.25 each), which is only meant to avoid crashes; to move toward the 0.969 target we need a real model-based prediction while keeping changes minimal. I keep your “produce submission.csv” flow intact, but replace the fallback with a simple, deterministic, image-based baseline using scikit-learn: load images, extract lightweight color features, train one-vs-rest logistic regression on `train.csv`, and predict probabilities for `test.csv`. This preserves the overall approach of “generate submission_avg then write submission.csv”, but makes the predictions informative instead of constant. I also make the data path resolution robust (use the provided `/kaggle/data/...` if `/kaggle/input/...` isn’t present), so it runs in your environment and always writes a valid submission.'
- What this solution (achieved 0.66662) has done: 'Your current baseline is likely underperforming because the features are too coarse (global channel stats) for a 4-label AUC task; without changing the modeling approach, we can make the same logistic-regression pipeline more informative by adding a small, deterministic color-histogram feature block. I keep your exact training/prediction flow and model class (OneVsRest LogisticRegression), but expand `extract_features()` to include per-channel histograms (captures lesion color/coverage signals) and switch to `saga` with a slightly larger `C` to better fit the richer feature space while remaining the same core linear model. I also ensure the submission rows align to `sample_submission.csv` by merging on `image_id` (prevents any accidental ordering mismatch from hurting AUC). These are minimal, safe changes aimed at increasing score toward 0.969 without introducing new training loops or deep models.'
- What this solution (achieved 0.67362) has done: 'Your current linear image-feature baseline is under the 0.969 target, so we should improve discriminative power without changing the overall approach (still: extract handcrafted features → StandardScaler → OneVsRest LogisticRegression → predict_proba). The smallest high-impact upgrade is to add a compact “texture/spotting” signal (edge magnitude histogram + simple gradient stats) that often correlates with lesions, while keeping the same model and training flow. I also make the submission strictly aligned to `sample_submission.csv` order by using it as the canonical `image_id` list (avoids any accidental row-order mismatch). Finally, I keep runtime within limits by using modest image size and feature dimensionality.'
- What this solution (achieved 0.69624) has done: 'Your current score (0.67362) is far below the target (0.9693), so we should increase discriminative power while keeping the same core pipeline (handcrafted features → StandardScaler → OneVsRest LogisticRegression → predict_proba). The smallest high-impact change is to enrich `extract_features()` with compact, deterministic local-texture signals (coarse HOG-like orientation histograms + local binary pattern histogram) that better capture lesion patterns than global color/edges alone. I keep image size, model type, training flow, and submission writing semantics intact, and I only adjust feature extraction and slightly raise `max_iter` to ensure stable convergence on the higher-dimensional feature vector. Submission alignment remains anchored to `sample_submission.csv` via merge on `image_id` to avoid any ordering mistakes.'
- What this solution (achieved 0.68402) has done: 'Your current score (0.69624) is far below the target (0.9693), so we should improve discriminative power while keeping the same core pipeline (handcrafted image features → StandardScaler → OneVsRest LogisticRegression → predict_proba). The smallest high-impact change without changing the model/training loop is to fix a feature bug in `_hog_like` that currently drops the last row/column of gradients (reducing useful signal) and to add a tiny amount of multi-scale robustness by concatenating the same feature set extracted at a second image size. This preserves identical semantics (same classifier, same fitting, same prediction), just with a slightly richer and more correct feature representation. I also keep submission alignment anchored to `sample_submission.csv` as you already do and still write a valid `submission.csv`.'
- What this solution (achieved 0.72571) has done: 'Your current score (0.684) is far below the target (0.9693), so we should improve discrimination while keeping the same core pipeline (handcrafted features → StandardScaler → OneVsRest LogisticRegression → predict_proba → submission.csv). The most likely score limiter is that the raw `predict_proba` outputs are poorly calibrated for ROC-AUC ranking on this dataset; a minimal, metric-aligned improvement is to add out-of-fold probability calibration using scikit-learn’s `CalibratedClassifierCV` on top of the same logistic regression base model. This keeps the model class and training semantics intact (still logistic regression per class), but improves probability ranking stability, which typically boosts mean ROC AUC. I also make sure class imbalance is handled with `class_weight="balanced"` (still logistic regression, same training loop) and keep submission alignment via `sample_submission.csv` unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from PIL import Image

from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV



## === cell 1
CANDIDATE_DATA_DIRS = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/input",
    "/kaggle/data",
]

DATA_DIR = None
for d in CANDIDATE_DATA_DIRS:
    if os.path.isfile(os.path.join(d, "train.csv")) and os.path.isdir(
        os.path.join(d, "images")
    ):
        DATA_DIR = d
        break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find dataset directory containing train.csv and images/. "
        f"Tried: {CANDIDATE_DATA_DIRS}"
    )

SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TEST_CSV_PATH = os.path.join(DATA_DIR, "test.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

SUBMISSIONS_PATH = "/kaggle/input/submissions/"

print("Using DATA_DIR:", DATA_DIR)
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_PATH))
print("Images dir exists:", os.path.isdir(IMAGES_DIR))



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found external submissions:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) == 0:
        raise ValueError("sub_idx is empty; nothing to ensemble.")
    if len(weights) != len(sub_idx):
        raise ValueError(
            f"weights length ({len(weights)}) must match sub_idx length ({len(sub_idx)})."
        )
    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty; no submission files found to ensemble."
        )
    for j in sub_idx:
        if j < 0 or j >= len(submissions_all):
            raise IndexError(
                f"Requested submission index {j} out of range for {len(submissions_all)} files."
            )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg, image_ids=None, submissions_all=None):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    if image_ids is None:
        submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    else:
        pred_df = pd.DataFrame(
            submission_avg,
            columns=["healthy", "multiple_diseases", "rust", "scab"],
        )
        pred_df["image_id"] = image_ids

        submission_df = submission_df[["image_id"]].merge(
            pred_df, on="image_id", how="left"
        )

        if (
            submission_df[["healthy", "multiple_diseases", "rust", "scab"]]
            .isna()
            .any()
            .any()
        ):
            missing = submission_df.loc[
                submission_df[["healthy", "multiple_diseases", "rust", "scab"]]
                .isna()
                .any(axis=1),
                "image_id",
            ].tolist()
            raise ValueError(
                f"Missing predictions for {len(missing)} image_id(s), e.g. {missing[:5]}"
            )
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _safe_open_image(path, img_size=(128, 128)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(img_size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def _channel_hist(arr, bins=16):
    hists = []
    for c in range(3):
        hist, _ = np.histogram(arr[..., c], bins=bins, range=(0.0, 1.0), density=False)
        hist = hist.astype(np.float32)
        hist = hist / (hist.sum() + 1e-8)
        hists.append(hist)
    return np.concatenate(hists, axis=0)


def _edge_features(arr, edge_bins=16):
    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    gx = np.diff(gray, axis=1)
    gy = np.diff(gray, axis=0)

    mag = np.sqrt(gx[:-1, :] * gx[:-1, :] + gy[:, :-1] * gy[:, :-1]).astype(np.float32)

    mag_mean = float(mag.mean())
    mag_std = float(mag.std())
    mag_p90 = float(np.quantile(mag, 0.90))

    mag_cap = np.clip(mag, 0.0, 0.5)
    hist, _ = np.histogram(mag_cap, bins=edge_bins, range=(0.0, 0.5), density=False)
    hist = hist.astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)

    return np.concatenate(
        [np.array([mag_mean, mag_std, mag_p90], dtype=np.float32), hist], axis=0
    )


def _hog_like(gray, cell_size=16, n_bins=9):
    gx = np.diff(gray, axis=1).astype(np.float32)  # (H, W-1)
    gy = np.diff(gray, axis=0).astype(np.float32)  # (H-1, W)

    gx = gx[:-1, :]
    gy = gy[:, :-1]

    mag = np.sqrt(gx * gx + gy * gy)
    ang = (np.degrees(np.arctan2(gy, gx)) + 180.0) % 180.0  # [0,180)

    H, W = mag.shape
    n_cells_y = max(1, H // cell_size)
    n_cells_x = max(1, W // cell_size)

    feats = []
    bin_edges = np.linspace(0.0, 180.0, n_bins + 1, dtype=np.float32)
    for cy in range(n_cells_y):
        y0 = cy * cell_size
        y1 = (cy + 1) * cell_size
        for cx in range(n_cells_x):
            x0 = cx * cell_size
            x1 = (cx + 1) * cell_size

            m = mag[y0:y1, x0:x1].ravel()
            a = ang[y0:y1, x0:x1].ravel()
            if m.size == 0:
                h = np.zeros(n_bins, dtype=np.float32)
            else:
                h, _ = np.histogram(a, bins=bin_edges, weights=m, density=False)
                h = h.astype(np.float32)
                h = h / (h.sum() + 1e-8)
            feats.append(h)

    return np.concatenate(feats, axis=0).astype(np.float32)


def _lbp_hist(gray):
    g = gray.astype(np.float32)
    center = g[1:-1, 1:-1]
    codes = np.zeros_like(center, dtype=np.uint8)

    neighbors = [
        g[:-2, :-2],
        g[:-2, 1:-1],
        g[:-2, 2:],
        g[1:-1, 2:],
        g[2:, 2:],
        g[2:, 1:-1],
        g[2:, :-2],
        g[1:-1, :-2],
    ]
    for i, n in enumerate(neighbors):
        codes |= (n >= center).astype(np.uint8) << i

    hist = np.bincount(codes.ravel(), minlength=256).astype(np.float32)
    hist = hist / (hist.sum() + 1e-8)
    return hist


def _extract_features_single_scale(
    image_ids,
    img_size=(128, 128),
    hist_bins=16,
    edge_bins=16,
    hog_cell_size=16,
    hog_bins=9,
):
    dummy = np.zeros((img_size[0], img_size[1], 3), dtype=np.float32)
    dummy_gray = (
        0.2989 * dummy[..., 0] + 0.5870 * dummy[..., 1] + 0.1140 * dummy[..., 2]
    ).astype(np.float32)
    hog_dim = _hog_like(dummy_gray, cell_size=hog_cell_size, n_bins=hog_bins).shape[0]
    n_feat = 14 + 3 * hist_bins + (3 + edge_bins) + hog_dim + 256

    feats = np.zeros((len(image_ids), n_feat), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        arr = _safe_open_image(img_path, img_size=img_size)

        flat = arr.reshape(-1, 3)
        ch_means = flat.mean(axis=0)
        ch_stds = flat.std(axis=0)
        ch_mins = flat.min(axis=0)
        ch_maxs = flat.max(axis=0)
        overall_mean = arr.mean()
        overall_std = arr.std()
        base = np.concatenate(
            [ch_means, ch_stds, ch_mins, ch_maxs, [overall_mean, overall_std]]
        ).astype(np.float32)

        hist = _channel_hist(arr, bins=hist_bins).astype(np.float32)
        edge = _edge_features(arr, edge_bins=edge_bins).astype(np.float32)

        gray = (
            0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]
        ).astype(np.float32)
        hog = _hog_like(gray, cell_size=hog_cell_size, n_bins=hog_bins).astype(
            np.float32
        )
        lbp = _lbp_hist(gray).astype(np.float32)

        feats[i, :] = np.concatenate([base, hist, edge, hog, lbp]).astype(np.float32)
    return feats


def extract_features(
    image_ids,
    img_size=(128, 128),
    hist_bins=16,
    edge_bins=16,
    hog_cell_size=16,
    hog_bins=9,
):
    f1 = _extract_features_single_scale(
        image_ids,
        img_size=img_size,
        hist_bins=hist_bins,
        edge_bins=edge_bins,
        hog_cell_size=hog_cell_size,
        hog_bins=hog_bins,
    )
    f2 = _extract_features_single_scale(
        image_ids,
        img_size=(96, 96),
        hist_bins=hist_bins,
        edge_bins=edge_bins,
        hog_cell_size=hog_cell_size,
        hog_bins=hog_bins,
    )
    return np.concatenate([f1, f2], axis=1).astype(np.float32)


def train_and_predict_baseline():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    X_train = extract_features(
        train_df["image_id"].values,
        img_size=(128, 128),
        hist_bins=16,
        edge_bins=16,
        hog_cell_size=16,
        hog_bins=9,
    )
    y_train = train_df[TARGET_COLS].values.astype(int)

    X_test = extract_features(
        test_df["image_id"].values,
        img_size=(128, 128),
        hist_bins=16,
        edge_bins=16,
        hog_cell_size=16,
        hog_bins=9,
    )

    base_lr = LogisticRegression(
        max_iter=4000,
        solver="saga",
        C=3.0,
        random_state=0,
        n_jobs=-1,
        class_weight="balanced",
    )

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ovr_cal",
                OneVsRestClassifier(
                    CalibratedClassifierCV(
                        estimator=base_lr,
                        method="sigmoid",
                        cv=3,
                    )
                ),
            ),
        ]
    )

    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)
    proba = np.clip(proba, 0.0, 1.0)

    if proba.shape[1] != 4:
        raise ValueError(f"Expected 4 target columns, got proba shape {proba.shape}")

    return test_df["image_id"].values, proba




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [1, 2], [0.5, 0.5])
    make_submission_file(
        submission_avg, image_ids=None, submissions_all=submissions_all
    )
else:
    test_image_ids, submission_avg = train_and_predict_baseline()
    make_submission_file(
        submission_avg, image_ids=test_image_ids, submissions_all=submissions_all
    )

out = pd.read_csv("submission.csv")
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Submission columns:", list(out.columns))
print("Min/max per target:\n", out[TARGET_COLS].min(), "\n", out[TARGET_COLS].max())
