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

0.9704488894833836

# 6. Current score

0.64301

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to ensemble several external submission CSVs that are not present in this Kaggle environment (`../input/plantpathology/...`), so `dsub` is never created and the later averaging crashes. To keep the core “simple average of predictions into sample_submission format” logic while making it runnable end-to-end, I replaced those missing inputs with a deterministic, data-only fallback: compute class priors from `train.csv` and use them as constant probabilities for every test image (a valid baseline). I also made the input path robust by using the provided `/kaggle/input/plant-pathology-2020-fgvc7/` dataset, and added column/order alignment safeguards so the submission is always valid. This produce `submission.csv` reliably; score won’t hit the 0.97 target (since no model training packages are available), but it run and submit correctly.'
- What this solution (achieved 0.556) has done: 'Your current 0.5 score comes from predicting constant class priors, which has little ranking power for ROC AUC. With only numpy/pandas/sklearn available (no deep learning / image feature packages), the smallest legitimate way to increase AUC toward 0.97 is to keep the same “data-only” approach but add simple image-derived signals: compute per-image grayscale summary stats (mean/std + coarse histogram bins) from the provided JPGs, then train a lightweight one-vs-rest logistic regression for each label. This preserves the overall pipeline structure (read CSVs → produce probabilities → write submission) while introducing real per-image variation to improve ranking. I also keep strict column/order alignment to the sample submission so the CSV is always valid.'
- What this solution (achieved 0.64301) has done: 'We keep your exact “simple grayscale feature extraction + one-vs-rest LogisticRegression” core logic, but make two minimal changes that typically increase ROC AUC ranking power: (1) include a small set of deterministic color features (RGB channel stats + channel histograms) alongside your existing grayscale features, and (2) set `class_weight="balanced"` to reduce bias from label imbalance (important for mean column-wise AUC). We also add an explicit `multi_class="ovr"` for clarity and bump `max_iter` modestly to ensure stable convergence with the slightly larger feature set (still fast). Submission formatting, paths, and the overall pipeline remain unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_CANDIDATES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
]
BASE = None
for p in BASE_CANDIDATES:
    if os.path.exists(p) and os.path.exists(os.path.join(p, "sample_submission.csv")):
        BASE = p
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate competition dataset directory containing sample_submission.csv. "
        f"Tried: {BASE_CANDIDATES}"
    )

print("Using BASE:", BASE)
print("Files in BASE (first 20):", sorted(os.listdir(BASE))[:20])



## === cell 2
train_path = os.path.join(BASE, "train.csv")
test_path = os.path.join(BASE, "test.csv")
sample_path = os.path.join(BASE, "sample_submission.csv")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sub = pd.read_csv(sample_path)

target_cols = [c for c in sub.columns if c != "image_id"]
missing = [c for c in target_cols if c not in train_df.columns]
if missing:
    raise ValueError(f"Train is missing expected target columns: {missing}")

IMAGES_CANDIDATES = [
    os.path.join(BASE, "images"),
    os.path.join(BASE, "Images"),
]
IMAGES_DIR = None
for p in IMAGES_CANDIDATES:
    if os.path.isdir(p):
        IMAGES_DIR = p
        break
if IMAGES_DIR is None:
    raise FileNotFoundError(
        f"Could not locate images directory. Tried: {IMAGES_CANDIDATES}"
    )

print("Using IMAGES_DIR:", IMAGES_DIR)



## === cell 3
from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression


def image_features_from_path(
    img_path, gray_hist_bins=16, rgb_hist_bins=8, size=(128, 128)
):
    """
    Deterministic features, designed to improve per-image ranking for ROC AUC while keeping the same pipeline.
    - Grayscale: mean, std (2)
    - Grayscale histogram (16)
    - Grayscale 4x4 grid mean/std (32)
    - Color: per-channel mean/std for RGB (6)
    - Color: per-channel histogram (8 bins each => 24)
    Total dims: 2 + 16 + 32 + 6 + 24 = 80
    """
    try:
        im_rgb = Image.open(img_path).convert("RGB").resize(size)
        arr_rgb = np.asarray(im_rgb, dtype=np.float32) / 255.0  # (H,W,3)
        arr_gray = (
            0.2989 * arr_rgb[..., 0]
            + 0.5870 * arr_rgb[..., 1]
            + 0.1140 * arr_rgb[..., 2]
        ).astype(np.float32)
    except Exception:
        return np.full(80, np.nan, dtype=np.float32)

    mean_g = float(arr_gray.mean())
    std_g = float(arr_gray.std())

    hist_g, _ = np.histogram(
        arr_gray, bins=gray_hist_bins, range=(0.0, 1.0), density=False
    )
    hist_g = hist_g.astype(np.float32)
    hist_g = hist_g / (hist_g.sum() + 1e-12)

    h, w = arr_gray.shape
    gh, gw = h // 4, w // 4
    grid_feats = []
    for i in range(4):
        for j in range(4):
            patch = arr_gray[i * gh : (i + 1) * gh, j * gw : (j + 1) * gw]
            grid_feats.append(float(patch.mean()))
            grid_feats.append(float(patch.std()))
    grid_feats = np.asarray(grid_feats, dtype=np.float32)

    ch_stats = []
    for k in range(3):
        ch = arr_rgb[..., k]
        ch_stats.append(float(ch.mean()))
        ch_stats.append(float(ch.std()))
    ch_stats = np.asarray(ch_stats, dtype=np.float32)

    ch_hists = []
    for k in range(3):
        ch = arr_rgb[..., k]
        hist_c, _ = np.histogram(
            ch, bins=rgb_hist_bins, range=(0.0, 1.0), density=False
        )
        hist_c = hist_c.astype(np.float32)
        hist_c = hist_c / (hist_c.sum() + 1e-12)
        ch_hists.append(hist_c)
    ch_hists = np.concatenate(ch_hists).astype(np.float32)  # 3*rgb_hist_bins

    feats = np.concatenate(
        [
            np.asarray([mean_g, std_g], dtype=np.float32),
            hist_g,
            grid_feats,
            ch_stats,
            ch_hists,
        ]
    ).astype(np.float32)
    return feats


def build_feature_matrix(image_ids):
    X = np.zeros((len(image_ids), 80), dtype=np.float32)
    for i, img_id in enumerate(image_ids):
        img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
        X[i] = image_features_from_path(img_path)

    col_means = np.nanmean(X, axis=0)
    inds = np.where(np.isnan(X))
    if len(inds[0]) > 0:
        X[inds] = np.take(col_means, inds[1])
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

print("X_train shape:", X_train.shape, "X_test shape:", X_test.shape)



## === cell 4
pred_test = pd.DataFrame({"image_id": test_df["image_id"].values})

for c in target_cols:
    y = train_df[c].astype(int).values

    if y.min() == y.max():
        const_p = float(y.mean())
        pred_test[c] = const_p
        continue

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "lr",
                LogisticRegression(
                    C=1.0,
                    solver="lbfgs",
                    max_iter=4000,  # small stability bump with slightly larger feature set
                    random_state=0,
                    class_weight="balanced",
                    multi_class="ovr",
                ),
            ),
        ]
    )
    clf.fit(X_train, y)
    pred_test[c] = clf.predict_proba(X_test)[:, 1].astype(np.float64)

pred_test[target_cols] = pred_test[target_cols].clip(1e-6, 1 - 1e-6)



## === cell 5
sub_out = sub[["image_id"] + target_cols].copy()
sub_out = sub_out.merge(pred_test, on="image_id", how="left", suffixes=("", "_pred"))

for c in target_cols:
    if f"{c}_pred" not in sub_out.columns:
        raise RuntimeError(f"Missing predictions for column {c}.")
    sub_out[c] = sub_out[f"{c}_pred"].astype(float)
    sub_out.drop(columns=[f"{c}_pred"], inplace=True)

sub_out[target_cols] = sub_out[target_cols].clip(0.0, 1.0)

sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## === cell 6
assert os.path.exists("submission.csv"), "submission.csv was not created."
loaded = pd.read_csv("submission.csv")
assert (
    list(loaded.columns) == ["image_id"] + target_cols
), "Submission columns mismatch."
assert len(loaded) == len(
    pd.read_csv(sample_path)
), "Submission row count mismatch vs sample_submission."
assert loaded["image_id"].equals(
    pd.read_csv(sample_path)["image_id"]
), "image_id order mismatch vs sample_submission."
assert loaded[target_cols].notnull().all().all(), "Found NaNs in prediction columns."
print("Submission format validated.")
