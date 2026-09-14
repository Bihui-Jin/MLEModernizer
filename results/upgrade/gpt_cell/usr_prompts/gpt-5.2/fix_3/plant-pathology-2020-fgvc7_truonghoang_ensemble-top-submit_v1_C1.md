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

0.64273

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Diagnosis: Cell 1 crashes with `FileNotFoundError` because it tries to load five external ensemble submission CSVs from `../input/plantpathology/`, a directory/files that do not exist in this environment. The available dataset provides only the competition files (e.g., `../input/plant-pathology-2020-fgvc7/sample_submission.csv`) but not those precomputed model submissions. To keep cell 2’s averaging logic intact (it expects `sub1..sub4` with the standard columns), we should create fallback DataFrames with the correct schema and deterministic values when those CSVs are missing.  

Patch summary: In cell 1 only, wrap each `pd.read_csv` in a safe loader that falls back to copying `sample_submission.csv` and setting class probabilities to a uniform distribution (0.25 each) if the target file is absent. This preserves the interface (`sub1..sub5` exist with expected columns) so cell 2 can run unchanged.  

Updated cells: Only cell 1 is modified.  

Compatibility notes for cell k+1: Cell 2 still find `sub1`, `sub2`, `sub3`, `sub4` with columns `healthy`, `multiple_diseases`, `rust`, `scab`, and aligned row order/length matching `sample_submission.csv`. `sub5` is also created to preserve the original variable, though it is not used in cell 2.  

Assumptions: When external ensemble CSVs are unavailable, using a uniform probability fallback is acceptable to unblock execution and maintain deterministic behavior without changing downstream averaging semantics.'
- What this solution (achieved 0.64273) has done: 'Your current 0.5 score comes from the uniform 0.25 fallback predictions when the external ensemble CSVs are missing. To move toward the 0.96796 target without changing the “average submissions” core logic, the minimal legitimate improvement is to replace the fallback with a lightweight image-based baseline trained on the provided train.csv/images and then use its test probabilities as stand-ins for the missing `sub1..sub4`. This preserves the downstream averaging semantics (cell 2 remains an averaging step) while producing informative probabilities from available data. The model is a simple scikit-learn multinomial logistic regression on color-histogram features extracted from the JPGs, which is fast enough and uses only installed packages plus PIL (available in Kaggle). The submission format and column alignment are kept identical to sample_submission.csv.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

os.listdir("../input/")



## === cell 1
import os
import numpy as np
import pandas as pd


DATA_ROOT = "../input/plant-pathology-2020-fgvc7"
IMG_DIR = os.path.join(DATA_ROOT, "images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _load_image_rgb(path):
    from PIL import Image

    with Image.open(path) as im:
        im = im.convert("RGB")
        return im


def _rgb_hist_features(img_path, size=128, bins=16):
    """
    Lightweight feature extraction: resized RGB image -> per-channel normalized histogram.
    Output dim: 3*bins.
    """
    im = _load_image_rgb(img_path)
    im = im.resize((size, size))
    arr = np.asarray(im, dtype=np.uint8)
    feats = []
    for ch in range(3):
        h, _ = np.histogram(arr[..., ch], bins=bins, range=(0, 256))
        feats.append(h.astype(np.float32))
    x = np.concatenate(feats)
    s = x.sum()
    if s > 0:
        x /= s
    return x


def _build_features(df, img_dir=IMG_DIR):
    X = np.zeros((len(df), 48), dtype=np.float32)  # 3*bins with bins=16 => 48
    missing = 0
    for i, image_id in enumerate(df["image_id"].values):
        img_path = os.path.join(img_dir, f"{image_id}.jpg")
        if not os.path.exists(img_path):
            missing += 1
            continue
        X[i] = _rgb_hist_features(img_path, size=128, bins=16)
    return X, missing


def _train_and_predict_probs():
    train_df = pd.read_csv(TRAIN_CSV)
    test_df = pd.read_csv(TEST_CSV)

    X_train, miss_tr = _build_features(train_df)
    X_test, miss_te = _build_features(test_df)

    y = train_df[TARGET_COLS].values.astype(np.int64)
    y_class = y.argmax(axis=1)

    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=400,
        C=3.0,
        n_jobs=1,
        random_state=42,
    )
    clf.fit(X_train, y_class)
    proba = clf.predict_proba(X_test)  # shape (n_test, 4)

    aligned = np.zeros((len(test_df), len(TARGET_COLS)), dtype=np.float32)
    for j, cls in enumerate(clf.classes_):
        aligned[:, int(cls)] = proba[:, j]

    out = pd.DataFrame({"image_id": test_df["image_id"].values})
    for k, c in enumerate(TARGET_COLS):
        out[c] = aligned[:, k].astype(np.float32)

    return out


def _safe_read_submission(path, template_path=SAMPLE_SUB):
    """
    Preserve original interface: try reading external submission; if missing,
    use our trained baseline predictions (informative) instead of uniform 0.25.
    """
    if os.path.exists(path):
        df = pd.read_csv(path)
        for c in ["image_id"] + TARGET_COLS:
            if c not in df.columns:
                raise ValueError(f"Submission at {path} missing required column: {c}")
        return df[["image_id"] + TARGET_COLS].copy()

    pred = _train_and_predict_probs()

    tmpl = pd.read_csv(template_path)[["image_id"]].copy()
    pred = tmpl.merge(pred, on="image_id", how="left")
    pred[TARGET_COLS] = pred[TARGET_COLS].fillna(0.25)
    return pred[["image_id"] + TARGET_COLS].copy()


sub1 = _safe_read_submission(
    "../input/plantpathology/effnet-fastai-folds-x5_version3.csv"
)
sub2 = _safe_read_submission(
    "../input/plantpathology/fork-of-plant-2020-tpu-915e9c_version1.csv"
)
sub3 = _safe_read_submission(
    "../input/plantpathology/plant-pathology-pytorch-efficientnet-b4-gpu_version_7.csv"
)
sub4 = _safe_read_submission(
    "../input/plantpathology/public-first-score-tpu-incepresnetv2-enb7_version8.csv"
)
sub5 = _safe_read_submission(
    "../input/plantpathology/tpu-ensemble-effnb7-effnb6-inceptresnetv2-etc_verion13.csv"
)



## === cell 2
import pandas as pd

sub = pd.read_csv("../input/plant-pathology-2020-fgvc7/sample_submission.csv")

sub.healthy = (sub1.healthy + sub2.healthy + sub3.healthy + sub4.healthy) / 4
sub.multiple_diseases = (
    sub1.multiple_diseases
    + sub2.multiple_diseases
    + sub3.multiple_diseases
    + sub4.multiple_diseases
) / 4
sub.rust = (sub1.rust + sub2.rust + sub3.rust + sub4.rust) / 4
sub.scab = (sub1.scab + sub2.scab + sub3.scab + sub4.scab) / 4

sub = sub[["image_id", "healthy", "multiple_diseases", "rust", "scab"]]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
