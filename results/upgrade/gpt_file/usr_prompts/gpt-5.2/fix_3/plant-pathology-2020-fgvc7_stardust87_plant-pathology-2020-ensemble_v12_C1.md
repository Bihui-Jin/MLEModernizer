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

0.9711824920467708

# 6. Current score

0.66867

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your code fails because `SUBMISSIONS_PATH` points to a non-existent folder, so `submissions_all` is empty and indexing `[0,1,2]` crashes. I fix this by switching the input path to the competition’s provided `sample_submission.csv` and generating a valid baseline submission directly from it (this is score-improving vs “no submission” while keeping changes minimal). I also add small safety checks (existence/shape) and ensure the output is written as `submission.csv` with the exact required columns and row order aligned to `test.csv`.'
- What this solution (achieved 0.66867) has done: 'Your current pipeline just copies the sample submission’s constant probabilities, which typically scores around 0.5 AUC. To move toward the 0.971 target without changing the overall “no-training” approach, I generate image-based probabilities using a simple RGB feature extractor plus a multi-output logistic regression (same basic structure: read CSVs → compute features → fit → predict → write submission). I also add a stratified train/validation split and print the mean ROC AUC locally so you can verify the direction before submitting, while keeping the submission formatting logic and paths intact. This should substantially improve score toward the target while remaining lightweight and finishing within the time limit.'

# 9. Code solution

## === cell 0
import os
import pandas as pd



## === cell 1
DATA_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_DIR, "images")

print("Exists TRAIN_CSV:", os.path.exists(TRAIN_CSV))
print("Exists TEST_CSV:", os.path.exists(TEST_CSV))
print("Exists SAMPLE_SUB_CSV:", os.path.exists(SAMPLE_SUB_CSV))
print("Exists IMAGES_DIR:", os.path.exists(IMAGES_DIR))




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    if weights is None:
        weights = [1.0] * len(sub_idx)

    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have the same length, got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError(
            "submissions_all is empty. Provide valid submission file paths to ensemble."
        )

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} out of range for submissions_all of length {len(submissions_all)}"
            )
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




## === cell 3
def make_submission_file_from_df(pred_df, out_path="submission.csv"):
    sample = pd.read_csv(SAMPLE_SUB_CSV)
    required_cols = ["image_id", "healthy", "multiple_diseases", "rust", "scab"]
    if list(sample.columns) != required_cols:
        sample = sample[required_cols]

    if "image_id" not in pred_df.columns:
        raise ValueError("pred_df must contain an 'image_id' column")

    pred_df = pred_df.copy()
    pred_df = pred_df[required_cols]
    merged = sample[["image_id"]].merge(pred_df, on="image_id", how="left")

    for c in required_cols[1:]:
        if merged[c].isna().any():
            merged[c] = merged[c].fillna(0.25)

    for c in required_cols[1:]:
        merged[c] = merged[c].clip(0.0, 1.0)

    merged.to_csv(out_path, index=False)
    print(
        f"Wrote {out_path} with shape {merged.shape} and columns {list(merged.columns)}"
    )




## === cell 4

import numpy as np

try:
    from PIL import Image
except Exception as e:
    raise ImportError(
        "PIL is required to read images. Please ensure Pillow is available in the environment."
    ) from e

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import roc_auc_score

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)


def image_path(image_id: str) -> str:
    return os.path.join(IMAGES_DIR, f"{image_id}.jpg")


def extract_rgb_stats(img_path: str, size=(96, 96)) -> np.ndarray:
    with Image.open(img_path) as im:
        im = im.convert("RGB")
        if size is not None:
            im = im.resize(size)
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (H,W,3)
    means = arr.mean(axis=(0, 1))  # 3
    stds = arr.std(axis=(0, 1))  # 3
    overall_mean = np.array([arr.mean()], dtype=np.float32)
    overall_std = np.array([arr.std()], dtype=np.float32)
    return np.concatenate([means, stds, overall_mean, overall_std], axis=0)  # 8-dim


def build_features(image_ids: pd.Series) -> np.ndarray:
    X = np.zeros((len(image_ids), 8), dtype=np.float32)
    for i, img_id in enumerate(image_ids.tolist()):
        p = image_path(img_id)
        if not os.path.exists(p):
            X[i] = 0.0
        else:
            X[i] = extract_rgb_stats(p)
    return X


X = build_features(train_df["image_id"])
y = train_df[TARGET_COLS].values.astype(int)

strat = train_df[TARGET_COLS].astype(str).agg("_".join, axis=1)

X_tr, X_va, y_tr, y_va = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=strat
)

clf = Pipeline(
    steps=[
        ("scaler", StandardScaler()),
        ("ovr", OneVsRestClassifier(LogisticRegression(max_iter=2000, solver="lbfgs"))),
    ]
)

clf.fit(X_tr, y_tr)

va_pred = clf.predict_proba(X_va)
auc_cols = []
for j, c in enumerate(TARGET_COLS):
    if len(np.unique(y_va[:, j])) < 2:
        continue
    auc_cols.append(roc_auc_score(y_va[:, j], va_pred[:, j]))
if len(auc_cols) > 0:
    print("Local mean column-wise ROC AUC (valid):", float(np.mean(auc_cols)))
else:
    print("Local AUC could not be computed (constant labels in validation split).")

clf.fit(X, y)

X_test = build_features(test_df["image_id"])
test_pred = clf.predict_proba(X_test)

pred_df = pd.DataFrame(test_pred, columns=TARGET_COLS)
pred_df.insert(0, "image_id", test_df["image_id"].values)

make_submission_file_from_df(pred_df, out_path="submission.csv")
