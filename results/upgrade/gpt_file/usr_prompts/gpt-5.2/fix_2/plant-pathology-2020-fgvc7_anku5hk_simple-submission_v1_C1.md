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

3.9

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

0.9715261650067112

# 6. Current score

0.68107

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.68107) has done: 'I remove the dependency on missing external “plant-best-sub” files that cause the FileNotFoundError, and instead generate predictions from the provided data only. To keep changes minimal and ensure an end-to-end run with the available packages, I implement a stable image-based baseline using simple color/statistical features extracted from the JPEGs and train one-vs-rest logistic regression models for the four labels. This preserves the evaluation semantics (probabilities per class) and produces a valid `submission.csv` with the exact required columns and test `image_id` order. I also make the input path resolution robust to your folder layout so it runs in this environment.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

from PIL import Image
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/data/plant-pathology-2020-fgvc7",
    "/kaggle/data",
    "/kaggle/input",
    "../input/plant-pathology-2020-fgvc7",
    "../input",
]


def find_file(filename: str):
    for base in CANDIDATE_BASES:
        path = os.path.join(base, filename)
        if os.path.exists(path):
            return path
    raise FileNotFoundError(
        f"Could not find {filename} under any candidate base paths: {CANDIDATE_BASES}"
    )


def find_images_dir():
    for base in CANDIDATE_BASES:
        path = os.path.join(base, "images")
        if os.path.isdir(path):
            return path
    raise FileNotFoundError(
        f"Could not find images/ directory under any candidate base paths: {CANDIDATE_BASES}"
    )


train_csv_path = find_file("train.csv")
test_csv_path = find_file("test.csv")
sample_sub_path = find_file("sample_submission.csv")
images_dir = find_images_dir()

train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(test_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
assert all(
    c in train_df.columns for c in ["image_id"] + target_cols
), "Train columns mismatch."
assert (
    list(sample_sub.columns) == ["image_id"] + target_cols
), "Sample submission columns mismatch."

train_df.head(), test_df.head(), sample_sub.head()




## === cell 2
def image_features(image_path, size=(96, 96)):
    try:
        img = Image.open(image_path).convert("RGB")
    except Exception:
        return np.zeros(24, dtype=np.float32)

    img = img.resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3 in [0,1]
    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))

    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    eps = 1e-6
    exg = 2 * g - r - b  # excess green
    exr = 1.4 * r - g  # excess red
    exg_mean = exg.mean()
    exg_std = exg.std()
    exr_mean = exr.mean()
    exr_std = exr.std()
    rg_ratio = (r.mean() + eps) / (g.mean() + eps)
    gb_ratio = (g.mean() + eps) / (b.mean() + eps)
    rb_ratio = (r.mean() + eps) / (b.mean() + eps)
    gray = 0.2989 * r + 0.5870 * g + 0.1140 * b
    gx = np.abs(np.diff(gray, axis=1)).mean()
    gy = np.abs(np.diff(gray, axis=0)).mean()

    feat = np.concatenate(
        [
            means,
            stds,
            mins,
            maxs,
            np.array(
                [
                    exg_mean,
                    exg_std,
                    exr_mean,
                    exr_std,
                    rg_ratio,
                    gb_ratio,
                    rb_ratio,
                    gx,
                    gy,
                ],
                dtype=np.float32,
            ),
        ]
    )
    if feat.shape[0] != 24:
        feat = np.pad(feat, (0, max(0, 24 - feat.shape[0])), mode="constant")[:24]
    return feat.astype(np.float32)


def build_feature_matrix(image_ids):
    X = np.zeros((len(image_ids), 24), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(images_dir, f"{image_id}.jpg")
        X[i] = image_features(img_path)
    return X


X_train = build_feature_matrix(train_df["image_id"].values)
X_test = build_feature_matrix(test_df["image_id"].values)

y_train = train_df[target_cols].astype(np.int32).values

X_train.shape, X_test.shape, y_train.shape



## === cell 3
models = []
test_pred = np.zeros((len(test_df), len(target_cols)), dtype=np.float64)

for j, col in enumerate(target_cols):
    clf = make_pipeline(
        StandardScaler(with_mean=True, with_std=True),
        LogisticRegression(
            solver="lbfgs",
            max_iter=500,
            C=2.0,
            class_weight="balanced",
            random_state=42,
        ),
    )
    clf.fit(X_train, y_train[:, j])
    test_pred[:, j] = clf.predict_proba(X_test)[:, 1]
    models.append(clf)

test_pred = np.clip(test_pred, 1e-6, 1 - 1e-6)

test_pred[:3]



## === cell 4
sub = sample_sub.copy()
sub = sub.merge(
    test_df[["image_id"]], on="image_id", how="right"
)  # ensure exactly test ids
sub[target_cols] = test_pred

sub = sub[["image_id"] + target_cols]

sub.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
