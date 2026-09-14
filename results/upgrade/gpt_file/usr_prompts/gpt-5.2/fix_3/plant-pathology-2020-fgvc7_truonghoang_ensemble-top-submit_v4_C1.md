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

0.9667180849324454

# 6. Current score

0.63456

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Your notebook fails because it tries to read several external “../input/plantpathology/*.csv” submission files that do not exist in this environment, so `dsub` is never created and later cells crash. The minimal safe fix is to remove that dependency and instead generate predictions directly from the provided training/test labels using a simple, deterministic baseline that outputs class priors (mean of each label in `train.csv`) for every test row. This run end-to-end on the given dataset, produce a correctly formatted `submission.csv`, and should yield a reasonable ROC AUC (better than an all-zeros submission) without changing any “model architecture” (there wasn’t one here). Paths are adjusted to the actual provided dataset location while keeping the same overall flow (load → build submission → write CSV).'
- What this solution (achieved 0.63456) has done: 'You’re currently submitting constant class-prior probabilities, which tends to score near random (≈0.5 AUC) because it cannot rank images. Since we must keep the core approach minimal and we don’t have deep-learning libraries available, the smallest legitimate improvement is to add a lightweight image-feature model using only installed packages: read each JPG, compute simple color statistics, and train one LogisticRegression per label. This preserves the “load → build features → fit → predict → write submission.csv” flow, produces a valid submission, and should move the score upward toward your target without changing evaluation semantics. I’m also keeping paths compatible with your provided directory layout and ensuring the submission column order matches `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd



## === cell 1
BASE_INPUT = "/kaggle/input"

CANDIDATES = [
    os.path.join("/kaggle", "data", "plant-pathology-2020-fgvc7"),
    os.path.join("/kaggle", "data"),
    os.path.join(BASE_INPUT, "plant-pathology-2020-fgvc7"),
    BASE_INPUT,
]

COMP_DIR = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
        os.path.join(c, "test.csv")
    ):
        COMP_DIR = c
        break

if COMP_DIR is None:
    raise FileNotFoundError(
        "Could not find train.csv/test.csv under expected Kaggle input/data directories."
    )

train_path = os.path.join(COMP_DIR, "train.csv")
test_path = os.path.join(COMP_DIR, "test.csv")
sample_path = os.path.join(COMP_DIR, "sample_submission.csv")

images_dir = os.path.join(COMP_DIR, "images")
if not os.path.isdir(images_dir):
    alt_images_dir = os.path.join("/kaggle", "data", "images")
    if os.path.isdir(alt_images_dir):
        images_dir = alt_images_dir

for p in [train_path, test_path, sample_path]:
    if not os.path.exists(p):
        raise FileNotFoundError(f"Required file not found: {p}")

if not os.path.isdir(images_dir):
    raise FileNotFoundError(f"Images directory not found: {images_dir}")

print("Using COMP_DIR =", COMP_DIR)
print("Using images_dir =", images_dir)
print(
    "Files:",
    [
        os.path.basename(train_path),
        os.path.basename(test_path),
        os.path.basename(sample_path),
    ],
)



## === cell 2

from PIL import Image
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

train = pd.read_csv(train_path)
test = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]

missing_train = [c for c in (["image_id"] + target_cols) if c not in train.columns]
missing_test = [c for c in (["image_id"]) if c not in test.columns]
missing_sub = [c for c in (["image_id"] + target_cols) if c not in sample_sub.columns]
if missing_train:
    raise ValueError(f"train.csv missing columns: {missing_train}")
if missing_test:
    raise ValueError(f"test.csv missing columns: {missing_test}")
if missing_sub:
    raise ValueError(f"sample_submission.csv missing columns: {missing_sub}")


def _safe_open_image(path):
    with Image.open(path) as im:
        return im.convert("RGB")


def extract_features(image_id, resize=(96, 96)):
    img_path = os.path.join(images_dir, f"{image_id}.jpg")
    im = _safe_open_image(img_path)
    if resize is not None:
        im = im.resize(resize)
    arr = np.asarray(im, dtype=np.float32) / 255.0  # H,W,3 in [0,1]

    ch_mean = arr.reshape(-1, 3).mean(axis=0)
    ch_std = arr.reshape(-1, 3).std(axis=0)

    gray = (0.2989 * arr[..., 0] + 0.5870 * arr[..., 1] + 0.1140 * arr[..., 2]).astype(
        np.float32
    )
    g_mean = float(gray.mean())
    g_std = float(gray.std())

    dh = np.abs(gray[:, 1:] - gray[:, :-1]).mean() if gray.shape[1] > 1 else 0.0
    dv = np.abs(gray[1:, :] - gray[:-1, :]).mean() if gray.shape[0] > 1 else 0.0

    feat = np.concatenate(
        [ch_mean, ch_std, np.array([g_mean, g_std, dh, dv], dtype=np.float32)], axis=0
    )
    return feat


X_train = np.vstack([extract_features(iid) for iid in train["image_id"].values])
X_test = np.vstack([extract_features(iid) for iid in test["image_id"].values])

models = {}
test_pred = np.zeros((len(test), len(target_cols)), dtype=np.float32)

for j, c in enumerate(target_cols):
    y = train[c].astype(int).values
    pipe = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    solver="lbfgs",
                    max_iter=1000,
                    class_weight="balanced",
                    random_state=42,
                ),
            ),
        ]
    )
    pipe.fit(X_train, y)
    test_pred[:, j] = pipe.predict_proba(X_test)[:, 1].astype(np.float32)
    models[c] = pipe

sub = pd.DataFrame({"image_id": test["image_id"].values})
for j, c in enumerate(target_cols):
    sub[c] = np.clip(test_pred[:, j], 0.0, 1.0)

sub = sub[sample_sub.columns.tolist()]
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
