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

0.68841

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist here, so `submissions_all` is empty and indexing `[1,2]` fails. I keep your ensembling logic intact, but add a safe fallback: if no external submissions are found, build a valid baseline submission by copying `sample_submission.csv` and filling equal probabilities. I also make `ensemble()` validate indices/weights and use the competition’s available `sample_submission.csv` path so the notebook always produces a proper `submission.csv` with the required columns.'
- What this solution (achieved 0.68841) has done: 'Your current 0.5 score comes from the equal-probability fallback (0.25 each), which is only meant to avoid crashes; to move toward the 0.969 target we need a real model-based prediction while keeping changes minimal. I keep your “produce submission.csv” flow intact, but replace the fallback with a simple, deterministic, image-based baseline using scikit-learn: load images, extract lightweight color features, train one-vs-rest logistic regression on `train.csv`, and predict probabilities for `test.csv`. This preserves the overall approach of “generate submission_avg then write submission.csv”, but makes the predictions informative instead of constant. I also make the data path resolution robust (use the provided `/kaggle/data/...` if `/kaggle/input/...` isn’t present), so it runs in your environment and always writes a valid submission.'

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
def make_submission_file(submission_avg, submissions_all=None):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]


def _safe_open_image(path, img_size=(128, 128)):
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(img_size, resample=Image.BILINEAR)
        arr = np.asarray(im, dtype=np.float32) / 255.0
    return arr


def extract_features(image_ids, img_size=(128, 128)):
    feats = np.zeros((len(image_ids), 14), dtype=np.float32)
    for i, image_id in enumerate(image_ids):
        img_path = os.path.join(IMAGES_DIR, f"{image_id}.jpg")
        arr = _safe_open_image(img_path, img_size=img_size)
        ch_means = arr.reshape(-1, 3).mean(axis=0)
        ch_stds = arr.reshape(-1, 3).std(axis=0)
        ch_mins = arr.reshape(-1, 3).min(axis=0)
        ch_maxs = arr.reshape(-1, 3).max(axis=0)
        overall_mean = arr.mean()
        overall_std = arr.std()
        feats[i, :] = np.concatenate(
            [ch_means, ch_stds, ch_mins, ch_maxs, [overall_mean, overall_std]]
        ).astype(np.float32)
    return feats


def train_and_predict_baseline():
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    test_df = pd.read_csv(TEST_CSV_PATH)

    X_train = extract_features(train_df["image_id"].values, img_size=(128, 128))
    y_train = train_df[TARGET_COLS].values.astype(int)

    X_test = extract_features(test_df["image_id"].values, img_size=(128, 128))

    clf = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "ovr",
                OneVsRestClassifier(
                    LogisticRegression(max_iter=1000, solver="lbfgs", random_state=0)
                ),
            ),
        ]
    )
    clf.fit(X_train, y_train)

    proba = clf.predict_proba(X_test)

    proba = np.clip(proba, 0.0, 1.0)
    if proba.shape[1] != 4:
        raise ValueError(f"Expected 4 target columns, got proba shape {proba.shape}")

    return proba




## === cell 6
if len(submissions_all) >= 3:
    submission_avg = ensemble(submissions_all, [1, 2], [0.5, 0.5])
else:
    submission_avg = train_and_predict_baseline()

make_submission_file(submission_avg, submissions_all)

out = pd.read_csv("submission.csv")
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
print("Submission columns:", list(out.columns))
print("Min/max per target:\n", out[TARGET_COLS].min(), "\n", out[TARGET_COLS].max())
