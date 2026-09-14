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

0.9629218283013236

# 6. Current score

0.56205

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The crash happens because `/kaggle/input/submissions/` doesn’t exist in this environment, so `submissions_all` is empty and indexing `[0,2,4]` fails. I keep the ensemble logic intact, but add a safe fallback: if no external submission files are found, we generate a valid baseline submission from the provided `sample_submission.csv` (uniform probabilities). This guarantees the notebook runs end-to-end and always writes `submission.csv` with the correct columns and row order. I also add minimal guards to prevent mismatched indices/weights from causing runtime errors.'
- What this solution (achieved 0.5) has done: 'Your current 0.5 score comes from outputting uniform probabilities (or failing to find any real model submissions), which is far from the 0.9629 target. The smallest legitimate improvement without changing the “core logic” (you’re not training any model here) is to generate stronger probabilities directly from the provided training labels: use the per-class prevalence as constant predictions for all test images, which usually beat uniform on mean ROC AUC. I keep the existing ensemble behavior unchanged when external submissions exist, but improve the no-submission fallback from uniform (0.25) to label-prior probabilities computed from `train.csv`. I also harden column ordering/alignment using `sample_submission.csv` so the output is always valid and correctly ordered.'
- What this solution (achieved 0.56205) has done: 'I fix the fallback model crash by correcting the image filename mapping: the `image_id` values in `train.csv`/`test.csv` don’t include the `.jpg` extension, so the loader was trying to open non-existent paths and loaded zero images. I add a minimal helper that resolves each `image_id` to an existing file in the images directory (trying `id`, `id.jpg`, `id.JPG`, etc.), and use it in both train/test feature loading. This preserves the core logic (pixel features + multinomial LogisticRegression fallback, and the ensemble path unchanged) while making the pipeline run end-to-end and produce a valid `submission.csv` with correctly ordered columns.'

# 9. Code solution

## === cell 0
import os
import pandas as pd

from PIL import Image
import numpy as np

from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression



## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_PATH = os.path.join(DATA_ROOT, "train.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")

TARGET_COLS = ["healthy", "multiple_diseases", "rust", "scab"]



## === cell 2
submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
        for filename in filenames:
            if filename.lower().endswith(".csv"):
                submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 3
def ensemble(submissions_all, sub_idx, weights=[]):
    if len(sub_idx) != len(weights):
        raise ValueError(
            f"sub_idx and weights must have same length. Got {len(sub_idx)} and {len(weights)}"
        )

    if len(submissions_all) == 0:
        raise ValueError("No submission files available to ensemble.")

    submission_with_weight = []
    for i in range(len(sub_idx)):
        if sub_idx[i] < 0 or sub_idx[i] >= len(submissions_all):
            raise IndexError(
                f"sub_idx[{i}]={sub_idx[i]} is out of range for {len(submissions_all)} available files."
            )
        print(
            f"I'm taking submission {submissions_all[sub_idx[i]]} with weight {weights[i]}"
        )
        submission = pd.read_csv(submissions_all[sub_idx[i]])

        missing = [c for c in TARGET_COLS if c not in submission.columns]
        if missing:
            raise ValueError(
                f"Submission {submissions_all[sub_idx[i]]} missing columns: {missing}"
            )

        submission = submission.loc[:, TARGET_COLS].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 4
def make_submission_file(submission_avg):
    submission_df = pd.read_csv(SAMPLE_SUB_PATH)
    submission_df = submission_df.loc[:, ["image_id"] + TARGET_COLS].copy()
    submission_df.loc[:, TARGET_COLS] = submission_avg
    submission_df.to_csv("submission.csv", index=False)




## === cell 5
def _resolve_image_path(image_dir, image_id):
    candidates = [
        image_id,
        f"{image_id}.jpg",
        f"{image_id}.JPG",
        f"{image_id}.jpeg",
        f"{image_id}.JPEG",
        f"{image_id}.png",
        f"{image_id}.PNG",
    ]
    for name in candidates:
        p = os.path.join(image_dir, name)
        if os.path.isfile(p):
            return p
    return None


def _load_image_feature(image_path, size=(64, 64)):
    try:
        img = Image.open(image_path).convert("RGB").resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (H,W,3)
        return arr.reshape(-1)  # flattened
    except Exception:
        return None


def train_predict_lr_on_pixels(
    train_df, test_df, image_dir, target_cols, size=(64, 64), seed=42
):
    X_train_list, y_list, ok_train_ids = [], [], []
    for _, row in train_df.iterrows():
        img_id = row["image_id"]
        img_path = _resolve_image_path(image_dir, img_id)
        if img_path is None:
            continue
        feat = _load_image_feature(img_path, size=size)
        if feat is None:
            continue
        X_train_list.append(feat)
        y_list.append(int(np.argmax(row[target_cols].values)))
        ok_train_ids.append(img_id)

    X_test_list, ok_test_ids = [], []
    for _, row in test_df.iterrows():
        img_id = row["image_id"]
        img_path = _resolve_image_path(image_dir, img_id)
        if img_path is None:
            X_test_list.append(None)
        else:
            feat = _load_image_feature(img_path, size=size)
            X_test_list.append(feat if feat is not None else None)
        ok_test_ids.append(img_id)

    if len(X_train_list) == 0:
        raise RuntimeError(
            "No training images could be loaded; cannot build fallback model. "
            "Check IMAGES_DIR and image_id filename resolution."
        )

    X_train = np.vstack(X_train_list)
    y = np.array(y_list)

    priors = train_df[target_cols].mean().astype(float).values  # (4,)

    clf = LogisticRegression(
        max_iter=300,
        solver="lbfgs",
        multi_class="multinomial",
        n_jobs=1,
        random_state=seed,
        C=2.0,
    )
    clf.fit(X_train, y)

    preds = np.zeros((len(test_df), len(target_cols)), dtype=np.float32)
    for i, feat in enumerate(X_test_list):
        if feat is None:
            preds[i, :] = priors
        else:
            preds[i, :] = clf.predict_proba(feat.reshape(1, -1))[0]

    return preds




## === cell 6
if len(submissions_all) == 0:
    train = pd.read_csv(TRAIN_PATH)
    test = pd.read_csv(TEST_PATH)

    for c in ["image_id"] + TARGET_COLS:
        if c not in train.columns:
            raise ValueError(f"train.csv missing expected column: {c}")
    if "image_id" not in test.columns:
        raise ValueError("test.csv missing expected column: image_id")

    submission_avg = train_predict_lr_on_pixels(
        train_df=train,
        test_df=test,
        image_dir=IMAGES_DIR,
        target_cols=TARGET_COLS,
        size=(64, 64),
        seed=42,
    )
    make_submission_file(submission_avg)
else:
    submission_avg = ensemble(submissions_all, [0, 2, 4], [0.15, 0.8, 0.05])
    make_submission_file(submission_avg)

print("Wrote submission.csv")
print(pd.read_csv("submission.csv").head())
