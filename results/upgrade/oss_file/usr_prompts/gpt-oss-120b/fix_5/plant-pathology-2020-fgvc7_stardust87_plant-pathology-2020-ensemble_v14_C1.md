# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")
IMAGES_DIR = os.path.join(DATA_ROOT, "images")


def load_image_vectors(df, image_dir, size=(128, 128)):
    """
    Load images, resize to a larger 128×128 resolution (instead of 64×64)
    to provide richer pixel information for the classifier.
    """
    vectors = []
    ids_kept = []
    for img_id in df["image_id"]:
        img_path = os.path.join(image_dir, f"{img_id}.jpg")
        if not os.path.isfile(img_path):
            candidates = [
                f for f in os.listdir(image_dir) if f.lower().startswith(img_id.lower())
            ]
            if candidates:
                img_path = os.path.join(image_dir, candidates[0])
            else:
                continue
        try:
            img = Image.open(img_path).convert("RGB").resize(size)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # normalise
            vectors.append(arr.ravel())
            ids_kept.append(img_id)
        except Exception:
            continue
    return np.stack(vectors), ids_kept




## === cell 1
def train_and_predict():
    train_df = pd.read_csv(TRAIN_CSV)
    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    X_train, kept_ids = load_image_vectors(train_df, IMAGES_DIR)
    y_train = train_df.loc[train_df["image_id"].isin(kept_ids), label_cols].values

    base_clf = LogisticRegression(
        max_iter=500,
        solver="saga",
        n_jobs=5,
        class_weight="balanced",
        C=2.0,
        penalty="l2",
        random_state=42,
    )
    clf = MultiOutputClassifier(base_clf, n_jobs=5)
    clf.fit(X_train, y_train)

    test_df = pd.read_csv(TEST_CSV)
    X_test, test_ids = load_image_vectors(test_df, IMAGES_DIR)

    probas = clf.predict_proba(X_test)  # list of arrays per label
    prob_matrix = np.column_stack([p[:, 1] for p in probas])  # (n_samples, 4)

    template = pd.read_csv(SAMPLE_SUBMISSION)
    submission = pd.DataFrame(
        {
            "image_id": test_df["image_id"],
            "healthy": prob_matrix[:, 0],
            "multiple_diseases": prob_matrix[:, 1],
            "rust": prob_matrix[:, 2],
            "scab": prob_matrix[:, 3],
        }
    )
    submission = submission[template.columns]
    submission_path = "submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Model‑based submission written to {submission_path}")

    return True




## === cell 2
fallback = True
try:
    import glob

    SUBMISSIONS_PATH = os.path.join(DATA_ROOT, "submissions")
    if not os.path.isdir(SUBMISSIONS_PATH):
        SUBMISSIONS_PATH = DATA_ROOT
    submission_files = [
        p
        for p in glob.glob(os.path.join(SUBMISSIONS_PATH, "**/*.csv"), recursive=True)
        if "submission" in os.path.basename(p).lower()
    ]
    if len(submission_files) >= 2:
        dfs = [
            pd.read_csv(p)[["healthy", "multiple_diseases", "rust", "scab"]].astype(
                float
            )
            for p in submission_files[:2]
        ]
        avg = (dfs[0] + dfs[1]) / 2.0
        tmpl = pd.read_csv(SAMPLE_SUBMISSION)
        tmpl[["healthy", "multiple_diseases", "rust", "scab"]] = avg.values
        tmpl.to_csv("submission.csv", index=False)
        print("Ensembled existing submissions.")
        fallback = False
except Exception as e:
    print(f"Ensembling step failed ({e}), will use model fallback.")
    fallback = True

if fallback:
    try:
        success = train_and_predict()
        if not success:
            raise RuntimeError("Model prediction failed.")
    except Exception as e:
        print(f"Model fallback failed ({e}), using mean‑label baseline.")
        train_df = pd.read_csv(TRAIN_CSV)
        baseline_probs = (
            train_df[["healthy", "multiple_diseases", "rust", "scab"]].mean().values
        )
        test_df = pd.read_csv(TEST_CSV)
        baseline_matrix = pd.DataFrame(
            [baseline_probs] * len(test_df),
            columns=["healthy", "multiple_diseases", "rust", "scab"],
        )
        submission_df = pd.concat([test_df, baseline_matrix], axis=1)
        submission_df.to_csv("submission.csv", index=False)
        print("Baseline submission.csv written successfully.")
