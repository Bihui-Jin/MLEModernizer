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

0.966306644390566

# 6. Current score

0.63546

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I added robust handling for the submissions directory: the script now searches for any CSV files, falls back to the provided `sample_submission.csv` when none are found, and adjusts the ensemble logic to work with the actual number of submissions (using equal weights if none are supplied). This prevents the original `IndexError`, guarantees a valid `submission.csv` is written, and keeps the core workflow unchanged.'
- What this solution (achieved 0.58501) has done: 'I add a lightweight image‑based model that creates its own prediction CSV when only the sample submission is available. The new predictions are inserted into the list of submissions before the existing ensemble step, so the script still writes a valid `submission.csv` while improving the score by providing non‑constant probabilities.'
- What this solution (achieved 0.63546) has done: 'We keep the existing workflow but add a second, stronger model (a RandomForest wrapped in OneVsRest) and average its predictions with the original LogisticRegression predictions. This modest ensemble is expected to raise the ROC‑AUC toward the target while preserving the core logic and still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import pandas as pd
import numpy as np




## === cell 1
SUBMISSIONS_PATH = "/kaggle/input/submissions/"

submissions_all = []
if os.path.isdir(SUBMISSIONS_PATH):
    submissions_all = sorted(
        glob.glob(os.path.join(SUBMISSIONS_PATH, "**", "*.csv"), recursive=True)
    )

if not submissions_all:
    possible_paths = [
        "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "sample_submission.csv",
    ]
    for p in possible_paths:
        if os.path.isfile(p):
            submissions_all = [p]
            break

print("Found submissions:", submissions_all)

if len(submissions_all) == 1 and submissions_all[0].endswith("sample_submission.csv"):
    try:
        from PIL import Image
        from sklearn.linear_model import LogisticRegression
        from sklearn.multiclass import OneVsRestClassifier
        from sklearn.ensemble import RandomForestClassifier
    except Exception as e:
        raise ImportError(
            "Required packages for image modeling (Pillow, scikit‑learn) are missing."
        ) from e

    train_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
    test_csv_path = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"
    images_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images"

    train_df = pd.read_csv(train_csv_path)
    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    y_train = train_df[label_cols].values

    def load_image(img_id):
        img_path = os.path.join(images_dir, f"{img_id}.jpg")
        img = Image.open(img_path).convert("RGB").resize((64, 64))
        return np.asarray(img, dtype=np.float32).reshape(-1) / 255.0

    X_train = np.stack([load_image(img_id) for img_id in train_df["image_id"]])
    lr_clf = OneVsRestClassifier(
        LogisticRegression(
            max_iter=1000, n_jobs=1, class_weight="balanced", solver="lbfgs"
        )
    )
    lr_clf.fit(X_train, y_train)

    test_df = pd.read_csv(test_csv_path)
    X_test = np.stack([load_image(img_id) for img_id in test_df["image_id"]])
    probs_lr = lr_clf.predict_proba(X_test)

    rf_clf = OneVsRestClassifier(
        RandomForestClassifier(
            n_estimators=200,
            n_jobs=-1,
            random_state=42,
            class_weight="balanced",
            max_depth=None,
        )
    )
    rf_clf.fit(X_train, y_train)
    probs_rf = rf_clf.predict_proba(X_test)

    probs = (probs_lr + probs_rf) / 2.0

    model_pred_path = "model_prediction.csv"
    pred_df = test_df.copy()
    pred_df[label_cols] = probs
    pred_df.to_csv(model_pred_path, index=False)
    print(f"Model predictions written to {model_pred_path}")

    submissions_all.insert(0, model_pred_path)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Average selected submissions with given weights.
    If weights are not provided, equal weighting is used.
    """
    if weights is None:
        weights = [1.0 / len(sub_idx)] * len(sub_idx)

    weight_sum = sum(weights)
    weights = [w / weight_sum for w in weights]

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        print(
            f"I'm taking submission {submissions_all[idx]} with weight {weights[i]:.4f}"
        )
        df = pd.read_csv(submissions_all[idx])
        arr = df.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].values
        submission_with_weight.append(arr * weights[i])
    submission_avg = sum(submission_with_weight)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg, submissions_all):
    """
    Create the final submission file using the same structure as the first
    CSV (usually the sample submission). The probability columns are replaced
    by the averaged predictions.
    """
    template_path = submissions_all[0]
    submission_df = pd.read_csv(template_path)
    submission_df[["healthy", "multiple_diseases", "rust", "scab"]] = submission_avg
    submission_df.to_csv("submission.csv", index=False)
    print("Submission written to submission.csv")




## === cell 4
available_indices = list(range(len(submissions_all)))

if len(available_indices) == 0:
    raise RuntimeError("No submission CSV files found to build a submission.")

sub_idx = available_indices
weights = None  # default to equal weighting

submission_avg = ensemble(submissions_all, sub_idx, weights)
make_submission_file(submission_avg, submissions_all)
