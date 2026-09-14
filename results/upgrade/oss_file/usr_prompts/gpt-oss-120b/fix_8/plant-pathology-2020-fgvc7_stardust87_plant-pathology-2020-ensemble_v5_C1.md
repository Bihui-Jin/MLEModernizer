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

0.9630283601755996

# 6. Current score

0.54803

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I correct the file paths so the script can read the dataset, make the fallback baseline work when no prior submissions exist, and adjust the submission‑creation logic to always write a proper `submission.csv` containing the required columns and predictions.'
- What this solution (achieved 0.5) has done: 'I broaden the ensembling step to average **all** previously saved CSV submissions (instead of just the first two) with equal weighting, which can capture more useful signal from existing models and therefore raise the ROC AUC toward the target. If no past submissions exist the script still falls back to the simple mean‑label baseline, preserving correctness. No core modeling logic is altered, only the way predictions are combined.'
- What this solution (achieved 0.5) has done: 'I keep the overall logic unchanged but modify the ensembling step so that more recent submission files receive higher weights (assuming newer files are better) and limit ensembling to the most recent five files to reduce noise from older, poorer predictions. This small weighting tweak should raise the ROC AUC toward the target without altering the core pipeline.'
- What this solution (achieved 0.50188) has done: 'I keep the overall pipeline unchanged but replace the simple global‑mean fallback with a lightweight nearest‑neighbor average based on the numeric part of each image_id. By borrowing label information from the most similar training images (instead of a constant mean), the predictions gain variability that better matches the true distribution, which should raise the ROC‑AUC from the current 0.5 toward the target while still respecting all core‑logic constraints.'
- What this solution (achieved 0.49192) has done: 'I improve the fallback nearest‑neighbor predictor by using a larger neighbourhood (k = 20) and weighting neighbours with the inverse distance + 1, so closer training images have more influence. This change keeps the overall pipeline unchanged but should raise the ROC‑AUC toward the target while still respecting the “minimal‑change” rule.'
- What this solution (achieved 0.54803) has done: 'I add image‑based features and a lightweight logistic‑regression fallback that is blended with the existing nearest‑neighbor baseline. This keeps the original ensembling logic unchanged, but improves the fallback predictions, moving the ROC‑AUC score closer to the target while still writing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression

DATA_ROOT = "/kaggle/input/plant-pathology-2020-fgvc7"
SUBMISSIONS_PATH = "/kaggle/input/submissions/submissions/"



## === cell 1
submissions_all = []
for dirname, _, filenames in os.walk(SUBMISSIONS_PATH):
    for filename in filenames:
        if filename.lower().endswith(".csv"):
            submissions_all.append(os.path.join(dirname, filename))
submissions_all = submissions_all[::-1]  # newest first after reversal
print("Found submissions:", submissions_all)




## === cell 2
def ensemble(submissions_all, sub_idx, weights=None):
    """
    Load selected submissions, weight them and return the averaged prediction matrix.
    If weights are None, assign higher weight to newer submissions (larger index in sub_idx)
    and limit to at most five most recent files.
    """
    if len(sub_idx) > 5:
        sub_idx = sub_idx[:5]

    if weights is None:
        weights = np.arange(1, len(sub_idx) + 1)[::-1]  # newest gets largest weight
        weights = weights.astype(float)

    if len(weights) != len(sub_idx):
        raise ValueError("Length of weights must match length of sub_idx.")

    submission_with_weight = []
    for i, idx in enumerate(sub_idx):
        if idx >= len(submissions_all):
            raise IndexError(f"sub_idx {idx} out of range for submissions list.")
        print(f"Loading submission {submissions_all[idx]} with weight {weights[i]}")
        submission = pd.read_csv(submissions_all[idx])
        submission = submission.loc[
            :, ["healthy", "multiple_diseases", "rust", "scab"]
        ].values
        submission_with_weight.append(submission * weights[i])

    submission_avg = sum(submission_with_weight) / sum(weights)
    return submission_avg




## === cell 3
def make_submission_file(submission_avg):
    """
    Write `submission.csv` using the test set image IDs and the provided averaged predictions.
    """
    test_path = os.path.join(DATA_ROOT, "test.csv")
    test_df = pd.read_csv(test_path)
    submission_df = pd.DataFrame(
        submission_avg, columns=["healthy", "multiple_diseases", "rust", "scab"]
    )
    submission_df.insert(0, "image_id", test_df["image_id"])
    submission_df.to_csv("submission.csv", index=False)
    print("Wrote final submission to submission.csv (", submission_df.shape[0], "rows)")




## === cell 4
def nearest_neighbor_predictions(train_df, test_df, k=20):
    """
    For each test image, find the k nearest training images based on the numeric part
    of the image_id and compute a distance‑weighted average of their label values.
    Closer neighbours receive higher weight (weight = 1 / (distance + 1)).
    """
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["num_id"] = train_df["image_id"].str.extract(r"(\d+)").astype(int)
    test_df["num_id"] = test_df["image_id"].str.extract(r"(\d+)").astype(int)

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    preds = np.zeros((len(test_df), len(label_cols)), dtype=float)

    train_sorted = train_df.sort_values("num_id")
    train_nums = train_sorted["num_id"].values
    train_labels = train_sorted[label_cols].values

    for i, test_num in enumerate(test_df["num_id"].values):
        diffs = np.abs(train_nums - test_num)
        nearest_idx = np.argpartition(diffs, k)[:k]
        nearest_diffs = diffs[nearest_idx]
        weights = 1.0 / (nearest_diffs + 1.0)
        weighted_sum = (train_labels[nearest_idx].T * weights).T.sum(axis=0)
        preds[i] = weighted_sum / weights.sum()

    return preds


def compute_image_features(df, img_dir):
    """
    Simple image feature: mean RGB values for each image.
    Returns a (n_samples, 3) numpy array.
    """
    features = []
    for img_id in df["image_id"]:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                arr = np.array(im)
                mean_rgb = arr.mean(axis=(0, 1))  # (R, G, B)
        except Exception:
            mean_rgb = np.zeros(3)
        features.append(mean_rgb)
    return np.stack(features)


def logistic_regression_predictions(train_df, test_df):
    """
    Train a separate LogisticRegression for each label using mean RGB features,
    then predict probabilities for the test set.
    """
    img_dir = os.path.join(DATA_ROOT, "images")
    X_train = compute_image_features(train_df, img_dir)
    X_test = compute_image_features(test_df, img_dir)

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    preds = np.zeros((len(test_df), len(label_cols)), dtype=float)

    for i, col in enumerate(label_cols):
        y = train_df[col].values
        if len(np.unique(y)) == 1:
            preds[:, i] = y[0]  # constant prediction
            continue
        clf = LogisticRegression(max_iter=200, solver="lbfgs")
        clf.fit(X_train, y)
        preds[:, i] = clf.predict_proba(X_test)[:, 1]  # probability of class 1

    return preds




## === cell 5
try:
    if len(submissions_all) >= 1:
        all_indices = list(range(len(submissions_all)))
        submission_avg = ensemble(submissions_all, all_indices)
    else:
        raise ValueError("No previous submissions found for ensembling.")
except Exception as e:
    print(f"Ensemble failed ({e}); falling back to enhanced baseline.")
    train_path = os.path.join(DATA_ROOT, "train.csv")
    test_path = os.path.join(DATA_ROOT, "test.csv")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    nn_preds = nearest_neighbor_predictions(train_df, test_df, k=20)

    lr_preds = logistic_regression_predictions(train_df, test_df)

    submission_avg = (nn_preds + lr_preds) / 2.0

make_submission_file(submission_avg)
