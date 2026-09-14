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

0.64447

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I correct the file paths so the script can read the dataset, make the fallback baseline work when no prior submissions exist, and adjust the submission‑creation logic to always write a proper `submission.csv` containing the required columns and predictions.'
- What this solution (achieved 0.5) has done: 'I broaden the ensembling step to average **all** previously saved CSV submissions (instead of just the first two) with equal weighting, which can capture more useful signal from existing models and therefore raise the ROC AUC toward the target. If no past submissions exist the script still falls back to the simple mean‑label baseline, preserving correctness. No core modeling logic is altered, only the way predictions are combined.'
- What this solution (achieved 0.5) has done: 'I keep the overall logic unchanged but modify the ensembling step so that more recent submission files receive higher weights (assuming newer files are better) and limit ensembling to the most recent five files to reduce noise from older, poorer predictions. This small weighting tweak should raise the ROC AUC toward the target without altering the core pipeline.'
- What this solution (achieved 0.50188) has done: 'I keep the overall pipeline unchanged but replace the simple global‑mean fallback with a lightweight nearest‑neighbor average based on the numeric part of each image_id. By borrowing label information from the most similar training images (instead of a constant mean), the predictions gain variability that better matches the true distribution, which should raise the ROC‑AUC from the current 0.5 toward the target while still respecting all core‑logic constraints.'
- What this solution (achieved 0.49192) has done: 'I improve the fallback nearest‑neighbor predictor by using a larger neighbourhood (k = 20) and weighting neighbours with the inverse distance + 1, so closer training images have more influence. This change keeps the overall pipeline unchanged but should raise the ROC‑AUC toward the target while still respecting the “minimal‑change” rule.'
- What this solution (achieved 0.54803) has done: 'I add image‑based features and a lightweight logistic‑regression fallback that is blended with the existing nearest‑neighbor baseline. This keeps the original ensembling logic unchanged, but improves the fallback predictions, moving the ROC‑AUC score closer to the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.53757) has done: 'I keep the overall pipeline unchanged but improve the fallback baseline used when no historic submissions are found. I increase the nearest‑neighbor neighbourhood to k=50 (more “peer” information) and blend the NN and Logistic‑Regression predictions with a 0.7 / 0.3 weighting, which tends to give a stronger signal. The Logistic‑Regression step now scales the mean‑RGB features with StandardScaler, a tiny change that often boosts classification performance without altering the core modelling approach. These minimal adjustments should raise the ROC‑AUC toward the target while preserving the original logic and output format.'
- What this solution (achieved 0.57861) has done: 'I correct the dataset root path typo (`fgvg7 → fgvc7`) so the script can locate the train/test CSV files and image folder. The rest of the logic is left unchanged, preserving the existing modeling and ensembling approach; this small fix enables the fallback baseline to run and produce a valid `submission.csv`.'
- What this solution (achieved 0.62231) has done: 'I tighten the fallback predictor by (1) using a smaller neighbourhood (k=30) for the nearest‑neighbor component, (2) giving equal weight to the nearest‑neighbor and logistic‑regression predictions (dropping the tiny global‑mean term), and (3) training each logistic regression with `class_weight='balanced'` to improve ROC‑AUC on the imbalanced disease labels. These modest tweaks keep the overall pipeline unchanged while steering the score upward toward the target.'
- What this solution (achieved 0.64447) has done: 'I slightly improve the fallback predictor that runs when no past submissions exist. First, I reduce the nearest‑neighbor neighbourhood from k=30 to k=20, which has shown a modest boost in ROC‑AUC. Second, I change the blend to give a larger share to the logistic‑regression model (0.65) and add a tiny contribution (0.05) of the global‑mean label frequencies to stabilize predictions. These tiny, targeted tweaks keep the core pipeline intact while moving the score upward toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

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
def nearest_neighbor_predictions(train_df, test_df, k=30):
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
    Enhanced image feature: mean and standard deviation of RGB channels for each image.
    Returns a (n_samples, 6) numpy array (mean_R, mean_G, mean_B, std_R, std_G, std_B).
    """
    features = []
    for img_id in df["image_id"]:
        img_path = os.path.join(img_dir, f"{img_id}.jpg")
        try:
            with Image.open(img_path) as im:
                im = im.convert("RGB")
                arr = np.array(im).astype(np.float32) / 255.0
                mean_rgb = arr.mean(axis=(0, 1))  # (R, G, B)
                std_rgb = arr.std(axis=(0, 1))  # (R, G, B)
                feat = np.concatenate([mean_rgb, std_rgb])
        except Exception:
            feat = np.zeros(6)
        features.append(feat)
    return np.stack(features)


def logistic_regression_predictions(train_df, test_df):
    """
    Train a separate LogisticRegression for each label using enhanced mean/std RGB features,
    then predict probabilities for the test set.
    Features are standardized to improve logistic regression performance.
    """
    img_dir = os.path.join(DATA_ROOT, "images")
    X_train_raw = compute_image_features(train_df, img_dir)
    X_test_raw = compute_image_features(test_df, img_dir)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    X_test = scaler.transform(X_test_raw)

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    preds = np.zeros((len(test_df), len(label_cols)), dtype=float)

    for i, col in enumerate(label_cols):
        y = train_df[col].values
        if len(np.unique(y)) == 1:
            preds[:, i] = y[0]  # constant prediction
            continue
        clf = LogisticRegression(
            max_iter=800, solver="lbfgs", n_jobs=1, class_weight="balanced"
        )
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

    label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
    global_mean = train_df[label_cols].mean().values  # shape (4,)

    submission_avg = (
        0.65 * lr_preds + 0.30 * nn_preds + 0.05 * global_mean
    )  # broadcasting adds the same mean to every row

make_submission_file(submission_avg)
