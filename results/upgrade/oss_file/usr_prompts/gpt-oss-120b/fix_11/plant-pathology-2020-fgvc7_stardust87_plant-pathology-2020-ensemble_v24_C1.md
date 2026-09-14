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

0.9678361818267092

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'Implemented a robust ensembling routine that skips any CSVs whose row count does not match the test set (preventing shape‑broadcast errors). The function now accepts an `expected_rows` argument, validates each submission, and only aggregates compatible files. In the main workflow we compute the test row count, pass it to `ensemble`, and safely select a matching template submission for writing the final CSV. If no valid submissions exist, the fallback baseline is used. This ensures a correct `submission.csv` is always produced without runtime crashes.'
- What this solution (achieved 0.5) has done: 'I add a lightweight image‑based fallback model that trains a simple logistic regression on tiny RGB thumbnails of the training images and uses it to predict probabilities for the test set. This model replaces the naïve class‑mean baseline when no valid ensemble is available, giving a more informed prediction and moving the ROC‑AUC score closer to the target while keeping the original ensembling logic intact.'
- What this solution (achieved 0.5) has done: 'Implemented a higher‑resolution thumbnail extractor (64 × 64) to give the logistic‑regression model richer visual information, increased its iteration limit and used a balanced class weight for better convergence, and blended its probability output with the simple class‑mean baseline (70 % model + 30 % mean). This modest enrichment keeps the original lightweight “fallback” architecture while providing a stronger, more calibrated prediction that should raise the ROC‑AUC toward the target. The rest of the pipeline (submission handling, ensembling, fallback logic) remains unchanged.'
- What this solution (achieved 0.5) has done: 'I remove the unsupported `n_jobs` argument from the LogisticRegression constructor so the fallback image‑based model can train without raising an exception. This change lets the model produce non‑constant predictions instead of falling back to the class‑mean baseline, which should raise the ROC‑AUC from 0.5 toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.5) has done: 'I boost the lightweight fallback model by scaling the thumbnail features, increasing the logistic‑regression iteration limit and giving the model’s predictions a higher share in the final blend (alpha = 0.9). These small, targeted adjustments keep the overall pipeline unchanged while expected to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I increase the thumbnail resolution used for feature extraction from 64×64 to 128×128, giving the logistic‑regression model richer visual information while keeping the same model architecture and training pipeline. This modest change should improve the predictive power and raise the ROC‑AUC toward the target score without altering any core logic.'
- What this solution (achieved 0.5) has done: 'I augment the training data with horizontally‑flipped thumbnails to give the logistic‑regression model more visual diversity, and slightly reduce the blending weight of the model (alpha = 0.6) so the calibrated output benefits from the more reliable class‑mean baseline. These small adjustments keep the original pipeline intact while providing the model with extra training examples and a better‑balanced final blend, which should move the ROC‑AUC score closer to the target.'
- What this solution (achieved 0.5) has done: 'I augment the fallback image model with additional flipped variants (horizontal, vertical, and both) to give it more training data, increase the LogisticRegression iteration limit for better convergence, and raise the blending weight α so the model’s predictions dominate the final submission. These modest changes stay within the existing lightweight pipeline while expected to lift the ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline




## === cell 1
POSSIBLE_DIRS = [
    "/kaggle/input/submissions",
    "/kaggle/input/plant-pathology-2020-fgvc7",
    "/kaggle/input",
]

submissions_all = []
for base_dir in POSSIBLE_DIRS:
    if os.path.isdir(base_dir):
        for dirname, _, filenames in os.walk(base_dir):
            for filename in filenames:
                if filename.lower().endswith(".csv"):
                    submissions_all.append(os.path.join(dirname, filename))
submissions_all.sort()
print("Found submission files:", submissions_all)




## === cell 2
def ensemble(submissions_list, sub_idx, weights, expected_rows=None):
    """
    Weighted average of a list of submission CSVs.
    Returns None if no valid submissions are found.
    """
    if not submissions_list:
        return None

    max_idx = len(submissions_list) - 1
    valid_idx = [i for i in sub_idx if 0 <= i <= max_idx]
    if not valid_idx:
        return None

    idx_weight_pairs = [(i, w) for i, w in zip(sub_idx, weights) if i in valid_idx]
    if not idx_weight_pairs:
        return None

    total_w = sum(w for _, w in idx_weight_pairs)
    if total_w == 0:
        return None
    idx_weight_pairs = [(i, w / total_w) for i, w in idx_weight_pairs]

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    weighted_sum = None

    for idx, weight in idx_weight_pairs:
        path = submissions_list[idx]
        print(f"Ensembling submission {path} with normalized weight {weight:.4f}")
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()

        if not all(col in df.columns for col in cols):
            print(f"Skipping {path}: missing required columns.")
            continue

        if expected_rows is not None and df.shape[0] != expected_rows:
            print(
                f"Skipping {path}: row count {df.shape[0]} does not match expected {expected_rows}."
            )
            continue

        df_vals = df[cols].values.astype(float)
        if weighted_sum is None:
            weighted_sum = df_vals * weight
        else:
            weighted_sum += df_vals * weight

    return weighted_sum




## === cell 3
def _load_image_features(image_ids, base_dir, size=(128, 128)):
    """
    Load images, resize to `size`, flatten and normalize.
    Returns a NumPy array of shape (len(image_ids), size[0]*size[1]*3).
    """
    features = []
    img_dir = os.path.join(base_dir, "images")
    for img_id in image_ids:
        img_path = os.path.join(img_dir, img_id)
        try:
            with Image.open(img_path) as img:
                img = img.convert("RGB")
                img = img.resize(size, Image.BILINEAR)
                arr = np.asarray(img, dtype=np.float32) / 255.0
                features.append(arr.ravel())
        except Exception as e:
            print(f"Warning: could not process {img_path}: {e}")
            features.append(np.zeros(size[0] * size[1] * 3, dtype=np.float32))
    return np.stack(features)


def train_fallback_model(train_path, test_path, out_path="submission.csv"):
    """
    Train a simple logistic‑regression model on low‑resolution RGB thumbnails,
    augment training data with horizontal, vertical and both‑axis flips,
    then blend its predictions with the class‑mean baseline.
    """
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    cols = ["healthy", "multiple_diseases", "rust", "scab"]
    base_dir = os.path.dirname(train_path)

    print("Loading training image features (128x128)...")
    X_train = _load_image_features(train_df["image_id"].values, base_dir)
    print("Loading test image features (128x128)...")
    X_test = _load_image_features(test_df["image_id"].values, base_dir)

    h, w, c = 128, 128, 3  # matches the size used in _load_image_features
    X_train_reshaped = X_train.reshape(-1, h, w, c)

    X_hflip = X_train_reshaped[:, :, ::-1, :].reshape(-1, h * w * c)
    X_vflip = X_train_reshaped[:, ::-1, :, :].reshape(-1, h * w * c)
    X_hvflip = X_train_reshaped[:, ::-1, ::-1, :].reshape(-1, h * w * c)

    X_train = np.concatenate([X_train, X_hflip, X_vflip, X_hvflip], axis=0)

    y_train = train_df[cols].values.astype(float)
    y_train = np.concatenate([y_train, y_train, y_train, y_train], axis=0)

    print("Training One‑Vs‑Rest logistic regression models (with scaling)...")
    clf = OneVsRestClassifier(
        make_pipeline(
            StandardScaler(),
            LogisticRegression(
                solver="lbfgs",
                max_iter=5000,  # more iterations for better convergence
                class_weight="balanced",
            ),
        )
    )
    clf.fit(X_train, y_train)

    print("Predicting on test images...")
    probs = clf.predict_proba(X_test)  # shape (n_test, n_labels)

    class_means = train_df[cols].mean().values  # shape (n_labels,)

    alpha = 0.9  # give more weight to the (now stronger) model
    blended = alpha * probs + (1 - alpha) * class_means

    sub_df = pd.DataFrame()
    sub_df["image_id"] = test_df["image_id"]
    for i, c in enumerate(cols):
        sub_df[c] = blended[:, i]

    sub_df.to_csv(out_path, index=False)
    print(f"Model‑based fallback submission written to {out_path}")




## === cell 4
def fallback_submission(train_path, test_path, out_path="submission.csv"):
    """
    Try the lightweight image model first; if it fails, fall back to class‑mean probabilities.
    """
    try:
        train_fallback_model(train_path, test_path, out_path=out_path)
    except Exception as e:
        print(f"Model fallback failed ({e}); using class‑mean baseline.")
        train_df = pd.read_csv(train_path)
        test_df = pd.read_csv(test_path)
        cols = ["healthy", "multiple_diseases", "rust", "scab"]
        class_means = train_df[cols].mean()
        sub_df = pd.DataFrame()
        sub_df["image_id"] = test_df["image_id"]
        for c in cols:
            sub_df[c] = class_means[c]
        sub_df.to_csv(out_path, index=False)
        print(f"Mean‑baseline submission written to {out_path}")




## === cell 5
TRAIN_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/train.csv"
TEST_CSV = "/kaggle/input/plant-pathology-2020-fgvc7/test.csv"

test_len = pd.read_csv(TEST_CSV).shape[0]

submission_avg = ensemble(
    submissions_all, [0, 2, 4], [0.15, 0.8, 0.05], expected_rows=test_len
)

if submission_avg is not None:
    template_path = None
    for path in submissions_all:
        df = pd.read_csv(path)
        df.columns = df.columns.str.strip()
        if df.shape[0] == test_len and all(
            col in df.columns
            for col in ["healthy", "multiple_diseases", "rust", "scab"]
        ):
            template_path = path
            break

    if template_path is not None:
        base_df = pd.read_csv(template_path)
        base_df.iloc[:, 1:] = submission_avg
        base_df.to_csv("submission.csv", index=False)
        print("Ensembled submission written to submission.csv")
    else:
        fallback_submission(TRAIN_CSV, TEST_CSV, out_path="submission.csv")
else:
    fallback_submission(TRAIN_CSV, TEST_CSV, out_path="submission.csv")
