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

0.9690752798341838

# 6. Current score

0.4746

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing‑file ensemble loading with a simple baseline that uses the average label frequencies from the training set. This removes the FileNotFoundError, defines the needed variables, and writes a correctly‑formatted `submission.csv` so the pipeline finishes successfully.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑frequency baseline with a very lightweight model that uses the numeric part of each image filename as a feature. A logistic regression is trained separately for each target column on this single feature, then used to predict probabilities for the test images. Where an image id also appears in the training set we fall back to the true labels (which boosts the ROC‑AUC). This adds minimal code, respects the existing workflow, and should raise the score toward the target.'
- What this solution (achieved 0.54806) has done: 'I add a simple binary feature indicating whether the image comes from the training set (“Train_…”) and use both this flag and the numeric part of the filename as inputs to the logistic‑regression models. This keeps the core approach unchanged while giving the model a little more information, which should raise the ROC‑AUC score toward the target.'
- What this solution (achieved 0.54806) has done: 'I add a lightweight image‑based feature extraction step (mean R, G, B values) using Pillow (fallbacking to zeros if Pillow is unavailable) and include these three new columns together with the existing numeric features for the logistic‑regression models. This should give the models genuine visual information and raise the ROC‑AUC toward the target while keeping the original workflow unchanged.'
- What this solution (achieved 0.47695) has done: 'The update adds richer image statistics (standard deviations for each colour channel) and switches from a simple logistic regression to a Gradient Boosting classifier, which can capture non‑linear patterns in the same lightweight feature set. These changes keep the overall workflow unchanged while providing more discriminative information, aiming to raise the ROC‑AUC from the current 0.55 toward the target 0.97.'
- What this solution (achieved 0.47785) has done: 'I add a simple polynomial feature (the square of the numeric part of the filename) and increase the GradientBoosting depth and number of trees to give the model more capacity. These changes keep the overall workflow unchanged while providing richer information that should raise the ROC‑AUC toward the target.'
- What this solution (achieved 0.4746) has done: 'I add lightweight color‑histogram features (16 bins per RGB channel) to the image statistics, include them in the feature set, and slightly increase the GradientBoosting capacity (more trees and a deeper depth). These changes keep the overall workflow unchanged while giving the model richer visual information, which should raise the ROC‑AUC toward the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


def compute_image_stats(img_path):
    """
    Return mean and standard deviation for R, G, B channels.
    If reading fails, return zeros for all six values.
    """
    if not _PIL_AVAILABLE:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img) / 255.0  # normalize to [0,1]
            r = arr[:, :, 0]
            g = arr[:, :, 1]
            b = arr[:, :, 2]
            return (
                float(r.mean()),
                float(g.mean()),
                float(b.mean()),
                float(r.std()),
                float(g.std()),
                float(b.std()),
            )
    except Exception:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def compute_color_histograms(img_path, bins=16):
    """
    Return flattened normalized histograms for R, G, B channels.
    If reading fails, return zeros for all bins.
    """
    if not _PIL_AVAILABLE:
        return np.zeros(bins * 3, dtype=float)
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img)
            hist_r, _ = np.histogram(
                arr[:, :, 0], bins=bins, range=(0, 256), density=True
            )
            hist_g, _ = np.histogram(
                arr[:, :, 1], bins=bins, range=(0, 256), density=True
            )
            hist_b, _ = np.histogram(
                arr[:, :, 2], bins=bins, range=(0, 256), density=True
            )
            return np.concatenate([hist_r, hist_g, hist_b]).astype(float)
    except Exception:
        return np.zeros(bins * 3, dtype=float)




## === cell 1
train_path = "../input/plant-pathology-2020-fgvc7/train.csv"
test_path = "../input/plant-pathology-2020-fgvc7/test.csv"
sample_sub_path = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
sample_sub = pd.read_csv(sample_sub_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def extract_num(img_id):
    base = os.path.splitext(img_id)[0]
    parts = base.split("_")
    for part in parts[::-1]:
        if part.isdigit():
            return int(part)
    return 0  # fallback if format unexpected


train_df["img_num"] = train_df["image_id"].apply(extract_num)
test_df["img_num"] = test_df["image_id"].apply(extract_num)

train_df["img_num_sq"] = train_df["img_num"] ** 2
test_df["img_num_sq"] = test_df["img_num"] ** 2

train_df["is_train"] = train_df["image_id"].str.startswith("Train").astype(int)
test_df["is_train"] = test_df["image_id"].str.startswith("Train").astype(int)

base_dir = os.path.abspath(os.path.join(train_path, os.pardir))
images_dir = os.path.join(base_dir, "images")


def add_image_features(df):
    r_means, g_means, b_means = [], [], []
    r_stds, g_stds, b_stds = [], [], []
    hist_features = []

    for img_id in df["image_id"]:
        img_path = os.path.join(images_dir, img_id)
        r, g, b, rs, gs, bs = compute_image_stats(img_path)
        r_means.append(r)
        g_means.append(g)
        b_means.append(b)
        r_stds.append(rs)
        g_stds.append(gs)
        b_stds.append(bs)

        hist = compute_color_histograms(img_path, bins=16)
        hist_features.append(hist)

    df["r_mean"] = r_means
    df["g_mean"] = g_means
    df["b_mean"] = b_means
    df["r_std"] = r_stds
    df["g_std"] = g_stds
    df["b_std"] = b_stds

    hist_array = np.stack(hist_features)  # shape (n_rows, 48)
    for i in range(48):
        df[f"hist_{i}"] = hist_array[:, i]

    return df


train_df = add_image_features(train_df)
test_df = add_image_features(test_df)




## === cell 2
models = {}
hist_cols = [f"hist_{i}" for i in range(48)]
feature_cols = [
    "img_num",
    "img_num_sq",
    "is_train",
    "r_mean",
    "g_mean",
    "b_mean",
    "r_std",
    "g_std",
    "b_std",
] + hist_cols

for col in target_cols:
    gbc = GradientBoostingClassifier(
        n_estimators=800,  # more trees for higher capacity
        learning_rate=0.05,
        max_depth=6,  # slightly deeper trees
        random_state=42,
    )
    gbc.fit(train_df[feature_cols], train_df[col])
    models[col] = gbc

label_lookup = train_df.set_index("image_id")[target_cols].to_dict(orient="index")




## === cell 3
submission = sample_sub.copy()

for col in target_cols:
    exact_vals = []
    pred_vals = []
    for idx, row in submission.iterrows():
        img_id = row["image_id"]
        if img_id in label_lookup:
            exact_vals.append(label_lookup[img_id][col])
            pred_vals.append(None)
        else:
            feats = test_df.loc[test_df["image_id"] == img_id, feature_cols]
            if not feats.empty:
                pred = models[col].predict_proba(feats.values)[0, 1]
            else:
                pred = train_df[col].mean()
            exact_vals.append(None)
            pred_vals.append(pred)
    submission[col] = [
        ev if ev is not None else pv for ev, pv in zip(exact_vals, pred_vals)
    ]




## === cell 4
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
