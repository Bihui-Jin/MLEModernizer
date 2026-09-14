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

0.54806

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the missing‑file ensemble loading with a simple baseline that uses the average label frequencies from the training set. This removes the FileNotFoundError, defines the needed variables, and writes a correctly‑formatted `submission.csv` so the pipeline finishes successfully.'
- What this solution (achieved 0.54806) has done: 'I replace the constant‑frequency baseline with a very lightweight model that uses the numeric part of each image filename as a feature. A logistic regression is trained separately for each target column on this single feature, then used to predict probabilities for the test images. Where an image id also appears in the training set we fall back to the true labels (which boosts the ROC‑AUC). This adds minimal code, respects the existing workflow, and should raise the score toward the target.'
- What this solution (achieved 0.54806) has done: 'I add a simple binary feature indicating whether the image comes from the training set (“Train_…”) and use both this flag and the numeric part of the filename as inputs to the logistic‑regression models. This keeps the core approach unchanged while giving the model a little more information, which should raise the ROC‑AUC score toward the target.'
- What this solution (achieved 0.54806) has done: 'I add a lightweight image‑based feature extraction step (mean R, G, B values) using Pillow (fallbacking to zeros if Pillow is unavailable) and include these three new columns together with the existing numeric features for the logistic‑regression models. This should give the models genuine visual information and raise the ROC‑AUC toward the target while keeping the original workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression

try:
    from PIL import Image

    _PIL_AVAILABLE = True
except Exception:
    _PIL_AVAILABLE = False


def compute_image_means(img_path):
    """Return mean R, G, B values for an image. If reading fails, return zeros."""
    if not _PIL_AVAILABLE:
        return 0.0, 0.0, 0.0
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            arr = np.array(img) / 255.0  # normalize
            r_mean = arr[:, :, 0].mean()
            g_mean = arr[:, :, 1].mean()
            b_mean = arr[:, :, 2].mean()
            return float(r_mean), float(g_mean), float(b_mean)
    except Exception:
        return 0.0, 0.0, 0.0




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

train_df["is_train"] = train_df["image_id"].str.startswith("Train").astype(int)
test_df["is_train"] = test_df["image_id"].str.startswith("Train").astype(int)

base_dir = os.path.abspath(
    os.path.join(train_path, os.pardir)
)  # .../plant-pathology-2020-fgvc7
images_dir = os.path.join(base_dir, "images")


def add_image_features(df):
    r_means, g_means, b_means = [], [], []
    for img_id in df["image_id"]:
        img_path = os.path.join(images_dir, img_id)
        r, g, b = compute_image_means(img_path)
        r_means.append(r)
        g_means.append(g)
        b_means.append(b)
    df["r_mean"] = r_means
    df["g_mean"] = g_means
    df["b_mean"] = b_means
    return df


train_df = add_image_features(train_df)
test_df = add_image_features(test_df)




## === cell 2
models = {}
feature_cols = ["img_num", "is_train", "r_mean", "g_mean", "b_mean"]
for col in target_cols:
    lr = LogisticRegression(solver="lbfgs", max_iter=300, C=2.0, n_jobs=1)
    lr.fit(train_df[feature_cols], train_df[col])
    models[col] = lr

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
