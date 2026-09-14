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

0.9663173272786793

# 6. Current score

0.51256

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing ensemble loading with a simple baseline that fills every test row with the average label values computed from the training data. This removes the missing‑file error, ensures the variable names exist, and creates a valid `submission.csv` file. The core logic (reading CSVs, averaging, writing submission) is kept minimal and deterministic.'
- What this solution (achieved 0.49261) has done: 'I add a lightweight, data‑driven adjustment to the constant‑baseline predictions by using each image’s file size as a proxy feature. The file size is read for every test image, normalized, and a small scaling factor is applied to the mean label probabilities; this introduces modest variation that can raise the ROC‑AUC above the constant‑baseline 0.5 while preserving the original simple‑averaging logic. The change is minimal, adds no heavy dependencies, and keeps the overall workflow unchanged.'
- What this solution (achieved 0.50739) has done: 'I add a lightweight logistic‑regression model that uses each image’s file‑size as a single feature. The model is trained on the known labels from the training set and then used to predict probabilities for the test images, replacing the previous constant‑plus‑size‑scaling approach. This keeps the pipeline simple, adds only a few lines, and is expected to raise the ROC‑AUC toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.49922) has done: 'I replace the single‑feature LogisticRegression with a distance‑weighted K‑Nearest‑Neighbors regressor, which can capture non‑linear trends of the image‑size proxy while keeping the overall pipeline unchanged. This lightweight change is expected to raise the ROC‑AUC toward the target without altering the core logic or adding heavy dependencies.'
- What this solution (achieved 0.48219) has done: 'I added a second lightweight feature – the numeric part of the image filename – and switched the K‑Nearest‑Neighbors model to use a single nearest neighbor (k = 1). This keeps the overall pipeline (size‑based feature, per‑label models, CSV output) intact while giving the model more discriminative information, which should move the ROC‑AUC closer to the target score.'
- What this solution (achieved 0.48385) has done: 'I keep the overall workflow unchanged but improve the predictor by using a slightly larger, distance‑weighted k‑Nearest‑Neighbors model (k = 5) and blend its output with the simple class‑frequency baseline. This smooths the predictions and adds a modest amount of prior information, which should raise the ROC‑AUC toward the target while preserving the original lightweight logic.'
- What this solution (achieved 0.56228) has done: 'I replace the simple distance‑weighted k‑Nearest‑Neighbors with a light LogisticRegression model that uses the same two features plus an interaction term, and increase the blend weight so the learned model dominates the baseline. This adds only a few lines, keeps the overall workflow unchanged, and is expected to raise the ROC‑AUC toward the target while still writing a valid `submission.csv`.'
- What this solution (achieved 0.50939) has done: 'I replace the single‑feature LogisticRegression models with lightweight RandomForest classifiers, which can capture non‑linear patterns in the file‑size and numeric‑id features while keeping the overall pipeline unchanged. This change is allowed because the current gap exceeds 30 % of the target, and the new models are expected to raise the ROC‑AUC toward the desired score. I also adjust the import statements accordingly and keep the same blending with the baseline probabilities.'
- What this solution (achieved 0.51256) has done: 'I strengthen the lightweight RandomForest models by giving them more trees, limiting depth to capture useful patterns without over‑fitting, and handling class imbalance with `class_weight='balanced'`. I also shift the blending weight to rely a bit more on the model predictions (0.95 × model + 0.05 × baseline), which should raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier  # enhanced model
from sklearn.linear_model import LogisticRegression  # kept for compatibility

BASE_DIR = "../input/plant-pathology-2020-fgvc7"
IMAGES_DIR = os.path.join(BASE_DIR, "images")



## === cell 1
train_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_path)

label_cols = ["healthy", "multiple_diseases", "rust", "scab"]
baseline_probs = train_df[label_cols].mean()

train_ids = train_df["image_id"].astype(str)


def extract_numeric_id(img_id):
    parts = img_id.split("_")
    try:
        return int(parts[-1])
    except ValueError:
        return 0


train_sizes = []
train_nums = []
for img_id in train_ids:
    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    try:
        size = os.path.getsize(img_path)
    except OSError:
        size = np.nan
    train_sizes.append(size)
    train_nums.append(extract_numeric_id(img_id))

train_sizes = np.array(train_sizes, dtype=float)
size_mean = np.nanmean(train_sizes)
size_std = np.nanstd(train_sizes) if np.nanstd(train_sizes) > 0 else 1.0
train_sizes = np.where(np.isnan(train_sizes), size_mean, train_sizes)

train_nums = np.array(train_nums, dtype=float)

size_norm = (train_sizes - size_mean) / size_std
interaction = size_norm * train_nums

X_train = np.column_stack((size_norm, train_nums, interaction))

models = {}
for col in label_cols:
    y = train_df[col].values
    rf = RandomForestClassifier(
        n_estimators=500,  # more trees for better stability
        max_depth=15,  # restrict depth to avoid excessive over‑fit
        class_weight="balanced",  # compensate for label imbalance
        random_state=42,
        n_jobs=5,
    )
    rf.fit(X_train, y)
    models[col] = rf



## === cell 2
sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sub = pd.read_csv(sample_sub_path)

test_ids = sub["image_id"].astype(str)

test_sizes = []
test_nums = []
for img_id in test_ids:
    img_path = os.path.join(IMAGES_DIR, f"{img_id}.jpg")
    try:
        size = os.path.getsize(img_path)
    except OSError:
        size = size_mean  # fallback to mean size
    test_sizes.append(size)
    test_nums.append(extract_numeric_id(img_id))

test_sizes = np.array(test_sizes, dtype=float)
test_sizes = np.where(np.isnan(test_sizes), size_mean, test_sizes)

test_nums = np.array(test_nums, dtype=float)

size_norm_test = (test_sizes - size_mean) / size_std
interaction_test = size_norm_test * test_nums

X_test = np.column_stack((size_norm_test, test_nums, interaction_test))

for col in label_cols:
    model = models[col]
    model_pred = model.predict_proba(X_test)[:, 1]
    model_pred = np.clip(model_pred, 0.0, 1.0)
    blended = 0.95 * model_pred + 0.05 * baseline_probs[col]
    sub[col] = np.clip(blended, 0.0, 1.0)



## === cell 3
output_path = "submission.csv"
sub.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
