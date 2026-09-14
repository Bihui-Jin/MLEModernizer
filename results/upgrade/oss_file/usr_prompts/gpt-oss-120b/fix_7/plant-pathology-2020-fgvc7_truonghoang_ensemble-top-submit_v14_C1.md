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

0.9704488894833836

# 6. Current score

0.66617

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I replace the failing ensemble loading with a simple baseline that uses the average label frequencies from the training data to fill the submission file. This removes the missing‑file error, ensures the `dsub` variable is defined, and writes a correctly‑formatted `submission.csv` so the notebook completes without runtime errors. The core logic (reading CSVs and writing the submission) remains unchanged.'
- What this solution (achieved 0.5) has done: 'I replace the constant‑frequency baseline with a tiny image‑based model: for each image I compute the mean RGB values, train a separate logistic‑regression classifier for every disease label on these three features, and use the learned probabilities for the submission. This keeps the original CSV‑reading and output logic while adding a lightweight, well‑behaved model that should raise the ROC‑AUC from the 0.5 baseline toward the target score.'
- What this solution (achieved 0.5) has done: 'I fix the NaN handling by adding a SimpleImputer, enrich the image features with per‑channel standard deviations (giving a more informative 6‑dimensional vector), and standardise the data before fitting the LogisticRegression models. These changes eliminate the runtime error, improve the model’s predictive power, and keep the overall pipeline unchanged, moving the ROC‑AUC score closer to the target.'
- What this solution (achieved 0.6452) has done: 'The fix corrects the image‑path handling so that feature extraction actually loads the images instead of returning all‑NaN vectors, which caused the feature matrix to have zero columns and broke scaling and model training. The `extract_features` function now builds filenames directly from the `image_id` column without adding an extra prefix, and the calls are updated accordingly. This restores a valid 6‑dimensional feature set, allowing the scaler and logistic regression models to run and produce meaningful predictions, moving the ROC‑AUC score toward the target.'
- What this solution (achieved 0.57023) has done: 'I enhance the feature extraction to include the full down‑sampled pixel values (32×32×3) together with the per‑channel mean and std, giving the logistic‑regression models much richer information while keeping the same model type. I also raise the maximum number of iterations to ensure convergence. These minimal changes keep the original pipeline intact but are expected to raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.66617) has done: 'I replace the simple LogisticRegression models with a modest RandomForestClassifier for each disease label, which can capture non‑linear patterns in the pixel‑level features and is allowed because the score gap exceeds 30 % of the target. The rest of the pipeline (feature extraction, imputation, scaling, and CSV handling) remains unchanged, ensuring a valid `submission.csv` while moving the ROC‑AUC closer to the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier  # new model

base_path = "../input/plant-pathology-2020-fgvc7"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
sample_sub_path = os.path.join(base_path, "sample_submission.csv")
images_dir = os.path.join(base_path, "images")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)
submission = pd.read_csv(sample_sub_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]
col_means = {col: train_df[col].mean() for col in target_cols}


def extract_features(image_id):
    """Return per‑channel mean, std, and flattened pixel values for an image."""
    filename = f"{image_id}.jpg"  # image_id already includes the proper prefix
    path = os.path.join(images_dir, filename)
    try:
        img = Image.open(path).convert("RGB")
        img = img.resize((32, 32))  # cheap down‑sampling
        arr = np.asarray(img) / 255.0  # shape (32, 32, 3), values in [0,1]
        means = arr.mean(axis=(0, 1))  # (3,)
        stds = arr.std(axis=(0, 1))  # (3,)
        flat = arr.reshape(-1)  # (32*32*3,) = 3072
        return np.concatenate([means, stds, flat])  # (3078,)
    except Exception:
        return np.full(6 + 32 * 32 * 3, np.nan)


train_features = np.vstack(
    [extract_features(img_id) for img_id in train_df["image_id"]]
)
test_features = np.vstack([extract_features(img_id) for img_id in test_df["image_id"]])

imputer = SimpleImputer(strategy="mean")
train_features = imputer.fit_transform(train_features)
test_features = imputer.transform(test_features)

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features)
test_features = scaler.transform(test_features)




## === cell 1
models = {}
for col in target_cols:
    y = train_df[col].values
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        class_weight="balanced",
    )
    clf.fit(train_features, y)
    models[col] = clf




## === cell 2
for col in target_cols:
    if col in models:
        probs = models[col].predict_proba(test_features)[:, 1]  # probability of class 1
        probs = np.clip(probs, 0.0, 1.0)
        submission[col] = probs
    else:
        submission[col] = col_means[col]




## === cell 3
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission written to {output_path}")
