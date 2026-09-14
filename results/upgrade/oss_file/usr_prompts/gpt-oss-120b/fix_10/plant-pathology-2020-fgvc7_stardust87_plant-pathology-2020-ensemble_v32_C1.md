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

0.9646650712510072

# 6. Current score

0.61011

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61324) has done: 'The script now safely loads the available train and test data, builds a simple image‑based model (flattened 64×64 RGB pixels with a RandomForest wrapped in a MultiOutputClassifier), predicts probabilities for each disease, and writes a correctly‑formatted `submission.csv`. It also retains the original submission‑listing logic but falls back to the model when no external submissions are found, fixing the IndexError and ensuring a valid CSV is produced. This minimal change fixes the runtime error and provides a reasonable baseline that moves the score toward the target.'
- What this solution (achieved 0.56726) has done: 'I increase the image resolution to 128×128 for richer visual information and replace the RandomForest with a scaled Logistic Regression wrapped in a MultiOutputClassifier, which better matches the ROC‑AUC metric. These modest changes keep the overall pipeline unchanged while providing stronger, calibrated probability estimates, moving the validation score upward toward the target.'
- What this solution (achieved 0.67561) has done: 'I replace the linear model with a stronger non‑linear one (a balanced RandomForest) wrapped in a MultiOutputClassifier. RandomForest can capture image‑pixel interactions better than a simple logistic regression on flattened pixels, which should raise the ROC‑AUC toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.62874) has done: 'I keep the existing image loading and preprocessing but add a calibrated Logistic Regression model alongside the RandomForest. Both models predict probabilities for each disease, and I average their outputs before writing the submission. This small ensemble usually raises ROC‑AUC without changing the overall pipeline structure.'
- What this solution (achieved 0.63534) has done: 'I add two additional, still lightweight models – an ExtraTrees classifier and a PCA‑reduced logistic regression – and average their predicted probabilities together with the existing RandomForest and logistic regression. These models keep the overall pipeline unchanged (still flattening images and using MultiOutputClassifier) but give the ensemble a richer, more diverse set of learners, which should raise the ROC‑AUC toward the target without altering the core logic.'
- What this solution (achieved 0.59992) has done: 'I enhance the image feature extraction by appending per‑channel mean and standard deviation to the flattened pixel vector, and I slightly boost model capacity (more trees and more PCA components). These changes keep the overall pipeline unchanged while providing richer information that should raise the ROC‑AUC toward the target score.'
- What this solution (achieved 0.61011) has done: 'I parallelize image loading with a thread pool and reduce the resize resolution from 224×224 to 64×64, which drastically cuts the feature dimensionality while preserving the same preprocessing logic (flattened RGB pixels plus channel statistics). This keeps the exact model types and hyper‑parameters unchanged, but makes feature extraction and all subsequent model training fit comfortably within the 600‑second limit. The rest of the pipeline (model definitions, training, ensembling, and submission) remains identical.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from concurrent.futures import ThreadPoolExecutor



## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
IMAGE_SIZE = (64, 64)




## === cell 2
def load_image_array(image_id, size=IMAGE_SIZE):
    """
    Load an image, resize to `size`, and return a feature vector.
    The vector consists of the flattened RGB pixels together with
    per‑channel mean and standard deviation (6 extra features).
    Returned as float32 for downstream efficiency.
    """
    filename = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32)
        flat = arr.flatten()
        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        return np.concatenate([flat, means, stds]).astype(np.float32)


def load_images_parallel(ids):
    """Load many images concurrently using a thread pool."""
    with ThreadPoolExecutor(max_workers=8) as executor:
        arrays = list(executor.map(load_image_array, ids))
    return np.stack(arrays).astype(np.float32)




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Loading training images...")
X_train = load_images_parallel(train_df["image_id"].values)
y_train = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
    np.int8
)

print("Loading test images...")
X_test = load_images_parallel(test_df["image_id"].values)

rf = RandomForestClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
rf_model = MultiOutputClassifier(rf, n_jobs=5)

print("Training RandomForest...")
rf_model.fit(X_train, y_train)

logit_pipe = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=5,
        solver="lbfgs",
        C=5.0,
    ),
)
logit_model = MultiOutputClassifier(logit_pipe, n_jobs=5)

print("Training Logistic Regression...")
logit_model.fit(X_train, y_train)

et = ExtraTreesClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
et_model = MultiOutputClassifier(et, n_jobs=5)

print("Training ExtraTrees...")
et_model.fit(X_train, y_train)

pca_logit_pipe = make_pipeline(
    StandardScaler(),
    PCA(n_components=500, random_state=42),
    LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        n_jobs=5,
        solver="lbfgs",
        C=5.0,
    ),
)
pca_logit_model = MultiOutputClassifier(pca_logit_pipe, n_jobs=5)

print("Training PCA‑Logistic Regression...")
pca_logit_model.fit(X_train, y_train)

rf_probas = []
logit_probas = []
et_probas = []
pca_logit_probas = []

for i in range(y_train.shape[1]):
    rf_probas.append(rf_model.estimators_[i].predict_proba(X_test)[:, 1])
    logit_probas.append(logit_model.estimators_[i].predict_proba(X_test)[:, 1])
    et_probas.append(et_model.estimators_[i].predict_proba(X_test)[:, 1])
    pca_logit_probas.append(pca_logit_model.estimators_[i].predict_proba(X_test)[:, 1])

rf_probas = np.column_stack(rf_probas)
logit_probas = np.column_stack(logit_probas)
et_probas = np.column_stack(et_probas)
pca_logit_probas = np.column_stack(pca_logit_probas)

submission_avg = (rf_probas + logit_probas + et_probas + pca_logit_probas) / 4.0




## === cell 4
def make_submission_file(preds, test_dataframe, output_path="submission.csv"):
    """
    Write predictions to a CSV with the required format:
    image_id,healthy,multiple_diseases,rust,scab
    """
    sub_df = test_dataframe.copy()
    sub_df[["healthy", "multiple_diseases", "rust", "scab"]] = preds
    sub_df.to_csv(output_path, index=False)
    print(f"Submission written to {output_path}")


make_submission_file(submission_avg, test_df)
