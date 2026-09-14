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
import numpy as np
from PIL import Image
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from concurrent.futures import ThreadPoolExecutor




## === cell 1
BASE_DIR = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_CSV = os.path.join(BASE_DIR, "test.csv")
IMAGES_DIR = os.path.join(BASE_DIR, "images")
IMAGE_SIZE = (128, 128)  # increased resolution for richer visual features




## === cell 2
def load_image_array(image_id, size=IMAGE_SIZE):
    """
    Load an image, resize to `size`, and return a feature vector.
    The vector consists of:
      * flattened RGB pixels
      * per‑channel mean and standard deviation (6 values)
      * HSV colour histogram (16 bins per channel → 48 values)
    Returned as float32 for downstream efficiency.
    """
    filename = image_id if image_id.lower().endswith(".jpg") else f"{image_id}.jpg"
    path = os.path.join(IMAGES_DIR, filename)
    with Image.open(path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32)  # shape (H, W, 3)

        flat = arr.flatten()

        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))

        hsv_img = img.convert("HSV")
        hsv_arr = np.asarray(hsv_img, dtype=np.uint8)  # shape (H, W, 3)
        hist_features = []
        for ch in range(3):
            hist, _ = np.histogram(
                hsv_arr[:, :, ch], bins=16, range=(0, 256), density=True
            )
            hist_features.append(hist.astype(np.float32))
        hist_feat = np.concatenate(hist_features)  # length 48

        return np.concatenate([flat, means, stds, hist_feat]).astype(np.float32)


def load_images_parallel(ids):
    """Load many images concurrently using a thread pool and pre‑allocate the result array."""
    first_feat = load_image_array(ids[0])
    n_samples = len(ids)
    n_features = first_feat.shape[0]
    result = np.empty((n_samples, n_features), dtype=np.float32)
    result[0] = first_feat

    def worker(idx, img_id):
        return idx, load_image_array(img_id)

    with ThreadPoolExecutor(max_workers=8) as executor:
        futures = [
            executor.submit(worker, i, img_id) for i, img_id in enumerate(ids[1:])
        ]
        for fut in futures:
            idx, feat = fut.result()
            result[idx] = feat
    return result




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

print("Loading training images...")
X_all = load_images_parallel(train_df["image_id"].values)
y_all = train_df[["healthy", "multiple_diseases", "rust", "scab"]].values.astype(
    np.int8
)

print("Loading test images...")
X_test = load_images_parallel(test_df["image_id"].values)

X_train, X_val, y_train, y_val = train_test_split(
    X_all, y_all, test_size=0.2, random_state=42, shuffle=True
)

rf = RandomForestClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
rf_model = MultiOutputClassifier(rf, n_jobs=5)

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

et = ExtraTreesClassifier(
    n_estimators=1500,
    n_jobs=5,
    class_weight="balanced",
    random_state=42,
)
et_model = MultiOutputClassifier(et, n_jobs=5)

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

print("Training RandomForest on split...")
rf_model.fit(X_train, y_train)
print("Training Logistic Regression on split...")
logit_model.fit(X_train, y_train)
print("Training ExtraTrees on split...")
et_model.fit(X_train, y_train)
print("Training PCA‑Logistic Regression on split...")
pca_logit_model.fit(X_train, y_train)


def get_val_probas(model, X):
    """Return column‑wise prediction probabilities for validation data."""
    return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])


rf_val = get_val_probas(rf_model, X_val)
logit_val = get_val_probas(logit_model, X_val)
et_val = get_val_probas(et_model, X_val)
pca_logit_val = get_val_probas(pca_logit_model, X_val)


def mean_auc(y_true, y_scores):
    return np.mean(
        [roc_auc_score(y_true[:, j], y_scores[:, j]) for j in range(y_true.shape[1])]
    )


auc_rf = mean_auc(y_val, rf_val)
auc_logit = mean_auc(y_val, logit_val)
auc_et = mean_auc(y_val, et_val)
auc_pca = mean_auc(y_val, pca_logit_val)

print(
    f"Validation AUCs -> RF: {auc_rf:.4f}, Logit: {auc_logit:.4f}, ET: {auc_et:.4f}, PCA‑Logit: {auc_pca:.4f}"
)

raw_weights = np.array([auc_rf, auc_logit, auc_et, auc_pca])
weights = raw_weights / raw_weights.sum()
print(
    f"Ensemble weights: RF {weights[0]:.3f}, Logit {weights[1]:.3f}, ET {weights[2]:.3f}, PCA‑Logit {weights[3]:.3f}"
)

print("Retraining models on the full training data...")
rf_model.fit(X_all, y_all)
logit_model.fit(X_all, y_all)
et_model.fit(X_all, y_all)
pca_logit_model.fit(X_all, y_all)


def get_test_probas(model, X):
    """Return column‑wise prediction probabilities for test data."""
    return np.column_stack([est.predict_proba(X)[:, 1] for est in model.estimators_])


rf_test = get_test_probas(rf_model, X_test)
logit_test = get_test_probas(logit_model, X_test)
et_test = get_test_probas(et_model, X_test)
pca_logit_test = get_test_probas(pca_logit_model, X_test)

submission_avg = (
    weights[0] * rf_test
    + weights[1] * logit_test
    + weights[2] * et_test
    + weights[3] * pca_logit_test
)




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
