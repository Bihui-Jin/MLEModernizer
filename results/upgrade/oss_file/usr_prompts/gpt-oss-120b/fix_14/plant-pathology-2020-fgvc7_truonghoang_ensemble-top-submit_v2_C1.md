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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scipy==1.15.3
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
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
import multiprocessing as mp
import concurrent.futures

np.random.seed(42)



## === cell 1
base_path = "../input/plant-pathology-2020-fgvc7/"
train_path = os.path.join(base_path, "train.csv")
test_path = os.path.join(base_path, "test.csv")
images_dir = os.path.join(base_path, "images")

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

target_cols = ["healthy", "multiple_diseases", "rust", "scab"]


def extract_features(img_path):
    """
    Return a feature vector consisting of:
    - mean of each RGB channel (3)
    - std  of each RGB channel (3)
    - median of each RGB channel (3)
    - skewness of each RGB channel (3)  <-- now computed vectorised
    - 32‑bin normalized histogram for each RGB channel (96)
    Total length = 108.
    """
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.array(img, dtype=np.float32) / 255.0  # scale to [0,1]

        means = arr.mean(axis=(0, 1))
        stds = arr.std(axis=(0, 1))
        medians = np.median(arr, axis=(0, 1))

        centered = arr - means  # broadcast over H,W
        m3 = (centered**3).mean(axis=(0, 1))
        skews = np.where(stds != 0, m3 / (stds**3 + 1e-12), 0.0)

        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(
                arr[:, :, c], bins=32, range=(0.0, 1.0), density=True
            )
            hist_features.extend(hist)

        return np.concatenate([means, stds, medians, skews, hist_features])
    except Exception:
        return np.full(108, np.nan, dtype=np.float32)


def _worker(img_id):
    img_file = os.path.join(images_dir, f"{img_id}.jpg")
    return extract_features(img_file)


def train_label(col, X, y):
    """Fit GBC and LR for a single target column and compute ensemble weights."""
    gbc = GradientBoostingClassifier(
        n_estimators=800,
        learning_rate=0.03,
        max_depth=4,
        subsample=0.9,
        random_state=42,
    )
    gbc.fit(X, y)

    lr = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="liblinear",
    )
    lr.fit(X, y)

    probs_gbc = gbc.predict_proba(X)[:, 1]
    probs_lr = lr.predict_proba(X)[:, 1]
    auc_gbc = roc_auc_score(y, probs_gbc)
    auc_lr = roc_auc_score(y, probs_lr)

    total_auc = auc_gbc + auc_lr
    w_gbc = auc_gbc / total_auc if total_auc > 0 else 0.5
    w_lr = 1.0 - w_gbc
    return col, gbc, lr, {"gbc": w_gbc, "lr": w_lr}


if __name__ == "__main__":
    train_ids = train_df["image_id"].tolist()
    proc_cnt = max(1, mp.cpu_count() // 2)
    with mp.Pool(processes=proc_cnt) as pool:
        train_feat_list = pool.map(_worker, train_ids)
    X_train = np.vstack(train_feat_list).astype(np.float32)

    col_means = np.nanmean(X_train, axis=0)
    nan_mask = np.isnan(X_train)
    X_train[nan_mask] = np.take(col_means, np.where(nan_mask)[1])

    gbc_models = {}
    lr_models = {}
    model_weights = {}

    with concurrent.futures.ThreadPoolExecutor(
        max_workers=len(target_cols)
    ) as executor:
        futures = [
            executor.submit(train_label, col, X_train, train_df[col])
            for col in target_cols
        ]
        for fut in concurrent.futures.as_completed(futures):
            col_name, gbc, lr, weight_dict = fut.result()
            gbc_models[col_name] = gbc
            lr_models[col_name] = lr
            model_weights[col_name] = weight_dict

    test_ids = test_df["image_id"].tolist()
    with mp.Pool(processes=proc_cnt) as pool:
        test_feat_list = pool.map(_worker, test_ids)
    X_test = np.vstack(test_feat_list).astype(np.float32)

    nan_mask_test = np.isnan(X_test)
    X_test[nan_mask_test] = np.take(col_means, np.where(nan_mask_test)[1])



## === cell 2
submission = pd.DataFrame()
submission["image_id"] = test_df["image_id"]

for col in target_cols:
    probs_gbc = gbc_models[col].predict_proba(X_test)[:, 1]
    probs_lr = lr_models[col].predict_proba(X_test)[:, 1]
    w = model_weights[col]
    submission[col] = w["gbc"] * probs_gbc + w["lr"] * probs_lr

output_path = "submission.csv"
submission.to_csv(output_path, index=False)
