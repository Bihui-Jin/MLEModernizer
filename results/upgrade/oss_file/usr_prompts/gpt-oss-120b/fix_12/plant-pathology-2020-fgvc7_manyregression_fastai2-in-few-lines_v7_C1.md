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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA




## === cell 1
BASE_PATH = Path("/kaggle/input/plant-pathology-2020-fgvc7")
TRAIN_CSV = BASE_PATH / "train.csv"
TEST_CSV = BASE_PATH / "test.csv"
SAMPLE_SUB = BASE_PATH / "sample_submission.csv"
IMG_DIR = BASE_PATH / "images"




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)
sample_sub = pd.read_csv(SAMPLE_SUB)

LABEL_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
y = train_df[LABEL_COLS].values.astype(np.float32)




## === cell 3
from concurrent.futures import ThreadPoolExecutor
import os


def extract_features(image_path, size=(64, 64)):
    """
    Load an image, resize, and return a feature vector composed of:
      - mean RGB (3,)
      - std RGB   (3,)
      - per‑channel colour histograms (16 bins each → 48,)
      - edge‑strength statistic (1,)
      - flattened grayscale (size[0] * size[1],)
    """
    with Image.open(image_path) as img:
        img = img.convert("RGB")
        img = img.resize(size)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # (h, w, 3)

        mean_rgb = arr.mean(axis=(0, 1))
        std_rgb = arr.std(axis=(0, 1))

        hist_bins = 16
        hist_features = np.concatenate(
            [
                np.histogram(
                    arr[:, :, ch], bins=hist_bins, range=(0.0, 1.0), density=True
                )[0]
                for ch in range(3)
            ]
        )

        gray = np.dot(arr, [0.2989, 0.5870, 0.1140])
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_mean = np.array([grad_mag.mean()], dtype=np.float32)

        gray_flat = gray.ravel()
        return np.concatenate([mean_rgb, std_rgb, hist_features, edge_mean, gray_flat])


def _extract_path(img_path):
    """Helper for ThreadPoolExecutor – returns feature vector for a single image."""
    return extract_features(img_path)


def build_feature_matrix(df):
    """Return a (n_samples, feature_dim) array of image features using parallel threads."""
    img_paths = [IMG_DIR / f"{img_id}.jpg" for img_id in df["image_id"]]
    sample_feat = extract_features(img_paths[0])
    dim = sample_feat.shape[0]
    n = len(img_paths)
    feats = np.empty((n, dim), dtype=np.float32)

    max_workers = os.cpu_count() or 1

    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for i, f in enumerate(executor.map(_extract_path, img_paths)):
            feats[i] = f
    return feats




## === cell 4
X_train = build_feature_matrix(train_df)
X_test = build_feature_matrix(test_df)




## === cell 5
X_tr, X_val, y_tr, y_val = train_test_split(
    X_train, y, test_size=0.2, random_state=42, stratify=y.argmax(axis=1)
)

scaler = StandardScaler()
X_tr_scaled = scaler.fit_transform(X_tr)
X_val_scaled = scaler.transform(X_val)

pca = PCA(n_components=0.99, random_state=42)
X_tr_pca = pca.fit_transform(X_tr_scaled)
X_val_pca = pca.transform(X_val_scaled)

base_clf = GradientBoostingClassifier(
    n_estimators=500, learning_rate=0.05, max_depth=4, random_state=42
)
clf = OneVsRestClassifier(base_clf)
clf.fit(X_tr_pca, y_tr)

val_preds = clf.predict_proba(X_val_pca)
val_auc = np.mean(
    [roc_auc_score(y_val[:, i], val_preds[:, i]) for i in range(len(LABEL_COLS))]
)
print(f"Validation mean ROC‑AUC: {val_auc:.4f}")




## === cell 6
X_train_scaled = scaler.fit_transform(X_train)
X_train_pca = pca.fit_transform(X_train_scaled)

clf.fit(X_train_pca, y)




## === cell 7
X_test_scaled = scaler.transform(X_test)
X_test_pca = pca.transform(X_test_scaled)

test_preds = clf.predict_proba(X_test_pca)  # shape (n_test, 4)




## === cell 8
submission = pd.DataFrame(test_preds, columns=LABEL_COLS)
submission.insert(0, "image_id", test_df["image_id"])
submission.to_csv("submission.csv", index=False, float_format="%.6f")
print("Submission file written to submission.csv")
