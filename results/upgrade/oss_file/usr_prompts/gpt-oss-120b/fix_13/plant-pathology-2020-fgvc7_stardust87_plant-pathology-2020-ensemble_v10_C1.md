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
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.multiclass import OneVsRestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
import concurrent.futures


def _kurtosis(arr, mean, std):
    """Calculate excess kurtosis; returns 0 if variance is zero."""
    if std == 0:
        return 0.0
    return ((arr - mean) ** 4).mean() / (std**4) - 3.0




## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TEST_CSV = os.path.join(BASE_PATH, "test.csv")
IMAGES_DIR = os.path.join(BASE_PATH, "images")
SUBMISSION_PATH = "submission.csv"

LABEL_COLS = ["healthy", "multiple_diseases", "rust", "scab"]
IMG_SIZE = 128  # keep higher resolution for richer features
HIST_BINS = 32  # richer colour histograms (increase from 16)




## === cell 2
def _skewness(arr, mean, std):
    if std == 0:
        return 0.0
    return ((arr - mean) ** 3).mean() / (std**3)


def _process_image(img_id, images_dir):
    """
    Compute the feature vector for a single image. Returns a 1‑D numpy array.
    """
    img_path = os.path.join(images_dir, f"{img_id}.jpg")
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize((IMG_SIZE, IMG_SIZE))
        rgb_arr = np.asarray(img, dtype=np.float32) / 255.0

        rgb_mean = rgb_arr.mean(axis=(0, 1))
        rgb_std = rgb_arr.std(axis=(0, 1))
        rgb_skew = np.array(
            [_skewness(rgb_arr[:, :, c], rgb_mean[c], rgb_std[c]) for c in range(3)]
        )
        rgb_kurt = np.array(
            [_kurtosis(rgb_arr[:, :, c], rgb_mean[c], rgb_std[c]) for c in range(3)]
        )

        hsv_arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        hsv_mean = hsv_arr.mean(axis=(0, 1))
        hsv_std = hsv_arr.std(axis=(0, 1))
        hsv_skew = np.array(
            [_skewness(hsv_arr[:, :, c], hsv_mean[c], hsv_std[c]) for c in range(3)]
        )
        hsv_kurt = np.array(
            [_kurtosis(hsv_arr[:, :, c], hsv_mean[c], hsv_std[c]) for c in range(3)]
        )

        gray_arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
        gray_mean = gray_arr.mean()
        gray_std = gray_arr.std()
        gray_skew = _skewness(gray_arr, gray_mean, gray_std)
        gray_kurt = _kurtosis(gray_arr, gray_mean, gray_std)

        gy, gx = np.gradient(gray_arr)
        grad_mag = np.sqrt(gx**2 + gy**2)
        edge_density = grad_mag.mean()

        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(rgb_arr[:, :, c], bins=HIST_BINS, range=(0.0, 1.0))
            hist = hist.astype(np.float32)
            if hist.sum() > 0:
                hist = hist / hist.sum()
            hist_features.extend(hist)

        lap_x2 = np.gradient(np.gradient(gray_arr, axis=0), axis=0)
        lap_y2 = np.gradient(np.gradient(gray_arr, axis=1), axis=1)
        laplacian = lap_x2 + lap_y2
        laplacian_var = laplacian.var()

        feat = np.concatenate(
            [
                rgb_mean,
                rgb_std,
                rgb_skew,
                rgb_kurt,
                hsv_mean,
                hsv_std,
                hsv_skew,
                hsv_kurt,
                [gray_mean, gray_std, gray_skew, gray_kurt],
                [edge_density],
                hist_features,
                [laplacian_var],
            ]
        )
        return feat.astype(np.float32)
    except Exception:
        dim = (
            (3 + 3 + 3 + 3) + (3 + 3 + 3 + 3) + (1 + 1 + 1 + 1) + 1 + 3 * HIST_BINS + 1
        )
        return np.zeros(dim, dtype=np.float32)


def load_features(df, images_dir, label_cols=None):
    """
    Parallelized feature loader. Returns X (n_samples, n_features) and optionally y.
    """
    img_ids = df["image_id"].tolist()
    with concurrent.futures.ProcessPoolExecutor() as executor:
        X_list = list(
            executor.map(_process_image, img_ids, [images_dir] * len(img_ids))
        )
    X = np.stack(X_list)
    if label_cols is not None:
        y = df[label_cols].values.astype(np.float32)
        return X, y
    return X




## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
X_train, y_train = load_features(train_df, IMAGES_DIR, LABEL_COLS)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)

clf = OneVsRestClassifier(
    GradientBoostingClassifier(
        n_estimators=1800,  # more trees
        learning_rate=0.01,  # a bit lower LR for smoother boosting
        max_depth=7,  # deeper trees
        subsample=0.9,  # use more samples per iteration
        max_features=None,  # use all features
        random_state=42,
    )
)
clf.fit(X_train, y_train)

test_df = pd.read_csv(TEST_CSV)
X_test = load_features(test_df, IMAGES_DIR)
X_test = scaler.transform(X_test)

test_pred_matrix = clf.predict_proba(X_test)




## === cell 4
submission_df = pd.DataFrame(test_pred_matrix, columns=LABEL_COLS)
submission_df.insert(0, "image_id", test_df["image_id"])
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
