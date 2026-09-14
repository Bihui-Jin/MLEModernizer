# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import concurrent.futures
import random

np.random.seed(42)
random.seed(42)

GLOBAL_SIZE = (224, 224)
GLOBAL_HIST_BINS = 32



## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_images_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"])


def _extract_features(img_path):
    """
    Load an image, resize, and compute an enriched feature vector:
      - RGB mean, std, min, max, skewness (15 values)
      - HSV mean, std, min, max, skewness (15 values)
      - RGB histogram (GLOBAL_HIST_BINS * 3)
      - HSV histogram (GLOBAL_HIST_BINS * 3)
      - Grayscale histogram (GLOBAL_HIST_BINS)
    Returns a float32 vector.
    """
    img = Image.open(img_path).convert("RGB").resize(GLOBAL_SIZE)
    rgb_arr = np.asarray(img, dtype=np.float32) / 255.0  # [0,1]

    rgb_means = rgb_arr.mean(axis=(0, 1))
    rgb_stds = rgb_arr.std(axis=(0, 1))
    rgb_mins = rgb_arr.min(axis=(0, 1))
    rgb_maxs = rgb_arr.max(axis=(0, 1))

    rgb_skew = np.empty(3, dtype=np.float32)
    for c in range(3):
        if rgb_stds[c] > 0:
            rgb_skew[c] = ((rgb_arr[:, :, c] - rgb_means[c]) ** 3).mean() / (
                rgb_stds[c] ** 3
            )
        else:
            rgb_skew[c] = 0.0

    rgb_hist = []
    for c in range(3):
        hist, _ = np.histogram(
            rgb_arr[:, :, c], bins=GLOBAL_HIST_BINS, range=(0.0, 1.0), density=True
        )
        rgb_hist.append(hist)
    rgb_hist = np.concatenate(rgb_hist)

    hsv_arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
    hsv_means = hsv_arr.mean(axis=(0, 1))
    hsv_stds = hsv_arr.std(axis=(0, 1))
    hsv_mins = hsv_arr.min(axis=(0, 1))
    hsv_maxs = hsv_arr.max(axis=(0, 1))

    hsv_skew = np.empty(3, dtype=np.float32)
    for c in range(3):
        if hsv_stds[c] > 0:
            hsv_skew[c] = ((hsv_arr[:, :, c] - hsv_means[c]) ** 3).mean() / (
                hsv_stds[c] ** 3
            )
        else:
            hsv_skew[c] = 0.0

    hsv_hist = []
    for c in range(3):
        hist, _ = np.histogram(
            hsv_arr[:, :, c], bins=GLOBAL_HIST_BINS, range=(0.0, 1.0), density=True
        )
        hsv_hist.append(hist)
    hsv_hist = np.concatenate(hsv_hist)

    gray_arr = np.asarray(img.convert("L"), dtype=np.float32) / 255.0
    gray_hist, _ = np.histogram(
        gray_arr, bins=GLOBAL_HIST_BINS, range=(0.0, 1.0), density=True
    )

    feature_vec = np.concatenate(
        [
            rgb_means,
            rgb_stds,
            rgb_mins,
            rgb_maxs,
            rgb_skew,
            hsv_means,
            hsv_stds,
            hsv_mins,
            hsv_maxs,
            hsv_skew,
            rgb_hist,
            hsv_hist,
            gray_hist,
        ]
    ).astype(np.float32)
    return feature_vec


def load_features(df, img_dir):
    """
    Parallel loading of images and extraction of richer features.
    Uses a ProcessPoolExecutor to achieve true CPU parallelism.
    Pre‑allocates the result array to avoid extra list‑to‑array copying.
    Returns a NumPy array of shape (len(df), feature_dim).
    """
    img_paths = [os.path.join(img_dir, img_id) for img_id in df["image_id"]]

    dummy_feat = _extract_features(img_paths[0])
    feature_dim = dummy_feat.shape[0]

    n = len(img_paths)
    feats = np.empty((n, feature_dim), dtype=np.float32)

    max_workers = min(os.cpu_count() or 1, 16)

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, feat in enumerate(
            executor.map(_extract_features, img_paths, chunksize=64)
        ):
            feats[idx] = feat

    return feats


X = load_features(train_df, train_images_dir)
y = train_df["label_enc"].values

X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=1200,
    max_features=None,
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
et.fit(X_train, y_train)

valid_pred = et.predict(X_valid)
val_acc = accuracy_score(y_valid, valid_pred)
print(f"Validation accuracy: {val_acc:.4f}")

test_filenames = sorted(
    [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
)

test_df = pd.DataFrame({"image_id": test_filenames})
X_test = load_features(test_df, test_images_dir)

test_pred_enc = et.predict(X_test)
test_pred_labels = le.inverse_transform(test_pred_enc)

submission = pd.DataFrame(
    {"image_id": test_filenames, "label": test_pred_labels.astype(int)}
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission.head())
