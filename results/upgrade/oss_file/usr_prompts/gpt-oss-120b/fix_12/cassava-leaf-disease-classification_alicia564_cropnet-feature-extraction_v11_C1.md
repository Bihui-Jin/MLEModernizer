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
from PIL import Image
from concurrent.futures import ThreadPoolExecutor
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUBMISSION = os.path.join(DATA_ROOT, "sample_submission.csv")

label_to_disease = pd.read_json(
    os.path.join(DATA_ROOT, "label_num_to_disease_map.json"), typ="series"
)

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_df["label"] = train_df["label"].astype(str)

train_meta, valid_meta = train_test_split(
    train_df, test_size=0.2, stratify=train_df["label"], random_state=42
)




## === cell 1
IMG_SIZE = (64, 64)  # keep small for speed; PCA will handle dimensionality


def load_image_array(img_path):
    """Load, convert to RGB, resize, and return a uint8 numpy array (H,W,3)."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(IMG_SIZE, Image.BILINEAR)
        return np.array(img, dtype=np.uint8)


def build_dataset(meta_df, augment=False):
    """
    Return X (n_samples, features) and y (int labels) for a metadata frame.
    If augment=True, adds horizontally‑flipped, vertically‑flipped,
    90°‑rotated and 180°‑rotated copies (training only) without re‑reading the image files.
    """
    paths = meta_df["path"].tolist()
    labels = meta_df["label"].astype(int).to_numpy()
    n = len(paths)
    feat_dim = IMG_SIZE[0] * IMG_SIZE[1] * 3

    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_image_array, paths))
    images = np.stack(images, axis=0)  # shape (n, H, W, 3)

    if augment:
        X = np.empty((n * 5, feat_dim), dtype=np.uint8)
        y = np.empty(n * 5, dtype=int)

        X[0:n] = images.reshape(n, -1)
        y[0:n] = labels

        X[n : 2 * n] = np.fliplr(images).reshape(n, -1)
        X[2 * n : 3 * n] = np.flipud(images).reshape(n, -1)
        X[3 * n : 4 * n] = np.rot90(images, k=1, axes=(1, 2)).reshape(n, -1)
        X[4 * n : 5 * n] = np.rot90(images, k=2, axes=(1, 2)).reshape(n, -1)

        y[n:] = np.repeat(labels, 4)
    else:
        X = images.reshape(n, -1)
        y = labels

    return X, y


print("Building training set (with augmentation)...")
X_train_raw, y_train = build_dataset(train_meta, augment=True)
print(f"Training set shape (raw): {X_train_raw.shape}")

print("Building validation set...")
X_valid_raw, y_valid = build_dataset(valid_meta, augment=False)
print(f"Validation set shape (raw): {X_valid_raw.shape}")




## === cell 2
pca = PCA(n_components=300, random_state=42, svd_solver="randomized")
X_train = pca.fit_transform(X_train_raw)
X_valid = pca.transform(X_valid_raw)

rf = RandomForestClassifier(
    n_estimators=1200,  # increased from 800
    max_depth=None,
    max_features="sqrt",
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
)
rf.fit(X_train, y_train)

valid_pred = rf.predict(X_valid)
val_acc = accuracy_score(y_valid, valid_pred)
print(f"Validation accuracy: {val_acc:.4f}")




## === cell 3
test_filenames = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)


def load_test_batch(file_list):
    """Parallel load of a batch of test images and apply the same PCA transform."""
    paths = [os.path.join(TEST_IMG_DIR, fname) for fname in file_list]
    max_workers = min(8, os.cpu_count() or 1)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        arrays = list(executor.map(load_image_array, paths))
    batch_flat = np.stack([arr.ravel() for arr in arrays], axis=0)
    return pca.transform(batch_flat)


batch_size = 256
predictions = []
image_ids = []

for i in range(0, len(test_filenames), batch_size):
    batch_files = test_filenames[i : i + batch_size]
    batch_X = load_test_batch(batch_files)
    batch_pred = rf.predict(batch_X)
    predictions.extend(batch_pred.astype(str))
    image_ids.extend(batch_files)

submission = pd.DataFrame({"image_id": image_ids, "label": predictions})

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission.head())
