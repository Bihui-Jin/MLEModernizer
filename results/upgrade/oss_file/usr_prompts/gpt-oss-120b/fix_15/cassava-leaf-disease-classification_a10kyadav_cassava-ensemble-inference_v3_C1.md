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

3.9

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

os.environ["OMP_NUM_THREADS"] = (
    "1"  # ensure sklearn's internal threads don’t oversubscribe
)
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import concurrent.futures

np.random.seed(42)

INPUT_ROOT = Path("/kaggle/input/cassava-leaf-disease-classification")
TRAIN_CSV = INPUT_ROOT / "train.csv"
SAMPLE_SUBMISSION = INPUT_ROOT / "sample_submission.csv"
TRAIN_IMG_ROOT = INPUT_ROOT / "train_images"
TEST_IMG_ROOT = INPUT_ROOT / "test_images"
OUTPUT_SUBMISSION = Path("submission.csv")

MAX_TRAIN_SAMPLES = None  # keep full dataset

train_df = pd.read_csv(TRAIN_CSV)


def load_and_preprocess(img_path, size=(64, 64)):
    """Load an image, resize to `size`, normalize, and return a flattened float32 array."""
    with Image.open(img_path) as img:
        img = img.convert("RGB")
        img = img.resize(size, Image.BILINEAR)
        arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]
    return arr.ravel()  # flatten to 1‑D vector


if MAX_TRAIN_SAMPLES is not None and len(train_df) > MAX_TRAIN_SAMPLES:
    train_df = (
        train_df.groupby("label", group_keys=False)
        .apply(
            lambda grp: grp.sample(
                n=int(np.round(MAX_TRAIN_SAMPLES * len(grp) / len(train_df))),
                random_state=42,
            )
        )
        .reset_index(drop=True)
    )

img_paths = []
labels = []
for _, row in train_df.iterrows():
    img_file = TRAIN_IMG_ROOT / row["image_id"]
    if img_file.is_file():
        img_paths.append(str(img_file))
        labels.append(row["label"])

print(f"Loading and preprocessing training images (using {len(img_paths)} samples)...")

max_workers = min(32, (os.cpu_count() or 1))
n_samples = len(img_paths)
feature_dim = 64 * 64 * 3  # 12288

train_features = np.empty((n_samples, feature_dim), dtype=np.float32)


def _load_fill(idx_path):
    idx, path = idx_path
    train_features[idx] = load_and_preprocess(path)


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
    list(exe.map(_load_fill, enumerate(img_paths)))

X = train_features  # already float32, shape (n_samples, 12288)
y = np.array(labels, dtype=np.int64)




## === cell 1
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

train_mean = X_train.mean(axis=0, keepdims=True).astype(np.float32)
train_std = X_train.std(axis=0, keepdims=True).astype(np.float32) + 1e-8

X_train_scaled = (X_train - train_mean) / train_std
X_val_scaled = (X_val - train_mean) / train_std

print("Training Logistic Regression model with manual scaling and class balancing...")
clf = LogisticRegression(
    multi_class="multinomial",
    solver="lbfgs",
    max_iter=1000,
    n_jobs=-1,
    class_weight="balanced",
    C=10.0,
    random_state=42,
)

clf.fit(X_train_scaled, y_train)

val_pred = clf.predict(X_val_scaled)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy: {val_acc:.4f}")

print("Retraining on the full dataset for final predictions...")
full_mean = X.mean(axis=0, keepdims=True).astype(np.float32)
full_std = X.std(axis=0, keepdims=True).astype(np.float32) + 1e-8
X_full_scaled = (X - full_mean) / full_std

clf.fit(X_full_scaled, y)




## === cell 2
sample_sub = pd.read_csv(SAMPLE_SUBMISSION)

test_img_paths = []
test_ids = []
for img_id in sample_sub["image_id"]:
    img_path = TEST_IMG_ROOT / img_id
    if img_path.is_file():
        test_img_paths.append(str(img_path))
        test_ids.append(img_id)
    else:
        test_img_paths.append(None)
        test_ids.append(img_id)

print("Loading and preprocessing test images...")


def _load_test(p):
    return load_and_preprocess(p) if p is not None else None


with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as exe:
    test_features = list(exe.map(_load_test, test_img_paths))

most_common_label = int(np.argmax(np.bincount(y)))

pred_labels = [None] * len(test_features)

valid_idx = np.nonzero([f is not None for f in test_features])[0]
if valid_idx.size > 0:
    valid_feats = np.stack([test_features[i] for i in valid_idx]).astype(np.float32)
    valid_feats_scaled = (valid_feats - full_mean) / full_std
    batch_preds = clf.predict(valid_feats_scaled)
    for idx, pred in zip(valid_idx, batch_preds):
        pred_labels[idx] = int(pred)

for i, lbl in enumerate(pred_labels):
    if lbl is None:
        pred_labels[i] = most_common_label

submission_df = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
submission_df.to_csv(OUTPUT_SUBMISSION, index=False)
print(f"Submission saved to {OUTPUT_SUBMISSION.resolve()}")
