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

2.7

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
from __future__ import print_function
import os
import numpy as np
import pandas as pd
from PIL import Image

BASE_PATH = "../input/cassava-leaf-disease-classification/"
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)
train_df["image_path"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
val_split = int(0.9 * len(train_df))
train_data = train_df.iloc[:val_split]
val_data = train_df.iloc[val_split:]




## === cell 1
class_means = {}
class_counts = {}

for _, row in train_df.iterrows():
    try:
        img = Image.open(row["image_path"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_colour = arr.mean(axis=(0, 1))
    except Exception:
        continue

    lbl = row["label"]
    if lbl not in class_means:
        class_means[lbl] = mean_colour
        class_counts[lbl] = 1
    else:
        class_means[lbl] = class_means[lbl] + (mean_colour - class_means[lbl]) / (
            class_counts[lbl] + 1
        )
        class_counts[lbl] += 1

all_labels = range(5)
overall_mean = np.mean(list(class_means.values()), axis=0)
for lbl in all_labels:
    if lbl not in class_means:
        class_means[lbl] = overall_mean
        class_counts[lbl] = 0

epsilon = 1e-5  # for numerical stability
class_cov = {}
class_inv_cov = {}

for lbl in all_labels:
    class_cov[lbl] = np.zeros((3, 3))

for _, row in train_df.iterrows():
    try:
        img = Image.open(row["image_path"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_colour = arr.mean(axis=(0, 1))
    except Exception:
        continue

    lbl = row["label"]
    diff = mean_colour - class_means[lbl]
    class_cov[lbl] += np.outer(diff, diff)

for lbl in all_labels:
    n = max(class_counts[lbl] - 1, 1)  # avoid division by zero
    cov = class_cov[lbl] / n
    cov += epsilon * np.eye(3)  # regularisation
    class_cov[lbl] = cov
    class_inv_cov[lbl] = np.linalg.inv(cov)




## === cell 2
test_files = os.listdir(TEST_IMG_DIR)
test_df = pd.DataFrame({"image_id": test_files})
test_df["image_path"] = test_df["image_id"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x)
)




## === cell 3
centroid_matrix = np.stack([class_means[lbl] for lbl in all_labels])  # (5,3)


def predict_label(mean_colour):
    """Mahalanobis distance to each class centroid; fallback to Euclidean."""
    best_lbl = None
    best_dist = None
    for lbl in all_labels:
        diff = mean_colour - class_means[lbl]
        try:
            dist = np.dot(np.dot(diff, class_inv_cov[lbl]), diff)
        except Exception:
            dist = np.linalg.norm(diff)
        if best_dist is None or dist < best_dist:
            best_dist = dist
            best_lbl = lbl
    return int(best_lbl)


val_correct = 0
val_total = 0
for _, row in val_data.iterrows():
    try:
        img = Image.open(row["image_path"]).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_colour = arr.mean(axis=(0, 1))
        pred = predict_label(mean_colour)
        if pred == row["label"]:
            val_correct += 1
        val_total += 1
    except Exception:
        continue
if val_total > 0:
    print(
        "Validation accuracy (approx.): {:.4f}".format(val_correct / float(val_total))
    )

pred_labels = []
most_common_label = train_df["label"].mode()[0]

for img_path in test_df["image_path"]:
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        mean_colour = arr.mean(axis=(0, 1))
        pred = predict_label(mean_colour)
    except Exception:
        pred = most_common_label  # fallback for unreadable files
    pred_labels.append(pred)




## === cell 4
submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission written to {}".format(submission_path))
