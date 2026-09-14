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

# 5. Target score

0.8841039588999697

# 6. Current score

0.54522

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script failed because it counted a non‑image entry (a nested folder) in the test directory, causing a row‑count mismatch, and it also used an f‑string which is invalid in Python 2.7. The fix filters for actual image files, validates the row count against the provided sample submission, and replaces the f‑string with a compatible `format` call. The corrected cells now run end‑to‑end and output a proper `submission.csv`.'
- What this solution (achieved 0.40732) has done: 'I keep the existing structure but add a lightweight heuristic that uses average image file size per class from the training set to predict test labels, which should improve accuracy beyond the simple most‑common‑label baseline while staying compatible with Python 2.7 and the current package set. The changes compute per‑label average file sizes, then assign each test image the label whose average size is closest to that image’s size, falling back to the most common label if needed, and all other logic (paths, CSV handling) remains unchanged.'
- What this solution (achieved 0.4219) has done: 'I replace the per‑class average size heuristic with a nearest‑neighbor lookup that assigns each test image the label of the training image whose file size is closest. This keeps the overall workflow unchanged, uses only the existing file‑size information, and is expected to raise accuracy from the current 0.40‑ish toward the target (while staying well below the ceiling). The most‑common label remains a fallback for any missing files.'
- What this solution (achieved 0.54522) has done: 'I replace the single‑nearest‑size lookup with a small k‑nearest‑neighbors vote (k=5). This keeps the same file‑size‑only approach but smooths noisy matches, which should raise accuracy toward the target while preserving the overall logic and compatibility with Python 2.7.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TRAIN_IMAGES_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMAGES_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUBMISSION_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)

most_common_label = train_df["label"].value_counts().idxmax()

train_sizes = []
train_labels = []
for _, row in train_df.iterrows():
    img_name = row["image_id"]
    label = row["label"]
    img_path = os.path.join(TRAIN_IMAGES_DIR, img_name)
    if os.path.isfile(img_path):
        size = os.path.getsize(img_path)
        train_sizes.append(size)
        train_labels.append(label)

train_sizes = np.array(train_sizes, dtype=np.int64)
train_labels = np.array(train_labels, dtype=np.int64)

if train_sizes.size == 0:
    train_sizes = np.array([0], dtype=np.int64)
    train_labels = np.array([most_common_label], dtype=np.int64)




## === cell 1
import collections

valid_extensions = (".jpg", ".jpeg", ".png")
test_filenames = [
    f
    for f in os.listdir(TEST_IMAGES_DIR)
    if f.lower().endswith(valid_extensions)
    and os.path.isfile(os.path.join(TEST_IMAGES_DIR, f))
]
test_filenames.sort()

expected_rows = pd.read_csv(SAMPLE_SUBMISSION_CSV).shape[0]
assert (
    len(test_filenames) == expected_rows
), "Number of test images ({}) does not match expected ({})".format(
    len(test_filenames), expected_rows
)

k = 5  # number of nearest neighbours to consider
pred_labels = []
for fname in test_filenames:
    test_path = os.path.join(TEST_IMAGES_DIR, fname)
    if os.path.isfile(test_path):
        test_size = os.path.getsize(test_path)

        diffs = np.abs(train_sizes - test_size)
        if train_sizes.shape[0] <= k:
            nearest_idxs = np.arange(train_sizes.shape[0])
        else:
            nearest_idxs = np.argpartition(diffs, k - 1)[:k]

        nearest_labels = train_labels[nearest_idxs]
        vote_counts = collections.Counter(nearest_labels.tolist())
        best_label = vote_counts.most_common(1)[0][0]
        pred_labels.append(best_label)
    else:
        pred_labels.append(most_common_label)

submission_df = pd.DataFrame({"image_id": test_filenames, "label": pred_labels})




## === cell 2
submission_df.to_csv(SUBMISSION_PATH, index=False)
print("Submission written to {}".format(SUBMISSION_PATH))
