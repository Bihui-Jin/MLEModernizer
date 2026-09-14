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
import os, json, random, time
import numpy as np, pandas as pd

tf_available = False

from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

np.random.seed(42)



## === cell 1
train_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
test_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_path)
test_df = pd.read_csv(test_path)

numeric_mode_label = int(train_df["label"].mode()[0])

train_df["label"] = train_df["label"].astype(str)
train_df["filepath"] = train_dir + train_df["image_id"].astype(str)
test_df["filepath"] = test_dir + test_df["image_id"].astype(str)

train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

from sklearn.ensemble import RandomForestClassifier
from PIL import Image
import multiprocessing as mp
import concurrent.futures
import gc

IMG_SIZE = (128, 128)
FLAT_DIM = IMG_SIZE[0] * IMG_SIZE[1] * 3
CHANNEL_STATS_DIM = 6  # mean & std for each of 3 channels
HIST_BINS = 16 * 3  # 16 bins per channel
FEATURE_DIM = FLAT_DIM + CHANNEL_STATS_DIM + HIST_BINS


def extract_features(img_path, size=IMG_SIZE):
    """Resize image, flatten RGB pixels, append channel stats, and colour histograms."""
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(size, Image.NEAREST)  # fast resize, same as original intent
        arr_uint8 = np.asarray(img, dtype=np.uint8)  # (H, W, 3) uint8

        flat = arr_uint8.flatten().astype(np.float32)

        channel_means = arr_uint8.mean(axis=(0, 1)).astype(np.float32)
        channel_stds = arr_uint8.std(axis=(0, 1)).astype(np.float32)

        hist_bins = 16
        hist_range = (0, 256)
        hist_r, _ = np.histogram(arr_uint8[:, :, 0], bins=hist_bins, range=hist_range)
        hist_g, _ = np.histogram(arr_uint8[:, :, 1], bins=hist_bins, range=hist_range)
        hist_b, _ = np.histogram(arr_uint8[:, :, 2], bins=hist_bins, range=hist_range)
        hist = np.concatenate([hist_r, hist_g, hist_b]).astype(np.float32)

        return np.concatenate([flat, channel_means, channel_stds, hist])
    except Exception:
        return np.zeros(FEATURE_DIM, dtype=np.float32)


def _extract_indexed(idx_path):
    """Helper for thread‑based parallel processing that keeps original index."""
    idx, path = idx_path
    return idx, extract_features(path)


def parallel_features(paths):
    """
    Extract features in parallel using a thread pool.
    Threads share memory, avoiding the huge pickling overhead of multiprocessing.
    """
    n = len(paths)
    result = np.empty((n, FEATURE_DIM), dtype=np.float32)

    indexed_paths = list(enumerate(paths))

    max_workers = max(1, mp.cpu_count() - 1)
    with concurrent.futures.ThreadPoolExecutor(max_workers=max_workers) as executor:
        for idx, feat in executor.map(_extract_indexed, indexed_paths):
            result[idx] = feat
    return result


def main():
    start_total = time.time()

    train_paths = train_df["filepath"].tolist()
    train_labels = train_df["label"].astype(int).values

    t0 = time.time()
    X_train = parallel_features(train_paths)
    print(f"Feature extraction (train) took {time.time() - t0:.2f}s")
    y_train = np.array(train_labels)

    gc.collect()

    rf = RandomForestClassifier(
        n_estimators=1200,  # slightly more trees for better stability
        max_depth=None,
        random_state=42,
        n_jobs=-1,
        min_samples_leaf=1,
        class_weight="balanced",  # mitigate class imbalance
    )
    t1 = time.time()
    rf.fit(X_train, y_train)
    print(f"RandomForest training took {time.time() - t1:.2f}s")

    test_paths = test_df["filepath"].tolist()
    t2 = time.time()
    X_test = parallel_features(test_paths)
    print(f"Feature extraction (test) took {time.time() - t2:.2f}s")

    pred_labels = rf.predict(X_test).astype(int)

    submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
    submission_path = "/kaggle/working/submission.csv"
    submission.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")

    print("Full run time:", time.time() - start_total)
    print(pd.read_csv(submission_path).head())


if __name__ == "__main__":
    main()
