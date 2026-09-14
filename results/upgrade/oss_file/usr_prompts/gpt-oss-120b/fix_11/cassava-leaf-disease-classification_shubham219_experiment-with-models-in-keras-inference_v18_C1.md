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
import glob
import numpy as np
import pandas as pd
from PIL import Image
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from sklearn.neighbors import NearestNeighbors
from sklearn.preprocessing import StandardScaler

SEED = 42
DEBUG = False
np.random.seed(SEED)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
if not os.path.exists(train_csv_path):
    train_csv_path = "data/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)

id_to_label = dict(zip(train_df["image_id"], train_df["label"]))




## === cell 2
CHUNKSIZE = 500  # larger chunk reduces task‑dispatch overhead
MAX_WORKERS = min(8, os.cpu_count())  # cap workers to avoid oversubscription

train_images = glob.glob(
    "../input/cassava-leaf-disease-classification/train_images/*.jpg"
)
if not train_images:
    train_images = glob.glob(
        "data/cassava-leaf-disease-classification/train_images/*.jpg"
    )
train_images.sort()  # deterministic order


def _hist16(channel_arr):
    """Return a 16‑bin density histogram for a single channel."""
    bins = (channel_arr // 16).ravel()
    hist = np.bincount(bins, minlength=16).astype(np.float32)
    total = hist.sum()
    if total > 0:
        hist /= total
    return hist


def _process_train(path):
    img_id = os.path.basename(path)
    if img_id not in id_to_label:
        return None  # skip stray files
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)  # (H, W, 3)
        mean_rgb = arr.mean(axis=(0, 1), dtype=np.float32)
        std_rgb = arr.std(axis=(0, 1), dtype=np.float32)
        hist = [_hist16(arr[:, :, ch]) for ch in range(3)]
        feat = np.concatenate([mean_rgb, std_rgb, np.concatenate(hist)])  # (54,)
        return (img_id, feat.astype(np.float32))  # store as float32
    except Exception:
        return None


with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
    results = list(exe.map(_process_train, train_images, chunksize=CHUNKSIZE))

train_ids = []
train_features = []
for res in results:
    if res is not None:
        img_id, feat = res
        train_ids.append(img_id)
        train_features.append(feat)

train_features = np.vstack(train_features)  # (N_train, 54), float32
train_labels = np.array([id_to_label[i] for i in train_ids])

scaler = StandardScaler()
train_features = scaler.fit_transform(train_features.astype(np.float32))




## === cell 3
test_images = glob.glob(
    "../input/cassava-leaf-disease-classification/test_images/*.jpg"
)
if not test_images:
    test_images = glob.glob(
        "data/cassava-leaf-disease-classification/test_images/*.jpg"
    )
test_images.sort()  # deterministic order


def _process_test(path):
    img_id = os.path.basename(path)
    try:
        img = Image.open(path).convert("RGB")
        arr = np.array(img, dtype=np.uint8)
        mean_rgb = arr.mean(axis=(0, 1), dtype=np.float32)
        std_rgb = arr.std(axis=(0, 1), dtype=np.float32)
        hist = [_hist16(arr[:, :, ch]) for ch in range(3)]
        feat = np.concatenate([mean_rgb, std_rgb, np.concatenate(hist)])  # (54,)
        return (img_id, feat.astype(np.float32))
    except Exception:
        return (img_id, np.full(54, np.nan, dtype=np.float32))


with ProcessPoolExecutor(max_workers=MAX_WORKERS) as exe:
    test_results = list(exe.map(_process_test, test_images, chunksize=CHUNKSIZE))

test_ids = [img_id for img_id, _ in test_results]
test_features = np.vstack([feat for _, feat in test_results])  # (N_test, 54)




## === cell 4
most_common_label = train_df["label"].mode()[0]
k = 7  # a slightly larger neighbourhood

nn = NearestNeighbors(n_neighbors=k, algorithm="ball_tree", metric="manhattan")
nn.fit(train_features)

nan_mask = np.isnan(test_features).any(axis=1)

valid_features = test_features[~nan_mask]
valid_features = scaler.transform(valid_features.astype(np.float32))

distances, indices = nn.kneighbors(valid_features, return_distance=True)

pred_labels = []


def _vote(neigh_idx, neigh_dists):
    """Weighted vote with exponential decay based on distance."""
    neigh_labels = train_labels[neigh_idx]
    weights = np.exp(-neigh_dists)
    label_weights = {}
    for lbl, w in zip(neigh_labels, weights):
        label_weights[lbl] = label_weights.get(lbl, 0.0) + w

    max_weight = max(label_weights.values())
    candidates = [lbl for lbl, w in label_weights.items() if w == max_weight]

    if len(candidates) == 1:
        return candidates[0]
    for lbl in neigh_labels:
        if lbl in candidates:
            return lbl
    return candidates[0]


for neigh_idx_row, neigh_dist_row in zip(indices, distances):
    pred_labels.append(_vote(neigh_idx_row, neigh_dist_row))

for _ in range(nan_mask.sum()):
    pred_labels.append(most_common_label)




## === cell 5
submission = pd.DataFrame({"image_id": test_ids, "label": pred_labels})
submission.to_csv("submission.csv", index=False)




## === cell 6
print(submission.head())
