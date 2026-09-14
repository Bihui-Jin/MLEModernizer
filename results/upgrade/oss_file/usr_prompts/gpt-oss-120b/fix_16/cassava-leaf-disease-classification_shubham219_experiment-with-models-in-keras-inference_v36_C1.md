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
import pandas as pd
import numpy as np
from PIL import Image
import concurrent.futures

SEED = 42
np.random.seed(SEED)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "../input/cassava-leaf-disease-classification/train_images/"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)


def _extract_feat(img_path):
    """Return a 15‑dim colour feature (mean, std, median, min, max) or None on error."""
    try:
        img = Image.open(img_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32)  # shape (H, W, 3)
        mean = arr.mean(axis=(0, 1))
        std = arr.std(axis=(0, 1))
        median = np.median(arr, axis=(0, 1))
        cmin = arr.min(axis=(0, 1))
        cmax = arr.max(axis=(0, 1))
        return np.concatenate([mean, std, median, cmin, cmax])
    except Exception:
        return None


train_paths = [
    os.path.join(train_images_dir, img_id) for img_id in train_df["image_id"]
]
train_labels_all = train_df["label"].astype(int).values

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    train_feats_raw = list(exe.map(_extract_feat, train_paths, chunksize=128))

train_feats = []
train_labels = []
for ft, lb in zip(train_feats_raw, train_labels_all):
    if ft is not None:
        train_feats.append(ft)
        train_labels.append(lb)

train_feats = np.stack(train_feats).astype(np.float32)  # (N_train, 15)
train_labels = np.array(train_labels, dtype=np.int32)  # (N_train,)

feat_mean = train_feats.mean(axis=0, keepdims=True)
feat_std = train_feats.std(axis=0, keepdims=True) + 1e-6
train_feats = (train_feats - feat_mean) / feat_std

train_norms = np.einsum("ij,ij->i", train_feats, train_feats).astype(
    np.float32
)  # (N_train,)

test_image_paths = sorted(glob.glob(os.path.join(test_images_dir, "*.jpg")))

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as exe:
    test_feats_raw = list(exe.map(_extract_feat, test_image_paths, chunksize=128))

fallback_label = int(train_df["label"].mode()[0])
k = 35  # neighbour count

valid_test_idx = [i for i, ft in enumerate(test_feats_raw) if ft is not None]

if valid_test_idx:
    test_feats = np.stack([test_feats_raw[i] for i in valid_test_idx]).astype(
        np.float32
    )
    test_feats = (test_feats - feat_mean) / feat_std

    N_train = train_feats.shape[0]
    N_test = test_feats.shape[0]
    batch_size = 256  # process test set in batches to limit memory
    pred_labels_valid = []
    eps = 1e-6

    for start in range(0, N_test, batch_size):
        end = min(start + batch_size, N_test)
        batch = test_feats[start:end]  # (B, 15)
        test_norms = np.einsum("ij,ij->i", batch, batch).astype(np.float32)  # (B,)

        dists_sq = (
            train_norms[:, None]  # (N_train, 1)
            - 2.0 * train_feats @ batch.T  # (N_train, B)
            + test_norms[None, :]  # (1, B)
        )
        dists = np.sqrt(np.maximum(dists_sq, 0.0, dtype=np.float32))  # (N_train, B)

        if k < N_train:
            knn_idx = np.argpartition(dists, k, axis=0)[:k, :]  # (k, B)
        else:
            knn_idx = np.arange(N_train)[:, None].repeat(dists.shape[1], axis=1)

        knn_labels = train_labels[knn_idx]  # (k, B)
        knn_dists = dists[knn_idx, np.arange(end - start)]  # (k, B)

        for col in range(end - start):
            labels_col = knn_labels[:, col]
            dists_col = knn_dists[:, col]
            weights = 1.0 / (dists_col + eps)

            label_weights = np.zeros(5, dtype=np.float32)
            for lbl, wgt in zip(labels_col, weights):
                label_weights[lbl] += wgt
            best_label = int(np.argmax(label_weights))
            pred_labels_valid.append(best_label)
else:
    pred_labels_valid = []

pred_labels = []
valid_iter = iter(pred_labels_valid)
for ft in test_feats_raw:
    if ft is None:
        pred_labels.append(fallback_label)
    else:
        pred_labels.append(next(valid_iter))

submission = pd.DataFrame(
    {"image_id": [os.path.basename(p) for p in test_image_paths], "label": pred_labels}
)




## === cell 2
output_path = "submission.csv"
submission.to_csv(output_path, index=False)
print(f"Submission file written to {output_path} with {len(submission)} rows.")
