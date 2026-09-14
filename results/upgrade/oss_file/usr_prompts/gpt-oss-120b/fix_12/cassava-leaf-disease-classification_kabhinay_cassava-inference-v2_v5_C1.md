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
import os
import numpy as np
import pandas as pd
from PIL import Image, ImageStat  # use ImageStat for fast mean/std
import concurrent.futures  # parallel image processing

np.random.seed(42)




## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_df = pd.read_csv(train_csv_path)




## === cell 2
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"


def _extract_feature(img_path):
    """Return a richer colour feature (mean/std + 32‑bin RGB & HSV histograms)."""
    try:
        with Image.open(img_path) as img:
            img = img.convert("RGB")
            stat = ImageStat.Stat(img)
            mean_std = stat.mean + stat.stddev  # list of 6 floats

            arr = np.array(img)  # (H, W, 3)
            hist_bins = 32
            rgb_hist = []
            for ch in range(3):
                h, _ = np.histogram(
                    arr[:, :, ch].ravel(),
                    bins=hist_bins,
                    range=(0, 256),
                    density=True,
                )
                rgb_hist.extend(h.astype(np.float32))

            hsv_img = img.convert("HSV")
            hsv_arr = np.array(hsv_img)
            hsv_hist = []
            for ch in range(3):
                h, _ = np.histogram(
                    hsv_arr[:, :, ch].ravel(),
                    bins=hist_bins,
                    range=(0, 256),
                    density=True,
                )
                hsv_hist.extend(h.astype(np.float32))

            feature = np.array(mean_std + rgb_hist + hsv_hist, dtype=np.float32)
            return feature
    except Exception:
        return None


train_paths_labels = [
    (os.path.join(train_images_dir, row["image_id"]), int(row["label"]))
    for _, row in train_df.iterrows()
]

train_features = []
train_labels = []
with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    for feat, (path, label) in zip(
        executor.map(lambda pl: _extract_feature(pl[0]), train_paths_labels),
        train_paths_labels,
    ):
        if feat is not None:
            train_features.append(feat)
            train_labels.append(label)

train_features_arr = np.stack(train_features)  # (N_train, 150)
train_labels_arr = np.array(train_labels, dtype=np.int32)  # (N_train,)

feat_mean = train_features_arr.mean(axis=0, keepdims=True)
feat_std = train_features_arr.std(axis=0, keepdims=True)
feat_std[feat_std == 0] = 1.0
train_features_arr = (train_features_arr - feat_mean) / feat_std




## === cell 3
test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
test_filenames = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
num_test = len(test_filenames)

fallback_label = int(train_df["label"].value_counts().idxmax())


def _extract_test_feature(filename):
    img_path = os.path.join(test_dir, filename)
    return _extract_feature(img_path)


with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
    test_features_list = list(executor.map(_extract_test_feature, test_filenames))

test_features_arr = np.array(
    [
        (
            feat
            if feat is not None
            else np.full(train_features_arr.shape[1], np.nan, dtype=np.float32)
        )
        for feat in test_features_list
    ]
)

test_features_arr = (test_features_arr - feat_mean) / feat_std

k = 5  # a slightly larger neighbourhood for more stable voting
predictions = []

for idx, test_feat in enumerate(test_features_arr):
    if np.isnan(test_feat).any():
        pred_label = fallback_label
    else:
        dists = np.linalg.norm(train_features_arr - test_feat, axis=1)
        kn = min(k, len(dists))
        nearest_idxs = np.argpartition(dists, kn - 1)[:kn]
        nearest_labels = train_labels_arr[nearest_idxs]
        nearest_dists = dists[nearest_idxs]
        weights = 1.0 / (nearest_dists + 1e-8)
        counts = np.bincount(nearest_labels, weights=weights, minlength=5)
        pred_label = int(np.argmax(counts))
    predictions.append(pred_label)




## === cell 4
submission_path = "/kaggle/working/submission.csv"
results = pd.DataFrame({"image_id": test_filenames, "label": predictions})
results.to_csv(submission_path, index=False)
