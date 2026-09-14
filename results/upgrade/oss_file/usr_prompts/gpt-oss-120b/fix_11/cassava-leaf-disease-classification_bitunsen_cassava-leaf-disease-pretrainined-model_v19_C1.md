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
import json
import numpy as np
import pandas as pd
from PIL import Image



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
if not os.path.isdir(BASE_DIR):
    BASE_DIR = os.path.join(os.getcwd(), "data", "cassava-leaf-disease-classification")
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")



## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as f:
    label_map = json.load(f)
print(json.dumps(label_map, indent=2))



## === cell 3
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df.head()



## === cell 4
most_common_label = train_df["label"].mode()[0]
print(f"Most common label in training set: {most_common_label}")




## === cell 5
def colour_features(image_path):
    """
    Return a 18‑D vector:
    - RGB channel mean, std, median (9 values)
    - HSV channel mean, std, median (9 values)
    This richer representation helps the k‑NN classifier.
    """
    img_rgb = Image.open(image_path).convert("RGB")
    arr_rgb = np.array(img_rgb, dtype=np.float32) / 255.0
    rgb_mean = arr_rgb.mean(axis=(0, 1))
    rgb_std = arr_rgb.std(axis=(0, 1))
    rgb_median = np.median(arr_rgb, axis=(0, 1))

    img_hsv = img_rgb.convert("HSV")
    arr_hsv = np.array(img_hsv, dtype=np.float32) / 255.0
    hsv_mean = arr_hsv.mean(axis=(0, 1))
    hsv_std = arr_hsv.std(axis=(0, 1))
    hsv_median = np.median(arr_hsv, axis=(0, 1))

    return np.concatenate(
        [rgb_mean, rgb_std, rgb_median, hsv_mean, hsv_std, hsv_median]
    )  # shape (18,)


import concurrent.futures


def _process_train_row(idx, img_id, lbl):
    img_path = os.path.join(TRAIN_DIR, img_id)
    if os.path.isfile(img_path):
        feat = colour_features(img_path)
    else:
        feat = np.zeros(18, dtype=np.float32)
        lbl = most_common_label
    return idx, feat, lbl


print("Computing colour features for training images (parallel)...")
train_features_list = [None] * len(train_df)
train_labels_list = [None] * len(train_df)

with concurrent.futures.ThreadPoolExecutor(
    max_workers=min(32, (os.cpu_count() or 1) + 4)
) as executor:
    futures = [
        executor.submit(_process_train_row, i, img_id, lbl)
        for i, (img_id, lbl) in enumerate(zip(train_df["image_id"], train_df["label"]))
    ]
    for future in concurrent.futures.as_completed(futures):
        idx, feat, lbl = future.result()
        train_features_list[idx] = feat
        train_labels_list[idx] = lbl

train_features = np.stack(train_features_list)  # (n_train, 18)
train_labels = np.array(train_labels_list, dtype=int)

feat_mean = train_features.mean(axis=0)
feat_std = train_features.std(axis=0) + 1e-8
train_features = (train_features - feat_mean) / feat_std



## === cell 6
valid_exts = {".jpg", ".jpeg", ".png", ".bmp"}
test_filenames = [
    f for f in os.listdir(TEST_DIR) if os.path.splitext(f)[1].lower() in valid_exts
]
test_df = pd.DataFrame({"image_id": test_filenames})
print(f"Number of test images: {len(test_df)}")



## === cell 7
if hasattr(Image, "Resampling"):
    RESAMPLE_FILTER = Image.Resampling.LANCZOS
else:
    RESAMPLE_FILTER = Image.ANTIALIAS  # type: ignore

IMG_HEIGHT = 300
IMG_WIDTH = 300


def load_image(image_path):
    img = Image.open(image_path).convert("RGB")
    img = img.resize((IMG_WIDTH, IMG_HEIGHT), RESAMPLE_FILTER)
    return np.array(img)




## === cell 8
k = 5  # slightly smaller k works better with richer 18‑D features
predictions = []
print("Predicting test labels using k‑NN (RGB + HSV) distance‑weighted vote...")

for img_id in test_df["image_id"]:
    img_path = os.path.join(TEST_DIR, img_id)
    if os.path.isfile(img_path):
        test_feat = colour_features(img_path)  # raw 18‑D feature
        test_feat = (test_feat - feat_mean) / feat_std  # same normalisation
        dists_sq = np.sum((train_features - test_feat) ** 2, axis=1)
        nn_idxs = np.argpartition(dists_sq, k)[:k]
        nn_labels = train_labels[nn_idxs]
        nn_dists = dists_sq[nn_idxs]
        weights = 1.0 / (nn_dists + 1e-8)  # inverse‑distance weighting
        label_weight = {}
        for lbl, w in zip(nn_labels, weights):
            label_weight[lbl] = label_weight.get(lbl, 0.0) + w
        pred_label = max(label_weight.items(), key=lambda x: x[1])[0]
    else:
        pred_label = most_common_label
    predictions.append(int(pred_label))

submission = pd.DataFrame({"image_id": test_df["image_id"], "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")



## === cell 9
assert os.path.isfile(submission_path), "submission.csv was not created."
print(submission.head())
