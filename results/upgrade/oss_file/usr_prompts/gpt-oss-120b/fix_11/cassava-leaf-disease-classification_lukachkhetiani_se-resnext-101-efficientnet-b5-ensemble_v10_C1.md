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
import pandas as pd
import numpy as np
from tqdm import tqdm
from PIL import Image
import concurrent.futures

test_dir = "../input/cassava-leaf-disease-classification/test_images"
train_dir = "../input/cassava-leaf-disease-classification/train_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"

sample_sub = pd.read_csv(sample_sub_path)
ordered_image_ids = sample_sub["image_id"].tolist()
train_df = pd.read_csv(train_csv_path)


def color_feature(image_path):
    """
    Return an 18‑dim vector:
        - mean and std of RGB channels (6 values)
        - mean and std of HSV channels (6 values)
        - mean and std of LAB channels (6 values)
    If reading fails, return a zero vector.
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            arr_rgb = np.asarray(img, dtype=np.float32)  # (H, W, 3)

            mean_rgb = arr_rgb.mean(axis=(0, 1))
            std_rgb = arr_rgb.std(axis=(0, 1))

            img_hsv = img.convert("HSV")
            arr_hsv = np.asarray(img_hsv, dtype=np.float32)
            mean_hsv = arr_hsv.mean(axis=(0, 1))
            std_hsv = arr_hsv.std(axis=(0, 1))

            img_lab = img.convert("LAB")
            arr_lab = np.asarray(img_lab, dtype=np.float32)
            mean_lab = arr_lab.mean(axis=(0, 1))
            std_lab = arr_lab.std(axis=(0, 1))

            return np.concatenate(
                [mean_rgb, std_rgb, mean_hsv, std_hsv, mean_lab, std_lab]
            )
    except Exception:
        return np.zeros(18, dtype=np.float32)


def process_train_row(row):
    """
    Convert a training row (namedtuple) into a feature vector and label.
    """
    img_name = row.image_id
    label = int(row.label)
    img_path = os.path.join(train_dir, img_name)
    feat = color_feature(img_path)
    return feat, label


print("Preparing training features (RGB+HSV+LAB mean + std) using parallel I/O…")
train_features = []
train_labels = []

rows = list(train_df.itertuples(index=False))

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    for feat, label in tqdm(
        executor.map(process_train_row, rows), total=len(rows), desc="Train images"
    ):
        train_features.append(feat)
        train_labels.append(label)

train_features = np.stack(train_features).astype(np.float32)  # (N_train, 18)
train_labels = np.array(train_labels, dtype=np.int32)  # (N_train,)

feat_mean = train_features.mean(axis=0, keepdims=True)
feat_std = train_features.std(axis=0, keepdims=True) + 1e-6
train_features = (train_features - feat_mean) / feat_std

k = 7  # odd number to reduce ties
eps = 1e-6  # avoid division by zero




## === cell 1
def predict_one(args):
    """
    Predict the label for a single test image using weighted k‑NN.
    """
    idx, img_name = args
    img_path = os.path.join(test_dir, img_name)

    test_feat = color_feature(img_path).astype(np.float32)
    test_feat = (test_feat - feat_mean.squeeze()) / feat_std.squeeze()

    dists = np.linalg.norm(train_features - test_feat, axis=1)

    knn_idx = np.argpartition(dists, k)[:k]
    knn_labels = train_labels[knn_idx]
    knn_dists = dists[knn_idx]

    weights = 1.0 / (knn_dists + eps)
    weight_sum = {}
    for lbl, w in zip(knn_labels, weights):
        weight_sum[lbl] = weight_sum.get(lbl, 0.0) + w

    max_weight = max(weight_sum.values())
    candidate_labels = [lbl for lbl, w in weight_sum.items() if w == max_weight]
    pred_label = min(candidate_labels)  # tie‑break by smallest label

    return idx, img_name, int(pred_label)


print("Predicting test images via weighted k‑NN on scaled RGB+HSV+LAB features…")
args_iter = list(enumerate(ordered_image_ids))

with concurrent.futures.ThreadPoolExecutor(max_workers=os.cpu_count() or 4) as executor:
    results = list(
        tqdm(
            executor.map(predict_one, args_iter),
            total=len(args_iter),
            desc="Test images",
        )
    )

results.sort(key=lambda x: x[0])
names = [img_name for _, img_name, _ in results]
preds = [pred for _, _, pred in results]

submission = pd.DataFrame({"image_id": names, "label": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
