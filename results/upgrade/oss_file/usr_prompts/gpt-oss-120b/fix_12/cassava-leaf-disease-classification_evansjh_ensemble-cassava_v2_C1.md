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
import pandas as pd
import numpy as np
import random

TF_AVAILABLE = False
print("TensorFlow disabled; using fallback image‑based classifier.")

try:
    from PIL import Image

    PIL_AVAILABLE = True
except Exception as e:
    PIL_AVAILABLE = False
    print("Pillow not usable:", e)




## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
sample_submission_path = os.path.join(base_path, "sample_submission.csv")
test_image_dir = os.path.join(base_path, "test_images")
train_image_dir = os.path.join(base_path, "train_images")

train_df = pd.read_csv(train_csv_path)
fallback_label = int(train_df["label"].mode()[0])

sample_df = pd.read_csv(sample_submission_path)




## === cell 2
max_per_class = 1000  # images used to compute class mean
max_prototypes = 500  # prototype images kept per class
img_target_sz = (96, 96)  # resolution

sums = {
    c: np.zeros((img_target_sz[0], img_target_sz[1], 3), dtype=np.float32)
    for c in range(5)
}
counts = {c: 0 for c in range(5)}
proto_lists = {c: [] for c in range(5)}

random.seed(42)

shuffled_df = train_df.sample(frac=1, random_state=42).itertuples(index=False)

class_done = {c: False for c in range(5)}
remaining = 5  # number of classes still needing data

for row in shuffled_df:
    label = int(row.label)
    if class_done[label]:
        continue  # this class already has enough data

    img_path = os.path.join(train_image_dir, row.image_id)
    if not os.path.exists(img_path):
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_target_sz)
        arr = np.asarray(img, dtype=np.float32) / 255.0
    except Exception:
        continue

    if counts[label] < max_per_class:
        sums[label] += arr
        counts[label] += 1

    if len(proto_lists[label]) < max_prototypes:
        proto_lists[label].append(arr)

    if counts[label] >= max_per_class and len(proto_lists[label]) >= max_prototypes:
        class_done[label] = True
        remaining -= 1
        if remaining == 0:
            break  # all classes satisfied, stop early

class_mean_images = {
    c: (
        sums[c] / counts[c]
        if counts[c] > 0
        else np.zeros((img_target_sz[0], img_target_sz[1], 3), dtype=np.float32)
    )
    for c in range(5)
}

class_prototypes = {
    c: (
        np.stack(proto_lists[c])
        if proto_lists[c]
        else np.empty((0, img_target_sz[0], img_target_sz[1], 3), dtype=np.float32)
    )
    for c in range(5)
}

D = img_target_sz[0] * img_target_sz[1] * 3
flat_prototypes = {}
for c in range(5):
    if class_prototypes[c].shape[0] > 0:
        flat_prototypes[c] = (
            class_prototypes[c]
            .reshape(class_prototypes[c].shape[0], -1)
            .astype(np.float32)
        )
    else:
        flat_prototypes[c] = np.empty((0, D), dtype=np.float32)

all_flat_prototypes = np.concatenate(
    [flat_prototypes[c] for c in range(5)], axis=0
)  # shape (total_prototypes, D)
prototype_labels = np.concatenate(
    [np.full(flat_prototypes[c].shape[0], c, dtype=np.int32) for c in range(5)], axis=0
)  # shape (total_prototypes,)




## === cell 3
predictions = []

if TF_AVAILABLE:
    pass
else:
    batch_size = 64
    test_ids = list(sample_df["image_id"])
    for start in range(0, len(test_ids), batch_size):
        batch_ids = test_ids[start : start + batch_size]
        batch_arrs = []
        for image_id in batch_ids:
            image_path = os.path.join(test_image_dir, image_id)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(img_target_sz)
                arr = np.asarray(img, dtype=np.float32) / 255.0
            except Exception:
                arr = None
            batch_arrs.append(arr)

        for arr in batch_arrs:
            if arr is None or all_flat_prototypes.shape[0] == 0:
                pred = fallback_label
            else:
                flat_arr = arr.reshape(-1).astype(np.float32)  # (D,)
                dists = np.linalg.norm(all_flat_prototypes - flat_arr, axis=1)
                best_idx = dists.argmin()
                pred = int(prototype_labels[best_idx])
            predictions.append(pred)

submission_df = pd.DataFrame({"image_id": sample_df["image_id"], "label": predictions})




## === cell 4
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file saved at: {output_path}")
