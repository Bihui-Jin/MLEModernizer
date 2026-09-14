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

# 5. Target score

0.8907524932003626

# 6. Current score

0.18909

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The script now safely handles missing TensorFlow/models, falls back to a simple baseline prediction (the most common label from the training set), and always writes a correctly‑named `submission.csv` so Kaggle can accept it.'
- What this solution (achieved 0.61099) has done: 'The script now safely falls back to a lightweight image‑based heuristic when TensorFlow models cannot be used. It imports Pillow, builds average RGB centroids for each class from a limited subset of the training images, and classifies each test image by nearest centroid. This replaces the previous “most common label” fallback, giving a much better accuracy while keeping the original model‑based path unchanged. The submission file is still written to `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fixed the import error handling and replaced the simple RGB‑centroid fallback with a more discriminative small‑image prototype approach. For each class we now compute an average 32×32 RGB image from many training samples (up to 500 per class). During inference the test image is resized to the same size and assigned to the class whose prototype image is closest in Euclidean distance. This richer representation boosts the accuracy toward the target while keeping the original TensorFlow path unchanged and preserving all existing logic.'
- What this solution (achieved 0.31054) has done: 'Implemented a safe import strategy that disables TensorFlow outright to avoid the protobuf crash and added a lightweight K‑Nearest‑Neighbor fallback using multiple class prototypes (up to 30 images per class). This richer image‑based heuristic replaces the single‑centroid approach, improving discrimination while keeping the original workflow intact. The script now builds prototype arrays during training preprocessing and uses them for fast Euclidean‑distance classification, finally writing a proper `submission.csv`.'
- What this solution (achieved 0.17564) has done: 'Implemented modest enhancements to the fallback image‑based classifier:
- Increased prototype count (`max_prototypes`) to capture richer class variability.
- Pre‑computed flattened mean images for each class.
- During inference, evaluated both prototype‑nearest and mean‑image distances, selecting the class with the smallest distance.  
These changes stay within the existing workflow and are expected to raise validation accuracy toward the target without altering the core model logic.'
- What this solution (achieved 0.18909) has done: 'I increase the image resolution and allow more training samples per class to enrich the prototype‐based fallback classifier. By using 64×64 images and up to 1 000 samples per class (with 200 prototypes each), the Euclidean‑distance matching should capture more visual detail and improve validation accuracy, moving the score closer to the target while keeping the overall workflow unchanged.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
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
max_per_class = 1000  # number of images used to compute the class mean
max_prototypes = 200  # number of prototype images kept per class
img_target_sz = (64, 64)  # higher resolution for better discrimination

sums = {
    c: np.zeros((img_target_sz[0], img_target_sz[1], 3), dtype=np.float64)
    for c in range(5)
}
counts = {c: 0 for c in range(5)}

proto_lists = {c: [] for c in range(5)}

random.seed(42)

for _, row in train_df.iterrows():
    label = int(row["label"])
    img_path = os.path.join(train_image_dir, row["image_id"])
    if not os.path.exists(img_path):
        continue
    try:
        img = Image.open(img_path).convert("RGB")
        img = img.resize(img_target_sz)
        arr = np.asarray(img, dtype=np.float64) / 255.0

        if counts[label] < max_per_class:
            sums[label] += arr
            counts[label] += 1

        if len(proto_lists[label]) < max_prototypes:
            proto_lists[label].append(arr)
    except Exception:
        continue

class_mean_images = {}
for c in range(5):
    if counts[c] > 0:
        class_mean_images[c] = sums[c] / counts[c]
    else:
        class_mean_images[c] = np.zeros(
            (img_target_sz[0], img_target_sz[1], 3), dtype=np.float64
        )

class_prototypes = {}
for c in range(5):
    if proto_lists[c]:
        class_prototypes[c] = np.stack(proto_lists[c])  # shape (N, H, W, 3)
    else:
        class_prototypes[c] = np.empty((0, img_target_sz[0], img_target_sz[1], 3))




## === cell 3
predictions = []

if TF_AVAILABLE:
    pass
else:
    if PIL_AVAILABLE and class_prototypes:
        flat_prototypes = {}
        for c, arrs in class_prototypes.items():
            if arrs.size > 0:
                flat_prototypes[c] = arrs.reshape(arrs.shape[0], -1)  # (N, D)
            else:
                flat_prototypes[c] = np.empty(
                    (0, img_target_sz[0] * img_target_sz[1] * 3)
                )

        flat_means = {c: class_mean_images[c].reshape(-1) for c in range(5)}  # (D,)

        for image_id in sample_df["image_id"]:
            image_path = os.path.join(test_image_dir, image_id)
            try:
                img = Image.open(image_path).convert("RGB")
                img = img.resize(img_target_sz)
                arr = np.asarray(img, dtype=np.float64) / 255.0
                flat_arr = arr.reshape(-1)  # (D,)

                best_label = fallback_label
                best_dist = None

                for c in range(5):
                    dist_mean = np.linalg.norm(flat_arr - flat_means[c])

                    proto_flat = flat_prototypes[c]
                    if proto_flat.shape[0] > 0:
                        dists = np.linalg.norm(proto_flat - flat_arr, axis=1)
                        dist_proto = dists.min()
                        cur_dist = min(dist_mean, dist_proto)
                    else:
                        cur_dist = dist_mean

                    if best_dist is None or cur_dist < best_dist:
                        best_dist = cur_dist
                        best_label = c

                pred = int(best_label)
            except Exception:
                pred = fallback_label
            predictions.append(pred)
    else:
        predictions = [fallback_label] * len(sample_df)

submission_df = pd.DataFrame({"image_id": sample_df["image_id"], "label": predictions})




## === cell 4
output_path = "/kaggle/working/submission.csv"
submission_df.to_csv(output_path, index=False)
print(f"Submission file saved at: {output_path}")
