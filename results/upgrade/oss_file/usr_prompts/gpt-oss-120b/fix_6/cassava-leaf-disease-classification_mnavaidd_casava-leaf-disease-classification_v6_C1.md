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

# 5. Target score

0.6976427923844062

# 6. Current score

0.18311

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.18087) has done: 'I replaced the TensorFlow‑based pipeline with a lightweight nearest‑centroid classifier that uses only OpenCV and NumPy. This avoids the protobuf import error, restores end‑to‑end execution, and still provides a reasonable accuracy (centroid‑based classification often reaches ~0.70 on this dataset). The script now loads and resizes images, computes per‑class mean feature vectors, predicts test labels by nearest centroid, and writes a proper `submission.csv`.'
- What this solution (achieved 0.28438) has done: 'I increase the image resolution to 64 × 64 and normalize every feature vector to unit length, then also normalize the class centroids. Using cosine similarity (via a dot product) instead of raw Euclidean distance usually gives a sharper nearest‑centroid decision on normalized data, which should boost accuracy toward the target without altering the overall pipeline. The changes are limited to preprocessing and the distance computation, keeping the core logic intact.'
- What this solution (achieved 0.18311) has done: 'I increase the image resolution to 96 × 96, compute both normalized (for cosine similarity) and raw (for Euclidean distance) class centroids, and combine these two similarity measures when predicting. This keeps the original nearest‑centroid idea but adds a small amount of extra information that should raise the accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2



## === cell 1
data_path = "../input/cassava-leaf-disease-classification/"
train_csv_path = data_path + "train.csv"
label_json_path = data_path + "label_num_to_disease_map.json"
train_images_dir = data_path + "train_images/"
test_images_dir = data_path + "test_images/"



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)  # ensure numeric labels

label_map = pd.read_json(label_json_path, orient="index")
label_names = label_map.values.flatten().tolist()

print("Label names:")
for i, name in enumerate(label_names):
    print(f" {i}. {name}")



## === cell 3
BATCH_SIZE = 32  # kept for compatibility, not used
IMG_SIZE = 96  # larger size for richer centroids (was 64)
SEED = 42




## === cell 4
def load_and_preprocess(img_path):
    """
    Read an image, convert to RGB, resize, flatten.
    Returns:
        norm_feat: L2‑normalized feature vector (float32)
        raw_feat:  raw (un‑normalized) feature vector (float32)
    """
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    raw_feat = img.flatten().astype(np.float32)

    norm = np.linalg.norm(raw_feat)
    if norm > 0:
        norm_feat = raw_feat / norm
    else:
        norm_feat = raw_feat
    return norm_feat, raw_feat


num_classes = 5
feature_len = IMG_SIZE * IMG_SIZE * 3

class_sums_norm = np.zeros((num_classes, feature_len), dtype=np.float64)
class_sums_raw = np.zeros((num_classes, feature_len), dtype=np.float64)
class_counts = np.zeros(num_classes, dtype=np.int32)

print("Computing class centroids from training data...")
for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_dir, row["image_id"])
    norm_feat, raw_feat = load_and_preprocess(img_path)
    label = row["label"]
    class_sums_norm[label] += norm_feat
    class_sums_raw[label] += raw_feat
    class_counts[label] += 1

class_means_norm = np.divide(
    class_sums_norm,
    class_counts[:, None],
    out=np.zeros_like(class_sums_norm),
    where=class_counts[:, None] != 0,
)

centroid_norms = np.linalg.norm(class_means_norm, axis=1, keepdims=True)
nonzero = centroid_norms.squeeze() > 0
class_means_norm[nonzero] = class_means_norm[nonzero] / centroid_norms[nonzero]

class_means_raw = np.divide(
    class_sums_raw,
    class_counts[:, None],
    out=np.zeros_like(class_sums_raw),
    where=class_counts[:, None] != 0,
)

print(
    "Centroids computed for classes:",
    {i: int(cnt) for i, cnt in enumerate(class_counts)},
)



## === cell 5
sample_sub = pd.read_csv(data_path + "sample_submission.csv")

print("Predicting test set labels using combined similarity...")
pred_labels = []
alpha = 0.7  # weight for cosine similarity
beta = 0.3  # weight for Euclidean distance (negative)

for img_id in sample_sub["image_id"]:
    img_path = os.path.join(test_images_dir, img_id)
    norm_feat, raw_feat = load_and_preprocess(img_path)

    cos_sims = np.dot(class_means_norm, norm_feat)

    euc_dists = np.linalg.norm(class_means_raw - raw_feat, axis=1)
    neg_euc = -euc_dists

    scores = alpha * cos_sims + beta * neg_euc
    pred_label = int(np.argmax(scores))
    pred_labels.append(pred_label)

submission = pd.DataFrame({"image_id": sample_sub["image_id"], "label": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)



## === cell 6
print(f"Submission file written to '{submission_path}'. First 5 rows:")
print(submission.head())
