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

0.7586884255061952

# 6. Current score

0.29671

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'Implemented fixes to eliminate TensorFlow import errors, removed unavailable model loading, and replaced model predictions with a simple majority‑class baseline derived from the training labels. Added safe handling for missing test images and ensured the script writes a correctly formatted `submission.csv` using the sample submission layout.'
- What this solution (achieved 0.18087) has done: 'I replace the majority‑class baseline with a very lightweight image‑based classifier: for each disease class I compute the average (centroid) image from the training set (resized to a small 32×32 size). Then, for every test image I assign the label of the nearest centroid using Euclidean distance. This keeps the original simple pipeline while adding a modest, data‑driven improvement that should move the accuracy from 0.61 toward the target 0.758.'
- What this solution (achieved 0.36584) has done: 'The changes increase the image resolution used for the class centroids from 32 to 64 pixels and apply per‑image standardization (zero‑mean, unit‑variance) before averaging and distance calculation. This provides richer visual detail and a more consistent feature scale, which should raise the classification accuracy and move the score closer to the target while keeping the original centroid‑based logic unchanged.'
- What this solution (achieved 0.20665) has done: 'The update keeps the same centroid‑based nearest‑neighbor approach but improves the representation: images are resized to 128 × 128 for richer detail, each class centroid is taken as the per‑pixel median (less blur than a mean), and predictions use cosine similarity (which works better with the standardized vectors) instead of Euclidean distance. These minimal changes are expected to raise the validation accuracy toward the target score while still producing a correct `submission.csv`.'
- What this solution (achieved 0.29671) has done: 'I replace the per‑class median aggregation with a mean (which is less lossy for image data) and add a tiny safety fallback: if cosine similarity is non‑positive the prediction defaults to the majority class. These tiny tweaks keep the overall centroid‑based approach unchanged while expected to raise the validation accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import cv2



## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
label_json_path = (
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
)
train_images_dir = "../input/cassava-leaf-disease-classification/train_images"
test_images_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_sub_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"



## === cell 2
train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype("string")
label_map = pd.read_json(label_json_path, orient="index").values.flatten().tolist()



## === cell 3
IMG_SIZE = 128
majority_label = train_df["label"].mode()[0]



## === cell 4
centroids = {}
for cls in train_df["label"].unique():
    cls_ids = train_df[train_df["label"] == cls].image_id.values
    img_stack = []
    for img_id in cls_ids:
        img_path = os.path.join(train_images_dir, str(img_id))
        img = cv2.imread(img_path)
        if img is None:
            continue
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
        mean = img.mean(axis=(0, 1), keepdims=True)
        std = img.std(axis=(0, 1), keepdims=True)
        std = np.where(std < 1e-6, 1.0, std)
        img = (img - mean) / std
        img_stack.append(img)
    if img_stack:  # safety check
        centroids[cls] = np.mean(np.stack(img_stack), axis=0)



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
preds = []
for img_id in sample_sub.image_id:
    img_path = os.path.join(test_images_dir, str(img_id))
    img = cv2.imread(img_path)
    if img is None:
        preds.append(majority_label)
        continue
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE)).astype(np.float32) / 255.0
    mean = img.mean(axis=(0, 1), keepdims=True)
    std = img.std(axis=(0, 1), keepdims=True)
    std = np.where(std < 1e-6, 1.0, std)
    img = (img - mean) / std
    img_vec = img.ravel()
    img_norm = np.linalg.norm(img_vec)
    best_lbl = None
    best_sim = -np.inf  # cosine similarity ranges [-1, 1]
    for cls, cent in centroids.items():
        cent_vec = cent.ravel()
        cent_norm = np.linalg.norm(cent_vec)
        if img_norm == 0 or cent_norm == 0:
            continue
        sim = np.dot(img_vec, cent_vec) / (img_norm * cent_norm)
        if sim > best_sim:
            best_sim = sim
            best_lbl = cls
    if best_lbl is None or best_sim <= 0:
        preds.append(majority_label)
    else:
        preds.append(best_lbl)

submission = pd.DataFrame({"image_id": sample_sub.image_id, "label": preds})
submission.to_csv("submission.csv", index=False)



## === cell 6
print("Submission file created (first few rows):")
print(submission.head())
