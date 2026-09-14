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

0.8768510123904503

# 6. Current score

0.25747

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.463) has done: 'I adjust the data‑folder path detection so the script works in the Kaggle environment (using `/kaggle/input/...` when present) and switch to a 1‑nearest‑neighbor rule, which is a minimal change that often improves simple color‑based classification accuracy. The rest of the pipeline stays unchanged, and the script now reliably writes a `submission.csv` file.'
- What this solution (achieved 0.25747) has done: 'We replace the original 1‑nearest‑neighbor search with a far more stable “nearest‑centroid” classifier: each test image is compared to the six‑dimensional normalized color‑mean/std centroids of the five classes and the label of the closest centroid is taken. This reduces noise from individual training samples and is expected to raise the accuracy, moving the score closer to the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.463) has done: 'I replace the centroid‑based classification with a simple 1‑nearest‑neighbor search using the same normalized color‑mean/std features. This keeps the original feature extraction unchanged while providing a classifier that previously achieved a higher validation score, moving the Kaggle accuracy closer to the target. The rest of the pipeline (path handling, feature computation, CSV writing) remains identical.'
- What this solution (achieved 0.25747) has done: 'I replace the 1‑nearest‑neighbor search with a centroid‑based classifier: each test image’s normalized feature vector is compared to the normalized class centroids, and the label of the closest centroid is used. This change keeps the original feature extraction and preprocessing while reducing noise from individual training samples, which should raise the accuracy toward the target.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from collections import Counter
from PIL import Image



## === cell 1
possible_paths = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "cassava-leaf-disease-classification",
]
for p in possible_paths:
    if os.path.isdir(p):
        project_folder = p
        break
else:
    raise FileNotFoundError(
        "Project folder not found. Tried: " + ", ".join(possible_paths)
    )

train_csv_path = os.path.join(project_folder, "train.csv")
train_images_folder = os.path.join(project_folder, "train_images")
test_folder = os.path.join(project_folder, "test_images")

if not os.path.isfile(train_csv_path):
    raise FileNotFoundError(f"Train CSV not found: {train_csv_path}")
if not os.path.isdir(train_images_folder):
    raise FileNotFoundError(f"Train images folder not found: {train_images_folder}")
if not os.path.isdir(test_folder):
    raise FileNotFoundError(f"Test images folder not found: {test_folder}")

train_df = pd.read_csv(train_csv_path)
train_df["label"] = train_df["label"].astype(int)

most_common_label = Counter(train_df["label"]).most_common(1)[0][0]




## === cell 2
def extract_color_features(image_path, size=(32, 32)):
    """
    Load an image, resize, and return a feature vector consisting of
    per‑channel mean and standard deviation (length 6).
    """
    try:
        with Image.open(image_path) as img:
            img = img.convert("RGB")
            img = img.resize(size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0  # shape (H, W, 3)
            means = arr.mean(axis=(0, 1))  # (3,)
            stds = arr.std(axis=(0, 1))  # (3,)
            return np.concatenate([means, stds])  # (6,)
    except Exception:
        return None




## === cell 3
class_features = {}
all_features = []  # raw (un‑scaled) feature vectors
all_labels = []  # label for each entry in all_features

for _, row in train_df.iterrows():
    img_path = os.path.join(train_images_folder, row["image_id"])
    feat = extract_color_features(img_path)
    if feat is None:
        continue
    lbl = row["label"]
    class_features.setdefault(lbl, []).append(feat)
    all_features.append(feat)
    all_labels.append(lbl)

if not class_features:
    raise RuntimeError("Failed to compute any class features.")

centroids = {lbl: np.mean(feats, axis=0) for lbl, feats in class_features.items()}

all_features_stack = np.vstack(all_features)
global_mean = all_features_stack.mean(axis=0)
global_std = all_features_stack.std(axis=0) + 1e-8  # avoid division by zero

norm_centroids = {
    lbl: (centroid - global_mean) / global_std for lbl, centroid in centroids.items()
}

train_norm_features = (
    all_features_stack - global_mean
) / global_std  # kept for compatibility
train_labels = np.array(all_labels)



## === cell 4
test_files = sorted([f for f in os.listdir(test_folder) if f.lower().endswith(".jpg")])
predictions = []

for img_name in test_files:
    img_path = os.path.join(test_folder, img_name)
    feat = extract_color_features(img_path)
    if feat is None:
        predictions.append(most_common_label)
        continue

    norm_feat = (feat - global_mean) / global_std

    dists = {
        lbl: np.linalg.norm(centroid - norm_feat)
        for lbl, centroid in norm_centroids.items()
    }
    pred_label = int(min(dists, key=dists.get))
    predictions.append(pred_label)



## === cell 5
submission_df = pd.DataFrame({"image_id": test_files, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Saved submission to {submission_path} with {len(submission_df)} rows.")
