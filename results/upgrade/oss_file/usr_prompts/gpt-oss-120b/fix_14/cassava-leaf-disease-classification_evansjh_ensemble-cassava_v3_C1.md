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

0.890601390148081

# 6. Current score

0.213

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'The fix removes the failing TensorFlow imports and model loading, replaces them with a simple majority‑class baseline (the most frequent label in the training set) to ensure the script runs end‑to‑end and writes a correctly formatted `submission.csv`. This resolves the import error, missing model files, and undefined‑variable crashes while still producing a valid submission file.'
- What this solution (achieved 0.13004) has done: 'I replace the simple majority‑class baseline with a very lightweight centroid classifier that uses the average RGB colour of each image (scaled to 32 × 32). For each disease class we compute a colour centroid from the training images, then assign each test image the label of the nearest centroid. This adds only a small feature‑extraction step and modest computation, but typically raises accuracy well above the 0.61 baseline, moving the score toward the target 0.89 while keeping the script simple and self‑contained.'
- What this solution (achieved 0.51084) has done: 'The fix adds a robust path‑resolution step that checks several common locations (e.g., the Kaggle input folder) and falls back gracefully if the CSV or image directories are missing. This prevents the FileNotFoundError and ensures the script always produces a correctly‑named `submission.csv`. The core feature extraction and nearest‑centroid prediction logic remain unchanged, preserving the original model while allowing it to run end‑to‑end.'
- What this solution (achieved 0.2003) has done: 'I replace the per‑image nearest‑neighbor lookup with a lightweight class‑centroid classifier: each disease class’s average colour‑histogram is pre‑computed from the training set, and test images are assigned the label of the nearest centroid. This retains the original histogram feature extraction while providing a more stable and discriminative decision rule, which should raise the accuracy toward the target score.'
- What this solution (achieved 0.51084) has done: 'I replace the centroid‑based rule with a straightforward 1‑Nearest‑Neighbour classifier that compares each test image’s histogram vector to all training vectors and picks the label of the closest one. This keeps the same feature extraction and data loading logic but uses a more discriminative decision rule, which should raise accuracy toward the target while still producing a valid `submission.csv`.'
- What this solution (achieved 0.2003) has done: 'I replace the per‑image 1‑Nearest‑Neighbour lookup with a lightweight class‑centroid classifier that still uses the same colour‑histogram features. After computing the mean feature vector for each disease class, each test image is assigned the label of the nearest centroid. This small change keeps the core feature extraction unchanged while providing a more robust decision rule, which should raise the validation accuracy and move the current score (0.51) closer to the target (0.89).'
- What this solution (achieved 0.51084) has done: 'I replace the centroid‑based rule with a simple 1‑Nearest‑Neighbour classifier that compares each test image’s histogram feature to all training‑image features and assigns the label of the closest neighbour. This keeps the original feature extraction unchanged, adds only a lightweight distance computation, and is expected to raise the accuracy from the current 0.20 → ≈0.5‑0.6, moving the score toward the target 0.89.'
- What this solution (achieved 0.2003) has done: 'We replace the exhaustive 1‑nearest‑neighbour search with a lightweight class‑centroid classifier: after extracting colour‑histogram features for the training set we compute the mean feature vector for each label and, for each test image, assign the label of the nearest centroid. This keeps the original feature extraction, adds only a few lines, and typically raises validation accuracy, moving the current 0.51 score closer to the 0.89 target while still writing a correct `submission.csv`.'
- What this solution (achieved 0.51084) has done: 'I replace the centroid‑based decision rule with a straightforward 1‑Nearest‑Neighbour classifier that uses the same colour‑histogram features. This keeps the feature extraction unchanged while providing a more discriminative prediction, which should raise the validation accuracy from the current 0.20 → ≈0.55 and move the score closer to the target. The rest of the script (paths, feature extraction, CSV handling) remains unchanged.'
- What this solution (achieved 0.21599) has done: 'I switch the classifier from a full 1‑nearest‑neighbour lookup to a lightweight nearest‑centroid rule and increase the colour histogram resolution (bins = 32). After extracting features for the training images we compute a mean feature vector for each disease class and, for each test image, assign the label of the closest centroid. This keeps the same feature extraction pipeline while providing a more stable decision rule and richer features, which should raise the validation accuracy from ~0.51 toward the target 0.89.'
- What this solution (achieved 0.213) has done: 'I enrich the image representation by enlarging the resize dimensions to 128 × 128 and using a finer colour‑histogram with 64 bins (instead of 64×64 / 32 bins). This provides a more discriminative feature while leaving the overall centroid‑based classifier unchanged, so the script still runs end‑to‑end and the submission format is preserved. The richer features are expected to raise the validation accuracy, moving the current score 0.21599 closer to the target 0.8906.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
from PIL import Image

candidate_dirs = [
    os.path.join("data", "cassava-leaf-disease-classification"),
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
    "cassava-leaf-disease-classification",
    "./cassava-leaf-disease-classification",
]
base_dir = None
for d in candidate_dirs:
    if os.path.isdir(d):
        base_dir = d
        break
if base_dir is None:
    raise RuntimeError("Could not locate the dataset directory.")

train_csv_path = os.path.join(base_dir, "train.csv")
train_image_dir = os.path.join(base_dir, "train_images")
test_image_dir = os.path.join(base_dir, "test_images")
submission_path = "submission.csv"




## === cell 1
def extract_feature(img_path, size=(128, 128), bins=64):
    """
    Load an image, resize, and return a concatenated colour histogram.
    Returns a (3*bins,) float32 vector or None on failure.
    """
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            im = im.resize(size)
            arr = np.asarray(im, dtype=np.float32) / 255.0  # normalise to [0, 1]
            hist_features = []
            for ch in range(3):
                h, _ = np.histogram(
                    arr[:, :, ch], bins=bins, range=(0, 1), density=True
                )
                hist_features.append(h.astype(np.float32))
            return np.concatenate(hist_features)  # shape (3*bins,)
    except Exception:
        return None


if not os.path.exists(train_csv_path):
    raise FileNotFoundError(f"Training CSV not found at {train_csv_path}")

train_df = pd.read_csv(train_csv_path)

most_common_label = int(train_df["label"].mode().iloc[0]) if not train_df.empty else 0

train_features_list = []
train_labels_list = []

for _, row in train_df.iterrows():
    img_name = row["image_id"]
    label = int(row["label"])
    img_path = os.path.join(train_image_dir, img_name)
    feat = extract_feature(img_path)
    if feat is not None:
        train_features_list.append(feat)
        train_labels_list.append(label)

if train_features_list:
    train_features = np.stack(train_features_list)  # (N_train, 3*bins)
    train_labels = np.array(train_labels_list, dtype=np.int32)
else:
    train_features = np.empty((0, 3 * 64), dtype=np.float32)
    train_labels = np.empty((0,), dtype=np.int32)

num_classes = 5
feature_dim = train_features.shape[1]
centroids = np.empty((num_classes, feature_dim), dtype=np.float32)
centroid_mask = np.zeros(num_classes, dtype=bool)  # True where centroid is valid

for lbl in range(num_classes):
    mask = train_labels == lbl
    if np.any(mask):
        centroids[lbl] = train_features[mask].mean(axis=0)
        centroid_mask[lbl] = True
centroids[~centroid_mask] = np.nan




## === cell 2
if not os.path.isdir(test_image_dir):
    raise FileNotFoundError(f"Test image directory not found at {test_image_dir}")

test_images = sorted(
    [f for f in os.listdir(test_image_dir) if f.lower().endswith((".jpg", ".png"))]
)

predictions = []

if train_features.shape[0] > 0 and np.any(centroid_mask):
    valid_labels = np.where(centroid_mask)[0]
    valid_centroids = centroids[centroid_mask]  # shape (C_valid, dim)

    for img_name in test_images:
        img_path = os.path.join(test_image_dir, img_name)
        feat = extract_feature(img_path)
        if feat is None:
            pred = most_common_label
        else:
            dists = np.sum((valid_centroids - feat) ** 2, axis=1)
            nearest_idx = np.argmin(dists)
            pred = int(valid_labels[nearest_idx])
        predictions.append(pred)
else:
    predictions = [most_common_label] * len(test_images)

submission_df = pd.DataFrame({"image_id": test_images, "label": predictions})
submission_df.to_csv(submission_path, index=False)

print(f"Submission written to {submission_path} with {len(submission_df)} rows.")
