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

0.8931701420368692

# 6. Current score

0.61547

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I remove the unused tensorflow_hub import that crashes on load, replace the missing TFSMLayer model loading with a single EfficientNetB0 classifier built, trained, and used for inference, and update the prediction cell to use this model. This fixes the import error, eliminates the “model not defined” issue, and creates a valid submission.csv while preserving the original data handling logic.'
- What this solution (achieved 0.6207) has done: 'I remove the TensorFlow imports that cause protobuf errors and replace the deep‑learning pipeline with a lightweight scikit‑learn model that uses down‑scaled image pixels as features. The script now loads the training images, encodes the labels, trains a RandomForest classifier, evaluates it on a validation split, predicts the test set, and writes a proper `submission.csv`. This fixes the runtime crashes and raises the validation accuracy substantially, moving the score toward the target.'
- What this solution (achieved 0.61734) has done: 'I increase the image resolution to capture more detail and replace the RandomForest with a stronger ExtraTrees classifier (more trees, limited feature sampling). These tweaks keep the overall pipeline unchanged while giving the model a better chance to close the gap toward the target accuracy.'
- What this solution (achieved 0.61286) has done: 'The changes focus on speeding up the heavy image‑loading and tree‑training steps while keeping the same overall workflow. Image loading now uses a process pool (avoiding the GIL) and pre‑allocates the size list once, which cuts I/O overhead. The ExtraTrees classifier keeps the same algorithm but uses fewer trees (300) and the default feature‑subsampling (`max_features='sqrt'`), dramatically reducing training time without altering the model type. Minor deterministic seeds are added to guarantee reproducible results.'
- What this solution (achieved 0.61584) has done: 'I replace the raw‑pixel flattening with a lightweight colour‑histogram + mean/std feature extractor (still using parallel loading) and apply PCA to keep the most informative components before training the ExtraTrees model. This richer yet low‑dimensional representation typically boosts classification accuracy on image data while keeping the overall pipeline unchanged, moving the validation score closer to the target.'
- What this solution (achieved 0.61547) has done: 'I keep the overall pipeline unchanged but improve the feature representation and model capacity:  
- Increase the image resize to (160,160) to capture more detail.  
- Remove dimensionality reduction by setting PCA to keep all 54 features (no information loss).  
- Strengthen the ExtraTrees model by using 800 trees and max_features='log2', which often yields higher accuracy on compact features.  
These minimal tweaks are expected to raise validation accuracy, moving the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from PIL import Image
import concurrent.futures
import random

np.random.seed(42)
random.seed(42)



## === cell 1
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
test_images_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

train_df = pd.read_csv(train_csv_path)

le = LabelEncoder()
train_df["label_enc"] = le.fit_transform(train_df["label"])


def _extract_features(img_path, size):
    """
    Load an image, resize, and compute a compact feature vector:
      - mean and std for each RGB channel (6 values)
      - 16‑bin histogram per channel (48 values, normalized)
    Returns a 54‑dim float32 vector.
    """
    img = Image.open(img_path).convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32) / 255.0  # normalize to [0,1]

    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))

    hist_features = []
    for c in range(3):
        hist, _ = np.histogram(arr[:, :, c], bins=16, range=(0.0, 1.0), density=True)
        hist_features.append(hist)
    hist_features = np.concatenate(hist_features)

    return np.concatenate([means, stds, hist_features])


def load_features(df, img_dir, size=(160, 160)):
    """Parallel loading of images and extraction of compact features."""
    img_paths = [os.path.join(img_dir, img_id) for img_id in df["image_id"]]
    n_images = len(img_paths)
    X = np.empty((n_images, 54), dtype=np.float32)  # 54 = 6 + 48

    size_args = [size] * n_images
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for idx, feats in enumerate(
            executor.map(_extract_features, img_paths, size_args)
        ):
            X[idx] = feats
    return X


X = load_features(train_df, train_images_dir, size=(160, 160))
y = train_df["label_enc"].values

pca = PCA(n_components=54, svd_solver="full", random_state=42)
X_reduced = pca.fit_transform(X)

X_train, X_valid, y_train, y_valid = train_test_split(
    X_reduced, y, test_size=0.2, stratify=y, random_state=42
)

et = ExtraTreesClassifier(
    n_estimators=800,  # more trees for better performance
    max_features="log2",  # alternative feature sub‑sampling
    class_weight="balanced",
    n_jobs=-1,
    random_state=42,
)
et.fit(X_train, y_train)

valid_pred = et.predict(X_valid)
val_acc = accuracy_score(y_valid, valid_pred)
print(f"Validation accuracy: {val_acc:.4f}")

test_filenames = sorted(
    [f for f in os.listdir(test_images_dir) if f.lower().endswith(".jpg")]
)

test_df = pd.DataFrame({"image_id": test_filenames})
X_test = load_features(test_df, test_images_dir, size=(160, 160))
X_test_reduced = pca.transform(X_test)

test_pred_enc = et.predict(X_test_reduced)
test_pred_labels = le.inverse_transform(test_pred_enc)

submission = pd.DataFrame(
    {"image_id": test_filenames, "label": test_pred_labels.astype(int)}
)

submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission.head())
