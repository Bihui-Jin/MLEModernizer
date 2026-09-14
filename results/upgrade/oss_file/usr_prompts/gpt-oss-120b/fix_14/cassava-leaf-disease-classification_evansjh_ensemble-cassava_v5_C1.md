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

0.8895436687821094

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I remove the failing TensorFlow model loading and replace it with a simple baseline that predicts the most frequent label from the training set for every test image. This fixes the import and file‑not‑found errors, guarantees a valid `submission.csv` with the correct columns, and lets the notebook run end‑to‑end. The core logic of reading files, building the submission DataFrame, and saving it remains unchanged.'
- What this solution (achieved 0.61809) has done: 'I replace the naive majority‑class predictor with a very lightweight image‑based model.  
The script now extracts simple colour‑histogram features from each training image, trains a small RandomForest classifier (validated with a train/validation split), and uses this model to predict labels for the test set. All original I/O paths and the final CSV‑writing logic remain unchanged, so the workflow still produces a valid `submission.csv`, but the expected accuracy should move much closer to the target score.'
- What this solution (achieved 0.61734) has done: 'I slightly enrich the image features (add per‑channel standard deviation and use finer histograms) and give the RandomForest a bit more capacity (more trees and a reasonable depth). These changes keep the overall pipeline identical while providing the model with more discriminative information, which should raise validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.62294) has done: 'I enhance the feature extractor to include HSV histograms and a simple edge‑density measure, increase the image size for richer detail, and strengthen the ensemble by using an ExtraTreesClassifier with more trees and unrestricted depth. These modest tweaks keep the overall pipeline unchanged while providing the model with more discriminative information, which should raise the validation accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.62556) has done: 'The changes replace the heavy process‑based parallel feature extraction with a thread‑based approach that avoids costly pickling of large NumPy arrays, pre‑allocate the feature matrix, and fill it in place. This keeps the exact same feature logic and model configuration while dramatically reducing overhead, allowing the whole pipeline to complete well within the 600‑second limit.'
- What this solution (achieved 0.62593) has done: 'I added a few lightweight, discriminative features (per‑channel min and max values) to the existing colour‑based extractor, and increased the number of trees in the ExtraTrees model slightly. These changes keep the same overall pipeline and model type while giving the classifier a bit more information, which should raise validation accuracy and move the Kaggle score closer to the target.'
- What this solution (achieved 0.61099) has done: 'I added the missing data loading, path definitions, a simple yet effective `extract_features` function (RGB + HSV histograms plus channel statistics), defined the classifier (`ExtraTreesClassifier`), and completed the workflow to build the feature matrices, train the model, predict on the test set and write a proper `submission.csv`. These fixes resolve the NameError issues and produce a valid submission while modestly improving accuracy toward the target score. The core pipeline and model type remain unchanged.'
- What this solution (achieved 0.11024) has done: 'I enrich the feature extractor by adding per‑channel minimum and maximum values (giving the model more discriminative signal) and increase the tree count in the ExtraTrees classifier to improve its capacity. These small, targeted tweaks keep the overall pipeline unchanged while expected to raise validation accuracy and thus move the score closer to the target.'
- What this solution (achieved 0.11024) has done: 'I improve the feature extractor by using a larger image resolution (128 × 128) and finer 32‑bin colour histograms, which provides more discriminative information while keeping the same overall pipeline. I also increase the number of trees in the ExtraTrees model to give it higher capacity. These small, targeted tweaks are expected to raise the validation accuracy and move the Kaggle score closer to the target without changing any core logic.'

# 9. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = (
    "1"  # limit NumPy/BLAS threads to avoid oversubscription
)

import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.metrics import accuracy_score
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor  # keep process‑based pool

base_path = Path("cassava-leaf-disease-classification")
train_csv_path = base_path / "train.csv"
train_image_dir = base_path / "train_images"
test_image_dir = base_path / "test_images"

train_df = pd.read_csv(train_csv_path)


def extract_features(img_path: str) -> np.ndarray:
    """Extract colour‑histogram and channel statistics from an image.

    Uses a larger 128×128 resize and 32‑bin histograms for finer detail.
    """
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return np.zeros(228, dtype=np.float32)  # match the feature length below

    img = img.resize((128, 128))
    arr = np.array(img)  # shape (128,128,3)

    rgb_hist = []
    for ch in range(3):
        hist, _ = np.histogram(arr[:, :, ch], bins=32, range=(0, 256), density=True)
        rgb_hist.extend(hist)

    hsv_img = img.convert("HSV")
    hsv_arr = np.array(hsv_img)
    hsv_hist = []
    for ch in range(3):
        hist, _ = np.histogram(hsv_arr[:, :, ch], bins=32, range=(0, 256), density=True)
        hsv_hist.extend(hist)

    means = arr.mean(axis=(0, 1))
    stds = arr.std(axis=(0, 1))
    mins = arr.min(axis=(0, 1))
    maxs = arr.max(axis=(0, 1))

    features = np.concatenate([rgb_hist, hsv_hist, means, stds, mins, maxs]).astype(
        np.float32
    )
    return features


train_paths = []
train_labels = []

for _, row in train_df.iterrows():
    img_path = train_image_dir / row["image_id"]
    if img_path.is_file():
        train_paths.append(str(img_path))
        train_labels.append(int(row["label"]))

sample_feat = extract_features(train_paths[0])
feat_len = sample_feat.shape[0]

train_features = np.empty((len(train_paths), feat_len), dtype=np.float32)


def _store(idx_path):
    idx, path = idx_path
    train_features[idx] = extract_features(path)


def _worker_init():
    os.environ["OMP_NUM_THREADS"] = "1"


with ProcessPoolExecutor(
    max_workers=os.cpu_count(), initializer=_worker_init
) as executor:
    executor.map(_store, enumerate(train_paths), chunksize=256)

X = train_features
y = np.array(train_labels, dtype=np.int32)




## === cell 1
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=42, stratify=y
)

model = ExtraTreesClassifier(
    n_estimators=1200,  # increased number of trees for higher capacity
    max_depth=None,
    min_samples_split=2,
    min_samples_leaf=1,
    bootstrap=False,
    n_jobs=1,  # respect the OMP_NUM_THREADS limit
    random_state=42,
    class_weight="balanced",
)

model.fit(X_train, y_train)

val_preds = model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.4f}")

model.fit(X, y)

test_image_names = [
    fn
    for fn in os.listdir(test_image_dir)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
]

test_paths = [str(test_image_dir / fn) for fn in test_image_names]

test_features = np.empty((len(test_paths), feat_len), dtype=np.float32)


def _store_test(idx_path):
    idx, path = idx_path
    test_features[idx] = extract_features(path)


with ProcessPoolExecutor(
    max_workers=os.cpu_count(), initializer=_worker_init
) as executor:
    executor.map(_store_test, enumerate(test_paths), chunksize=256)

test_preds = model.predict(test_features).astype(int)

submission_df = pd.DataFrame({"image_id": test_image_names, "label": test_preds})

submission_path = Path("submission.csv")
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path.resolve()}")
