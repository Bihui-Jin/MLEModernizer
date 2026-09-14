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
import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor

tf = None
print(
    "TensorFlow is not used; fallback dummy or lightweight sklearn predictor will be employed."
)

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUBMISSION_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
SUBMISSION_PATH = "submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
submission_df = pd.read_csv(SAMPLE_SUBMISSION_CSV)

majority_label = int(train_df["label"].mode().iloc[0])

np.random.seed(42)  # ensure reproducibility for any randomness downstream

MAX_WORKERS = min(8, os.cpu_count() or 1)




## === cell 1
from PIL import Image


def extract_features(image_path):
    """Create a richer feature vector for an image."""
    try:
        with Image.open(image_path) as img:
            img_rgb = img.convert("RGB")
            arr_rgb = np.asarray(img_rgb, dtype=np.float32) / 255.0
            mean_rgb = arr_rgb.mean(axis=(0, 1))
            std_rgb = arr_rgb.std(axis=(0, 1))

            img_hsv = img.convert("HSV")
            arr_hsv = np.asarray(img_hsv, dtype=np.float32) / 255.0
            mean_hsv = arr_hsv.mean(axis=(0, 1))
            std_hsv = arr_hsv.std(axis=(0, 1))

            hue = arr_hsv[..., 0]
            hist, _ = np.histogram(hue, bins=8, range=(0.0, 1.0), density=True)

            features = np.concatenate([mean_rgb, std_rgb, mean_hsv, std_hsv, hist])
            return features
    except Exception:
        return np.zeros(20, dtype=np.float32)




## === cell 2
model = None
try:
    from sklearn.ensemble import (
        RandomForestClassifier,
        ExtraTreesClassifier,
        GradientBoostingClassifier,
    )
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import accuracy_score

    train_image_paths = [
        os.path.join(TRAIN_IMG_DIR, img_name) for img_name in train_df["image_id"]
    ]

    sample_feat = extract_features(train_image_paths[0])
    feat_len = sample_feat.shape[0]

    X_train = np.empty((len(train_image_paths), feat_len), dtype=np.float32)

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        for idx, feat in enumerate(
            executor.map(extract_features, train_image_paths, chunksize=10)
        ):
            X_train[idx] = feat

    y_train = train_df["label"].values

    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.1, stratify=y_train, random_state=42
    )

    candidates = []

    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    candidates.append(("rf", rf))

    et = ExtraTreesClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1,
    )
    candidates.append(("et", et))

    gb = GradientBoostingClassifier(
        n_estimators=200,
        learning_rate=0.1,
        max_depth=3,
        random_state=42,
    )
    candidates.append(("gb", gb))

    lr = LogisticRegression(
        multi_class="multinomial",
        solver="lbfgs",
        max_iter=200,
        n_jobs=1,
        random_state=42,
    )
    candidates.append(("lr", lr))

    best_name, best_model = None, None
    best_score = -1.0

    for name, clf in candidates:
        clf.fit(X_tr, y_tr)
        if hasattr(clf, "predict"):
            preds = clf.predict(X_val)
        else:
            preds = np.argmax(clf.predict_proba(X_val), axis=1)
        acc = accuracy_score(y_val, preds)
        if acc > best_score:
            best_score = acc
            best_name = name
            best_model = clf

    best_model.fit(X_train, y_train)
    model = best_model
    print(f"Selected model '{best_name}' with validation accuracy {best_score:.4f}")

except Exception as e_main:
    print(f"Model training failed ({e_main}); falling back to dummy predictor.")

    class DummyModel:
        def __init__(self, pred_class, num_classes=5):
            self.pred_class = pred_class
            self.num_classes = num_classes

        def predict_proba(self, X):
            batch_size = X.shape[0]
            probs = np.zeros((batch_size, self.num_classes), dtype=np.float32)
            probs[np.arange(batch_size), self.pred_class] = 1.0
            return probs

    model = DummyModel(pred_class=majority_label)




## === cell 3
test_ids = submission_df["image_id"].values
test_image_paths = [os.path.join(TEST_IMG_DIR, img_name) for img_name in test_ids]

X_test = np.empty((len(test_image_paths), feat_len), dtype=np.float32)

with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
    for idx, feat in enumerate(
        executor.map(extract_features, test_image_paths, chunksize=10)
    ):
        X_test[idx] = feat

if hasattr(model, "predict"):
    pred_labels = model.predict(X_test)
elif hasattr(model, "predict_proba"):
    pred_labels = np.argmax(model.predict_proba(X_test), axis=1)
else:
    pred_labels = np.full(len(test_ids), majority_label, dtype=int)

submission_output = pd.DataFrame(
    {"image_id": test_ids, "label": pred_labels.astype(int)}
)
submission_output.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")
