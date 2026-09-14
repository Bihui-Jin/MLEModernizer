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
import numpy as np
import pandas as pd
import concurrent.futures  # added for parallel feature extraction
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier, ExtraTreesClassifier
from sklearn.metrics import accuracy_score
from PIL import Image

base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_images_path = os.path.join(base_path, "train_images")
test_images_path = os.path.join(base_path, "test_images")

train_df = pd.read_csv(train_csv_path)

label_encoder = LabelEncoder()
train_df["label_enc"] = label_encoder.fit_transform(train_df["label"])


def extract_features(img_path):
    """
    Feature vector:
    1) Flattened 64×64 RGB pixels (normalized)                     -> 12288
    2) Mean, std, min, max, skew for each RGB channel            -> 15
    3) Overall grayscale mean                                    -> 1
    4) 16‑bin HSV histogram per channel                         -> 48
    5) 16‑bin RGB histogram per channel                         -> 48
    6) Edge‑density (fraction of strong gradients)              -> 1
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((64, 64))
    rgb_arr = np.asarray(img).astype(np.float32) / 255.0  # (64,64,3)

    flat = rgb_arr.flatten()

    means = rgb_arr.mean(axis=(0, 1))
    stds = rgb_arr.std(axis=(0, 1))
    mins = rgb_arr.min(axis=(0, 1))
    maxs = rgb_arr.max(axis=(0, 1))
    diffs = rgb_arr - means
    skew = np.mean(diffs**3, axis=(0, 1))

    gray = (
        0.2989 * rgb_arr[:, :, 0]
        + 0.5870 * rgb_arr[:, :, 1]
        + 0.1140 * rgb_arr[:, :, 2]
    )
    gray_mean = gray.mean()

    hsv_img = img.convert("HSV")
    hsv_arr = np.asarray(hsv_img).astype(np.float32) / 255.0
    hsv_hist = []
    for ch in range(3):
        hist, _ = np.histogram(
            hsv_arr[:, :, ch], bins=16, range=(0.0, 1.0), density=True
        )
        hsv_hist.append(hist)
    hsv_hist = np.concatenate(hsv_hist)

    rgb_hist = []
    for ch in range(3):
        hist, _ = np.histogram(
            rgb_arr[:, :, ch], bins=16, range=(0.0, 1.0), density=True
        )
        rgb_hist.append(hist)
    rgb_hist = np.concatenate(rgb_hist)

    gx, gy = np.gradient(gray)
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_density = (grad_mag > 0.1).mean()

    return np.concatenate(
        [
            flat,
            means,
            stds,
            mins,
            maxs,
            skew,
            [gray_mean],
            hsv_hist,
            rgb_hist,
            [edge_density],
        ]
    )


def compute_features(image_paths):
    """Parallel extraction keeping input order; returns float32 array."""
    with concurrent.futures.ProcessPoolExecutor() as executor:
        feats = list(executor.map(extract_features, image_paths))
    return np.stack(feats, axis=0).astype(np.float32)


train_image_paths = [
    os.path.join(train_images_path, img_name) for img_name in train_df["image_id"]
]
train_features = compute_features(train_image_paths)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_features,
    train_df["label_enc"].values,
    test_size=0.2,
    stratify=train_df["label_enc"],
    random_state=42,
)

rf = RandomForestClassifier(
    n_estimators=2000,
    max_depth=30,
    class_weight="balanced",
    n_jobs=4,
    random_state=42,
)
et = ExtraTreesClassifier(
    n_estimators=2000,
    max_depth=30,
    class_weight="balanced",
    n_jobs=4,
    random_state=42,
)

rf.fit(X_train, y_train)
et.fit(X_train, y_train)

rf_prob = rf.predict_proba(X_valid)
et_prob = et.predict_proba(X_valid)
avg_prob = (rf_prob + et_prob) / 2.0
valid_pred = np.argmax(avg_prob, axis=1)

print("Validation accuracy:", accuracy_score(y_valid, valid_pred))




## === cell 1
test_filenames = [f for f in os.listdir(test_images_path) if f.lower().endswith(".jpg")]
test_paths = [os.path.join(test_images_path, f) for f in test_filenames]

test_features = compute_features(test_paths)

rf_test_prob = rf.predict_proba(test_features)
et_test_prob = et.predict_proba(test_features)
avg_test_prob = (rf_test_prob + et_test_prob) / 2.0
test_pred_enc = np.argmax(avg_test_prob, axis=1)
test_pred = label_encoder.inverse_transform(test_pred_enc)

submission = pd.DataFrame({"image_id": test_filenames, "label": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file created at {submission_path}")
print(submission.head())
