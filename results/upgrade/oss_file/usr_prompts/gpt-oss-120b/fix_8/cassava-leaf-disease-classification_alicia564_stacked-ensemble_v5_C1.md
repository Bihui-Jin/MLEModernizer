# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.8961922030825022

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61996) has done: 'I replace the failing TensorFlow pipeline with a lightweight image‑feature extractor (resize to 32×32 RGB) and a scikit‑learn classifier. This avoids the protobuf error, correctly builds train/validation splits, trains a simple model, and writes a properly formatted `submission.csv` so the notebook runs end‑to‑end. The changes are minimal to the surrounding workflow and keep the overall training‑validation‑prediction structure.'
- What this solution (achieved 0.61996) has done: 'I enhance the feature extractor by appending per‑channel mean and standard‑deviation statistics to the flattened 32×32 RGB vector, and increase the RandomForest size to give the model more capacity. These small, targeted changes keep the overall pipeline unchanged while providing richer information that should raise validation accuracy and move the score closer to the target.'
- What this solution (achieved 0.61921) has done: 'I enrich the image feature vector with per‑channel minimum and maximum values plus an overall grayscale mean, and slightly increase the RandomForest capacity (more trees and a limited depth) so the model can use the extra information without over‑fitting. These minimal tweaks keep the overall pipeline unchanged while giving the classifier stronger signals, which should raise validation accuracy and move the score closer to the target.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
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
    Return a feature vector composed of:
    1) Flattened 32×32 RGB pixels (normalized)
    2) Mean, std, min, max for each RGB channel
    3) Overall grayscale mean (brightness)
    4) 16‑bin HSV colour histogram for each channel (48 values)
    5) Edge‑density (fraction of strong gradients)
    These inexpensive statistics give the model richer colour and texture cues.
    """
    img = Image.open(img_path).convert("RGB")
    img = img.resize((32, 32))
    rgb_arr = np.asarray(img).astype(np.float32) / 255.0  # (32,32,3)

    flat = rgb_arr.flatten()

    means = rgb_arr.mean(axis=(0, 1))  # (3,)
    stds = rgb_arr.std(axis=(0, 1))  # (3,)
    mins = rgb_arr.min(axis=(0, 1))  # (3,)
    maxs = rgb_arr.max(axis=(0, 1))  # (3,)

    gray = (
        0.2989 * rgb_arr[:, :, 0]
        + 0.5870 * rgb_arr[:, :, 1]
        + 0.1140 * rgb_arr[:, :, 2]
    )
    gray_mean = gray.mean()

    hsv_img = img.convert("HSV")
    hsv_arr = np.asarray(hsv_img).astype(np.float32) / 255.0  # (32,32,3) in [0,1]
    hist_features = []
    for ch in range(3):
        hist, _ = np.histogram(
            hsv_arr[:, :, ch], bins=16, range=(0.0, 1.0), density=True
        )
        hist_features.append(hist)
    hist_features = np.concatenate(hist_features)  # 48 values

    sobel_x = np.array([[1, 0, -1], [2, 0, -2], [1, 0, -1]], dtype=np.float32)
    sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
    gx = np.abs(np.convolve(gray.ravel(), sobel_x.ravel(), mode="valid")).reshape(
        gray.shape[0] - 2, gray.shape[1] - 2
    )
    gy = np.abs(np.convolve(gray.ravel(), sobel_y.ravel(), mode="valid")).reshape(
        gray.shape[0] - 2, gray.shape[1] - 2
    )
    grad_mag = np.sqrt(gx**2 + gy**2)
    edge_thresh = 0.1  # heuristic
    edge_density = (grad_mag > edge_thresh).mean()

    return np.concatenate(
        [flat, means, stds, mins, maxs, [gray_mean], hist_features, [edge_density]]
    )


train_features = []
for img_name in train_df["image_id"]:
    img_path = os.path.join(train_images_path, img_name)
    train_features.append(extract_features(img_path))
train_features = np.stack(train_features, axis=0)

X_train, X_valid, y_train, y_valid = train_test_split(
    train_features,
    train_df["label_enc"].values,
    test_size=0.2,
    stratify=train_df["label_enc"],
    random_state=42,
)

clf = RandomForestClassifier(
    n_estimators=1000,  # slightly more trees for stability
    max_depth=30,
    n_jobs=-1,
    random_state=42,
)
clf.fit(X_train, y_train)

valid_pred = clf.predict(X_valid)
print("Validation accuracy:", accuracy_score(y_valid, valid_pred))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1724829452.py in <cell line: 0>()
     86 for img_name in train_df["image_id"]:
     87     img_path = os.path.join(train_images_path, img_name)
---> 88     train_features.append(extract_features(img_path))
     89 train_features = np.stack(train_features, axis=0)
     90 

/tmp/ipykernel_11/1724829452.py in extract_features(img_path)
     66     sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
     67     # apply on grayscale
---> 68     gx = np.abs(np.convolve(gray.ravel(), sobel_x.ravel(), mode="valid")).reshape(
     69         gray.shape[0] - 2, gray.shape[1] - 2
     70     )

ValueError: cannot reshape array of size 1016 into shape (30,30)

## === cell 1
test_filenames = [f for f in os.listdir(test_images_path) if f.lower().endswith(".jpg")]
test_paths = [os.path.join(test_images_path, f) for f in test_filenames]

test_features = []
for p in test_paths:
    test_features.append(extract_features(p))
test_features = np.stack(test_features, axis=0)

test_pred_enc = clf.predict(test_features)
test_pred = label_encoder.inverse_transform(test_pred_enc)

submission = pd.DataFrame({"image_id": test_filenames, "label": test_pred})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)

print(f"Submission file created at {submission_path}")
print(submission.head())

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2877948413.py in <cell line: 0>()
      4 test_features = []
      5 for p in test_paths:
----> 6     test_features.append(extract_features(p))
      7 test_features = np.stack(test_features, axis=0)
      8 

/tmp/ipykernel_11/1724829452.py in extract_features(img_path)
     66     sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]], dtype=np.float32)
     67     # apply on grayscale
---> 68     gx = np.abs(np.convolve(gray.ravel(), sobel_x.ravel(), mode="valid")).reshape(
     69         gray.shape[0] - 2, gray.shape[1] - 2
     70     )

ValueError: cannot reshape array of size 1016 into shape (30,30)
