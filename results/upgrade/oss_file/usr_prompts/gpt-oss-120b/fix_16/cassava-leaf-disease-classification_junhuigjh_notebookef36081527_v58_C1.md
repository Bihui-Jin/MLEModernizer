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
import pandas as pd
import numpy as np
from PIL import Image
import torch
from concurrent.futures import ThreadPoolExecutor
import warnings

warnings.filterwarnings("ignore", category=FutureWarning)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images"
test_image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

train_df = pd.read_csv(train_csv_path)




## === cell 1
def image_stats(image_path):
    """Return an enriched set of colour and texture statistics for an image.
    Features (27‑dim):
    - RGB: mean, std, median (9)
    - ratios: G/R, G/B (2)
    - brightness (mean of RGB means) (1)
    - HSV: mean, std (6)
    - LAB: mean, std (6)
    - Gradient magnitude: mean, std (2)
    """
    try:
        img = Image.open(image_path).convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0  # HxWx3
        mean = arr.mean(axis=(0, 1))  # (3,)
        std = arr.std(axis=(0, 1))
        median = np.median(arr, axis=(0, 1))
        eps = 1e-6
        ratio_g_r = mean[1] / (mean[0] + eps)
        ratio_g_b = mean[1] / (mean[2] + eps)
        brightness = mean.mean()
        hsv_arr = np.asarray(img.convert("HSV"), dtype=np.float32) / 255.0
        hsv_mean = hsv_arr.mean(axis=(0, 1))
        hsv_std = hsv_arr.std(axis=(0, 1))
        lab_arr = np.asarray(img.convert("LAB"), dtype=np.float32) / 255.0
        lab_mean = lab_arr.mean(axis=(0, 1))
        lab_std = lab_arr.std(axis=(0, 1))
        gray = arr.mean(axis=2)  # (H, W)
        gy, gx = np.gradient(gray)
        grad_mag = np.sqrt(gx**2 + gy**2)
        grad_mean = grad_mag.mean()
        grad_std = grad_mag.std()
        return np.concatenate(
            [
                mean,
                std,
                median,
                [ratio_g_r, ratio_g_b, brightness],
                hsv_mean,
                hsv_std,
                lab_mean,
                lab_std,
                [grad_mean, grad_std],
            ]
        )
    except Exception:
        return np.full(27, 0.5, dtype=np.float32)


existing_mask = train_df["image_id"].apply(
    lambda x: os.path.exists(os.path.join(train_images_dir, x))
)
train_paths = (
    train_df.loc[existing_mask, "image_id"]
    .apply(lambda x: os.path.join(train_images_dir, x))
    .tolist()
)
train_labels = train_df.loc[existing_mask, "label"].astype(int).values

with ThreadPoolExecutor(max_workers=8) as executor:
    train_features = list(executor.map(image_stats, train_paths, chunksize=32))

train_features = np.stack(train_features) if train_features else np.empty((0, 27))
train_labels = np.array(train_labels, dtype=int)




## === cell 2
os.environ["OMP_NUM_THREADS"] = "1"

from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier

scaler = StandardScaler()
train_features_scaled = scaler.fit_transform(train_features)

rf_clf = RandomForestClassifier(
    n_estimators=2000,  # a bit larger forest for higher accuracy
    max_depth=None,
    random_state=42,
    n_jobs=-1,
    class_weight="balanced",
)

rf_clf.fit(train_features_scaled, train_labels)




## === cell 3
test_image_names = sorted(
    [
        entry.name
        for entry in os.scandir(test_image_dir)
        if entry.is_file() and entry.name.lower().endswith(".jpg")
    ]
)
test_image_paths = [os.path.join(test_image_dir, name) for name in test_image_names]

with ThreadPoolExecutor(max_workers=8) as executor:
    test_features = list(executor.map(image_stats, test_image_paths, chunksize=32))

test_features = np.stack(test_features) if test_features else np.empty((0, 27))
test_features_scaled = scaler.transform(test_features)




## === cell 4
predictions = rf_clf.predict(test_features_scaled).astype(int).tolist()

if len(predictions) < len(test_image_names):
    most_common = int(pd.Series(train_labels).mode()[0])
    missing_count = len(test_image_names) - len(predictions)
    predictions.extend([most_common] * missing_count)
    valid_names = test_image_names
else:
    valid_names = test_image_names




## === cell 5
submission = pd.DataFrame({"image_id": valid_names, "label": predictions})
submission_path = "submission.csv"  # Kaggle expects this in the working directory
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path} with {len(submission)} rows.")
