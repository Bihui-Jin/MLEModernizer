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

0.853732245391357

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61659) has done: 'I replace the failing TensorFlow and Albumentations imports with lightweight NumPy, pandas, OpenCV, and scikit‑learn code, train a simple RandomForest classifier on resized images, and generate a valid submission.csv file. This removes the protobuf/Albumentations errors, ensures the script runs end‑to‑end, and produces a correctly‑formatted prediction file.'
- What this solution (achieved 0.62481) has done: 'I replace the RandomForest with a HistGradientBoostingClassifier (which generally gives higher accuracy on high‑dimensional image vectors) and ensure the feature matrix is a float type expected by the new model. This change keeps the overall pipeline unchanged while moving the validation accuracy closer to the target score.'
- What this solution (achieved 0.61921) has done: 'I normalize pixel values to the [0, 1] range and give the HistGradientBoosting model a bit more capacity (increase max_iter and lower learning_rate) so it can learn richer patterns from the images. These small, targeted tweaks keep the overall pipeline unchanged while aiming to raise validation accuracy toward the target score.'

# 9. Code solution

## === cell 0
import os

os.environ["OMP_NUM_THREADS"] = str(os.cpu_count())
os.environ["OPENBLAS_NUM_THREADS"] = str(os.cpu_count())
os.environ["MKL_NUM_THREADS"] = str(os.cpu_count())

import cv2
import json
import warnings
import numpy as np
import pandas as pd
import multiprocessing as mp
from concurrent.futures import (
    ThreadPoolExecutor,
)  # threads are fast for I/O bound cv2 ops

warnings.simplefilter("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.utils import parallel_backend  # enforce multithreaded backend during fit

np.random.seed(42)
import random

random.seed(42)




## === cell 1
img_width, img_height = 64, 64
feat_len = img_width * img_height * 3


def load_image(img_path):
    """Read an image, resize, and return as a float32 flattened array scaled to [0, 1]."""
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (img_width, img_height))
    return (img / 255.0).astype(np.float32).ravel()


train_images_dir = os.path.join(general_path, "train_images")
N = df_train.shape[0]
train_img_paths = [
    os.path.join(train_images_dir, img_id) for img_id in df_train["image_id"]
]

print(
    "Loading and preprocessing training images (thread pool, may take a few minutes)..."
)
pool_size = min(8, max(1, mp.cpu_count() - 1))

X_orig = np.empty((N, feat_len), dtype=np.float32)

with ThreadPoolExecutor(max_workers=pool_size) as executor:
    for idx, arr in enumerate(executor.map(load_image, train_img_paths, chunksize=32)):
        X_orig[idx] = arr

y_orig = df_train["label"].values.astype(int)

X_flip = X_orig.reshape(N, img_height, img_width, 3)[:, :, ::-1, :].reshape(N, feat_len)

X = np.vstack((X_orig, X_flip))
y = np.concatenate((y_orig, y_orig))

print("Feature matrix shape:", X.shape)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3475865361.py in <cell line: 0>()
     11 
     12 
---> 13 train_images_dir = os.path.join(general_path, "train_images")
     14 N = df_train.shape[0]
     15 train_img_paths = [

NameError: name 'general_path' is not defined

## === cell 2
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

clf = HistGradientBoostingClassifier(
    max_iter=500,  # retained from original script
    learning_rate=0.02,
    random_state=42,
)

print("Training HistGradientBoosting classifier...")

with parallel_backend("threading", n_jobs=os.cpu_count()):
    clf.fit(X_train, y_train)

val_pred = clf.predict(X_val)
val_acc = accuracy_score(y_val, val_pred)
print(f"Validation accuracy (approx.): {val_acc:.4f}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/915741291.py in <cell line: 0>()
      1 X_train, X_val, y_train, y_val = train_test_split(
----> 2     X, y, test_size=0.2, random_state=42, stratify=y
      3 )
      4 
      5 clf = HistGradientBoostingClassifier(

NameError: name 'X' is not defined

## === cell 3
test_df = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
test_images_dir = os.path.join(general_path, "test_images")

M = test_df.shape[0]
test_ids = test_df["image_id"].tolist()
test_img_paths = [os.path.join(test_images_dir, img_id) for img_id in test_ids]

print("Loading and preprocessing test images (thread pool)...")
X_test = np.empty((M, feat_len), dtype=np.float32)

with ThreadPoolExecutor(max_workers=pool_size) as executor:
    for idx, arr in enumerate(executor.map(load_image, test_img_paths, chunksize=32)):
        X_test[idx] = arr

test_preds = clf.predict(X_test)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/82488323.py in <cell line: 0>()
----> 1 test_df = pd.read_csv(os.path.join(general_path, "sample_submission.csv"))
      2 test_images_dir = os.path.join(general_path, "test_images")
      3 
      4 M = test_df.shape[0]
      5 test_ids = test_df["image_id"].tolist()

NameError: name 'general_path' is not defined
