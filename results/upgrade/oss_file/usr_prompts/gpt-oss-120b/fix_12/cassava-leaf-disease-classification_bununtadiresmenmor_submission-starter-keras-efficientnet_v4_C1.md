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

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8754910849199153

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.62481) has done: 'I remove the TensorFlow imports that crash because of a protobuf incompatibility, replace the deep‑learning model with a lightweight scikit‑learn classifier that uses simple resized‑image pixel features, and adjust all paths so the script can locate the train and test image folders. This fixes the runtime errors, creates a valid `submission.csv`, and—while keeping the core logic minimal—provides a reasonable baseline that should approach the target accuracy.'
- What this solution (achieved 0.61547) has done: 'The update enlarges the image size to capture more detail, adds normalized color‑histogram features to the pixel vector, and strengthens the RandomForest with more trees and balanced class weighting—small, targeted changes expected to raise validation accuracy toward the target while keeping the overall pipeline unchanged.'
- What this solution (achieved 0.61584) has done: 'I increase the image resolution from 64 to 96 pixels, double the histogram bin count, and strengthen the RandomForest by using more trees and the sqrt feature‑sampling strategy. These modest tweaks give the model richer visual information and better regularisation, which should raise validation accuracy toward the target without altering the overall pipeline.'

# 9. Code solution

## === cell 0
import os
import cv2
import json
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import concurrent.futures  # ProcessPoolExecutor for true parallel CPU‑bound preprocessing




## === cell 1
BASE_INPUT = Path("../input/cassava-leaf-disease-classification")
TRAIN_CSV = BASE_INPUT / "train.csv"
TEST_SAMPLE_SUB = BASE_INPUT / "sample_submission.csv"
TRAIN_IMG_DIR = BASE_INPUT / "train_images"
TEST_IMG_DIR = BASE_INPUT / "test_images"

IMG_SIZE = (96, 96)  # modest size to stay memory‑friendly
SEED = 42
HIST_BINS = 64  # finer colour histograms (was 32)


def load_and_preprocess(img_path: Path) -> np.ndarray:
    """Read an image, resize, convert to RGB, normalise, flatten and append colour histograms plus channel statistics."""
    img = cv2.imread(str(img_path))
    if img is None:
        flat_len = IMG_SIZE[0] * IMG_SIZE[1] * 3
        hist_len = HIST_BINS * 3
        stats_len = 6  # 3 means + 3 stds
        return np.zeros(flat_len + hist_len + stats_len, dtype=np.float32)

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, IMG_SIZE, interpolation=cv2.INTER_AREA)
    img_norm = img_resized.astype(np.float32) / 255.0
    flat = img_norm.ravel()  # already float32, shape (IMG_SIZE[0]*IMG_SIZE[1]*3,)

    hist_features = []
    for ch in range(3):
        channel = img_resized[:, :, ch].ravel()
        hist = np.bincount(channel, minlength=HIST_BINS).astype(np.float32)
        hist_sum = hist.sum()
        if hist_sum > 0:
            hist /= hist_sum
        hist_features.append(hist)
    hist_vec = np.concatenate(hist_features)

    means = img_norm.mean(axis=(0, 1))  # (3,)
    stds = img_norm.std(axis=(0, 1))  # (3,)
    stats_vec = np.concatenate([means, stds])  # (6,)

    return np.concatenate([flat, hist_vec, stats_vec])


train_df = pd.read_csv(TRAIN_CSV)
train_img_paths = [TRAIN_IMG_DIR / img_id for img_id in train_df["image_id"]]

sample_feat = load_and_preprocess(train_img_paths[0])
feat_len = sample_feat.shape[0]
num_train = len(train_img_paths)

max_workers = min(os.cpu_count() or 1, 8)
X = np.empty((num_train, feat_len), dtype=np.float32)
X[0] = sample_feat

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(
        executor.map(load_and_preprocess, train_img_paths[1:], chunksize=20), start=1
    ):
        X[idx] = feat

X = np.ascontiguousarray(X)  # ensure contiguous memory for faster RandomForest access
y = train_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.1, random_state=SEED, stratify=y
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1677143993.py in <cell line: 0>()
     58         executor.map(load_and_preprocess, train_img_paths[1:], chunksize=20), start=1
     59     ):
---> 60         X[idx] = feat
     61 
     62 X = np.ascontiguousarray(X)  # ensure contiguous memory for faster RandomForest access

ValueError: could not broadcast input array from shape (28393,) into shape (28417,)

## === cell 2
rf_clf = RandomForestClassifier(
    n_estimators=1500,  # same number of trees (core algorithm unchanged)
    max_depth=None,
    max_features="sqrt",  # use sqrt of features per split – reduces cost per tree
    max_samples=0.8,  # each tree sees only 80 % of samples – speeds up training
    class_weight="balanced",
    n_jobs=-1,
    random_state=SEED,
    verbose=0,
)
rf_clf.fit(X_train, y_train)

val_preds = rf_clf.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.5f}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3049488432.py in <cell line: 0>()
      9     verbose=0,
     10 )
---> 11 rf_clf.fit(X_train, y_train)
     12 
     13 val_preds = rf_clf.predict(X_val)

NameError: name 'X_train' is not defined

## === cell 3
test_sub = pd.read_csv(TEST_SAMPLE_SUB)
test_images = test_sub["image_id"].tolist()
test_img_paths = [TEST_IMG_DIR / img_id for img_id in test_images]

sample_test_feat = load_and_preprocess(test_img_paths[0])
test_feat_len = sample_test_feat.shape[0]
num_test = len(test_img_paths)

X_test = np.empty((num_test, test_feat_len), dtype=np.float32)
X_test[0] = sample_test_feat

with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
    for idx, feat in enumerate(
        executor.map(load_and_preprocess, test_img_paths[1:], chunksize=20), start=1
    ):
        X_test[idx] = feat

X_test = np.ascontiguousarray(X_test)

test_preds = rf_clf.predict(X_test)

submission = pd.DataFrame({"image_id": test_images, "label": test_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/4080389389.py in <cell line: 0>()
     14         executor.map(load_and_preprocess, test_img_paths[1:], chunksize=20), start=1
     15     ):
---> 16         X_test[idx] = feat
     17 
     18 X_test = np.ascontiguousarray(X_test)

ValueError: could not broadcast input array from shape (28387,) into shape (28410,)

## === cell 4
submission.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined
