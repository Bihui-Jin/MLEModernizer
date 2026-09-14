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

0.8443638561498942

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.60538) has done: 'The changes keep the exact feature‑extraction and model logic but avoid the expensive repeated startup of separate process pools. A single global `ProcessPoolExecutor` is created once, reused for both training and test feature extraction, and shut down after all work is done. Using a larger `chunksize` further reduces inter‑process overhead. These tweaks cut the overall runtime while producing identical features and predictions.'
- What this solution (achieved 0.60688) has done: 'The changes switch to a lightweight `ThreadPoolExecutor` (avoiding expensive process pickling) and replace the NumPy‑based pixel loop with Pillow’s `ImageStat` which computes channel means and standard deviations directly on the image without allocating a full array. This yields the same 6‑dimensional color statistics (just scaled), dramatically reducing I/O and computation time while preserving the model’s exact logic and deterministic behavior.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
from pathlib import Path
from PIL import Image, ImageStat
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from concurrent.futures import ThreadPoolExecutor  # lighter than processes

_executor = ThreadPoolExecutor(max_workers=os.cpu_count())


def extract_features(pil_img):
    """
    Compute extended color statistics using Pillow's ImageStat.
    Returns a 12‑element numpy array:
    [mean_R, mean_G, mean_B,
     std_R,  std_G,  std_B,
     min_R,  min_G,  min_B,
     max_R,  max_G,  max_B],
    all scaled to [0, 1] to match the original implementation.
    """
    stat = ImageStat.Stat(pil_img)  # works on 0‑255 pixel values
    means = np.array(stat.mean) / 255.0
    stds = np.array(stat.stddev) / 255.0
    mins = np.array(stat.min) / 255.0
    maxs = np.array(stat.max) / 255.0
    return np.concatenate([means, stds, mins, maxs])


def compute_feature_from_path(path_str):
    """Load image and compute its feature vector – used in parallel execution."""
    with Image.open(path_str) as img:
        pil_img = img.convert("RGB")
        return extract_features(pil_img)




## === cell 1
base_path = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_img_dir = os.path.join(base_path, "train_images")

train_df = pd.read_csv(train_csv_path)
train_image_ids = train_df["image_id"].tolist()
train_labels = train_df["label"].values

train_image_paths = [os.path.join(train_img_dir, img_id) for img_id in train_image_ids]

train_features_iter = _executor.map(
    compute_feature_from_path, train_image_paths, chunksize=20
)
train_features = np.stack(list(train_features_iter), axis=0)  # (N_train, 12)

X_tr, X_val, y_tr, y_val = train_test_split(
    train_features,
    train_labels,
    test_size=0.2,
    random_state=42,
    stratify=train_labels,
)

best_depth = None
best_acc = 0.0
for depth in [8, 12, 16, 20, 24, None]:
    clf = DecisionTreeClassifier(
        criterion="gini",
        max_depth=depth,
        min_samples_split=2,
        random_state=42,
    )
    clf.fit(X_tr, y_tr)
    preds = clf.predict(X_val)
    acc = accuracy_score(y_val, preds)
    if acc > best_acc:
        best_acc = acc
        best_depth = depth

decision_tree = DecisionTreeClassifier(
    criterion="gini",
    max_depth=best_depth,
    min_samples_split=2,
    random_state=42,
)
decision_tree.fit(train_features, train_labels)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/2758735143.py in <cell line: 0>()
     12     compute_feature_from_path, train_image_paths, chunksize=20
     13 )
---> 14 train_features = np.stack(list(train_features_iter), axis=0)  # (N_train, 12)
     15 
     16 X_tr, X_val, y_tr, y_val = train_test_split(

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/3721372371.py in compute_feature_from_path(path_str)
     34     with Image.open(path_str) as img:
     35         pil_img = img.convert("RGB")
---> 36         return extract_features(pil_img)
     37 
     38 

/tmp/ipykernel_55/3721372371.py in extract_features(pil_img)
     25     means = np.array(stat.mean) / 255.0
     26     stds = np.array(stat.stddev) / 255.0
---> 27     mins = np.array(stat.min) / 255.0
     28     maxs = np.array(stat.max) / 255.0
     29     return np.concatenate([means, stds, mins, maxs])

AttributeError: 'Stat' object has no attribute 'min'

## === cell 2
test_root = Path(base_path) / "test_images"
if not test_root.is_dir():
    candidates = list(Path(base_path).rglob("test_images"))
    test_root = next((p for p in candidates if p.is_dir()), None)
    if test_root is None:
        raise FileNotFoundError("Test image directory not found.")

valid_ext = {".jpg", ".jpeg", ".png"}
test_image_paths = sorted(
    [p for p in test_root.rglob("*") if p.suffix.lower() in valid_ext]
)

test_image_ids = [p.name for p in test_image_paths]

test_features_iter = _executor.map(
    compute_feature_from_path, (str(p) for p in test_image_paths), chunksize=20
)
test_features = np.stack(list(test_features_iter), axis=0)  # (N_test, 12)

test_predictions = decision_tree.predict(test_features)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3794384472.py in <cell line: 0>()
     16     compute_feature_from_path, (str(p) for p in test_image_paths), chunksize=20
     17 )
---> 18 test_features = np.stack(list(test_features_iter), axis=0)  # (N_test, 12)
     19 
     20 test_predictions = decision_tree.predict(test_features)

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_55/3721372371.py in compute_feature_from_path(path_str)
     34     with Image.open(path_str) as img:
     35         pil_img = img.convert("RGB")
---> 36         return extract_features(pil_img)
     37 
     38 

/tmp/ipykernel_55/3721372371.py in extract_features(pil_img)
     25     means = np.array(stat.mean) / 255.0
     26     stds = np.array(stat.stddev) / 255.0
---> 27     mins = np.array(stat.min) / 255.0
     28     maxs = np.array(stat.max) / 255.0
     29     return np.concatenate([means, stds, mins, maxs])

AttributeError: 'Stat' object has no attribute 'min'

## === cell 3
submission = pd.DataFrame(
    {"image_id": test_image_ids, "label": test_predictions.astype(int)}
)
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")

_executor.shutdown(wait=True)

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3360938771.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"image_id": test_image_ids, "label": test_predictions.astype(int)}
      3 )
      4 submission_path = "submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'test_predictions' is not defined
