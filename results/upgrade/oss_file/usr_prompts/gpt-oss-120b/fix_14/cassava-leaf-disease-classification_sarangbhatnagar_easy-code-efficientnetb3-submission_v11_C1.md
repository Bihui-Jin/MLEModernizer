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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

0.8507101843457238

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fixed the protobuf import issue, added a fallback model builder (EfficientNetB0 pretrained on ImageNet) that trains quickly on the provided training set, and rewrote the inference loop to use this model. The script now creates a proper `submission.csv` with the required columns, so it runs end‑to‑end and produces a valid Kaggle submission file.'
- What this solution (achieved 0.61996) has done: 'The fix replaces the TensorFlow‑based pipeline (which crashes due to protobuf incompatibility) with a plain NumPy + scikit‑learn workflow: images are loaded with Pillow, resized, normalized and flattened, then a RandomForest classifier is trained and used for inference. This eliminates the protobuf error while keeping the overall data‑handling and submission steps unchanged, and it modestly improves accuracy toward the target score.'
- What this solution (achieved 0.61584) has done: 'I keep the overall RandomForest‑style workflow but improve the pixel representation and the tree ensemble:  
- Increase the image resolution to 96×96 so the model sees finer details while still fitting in memory.  
- Switch to `ExtraTreesClassifier`, which generally yields higher accuracy on high‑dimensional image vectors, and raise the number of trees to 600 with `max_features='sqrt'`.  
These small, targeted tweaks should raise validation accuracy toward the target without altering the core pipeline.'
- What this solution (achieved 0.61398) has done: 'I increase the image resolution to 112×112, add a balanced class weighting, raise the number of trees to 800, and augment the training set with horizontal flips. These modest hyper‑parameter and data‑augmentation tweaks stay within the original ExtraTrees‑based pipeline while expected to raise validation accuracy toward the target score.'
- What this solution (achieved 0.61547) has done: 'The updates keep the same model and augmentation strategy but drastically cut memory and I/O overhead: images are kept as uint8 (no unnecessary float‑scaling) and are loaded in parallel threads, reducing the image‑loading bottleneck. The training set now occupies ~1 GB instead of >6 GB, and the ExtraTrees ensemble size is lowered to 400 trees (still a strong classifier) to keep total runtime well under the 600 s limit while preserving the original algorithmic logic. Seed settings and data paths remain unchanged.'
- What this solution (achieved 0.61435) has done: 'I modestly boost the model by using a larger image size (112 × 112), adding a few simple color‑statistics features (mean and std per channel), and increasing the ExtraTrees ensemble to 600 trees. These changes keep the same overall pipeline (pixel‑based ExtraTrees) while giving the classifier richer information and a slightly stronger ensemble, which should raise validation accuracy toward the target without breaking the existing workflow.'
- What this solution (achieved 0.61248) has done: 'We shrink the image resolution to 96 × 96 and stop the heavy data‑augmentation step so the training matrix fits in memory and the script finishes. The label array is aligned to the (now non‑augmented) training data, and we slightly reduce the ensemble size to keep runtime low while preserving the original ExtraTrees‑based approach.'
- What this solution (achieved 0.61323) has done: 'The fix aligns the training labels with the augmented (or non‑augmented) feature matrix and switches to a larger image size for better visual detail, while keeping the ExtraTrees pipeline unchanged. This resolves the sample‑size mismatch error and should improve validation accuracy toward the target.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.ensemble import ExtraTreesClassifier  # use a stronger tree ensemble
import concurrent.futures

np.random.seed(42)



## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

train_df = pd.read_csv(TRAIN_CSV)



## === cell 2
train_df, val_df = train_test_split(
    train_df, test_size=0.1, stratify=train_df["label"], random_state=42
)

IMG_SIZE = (96, 96)


def _process_image_features(path):
    """
    Load an image, resize and compute:
    - flattened pixel values (uint8 → float32)
    - per‑channel mean & std
    - per‑channel 16‑bin histograms (density)
    Returns the raw pixel array, the statistics and histograms for reuse.
    """
    with Image.open(path) as im:
        im = im.convert("RGB")
        im = im.resize(IMG_SIZE)
        im_np = np.array(im, dtype=np.uint8)  # (H, W, 3)

        channel_means = im_np.mean(axis=(0, 1), dtype=np.float32)  # (3,)
        channel_stds = im_np.std(axis=(0, 1), dtype=np.float32)  # (3,)

        hist_features = []
        for c in range(3):
            hist, _ = np.histogram(
                im_np[:, :, c], bins=16, range=(0, 255), density=True
            )
            hist_features.append(hist.astype(np.float32))
        hist_features = np.concatenate(hist_features)  # (48,)

        orig_pixels = im_np.ravel().astype(np.float32)  # (96*96*3,)

        return orig_pixels, channel_means, channel_stds, hist_features, im_np


def load_images(df, img_dir, augment=False):
    """
    Parallel image loading with a pre‑allocated NumPy array.
    For each image we store the original features; if augment=True we also store the
    horizontally flipped version (using the same statistics to match the original code).
    """
    paths = [os.path.join(img_dir, fname) for fname in df["image_id"]]
    sample_feat, _, _, _, _ = _process_image_features(paths[0])
    feat_len = sample_feat.size + 3 + 3 + 48  # pixels + means + stds + histograms

    repeat = 2 if augment else 1
    total_samples = len(paths) * repeat
    X = np.empty((total_samples, feat_len), dtype=np.float32)

    def worker(path):
        return _process_image_features(path)

    idx = 0
    with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
        for result in executor.map(worker, paths):
            orig_pixels, ch_means, ch_stds, hist_feat, im_np = result

            X[idx, : orig_pixels.size] = orig_pixels
            X[idx, orig_pixels.size : orig_pixels.size + 3] = ch_means
            X[idx, orig_pixels.size + 3 : orig_pixels.size + 6] = ch_stds
            X[idx, orig_pixels.size + 6 :] = hist_feat
            idx += 1

            if augment:
                flip_np = np.fliplr(im_np)
                flip_pixels = flip_np.ravel().astype(np.float32)

                X[idx, : flip_pixels.size] = flip_pixels
                X[idx, flip_pixels.size : flip_pixels.size + 3] = ch_means
                X[idx, flip_pixels.size + 3 : flip_pixels.size + 6] = ch_stds
                X[idx, flip_pixels.size + 6 :] = hist_feat
                idx += 1
    return X


X_train = load_images(train_df, TRAIN_IMG_DIR, augment=True)
y_train = np.repeat(train_df["label"].values, 2)

X_val = load_images(val_df, TRAIN_IMG_DIR, augment=False)
y_val = val_df["label"].values



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object 'load_images.<locals>.worker'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/476922922.py in <cell line: 0>()
     81 
     82 
---> 83 X_train = load_images(train_df, TRAIN_IMG_DIR, augment=True)
     84 y_train = np.repeat(train_df["label"].values, 2)
     85 

/tmp/ipykernel_55/476922922.py in load_images(df, img_dir, augment)
     58     # Use ProcessPoolExecutor to bypass the GIL for CPU‑bound work
     59     with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
---> 60         for result in executor.map(worker, paths):
     61             orig_pixels, ch_means, ch_stds, hist_feat, im_np = result
     62 

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

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
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'load_images.<locals>.worker'

## === cell 3
et_model = ExtraTreesClassifier(
    n_estimators=800,
    max_features=0.7,  # use ~70 % of features at each split
    max_depth=None,
    n_jobs=-1,
    random_state=42,
    class_weight="balanced",
    verbose=0,
)

et_model.fit(X_train, y_train)

val_preds = et_model.predict(X_val)
val_acc = accuracy_score(y_val, val_preds)
print(f"Validation accuracy: {val_acc:.5f}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/658480927.py in <cell line: 0>()
      9 )
     10 
---> 11 et_model.fit(X_train, y_train)
     12 
     13 val_preds = et_model.predict(X_val)

NameError: name 'X_train' is not defined

## === cell 4
test_df = pd.read_csv(SAMPLE_SUB)
X_test = load_images(test_df, TEST_IMG_DIR, augment=False)

test_pred_labels = et_model.predict(X_test)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"], "label": test_pred_labels.astype(int)}
)
submission.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object 'load_images.<locals>.worker'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3287474981.py in <cell line: 0>()
      1 test_df = pd.read_csv(SAMPLE_SUB)
----> 2 X_test = load_images(test_df, TEST_IMG_DIR, augment=False)
      3 
      4 test_pred_labels = et_model.predict(X_test)
      5 

/tmp/ipykernel_55/476922922.py in load_images(df, img_dir, augment)
     58     # Use ProcessPoolExecutor to bypass the GIL for CPU‑bound work
     59     with concurrent.futures.ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
---> 60         for result in executor.map(worker, paths):
     61             orig_pixels, ch_means, ch_stds, hist_feat, im_np = result
     62 

/usr/lib/python3.11/concurrent/futures/process.py in _chain_from_iterable_of_lists(iterable)
    618     careful not to keep references to yielded objects.
    619     """
--> 620     for element in iterable:
    621         element.reverse()
    622         while element:

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
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object 'load_images.<locals>.worker'
