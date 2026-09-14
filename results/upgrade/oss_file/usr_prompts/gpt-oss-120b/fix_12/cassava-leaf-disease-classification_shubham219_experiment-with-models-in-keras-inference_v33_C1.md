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

0.6406769416742218

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I set the protobuf implementation before importing TensorFlow to avoid the import error, replace the missing model load with a simple dummy model that produces uniform predictions, and adjust the inference code to work with this dummy model. This ensures the script runs end‑to‑end and writes a correctly formatted `submission.csv` file.'
- What this solution (achieved 0.06129) has done: 'I replace the dummy model with a lightweight CNN that is trained for a few epochs on the provided training images, fixing the import error and ensuring a proper model is used for inference. This adds proper data loading, model definition, training, and keeps the original submission‑writing logic, which raise the accuracy from the uniform‑prediction baseline toward the target score.'
- What this solution (achieved 0.51682) has done: 'I replace the TensorFlow‑based image pipeline with a lightweight scikit‑learn model that loads the images via Pillow, resizes them to a small 64×64 resolution, flattens them, and trains a multinomial Logistic Regression classifier. This removes the protobuf import error, ensures the script runs end‑to‑end, and provides a reasonable accuracy boost over the uniform‑prediction baseline while keeping the overall workflow (data loading → model training → test prediction → CSV submission) intact. The final CSV is written to `submission.csv` with the required columns.'
- What this solution (achieved 0.50037) has done: 'I increase the image resolution to 96×96 and replace the simple logistic regression with a lightweight MLP (standard‑scaled) which usually captures non‑linear patterns better while keeping the overall pipeline unchanged. This modest model upgrade is expected to lift validation accuracy closer to the target 0.64 without altering the core data loading or submission logic.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import glob
from PIL import Image
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline
import concurrent.futures
import hashlib

SEED = 42
IMG_SIZE = (96, 96)  # slightly larger resolution for better features
NUM_CLASSES = 5
BATCH_SIZE = 64  # kept for compatibility but not used
EPOCHS = 3  # kept for compatibility but not used
np.random.seed(SEED)

CACHE_DIR = "/tmp/cassava_image_cache"
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
train_csv_path = "../input/cassava-leaf-disease-classification/train.csv"
train_images_dir = "../input/cassava-leaf-disease-classification/train_images/"

train_df = pd.read_csv(train_csv_path)
train_df["path"] = train_images_dir + train_df["image_id"].astype(str)




## === cell 2
def _cache_path_from_paths(paths):
    """
    Generate a deterministic cache file name based on the list of image paths.
    """
    hash_input = "".join(paths).encode()
    key = hashlib.md5(hash_input).hexdigest()[:12]
    return os.path.join(CACHE_DIR, f"preproc_{key}.npy")


def load_and_preprocess(df, img_size):
    """
    Load images, resize, normalize, flatten, and cache the result.
    The cache is reused if the same list of image paths has been processed before.
    """
    paths = df["path"].values
    cache_path = _cache_path_from_paths(paths)

    if os.path.exists(cache_path):
        return np.load(cache_path, mmap_mode="r")  # memory‑map for low RAM pressure

    n_samples = len(paths)
    h, w = img_size
    result = np.empty((n_samples, h * w * 3), dtype=np.float32)

    def _process(idx_path):
        idx, p = idx_path
        with Image.open(p) as img:
            img = img.convert("RGB")
            img = img.resize(img_size, Image.BILINEAR)
            arr = np.asarray(img, dtype=np.float32) / 255.0
        return idx, arr.ravel()

    max_workers = os.cpu_count() or 1  # use all available cores
    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        for idx, flat_arr in executor.map(_process, enumerate(paths), chunksize=64):
            result[idx] = flat_arr

    np.save(cache_path, result, allow_pickle=False)
    return result




## === cell 3
X = load_and_preprocess(train_df, IMG_SIZE)
y = train_df["label"].values

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)




## --- ERROR in cell 3, traceback:
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
AttributeError: Can't pickle local object 'load_and_preprocess.<locals>._process'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2774853507.py in <cell line: 0>()
----> 1 X = load_and_preprocess(train_df, IMG_SIZE)
      2 y = train_df["label"].values
      3 
      4 X_train, X_val, y_train, y_val = train_test_split(
      5     X, y, test_size=0.2, random_state=SEED, stratify=y

/tmp/ipykernel_11/3227009142.py in load_and_preprocess(df, img_size)
     35     # enumerate returns (idx, path); chunksize speeds up large maps
     36     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
---> 37         for idx, flat_arr in executor.map(_process, enumerate(paths), chunksize=64):
     38             result[idx] = flat_arr
     39 

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

AttributeError: Can't pickle local object 'load_and_preprocess.<locals>._process'

## === cell 4
model = make_pipeline(
    StandardScaler(),
    MLPClassifier(
        hidden_layer_sizes=(512, 256),
        activation="relu",
        solver="adam",
        batch_size=64,
        max_iter=200,
        random_state=SEED,
        n_iter_no_change=10,
        warm_start=True,
    ),
)

model.fit(X_train, y_train)
val_acc = model.score(X_val, y_val)
print(f"Validation accuracy: {val_acc:.4f}")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/872017897.py in <cell line: 0>()
     13 )
     14 
---> 15 model.fit(X_train, y_train)
     16 val_acc = model.score(X_val, y_val)
     17 print(f"Validation accuracy: {val_acc:.4f}")

NameError: name 'X_train' is not defined

## === cell 5
test_images = glob.glob(
    "../input/cassava-leaf-disease-classification/test_images/*.jpg"
)
df_test = pd.DataFrame(test_images, columns=["path"])
df_test["image_id"] = df_test["path"].str.split("/").str[-1]

X_test = load_and_preprocess(df_test, IMG_SIZE)




## --- ERROR in cell 5, traceback:
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
AttributeError: Can't pickle local object 'load_and_preprocess.<locals>._process'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3546028852.py in <cell line: 0>()
      5 df_test["image_id"] = df_test["path"].str.split("/").str[-1]
      6 
----> 7 X_test = load_and_preprocess(df_test, IMG_SIZE)
      8 
      9 

/tmp/ipykernel_11/3227009142.py in load_and_preprocess(df, img_size)
     35     # enumerate returns (idx, path); chunksize speeds up large maps
     36     with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
---> 37         for idx, flat_arr in executor.map(_process, enumerate(paths), chunksize=64):
     38             result[idx] = flat_arr
     39 

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

AttributeError: Can't pickle local object 'load_and_preprocess.<locals>._process'

## === cell 6
pred_test_labels = model.predict(X_test)

final_submission = pd.DataFrame(
    {"image_id": df_test["image_id"], "label": pred_test_labels}
)

final_submission.to_csv("submission.csv", index=False)
print("submission.csv written with", len(final_submission), "rows.")




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/271871201.py in <cell line: 0>()
----> 1 pred_test_labels = model.predict(X_test)
      2 
      3 final_submission = pd.DataFrame(
      4     {"image_id": df_test["image_id"], "label": pred_test_labels}
      5 )

NameError: name 'X_test' is not defined

## === cell 7
final_submission.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1367403853.py in <cell line: 0>()
----> 1 final_submission.head()

NameError: name 'final_submission' is not defined
