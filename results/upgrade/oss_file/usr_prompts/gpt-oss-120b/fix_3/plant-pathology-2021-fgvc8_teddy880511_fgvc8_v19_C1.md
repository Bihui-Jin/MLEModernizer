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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.2794275161588194

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
import warnings
import concurrent.futures

warnings.filterwarnings("ignore")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_imgpath = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
train_csvpath = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"

train_files = sorted(os.listdir(train_imgpath))
num_train = len(train_files)


def _load_img(fname):
    img_path = os.path.join(train_imgpath, fname)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {fname}")
    if img.shape != (160, 240, 3):
        img = cv2.resize(img, (240, 160))
    return img


x_train = np.empty((num_train, 160, 240, 3), dtype=np.uint8)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for idx, img in enumerate(executor.map(_load_img, train_files)):
        x_train[idx] = img




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3654969546.py in <cell line: 0>()
     19 x_train = np.empty((num_train, 160, 240, 3), dtype=np.uint8)
     20 with concurrent.futures.ThreadPoolExecutor() as executor:
---> 21     for idx, img in enumerate(executor.map(_load_img, train_files)):
     22         x_train[idx] = img
     23 

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

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/3654969546.py in _load_img(fname)
     11     img = cv2.imread(img_path)
     12     if img is None:
---> 13         raise FileNotFoundError(f"Image not found: {fname}")
     14     if img.shape != (160, 240, 3):
     15         img = cv2.resize(img, (240, 160))

FileNotFoundError: Image not found: train_images

## === cell 2
label_classes = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",
]
train_df = pd.read_csv(train_csvpath)

train_df["primary_label"] = train_df["labels"].apply(lambda x: x.split()[0])
train_df["label_num"] = (
    train_df["primary_label"]
    .map({lbl: i for i, lbl in enumerate(label_classes)})
    .fillna(-1)
    .astype(int)
)

x_train = x_train[train_df["label_num"] >= 0]
y_train = to_categorical(
    train_df.loc[train_df["label_num"] >= 0, "label_num"],
    num_classes=len(label_classes),
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/715864557.py in <cell line: 0>()
     18 )
     19 
---> 20 x_train = x_train[train_df["label_num"] >= 0]
     21 y_train = to_categorical(
     22     train_df.loc[train_df["label_num"] >= 0, "label_num"],

IndexError: boolean index did not match indexed array along dimension 0; dimension is 14906 but corresponding boolean dimension is 14905

## === cell 3
model = ResNet50(
    include_top=True,
    weights=None,
    input_shape=(160, 240, 3),
    classes=len(label_classes),
)
model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])




## === cell 4
model.fit(x_train, y_train, batch_size=20, epochs=1, verbose=2)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3727482862.py in <cell line: 0>()
----> 1 model.fit(x_train, y_train, batch_size=20, epochs=1, verbose=2)
      2 
      3 

NameError: name 'y_train' is not defined

## === cell 5
test_imgpath = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"
test_files = sorted(os.listdir(test_imgpath))
num_test = len(test_files)


def _load_test_img(fname):
    img_path = os.path.join(test_imgpath, fname)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Test image not found: {fname}")
    if img.shape != (160, 240, 3):
        img = cv2.resize(img, (240, 160))
    return img


x_test = np.empty((num_test, 160, 240, 3), dtype=np.uint8)
with concurrent.futures.ThreadPoolExecutor() as executor:
    for idx, img in enumerate(executor.map(_load_test_img, test_files)):
        x_test[idx] = img

pred_probs = model.predict(x_test, batch_size=20, verbose=0)
pred_labels = [label_classes[np.argmax(p)] for p in pred_probs]

submission = pd.DataFrame({"image": test_files, "labels": pred_labels})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2683080611.py in <cell line: 0>()
     16 x_test = np.empty((num_test, 160, 240, 3), dtype=np.uint8)
     17 with concurrent.futures.ThreadPoolExecutor() as executor:
---> 18     for idx, img in enumerate(executor.map(_load_test_img, test_files)):
     19         x_test[idx] = img
     20 

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

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/2683080611.py in _load_test_img(fname)
      8     img = cv2.imread(img_path)
      9     if img is None:
---> 10         raise FileNotFoundError(f"Test image not found: {fname}")
     11     if img.shape != (160, 240, 3):
     12         img = cv2.resize(img, (240, 160))

FileNotFoundError: Test image not found: test_images
