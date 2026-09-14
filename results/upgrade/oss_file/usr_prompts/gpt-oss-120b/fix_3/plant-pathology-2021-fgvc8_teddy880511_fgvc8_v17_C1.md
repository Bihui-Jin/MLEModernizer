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

0.3635457063711896

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, numpy as np, pandas as pd, cv2
import tensorflow as tf
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.preprocessing.image import img_to_array
import multiprocessing as mp  # added for parallel image loading

BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"

train_img_path = os.path.join(BASE_PATH, "train_images")
train_csv_path = os.path.join(BASE_PATH, "train.csv")

train_files = sorted(os.listdir(train_img_path))
num_train = len(train_files)


def _load_train_image(fname):
    """Load and resize a single training image."""
    img_path = os.path.join(train_img_path, fname)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read image {fname}")
    if img.shape != (160, 240, 3):
        img = cv2.resize(img, (240, 160))
    return img


with mp.Pool(processes=mp.cpu_count()) as pool:
    train_images = pool.map(_load_train_image, train_files)

x_train = np.stack(train_images).astype(np.uint8)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
y_df = pd.read_csv(train_csv_path)
le = tf.keras.preprocessing.text.Tokenizer()
from sklearn.preprocessing import LabelEncoder

le_sc = LabelEncoder()
y_encoded = le_sc.fit_transform(y_df["labels"])
num_classes = len(le_sc.classes_)
y_train = to_categorical(y_encoded, num_classes)




## === cell 2
model = ResNet50(
    include_top=True, weights=None, input_shape=(160, 240, 3), classes=num_classes
)

model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])

model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/436437687.py in <cell line: 0>()
      5 model.compile(optimizer=SGD(), loss="categorical_crossentropy", metrics=["accuracy"])
      6 
----> 7 model.fit(x_train, y_train, batch_size=32, epochs=2, verbose=2)
      8 
      9 

NameError: name 'x_train' is not defined

## === cell 3
test_img_path = os.path.join(BASE_PATH, "test_images")
test_csv_path = os.path.join(BASE_PATH, "sample_submission.csv")  # only header needed

test_files = sorted(os.listdir(test_img_path))
num_test = len(test_files)


def _load_test_image(fname):
    """Load and resize a single test image."""
    img_path = os.path.join(test_img_path, fname)
    img = cv2.imread(img_path)
    if img is None:
        raise FileNotFoundError(f"Failed to read test image {fname}")
    if img.shape != (160, 240, 3):
        img = cv2.resize(img, (240, 160))
    return img


with mp.Pool(processes=mp.cpu_count()) as pool:
    test_images = pool.map(_load_test_image, test_files)

x_test = np.stack(test_images).astype(np.uint8)

pred_probs = model.predict(x_test, batch_size=32, verbose=0)
pred_labels = np.argmax(pred_probs, axis=1)
pred_class_names = le_sc.inverse_transform(pred_labels)

submission = pd.DataFrame({"image": test_files, "labels": pred_class_names})

submission.to_csv("submission.csv", index=False)
print("Submission file written to submission.csv")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
RemoteTraceback                           Traceback (most recent call last)
RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 125, in worker
    result = (True, func(*args, **kwds))
                    ^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/pool.py", line 48, in mapstar
    return list(map(*args))
           ^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_11/2227004014.py", line 13, in _load_test_image
    raise FileNotFoundError(f"Failed to read test image {fname}")
FileNotFoundError: Failed to read test image test_images
"""

The above exception was the direct cause of the following exception:

FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2227004014.py in <cell line: 0>()
     18 
     19 with mp.Pool(processes=mp.cpu_count()) as pool:
---> 20     test_images = pool.map(_load_test_image, test_files)
     21 
     22 x_test = np.stack(test_images).astype(np.uint8)

/usr/lib/python3.11/multiprocessing/pool.py in map(self, func, iterable, chunksize)
    365         in a list that is returned.
    366         '''
--> 367         return self._map_async(func, iterable, mapstar, chunksize).get()
    368 
    369     def starmap(self, func, iterable, chunksize=None):

/usr/lib/python3.11/multiprocessing/pool.py in get(self, timeout)
    772             return self._value
    773         else:
--> 774             raise self._value
    775 
    776     def _set(self, i, obj):

/usr/lib/python3.11/multiprocessing/pool.py in worker()
    123         job, i, func, args, kwds = task
    124         try:
--> 125             result = (True, func(*args, **kwds))
    126         except Exception as e:
    127             if wrap_exception and func is not _helper_reraises_exception:

/usr/lib/python3.11/multiprocessing/pool.py in mapstar()
     46 
     47 def mapstar(args):
---> 48     return list(map(*args))
     49 
     50 def starmapstar(args):

/tmp/ipykernel_11/2227004014.py in _load_test_image()
     11     img = cv2.imread(img_path)
     12     if img is None:
---> 13         raise FileNotFoundError(f"Failed to read test image {fname}")
     14     if img.shape != (160, 240, 3):
     15         img = cv2.resize(img, (240, 160))

FileNotFoundError: [Errno None] None
