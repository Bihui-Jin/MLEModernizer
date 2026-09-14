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

0.7410156971375809

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.25492) has done: 'The changes speed up data loading and model training by using more worker processes for the ImageDataGenerator and by increasing the test generator batch size, which reduces the number of prediction steps. These adjustments keep the same model, augmentation, and training epochs, so the resulting predictions remain identical while the runtime drops well below the 600‑second limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense
from tensorflow.keras.models import Sequential
from tensorflow.keras.utils import Sequence
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.model_selection import train_test_split
import math

print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/plant-pathology-2021-fgvc8"
train_dir = os.path.join(BASE_PATH, "train_images")
test_dir = os.path.join(BASE_PATH, "test_images")
train_csv_path = os.path.join(BASE_PATH, "train.csv")
sample_submission_path = os.path.join(BASE_PATH, "sample_submission.csv")




## === cell 2
train_df = pd.read_csv(train_csv_path)
test_df = pd.read_csv(
    sample_submission_path
)  # contains correct ordering of test images
test_ids = test_df["image"].tolist()




## === cell 3
train_df["label_list"] = train_df["labels"].apply(lambda x: x.split(" "))




## === cell 4
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(train_df["label_list"])
class_names = mlb.classes_.tolist()
num_classes = len(class_names)




## === cell 5
def _load_and_preprocess(path, label=None):
    """Read an image file, decode, resize, and apply ResNet‑50 preprocessing."""
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 336])
    img = tf.keras.applications.resnet50.preprocess_input(tf.cast(img, tf.float32))
    if label is None:
        return img
    else:
        return img, label


def make_dataset(file_paths, labels=None, batch_size=32, shuffle=False, seed=42):
    """Create a tf.data.Dataset for given file paths and optional labels."""
    paths_ds = tf.data.Dataset.from_tensor_slices(file_paths)
    if labels is not None:
        labels_ds = tf.data.Dataset.from_tensor_slices(labels.astype(np.float32))
        ds = tf.data.Dataset.zip((paths_ds, labels_ds))
        ds = ds.map(_load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    else:
        ds = paths_ds.map(
            lambda p: _load_and_preprocess(p, None), num_parallel_calls=tf.data.AUTOTUNE
        )

    if shuffle:
        ds = ds.shuffle(
            buffer=len(file_paths), seed=seed, reshuffle_each_iteration=True
        )
    ds = ds.batch(batch_size)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 6
train_idx, val_idx = train_test_split(
    np.arange(len(train_df)), test_size=0.1, random_state=42, stratify=y
)

train_paths = (
    train_df.iloc[train_idx]["image"].apply(lambda f: os.path.join(train_dir, f)).values
)
val_paths = (
    train_df.iloc[val_idx]["image"].apply(lambda f: os.path.join(train_dir, f)).values
)

train_labels = y[train_idx]
val_labels = y[val_idx]

batch_size = 32

train_ds = make_dataset(
    train_paths, train_labels, batch_size=batch_size, shuffle=True, seed=42
)
val_ds = make_dataset(val_paths, val_labels, batch_size=batch_size, shuffle=False)

test_paths = test_df["image"].apply(lambda f: os.path.join(test_dir, f)).values
test_ds = make_dataset(test_paths, labels=None, batch_size=batch_size, shuffle=False)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/602092931.py in <cell line: 0>()
     17 batch_size = 32
     18 
---> 19 train_ds = make_dataset(
     20     train_paths, train_labels, batch_size=batch_size, shuffle=True, seed=42
     21 )

/tmp/ipykernel_11/840545608.py in make_dataset(file_paths, labels, batch_size, shuffle, seed)
     27 
     28     if shuffle:
---> 29         ds = ds.shuffle(
     30             buffer=len(file_paths), seed=seed, reshuffle_each_iteration=True
     31         )

TypeError: DatasetV2.shuffle() got an unexpected keyword argument 'buffer'

## === cell 7
model = Sequential(
    [
        Conv2D(32, (3, 3), activation="relu", input_shape=(224, 336, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu"),
        GlobalAveragePooling2D(),
        Dense(128, activation="relu"),
        Dense(num_classes, activation="sigmoid"),
    ]
)

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])




## === cell 8
steps_per_epoch = math.ceil(len(train_idx) / batch_size)
validation_steps = math.ceil(len(val_idx) / batch_size)

model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4214737943.py in <cell line: 0>()
      4 
      5 model.fit(
----> 6     train_ds,
      7     validation_data=val_ds,
      8     epochs=10,

NameError: name 'train_ds' is not defined

## === cell 9
test_preds = model.predict(test_ds, verbose=0)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1379483621.py in <cell line: 0>()
----> 1 test_preds = model.predict(test_ds, verbose=0)
      2 
      3 

NameError: name 'test_ds' is not defined

## === cell 10
threshold = 0.5
pred_labels = []
for probs in test_preds:
    indices = np.where(probs >= threshold)[0]
    if len(indices) == 0:
        indices = [np.argmax(probs)]
    labels = [class_names[i] for i in indices]
    pred_labels.append(" ".join(labels))




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2390951155.py in <cell line: 0>()
      1 threshold = 0.5
      2 pred_labels = []
----> 3 for probs in test_preds:
      4     indices = np.where(probs >= threshold)[0]
      5     if len(indices) == 0:

NameError: name 'test_preds' is not defined

## === cell 11
submission = pd.DataFrame({"image": test_ids, "labels": pred_labels})
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2939963316.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image": test_ids, "labels": pred_labels})
      2 submission.to_csv("submission.csv", index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
