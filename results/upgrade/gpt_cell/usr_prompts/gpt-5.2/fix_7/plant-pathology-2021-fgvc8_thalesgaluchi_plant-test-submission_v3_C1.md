# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.15789

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os


import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

import zipfile

from PIL import Image

from sklearn.preprocessing import LabelEncoder  # LabelBinarizer

try:
    from google.protobuf import message_factory as _message_factory

    if not hasattr(_message_factory.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            if hasattr(self, "GetMessageClass"):
                return self.GetMessageClass(descriptor)
            from google.protobuf import symbol_database as _symbol_database

            return _symbol_database.Default().GetPrototype(descriptor)

        _message_factory.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass

import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    BatchNormalization,
    Dropout,
    Flatten,
    Input,
)

from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping


class F1Score(tf.keras.metrics.Metric):
    def __init__(
        self,
        num_classes=None,
        average="macro",
        threshold=0.5,
        name="f1_score",
        dtype=None,
    ):
        super().__init__(name=name, dtype=dtype)
        self.num_classes = num_classes
        self.average = average
        self.threshold = threshold

        shape = (
            (num_classes,) if (num_classes is not None and average != "micro") else ()
        )
        self.tp = self.add_weight(name="tp", initializer="zeros", shape=shape)
        self.fp = self.add_weight(name="fp", initializer="zeros", shape=shape)
        self.fn = self.add_weight(name="fn", initializer="zeros", shape=shape)

    def update_state(self, y_true, y_pred, sample_weight=None):
        y_true = tf.cast(y_true, tf.float32)
        y_pred = tf.cast(y_pred, tf.float32)

        y_pred = tf.cast(y_pred >= self.threshold, tf.float32)

        if tf.rank(y_true) == 1:
            y_true = tf.expand_dims(y_true, axis=-1)
        if tf.rank(y_pred) == 1:
            y_pred = tf.expand_dims(y_pred, axis=-1)

        if self.average == "micro":
            axes = None  # reduce all elements
            tp = tf.reduce_sum(y_true * y_pred)
            fp = tf.reduce_sum((1.0 - y_true) * y_pred)
            fn = tf.reduce_sum(y_true * (1.0 - y_pred))
        else:
            axes = 0  # reduce over batch -> per class
            tp = tf.reduce_sum(y_true * y_pred, axis=axes)
            fp = tf.reduce_sum((1.0 - y_true) * y_pred, axis=axes)
            fn = tf.reduce_sum(y_true * (1.0 - y_pred), axis=axes)

        if sample_weight is not None:
            sw = tf.cast(sample_weight, tf.float32)
            sw = tf.reshape(sw, (-1, 1))
            if self.average == "micro":
                tp = tf.reduce_sum((y_true * y_pred) * sw)
                fp = tf.reduce_sum(((1.0 - y_true) * y_pred) * sw)
                fn = tf.reduce_sum((y_true * (1.0 - y_pred)) * sw)
            else:
                tp = tf.reduce_sum((y_true * y_pred) * sw, axis=0)
                fp = tf.reduce_sum(((1.0 - y_true) * y_pred) * sw, axis=0)
                fn = tf.reduce_sum((y_true * (1.0 - y_pred)) * sw, axis=0)

        self.tp.assign_add(tp)
        self.fp.assign_add(fp)
        self.fn.assign_add(fn)

    def result(self):
        precision = self.tp / (self.tp + self.fp + tf.keras.backend.epsilon())
        recall = self.tp / (self.tp + self.fn + tf.keras.backend.epsilon())
        f1 = (
            2.0 * precision * recall / (precision + recall + tf.keras.backend.epsilon())
        )

        if self.average == "macro":
            return tf.reduce_mean(f1)
        if self.average == "weighted":
            support = self.tp + self.fn
            return tf.reduce_sum(f1 * support) / (
                tf.reduce_sum(support) + tf.keras.backend.epsilon()
            )
        return f1

    def reset_states(self):
        for v in self.variables:
            v.assign(tf.zeros_like(v))


## === cell 1
y_train = pd.read_csv('/kaggle/input/plant-pathology-2021-fgvc8/train.csv')


## === cell 2
file_path_test = '/kaggle/input/plant-pathology-2021-fgvc8/test_images'
test_filenames = os.listdir(file_path_test)


## === cell 3
sumb_sample = pd.read_csv('../input/plant-pathology-2021-fgvc8/sample_submission.csv')
sumb_sample


## === cell 4
subm = [(item, 'healthy') for item in test_filenames]
subm


## === cell 5
submission = pd.DataFrame(subm, columns= ['image', 'labels'])
submission.set_index('image', inplace = True)
submission.to_csv('./submission.csv')


## === cell 6
submited = pd.read_csv('./submission.csv')
submited
