# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.10

# 2. Installed packages

geopandas==0.14.4
lightgbm==4.6.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import importlib

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as _pb_ver

    _pb_major = int(_pb_ver.split(".")[0])
    if _pb_major >= 5:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )

        for _m in list(sys.modules.keys()):
            if _m.startswith("google.protobuf"):
                del sys.modules[_m]

        importlib.invalidate_caches()
except Exception:
    pass

import random

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import lightgbm as lgb
import tensorflow as tf
from tensorflow.keras import Sequential, Model
from tensorflow.keras.layers import (
    Dense,
    Conv2D,
    Flatten,
    Dropout,
    Input,
    Concatenate,
    BatchNormalization,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers.schedules import ExponentialDecay
from PIL import Image
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


## === cell 1

AUTOTUNE = tf.data.experimental.AUTOTUNE  
img_size = 224
channels = 3
Batch_size = 64


train_dir = "/kaggle/input/petfinder-pawpularity-score/train/"
test_dir = "/kaggle/input/petfinder-pawpularity-score/test/"

def seed_everything():
    np.random.seed(123)
    random.seed(123)
    tf.random.set_seed(123)
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ['PYTHONHASHSEED'] = str(123)

seed_everything()


## === cell 2

df = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/train.csv")
df_test = pd.read_csv("/kaggle/input/petfinder-pawpularity-score/test.csv")
Id = df_test["Id"].copy()



df["Id"] = df["Id"].apply(lambda x : "/kaggle/input/petfinder-pawpularity-score/train/" + x + ".jpg")
df_test["Id"] = df_test["Id"].apply(lambda x : "/kaggle/input/petfinder-pawpularity-score/test/" + x + ".jpg")


## === cell 3

def image_preprocess(is_labelled):  
    def augment(image):
        image = tf.image.random_flip_left_right(image)
        image = tf.image.random_saturation(image, 0.95, 1.05)
        image = tf.image.random_contrast(image, 0.95, 1.05)
        return image
    
    def can_be_augmented(img, label):
        return augment(img), label
    
    return can_be_augmented if is_labelled else augment



def image_read(is_labelled):
    def decode(path):
        image = tf.io.read_file(path)
        image = tf.image.decode_jpeg(image, channels=channels)
        image = tf.cast(image, tf.float32)
        image = tf.image.resize(image, (img_size, img_size))
        return image
    
    def can_be_decoded(path, label):
        return decode(path), label
    

    return can_be_decoded if is_labelled else decode


def create_dataset(df, batch_size, is_labelled = False, augment = False, shuffle = False):
    image_read_fn = image_read(is_labelled)
    image_preprocess_fn = image_preprocess(is_labelled)
    
    if is_labelled:
        dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values, df["Pawpularity"].values))
    else:
        dataset = tf.data.Dataset.from_tensor_slices((df["Id"].values))
    
    dataset = dataset.map(image_read_fn, num_parallel_calls=AUTOTUNE)
    dataset = dataset.map(image_preprocess_fn, num_parallel_calls=AUTOTUNE) if augment else dataset
    dataset = dataset.shuffle(1024, reshuffle_each_iteration=True) if shuffle else dataset
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset


## === cell 4

trn = df.iloc[:9000]
val = df.iloc[9001:]

train = create_dataset(trn, Batch_size, is_labelled = True, augment = True, shuffle = True)
validation = create_dataset(val, Batch_size, is_labelled = True, augment = False, shuffle = False)
test = create_dataset(df_test, Batch_size, is_labelled = False, augment = False, shuffle=False)


## === cell 5
train


## === cell 8
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.utils import to_categorical


## === cell 9
def get_model03():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=(224,224,3))

    h = keras.layers.experimental.preprocessing.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="Conv_3")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)
    
    
    h = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name="Conv_5")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)
    

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(512, activation="relu")(h)

    output = keras.layers.Dense(1, activation="relu")(h)

    model = keras.Model(inpt, output)
    
    return model


## === cell 10
def get_model03():
    np.random.seed(0)
    random.seed(12)
    tf.random.set_seed(12)

    inpt = keras.Input(shape=(224, 224, 3))

    h = keras.layers.Rescaling(1.0 / 255)(inpt)

    h = keras.layers.Conv2D(32, kernel_size=(3, 3), activation="relu", name="Conv_1")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(64, kernel_size=(3, 3), activation="relu", name="Conv_3")(h)
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Conv2D(128, kernel_size=(3, 3), activation="relu", name="Conv_5")(
        h
    )
    h = keras.layers.MaxPool2D(pool_size=(2, 2))(h)

    h = keras.layers.Dropout(0.1)(h)

    h = keras.layers.Flatten()(h)
    h = keras.layers.Dense(512, activation="relu")(h)

    output = keras.layers.Dense(1, activation="relu")(h)

    model = keras.Model(inpt, output)

    return model


## === cell 11

early_stopping = EarlyStopping(patience = 5,restore_best_weights=True)


lr_schedule = ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100, decay_rate=0.96,
    staircase=True)


## === cell 12

model.compile(loss="mse", 
              optimizer = tf.keras.optimizers.Adam(learning_rate = lr_schedule), 
              metrics=[tf.keras.metrics.RootMeanSquaredError()])


## --- ERROR in cell 12, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mNameError[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/1355172600.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m      1[0m [0;31m# Compiling and Fitting the model[0m[0;34m[0m[0;34m[0m[0m
[1;32m      2[0m [0;34m[0m[0m
[0;32m----> 3[0;31m model.compile(loss="mse", 
[0m[1;32m      4[0m               [0moptimizer[0m [0;34m=[0m [0mtf[0m[0;34m.[0m[0mkeras[0m[0;34m.[0m[0moptimizers[0m[0;34m.[0m[0mAdam[0m[0;34m([0m[0mlearning_rate[0m [0;34m=[0m [0mlr_schedule[0m[0;34m)[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m               metrics=[tf.keras.metrics.RootMeanSquaredError()])

[0;31mNameError[0m: name 'model' is not defined

## === cell 13
model.summary()
