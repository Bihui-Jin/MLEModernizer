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
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

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

# 5. Target score

18.474118485710264

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random
import numpy as np

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

train_mode = False  # is it training mode or submission mode

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

_WORKERS = max(2, (os.cpu_count() or 2) - 1)

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/1688230625.py in <cell line: 0>()
     16 from sklearn.model_selection import train_test_split
     17 
---> 18 import tensorflow as tf
     19 from tensorflow import keras
     20 from tensorflow.keras import models, layers

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
main_folder = "/kaggle/input/petfinder-pawpularity-score/"

test_image_folder = os.path.join(main_folder, "test")
test_meta = pd.read_csv(os.path.join(main_folder, "test.csv"))

test_meta["img_fnm"] = test_image_folder + "/" + test_meta["Id"].astype(str) + ".jpg"

if train_mode:
    train_image_folder = os.path.join(main_folder, "train")
    train_meta = pd.read_csv(os.path.join(main_folder, "train.csv"))
    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )




## === cell 2
if not train_mode:
    train_image_folder = os.path.join(main_folder, "train")
    train_meta = pd.read_csv(os.path.join(main_folder, "train.csv"))

    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )

    trn_df, val_df = train_test_split(
        train_meta, test_size=0.15, random_state=SEED, shuffle=True
    )
    trn_df = trn_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)




## === cell 3
target_size = 299

inceptionv3_pretrained = InceptionV3(
    input_shape=(target_size, target_size, 3), include_top=False, weights="imagenet"
)

inceptionv3_pretrained.trainable = False

model = models.Sequential()
model.add(inceptionv3_pretrained)
model.add(layers.Flatten())
model.add(Dropout(0.25))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=2e-5),
    metrics=["mse"],
    run_eagerly=False,
)

_ = model.summary()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2312655301.py in <cell line: 0>()
      1 target_size = 299
      2 
----> 3 inceptionv3_pretrained = InceptionV3(
      4     input_shape=(target_size, target_size, 3), include_top=False, weights="imagenet"
      5 )

NameError: name 'InceptionV3' is not defined

## === cell 4
EPOCH = 2
BATCH = 32

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=1.0, patience=50, restore_best_weights=True
)

steps_per_epoch = int(np.ceil(trn_df.shape[0] / BATCH))
validation_steps = int(np.ceil(val_df.shape[0] / BATCH))


def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [target_size, target_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


data_augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(factor=30.0 / 360.0, fill_mode="reflect", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augment",
)



def make_ds(df, training: bool):
    paths = df["img_fnm"].astype(str).values
    if training:
        labels = df["Pawpularity"].astype(np.float32).values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        labels = df["Pawpularity"].astype(np.float32).values
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

    def _map_fn(path, y):
        x = _decode_and_resize(path)
        if training:
            x = data_augment(x, training=True)
        return x, y

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(trn_df, training=True)
val_ds = make_ds(val_df, training=False)

log = model.fit(
    x=train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=EPOCH,
    callbacks=[early_stopping],
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1588181595.py in <cell line: 0>()
      6 BATCH = 32
      7 
----> 8 early_stopping = EarlyStopping(
      9     monitor="val_loss", min_delta=1.0, patience=50, restore_best_weights=True
     10 )

NameError: name 'EarlyStopping' is not defined

## === cell 5
model.optimizer.learning_rate = 0.5e-6

log2 = model.fit(
    x=train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=2,
    callbacks=[early_stopping],
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/725312523.py in <cell line: 0>()
----> 1 model.optimizer.learning_rate = 0.5e-6
      2 
      3 log2 = model.fit(
      4     x=train_ds,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'model' is not defined

## === cell 6
local_model_path = "InceptionV3_2_64FC_191121.h5"
model.save(local_model_path)
print("Saved model to:", local_model_path)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1308568434.py in <cell line: 0>()
      1 local_model_path = "InceptionV3_2_64FC_191121.h5"
----> 2 model.save(local_model_path)
      3 print("Saved model to:", local_model_path)
      4 
      5 

NameError: name 'model' is not defined

## === cell 7
if not train_mode:
    pass




## === cell 8
if not train_mode:
    print("Test rows:", len(test_meta))
    print(
        "Example test image path exists:", os.path.exists(test_meta.loc[0, "img_fnm"])
    )




## === cell 9
if not train_mode:
    test_paths = test_meta["img_fnm"].astype(str).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    test_ds = test_ds.with_options(options)

    def _map_test(path):
        return _decode_and_resize(path)

    test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)
    test_ds = test_ds.batch(BATCH, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE)

    steps = int(np.ceil(test_meta.shape[0] / BATCH))

    pred = model.predict(
        test_ds,
        steps=steps,
        verbose=1,
    )

    pred = np.asarray(pred).reshape(-1)

    if len(pred) != len(test_meta):
        pred = pred[: len(test_meta)]

    pred = np.clip(pred, 0.0, 100.0)

    test_meta["Pawpularity"] = pred
    submission_df = test_meta[["Id", "Pawpularity"]].copy()
    submission_df.to_csv("submission.csv", index=False)

    print(submission_df.head())
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/102600061.py in <cell line: 0>()
      3     # enable parallel decode/resize/prefetch. Semantics identical: rescale 1/255 and predict in order.
      4     test_paths = test_meta["img_fnm"].astype(str).values
----> 5     test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
      6 
      7     options = tf.data.Options()

NameError: name 'tf' is not defined
