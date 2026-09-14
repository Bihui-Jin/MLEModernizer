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

- What this solution (achieved 0.28148) has done: 'I fix the TensorFlow import crash by removing the forced protobuf Python implementation setting that breaks TF’s protobuf bindings in this environment, while keeping the same TF/Keras workflow. Then I fix the missing model path issue by loading an available EfficientNet family model from `tf.keras.applications` (same inference-style pipeline) instead of a non-existent `/kaggle/input/effnet3/...` SavedModel, so the notebook can run end-to-end. I also correct the label handling: `flow_from_dataframe(..., class_mode="categorical")` cannot directly consume the space-delimited multi-label strings, so I build a proper multi-label one-hot matrix and use `class_mode="raw"` with a custom dataframe. Finally, I generate submission labels by thresholding sigmoid outputs into a space-delimited list (falling back to the top-1 label if none exceed the threshold) and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/97173096.py in <cell line: 0>()
----> 1 import tensorflow as tf
      2 from tensorflow import keras
      3 from tensorflow.keras import backend as K
      4 
      5 tf.random.set_seed(42)

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

## === cell 2
from sklearn.model_selection import train_test_split



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 6
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()



## === cell 7
all_labels = sorted(
    {lab for s in train["labels"].fillna("").values for lab in s.split(" ") if lab}
)
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
num_classes = len(all_labels)

labels_series = train["labels"].fillna("").astype(str)
row_counts = labels_series.str.count(" ").to_numpy() + (
    labels_series.ne("").to_numpy().astype(np.int32)
)

rows = np.repeat(np.arange(len(train), dtype=np.int32), row_counts.astype(np.int32))

labs_flat = labels_series.str.split(" ").to_list()
labs_flat = [
    lab for row in labs_flat for lab in row if lab
]  # flatten; still faster than per-row extends in loop
cols = np.fromiter(
    (label2idx[lab] for lab in labs_flat), dtype=np.int32, count=len(labs_flat)
)

y = np.zeros((len(train), num_classes), dtype=np.float32)
if len(rows):
    y[rows, cols] = 1.0

y_cols = [f"y_{lab}" for lab in all_labels]

train_ml = train[["image"]].copy()
for j, col in enumerate(y_cols):
    train_ml[col] = y[:, j]

train_ml.head()



## === cell 8
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 336)
BATCH_SIZE = 16

train_df, val_df = train_test_split(
    train_ml, test_size=0.15, random_state=42, shuffle=True
)

options = tf.data.Options()
options.deterministic = True
options.experimental_slack = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True

CACHE_DIR = "/kaggle/working/tf_cache_pp2021"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    return img


@tf.function
def _decode_resize_with_label_from_bytes(img_bytes, label):
    img = _decode_resize_from_bytes(img_bytes)
    return img, label


@tf.function
def _decode_resize_from_bytes_only(img_bytes):
    return _decode_resize_from_bytes(img_bytes)


def _make_train_ds(df):
    paths_np = np.array(
        [os.path.join(train_dir, f) for f in df["image"].values], dtype=object
    )
    labels_np = df[y_cols].to_numpy(dtype=np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(options)

    ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_read_bytes(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache(os.path.join(CACHE_DIR, "train_bytes_cache"))
    ds = ds.map(
        _decode_resize_with_label_from_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(df):
    paths_np = np.array(
        [os.path.join(train_dir, f) for f in df["image"].values], dtype=object
    )
    labels_np = df[y_cols].to_numpy(dtype=np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(options)

    ds = ds.map(
        lambda p, y: (_read_bytes(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.cache(os.path.join(CACHE_DIR, "val_bytes_cache"))
    ds = ds.map(
        _decode_resize_with_label_from_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = _make_train_ds(train_df)
val_generator = _make_val_ds(val_df)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1074529735.py in <cell line: 0>()
----> 1 AUTOTUNE = tf.data.AUTOTUNE
      2 IMG_SIZE = (224, 336)
      3 BATCH_SIZE = 16
      4 
      5 train_df, val_df = train_test_split(

NameError: name 'tf' is not defined

## === cell 9
def _make_test_ds(df):
    paths_np = np.array(
        [os.path.join(test_dir, f) for f in df["image"].values], dtype=object
    )
    ds = tf.data.Dataset.from_tensor_slices(paths_np).with_options(options)

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(os.path.join(CACHE_DIR, "test_bytes_cache"))
    ds = ds.map(
        _decode_resize_from_bytes_only, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_generator = _make_test_ds(test_df)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3435520120.py in <cell line: 0>()
     17 
     18 
---> 19 test_generator = _make_test_ds(test_df)
     20 

/tmp/ipykernel_11/3435520120.py in _make_test_ds(df)
      3         [os.path.join(test_dir, f) for f in df["image"].values], dtype=object
      4     )
----> 5     ds = tf.data.Dataset.from_tensor_slices(paths_np).with_options(options)
      6 
      7     # SPEED FIX (equivalent): Cache raw bytes before decode/resize.

NameError: name 'tf' is not defined

## === cell 10
data_aug = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(factor=20.0 / 360.0, fill_mode="reflect", seed=42),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=42
        ),
    ],
    name="data_augmentation",
)

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 336, 3),
    pooling="avg",
)

inp = keras.Input(shape=(224, 336, 3), name="image")
x = data_aug(inp)
x = keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
trained_model_sub = keras.Model(inp, out)

base.trainable = False
for layer in trained_model_sub.layers:
    if isinstance(layer, keras.layers.Dense):
        layer.trainable = True

trained_model_sub.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

trained_model_sub.summary()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/219180862.py in <cell line: 0>()
----> 1 data_aug = keras.Sequential(
      2     [
      3         keras.layers.RandomFlip("horizontal", seed=42),
      4         keras.layers.RandomRotation(factor=20.0 / 360.0, fill_mode="reflect", seed=42),
      5         keras.layers.RandomTranslation(

NameError: name 'keras' is not defined

## === cell 11
EPOCHS = 3
_ = trained_model_sub.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/940279227.py in <cell line: 0>()
      1 EPOCHS = 3
----> 2 _ = trained_model_sub.fit(
      3     train_generator,
      4     validation_data=val_generator,
      5     epochs=EPOCHS,

NameError: name 'trained_model_sub' is not defined

## === cell 12
y_pred = trained_model_sub.predict(test_generator, verbose=1)
y_pred = np.asarray(y_pred)
print("y_pred shape:", y_pred.shape)

threshold = 0.5
mask = y_pred >= threshold
any_pos = mask.any(axis=1)
argmax_idx = y_pred.argmax(axis=1)

pos_lists = [np.flatnonzero(row).tolist() for row in mask]
for i in np.flatnonzero(~any_pos):
    pos_lists[i] = [int(argmax_idx[i])]

pred_label_strs = [" ".join(idx2label[int(j)] for j in inds) for inds in pos_lists]

gen_images = test_df["image"].tolist()
sub = pd.DataFrame({"image": gen_images, "labels": pred_label_strs})
sub.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/686506965.py in <cell line: 0>()
----> 1 y_pred = trained_model_sub.predict(test_generator, verbose=1)
      2 y_pred = np.asarray(y_pred)
      3 print("y_pred shape:", y_pred.shape)
      4 
      5 threshold = 0.5

NameError: name 'trained_model_sub' is not defined

## === cell 13
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].nunique() == len(sub)
assert sub["labels"].isna().sum() == 0

sub = sub.set_index("image").reindex(sam_sub["image"]).reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3731404143.py in <cell line: 0>()
----> 1 assert list(sub.columns) == ["image", "labels"]
      2 assert sub["image"].nunique() == len(sub)
      3 assert sub["labels"].isna().sum() == 0
      4 
      5 sub = sub.set_index("image").reindex(sam_sub["image"]).reset_index()

NameError: name 'sub' is not defined
