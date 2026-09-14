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

0.8674826231489876

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"

os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
import json
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(123)
np.random.seed(123)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("tf:", tf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(train.head())
print("train shape:", train.shape)



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)
classes



## === cell 4
train = train.copy()
train["class"] = train["label"].astype(str).map(classes)

train["path"] = (TRAIN_PATH + "/" + train["image_id"].astype(str)).astype("string")

train = train.astype(
    {"image_id": "string", "label": "string", "path": "string", "class": "string"}
)
train_df, val_df = train_test_split(
    train,
    test_size=0.05,
    random_state=100,
    stratify=train["label"].values,
)
print("train_df:", train_df.shape, "val_df:", val_df.shape)



## === cell 5
batch_size = 4
IMG_SIZE = (512, 512)

AUTOTUNE = tf.data.AUTOTUNE

CLASS_NAMES = [str(i) for i in sorted(train["label"].astype(int).unique().tolist())]
num_classes = len(CLASS_NAMES)
CLASS_TABLE = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(CLASS_NAMES),
        values=tf.constant(list(range(num_classes)), dtype=tf.int32),
    ),
    default_value=-1,
)

_TARGET_SIZE = tf.constant(list(IMG_SIZE), dtype=tf.int32)


@tf.function(jit_compile=True)
def _decode_resize_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, _TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


@tf.function(jit_compile=True)
def _label_to_onehot(label_s):
    y = CLASS_TABLE.lookup(label_s)
    y = tf.one_hot(y, depth=num_classes, dtype=tf.float32)
    return y


@tf.function(jit_compile=True)
def _augment_tf_stateless(x, seed2):
    x = tf.image.stateless_random_flip_left_right(x, seed2)
    seed2 = seed2 + tf.constant([1, 0], tf.int32)
    x = tf.image.stateless_random_flip_up_down(x, seed2)
    seed2 = seed2 + tf.constant([1, 0], tf.int32)

    k = tf.random.stateless_uniform([], seed2, minval=0, maxval=4, dtype=tf.int32)
    x = tf.image.rot90(x, k)
    seed2 = seed2 + tf.constant([1, 0], tf.int32)

    do_transpose = (
        tf.random.stateless_uniform([], seed2, minval=0.0, maxval=1.0, dtype=tf.float32)
        < 0.5
    )
    x = tf.cond(do_transpose, lambda: tf.transpose(x, perm=[1, 0, 2]), lambda: x)
    return x


def _make_dataset(df, training: bool, cache_path: str):
    paths = tf.constant(df["path"].astype(str).values)
    labels_str = tf.constant(df["label"].astype(str).values)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels_str))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    if training:
        shuffle_buf = int(min(len(df), 2048))
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=123, reshuffle_each_iteration=True
        )

    def _map_decode_and_label(path, label_s):
        x = _decode_resize_from_path(path)
        y = _label_to_onehot(label_s)
        return x, y

    ds = ds.map(_map_decode_and_label, num_parallel_calls=AUTOTUNE)


    if training:
        ds = ds.enumerate()

        def _map_aug(i, data):
            x, y = data
            seed2 = tf.stack([tf.cast(123, tf.int32), tf.cast(i, tf.int32)], axis=0)
            x = _augment_tf_stateless(x, seed2)
            return x, y

        ds = ds.map(_map_aug, num_parallel_calls=AUTOTUNE)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_cache = os.path.join(OUTPUT_DIR, "cache_train_512")
val_cache = os.path.join(OUTPUT_DIR, "cache_val_512")
train_gen = _make_dataset(train_df, training=True, cache_path=train_cache)
val_gen = _make_dataset(val_df, training=False, cache_path=val_cache)

print("Num classes inferred:", num_classes)
print("Class mapping:", {k: i for i, k in enumerate(CLASS_NAMES)})



## === cell 6
MODEL_PATH = "/kaggle/input/pass-4/weightEffnetB4_v6.h5"


def build_fallback_model(num_classes: int, img_size=(512, 512)):
    inputs = keras.Input(shape=(img_size[0], img_size[1], 3))
    base = keras.applications.EfficientNetB0(
        include_top=False, weights="imagenet", input_tensor=inputs
    )
    x = layers.GlobalAveragePooling2D()(base.output)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model


loaded_model = None
if os.path.exists(MODEL_PATH):
    try:
        try:
            tf.keras.config.enable_unsafe_deserialization()
        except Exception:
            pass

        loaded_model = tf.keras.models.load_model(MODEL_PATH, compile=False)
        print("Loaded external model:", MODEL_PATH)
    except Exception as e:
        print("Failed to load external model; will train fallback. Error:", repr(e))
        loaded_model = None
else:
    print("External model path not found; will train fallback:", MODEL_PATH)

if loaded_model is None:
    loaded_model = build_fallback_model(num_classes, IMG_SIZE)
    loaded_model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=2,
        verbose=1,
    )

print("Model input shape:", loaded_model.input_shape)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/840195334.py in <cell line: 0>()
     38 if loaded_model is None:
     39     loaded_model = build_fallback_model(num_classes, IMG_SIZE)
---> 40     loaded_model.fit(
     41         train_gen,
     42         validation_data=val_gen,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 7
sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
test_ids = sub["image_id"].tolist()

input_shape = loaded_model.input_shape
if isinstance(input_shape, list):
    input_shape = input_shape[0]
req_h, req_w = int(input_shape[1]), int(input_shape[2])
target_size = (req_h, req_w)

test_paths = (TEST_PATH + "/" + sub["image_id"].astype(str)).values

_TEST_TARGET_SIZE = tf.constant([req_h, req_w], dtype=tf.int32)


@tf.function(jit_compile=True)
def _decode_resize_test_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, _TEST_TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _make_test_dataset(paths, batch=128, cache_path=None):
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths))
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_resize_test_from_path, num_parallel_calls=AUTOTUNE)


    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_cache = os.path.join(OUTPUT_DIR, f"cache_test_{req_h}x{req_w}")
test_ds = _make_test_dataset(test_paths, batch=128, cache_path=test_cache)

proba = loaded_model.predict(test_ds, verbose=0)
preds = np.argmax(proba, axis=1).astype(int).tolist()

print("preds:", len(preds), "test_ids:", len(test_ids))



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
UnboundLocalError                         Traceback (most recent call last)
/tmp/ipykernel_11/3404431153.py in <cell line: 0>()
     48 test_ds = _make_test_dataset(test_paths, batch=128, cache_path=test_cache)
     49 
---> 50 proba = loaded_model.predict(test_ds, verbose=0)
     51 preds = np.argmax(proba, axis=1).astype(int).tolist()
     52 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/trainer.py in predict(self, x, batch_size, verbose, steps, callbacks)
    567         callbacks.on_predict_end()
    568         outputs = tree.map_structure_up_to(
--> 569             batch_outputs, potentially_ragged_concat, outputs
    570         )
    571         return tree.map_structure(convert_to_np_if_not_ragged, outputs)

UnboundLocalError: cannot access local variable 'batch_outputs' where it is not associated with a value

## === cell 8
submission = pd.DataFrame({"image_id": test_ids, "label": preds})
submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)

print(submission.head())
print("Wrote:", os.path.join(OUTPUT_DIR, "submission.csv"), "rows:", len(submission))
print("Columns:", submission.columns.tolist())

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2249253448.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_ids, "label": preds})
      2 submission.to_csv(os.path.join(OUTPUT_DIR, "submission.csv"), index=False)
      3 
      4 print(submission.head())
      5 print("Wrote:", os.path.join(OUTPUT_DIR, "submission.csv"), "rows:", len(submission))

NameError: name 'preds' is not defined
