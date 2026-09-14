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

0.6128739800543971

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The main bottleneck is input pipeline overhead (JPEG decode/resize on CPU) plus extra dataset passes during test-time name extraction; this can easily push end-to-end runtime over 600s at 512px. I keep the exact same model, loss, epochs, and data semantics, but speed up by (1) using the TFRecord-provided `height/width` to decode at a smaller scale before the final resize (provably equivalent to “decode full then resize” up to negligible interpolation differences), (2) enabling nondeterministic parallelism only for training while keeping eval/test deterministic, and (3) avoiding a second full pass over the test dataset by collecting `image_name` in the same pipeline pass used for prediction. I also set TF data options to reduce unnecessary overhead and keep everything deterministic where it affects evaluation outputs.'
- What this solution (achieved 0.11584) has done: 'I fix the environment-crashing import issue by removing the unnecessary standalone `keras` import (it can trigger a protobuf incompatibility in Kaggle images) and rely consistently on `tf.keras`. Then I fix the TFRecord JPEG decoder error by making `ratio` a Python `int` (the API requires a compile-time integer, not a Tensor), while keeping the same decode/resize/preprocess semantics. Finally, I ensure the test prediction path runs end-to-end and writes a valid `submission.csv` with the required `image_id,label` columns.'

# 9. Code solution

## === cell 0
import os
import glob
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import tensorflow.keras as k
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

print("TF:", tf.__version__)
print("TF-Keras:", tf.keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
image_size = 512
batch_size = 8
epochs = 2  # keep as provided

train_df = pd.read_csv(TRAIN_CSV, usecols=["label"])
num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE

ds_opts_train = tf.data.Options()
ds_opts_train.experimental_deterministic = False
try:
    ds_opts_train.threading.private_threadpool_size = 16
except Exception:
    pass

ds_opts_eval = tf.data.Options()
ds_opts_eval.experimental_deterministic = True
try:
    ds_opts_eval.threading.private_threadpool_size = 16
except Exception:
    pass

FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
}

JPEG_DECODE_RATIO = 1


@tf.function
def _decode_resize_preprocess(img_bytes, height, width):
    h = tf.cast(height, tf.int32)
    w = tf.cast(width, tf.int32)

    img = tf.io.decode_and_crop_jpeg(
        img_bytes,
        crop_window=[0, 0, h, w],  # full image, no cropping
        channels=3,
        ratio=JPEG_DECODE_RATIO,
        dct_method="INTEGER_FAST",
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img.set_shape([image_size, image_size, 3])
    img = effnet_preprocess(img)
    img.set_shape([image_size, image_size, 3])
    return img


@tf.function
def _parse_name_img_label(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    name = ex["image_name"]
    img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    label = tf.cast(ex["target"], tf.int32)
    return name, img, label


@tf.function
def _drop_name(name, img, label):
    return img, label


@tf.function
def _parse_name_img_test(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    name = ex["image_name"]
    img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    return name, img


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_tfrecs))
n_val_files = max(1, int(round(0.15 * len(train_tfrecs))))
val_file_idx = np.sort(perm[:n_val_files])
train_file_idx = np.sort(perm[n_val_files:])

train_tfrecs_split = [train_tfrecs[i] for i in train_file_idx]
val_tfrecs_split = [train_tfrecs[i] for i in val_file_idx]

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))


def _maybe_prefetch_to_device(ds):
    if _HAS_GPU:
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            pass
    return ds


def _make_ds_from_tfrecs(tfrecs, is_train: bool):
    opts = ds_opts_train if is_train else ds_opts_eval

    files = tf.data.Dataset.from_tensor_slices(tfrecs).with_options(opts)
    cycle_len = min(8, max(1, len(tfrecs)))

    def _make_tfr(x):
        return tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE)

    ds = files.interleave(
        _make_tfr,
        cycle_length=cycle_len,
        num_parallel_calls=AUTOTUNE,
        deterministic=not is_train,
    )

    ds = ds.map(
        _parse_name_img_label, num_parallel_calls=AUTOTUNE, deterministic=not is_train
    )

    if is_train:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=not is_train)

    if not is_train:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=is_train)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_ds = _make_ds_from_tfrecs(train_tfrecs_split, is_train=True)
val_ds = _make_ds_from_tfrecs(val_tfrecs_split, is_train=False)

inp = Input(shape=(image_size, image_size, 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(x)
model = Model(inputs=inp, outputs=out)

try:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
except TypeError:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)

models = [model]




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2984023671.py in <cell line: 0>()
    173 model.summary()
    174 
--> 175 history = model.fit(
    176     train_ds,
    177     validation_data=val_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::ParallelMapV2: Feature: height (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_54332]

## === cell 2
sub = pd.read_csv(SAMPLE_SUB)

raw_test = tf.data.Dataset.from_tensor_slices(test_tfrecs).with_options(ds_opts_eval)
cycle_len = min(8, max(1, len(test_tfrecs)))

raw_test = raw_test.interleave(
    lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
    cycle_length=cycle_len,
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_ds = (
    raw_test.map(_parse_name_img_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
test_ds = _maybe_prefetch_to_device(test_ds)

probs_mean = models[0].predict(
    test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE), verbose=1
)

name_batches = []
for nb, _ in test_ds:
    name_batches.append(nb.numpy())
name_bytes = np.concatenate(name_batches, axis=0)
test_names = np.char.decode(name_bytes.astype("S"), "utf-8")

pred_labels = np.argmax(probs_mean, axis=1).astype(int)
pred_df = pd.DataFrame({"image_id": test_names, "label": pred_labels})

sub = sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert sub["label"].notna().all(), "Some test images were not found in TFRecords"
sub["label"] = sub["label"].astype(int)

assert len(sub) == 2676, f"Unexpected test rows: {len(sub)}"
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub.columns.tolist()}"
assert sub["label"].between(0, 4).all(), "Labels must be in [0,4]"

sub.head()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3001248176.py in <cell line: 0>()
     19 test_ds = _maybe_prefetch_to_device(test_ds)
     20 
---> 21 probs_mean = models[0].predict(
     22     test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE), verbose=1
     23 )

NameError: name 'models' is not defined

## === cell 3
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
