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

0.5480507706255666

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Main runtime is dominated by CPU image decoding/resizing inside `Sequence.__getitem__` (Python loop + cv2 I/O) and by extra overhead during prediction. To keep identical model/training logic while cutting wall time, I switch both train/test pipelines to `tf.data` with parallel JPEG decode/resize, cached file-path tensors, deterministic options, and `prefetch(AUTOTUNE)` so the model is continuously fed. I also eliminate repeated pandas `.loc` lookups in inner loops by precomputing image paths/labels arrays (equivalent data, less Python overhead), and I keep batch size/epochs/model architecture unchanged. These changes are provably equivalent in semantics (same files, same resizing to 224×224, same normalization /255, same labels), but significantly reduce input pipeline bottlenecks to fit within 600s.'
- What this solution (achieved 0.11584) has done: 'Main bottlenecks are (1) training a full ResNet50 for an epoch on CPU (very slow) when no external weights exist, and (2) expensive per-image decoding/resizing inside the input pipeline without caching. To fit the 600s budget while preserving the exact core model/training logic and semantics, the script below makes the pipeline faster and avoids the “train-from-scratch” path by requiring the provided external weights (the intended fast inference path for this kind of submission script). For inference speed, it also switches TFRecord parsing to return `(img, name)` so Keras doesn’t do extra structure handling, enables dataset caching for test (safe because test is iterated once per weight file), and uses a vectorized name-to-index mapping via a prebuilt dict to avoid repeated pandas indexing overhead. All changes are strictly runtime-focused and keep deterministic settings and prediction semantics the same.'
- What this solution (achieved 0.11584) has done: 'Main bottlenecks are (1) training a full ResNet50 for 200 steps (unnecessary for producing test predictions and very slow on CPU) and (2) iterating the TFRecord test set twice (once for images and once for names), forcing a second full decode/resize pass. The optimized version keeps the exact same model/processing/prediction semantics, but removes the redundant training when no external weights exist (equivalent because predictions were otherwise from random/partially-trained weights) and makes the TFRecord test pipeline yield `(image, name)` in a single pass so decode happens once. It also replaces the expensive Python dict-based reordering with a vectorized merge against `sample_submission` for identical ordering while avoiding Python loops.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
from math import ceil

from tensorflow.keras.layers import Dense, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.applications.resnet50 import ResNet50, preprocess_input
from tensorflow.keras.optimizers import Adam

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF decide
except Exception:
    pass

tf.keras.backend.clear_session()
print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gm_exp = tf.Variable(3.0, dtype=tf.float32)


def generalized_mean_pool_2d(X):
    pool = (
        tf.reduce_mean(tf.abs(X ** (gm_exp)), axis=[1, 2], keepdims=False) + 1.0e-7
    ) ** (1.0 / gm_exp)
    return pool


def create_model(input_shape):
    inp = Input(shape=input_shape)

    x_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_tensor=inp,
        pooling=None,
        classes=None,
    )
    for layer in x_model.layers:
        layer.trainable = True

    lambda_layer = Lambda(generalized_mean_pool_2d)
    lambda_layer.trainable_weights.extend([gm_exp])

    x = lambda_layer(x_model.output)
    out = Dense(5, activation="softmax", name="plan_diseases")(x)
    model = Model(inputs=x_model.input, outputs=out)
    return model




## === cell 2
path = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(path, "train.csv")
sample_sub_path = os.path.join(path, "sample_submission.csv")
train_img_dir = os.path.join(path, "train_images") + "/"
test_img_dir = os.path.join(path, "test_images") + "/"
train_tfrec_dir = os.path.join(path, "train_tfrecords")
test_tfrec_dir = os.path.join(path, "test_tfrecords")

train_df = pd.read_csv(train_csv_path)
sample_submission = pd.read_csv(sample_sub_path)

print("train_df:", train_df.shape, "sample_submission:", sample_submission.shape)
print(train_df.head())




## === cell 3
IMG_H = 224
IMG_W = 224

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function(reduce_retracing=True)
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function(reduce_retracing=True)
def _parse_test_example_with_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    name = ex["image_name"]
    return img, name


@tf.function(reduce_retracing=True)
def _parse_image_label_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


def _list_tfrecords(dir_path, prefix):
    if not tf.io.gfile.exists(dir_path):
        return []
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    return sorted(files)


def _build_dataset_from_tfrecords(
    tfrecord_files,
    batch_size=16,
    training=False,
    seed=42,
    include_names=False,
):
    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    options.experimental_slack = True
    ds = ds.with_options(options)

    if training:
        ds = ds.map(
            _parse_train_example,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds = ds.shuffle(2048, seed=seed, reshuffle_each_iteration=True)
        ds = ds.repeat()
    else:
        ds = ds.map(
            _parse_test_example_with_name,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        if not include_names:
            ds = ds.map(
                lambda img, name: img,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


def _build_image_label_dataset_from_tfrecords(
    tfrecord_files,
    batch_size=16,
):
    ds = tf.data.TFRecordDataset(
        tfrecord_files,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=8 * 1024 * 1024,
    )

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    options.experimental_slack = True
    ds = ds.with_options(options)

    ds = ds.map(
        _parse_image_label_example,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_preprocess_from_bytes(img_bytes)


def _build_dataset_from_paths(
    image_paths,
    labels=None,
    batch_size=16,
    shuffle=False,
    seed=42,
):
    image_paths = tf.convert_to_tensor(image_paths, dtype=tf.string)

    if labels is not None:
        labels = tf.convert_to_tensor(labels, dtype=tf.int32)
        ds = tf.data.Dataset.from_tensor_slices((image_paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(image_paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.threading.private_threadpool_size = 0
    options.threading.max_intra_op_parallelism = 0
    options.experimental_slack = True
    ds = ds.with_options(options)

    if shuffle:
        buffer_size = int(image_paths.shape[0])
        ds = ds.shuffle(
            buffer_size=buffer_size,
            seed=seed,
            reshuffle_each_iteration=True,
        )

    if labels is not None:

        def _map_fn(path, y):
            return _decode_resize_preprocess(path), y

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    else:

        def _map_img(path):
            return _decode_resize_preprocess(path)

        ds = ds.map(_map_img, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
path_weights = "../input/weights/"
has_external_weights = os.path.isdir(path_weights) and len(os.listdir(path_weights)) > 0
print("External weights found:", has_external_weights)

test_tfrecord_files = _list_tfrecords(test_tfrec_dir, "ld_test")
train_tfrecord_files = _list_tfrecords(train_tfrec_dir, "ld_train")
print(
    "Found test tfrecords:",
    len(test_tfrecord_files),
    "train tfrecords:",
    len(train_tfrecord_files),
)




## === cell 5
input_shape = (224, 224, 3)
batch_size = 16

X_test = sample_submission.copy()
if "label" in X_test.columns:
    X_test = X_test.drop(columns=["label"])

use_tfrecord_test = len(test_tfrecord_files) > 0
if use_tfrecord_test:
    ds_test = _build_dataset_from_tfrecords(
        test_tfrecord_files,
        batch_size=batch_size,
        training=False,
        seed=SEED,
        include_names=True,
    )
else:
    test_image_paths = (test_img_dir + X_test["image_id"].astype(str)).to_numpy()
    ds_test = _build_dataset_from_paths(
        test_image_paths,
        labels=None,
        batch_size=batch_size,
        shuffle=False,
        seed=SEED,
    )




## === cell 6
all_pred = []

model = create_model(input_shape)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
    run_eagerly=False,
    steps_per_execution=16,
)

if has_external_weights:
    models = sorted(os.listdir(path_weights))
    weight_paths = [os.path.join(path_weights, w) for w in models]
else:
    weight_paths = [None]

for wpath in weight_paths:
    if wpath is not None:
        model.load_weights(wpath)

    if use_tfrecord_test:
        preds_batches = []
        names_batches = []
        for img_batch, name_batch in ds_test:
            preds_batches.append(model.predict_on_batch(img_batch).numpy())
            names_batches.append(name_batch.numpy())

        preds = np.concatenate(preds_batches, axis=0)
        names = np.concatenate(names_batches, axis=0).astype("U")

        preds = preds[: sample_submission.shape[0]]
        names = names[: sample_submission.shape[0]]

        pred_df = pd.DataFrame({"image_id": names})
        pred_df["row_id"] = np.arange(pred_df.shape[0], dtype=np.int32)

        merged = sample_submission[["image_id"]].merge(
            pred_df, on="image_id", how="left", sort=False, validate="one_to_one"
        )
        row_idx = merged["row_id"].to_numpy(np.int64)
        preds = preds[row_idx]
    else:
        preds = model.predict(ds_test, verbose=1)
        preds = preds[: sample_submission.shape[0]]

    all_pred.append(preds)

if len(all_pred) == 0:
    raise RuntimeError(
        "No predictions were generated (all_pred is empty); cannot build submission."
    )




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2493531432.py in <cell line: 0>()
     36         names_batches = []
     37         for img_batch, name_batch in ds_test:
---> 38             preds_batches.append(model.predict_on_batch(img_batch).numpy())
     39             names_batches.append(name_batch.numpy())
     40 

AttributeError: 'numpy.ndarray' object has no attribute 'numpy'

## === cell 7
sum_pred = np.mean(np.stack(all_pred, axis=0), axis=0)

diagnos = np.argmax(sum_pred, axis=1).astype(int).tolist()

sample_submission["label"] = diagnos
print(sample_submission.head())
print("Label value counts:\n", sample_submission["label"].value_counts())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2742660750.py in <cell line: 0>()
----> 1 sum_pred = np.mean(np.stack(all_pred, axis=0), axis=0)
      2 
      3 diagnos = np.argmax(sum_pred, axis=1).astype(int).tolist()
      4 
      5 sample_submission["label"] = diagnos

/usr/local/lib/python3.11/dist-packages/numpy/core/shape_base.py in stack(arrays, axis, out, dtype, casting)
    443     arrays = [asanyarray(arr) for arr in arrays]
    444     if not arrays:
--> 445         raise ValueError('need at least one array to stack')
    446 
    447     shapes = {arr.shape for arr in arrays}

ValueError: need at least one array to stack

## === cell 8
sample_submission[["image_id", "label"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", sample_submission[["image_id", "label"]].shape
)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
