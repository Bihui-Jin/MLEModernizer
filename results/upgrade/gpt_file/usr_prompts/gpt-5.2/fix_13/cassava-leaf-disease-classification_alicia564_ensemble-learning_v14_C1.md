# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.13

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

try:
    tf.config.optimizer.set_jit(True)  # XLA JIT (speed; numerically equivalent)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.keras.utils.set_random_seed(42)

print("TF version:", tf.__version__)



## === cell 1
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = (
    LabelEncoder().fit_transform(train_csv["disease"]).astype(np.int32)
)

train_csv["disease"] = train_csv["disease"].astype(str)
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

print("Train/valid sizes:", len(train), len(valid))



## === cell 2
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input
from tensorflow.keras.applications import EfficientNetB0, DenseNet169, MobileNetV2

NUM_CLASSES = 5
IMG_SIZE = (224, 224)


def build_classifier(
    backbone_fn,
    input_shape=(224, 224, 3),
    num_classes=5,
    weights="imagenet",
    pooling="avg",
):
    inp = Input(shape=input_shape)
    backbone = backbone_fn(
        include_top=False, weights=weights, input_tensor=inp, pooling=pooling
    )
    x = backbone.output
    out = Dense(num_classes, activation="softmax")(x)
    return Model(inputs=inp, outputs=out)


cropnet_model = build_classifier(
    EfficientNetB0, input_shape=(224, 224, 3), num_classes=NUM_CLASSES
)
old_densenet_model = build_classifier(
    DenseNet169, input_shape=(224, 224, 3), num_classes=NUM_CLASSES
)
efficientnet_model = build_classifier(
    MobileNetV2, input_shape=(224, 224, 3), num_classes=NUM_CLASSES
)

print("Models built.")



## === cell 3
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_preprocess,
)

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

TFREC_TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TFREC_TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

_train_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TFREC_TRAIN_DIR, "*.tfrec")))
_test_tfrec_files = sorted(tf.io.gfile.glob(os.path.join(TFREC_TEST_DIR, "*.tfrec")))

_data_opts = tf.data.Options()
_data_opts.experimental_deterministic = True
try:
    _data_opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) - 1)
except Exception:
    pass
try:
    _data_opts.autotune.enabled = True
except Exception:
    pass
try:
    _data_opts.autotune.cpu_budget = os.cpu_count() or 8
except Exception:
    pass

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function(reduce_retracing=True)
def _parse_tfrec_train(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)  # identical preprocessing
    label = tf.cast(ex["label"], tf.int32)
    img_id = ex["image_name"]
    return img_id, img, label


_train_ids = tf.constant(train["image_id"].to_numpy(dtype=np.str_), dtype=tf.string)
_valid_ids = tf.constant(valid["image_id"].to_numpy(dtype=np.str_), dtype=tf.string)
_train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        _train_ids, tf.ones_like(_train_ids, dtype=tf.int32)
    ),
    default_value=0,
)
_valid_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        _valid_ids, tf.ones_like(_valid_ids, dtype=tf.int32)
    ),
    default_value=0,
)


@tf.function(reduce_retracing=True)
def _is_in_train(img_id, img, label):
    return _train_id_table.lookup(img_id) > 0


@tf.function(reduce_retracing=True)
def _is_in_valid(img_id, img, label):
    return _valid_id_table.lookup(img_id) > 0


@tf.function(reduce_retracing=True)
def _drop_id(img_id, img, label):
    return img, label


def make_ds_from_tfrecord(files, training=True):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        _data_opts
    )
    ds = ds.map(_parse_tfrec_train, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.filter(_is_in_train)
    else:
        ds = ds.filter(_is_in_valid)

    ds = ds.map(_drop_id, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(8192, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if not training:
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _decode_and_preprocess(path, label):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)  # identical preprocessing
    return img, tf.cast(label, tf.int32)


def make_ds_from_paths(df, training=True):
    paths = df["path"].to_numpy(dtype=np.str_)
    labels = df["label_encoded"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_data_opts)

    if training:
        ds = ds.shuffle(min(len(df), 8192), seed=42, reshuffle_each_iteration=True)

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    if not training:
        ds = ds.cache()

    ds = ds.prefetch(AUTOTUNE)
    return ds


if _train_tfrec_files:
    train_ds = make_ds_from_tfrecord(_train_tfrec_files, training=True)
    valid_ds = make_ds_from_tfrecord(_train_tfrec_files, training=False)
else:
    train_ds = make_ds_from_paths(train, training=True)
    valid_ds = make_ds_from_paths(valid, training=False)


def compile_and_fit(model, epochs=2, lr=1e-4):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=lr),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=[tf.keras.metrics.SparseCategoricalAccuracy(name="acc")],
        run_eagerly=False,  # graph mode for speed; same semantics
    )
    return model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=epochs,
        verbose=2,
    )


def set_backbone_trainable(model, trainable):
    for layer in model.layers:
        layer.trainable = trainable
    model.layers[-1].trainable = True


for m in [cropnet_model, old_densenet_model, efficientnet_model]:
    set_backbone_trainable(m, False)

print("Training heads...")
_ = compile_and_fit(cropnet_model, epochs=2, lr=3e-4)
_ = compile_and_fit(old_densenet_model, epochs=2, lr=3e-4)
_ = compile_and_fit(efficientnet_model, epochs=2, lr=3e-4)


def unfreeze_top_layers(model, n_layers):
    for layer in model.layers[-n_layers:]:
        layer.trainable = True


unfreeze_top_layers(cropnet_model, 30)
unfreeze_top_layers(old_densenet_model, 30)
unfreeze_top_layers(efficientnet_model, 30)

print("Fine-tuning...")
_ = compile_and_fit(cropnet_model, epochs=1, lr=1e-5)
_ = compile_and_fit(old_densenet_model, epochs=1, lr=1e-5)
_ = compile_and_fit(efficientnet_model, epochs=1, lr=1e-5)



## === cell 4
import numpy as np
import pandas as pd

image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
test_image_ids = sample_sub["image_id"].to_numpy(dtype=np.str_)

BATCH_SIZE = 64  # batching only; same predictions
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _load_decode_resize_preprocess(img_id):
    path = tf.strings.join([image_dir, "/", img_id])
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)
    return img_id, img


def _parse_tfrec_test(example_proto):
    features = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    ex = tf.io.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)
    img_id = ex["image_name"]
    return img_id, img


if _test_tfrec_files:
    files = _test_tfrec_files
    test_ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE).with_options(
        _data_opts
    )
    test_ds = (
        test_ds.map(_parse_tfrec_test, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_image_ids).with_options(
        _data_opts
    )
    test_ds = (
        test_ds.map(_load_decode_resize_preprocess, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1


@tf.function(jit_compile=True, reduce_retracing=True)
def _ensemble_predict(batch_imgs):
    cropnet_pred = cropnet_model(batch_imgs, training=False)
    densenet_pred = old_densenet_model(batch_imgs, training=False)
    efficientnet_pred = efficientnet_model(batch_imgs, training=False)
    avg_pred = (
        cropnet_weight * cropnet_pred
        + densenet_weight * densenet_pred
        + efficientnet_weight * efficientnet_pred
    )
    return tf.argmax(avg_pred, axis=1, output_type=tf.int32)


n_test = len(sample_sub)
out_labels = np.empty((n_test,), dtype=np.int32)
out_ids = np.empty((n_test,), dtype=object)

i = 0
for batch_ids, batch_imgs in test_ds:
    batch_classes = _ensemble_predict(batch_imgs)
    b = int(batch_classes.shape[0])
    out_ids[i : i + b] = batch_ids.numpy().astype("U")
    out_labels[i : i + b] = batch_classes.numpy()
    i += b

submission_df = pd.DataFrame(
    {"image_id": out_ids[:i].tolist(), "label": out_labels[:i].tolist()}
)
submission_df = sample_sub[["image_id"]].merge(submission_df, on="image_id", how="left")
submission_df["label"] = submission_df["label"].astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission file created: {submission_path}")
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
