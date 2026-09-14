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

if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION") == "python":
    os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # matches EfficientNet preprocessing
    return image, label


def load_and_preprocess_image_nolabel(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)
    return image


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

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

train_paths = train["path"].values
train_labels_np = train["label_encoded"].values.astype(np.int64, copy=False)
valid_paths = valid["path"].values
valid_labels_np = valid["label_encoded"].values.astype(np.int64, copy=False)

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_np))
valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_np))

options = tf.data.Options()
options.experimental_deterministic = True

BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224


_TFREC_FEATURE_DESC_LABELED = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TFREC_FEATURE_DESC_UNLABELED = {"image": tf.io.FixedLenFeature([], tf.string)}


def _parse_tfrec(example_proto, labeled=True):
    feature_desc = (
        _TFREC_FEATURE_DESC_LABELED if labeled else _TFREC_FEATURE_DESC_UNLABELED
    )
    ex = tf.io.parse_single_example(example_proto, feature_desc)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE])
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    if labeled:
        return img, tf.cast(ex["target"], tf.int64)
    return img


def _tfrec_files(pattern):
    return tf.io.gfile.glob(pattern)


train_tfrec_files = sorted(
    _tfrec_files(
        "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec"
    )
)
test_tfrec_files = sorted(
    _tfrec_files(
        "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec"
    )
)

USE_TFRECORDS = (len(train_tfrec_files) > 0) and (len(test_tfrec_files) > 0)
print(
    "Using TFRecords:",
    USE_TFRECORDS,
    "| train tfrecs:",
    len(train_tfrec_files),
    "| test tfrecs:",
    len(test_tfrec_files),
)

_tfrec_ds_kwargs = dict(
    num_parallel_reads=AUTOTUNE,
    buffer_size=64 * 1024 * 1024,  # 64MB per file reader
)

if USE_TFRECORDS:
    train_ds = (
        tf.data.TFRecordDataset(train_tfrec_files, **_tfrec_ds_kwargs)
        .map(lambda x: _parse_tfrec(x, labeled=True), num_parallel_calls=AUTOTUNE)
        .shuffle(2048, seed=42, reshuffle_each_iteration=True)
        .cache()
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    valid_ds = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels_np))
    valid_ds = (
        valid_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()  # safe: deterministic, finite, reused
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
else:
    train_ds = (
        train_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    valid_ds = (
        valid_ds.map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()  # safe: deterministic, finite, reused
        .prefetch(AUTOTUNE)
        .with_options(options)
    )

print("Train/Valid sizes:", len(train), len(valid))
print("Num classes:", len(le.classes_))




## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetB0, EfficientNetB1, EfficientNetB2

NUM_CLASSES = len(le.classes_)
INPUT_SHAPE = (224, 224, 3)


def build_base_model(backbone_fn, input_shape=INPUT_SHAPE, num_classes=NUM_CLASSES):
    inp = layers.Input(shape=input_shape)
    base = backbone_fn(
        include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
    )
    x = layers.Dropout(0.2)(base.output)
    out = layers.Dense(num_classes, activation="softmax")(x)
    model = Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


cropnet_model = build_base_model(
    EfficientNetB2
)  # strongest -> will be weighted highest
densenet_model = build_base_model(EfficientNetB1)  # mid
efficientnet_model = build_base_model(EfficientNetB0)  # light

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1

print(
    "Built base models:",
    [m.name for m in [cropnet_model, densenet_model, efficientnet_model]],
)




## === cell 4
history = cropnet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=8,
    callbacks=[early_stopping, learning_rate_reduction],
    verbose=1,
)

_ = densenet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)
_ = efficientnet_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)




## === cell 5
from sklearn.linear_model import LogisticRegression


def to_probs(pred_output):
    """
    Keras model output is typically an ndarray; keep compatibility with dict/tensor cases.
    Return a numpy array of probabilities/logits as (batch, num_classes).
    """
    if isinstance(pred_output, dict):
        if "output_0" in pred_output:
            arr = pred_output["output_0"]
        else:
            arr = next(iter(pred_output.values()))
    else:
        arr = pred_output
    return np.asarray(arr)


if USE_TFRECORDS:
    train_meta_ds = (
        train_ds.unbatch().batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    )
    train_images_ds = train_meta_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)

    soft_voting_labels = np.concatenate([y.numpy() for _, y in train_meta_ds], axis=0)
else:
    train_meta_ds = (
        tf.data.Dataset.from_tensor_slices((train_paths, train_labels_np))
        .map(load_and_preprocess_image, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
    train_images_ds = train_meta_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE)
    soft_voting_labels = train_labels_np

cropnet_probs_all = to_probs(cropnet_model.predict(train_images_ds, verbose=0))
densenet_probs_all = to_probs(densenet_model.predict(train_images_ds, verbose=0))
efficientnet_probs_all = to_probs(
    efficientnet_model.predict(train_images_ds, verbose=0)
)

soft_voting_features = (
    cropnet_weight * cropnet_probs_all
    + densenet_weight * densenet_probs_all
    + efficientnet_weight * efficientnet_probs_all
)

meta_model = LogisticRegression(max_iter=2000, multi_class="multinomial", n_jobs=-1)
meta_model.fit(soft_voting_features, soft_voting_labels)

print("Meta-model trained on features:", soft_voting_features.shape)




## === cell 6
from sklearn.metrics import accuracy_score

meta_model_predictions = meta_model.predict(soft_voting_features)
accuracy = accuracy_score(soft_voting_labels, meta_model_predictions)
print("Meta-model (Logistic Regression) Train Accuracy:", accuracy)




## === cell 7
sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"

image_names = sample_sub["image_id"].tolist()
test_paths = [os.path.join(image_dir, fn) for fn in image_names]

if USE_TFRECORDS:
    test_ds = (
        tf.data.TFRecordDataset(test_tfrec_files, **_tfrec_ds_kwargs)
        .map(lambda x: _parse_tfrec(x, labeled=False), num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(options)
    )
else:
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = (
        test_ds.map(load_and_preprocess_image_nolabel, num_parallel_calls=AUTOTUNE)
        .batch(BATCH_SIZE, drop_remainder=False)
        .cache()
        .prefetch(AUTOTUNE)
        .with_options(options)
    )

cropnet_test = to_probs(cropnet_model.predict(test_ds, verbose=0))
densenet_test = to_probs(densenet_model.predict(test_ds, verbose=0))
efficientnet_test = to_probs(efficientnet_model.predict(test_ds, verbose=0))

soft_voting_test = (
    cropnet_weight * cropnet_test
    + densenet_weight * densenet_test
    + efficientnet_weight * efficientnet_test
)

predictions = meta_model.predict(soft_voting_test).astype(int).tolist()

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)

assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be image_id,label."
assert out_path.endswith(".csv")
