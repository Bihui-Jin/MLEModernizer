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

# 5. Code solution

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
    weight_paths = []

    if len(train_tfrecord_files) > 0:
        ds_train = _build_dataset_from_tfrecords(
            train_tfrecord_files,
            batch_size=batch_size,
            training=True,
            seed=SEED,
            include_names=False,
        )
        steps_per_epoch = 200
    else:
        train_paths = (train_img_dir + train_df["image_id"].astype(str)).to_numpy()
        train_labels = train_df["label"].astype(np.int32).to_numpy()
        ds_train = _build_dataset_from_paths(
            train_paths,
            labels=train_labels,
            batch_size=batch_size,
            shuffle=True,
            seed=SEED,
        ).repeat()
        steps_per_epoch = 200

    model.fit(
        ds_train,
        epochs=1,
        steps_per_epoch=steps_per_epoch,
        verbose=1,
    )

    weight_paths = [None]

id_to_pos = {img_id: i for i, img_id in enumerate(sample_submission["image_id"].values)}

for wpath in weight_paths:
    if wpath is not None:
        model.load_weights(wpath)

    if use_tfrecord_test:
        ds_test_imgs = ds_test.map(
            lambda img, name: img,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds_test_names = ds_test.map(
            lambda img, name: name,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )

        preds = model.predict(ds_test_imgs, verbose=0)
        names = np.concatenate([n.numpy() for n in ds_test_names], axis=0).astype("U")

        order = np.fromiter(
            (id_to_pos[n] for n in names), dtype=np.int64, count=names.shape[0]
        )
        preds = preds[np.argsort(order)]
    else:
        preds = model.predict(ds_test, verbose=1)

    preds = preds[: sample_submission.shape[0]]
    all_pred.append(preds)

if len(all_pred) == 0:
    raise RuntimeError(
        "No predictions were generated (all_pred is empty); cannot build submission."
    )



## === cell 7
sum_pred = np.mean(np.stack(all_pred, axis=0), axis=0)

diagnos = np.argmax(sum_pred, axis=1).astype(int).tolist()

sample_submission["label"] = diagnos
print(sample_submission.head())
print("Label value counts:\n", sample_submission["label"].value_counts())



## === cell 8
sample_submission[["image_id", "label"]].to_csv("submission.csv", index=False)
print(
    "Wrote submission.csv with shape:", sample_submission[["image_id", "label"]].shape
)
print("submission.csv preview:")
print(pd.read_csv("submission.csv").head())
