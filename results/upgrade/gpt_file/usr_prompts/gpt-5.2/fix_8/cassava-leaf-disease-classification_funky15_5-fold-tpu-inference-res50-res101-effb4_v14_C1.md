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

if "PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION" not in os.environ:
    pass

import glob, math, re
import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)
print("Eager:", tf.executing_eagerly())




## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64
NUM_CLASSES = 5

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass




## === cell 2
def _find_existing_model_paths():
    candidates = [
        "../input/tpus-resnet101-with-5-fold/resnet101_0.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_1.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_2.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_3.h5",
        "../input/tpus-resnet101-with-5-fold/resnet101_4.h5",
        "../input/resnet50-5fold-0/resnet50_0.h5",
        "../input/resnet50-5fold-0/resnet50_1.h5",
        "../input/resnet50-5fold-0/resnet50_2.h5",
        "../input/resnet50-5fold-0/resnet50_3.h5",
        "../input/resnet50-5fold-0/resnet50_4.h5",
    ]

    seen = set()
    existing = []
    for p in candidates:
        if p and p not in seen and os.path.isfile(p):
            existing.append(p)
            seen.add(p)
    return existing


def _build_fallback_model(image_size=512, num_classes=5):
    inp = keras.Input(shape=(image_size, image_size, 3), name="image")
    x = keras.applications.resnet.preprocess_input(inp)
    base = keras.applications.ResNet50(
        include_top=False, weights="imagenet", input_tensor=x, pooling="avg"
    )
    base.trainable = False
    out = keras.layers.Dense(num_classes, activation="softmax", name="probs")(
        base.output
    )
    model = keras.Model(inp, out)
    return model


existing_model_paths = _find_existing_model_paths()
loaded_models = []

if len(existing_model_paths) > 0:
    for p in existing_model_paths:
        try:
            m = keras.models.load_model(p, compile=False)
            loaded_models.append(m)
        except Exception as e:
            print(f"Warning: failed to load model {p}: {repr(e)}")

if len(loaded_models) == 0:
    print("No external .h5 models found/loaded; using fallback ResNet50 model.")
    loaded_models = [_build_fallback_model(IMAGE_SIZE, NUM_CLASSES)]

mod_lst = loaded_models
print(f"Number of models in ensemble: {len(mod_lst)}")




## === cell 3
test_dir = "/kaggle/data/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_dir):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]:
        if os.path.isdir(alt):
            test_dir = alt
            break

sample_sub_path = (
    "/kaggle/data/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.isfile(sample_sub_path):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]:
        if os.path.isfile(alt):
            sample_sub_path = alt
            break

train_csv_path = "/kaggle/data/input/cassava-leaf-disease-classification/train.csv"
if not os.path.isfile(train_csv_path):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/cassava-leaf-disease-classification/train.csv",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train.csv",
    ]:
        if os.path.isfile(alt):
            train_csv_path = alt
            break

train_dir = "/kaggle/data/input/cassava-leaf-disease-classification/train_images"
if not os.path.isdir(train_dir):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/cassava-leaf-disease-classification/train_images",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_images",
    ]:
        if os.path.isdir(alt):
            train_dir = alt
            break

train_tfrecords_dir = (
    "/kaggle/data/input/cassava-leaf-disease-classification/train_tfrecords"
)
if not os.path.isdir(train_tfrecords_dir):
    for alt in [
        "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords",
        "/kaggle/data/cassava-leaf-disease-classification/train_tfrecords",
        "/kaggle/data/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/train_tfrecords",
    ]:
        if os.path.isdir(alt):
            train_tfrecords_dir = alt
            break

print("train_csv_path:", train_csv_path)
print("train_dir:", train_dir)
print("test_dir:", test_dir)
print("sample_sub_path:", sample_sub_path)
print("train_tfrecords_dir:", train_tfrecords_dir)

assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"
assert os.path.isfile(
    sample_sub_path
), f"sample submission not found: {sample_sub_path}"
assert os.path.isfile(train_csv_path), f"train.csv not found: {train_csv_path}"
assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"




## === cell 4
@tf.function
def _decode_and_resize_from_path(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _decode_and_resize_with_label(path, label):
    return _decode_and_resize_from_path(path), label


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32)
    label = tf.cast(ex["target"], tf.int32)
    return img, label


def _make_dataset_from_df(df, image_root, batch_size, training):
    image_ids = df["image_id"].to_numpy(dtype=str, copy=False)
    paths = np.char.add(image_root.rstrip("/") + "/", image_ids)
    labels = df["label"].to_numpy(dtype=np.int32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_and_resize_with_label, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def _make_tfrecord_dataset(tfrec_files, batch_size, training):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def _train_fallback_if_needed(models, train_csv, train_images_dir, train_tfrecords_dir):
    if len(existing_model_paths) > 0:
        return models  # keep original intent if external models exist

    model = models[0]
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.SparseCategoricalCrossentropy(),
        metrics=["sparse_categorical_accuracy"],
    )

    df = pd.read_csv(train_csv)
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)
    n_val = int(0.1 * len(df))
    val_df = df.iloc[:n_val].reset_index(drop=True)
    tr_df = df.iloc[n_val:].reset_index(drop=True)

    tr_ds = _make_dataset_from_df(
        tr_df, train_images_dir, BATCH_SIZE, training=True
    ).cache()
    val_ds = _make_dataset_from_df(
        val_df, train_images_dir, BATCH_SIZE, training=False
    ).cache()

    epochs = 3
    print(
        f"Training fallback model for {epochs} epochs on {len(tr_df)} images, validating on {len(val_df)} images..."
    )
    model.fit(tr_ds, validation_data=val_ds, epochs=epochs, verbose=2)
    return [model]


mod_lst = _train_fallback_if_needed(
    mod_lst, train_csv_path, train_dir, train_tfrecords_dir
)




## === cell 5
def _make_test_dataset(image_ids, image_dir, batch_size):
    image_ids = np.asarray(image_ids, dtype=str)
    paths = np.char.add(image_dir.rstrip("/") + "/", image_ids)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_and_resize_from_path, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


def get_preds_model_list_from_ids(image_dir, model_obj_list, image_ids):
    test_ds = _make_test_dataset(image_ids, image_dir, BATCH_SIZE)

    probs_sum = None
    for mod in model_obj_list:
        p = mod.predict(test_ds, verbose=0)
        probs_sum = p if probs_sum is None else (probs_sum + p)

    avg_pred = probs_sum / float(len(model_obj_list))
    preds = np.argmax(avg_pred, axis=1).astype(int)
    return pd.DataFrame({"image_id": list(image_ids), "label": preds})


sample_df = pd.read_csv(sample_sub_path)
image_ids = sample_df["image_id"].tolist()

predict_df = get_preds_model_list_from_ids(test_dir, mod_lst, image_ids)
predict_df["label"] = predict_df["label"].astype(int)
predict_df = predict_df[["image_id", "label"]]

predict_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", predict_df.shape)
print(predict_df.head())




## === cell 6
print(predict_df.tail())
print("Unique predicted labels:", sorted(predict_df["label"].unique().tolist()))
print("submission.csv exists:", os.path.isfile("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.isfile("submission.csv") else None,
)
