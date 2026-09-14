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
import os, math, re, glob, random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from PIL import Image

print("Tensorflow version " + tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

tf.keras.backend.clear_session()



## === cell 1
IMAGE_SIZE = 512
BATCH_SIZE = 64

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_DIR, "train_images")
TEST_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing dir: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing dir: {TEST_TFREC_DIR}"



## === cell 2
test_dir = TEST_DIR




## === cell 3
def _tfrecord_files(tfrecord_dir):
    files = sorted(glob.glob(os.path.join(tfrecord_dir, "*.tfrec")))
    if not files:
        raise FileNotFoundError(f"No .tfrec files found in {tfrecord_dir}")
    return files


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_resize_normalize(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def build_tfrecord_ds(tfrecord_files, training, labeled=True, return_names=False):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.TFRecordDataset(
        tfrecord_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)

    def _parse(ex):
        ex = tf.io.parse_single_example(ex, _TFREC_FEATURES)
        img = _decode_resize_normalize(ex["image"])
        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
        if return_names and labeled:
            return (img, ex["target"]), ex["image_name"]
        if return_names and not labeled:
            return img, ex["image_name"]
        if labeled:
            return img, ex["target"]
        return img

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def build_test_ds(img_paths):
    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, (IMAGE_SIZE, IMAGE_SIZE), method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.Dataset.from_tensor_slices(img_paths).with_options(options)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def get_preds_model_list(image_dir, model_obj_list):
    test_tfrec_files = _tfrecord_files(TEST_TFREC_DIR)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)

    def _parse(ex):
        ex = tf.io.parse_single_example(ex, _TFREC_FEATURES)
        img = _decode_resize_normalize(ex["image"])
        return img, ex["image_name"]

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE).cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    try:
        import tensorflow_datasets as tfds  # usually present on Kaggle TF images

        names_np = np.concatenate([b for _, b in tfds.as_numpy(ds)], axis=0)
    except Exception:
        names_batches = []
        for _, names in ds:
            names_batches.append(names.numpy())
        names_np = np.concatenate(names_batches, axis=0)

    names_out = names_np.astype("U")  # bytes->str if needed
    if names_out.dtype.kind == "S":
        names_out = np.char.decode(names_out, "utf-8")

    ds_imgs = ds.map(lambda imgs, names: imgs, num_parallel_calls=AUTOTUNE)

    first = model_obj_list[0]
    all_same_obj = all(mod is first for mod in model_obj_list)

    if all_same_obj:
        avg_pred = first.predict(ds_imgs, verbose=0)
    else:
        pred_sum = None
        for mod in model_obj_list:
            p_full = mod.predict(ds_imgs, verbose=0)
            pred_sum = p_full if pred_sum is None else (pred_sum + p_full)
        avg_pred = pred_sum / float(len(model_obj_list))

    labels = np.argmax(avg_pred, axis=-1).astype(int)
    img_ids = [os.path.basename(n) for n in names_out.tolist()]
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 4
def get_preds(image_dir, model_obj):
    img_paths = sorted(glob.glob(os.path.join(image_dir, "*.jpg")))
    img_ids = [os.path.basename(p) for p in img_paths]

    ds = build_test_ds(img_paths)
    probs = model_obj.predict(ds, verbose=0)
    labels = np.argmax(probs, axis=-1).astype(int)
    return pd.DataFrame({"image_id": img_ids, "label": labels})




## === cell 5
train_df = pd.read_csv(TRAIN_CSV)
num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"


def make_model():
    inputs = keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    outputs = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    return model


def make_dataset_sharded(
    tfrecord_dir,
    training=True,
    shard_index=0,
    num_shards=1,
    cache_when_not_training=True,
):
    tfrec_files = _tfrecord_files(tfrecord_dir)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True

    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds = ds.shard(num_shards=num_shards, index=shard_index)

    def _parse(ex):
        ex = tf.io.parse_single_example(ex, _TFREC_FEATURES)
        img = _decode_resize_normalize(ex["image"])
        if training:
            img = tf.image.random_flip_left_right(img, seed=SEED)
        return img, ex["target"]

    ds = ds.map(_parse, num_parallel_calls=AUTOTUNE)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    if (not training) and cache_when_not_training:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


from sklearn.model_selection import train_test_split

tr_df, va_df = train_test_split(
    train_df, test_size=0.1, random_state=SEED, stratify=train_df["label"]
)

NUM_SHARDS = 10
train_ds = make_dataset_sharded(
    TRAIN_TFREC_DIR, training=True, shard_index=0, num_shards=NUM_SHARDS
)
val_ds = make_dataset_sharded(
    TRAIN_TFREC_DIR, training=False, shard_index=1, num_shards=NUM_SHARDS
)

EPOCHS = 2
m = make_model()
m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)

mod_lst = [m, m, m, m, m]

raw_pred_df = get_preds_model_list(test_dir, mod_lst)
sample_df = pd.read_csv(SAMPLE_SUB)

pred_map = dict(zip(raw_pred_df["image_id"].values, raw_pred_df["label"].values))
sample_df["label"] = sample_df["image_id"].map(pred_map).fillna(0).astype(int)

sample_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_df.shape)
print(sample_df.head())



## === cell 6
print(sample_df.head(10))
print("submission.csv exists:", os.path.exists("submission.csv"))
print("Unique predicted labels:", sorted(sample_df["label"].unique().tolist()))
