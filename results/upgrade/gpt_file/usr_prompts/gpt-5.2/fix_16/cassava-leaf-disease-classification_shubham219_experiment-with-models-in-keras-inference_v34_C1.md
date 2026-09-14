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

import random
import glob

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers

SEED = 42
DEBUG = False


def seed_everything(seed=SEED):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)


seed_everything(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass
tf.keras.utils.set_random_seed(SEED)

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(r):
        DATA_ROOT = r
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset folder. "
        f"Tried: {CANDIDATE_ROOTS}"
    )

TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

print("Using DATA_ROOT:", DATA_ROOT)
print("TensorFlow:", tf.__version__)



## === cell 1
IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 8
EPOCHS = 3 if not DEBUG else 1  # keep modest for runtime; no early stopping introduced
NUM_CLASSES = 5

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].copy()
trn_df = train_df.iloc[val_size:].copy()


def _cache_path(tag: str) -> str:
    return os.path.join(
        "/kaggle/working", f"tfdata_cache_{tag}_{IMG_SIZE[0]}x{IMG_SIZE[1]}.cache"
    )


def _with_ds_options(ds, training: bool):
    options = tf.data.Options()
    options.experimental_deterministic = not training
    try:
        options.experimental_optimization.map_fusion = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    try:
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return ds.with_options(options)


def _augment(img):
    h = tf.cast(tf.shape(img)[0], tf.float32)
    w = tf.cast(tf.shape(img)[1], tf.float32)

    dx = tf.random.uniform([], -0.05, 0.05, seed=SEED) * w
    dy = tf.random.uniform([], -0.05, 0.05, seed=SEED) * h

    scale = tf.random.uniform([], 0.9, 1.1, seed=SEED)

    ang = tf.random.uniform([], -10.0, 10.0, seed=SEED) * (np.pi / 180.0)
    cos_a = tf.cos(ang) / scale
    sin_a = tf.sin(ang) / scale

    cx = w / 2.0
    cy = h / 2.0

    a0 = cos_a
    a1 = -sin_a
    a2 = cx - a0 * cx - a1 * cy - dx
    b0 = sin_a
    b1 = cos_a
    b2 = cy - b0 * cx - b1 * cy - dy

    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        fill_mode="REFLECT",
        interpolation="BILINEAR",
        fill_value=0.0,
    )[0]

    img = tf.image.random_flip_left_right(img, seed=SEED)
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_from_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TFREC_FEATURES)
    img = tf.io.decode_jpeg(
        ex["image"], channels=3, dct_method="INTEGER_FAST", try_recover_truncated=True
    )
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    y = tf.cast(ex["target"], tf.int32)
    name = ex["image_name"]
    return img, y, name


def _map_onehot(img, y, name):
    return img, tf.one_hot(y, NUM_CLASSES), name


def _strip_name(img, y, name):
    return img, y


def _list_tfrecords(tfrecord_dir):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, "*.tfrec"))
    files = sorted(files)
    if len(files) == 0:
        raise FileNotFoundError(f"No TFRecord shards found in {tfrecord_dir}")
    return files


def make_dataset_from_tfrecords(tfrecord_files, training: bool, cache_tag: str):
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = _with_ds_options(ds, training=training)

    if training:
        ds = ds.shuffle(len(tfrecord_files), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )

    ds = ds.map(
        _decode_from_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=not training
    )

    if not training:
        ds = ds.cache(_cache_path(cache_tag))

    def _maybe_aug(img, y, name):
        if training:
            img = _augment(img)
        return img, y, name

    ds = ds.map(_maybe_aug, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.map(_map_onehot, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.map(_strip_name, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_tfrecord_files = _list_tfrecords(TRAIN_TFREC_DIR)

trn_ids = set(trn_df["image_id"].astype(str).tolist())
val_ids = set(val_df["image_id"].astype(str).tolist())


def make_filtered_split_ds(
    tfrecord_files, keep_ids_set, training: bool, cache_tag: str
):
    keep = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(
            keys=tf.constant(list(keep_ids_set), dtype=tf.string),
            values=tf.ones([len(keep_ids_set)], dtype=tf.int32),
        ),
        default_value=tf.constant(0, dtype=tf.int32),
    )

    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = _with_ds_options(ds, training=training)
    if training:
        ds = ds.shuffle(len(tfrecord_files), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=not training,
    )
    ds = ds.map(
        _decode_from_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=not training
    )

    ds = ds.filter(lambda img, y, name: tf.equal(keep.lookup(name), 1))

    if not training:
        ds = ds.cache(_cache_path(cache_tag))

    def _maybe_aug(img, y, name):
        if training:
            img = _augment(img)
        return img, y, name

    ds = ds.map(_maybe_aug, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.map(_map_onehot, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.map(_strip_name, num_parallel_calls=AUTOTUNE, deterministic=not training)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = make_filtered_split_ds(
    train_tfrecord_files, trn_ids, training=True, cache_tag="train_tfrec_decoded"
)
val_ds = make_filtered_split_ds(
    train_tfrecord_files, val_ids, training=False, cache_tag="val_tfrec_decoded"
)

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # stabilize + faster; preserves core approach

inputs = layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(5, activation="softmax")(x)
my_model = Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 2
if not os.path.exists(SAMPLE_SUB):
    test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
    if len(test_images) == 0:
        raise FileNotFoundError(f"No test images found in {TEST_IMG_DIR}")
    df_test = pd.DataFrame({"path": test_images})
else:
    sample = pd.read_csv(SAMPLE_SUB, usecols=["image_id"])
    df_test = sample.copy()
    df_test["path"] = (
        TEST_IMG_DIR.rstrip("/") + "/" + df_test["image_id"].astype(str)
    ).astype(str)


def make_test_ds_from_tfrecords(tfrecord_files, batch_size=128):
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    ds = _with_ds_options(ds, training=False)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=AUTOTUNE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(_decode_from_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(_cache_path("test_tfrec_decoded"))
    names_ds = ds.map(
        lambda img, y, name: name, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    img_ds = ds.map(
        lambda img, y, name: img, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    img_ds = img_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    names_ds = names_ds.batch(batch_size, drop_remainder=False)
    return img_ds, names_ds


test_tfrecord_files = _list_tfrecords(TEST_TFREC_DIR)
test_img_ds, test_names_ds = make_test_ds_from_tfrecords(
    test_tfrecord_files, batch_size=128
)

preds_list = []
names_list = []
for x_batch, n_batch in zip(test_img_ds, test_names_ds):
    preds_list.append(my_model.predict_on_batch(x_batch))
    names_list.append(n_batch.numpy())

pred_test = np.concatenate(preds_list, axis=0)
names = np.concatenate(names_list, axis=0).tolist()
image_ids = [nn.decode("utf-8") for nn in names]

pred_test_labels = np.argmax(pred_test, axis=-1)



## === cell 3
final_submission = pd.DataFrame(
    {
        "image_id": np.asarray(image_ids),
        "label": pred_test_labels.astype(int),
    }
)

final_csv = final_submission[["image_id", "label"]].copy()

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        mode_label = int(pd.Series(pred_test_labels).mode().iloc[0])
        final_csv["label"] = final_csv["label"].fillna(mode_label).astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())
