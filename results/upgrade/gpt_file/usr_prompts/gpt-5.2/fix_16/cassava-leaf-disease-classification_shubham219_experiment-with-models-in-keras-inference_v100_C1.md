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

3.11

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

0.6925052886068298

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, glob, random

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

TRAIN_TFREC_DIR = os.path.join(DATA_ROOT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_ROOT, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

print("TF:", tf.__version__)
print("Data root:", DATA_ROOT)

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model

import math



## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(np.int32)

rng = np.random.RandomState(SEED)
val_frac = 0.1
val_indices = []
for lbl, g in train_df.groupby("label", sort=False):
    idx = g.index.to_numpy()
    rng.shuffle(idx)
    n_val = int(round(len(idx) * val_frac))
    val_indices.append(idx[:n_val])
val_indices = (
    np.concatenate(val_indices) if len(val_indices) else np.array([], dtype=int)
)
val_mask = train_df.index.isin(val_indices)
val_df = train_df.loc[val_mask].reset_index(drop=True)
train_df = train_df.loc[~val_mask].reset_index(drop=True)

IMG_SIZE = (256, 256)
BATCH_SIZE = 16
N_CLASSES = 5

preprocess = tf.keras.applications.resnet50.preprocess_input
AUTOTUNE = tf.data.AUTOTUNE

FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


rot_layer = tf.keras.layers.RandomRotation(
    factor=10.0 / 180.0, fill_mode="reflect", seed=SEED
)
trans_layer = tf.keras.layers.RandomTranslation(
    height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED + 1
)
zoom_layer = tf.keras.layers.RandomZoom(
    height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=SEED + 2
)
flip_layer = tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED + 3)


@tf.function
def _augment(img):
    img = flip_layer(img, training=True)
    img = rot_layer(img, training=True)
    img = trans_layer(img, training=True)
    img = zoom_layer(img, training=True)
    return img


@tf.function
def _prep_train_from_bytes(img_bytes, y_int):
    img = _decode_resize_from_bytes(img_bytes)
    img = _augment(img)
    img = preprocess(img)
    y_oh = tf.one_hot(tf.cast(y_int, tf.int32), depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


@tf.function
def _prep_val_from_bytes(img_bytes, y_int):
    img = _decode_resize_from_bytes(img_bytes)
    img = preprocess(img)
    y_oh = tf.one_hot(tf.cast(y_int, tf.int32), depth=N_CLASSES, dtype=tf.float32)
    return img, y_oh


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES)
    return ex["image"], ex["label"]


@tf.function
def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES)
    return ex["image_name"], ex["image"]


ds_opts = tf.data.Options()
ds_opts.experimental_deterministic = True
try:
    ds_opts.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
for _name in [
    "map_parallelization",
    "map_and_batch_fusion",
    "parallel_batch",
    "filter_fusion",
    "map_fusion",
    "map_vectorization",
]:
    try:
        setattr(ds_opts.experimental_optimization, _name, True)
    except Exception:
        pass
try:
    ds_opts.experimental_slack = True
except Exception:
    pass
try:
    ds_opts.threading.private_threadpool_size = max(4, (os.cpu_count() or 8) // 2)
except Exception:
    pass

train_tfrec_files = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrec_files) > 0, "No train TFRecords found"
assert len(test_tfrec_files) > 0, "No test TFRecords found"

train_image_ids = train_df["image_id"].to_numpy(dtype=object)
val_image_ids = val_df["image_id"].to_numpy(dtype=object)

train_id_keys = tf.constant(train_image_ids)
train_id_vals = tf.ones([len(train_image_ids)], dtype=tf.int32)
val_id_keys = tf.constant(val_image_ids)
val_id_vals = tf.ones([len(val_image_ids)], dtype=tf.int32)

train_tbl = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(train_id_keys, train_id_vals),
    default_value=0,
)
val_tbl = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_id_keys, val_id_vals),
    default_value=0,
)

raw_train = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTOTUNE
).with_options(ds_opts)


@tf.function
def _keep_train(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES)
    return tf.equal(train_tbl.lookup(ex["image_name"]), 1)


@tf.function
def _keep_val(serialized):
    ex = tf.io.parse_single_example(serialized, FEATURES)
    return tf.equal(val_tbl.lookup(ex["image_name"]), 1)


shuffle_buf = int(min(len(train_df), 4096))

train_ds = raw_train.filter(_keep_train)
train_ds = train_ds.shuffle(
    buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.map(_prep_train_from_bytes, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.prefetch(AUTOTUNE)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

val_ds = raw_train.filter(_keep_val)
val_ds = val_ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(_prep_val_from_bytes, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache()
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False)
val_ds = val_ds.prefetch(AUTOTUNE)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
)
base.trainable = False  # keep minimal + stable

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = tfl.GlobalAveragePooling2D()(x)
x = tfl.Dropout(0.2)(x)
outputs = tfl.Dense(N_CLASSES, activation="softmax")(x)
my_model = tf.keras.Model(inputs, outputs)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 3

steps_per_epoch = int(math.ceil(len(train_df) / BATCH_SIZE))
val_steps = int(math.ceil(len(val_df) / BATCH_SIZE))

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 3
@tf.function
def _prep_test_from_bytes(img_bytes):
    img = _decode_resize_from_bytes(img_bytes)
    img = preprocess(img)
    return img


def make_test_ds(batch_size=16):
    ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(ds_opts)
    ds = ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE
    )  # (image_name, image_bytes)
    ds = ds.map(
        lambda name, img_b: (name, _prep_test_from_bytes(img_b)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds




## === cell 4
test_ds = make_test_ds(batch_size=16)

test_names = []
test_imgs_ds = []
for batch_names, batch_imgs in test_ds:
    test_names.append(batch_names.numpy())
    test_imgs_ds.append(batch_imgs)
test_names = np.concatenate(test_names).astype("U")
test_imgs_ds = (
    tf.data.Dataset.from_tensor_slices(tf.concat(test_imgs_ds, axis=0))
    .batch(16)
    .prefetch(AUTOTUNE)
)

pred_test = my_model.predict(test_imgs_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

sample_sub = pd.read_csv(SAMPLE_SUB)
pred_df = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})
sample_sub = sample_sub.drop(columns=["label"]).merge(
    pred_df, on="image_id", how="left"
)
sample_sub["label"] = sample_sub["label"].astype(int)

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
sample_sub.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3505848231.py in <cell line: 0>()
      8     test_names.append(batch_names.numpy())
      9     test_imgs_ds.append(batch_imgs)
---> 10 test_names = np.concatenate(test_names).astype("U")
     11 test_imgs_ds = (
     12     tf.data.Dataset.from_tensor_slices(tf.concat(test_imgs_ds, axis=0))

ValueError: need at least one array to concatenate
