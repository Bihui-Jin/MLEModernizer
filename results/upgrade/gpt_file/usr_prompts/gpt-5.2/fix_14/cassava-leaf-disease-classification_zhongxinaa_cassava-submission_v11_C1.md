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

3.12

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

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import efficientnet_v2

print("TF version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
WORK_DIR = "../input/cassava-leaf-disease-classification/"
TRAIN_CSV = os.path.join(WORK_DIR, "train.csv")
SAMPLE_SUB = os.path.join(WORK_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(WORK_DIR, "train_images")
TEST_IMG_DIR = os.path.join(WORK_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(WORK_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(WORK_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = 224
BATCH_SIZE = 64
NUM_CLASSES = 5

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

INTERLEAVE_CYCLE = min(16, max(1, (os.cpu_count() or 8) // 2))




## === cell 2
class SigmoidFocalCrossEntropy(tf.keras.losses.Loss):
    def __init__(self, alpha=0.25, gamma=2.0, from_logits=False, **kwargs):
        super().__init__(**kwargs)
        self.alpha = alpha
        self.gamma = gamma
        self.from_logits = from_logits

    def call(self, y_true, y_pred):
        if self.from_logits:
            y_pred = tf.sigmoid(y_pred)
        y_pred = tf.clip_by_value(
            y_pred, tf.keras.backend.epsilon(), 1 - tf.keras.backend.epsilon()
        )
        cross_entropy = -y_true * tf.math.log(y_pred) - (1 - y_true) * tf.math.log(
            1 - y_pred
        )
        weight = self.alpha * y_true + (1 - self.alpha) * (1 - y_true)
        focal_loss = weight * ((1 - y_pred) ** self.gamma) * cross_entropy
        return tf.reduce_sum(focal_loss, axis=-1)


custom_objects = {"SigmoidFocalCrossEntropy": SigmoidFocalCrossEntropy}



## === cell 3
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = (
    TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
)

assert len(train_df) > 0, "No training rows found."

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.15,
    random_state=SEED,
    stratify=train_df["label"].values,
)
tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print(
    "Label distribution (train):\n",
    tr_df["label"].value_counts(normalize=True).sort_index(),
)
print(
    "Label distribution (val):\n",
    va_df["label"].value_counts(normalize=True).sort_index(),
)



## === cell 4
aug = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ]
)


@tf.function
def _decode_resize_preprocess_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = efficientnet_v2.preprocess_input(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function
def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURES)
    x = _decode_resize_preprocess_from_bytes(ex["image"])
    y = tf.one_hot(tf.cast(ex["target"], tf.int32), NUM_CLASSES)
    return x, y


@tf.function
def _apply_aug(x, y):
    x = aug(x, training=True)
    return x, y


def _list_tfrec_files(tfrecord_dir, prefix):
    files = tf.io.gfile.glob(os.path.join(tfrecord_dir, f"{prefix}*.tfrec"))
    files = sorted(files)
    return files


def make_tfrec_ds(tfrecord_files, training, cache=None):
    ds = tf.data.Dataset.from_tensor_slices(tfrecord_files)
    if training:
        ds = ds.shuffle(len(tfrecord_files), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=INTERLEAVE_CYCLE,
        num_parallel_calls=AUTOTUNE,
        deterministic=(not training),
        block_length=1,
    )
    ds = ds.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=(not training)
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    if training:
        ds = ds.map(_apply_aug, num_parallel_calls=AUTOTUNE, deterministic=False)

    if cache is True:
        ds = ds.cache()
    elif isinstance(cache, str):
        ds = ds.cache(cache)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = not training
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    ds = ds.with_options(options)
    return ds


def make_img_ds(df, training, cache=None):
    paths = df["filepath"].astype(str).values
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(len(paths), seed=SEED, reshuffle_each_iteration=True)

    @tf.function
    def _load_and_prepare(path, label):
        x = _decode_resize_preprocess_from_path(path)
        y = tf.one_hot(label, NUM_CLASSES)
        if training:
            x = aug(x, training=True)
        return x, y

    ds = ds.map(
        _load_and_prepare,
        num_parallel_calls=AUTOTUNE,
        deterministic=(not training),
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache is True:
        ds = ds.cache()
    elif isinstance(cache, str):
        ds = ds.cache(cache)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = not training
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    ds = ds.with_options(options)
    return ds


train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
use_tfrecords = len(train_tfrec_files) > 0
print(
    "Using TFRecords for training:", use_tfrecords, "num_files:", len(train_tfrec_files)
)

if use_tfrecords:
    train_ds = make_tfrec_ds(train_tfrec_files, training=True, cache=None)
    val_cache_path = "/kaggle/working/val_cache"
    val_ds = make_tfrec_ds(train_tfrec_files, training=False, cache=val_cache_path)
else:
    train_ds = make_img_ds(tr_df, training=True, cache=None)
    val_cache_path = "/kaggle/working/val_cache"
    val_ds = make_img_ds(va_df, training=False, cache=val_cache_path)

train_steps = int(np.ceil(len(tr_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(va_df) / BATCH_SIZE))
print("train_steps:", train_steps, "val_steps:", val_steps)



## === cell 5
base = efficientnet_v2.EfficientNetV2B0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
base.trainable = False  # start with frozen backbone for stability/speed

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inputs, outputs)

try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=SigmoidFocalCrossEntropy(from_logits=False),
        metrics=["accuracy"],
        jit_compile=True,
    )
except TypeError:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=SigmoidFocalCrossEntropy(from_logits=False),
        metrics=["accuracy"],
    )

model.summary()



## === cell 6
ckpt_path = "/kaggle/working/best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, save_weights_only=False
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=6,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

try:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=SigmoidFocalCrossEntropy(from_logits=False),
        metrics=["accuracy"],
        jit_compile=True,
    )
except TypeError:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=SigmoidFocalCrossEntropy(from_logits=False),
        metrics=["accuracy"],
    )

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=3,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    callbacks=callbacks,
    verbose=1,
)

if not tf.io.gfile.exists(ckpt_path):
    model.save(ckpt_path)

model = tf.keras.models.load_model(ckpt_path, custom_objects=custom_objects)



## === cell 7
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = TEST_IMG_DIR.rstrip("/") + "/" + sub["image_id"].astype(str)

test_tfrec_files = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")

if len(test_tfrec_files) > 0:
    print("Using TFRecords for test:", True, "num_files:", len(test_tfrec_files))

    @tf.function
    def _parse_test_example(serialized):
        ex = tf.io.parse_single_example(serialized, _TEST_FEATURES)
        x = _decode_resize_preprocess_from_bytes(ex["image"])
        return x, ex["image_name"]

    ds = tf.data.Dataset.from_tensor_slices(test_tfrec_files)
    ds = ds.interleave(
        lambda f: tf.data.TFRecordDataset(f, num_parallel_reads=AUTOTUNE),
        cycle_length=INTERLEAVE_CYCLE,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
        block_length=1,
    )
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    options = tf.data.Options()
    options.deterministic = True
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    ds = ds.with_options(options)

    names = []
    probs_list = []

    for batch_imgs, batch_names in ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(
        AUTOTUNE
    ):
        probs_batch = model(batch_imgs, training=False).numpy()
        probs_list.append(probs_batch)
        names.extend([n.decode("utf-8") for n in batch_names.numpy().tolist()])

    probs = np.concatenate(probs_list, axis=0)
    preds = np.argmax(probs, axis=1).astype(int)

    name_to_pred = dict(zip(names, preds.tolist()))
    ordered_preds = sub["image_id"].map(name_to_pred).astype(int).values
    sub_out = pd.DataFrame({"image_id": sub["image_id"].values, "label": ordered_preds})
else:
    print("Using TFRecords for test:", False, "(fallback to JPEG paths)")
    test_paths = sub["filepath"].astype(str).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.map(
        _decode_resize_preprocess_from_path,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = True
    options.autotune.enabled = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    test_ds = test_ds.with_options(options)

    probs = model.predict(test_ds, verbose=1)
    preds = np.argmax(probs, axis=1).astype(int)
    sub_out = pd.DataFrame({"image_id": sub["image_id"].values, "label": preds})

sub_out.to_csv("submission.csv", index=False)

print(sub_out.head())
print("Wrote submission.csv with shape:", sub_out.shape)
print("Unique labels:", np.unique(sub_out["label"].values))
