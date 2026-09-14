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

0.7892112420670897

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'The timeout is most likely caused by building two separate TFRecord parsing pipelines over the same `raw_all` dataset (doubling decode/resize cost) and then filtering twice with costly hashing per element, plus an extra full pass over `test_ds` just to collect IDs. I keep the exact model and training loop, but refactor the tf.data input pipeline to parse/decode each training example only once, compute the split boolean once, and then route into train/val datasets with single-pass caching where safe. I also extract test `image_id`s in the same pass used for prediction (no second iteration), and add tf.data performance knobs (options/threading) that are semantically equivalent. These changes reduce redundant CPU work and dataset passes while preserving the exact data content, augmentations, loss, and evaluation behavior.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate runtime crash by avoiding the protobuf implementation toggle that breaks TensorFlow in this environment, while keeping the rest of your setup (seeds, determinism attempts, XLA flags) intact. Then I correct the train/val split logic mismatch: you currently compute `trn_df/val_df` by shuffled indices but build `train_ds/val_ds` via hashed `image_id`, so `steps_per_epoch`/`val_steps` don’t correspond to the actual dataset sizes and training effectively under/over-trains, hurting accuracy. I minimally align the tf.data filtering with your already-created `trn_df/val_df` by filtering TFRecords using a lookup table of the selected IDs, preserving your model/augmentations/training loop. Finally, I make test ID extraction come from the same batched dataset pass used for prediction to avoid any ordering/id mismatch while keeping semantics the same.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate TensorFlow import crash caused by an incompatible protobuf runtime by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow`. Then, to move the accuracy score toward your target (your current score suggests predictions are badly misaligned/mostly wrong), I minimally correct the TFRecord `image_id` type mismatch: the lookup tables are built from Python strings but TFRecords provide `tf.string` bytes, so your train/val filters currently drop almost everything and the model effectively trains on near-empty data. I keep your model, augmentations, optimizer, and training loop unchanged, only ensuring that IDs are compared in the same bytes dtype for correct splitting and stable dataset sizes. Finally, I keep the submission-writing logic but ensure `pred_test_labels` length matches `sample_submission` length via the existing ID fallback path.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import glob
import random
import numpy as np
import pandas as pd

import tensorflow as tf

from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = os.path.join(BASE_INPUT, "train.csv")
sample_sub_path = os.path.join(BASE_INPUT, "sample_submission.csv")

train_img_dir = os.path.join(BASE_INPUT, "train_images")
test_img_dir = os.path.join(BASE_INPUT, "test_images")

train_tfrec_dir = os.path.join(BASE_INPUT, "train_tfrecords")
test_tfrec_dir = os.path.join(BASE_INPUT, "test_tfrecords")

assert os.path.exists(train_csv_path), f"Missing: {train_csv_path}"
assert os.path.exists(sample_sub_path), f"Missing: {sample_sub_path}"
assert os.path.isdir(train_img_dir), f"Missing dir: {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing dir: {test_img_dir}"
assert os.path.isdir(train_tfrec_dir), f"Missing dir: {train_tfrec_dir}"
assert os.path.isdir(test_tfrec_dir), f"Missing dir: {test_tfrec_dir}"

train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

assert {"image_id", "label"}.issubset(train_df.columns)
assert {"image_id", "label"}.issubset(sample_sub.columns)

train_df["path"] = train_img_dir + "/" + train_df["image_id"].astype(str)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

if DEBUG:
    train_df = train_df.sample(1024, random_state=SEED).reset_index(drop=True)

train_df.head()



## === cell 2
idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.15
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_df = train_df.iloc[val_idx].reset_index(drop=True)

trn_df.shape, val_df.shape



## === cell 3
IMG_SIZE = (300, 300)
BATCH_SIZE = 32 if not DEBUG else 16
AUTO = tf.data.AUTOTUNE

train_tfrec_files = sorted(glob.glob(os.path.join(train_tfrec_dir, "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(test_tfrec_dir, "*.tfrec")))
assert len(train_tfrec_files) > 0, f"No TFRecords found in {train_tfrec_dir}"
assert len(test_tfrec_files) > 0, f"No TFRecords found in {test_tfrec_dir}"

data_augmentation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(
            factor=15.0 / 180.0, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="nearest", seed=SEED
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="nearest",
            seed=SEED,
        ),
        tf.keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="data_augmentation",
)

_FEATURES_TRAIN_SAFE = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}

_FEATURES_TEST_WITH_ID = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _decode_image_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _has_valid_label(img, label, image_id):
    return tf.greater_equal(label, 0)


def _parse_train_example_once(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TRAIN_SAFE)
    img = _decode_image_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    image_id = ex["image_id"]  # tf.string (bytes)
    return img, label, image_id


def _aug_and_drop_id(img, label, image_id):
    img = data_augmentation(img, training=True)
    return img, label


def _noaug_and_drop_id(img, label, image_id):
    return img, label


trn_ids = tf.constant(trn_df["image_id"].astype(str).values, dtype=tf.string)
val_ids = tf.constant(val_df["image_id"].astype(str).values, dtype=tf.string)

trn_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(trn_ids, tf.ones_like(trn_ids, dtype=tf.int32)),
    default_value=0,
)
val_lookup = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids, tf.ones_like(val_ids, dtype=tf.int32)),
    default_value=0,
)


def _is_in_train(img, label, image_id):
    return tf.equal(trn_lookup.lookup(image_id), 1)


def _is_in_val(img, label, image_id):
    return tf.equal(val_lookup.lookup(image_id), 1)


options = tf.data.Options()
options.experimental_deterministic = False
options.experimental_optimization.apply_default_optimizations = True

raw_all = tf.data.TFRecordDataset(
    train_tfrec_files, num_parallel_reads=AUTO
).with_options(options)

parsed_all = (
    raw_all.map(_parse_train_example_once, num_parallel_calls=AUTO)
    .filter(_has_valid_label)
    .cache()
)

train_part = parsed_all.filter(_is_in_train)
val_part = parsed_all.filter(_is_in_val)

train_ds = (
    train_part.map(_aug_and_drop_id, num_parallel_calls=AUTO)
    .shuffle(min(8192, len(trn_df)), seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

val_ds = (
    val_part.map(_noaug_and_drop_id, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

print("steps_per_epoch:", steps_per_epoch, "val_steps:", val_steps)




## === cell 4
def build_model():
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    )
    x = GlobalAveragePooling2D()(base.output)
    x = Dropout(0.3)(x)
    out = Dense(5, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=out)
    return model


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,  # keep stable; global XLA already enabled above
)

model.summary()



## === cell 5
ckpt_path = "/kaggle/working/best_model.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_accuracy",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=2,
        verbose=1,
        min_lr=1e-6,
    ),
]

EPOCHS = 6 if not DEBUG else 2

best_model = model

history = model.fit(
    train_ds,
    validation_data=val_ds,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
)

if os.path.exists(ckpt_path):
    best_model = load_model(ckpt_path)
else:
    best_model = model




## === cell 6
def _parse_test_example_safe(ex):
    ex = tf.io.parse_single_example(ex, _FEATURES_TEST_WITH_ID)
    img = _decode_image_bytes(ex["image"])
    image_id = ex["image_id"]
    return img, image_id


test_ds = (
    tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTO)
    .with_options(options)
    .map(_parse_test_example_safe, num_parallel_calls=AUTO)
    .batch(128, drop_remainder=False)
    .prefetch(AUTO)
)

test_ids = []
pred_labels = []

for batch_imgs, batch_ids in test_ds:
    probs = best_model(batch_imgs, training=False).numpy()
    pred_labels.append(np.argmax(probs, axis=-1).astype(int))
    test_ids.append(batch_ids.numpy())

test_ids_bytes = (
    np.concatenate(test_ids, axis=0) if len(test_ids) else np.array([], dtype="S")
)
pred_test_labels = (
    np.concatenate(pred_labels, axis=0) if len(pred_labels) else np.array([], dtype=int)
)

test_ids = np.asarray(test_ids_bytes, dtype="S").astype(str)

if (len(test_ids) != len(sample_sub)) or np.all(test_ids == ""):
    test_ids = sample_sub["image_id"].astype(str).values

print(
    "n_test:",
    len(test_ids),
    "preds:",
    len(pred_test_labels),
    "IDs example:",
    test_ids[:3],
)



## === cell 7
pred_df = pd.DataFrame({"image_id": test_ids, "label": pred_test_labels})

final_csv = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if final_csv["label"].isna().any():
    fallback = int(train_df["label"].mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fallback).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
final_csv.head()



## === cell 8
assert os.path.exists("submission.csv")
sub = pd.read_csv("submission.csv")
assert list(sub.columns) == ["image_id", "label"]
assert len(sub) == len(sample_sub)
assert sub["label"].between(0, 4).all()
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
