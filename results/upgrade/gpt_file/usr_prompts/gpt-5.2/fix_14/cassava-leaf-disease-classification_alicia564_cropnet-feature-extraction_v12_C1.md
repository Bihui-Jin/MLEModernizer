# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

# 5. Target score

0.8881837413115745

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.19955) has done: 'I remove `tensorflow_hub` usage (it’s triggering the protobuf `MessageFactory.GetPrototype` crash in this environment) and instead build the same kind of image classifier directly with `tf.keras` EfficientNetB0, which you already import. I also fix the broken SavedModel/TFSMLayer path logic by deleting that dependency and ensuring `model` is always defined via the trained Keras model. To keep the core approach intact (transfer learning on 224×224 images with augmentation and categorical loss), I train using your existing `ImageDataGenerator` generators and then generate predictions aligned to `sample_submission.csv` order. Finally, I guarantee a valid `/kaggle/working/submission.csv` with `image_id,label` columns is written.'
- What this solution (achieved 0.23057) has done: 'The timeout is dominated by expensive JPEG decode/resize every epoch plus heavy on-the-fly augmentation, and by an inefficient caching setup that forces large disk cache writes and still re-runs the augmentation pipeline each epoch. I keep the same model, losses, epochs, and augmentation, but make the input pipeline provably equivalent while reducing redundant work: (1) switch to TFRecords (already provided) to avoid filesystem JPEG overhead, (2) cache only the *decoded+resized* tensors in-memory (not to disk) to eliminate slow cache file I/O, and (3) add pipeline knobs that improve throughput without changing semantics (parallel reads, non-blocking prefetch, and deterministic options retained). Prediction also read TFRecord test shards instead of individual JPEG files, preserving the same preprocessing and output format.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow.keras import backend as K

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("enable_op_determinism() not available; continuing. Reason:", repr(e))

tf.config.run_functions_eagerly(False)

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = f"{DATA_DIR}/train.csv"
SAMPLE_SUB = f"{DATA_DIR}/sample_submission.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
MAP_JSON = f"{DATA_DIR}/label_num_to_disease_map.json"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

label_to_disease = pd.read_json(MAP_JSON, typ="series")
train_csv = pd.read_csv(TRAIN_CSV)

train_csv["path"] = TRAIN_IMG_DIR + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(int)

train_df, valid_df = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=SEED
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = train_csv["label"].nunique()

print("NUM_CLASSES:", NUM_CLASSES)



## === cell 2
base_model = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base_model.trainable = False  # feature extraction stage

x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.2)(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()



## === cell 3
from tensorflow.keras.callbacks import ReduceLROnPlateau, Callback
from tensorflow.keras import layers as L


class EpochEndPrinter(Callback):
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        msg = {
            k: float(v)
            for k, v in logs.items()
            if isinstance(v, (int, float, np.floating))
        }
        print(f"Epoch {epoch+1} end logs:", msg)


learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)

augmenter = tf.keras.Sequential(
    [
        L.RandomRotation(factor=0.125, fill_mode="nearest", seed=SEED),  # ~45 degrees
        L.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=SEED),
        L.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=SEED),
        L.RandomFlip("horizontal_and_vertical", seed=SEED),
    ],
    name="aug",
)

AUTOTUNE = tf.data.AUTOTUNE
SHUFFLE_BUFFER = 4096

TRAIN_TFRECS = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord")
    ]
)
TEST_TFRECS = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec") or f.endswith(".tfrecord")
    ]
)

print("Train TFRecords:", len(TRAIN_TFRECS), "Test TFRecords:", len(TEST_TFRECS))

train_ids = set(train_df["image_id"].astype(str).tolist())
valid_ids = set(valid_df["image_id"].astype(str).tolist())


def _with_ds_options(ds):
    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    return ds.with_options(opts)


def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


_TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_tfrecord(example):
    ex = tf.io.parse_single_example(example, _TFREC_FEATURES)
    img = _decode_resize_from_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    image_id = ex["image_name"]
    return img, label, image_id


def _augment_map(img, label):
    img = augmenter(img, training=True)
    label = tf.cast(label, tf.int32)
    return img, label


def _valid_map(img, label):
    label = tf.cast(label, tf.int32)
    return img, label


_MINI_FEATURES = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


def _parse_name_target(example):
    ex = tf.io.parse_single_example(example, _MINI_FEATURES)
    return ex["image_name"], tf.cast(ex["target"], tf.int32)


def _normalize_image_id_tf(x):
    x = tf.strings.strip(x)
    x = tf.strings.regex_replace(x, r"\x00", "")
    x = tf.strings.split(x, os.sep)[-1]
    return x


def _collect_split_indices():
    files = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_ds_options(ds)
    ds = ds.map(_parse_name_target, num_parallel_calls=AUTOTUNE, deterministic=True)

    train_idx = []
    valid_idx = []
    for i, (iid, _) in enumerate(ds.as_numpy_iterator()):
        if isinstance(iid, (bytes, bytearray, np.bytes_)):
            s = iid.decode("utf-8", errors="ignore")
        else:
            s = str(iid)
        s = s.strip().strip("\x00")
        s = os.path.basename(s)
        if s in train_ids:
            train_idx.append(i)
        elif s in valid_ids:
            valid_idx.append(i)
    return np.asarray(train_idx, dtype=np.int64), np.asarray(valid_idx, dtype=np.int64)


train_indices, valid_indices = _collect_split_indices()
print("Split indices:", "train", len(train_indices), "valid", len(valid_indices))

_train_idx_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(train_indices, dtype=tf.int64),
        values=tf.ones([len(train_indices)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)
_valid_idx_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(valid_indices, dtype=tf.int64),
        values=tf.ones([len(valid_indices)], dtype=tf.int32),
    ),
    default_value=tf.constant(0, dtype=tf.int32),
)


def _in_train_by_index(i):
    return tf.equal(_train_idx_table.lookup(i), 1)


def _in_valid_by_index(i):
    return tf.equal(_valid_idx_table.lookup(i), 1)


_CACHE_DIR = "/kaggle/working/tfdata_cache"
tf.io.gfile.makedirs(_CACHE_DIR)
TRAIN_CACHE_PATH = os.path.join(_CACHE_DIR, "train.cache")
VALID_CACHE_PATH = os.path.join(_CACHE_DIR, "valid.cache")


def make_train_ds_from_tfrecords():
    files = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
    files = files.shuffle(len(TRAIN_TFRECS), seed=SEED, reshuffle_each_iteration=True)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_ds_options(ds)

    ds = ds.enumerate()  # (index, serialized_example)
    ds = ds.filter(lambda i, _: _in_train_by_index(i))
    ds = ds.map(lambda _, ex: ex, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(
        lambda img, label, image_id: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.cache(TRAIN_CACHE_PATH)
    ds = ds.map(_augment_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds_from_tfrecords():
    files = tf.data.Dataset.from_tensor_slices(TRAIN_TFRECS)
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = _with_ds_options(ds)

    ds = ds.enumerate()
    ds = ds.filter(lambda i, _: _in_valid_by_index(i))
    ds = ds.map(lambda _, ex: ex, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(
        lambda img, label, image_id: (img, label),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache(VALID_CACHE_PATH)
    ds = ds.map(_valid_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds_from_tfrecords()
valid_ds = make_valid_ds_from_tfrecords()

_xb, _yb = next(iter(train_ds.take(1)))
print("Train batch:", _xb.shape, _yb.shape)



## === cell 4
EPOCHS = 10  # keep as originally set; no early stopping now

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=[learning_rate_reduction, EpochEndPrinter()],
    verbose=1,
)

val_metrics = model.evaluate(valid_ds, verbose=0)
print({"val_loss": float(val_metrics[0]), "val_accuracy": float(val_metrics[1])})



## === cell 5
FINE_TUNE = True
if FINE_TUNE:
    base_model.trainable = True
    for layer in base_model.layers[:-20]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=16,
    )

    history_ft = model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=3,
        callbacks=[learning_rate_reduction, EpochEndPrinter()],
        verbose=1,
    )

    val_metrics = model.evaluate(valid_ds, verbose=0)
    print(
        {
            "val_loss_after_ft": float(val_metrics[0]),
            "val_accuracy_after_ft": float(val_metrics[1]),
        }
    )



## === cell 6
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["path"] = TEST_IMG_DIR + "/" + sample_sub["image_id"]


def _parse_test_tfrecord(example):
    ex = tf.io.parse_single_example(
        example,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
        },
    )
    img = _decode_resize_from_bytes(ex["image"])
    image_id = ex["image_name"]
    return img, image_id


test_files = tf.data.Dataset.from_tensor_slices(TEST_TFRECS)
test_ds_kv = tf.data.TFRecordDataset(test_files, num_parallel_reads=AUTOTUNE)
test_ds_kv = _with_ds_options(test_ds_kv)
test_ds_kv = test_ds_kv.map(
    _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
)

TEST_CACHE_PATH = os.path.join(_CACHE_DIR, "test.cache")
test_ds_kv = test_ds_kv.cache(TEST_CACHE_PATH)
test_ds_kv = test_ds_kv.batch(BATCH_SIZE, drop_remainder=False)
test_ds_kv = test_ds_kv.prefetch(AUTOTUNE)

test_ds = test_ds_kv.map(
    lambda img, image_id: img, num_parallel_calls=AUTOTUNE, deterministic=True
)



## === cell 7
assert "model" in globals() and model is not None


def _normalize_image_id(x):
    if isinstance(x, (bytes, bytearray, np.bytes_)):
        s = x.decode("utf-8", errors="ignore")
    else:
        s = str(x)
    s = s.strip().strip("\x00")
    s = os.path.basename(s)
    return s


probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

image_ids = []
for _, batch_ids in test_ds_kv.as_numpy_iterator():
    image_ids.extend([_normalize_image_id(x) for x in batch_ids])

assert len(image_ids) == len(
    preds
), f"ids/preds length mismatch: {len(image_ids)} vs {len(preds)}"

pred_map = {}
for iid, p in zip(image_ids, preds):
    if iid not in pred_map:
        pred_map[iid] = int(p)

mode_label = (
    int(pd.Series(list(pred_map.values())).mode().iloc[0]) if len(pred_map) else 0
)
ordered_preds = [
    pred_map.get(iid, mode_label) for iid in sample_sub["image_id"].astype(str).tolist()
]

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].astype(str).tolist(), "label": ordered_preds}
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())

assert out_path.endswith(".csv")
assert len(submission_df) == len(sample_sub)
assert submission_df.columns.tolist() == ["image_id", "label"]
assert submission_df["label"].isna().sum() == 0
assert submission_df["image_id"].isna().sum() == 0
assert submission_df["image_id"].nunique() == len(
    submission_df
), "Duplicate image_id rows found in submission."
assert submission_df["label"].dtype != object, "Label column must be numeric."
