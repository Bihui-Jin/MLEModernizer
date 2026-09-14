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
import random
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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

CACHE_DIR = "/kaggle/working/tf_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _read_bytes(path, label):
    img_bytes = tf.io.read_file(path)
    label = tf.cast(label, tf.int32)
    return img_bytes, label


def _decode_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


def _decode_map(img_bytes, label):
    img = _decode_resize_from_bytes(img_bytes)
    label = tf.cast(label, tf.int32)
    return img, label


def _augment_map(img, label):
    img = augmenter(img, training=True)
    label = tf.cast(label, tf.int32)
    return img, label


def _valid_map(img, label):
    label = tf.cast(label, tf.int32)
    return img, label


def _with_ds_options(ds):
    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    return ds.with_options(opts)


def make_train_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label"].to_numpy(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _with_ds_options(ds)

    ds = ds.shuffle(
        buffer_size=min(len(df), SHUFFLE_BUFFER),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache(os.path.join(CACHE_DIR, "train_decoded.cache"))

    ds = ds.map(_augment_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths = df["path"].to_numpy()
    labels = df["label"].to_numpy(np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = _with_ds_options(ds)

    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.cache(os.path.join(CACHE_DIR, "valid_decoded.cache"))

    ds = ds.map(_valid_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_df)
valid_ds = make_valid_ds(valid_df)




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

test_paths = sample_sub["path"].to_numpy()


def _read_bytes_test(path):
    return tf.io.read_file(path)


def _test_decode_map(img_bytes):
    return _decode_resize_from_bytes(img_bytes)


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = _with_ds_options(test_ds)
test_ds = test_ds.map(_read_bytes_test, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.map(_test_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

test_ds = test_ds.cache(os.path.join(CACHE_DIR, "test_decoded.cache"))

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False)
test_ds = test_ds.prefetch(AUTOTUNE)




## === cell 7
assert "model" in globals() and model is not None

probs = model.predict(test_ds, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission_df = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": preds}
)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)

print("Submission file created:", out_path)
print(submission_df.head())
print("Rows:", len(submission_df), "Cols:", submission_df.columns.tolist())
assert out_path.endswith(".csv") and len(submission_df) == len(sample_sub)
assert submission_df.columns.tolist() == ["image_id", "label"]
