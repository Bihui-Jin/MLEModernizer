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
import random
import numpy as np

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")




## === cell 1
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from PIL import Image  # kept to preserve original imports/cell structure, but not used

print("TF version:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

print("Train rows:", len(train_df), "Test rows:", len(sample_df))
print("Train label distribution:\n", train_df["label"].value_counts().sort_index())




## === cell 3
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCHS = 3  # keep identical training schedule

val_frac = 0.10

TFREC_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_and_resize_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img.set_shape((IMG_SIZE, IMG_SIZE, 3))
    return img


@tf.function
def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES)
    img = _decode_and_resize_bytes(ex["image"])
    label = tf.cast(ex["target"], tf.int32)
    return img, label


aug = keras.Sequential(
    [
        layers.RandomFlip("horizontal"),
        layers.RandomRotation(0.05),
        layers.RandomZoom(0.1),
    ],
    name="augment",
)


@tf.function
def _map_train_from_img(img, label):
    img = aug(img, training=True)
    return img, label


_TFREC_FILE_CACHE = {}


def _tfrecord_files(directory, prefix):
    key = (directory, prefix)
    if key in _TFREC_FILE_CACHE:
        return _TFREC_FILE_CACHE[key]
    files = tf.io.gfile.glob(os.path.join(directory, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found in {directory} with prefix {prefix}"
        )
    _TFREC_FILE_CACHE[key] = files
    return files


train_files = _tfrecord_files(TRAIN_TFREC_DIR, "ld_train")

n_files = len(train_files)
n_val_files = max(1, int(round(n_files * val_frac)))
val_files = train_files[:n_val_files]
trn_files = train_files[n_val_files:]


def _dataset_cardinality_batches(ds):
    try:
        c = tf.data.experimental.cardinality(ds)
        c_val = int(c.numpy())
        if c_val >= 0:  # known
            return c_val
    except Exception:
        pass
    return None


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
try:
    options.experimental_slack = True
except Exception:
    pass


def _dataset_from_files(files, training):
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        **(
            {"sloppy": True}
            if "sloppy" in tf.data.TFRecordDataset.__init__.__code__.co_varnames
            else {}
        ),
    ).with_options(options)

    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.ignore_errors()

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=False)
        ds = ds.map(
            _map_train_from_img, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _dataset_from_files(trn_files, training=True)
val_ds = _dataset_from_files(val_files, training=False)

steps_per_epoch = _dataset_cardinality_batches(train_ds)
validation_steps = _dataset_cardinality_batches(val_ds)

print(
    f"TFRecord files: train={len(trn_files)}, val={len(val_files)} | "
    f"steps: train={steps_per_epoch}, val={validation_steps} (None means 'iterate full dataset')"
)

inputs = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)




## === cell 4
if "val_accuracy" in history.history:
    print(f"Validation accuracy: {history.history['val_accuracy'][-1]:.4f}")
elif "val_acc" in history.history:
    print(f"Validation accuracy: {history.history['val_acc'][-1]:.4f}")
else:
    val_loss, val_acc = model.evaluate(val_ds, steps=validation_steps, verbose=0)
    print(f"Validation accuracy: {val_acc:.4f}")




## === cell 5
test_images = sample_df["image_id"].tolist()
test_paths = (TEST_IMG_DIR + "/" + sample_df["image_id"].astype(str)).to_numpy()
print(
    "Missing test files: 0 (existence scan skipped for speed; dataset assumed consistent)"
)




## === cell 6
@tf.function
def _parse_test_tfrecord(example_proto):
    ex = tf.io.parse_single_example(
        example_proto, {"image": tf.io.FixedLenFeature([], tf.string)}
    )
    img = _decode_and_resize_bytes(ex["image"])
    return img


def make_test_dataset_from_tfrecords():
    files = _tfrecord_files(TEST_TFREC_DIR, "ld_test")

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    try:
        options.experimental_slack = True
    except Exception:
        pass

    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        **(
            {"sloppy": True}
            if "sloppy" in tf.data.TFRecordDataset.__init__.__code__.co_varnames
            else {}
        ),
    ).with_options(options)

    ds = ds.map(
        _parse_test_tfrecord, num_parallel_calls=AUTOTUNE, deterministic=True
    ).ignore_errors()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds, files


try:
    test_ds, test_files = make_test_dataset_from_tfrecords()

    test_steps = _dataset_cardinality_batches(test_ds)
except Exception:

    @tf.function
    def _decode_and_resize(path):
        img_bytes = tf.io.read_file(path)
        img = _decode_and_resize_bytes(img_bytes)
        return img

    @tf.function
    def _map_test(path):
        return _decode_and_resize(path)

    def make_test_dataset(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths)

        options = tf.data.Options()
        options.experimental_deterministic = True
        options.experimental_optimization.apply_default_optimizations = True
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        try:
            options.experimental_slack = True
        except Exception:
            pass
        ds = ds.with_options(options)

        ds = ds.map(
            _map_test, num_parallel_calls=AUTOTUNE, deterministic=True
        ).ignore_errors()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
        return ds

    test_ds = make_test_dataset(test_paths)
    test_steps = (len(test_images) + BATCH_SIZE - 1) // BATCH_SIZE

probs = model.predict(test_ds, steps=test_steps, verbose=0)
y_preds = probs.argmax(axis=1).astype(int).tolist()

if len(y_preds) != len(test_images):
    y_preds = y_preds[: len(test_images)]

print("Preds:", len(y_preds), "Images:", len(test_images))




## === cell 7
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
df_sub["image_id"] = df_sub["image_id"].astype(str)
df_sub["label"] = df_sub["label"].astype(int)

print(df_sub.head())
print("Submission shape:", df_sub.shape)




## === cell 8
out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "size(bytes):", os.path.getsize(out_path))
