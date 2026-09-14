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
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)




## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import load_model

tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    _cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu_cnt)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## === cell 2
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

num_classes = int(train_df["label"].nunique())
print(
    "Train rows:",
    len(train_df),
    "Num classes:",
    num_classes,
    "Test rows:",
    len(sample_df),
)




## === cell 3
MODEL_PATH = "../input/keras-available-models-part-1-training/model.h5"

model = None
if os.path.exists(MODEL_PATH):
    model = load_model(MODEL_PATH, compile=False)
    print("Loaded external model:", MODEL_PATH)
else:
    print(
        "External model not found; will train a small fallback model on provided training images."
    )




## === cell 4
img_size = 224
batch_size = 32
epochs = 5


@tf.function
def _decode_resize_normalize(path, label=None):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, label


@tf.function
def _parse_tfrecord(example_proto, labeled=True, with_id=False):
    features = {"image": tf.io.FixedLenFeature([], tf.string)}
    if labeled:
        features["target"] = tf.io.FixedLenFeature([], tf.int64)
    if with_id:
        features["image_name"] = tf.io.FixedLenFeature([], tf.string)

    parsed = tf.io.parse_single_example(example_proto, features)

    img = tf.image.decode_jpeg(parsed["image"], channels=3)
    img = tf.image.resize(
        img, [img_size, img_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0

    if labeled:
        label = tf.cast(parsed["target"], tf.int32)
        if with_id:
            return img, label, parsed["image_name"]
        return img, label

    if with_id:
        return img, parsed["image_name"]
    return img


def _list_tfrec_files(tfrec_dir, prefix):
    if not os.path.isdir(tfrec_dir):
        return []
    files = tf.io.gfile.glob(os.path.join(tfrec_dir, f"{prefix}*.tfrec"))
    return sorted(files)


opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.autotune = True
except Exception:
    pass


if model is None:
    train_tfrec_files = _list_tfrec_files(TRAIN_TFREC_DIR, "ld_train")

    if train_tfrec_files:
        rng = np.random.RandomState(SEED)
        file_idx = np.arange(len(train_tfrec_files))
        rng.shuffle(file_idx)
        split_files = max(1, int(0.9 * len(file_idx)))
        tr_files = [train_tfrec_files[i] for i in file_idx[:split_files]]
        va_files = [train_tfrec_files[i] for i in file_idx[split_files:]] or [
            train_tfrec_files[file_idx[-1]]
        ]

        def tfrec_to_dataset(files, shuffle=False):
            ds = tf.data.TFRecordDataset(
                files, num_parallel_reads=tf.data.AUTOTUNE
            ).with_options(opts)
            ds = ds.map(
                lambda x: _parse_tfrecord(x, labeled=True, with_id=False),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.apply(tf.data.experimental.ignore_errors())

            ds = ds.cache()

            if shuffle:
                ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = tfrec_to_dataset(tr_files, shuffle=True)
        val_ds = tfrec_to_dataset(va_files, shuffle=False)

        steps_per_epoch = None
        validation_steps = None

    else:
        idx = np.arange(len(train_df))
        rng = np.random.RandomState(SEED)
        rng.shuffle(idx)
        split = int(0.9 * len(idx))
        tr_idx, va_idx = idx[:split], idx[split:]

        tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
        va_df = train_df.iloc[va_idx].reset_index(drop=True)

        def df_to_dataset(df, shuffle=False):
            paths = tf.constant(
                np.char.add(TRAIN_IMG_DIR + "/", df["image_id"].to_numpy(dtype=str))
            )
            labels = tf.constant(df["label"].to_numpy(dtype=np.int32))
            ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)
            if shuffle:
                ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
            ds = ds.map(
                _decode_resize_normalize,
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.apply(tf.data.experimental.ignore_errors())

            ds = ds.cache()

            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = df_to_dataset(tr_df, shuffle=True)
        val_ds = df_to_dataset(va_df, shuffle=False)

        steps_per_epoch = int(np.ceil(len(tr_df) / batch_size))
        validation_steps = int(np.ceil(len(va_df) / batch_size))

    model = keras.Sequential(
        [
            layers.Input(shape=(img_size, img_size, 3)),
            layers.Conv2D(32, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(64, 3, padding="same", activation="relu"),
            layers.MaxPooling2D(),
            layers.Conv2D(128, 3, padding="same", activation="relu"),
            layers.GlobalAveragePooling2D(),
            layers.Dropout(0.2),
            layers.Dense(num_classes, activation="softmax"),
        ]
    )

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
    model.summary()

    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        verbose=2,
    )




## === cell 5
test_images = sample_df["image_id"].tolist()
assert len(test_images) == len(sample_df), "Sample submission parsing issue"




## === cell 6
test_tfrec_files = _list_tfrec_files(TEST_TFREC_DIR, "ld_test")

if test_tfrec_files:
    test_ds = tf.data.TFRecordDataset(
        test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE
    ).with_options(opts)
    test_ds = test_ds.map(
        lambda x: _parse_tfrecord(x, labeled=False, with_id=False),
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
else:
    test_paths = tf.constant(
        np.char.add(TEST_IMG_DIR + "/", sample_df["image_id"].to_numpy(dtype=str))
    )
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
    test_ds = test_ds.map(
        _decode_resize_normalize,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )
    test_ds = test_ds.apply(tf.data.experimental.ignore_errors())
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

probs = model.predict(test_ds, verbose=0)
y_preds = probs.argmax(axis=1).astype(np.int64).tolist()
assert len(y_preds) == len(test_images), "Prediction length mismatch"




## === cell 7
df_sub = pd.DataFrame({"image_id": test_images, "label": y_preds})
print(df_sub.head())
print("Submission shape:", df_sub.shape)
print("Label value counts:\n", df_sub["label"].value_counts().sort_index())




## === cell 8
out_path = "submission.csv"
df_sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "bytes:", os.path.getsize(out_path))
