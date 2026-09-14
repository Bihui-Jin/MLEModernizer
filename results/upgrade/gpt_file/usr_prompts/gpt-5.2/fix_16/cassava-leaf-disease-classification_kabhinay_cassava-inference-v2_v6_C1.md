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

2.7

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
from __future__ import print_function
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

np.random.seed(42)
random.seed(42)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None
try:
    import tensorflow as tf

    try:
        tf.random.set_seed(42)
    except Exception:
        try:
            tf.compat.v1.set_random_seed(42)
        except Exception:
            pass

    try:
        tf.config.experimental.enable_op_determinism(True)
    except Exception:
        pass

    print("TF version:", getattr(tf, "__version__", "unknown"))
    print("Eager execution:", tf.executing_eagerly())
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e
    tf = None
    print("TensorFlow import failed; will write baseline submission. Error:", repr(e))

print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFREC_DIR))




## === cell 1
if not TF_AVAILABLE:
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    train_df = pd.read_csv(TRAIN_CSV_PATH)

    fallback_label = int(train_df["label"].value_counts().index[0])
    submission = sample_sub.copy()
    submission["label"] = fallback_label

    out_path = "/kaggle/working/submission.csv"
    submission.to_csv(out_path, index=False)
    print("Wrote baseline:", out_path)
    print("fallback_label:", fallback_label)
    print(submission.head())




## === cell 2
if TF_AVAILABLE:
    IMG_SIZE = 448  # keep consistent with the original script's input size
    NUM_CLASSES = 5

    _ID_KEYS_TO_TRY = ["image_id", "id"]
    _LABEL_KEYS_TO_TRY = [
        "label",
        "target",
        "class",
    ]  # actual TFRecords often use "target"

    def _first_present_key(feature_dict, keys):
        for k in keys:
            if k in feature_dict:
                return k
        return None

    def _decode_and_resize(img_bytes):
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
        return img

    def parse_train_example(example_proto):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "class": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
            "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }
        ex = tf.io.parse_single_example(example_proto, features)

        label_key = _first_present_key(ex, _LABEL_KEYS_TO_TRY)
        label_raw = (
            ex[label_key] if label_key is not None else tf.constant(-1, tf.int64)
        )
        label = tf.cast(label_raw, tf.int32)
        label = tf.clip_by_value(label, 0, NUM_CLASSES - 1)

        img = _decode_and_resize(ex["image"])
        label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, label_oh

    def parse_test_example(example_proto):
        features = {
            "image": tf.io.FixedLenFeature([], tf.string),
            "id": tf.io.FixedLenFeature([], tf.string, default_value=""),
            "image_id": tf.io.FixedLenFeature([], tf.string, default_value=""),
        }
        ex = tf.io.parse_single_example(example_proto, features)
        img = _decode_and_resize(ex["image"])

        image_id = ex["image_id"]
        image_id = tf.where(tf.equal(image_id, ""), ex["id"], image_id)
        return img, image_id

    def list_tfrecs(directory):
        if not os.path.isdir(directory):
            return []
        files = []
        for fn in sorted(os.listdir(directory)):
            if (
                fn.endswith(".tfrec")
                or fn.endswith(".tfrecord")
                or fn.endswith(".tfrecords")
            ):
                files.append(os.path.join(directory, fn))
        return files

    train_tfrecs = list_tfrecs(TRAIN_TFREC_DIR)
    test_tfrecs = list_tfrecs(TEST_TFREC_DIR)

    print("Num train tfrecs:", len(train_tfrecs))
    print("Num test tfrecs:", len(test_tfrecs))
    if len(train_tfrecs) == 0 or len(test_tfrecs) == 0:
        raise IOError(
            "Expected TFRecords under %s and %s" % (TRAIN_TFREC_DIR, TEST_TFREC_DIR)
        )




## === cell 3
if TF_AVAILABLE:
    BATCH_SIZE = 16  # preserve original inference batch size
    EPOCHS = 3  # preserve original setting

    try:
        AUTOTUNE = tf.data.AUTOTUNE
    except Exception:
        AUTOTUNE = tf.data.experimental.AUTOTUNE

    _ds_options = tf.data.Options()
    try:
        _ds_options.experimental_deterministic = True
    except Exception:
        pass
    try:
        _ds_options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    def make_train_dataset(tfrecs, shuffle_buffer=2048):
        ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_ds_options)
        ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE)

        ds = ds.cache()

        ds = ds.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def make_valid_dataset(tfrecs):
        ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_ds_options)
        ds = ds.map(parse_train_example, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    def make_test_dataset(tfrecs):
        ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=AUTOTUNE)
        ds = ds.with_options(_ds_options)
        ds = ds.map(parse_test_example, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds

    if len(train_tfrecs) >= 4:
        valid_tfrecs = train_tfrecs[-2:]
        trn_tfrecs = train_tfrecs[:-2]
    else:
        valid_tfrecs = train_tfrecs[-1:]
        trn_tfrecs = train_tfrecs[:-1]

    train_ds = make_train_dataset(trn_tfrecs)
    valid_ds = make_valid_dataset(valid_tfrecs)
    test_ds = make_test_dataset(test_tfrecs)

    train_df = pd.read_csv(TRAIN_CSV_PATH)
    num_train = int(train_df.shape[0])

    frac_valid = (
        (len(valid_tfrecs) / float(len(train_tfrecs))) if len(train_tfrecs) else 0.0
    )
    num_valid = int(np.round(num_train * frac_valid))
    num_valid = max(1, min(num_train - 1, num_valid)) if num_train > 1 else 0
    num_train_eff = max(1, num_train - num_valid)

    steps_per_epoch = max(1, int(np.ceil(num_train_eff / float(BATCH_SIZE))))
    validation_steps = (
        max(1, int(np.ceil(num_valid / float(BATCH_SIZE)))) if num_valid else 1
    )

    print(
        "num_train:",
        num_train,
        "num_valid(approx):",
        num_valid,
        "num_train_eff:",
        num_train_eff,
    )
    print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)




## === cell 4
if TF_AVAILABLE:
    keras = tf.keras
    layers = tf.keras.layers

    inp = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    out = layers.Dense(NUM_CLASSES, activation="softmax", name="probs")(x)
    model = keras.Model(inp, out)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    print("Model built and compiled.")




## === cell 5
if TF_AVAILABLE:
    history = model.fit(
        train_ds,
        epochs=EPOCHS,
        steps_per_epoch=steps_per_epoch,
        validation_data=valid_ds,
        validation_steps=validation_steps,
        verbose=1,
    )




## === cell 6
if TF_AVAILABLE:
    sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
    sample_image_ids = sample_sub["image_id"].astype(str).values

    sample_index = dict((img_id, i) for i, img_id in enumerate(sample_image_ids))

    test_preds = np.zeros((len(sample_image_ids),), dtype=np.int64)

    num_seen = 0
    num_mapped = 0

    for bx, bid in test_ds:
        bid_np = bid.numpy()

        try:
            ids = np.char.decode(bid_np.astype(np.bytes_), "utf-8")
        except Exception:
            ids = bid_np.astype(str)

        probs = model.predict_on_batch(bx)
        cls = np.argmax(probs, axis=1).astype(np.int64)

        num_seen += int(ids.shape[0])

        for img_id, c in zip(ids.tolist(), cls.tolist()):
            idx = sample_index.get(img_id, None)
            if idx is not None:
                test_preds[idx] = int(c)
                num_mapped += 1

    print("Test samples seen:", num_seen, "Expected:", len(sample_image_ids))
    print("Mapped into sample_submission order:", num_mapped)

    if num_mapped < len(sample_image_ids):
        fallback_label = int(train_df["label"].value_counts().index[0])
        missing = np.where(test_preds == 0)[0]  # preserve original logic verbatim
        test_preds[missing] = fallback_label
        print(
            "Warning: mapping incomplete; filled potential unmapped rows with fallback_label:",
            fallback_label,
        )

    submission = pd.DataFrame(
        {"image_id": sample_image_ids.tolist(), "label": test_preds.astype(np.int64)}
    )
    assert submission.shape[0] == sample_sub.shape[0]
    assert list(submission.columns) == ["image_id", "label"]

    out_path = "/kaggle/working/submission.csv"
    submission.to_csv(out_path, index=False)
    print("Wrote:", out_path)
    print(submission.head())
