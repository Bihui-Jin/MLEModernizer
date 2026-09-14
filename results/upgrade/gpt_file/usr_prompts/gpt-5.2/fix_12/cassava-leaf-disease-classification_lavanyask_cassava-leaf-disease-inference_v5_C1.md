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

0.8609851919008764

# 6. Current score

0.6719

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.6719) has done: 'Main bottlenecks are (1) expensive JPEG decode+resize being executed every epoch without caching, and (2) iterating the test dataset twice (once for predict, once to collect names), causing a full second pass through TFRecords. I add safe `tf.data` caching (to RAM or disk fallback) for train/val/test after parsing to avoid repeated decode/resize work while keeping the same examples and labels. I also restructure test inference to produce `(image, name)` in one pipeline and collect names from the same iterator used for prediction, eliminating the second TFRecord pass while preserving identical prediction semantics. Finally, I keep determinism/seed behavior intact and only tune input pipeline options that don’t change the model or training loop logic.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

ROOT_DIR = "/kaggle/input/cassava-leaf-disease-classification"
print("ROOT_DIR exists:", os.path.exists(ROOT_DIR))
print("ROOT_DIR sample:", os.listdir(ROOT_DIR)[:10])



## === cell 1
import tensorflow as tf
from tensorflow.keras import layers, models

print("TensorFlow:", tf.__version__)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism not fully enabled:", repr(e))

try:
    cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(cpu_cnt)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception as e:
    print("Threading config not set:", repr(e))

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(ROOT_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(ROOT_DIR, "test_tfrecords")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sample_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_df))



## === cell 3
IMG_SIZE = 300
NUM_CLASSES = 5
BATCH_SIZE = 32


def _sorted_tfrec_files(dir_path, prefix):
    files = tf.io.gfile.glob(os.path.join(dir_path, f"{prefix}*.tfrec"))
    files = sorted(files)
    if not files:
        raise FileNotFoundError(
            f"No TFRecord files found under {dir_path} with prefix={prefix}"
        )
    return files


TRAIN_TFRECS = _sorted_tfrec_files(TRAIN_TFREC_DIR, "ld_train")
TEST_TFRECS = _sorted_tfrec_files(TEST_TFREC_DIR, "ld_test")

_FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
_FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_normalize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TRAIN)
    img = _decode_resize_normalize_from_bytes(ex["image"])
    y = ex["target"]
    return img, y


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURE_DESC_TEST)
    img = _decode_resize_normalize_from_bytes(ex["image"])
    image_name = ex["image_name"]
    return img, image_name


def _cache_ds(ds, cache_path: str = None):
    if cache_path is None:
        return ds.cache()
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        return ds.cache(cache_path)
    except Exception as e:
        print("Cache fallback to in-memory due to:", repr(e))
        return ds.cache()


def make_train_val_datasets_from_tfrecords(tfrec_files):
    files = list(tfrec_files)
    rng = np.random.RandomState(SEED)
    perm = rng.permutation(len(files))
    split = int(0.9 * len(files))
    tr_files = [files[i] for i in perm[:split]]
    va_files = [files[i] for i in perm[split:]]

    options = tf.data.Options()
    options.experimental_deterministic = False
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.map_and_batch_fusion = True
    except Exception:
        pass

    AUTOTUNE = tf.data.AUTOTUNE

    ds_tr = tf.data.TFRecordDataset(tr_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds_tr = ds_tr.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds_tr = _cache_ds(ds_tr, cache_path="/kaggle/working/cache_train_parsed")
    ds_tr = ds_tr.shuffle(buffer_size=8192, seed=SEED, reshuffle_each_iteration=True)
    ds_tr = ds_tr.batch(BATCH_SIZE, drop_remainder=False)
    ds_tr = ds_tr.prefetch(AUTOTUNE)

    ds_va = tf.data.TFRecordDataset(va_files, num_parallel_reads=AUTOTUNE).with_options(
        options
    )
    ds_va = ds_va.map(
        _parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=False
    )
    ds_va = _cache_ds(ds_va, cache_path="/kaggle/working/cache_val_parsed")
    ds_va = ds_va.batch(BATCH_SIZE, drop_remainder=False)
    ds_va = ds_va.prefetch(AUTOTUNE)

    return ds_tr, ds_va


train_ds, val_ds = make_train_val_datasets_from_tfrecords(TRAIN_TFRECS)
print("TFRecord train files:", len(TRAIN_TFRECS), "batch size:", BATCH_SIZE)



## === cell 4
new_model = models.Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        layers.Conv2D(16, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ]
)

new_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

new_model.summary()



## === cell 5
history = new_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)



## === cell 6
AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

test_ds = tf.data.TFRecordDataset(
    TEST_TFRECS, num_parallel_reads=AUTOTUNE
).with_options(options)
test_ds = test_ds.map(
    _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=False
)
test_ds = _cache_ds(test_ds, cache_path="/kaggle/working/cache_test_parsed")
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

proba = new_model.predict(
    test_ds.map(lambda x, name: x, num_parallel_calls=AUTOTUNE), verbose=0
)
preds = proba.argmax(axis=1).astype(np.int64)

names = np.concatenate([bn.numpy() for _, bn in test_ds], axis=0).astype(str).tolist()

print(
    "Preds length:",
    len(preds),
    "Names length:",
    len(names),
    "Expected:",
    len(sample_df),
)

pred_map = pd.DataFrame({"image_id": names, "label": preds})
sub = sample_df[["image_id"]].merge(pred_map, on="image_id", how="left")

assert sub.shape[0] == sample_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]
assert (
    sub["label"].isna().sum() == 0
), "Some test image_ids were not found in TFRecords."

sub["label"] = sub["label"].astype(np.int64)

print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub))
