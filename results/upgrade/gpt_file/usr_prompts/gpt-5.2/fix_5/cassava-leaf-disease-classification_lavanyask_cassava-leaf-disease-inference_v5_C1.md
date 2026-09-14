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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
TRAIN_CSV = os.path.join(ROOT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(ROOT_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(ROOT_DIR, "train_images")
TEST_DIR = os.path.join(ROOT_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sample_df.head())
print("Train rows:", len(train_df), "Test rows:", len(sample_df))



## === cell 3
IMG_SIZE = 300
NUM_CLASSES = 5
BATCH_SIZE = 32


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # faster than tf.image.decode_jpeg
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_train_val_datasets(train_df, train_dir):
    image_ids = train_df["image_id"].values
    labels = train_df["label"].astype(np.int64).values

    idx = np.arange(len(image_ids))
    np.random.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_paths_np = np.array(
        [os.path.join(train_dir, fn) for fn in image_ids[tr_idx]], dtype=np.str_
    )
    va_paths_np = np.array(
        [os.path.join(train_dir, fn) for fn in image_ids[va_idx]], dtype=np.str_
    )
    y_tr = labels[tr_idx]
    y_va = labels[va_idx]

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.autotune_buffers = True

    AUTOTUNE = tf.data.AUTOTUNE
    num_map_calls = AUTOTUNE  # let TF autotune parallelism for best throughput

    tr_paths = tf.constant(tr_paths_np)
    va_paths = tf.constant(va_paths_np)
    y_tr_tf = tf.constant(y_tr, dtype=tf.int64)
    y_va_tf = tf.constant(y_va, dtype=tf.int64)

    ds_tr = tf.data.Dataset.from_tensor_slices((tr_paths, y_tr_tf))
    ds_tr = ds_tr.with_options(options)

    shuffle_buf = min(int(tr_paths.shape[0]), 8192)
    ds_tr = ds_tr.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )

    ds_tr = ds_tr.map(
        lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=num_map_calls
    )

    ds_tr = ds_tr.batch(BATCH_SIZE, drop_remainder=False)
    ds_tr = ds_tr.prefetch(AUTOTUNE)

    ds_va = tf.data.Dataset.from_tensor_slices((va_paths, y_va_tf))
    ds_va = ds_va.with_options(options)
    ds_va = ds_va.map(
        lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=num_map_calls
    )

    ds_va = ds_va.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    return ds_tr, ds_va, (int(tr_paths.shape[0]), int(va_paths.shape[0]))


train_ds, val_ds, (n_train, n_val) = make_train_val_datasets(train_df, TRAIN_DIR)
print(
    "Train batches:",
    int(np.ceil(n_train / BATCH_SIZE)),
    "Val batches:",
    int(np.ceil(n_val / BATCH_SIZE)),
)
print("Train samples:", n_train, "Val samples:", n_val)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3855129790.py in <cell line: 0>()
     84 
     85 
---> 86 train_ds, val_ds, (n_train, n_val) = make_train_val_datasets(train_df, TRAIN_DIR)
     87 print(
     88     "Train batches:",

/tmp/ipykernel_11/3855129790.py in make_train_val_datasets(train_df, train_dir)
     42     options.experimental_optimization.parallel_batch = True
     43     options.experimental_optimization.map_and_batch_fusion = True
---> 44     options.experimental_optimization.autotune_buffers = True
     45 
     46     AUTOTUNE = tf.data.AUTOTUNE

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

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



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/914573334.py in <cell line: 0>()
----> 1 history = new_model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)
      2 

NameError: name 'train_ds' is not defined

## === cell 6
test_images = sample_df["image_id"].values.tolist()
test_paths_np = np.array(
    [os.path.join(TEST_DIR, fn) for fn in test_images], dtype=np.str_
)

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.experimental_optimization.map_and_batch_fusion = True
options.experimental_optimization.autotune_buffers = True

test_paths = tf.constant(test_paths_np)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = test_ds.map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

proba = new_model.predict(test_ds, verbose=0)
preds = proba.argmax(axis=1).astype(np.int64).tolist()

print("Preds length:", len(preds), "Expected:", len(test_images))
print("First 10 preds:", preds[:10])

sub = pd.DataFrame({"image_id": test_images, "label": preds})

assert sub.shape[0] == sample_df.shape[0]
assert list(sub.columns) == ["image_id", "label"]

print(sub.head())
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub))

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4171719237.py in <cell line: 0>()
     10 options.experimental_optimization.parallel_batch = True
     11 options.experimental_optimization.map_and_batch_fusion = True
---> 12 options.experimental_optimization.autotune_buffers = True
     13 
     14 # --- Performance: tf.constant string tensor + autotuned parallel calls for decode/resize.

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.
