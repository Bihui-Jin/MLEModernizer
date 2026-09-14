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

0.6158960411000303

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14425) has done: 'The timeout is dominated by slow Python-based JPEG decoding/augmentation via `ImageDataGenerator.flow_from_dataframe` plus oversized `IMG_SIZE=300`, causing the CPU to spend most time in Python generators rather than TensorFlow. Without changing the model or training loop semantics, I switch the input pipeline to a `tf.data` pipeline that uses TensorFlow ops for JPEG decode/resize/rescale and applies the same augmentations (rotation/shift/zoom/flip) inside the graph, with caching/prefetching and parallel mapping to remove Python overhead. I also ensure deterministic behavior via seeds and deterministic dataset options, and I keep the same split, labels, loss, optimizer, epochs, and prediction semantics (argmax of softmax). Paths remain unchanged and the submission format is identical.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"

print("TF version:", tf.__version__)


def _fast_jpg_count(img_dir: str) -> int:
    try:
        return sum(1 for n in os.listdir(img_dir) if n.lower().endswith(".jpg"))
    except Exception:
        return -1


print("Train images:", _fast_jpg_count(TRAIN_IMG_DIR))
print("Test images:", _fast_jpg_count(TEST_IMG_DIR))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)

df["path"] = (TRAIN_IMG_DIR + os.sep + df["image_id"].astype(str)).astype(str)
df["label"] = df["label"].astype(str)

train_df, val_df = train_test_split(
    df, test_size=0.15, random_state=SEED, stratify=df["label"]
)

IMG_SIZE = 300
BATCH_SIZE = 32

train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

class_names = sorted(df["label"].unique().tolist())
NUM_CLASSES = len(class_names)
print("Num classes:", NUM_CLASSES)
print("Class names:", class_names)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"

class_to_idx = {c: i for i, c in enumerate(class_names)}
train_df["label_idx"] = train_df["label"].map(class_to_idx).astype(np.int32)
val_df["label_idx"] = val_df["label"].map(class_to_idx).astype(np.int32)

AUTOTUNE = tf.data.AUTOTUNE
MAP_PARALLEL_CALLS = AUTOTUNE
PREFETCH_BUFFER = AUTOTUNE
SHUFFLE_BUFFER_CAP = 2048


def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_ds(
    paths, labels=None, training=False, batch_size=32, shuffle=False, cache_id=""
):
    paths = tf.constant(paths)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        labels = tf.constant(labels)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    ds = ds.with_options(options)

    if shuffle:
        buf = int(min(int(paths.shape[0]), SHUFFLE_BUFFER_CAP))
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    if labels is None:

        def _map_fn(path):
            img = _decode_resize_rescale(path)
            return img

        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                _map_fn,
                batch_size=batch_size,
                num_parallel_calls=MAP_PARALLEL_CALLS,
                drop_remainder=False,
                deterministic=True,
            )
        )
        if cache_id:
            ds = ds.cache()

    else:

        def _map_fn(path, label_idx):
            img = _decode_resize_rescale(path)
            y = tf.one_hot(tf.cast(label_idx, tf.int32), NUM_CLASSES)
            return img, y

        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                _map_fn,
                batch_size=batch_size,
                num_parallel_calls=MAP_PARALLEL_CALLS,
                drop_remainder=False,
                deterministic=True,
            )
        )
        if cache_id and (not training) and (not shuffle):
            ds = ds.cache()

    ds = ds.prefetch(PREFETCH_BUFFER)
    return ds


train_paths = train_df["path"].values.astype(str)
val_paths = val_df["path"].values.astype(str)

train_labels_idx = train_df["label_idx"].values.astype(np.int32)
val_labels_idx = val_df["label_idx"].values.astype(np.int32)

train_ds = make_ds(
    train_paths,
    train_labels_idx,
    training=True,
    batch_size=BATCH_SIZE,
    shuffle=True,
    cache_id="train_300x300",
)
val_ds = make_ds(
    val_paths,
    val_labels_idx,
    training=False,
    batch_size=BATCH_SIZE,
    shuffle=False,
    cache_id="val_300x300",
)

print("Datasets ready:", train_ds, val_ds)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/20028726.py in <cell line: 0>()
    115 val_labels_idx = val_df["label_idx"].values.astype(np.int32)
    116 
--> 117 train_ds = make_ds(
    118     train_paths,
    119     train_labels_idx,

/tmp/ipykernel_11/20028726.py in make_ds(paths, labels, training, batch_size, shuffle, cache_id)
     93 
     94         ds = ds.apply(
---> 95             tf.data.experimental.map_and_batch(
     96                 _map_fn,
     97                 batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    381               'in a future version' if date is None else ('after %s' % date),
    382               instructions)
--> 383       return func(*args, **kwargs)
    384 
    385     doc_controls.set_deprecated(new_func)

TypeError: map_and_batch() got an unexpected keyword argument 'deterministic'

## === cell 2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Activation,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras import layers

data_augmentation = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(0.06, seed=SEED),  # ~ +/- 10 degrees
        layers.RandomZoom(0.05, seed=SEED),
        layers.RandomTranslation(0.05, 0.05, seed=SEED),
    ],
    name="data_augmentation",
)

model = Sequential(
    [
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        data_augmentation,
        Conv2D(32, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(128, (3, 3)),
        Activation("relu"),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(256),
        Activation("relu"),
        Dropout(0.5),
        Dense(NUM_CLASSES, activation="softmax"),
    ]
)

model.compile(
    loss="categorical_crossentropy",
    optimizer="adam",
    metrics=["accuracy"],
    steps_per_execution=16,
)
model.summary()

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/709685795.py in <cell line: 0>()
     52 
     53 history = model.fit(
---> 54     train_ds,
     55     validation_data=val_ds,
     56     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 3
sub = pd.read_csv(SAMPLE_SUB)

sub["path"] = (TEST_IMG_DIR + os.sep + sub["image_id"].astype(str)).astype(str)

test_paths = sub["path"].values.astype(str)

test_ds = make_ds(
    test_paths,
    labels=None,
    training=False,
    batch_size=64,
    shuffle=False,
    cache_id="test_300x300",
)

pred_test = model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

submission = pd.DataFrame(
    {"image_id": sub["image_id"].values, "label": pred_test_labels}
)
assert list(submission.columns) == ["image_id", "label"]
assert submission.shape[0] == sub.shape[0]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1026154090.py in <cell line: 0>()
      5 test_paths = sub["path"].values.astype(str)
      6 
----> 7 test_ds = make_ds(
      8     test_paths,
      9     labels=None,

/tmp/ipykernel_11/20028726.py in make_ds(paths, labels, training, batch_size, shuffle, cache_id)
     73 
     74         ds = ds.apply(
---> 75             tf.data.experimental.map_and_batch(
     76                 _map_fn,
     77                 batch_size=batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    381               'in a future version' if date is None else ('after %s' % date),
    382               instructions)
--> 383       return func(*args, **kwargs)
    384 
    385     doc_controls.set_deprecated(new_func)

TypeError: map_and_batch() got an unexpected keyword argument 'deterministic'
