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

0.6143850105772136

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_FORCE_GPU_ALLOW_GROWTH", "true")

import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF version:", tf.__version__)

BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")
TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)
df["label"] = df["label"].astype(str)  # categorical mode expects string labels

df["filepath"] = (
    TRAIN_IMG_DIR + os.sep + df["image_id"].astype(str).to_numpy()
).astype(str)

sample_paths = df["filepath"].head(10).to_list()
missing = sum(not os.path.exists(p) for p in sample_paths)
print("Train rows:", len(df), "| Missing (first 10 checked):", missing)
df.head()




## === cell 2
def stratified_split_dataframe(dataframe, label_col, test_size=0.15, seed=42):
    rng = np.random.RandomState(seed)
    labels = dataframe[label_col].to_numpy()
    all_idx = np.arange(len(dataframe))

    train_idx_list = []
    valid_idx_list = []

    uniq, inv = np.unique(labels, return_inverse=True)
    for k in range(len(uniq)):
        idx = all_idx[inv == k]
        rng.shuffle(idx)
        n_valid = int(np.round(len(idx) * test_size))
        n_valid = min(max(n_valid, 1), len(idx) - 1) if len(idx) > 1 else 0
        valid_idx_list.append(idx[:n_valid])
        train_idx_list.append(idx[n_valid:])

    train_idx = (
        np.concatenate(train_idx_list) if train_idx_list else np.array([], dtype=int)
    )
    valid_idx = (
        np.concatenate(valid_idx_list) if valid_idx_list else np.array([], dtype=int)
    )

    rng2 = np.random.RandomState(seed)
    rng2.shuffle(train_idx)
    rng2.shuffle(valid_idx)

    train_df = dataframe.iloc[train_idx].reset_index(drop=True)
    valid_df = dataframe.iloc[valid_idx].reset_index(drop=True)
    return train_df, valid_df


train_df, valid_df = stratified_split_dataframe(df, "label", test_size=0.15, seed=SEED)
print("Train:", len(train_df), "Valid:", len(valid_df))



## === cell 3
IMG_SIZE = 380
BATCH_SIZE = 16  # keep moderate to fit typical Kaggle GPU memory
NUM_CLASSES = 5
EPOCHS = 5  # unchanged to preserve training approach/semantics

AUTO = tf.data.AUTOTUNE

train_tfrecs = tf.io.gfile.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec"))
test_tfrecs = tf.io.gfile.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec"))
assert len(train_tfrecs) > 0, "No train TFRecords found."
assert len(test_tfrecs) > 0, "No test TFRecords found."

train_tfrecs = sorted(train_tfrecs)
test_tfrecs = sorted(test_tfrecs)

FEATURE_DESC_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
FEATURE_DESC_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

_rot_layer = tf.keras.layers.RandomRotation(
    factor=15.0 / 180.0, fill_mode="nearest", seed=SEED
)


def _decode_resize(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    return img


def _augment(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    img = _rot_layer(img[None, ...], training=True)[0]
    scale = tf.random.stateless_uniform([], seed=[SEED, 2], minval=0.9, maxval=1.1)
    new_size = tf.cast(tf.round(scale * IMG_SIZE), tf.int32)
    img2 = tf.image.resize(img, [new_size, new_size], method="bilinear")

    max_shift = tf.cast(tf.round(0.05 * IMG_SIZE), tf.int32)
    dx = tf.random.stateless_uniform(
        [], seed=[SEED, 3], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )
    dy = tf.random.stateless_uniform(
        [], seed=[SEED, 4], minval=-max_shift, maxval=max_shift + 1, dtype=tf.int32
    )

    img2 = tf.image.resize_with_crop_or_pad(
        img2, IMG_SIZE + 2 * max_shift, IMG_SIZE + 2 * max_shift
    )
    start_x = max_shift + dx
    start_y = max_shift + dy
    img2 = tf.image.crop_to_bounding_box(img2, start_y, start_x, IMG_SIZE, IMG_SIZE)
    return img2


def _preprocess(img):
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # identical Keras ResNet50 preprocessing
    return img


def _parse_train(example):
    ex = tf.io.parse_single_example(example, FEATURE_DESC_TRAIN)
    img = _decode_resize(ex["image"])
    img = _augment(img)
    img = _preprocess(img)
    label = tf.one_hot(tf.cast(ex["label"], tf.int32), NUM_CLASSES)
    return img, label


def _parse_valid(example):
    ex = tf.io.parse_single_example(example, FEATURE_DESC_TRAIN)
    img = _decode_resize(ex["image"])
    img = _preprocess(img)
    label = tf.one_hot(tf.cast(ex["label"], tf.int32), NUM_CLASSES)
    return img, label


def _parse_test(example):
    ex = tf.io.parse_single_example(example, FEATURE_DESC_TEST)
    img = _decode_resize(ex["image"])
    img = _preprocess(img)
    return img, ex["image_name"]


train_ids = set(train_df["image_id"].astype(str).tolist())
valid_ids = set(valid_df["image_id"].astype(str).tolist())

all_ids = sorted(train_ids | valid_ids)
keys = tf.constant(all_ids, dtype=tf.string)
vals = tf.constant([1 if k in train_ids else 2 for k in all_ids], dtype=tf.int64)

split_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(keys=keys, values=vals),
    default_value=tf.constant(0, dtype=tf.int64),
)

raw_train = tf.data.TFRecordDataset(train_tfrecs, num_parallel_reads=AUTO)


def _get_example_and_name(example):
    ex = tf.io.parse_single_example(example, FEATURE_DESC_TRAIN)
    return example, ex["image_name"]


named = raw_train.map(_get_example_and_name, num_parallel_calls=AUTO)

train_raw = named.filter(lambda ex, name: tf.equal(split_table.lookup(name), 1)).map(
    lambda ex, name: ex, num_parallel_calls=AUTO
)
valid_raw = named.filter(lambda ex, name: tf.equal(split_table.lookup(name), 2)).map(
    lambda ex, name: ex, num_parallel_calls=AUTO
)

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = (
    train_raw.with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_parse_train, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

valid_ds = (
    valid_raw.with_options(options)
    .map(_parse_valid, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)

train_steps = int(np.ceil(len(train_df) / BATCH_SIZE))
valid_steps = int(np.ceil(len(valid_df) / BATCH_SIZE))

print("Steps - train:", train_steps, "valid:", valid_steps)



## === cell 4
base = ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE, IMG_SIZE, 3)
)
base.trainable = False  # unchanged transfer-learning approach

inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=16,
)

model.summary()



## === cell 5
ckpt_path = "/kaggle/working/best_model.keras"
callbacks = [
    ModelCheckpoint(
        ckpt_path, monitor="val_accuracy", save_best_only=True, mode="max", verbose=1
    ),
    ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=1, verbose=1),
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)

if os.path.exists(ckpt_path):
    best_model = tf.keras.models.load_model(ckpt_path)
else:
    best_model = model



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/193173557.py in <cell line: 0>()
      7 ]
      8 
----> 9 history = model.fit(
     10     train_ds,
     11     validation_data=valid_ds,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

InvalidArgumentError: Graph execution error:

Detected at node ParseSingleExample/ParseExample/ParseExampleV2 defined at (most recent call last):
<stack traces unavailable>
Error in user-defined function passed to ParallelMapDatasetV2:2 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::ParallelMapV2::Filter::ParallelMapV2: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNextAsOptional]] [Op:__inference_multi_step_on_iterator_13601]

## === cell 6
sub = pd.read_csv(SAMPLE_SUB)
sub["filepath"] = (
    TEST_IMG_DIR + os.sep + sub["image_id"].astype(str).to_numpy()
).astype(str)

sample_test_paths = sub["filepath"].head(10).to_list()
missing_test = sum(not os.path.exists(p) for p in sample_test_paths)
print("Test rows:", len(sub), "| Missing (first 10 checked):", missing_test)

raw_test = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTO).with_options(
    options
)
test_ds = (
    raw_test.map(_parse_test, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTO)
)

test_names = []
for _, names in test_ds:
    test_names.append(names.numpy())
test_names = np.concatenate(test_names).astype(str)

test_img_ds = test_ds.map(lambda x, n: x, num_parallel_calls=AUTO)
test_steps = int(np.ceil(len(test_names) / BATCH_SIZE))

probs = best_model.predict(test_img_ds, verbose=1, steps=test_steps)
preds = probs.argmax(axis=1).astype(int)

pred_df = pd.DataFrame({"image_id": test_names, "label": preds})

submission = sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert (
    submission["label"].notna().all()
), "Some test image_ids did not receive predictions."

submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "| rows:", len(submission))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4198351289.py in <cell line: 0>()
     27 test_steps = int(np.ceil(len(test_names) / BATCH_SIZE))
     28 
---> 29 probs = best_model.predict(test_img_ds, verbose=1, steps=test_steps)
     30 preds = probs.argmax(axis=1).astype(int)
     31 

NameError: name 'best_model' is not defined

## === cell 7
assert os.path.exists("submission.csv"), "submission.csv was not created."
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == [
    "image_id",
    "label",
], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(
    pd.read_csv(SAMPLE_SUB)
), "Row count mismatch vs sample_submission."
chk.head()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1925187473.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv"), "submission.csv was not created."
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == [
      4     "image_id",
      5     "label",

AssertionError: submission.csv was not created.
