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

0.8529767301299487

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.19955) has done: 'Your notebook fails early due to an unnecessary `tensorflow_hub` import that crashes protobuf in this environment, and then later because it references SavedModels under `/kaggle/input/...` paths that don’t exist. To make it run end-to-end and produce a valid `submission.csv`, I remove the hub dependency, drop the missing external SavedModel ensemble, and keep the core idea of transfer learning with EfficientNet + Keras generators. I also fix label handling to be numeric `0..4` (as required), ensure `steps_per_epoch` are correct, and generate predictions in the exact order of `sample_submission.csv` so image_id alignment is guaranteed. This should yield a reasonable accuracy submission and, most importantly, a valid CSV output.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")
train_tfrec_files = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrec_files = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in tf.io.gfile.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrec_files) > 0, "No train TFRecords found."
assert len(test_tfrec_files) > 0, "No test TFRecords found."

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["image_id"].astype(str)
train_df["label"] = train_df["label"].astype(int)
assert (
    train_df["path"].map(os.path.exists).mean() > 0.99
), "Training image paths look wrong."
assert set(train_df["label"].unique()).issubset(
    set(range(5))
), "Labels must be in {0,1,2,3,4}."
train_df["label_str"] = train_df["label"].astype(str)

train_split, valid_split = train_test_split(
    train_df,
    test_size=0.2,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32
NUM_CLASSES = 5

callbacks = [
    EarlyStopping(monitor="val_loss", patience=3, restore_best_weights=True),
    ReduceLROnPlateau(
        monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
    ),
]

AUTOTUNE = tf.data.AUTOTUNE

data_augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(0.125, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=SEED),
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=SEED),
    ],
    name="data_augment",
)


@tf.function
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _decode_resize_preprocess_from_path(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_preprocess_from_bytes(img_bytes)


@tf.function
def _augment_batch(x, y):
    return data_augment(x, training=True), y


SHUFFLE_BUFFER = 8192

_TRAIN_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _parse_train_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TRAIN_FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int64)
    return img, label


def _make_cache_path(name: str) -> str:
    return os.path.join("/kaggle/working", name)


def make_train_ds_from_tfrecords(tfrec_files, n_items):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.shuffle(SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(_make_cache_path("train_decoded_cache.tf-data"))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.prefetch(AUTOTUNE)
    steps_per_epoch = int(np.ceil(n_items / BATCH_SIZE))
    return ds, steps_per_epoch


def make_valid_ds_from_tfrecords(tfrec_files, n_items):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(_make_cache_path("valid_decoded_cache.tf-data"))
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    steps = int(np.ceil(n_items / BATCH_SIZE))
    return ds, steps


train_paths = train_split["path"].astype(str).values
train_labels = train_split["label"].values.astype(np.int64)
valid_paths = valid_split["path"].astype(str).values
valid_labels = valid_split["label"].values.astype(np.int64)

n_train = len(train_split)
n_valid = len(valid_split)

train_ds, train_steps = make_train_ds_from_tfrecords(train_tfrec_files, n_train)
valid_ds, valid_steps = make_valid_ds_from_tfrecords(train_tfrec_files, n_valid)

class_indices = {str(i): i for i in range(NUM_CLASSES)}
idx_to_classname = {v: k for k, v in class_indices.items()}
print("class_indices:", class_indices)
print("train_steps:", train_steps, "valid_steps:", valid_steps)



## === cell 2
base = EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3)
)
base.trainable = False  # minimal/training-stable

x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.2)(x)
out = Dense(NUM_CLASSES, activation="softmax")(x)

model = Model(inputs=base.input, outputs=out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()



## === cell 3
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=10,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

history_ft = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=5,
    callbacks=callbacks,
    verbose=1,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1655026423.py in <cell line: 0>()
      2 # train_ds is now infinite due to repeat(); specify steps_per_epoch/validation_steps
      3 # so training runs the same number of samples per epoch as the original finite dataset.
----> 4 history = model.fit(
      5     train_ds,
      6     validation_data=valid_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:4 transformation with iterator: Iterator::Root::Prefetch::Map::BatchV2::FileCacheImpl::Map: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_17090]

## === cell 4
sample_df["path"] = TEST_IMG_DIR + "/" + sample_df["image_id"].astype(str)

missing = sample_df["path"].map(os.path.exists)
if missing.mean() < 0.99:
    bad = sample_df.loc[~missing, "path"].head(5).tolist()
    raise FileNotFoundError(f"Some test images not found. Example missing paths: {bad}")

_TEST_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, _TEST_FEATURES)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    return img, ex["image_name"]


def make_test_ds_from_tfrecords(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds_from_tfrecords(test_tfrec_files)

probs, names = model.predict(test_ds, verbose=1)
names = names.astype("U")

pred_indices = np.argmax(probs, axis=1).astype(int)
name_to_pred = dict(zip(names.tolist(), pred_indices.tolist()))
pred_labels = sample_df["image_id"].map(name_to_pred).values.astype(int)

submission_df = pd.DataFrame(
    {"image_id": sample_df["image_id"].values, "label": pred_labels}
)
submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission_df.head())
print(submission_df["label"].value_counts().sort_index())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2064618660.py in <cell line: 0>()
     29 test_ds = make_test_ds_from_tfrecords(test_tfrec_files)
     30 
---> 31 probs, names = model.predict(test_ds, verbose=1)
     32 # names is a numpy array of bytes; decode once deterministically.
     33 names = names.astype("U")

ValueError: too many values to unpack (expected 2)
