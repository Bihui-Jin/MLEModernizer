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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.7620126926563917

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow/Keras import crash caused by an incompatible protobuf version by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I remove the broken dependency on a non-existent external pretrained model file and instead train the same kind of image classifier directly from the provided TFRecords so the notebook can run end-to-end. Finally, I ensure inference preserves the exact test image order and writes a valid `submission.csv` with the required `image_id,label` columns. These changes are the minimum needed to unblock execution and produce a submission (and should yield a reasonable accuracy score versus failing to submit).'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow/protobuf crash by removing the incompatible “force pure-Python protobuf” workaround and instead pin protobuf to the C++ implementation (which avoids the `MessageFactory.GetPrototype` error under TF 2.18 + protobuf 6). Then I fix the TFRecord parsing bug by using the correct feature keys for Cassava TFRecords (`class` for labels and `image_name` for IDs), which currently causes training to fail and also explains the very low score when it did run. Finally, I keep your model/training loop intact but ensure the datasets are properly repeated/consumed and the submission is aligned exactly to `sample_submission.csv` order so the produced `submission.csv` is always valid and score-improving.'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation before importing TensorFlow (this is the most reliable workaround for the `MessageFactory.GetPrototype` error in this environment). Then I fix the TFRecord parsing bug that prevents training by using the correct Cassava TFRecord label key (`label` instead of `class`) while keeping your dataset pipeline and model architecture unchanged. Finally, I keep the submission-writing logic but ensure all code executes end-to-end and produces a valid `submission.csv` with `image_id,label` aligned to `sample_submission.csv`, which should raise accuracy substantially from the current near-random score.'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation (the current `"cpp"` setting fails because `google.protobuf.pyext._message` is unavailable in this environment). Then I keep your TFRecord-based pipeline and CNN model intact, but correct the TFRecord label key to the Cassava TFRecord schema (`"label"` rather than `"class"`) so training/inference work properly and the score moves up from “not yielded”. Finally, I make the train/validation split deterministic but properly shuffled before splitting (to avoid taking the first TFRecord shard as validation), and ensure the submission is written as `submission.csv` with `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow import crash by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before *any* TensorFlow/protobuf import, since this environment lacks the protobuf C++ extension (`google.protobuf.pyext._message`). Then I keep your TFRecord-based pipeline and CNN model intact, but make the TFRecord parsing robust to the actual Cassava schema by accepting either `label` or `class` keys for targets and either `image_name` or `image_id` for test IDs. Finally, I add explicit dataset `.repeat()` plus computed `steps_per_epoch`/`validation_steps` so training terminates correctly, and ensure a valid `submission.csv` is always written in the exact `sample_submission.csv` order.'
- What this solution (achieved 0.11024) has done: 'I fix the TensorFlow import crash (`MessageFactory.GetPrototype`) by forcing protobuf’s pure-Python backend in the most reliable way for TF 2.18 + protobuf 6: setting both `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` and `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=2` before *any* TensorFlow/protobuf import. Then I make TFRecord parsing robust to the actual Cassava TFRecord schema by accepting both possible key names for labels and image IDs, ensuring training labels aren’t all `-1` (which would destroy accuracy). Finally, I keep your model and training loop intact but add a safe label-range assertion early and ensure the submission is aligned to `sample_submission.csv` order and always written as `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_TFREC_GLOB = f"{DATA_ROOT}/train_tfrecords/*.tfrec"
TEST_TFREC_GLOB = f"{DATA_ROOT}/test_tfrecords/*.tfrec"
SAMPLE_SUB_PATH = f"{DATA_ROOT}/sample_submission.csv"

print("DATA_ROOT:", DATA_ROOT)
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
import tensorflow as tf
from tensorflow import keras
import re

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

train_filenames = sorted(tf.io.gfile.glob(TRAIN_TFREC_GLOB))
test_filenames = sorted(tf.io.gfile.glob(TEST_TFREC_GLOB))

assert len(train_filenames) > 0, "No train tfrecords found."
assert len(test_filenames) > 0, "No test tfrecords found."

print("Train tfrecs:", len(train_filenames), "Test tfrecs:", len(test_filenames))
print("First train tfrec:", train_filenames[0])
print("First test tfrec:", test_filenames[0])




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
IMAGE_SIZE = (512, 512)
BATCH_SIZE = 16
NUM_CLASSES = 5
AUTOTUNE = tf.data.AUTOTUNE


def read_tfrec_train(example):
    fmt = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
        "class": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }
    ex = tf.io.parse_single_example(example, fmt)

    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.reshape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    label = tf.cast(ex["label"], tf.int32)
    alt = tf.cast(ex["class"], tf.int32)
    label = tf.where(label >= 0, label, alt)
    return img, label


def read_tfrec_test(example):
    fmt = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    }
    ex = tf.io.parse_single_example(example, fmt)

    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, IMAGE_SIZE)
    img = tf.reshape(img, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])

    image_id = tf.where(
        tf.strings.length(ex["image_name"]) > 0, ex["image_name"], ex["image_id"]
    )
    return img, image_id




## === cell 3
SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(fn).group(1)) for fn in filenames]
    return int(np.sum(n))


TRAIN_NUM = count_data_items(train_filenames)
VAL_FRAC = 0.1
VAL_NUM = int(TRAIN_NUM * VAL_FRAC)
TRN_NUM = TRAIN_NUM - VAL_NUM

all_train = tf.data.TFRecordDataset(train_filenames, num_parallel_reads=AUTOTUNE)
ds = all_train.map(read_tfrec_train, num_parallel_calls=AUTOTUNE)

ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=False)

val_ds = ds.take(VAL_NUM).batch(BATCH_SIZE).prefetch(AUTOTUNE)
train_ds = (
    ds.skip(VAL_NUM)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

steps_per_epoch = int(np.ceil(TRN_NUM / BATCH_SIZE))
validation_steps = int(np.ceil(VAL_NUM / BATCH_SIZE))

train_ds = train_ds.repeat()
val_ds = val_ds.repeat()

print("TRAIN_NUM:", TRAIN_NUM, "TRN_NUM:", TRN_NUM, "VAL_NUM:", VAL_NUM)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)

label_check = ds.map(lambda img, y: y).take(5000)
mins, maxs = [], []
for y in label_check.batch(1024):
    mins.append(int(tf.reduce_min(y).numpy()))
    maxs.append(int(tf.reduce_max(y).numpy()))
print("Label range sample min/max:", min(mins), max(maxs))
assert (
    min(mins) >= 0 and max(maxs) < NUM_CLASSES
), "Parsed labels out of expected range; check TFRecord keys."




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_55/797713463.py in <cell line: 0>()
     48 print("Label range sample min/max:", min(mins), max(maxs))
     49 assert (
---> 50     min(mins) >= 0 and max(maxs) < NUM_CLASSES
     51 ), "Parsed labels out of expected range; check TFRecord keys."
     52 

AssertionError: Parsed labels out of expected range; check TFRecord keys.

## === cell 4
inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPool2D()(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 5
EPOCHS = 3  # keep as-is to respect original runtime intent
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)




## === cell 6
test_ds = tf.data.TFRecordDataset(test_filenames, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.map(read_tfrec_test, num_parallel_calls=AUTOTUNE)

test_images = test_ds.map(lambda img, name: img).batch(BATCH_SIZE).prefetch(AUTOTUNE)
test_names = test_ds.map(lambda img, name: name).batch(1024).prefetch(AUTOTUNE)




## === cell 7
prob = model.predict(test_images, verbose=1)
pred = np.argmax(prob, axis=-1).astype(np.int64)

print("Pred shape:", pred.shape, "prob shape:", prob.shape)




## === cell 8
name_list = []
for batch in test_names:
    arr = batch.numpy()
    if arr.dtype.kind in ("S", "O"):
        name_list.extend(
            [
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                for x in arr.tolist()
            ]
        )
    else:
        name_list.extend(arr.astype("U").tolist())

test_ids = np.array(name_list, dtype="U")

print("IDs:", test_ids.shape, "Pred:", pred.shape)
assert test_ids.shape[0] == pred.shape[0], "Mismatch between test ids and predictions."




## === cell 9
sub = pd.DataFrame({"image_id": test_ids, "label": pred})

sample = pd.read_csv(SAMPLE_SUB_PATH)
sub = sample[["image_id"]].merge(sub, on="image_id", how="left")
assert sub["label"].notna().all(), "Some test image_ids were missing predictions."

sub["label"] = sub["label"].astype(int)
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)




## === cell 10
import os as _os

print(
    "submission.csv exists:",
    _os.path.exists("submission.csv"),
    "size:",
    _os.path.getsize("submission.csv") if _os.path.exists("submission.csv") else None,
)
with open("submission.csv", "r", encoding="utf-8") as f:
    for _ in range(5):
        print(f.readline().strip())
