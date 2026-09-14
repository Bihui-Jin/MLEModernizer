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

# 5. Target score

0.8819885161680266

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the immediate runtime/import failure caused by an incompatible protobuf implementation (the `MessageFactory.GetPrototype` error) by forcing TensorFlow to use the pure-Python protobuf backend before importing TensorFlow. Next, I make the SavedModel loading robust: those `/kaggle/input/...` model directories are not present in your provided dataset, so I detect missing model paths and fall back to a simple, deterministic baseline submission (majority class from `train.csv`) to ensure a valid `submission.csv` is always produced end-to-end. This keeps the core ensemble/prediction logic intact when the models exist, but prevents crashes when they don’t. The output always match the required submission format and `.csv` suffix.'
- What this solution (achieved 0.05531) has done: 'I fix the immediate TensorFlow import crash caused by the protobuf `MessageFactory.GetPrototype` mismatch by forcing the pure-Python protobuf runtime **and** ensuring TensorFlow is imported via the v1-compat path that works reliably in this environment. Next, I remove the dependency on missing external SavedModel directories (which currently forces a weak majority-class fallback and caps accuracy) by switching to using the competition-provided TFRecords for inference with a small Keras CNN trained on `train.csv` labels—this keeps the overall “train a classifier then predict test” semantics while making the pipeline self-contained. I keep the data paths unchanged, ensure deterministic behavior, and write a valid `submission.csv` with the exact required columns. This should substantially improve score from ~0.61 toward the target by using actual image content rather than a constant label.'

# 9. Code solution

## === cell 0
from __future__ import print_function
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import json
import random
import numpy as np
import pandas as pd

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from tensorflow import keras
from tensorflow.keras import layers

np.random.seed(42)
random.seed(42)
try:
    tf.set_random_seed(42)
except Exception:
    pass

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
TRAIN_TFREC_DIR = os.path.join(DATA_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(DATA_DIR, "test_tfrecords")

print("TF version:", tf.__version__)
print("Keras version:", getattr(keras, "__version__", "unknown"))
print("Train TFRecords dir exists:", os.path.isdir(TRAIN_TFREC_DIR))
print("Test TFRecords dir exists:", os.path.isdir(TEST_TFREC_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1

IMG_SIZE = 448  # keep consistent with the original script's input size
NUM_CLASSES = 5


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[int(value)]))


def parse_train_example(example_proto):
    features = {
        "image": tf.FixedLenFeature([], tf.string),
        "image_id": tf.FixedLenFeature([], tf.string),
        "label": tf.FixedLenFeature([], tf.int64),
    }
    ex = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize_images(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    label = tf.cast(ex["label"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


def parse_test_example(example_proto):
    features = {
        "image": tf.FixedLenFeature([], tf.string),
        "image_id": tf.FixedLenFeature([], tf.string),
    }
    ex = tf.parse_single_example(example_proto, features)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize_images(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    image_id = ex["image_id"]
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



## === cell 2
BATCH_SIZE = 16  # preserve original inference batch size
EPOCHS = 3  # small, to fit within runtime while improving over baseline


def make_train_dataset(tfrecs, shuffle_buffer=2048):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=2)
    ds = ds.map(parse_train_example, num_parallel_calls=2)
    ds = ds.shuffle(shuffle_buffer, seed=42, reshuffle_each_iteration=True)
    ds = ds.repeat()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(2)
    return ds


def make_valid_dataset(tfrecs):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=1)
    ds = ds.map(parse_train_example, num_parallel_calls=1)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(1)
    return ds


def make_test_dataset(tfrecs):
    ds = tf.data.TFRecordDataset(tfrecs, num_parallel_reads=2)
    ds = ds.map(parse_test_example, num_parallel_calls=2)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(2)
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
num_valid = (
    int(np.ceil(num_train * (len(valid_tfrecs) / float(len(train_tfrecs)))))
    if len(train_tfrecs)
    else 0
)

steps_per_epoch = max(1, int(np.ceil((num_train - num_valid) / float(BATCH_SIZE))))
validation_steps = max(1, int(np.ceil(num_valid / float(BATCH_SIZE))))
print("num_train:", num_train, "num_valid(approx):", num_valid)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## === cell 3

train_it = tf.data.make_one_shot_iterator(train_ds)
valid_it = tf.data.make_one_shot_iterator(valid_ds)
test_it = tf.data.make_one_shot_iterator(test_ds)

train_batch = train_it.get_next()
valid_batch = valid_it.get_next()

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
    optimizer=keras.optimizers.Adam(lr=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

print(model.summary())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3184636101.py in <cell line: 0>()
     23 
     24 model.compile(
---> 25     optimizer=keras.optimizers.Adam(lr=1e-3),
     26     loss="categorical_crossentropy",
     27     metrics=["accuracy"],

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/adam.py in __init__(self, learning_rate, beta_1, beta_2, epsilon, amsgrad, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     60         **kwargs,
     61     ):
---> 62         super().__init__(
     63             learning_rate=learning_rate,
     64             name=name,

/usr/local/lib/python3.11/dist-packages/keras/src/backend/tensorflow/optimizer.py in __init__(self, *args, **kwargs)
     19 class TFOptimizer(KerasAutoTrackable, base_optimizer.BaseOptimizer):
     20     def __init__(self, *args, **kwargs):
---> 21         super().__init__(*args, **kwargs)
     22         self._distribution_strategy = tf.distribute.get_strategy()
     23 

/usr/local/lib/python3.11/dist-packages/keras/src/optimizers/base_optimizer.py in __init__(self, learning_rate, weight_decay, clipnorm, clipvalue, global_clipnorm, use_ema, ema_momentum, ema_overwrite_frequency, loss_scale_factor, gradient_accumulation_steps, name, **kwargs)
     88             )
     89         if kwargs:
---> 90             raise ValueError(f"Argument(s) not recognized: {kwargs}")
     91 
     92         if name is None:

ValueError: Argument(s) not recognized: {'lr': 0.001}

## === cell 4
sess = keras.backend.get_session()
sess.run(tf.global_variables_initializer())
sess.run(tf.local_variables_initializer())


def fetch_batch(batch_tensors):
    bx, by = sess.run(batch_tensors)
    return bx, by


for epoch in range(EPOCHS):
    tr_losses = []
    tr_accs = []
    for step in range(steps_per_epoch):
        bx, by = fetch_batch(train_batch)
        metrics = model.train_on_batch(bx, by)
        tr_losses.append(float(metrics[0]))
        tr_accs.append(float(metrics[1]))
    va_losses = []
    va_accs = []
    for step in range(validation_steps):
        bx, by = fetch_batch(valid_batch)
        metrics = model.test_on_batch(bx, by)
        va_losses.append(float(metrics[0]))
        va_accs.append(float(metrics[1]))
    print(
        "Epoch %d/%d - loss: %.4f acc: %.4f - val_loss: %.4f val_acc: %.4f"
        % (
            epoch + 1,
            EPOCHS,
            np.mean(tr_losses),
            np.mean(tr_accs),
            np.mean(va_losses),
            np.mean(va_accs),
        )
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2033068380.py in <cell line: 0>()
      1 # Train using explicit numpy fetches from the dataset to stay compatible with TF1 graph mode.
      2 # This avoids relying on TF2-only model.fit(dataset) behavior.
----> 3 sess = keras.backend.get_session()
      4 sess.run(tf.global_variables_initializer())
      5 sess.run(tf.local_variables_initializer())

AttributeError: module 'keras._tf_keras.keras.backend' has no attribute 'get_session'

## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_image_ids = sample_sub["image_id"].astype(str).values.tolist()
sample_index = {img_id: i for i, img_id in enumerate(sample_image_ids)}

test_preds = np.zeros((len(sample_image_ids),), dtype=np.int64)

test_next = test_it.get_next()
num_seen = 0
while True:
    try:
        bx, bid = sess.run(test_next)
        ids = [
            x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
            for x in bid
        ]
        probs = model.predict_on_batch(bx)
        cls = np.argmax(probs, axis=1).astype(np.int64)
        for img_id, c in zip(ids, cls):
            if img_id in sample_index:
                test_preds[sample_index[img_id]] = c
        num_seen += len(ids)
    except tf.errors.OutOfRangeError:
        break

print("Test samples seen:", num_seen, "Expected:", len(sample_image_ids))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2796607508.py in <cell line: 0>()
     11 while True:
     12     try:
---> 13         bx, bid = sess.run(test_next)
     14         # bid is bytes; decode to str
     15         ids = [

NameError: name 'sess' is not defined

## === cell 6
submission = pd.DataFrame({"image_id": sample_image_ids, "label": test_preds})
assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["image_id", "label"]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submission.head())
