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

0.9073738289513448

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

os.environ["PYTHONHASHSEED"] = "42"
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder

label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

train_csv["label_encoded"] = LabelEncoder().fit_transform(train_csv["disease"])
train_csv["label"] = train_csv["label"].astype(str)
train_csv["disease"] = train_csv["disease"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=42,
)

_unique_labels_sorted = sorted(train_csv["label"].unique().tolist())
class_indices = {lab: i for i, lab in enumerate(_unique_labels_sorted)}
NUM_CLASSES = len(class_indices)
print("Detected classes:", NUM_CLASSES)
print("Class indices:", class_indices)

idx_to_label = {v: int(k) for k, v in class_indices.items()}
print("idx_to_label:", idx_to_label)

_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.autotune.enabled = True

TRAIN_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords"
TEST_TFREC_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords"

train_tfrecs = sorted(
    [
        os.path.join(TRAIN_TFREC_DIR, f)
        for f in os.listdir(TRAIN_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TEST_TFREC_DIR, f)
        for f in os.listdir(TEST_TFREC_DIR)
        if f.endswith(".tfrec")
    ]
)

print("Train tfrecords:", len(train_tfrecs), "Test tfrecords:", len(test_tfrecs))

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "id": tf.io.FixedLenFeature([], tf.string),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "id": tf.io.FixedLenFeature([], tf.string),
}

IMG_SIZE = (224, 224)
BATCH_SIZE = 32


def _decode_and_resize(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess(img):
    return preprocess_input(img)


def _parse_train_with_id(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    label = tf.cast(ex["target"], tf.int32)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    image_id = ex["id"]
    return img, label_oh, image_id


def _parse_train(example_proto):
    img, label_oh, _ = _parse_train_with_id(example_proto)
    return img, label_oh


def _parse_test(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    img = _decode_and_resize(ex["image"])
    img = _preprocess(img)
    image_id = ex["id"]
    return img, image_id


import math


@tf.function
def _random_shear(img, shear_range=0.2):
    shear_deg = tf.random.uniform([], -shear_range, shear_range, dtype=tf.float32)
    shear = shear_deg * (math.pi / 180.0)
    sinv = tf.math.sin(shear)
    transform = tf.stack([1.0, -sinv, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0])[tf.newaxis, :]
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        fill_value=0.0,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
    )[0]
    return out


augment_layers = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip("horizontal_and_vertical", seed=42),
        tf.keras.layers.RandomRotation(
            factor=45.0 / 360.0, fill_mode="nearest", seed=42
        ),
        tf.keras.layers.RandomTranslation(0.2, 0.2, fill_mode="nearest", seed=42),
        tf.keras.layers.RandomZoom(0.2, 0.2, fill_mode="nearest", seed=42),
    ],
    name="augment",
)


def _augment(img, label):
    img = augment_layers(img, training=True)
    img = _random_shear(img, shear_range=0.2)
    return img, label


n_train = len(train)
n_valid = len(valid)
print("CSV split sizes:", n_train, n_valid)

train_ids = tf.constant(train["image_id"].values.astype("S"), dtype=tf.string)
valid_ids = tf.constant(valid["image_id"].values.astype("S"), dtype=tf.string)
train_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        train_ids, tf.ones_like(train_ids, dtype=tf.int64)
    ),
    default_value=0,
)
valid_id_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        valid_ids, tf.ones_like(valid_ids, dtype=tf.int64)
    ),
    default_value=0,
)


def _is_in_train(img, label, image_id):
    return tf.equal(train_id_table.lookup(image_id), 1)


def _is_in_valid(img, label, image_id):
    return tf.equal(valid_id_table.lookup(image_id), 1)


train_raw = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(_ds_options)
train_parsed = train_raw.map(
    _parse_train_with_id, num_parallel_calls=AUTOTUNE
).with_options(_ds_options)

SHUFFLE_BUF = 8192

train_ds = (
    train_parsed.filter(_is_in_train)
    .map(lambda img, label, _id: (img, label), num_parallel_calls=AUTOTUNE)
    .shuffle(SHUFFLE_BUF, seed=42, reshuffle_each_iteration=False)
    .map(_augment, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

valid_ds = (
    train_parsed.filter(_is_in_valid)
    .map(lambda img, label, _id: (img, label), num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .cache()
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

train_steps = (n_train + BATCH_SIZE - 1) // BATCH_SIZE
valid_steps = (n_valid + BATCH_SIZE - 1) // BATCH_SIZE




## === cell 2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback
from tensorflow.keras.layers import Input


class EarlyStoppingCallback(Callback):
    def on_epoch_end(self, epoch, logs=None):
        if self.model.stop_training:
            print(f"Early stopping triggered at epoch {epoch + 1}.")


early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)




## === cell 3
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)
x = base_model.output
x = GlobalAveragePooling2D()(x)
outputs = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.summary()




## === cell 4
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    steps_per_epoch=train_steps,
    validation_steps=valid_steps,
    epochs=10,
    callbacks=[early_stopping, learning_rate_reduction, EarlyStoppingCallback()],
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_12/849389210.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_ds,
      3     validation_data=valid_ds,
      4     steps_per_epoch=train_steps,
      5     validation_steps=valid_steps,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::Map::Shuffle::Map::Filter::Map: Feature: id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_49808]

## === cell 5
import pandas as pd
import numpy as np
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

test_raw = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(_ds_options)
test_parsed = test_raw.map(_parse_test, num_parallel_calls=AUTOTUNE).with_options(
    _ds_options
)

test_ds = (
    test_parsed.batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
    .with_options(_ds_options)
)

pred_probs = model.predict(
    test_ds.map(lambda img, image_id: img, num_parallel_calls=AUTOTUNE),
    verbose=1,
)

pred_idx = np.argmax(pred_probs, axis=1).astype(int)
pred_labels = np.vectorize(idx_to_label.get)(pred_idx).astype(int)

test_image_ids = []
for batch_ids in test_ds.map(
    lambda img, image_id: image_id, num_parallel_calls=AUTOTUNE
):
    test_image_ids.extend(batch_ids.numpy().astype("U"))

test_image_ids = np.asarray(test_image_ids, dtype=object)

pred_map = dict(zip(test_image_ids.tolist(), pred_labels.tolist()))
submission_df = sample_sub.copy()

submission_df["label"] = submission_df["image_id"].map(pred_map).fillna(0).astype(int)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)

print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print("Missing mapped ids:", int(submission_df["label"].isna().sum()))
print("Label distribution:\n", submission_df["label"].value_counts().sort_index())

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_12/3408464916.py in <cell line: 0>()
     23 # Speed fix: get predictions and ids in a single pass without a second full dataset iteration.
     24 # This preserves correctness because it uses the same model outputs and the same ids coming from TFRecords.
---> 25 pred_probs = model.predict(
     26     test_ds.map(lambda img, image_id: img, num_parallel_calls=AUTOTUNE),
     27     verbose=1,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:25 transformation with iterator: Iterator::Root::ParallelMapV2::Prefetch::BatchV2::ParallelMapV2: Feature: id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name:
