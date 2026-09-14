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

0.7889090359625265

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")

import glob
import random
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.optimizers import Adam

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass

BASE = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing {TEST_TFREC_DIR}"

print("TensorFlow:", tf.__version__)
print("Train CSV:", TRAIN_CSV)
print("Train images:", TRAIN_IMG_DIR)
print("Test images:", TEST_IMG_DIR)
print("Train TFRecords:", TRAIN_TFREC_DIR)
print("Test TFRecords:", TEST_TFREC_DIR)

AUTOTUNE = tf.data.AUTOTUNE

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)

if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

num_classes = train_df["label"].nunique()
assert num_classes == 5, f"Expected 5 classes, got {num_classes}"

IMG_SIZE = 512
BATCH_SIZE = 16 if not DEBUG else 8
EPOCHS = 5 if not DEBUG else 2


def _augment_stateless(image, seed_pair):
    image = tf.image.stateless_random_flip_left_right(image, seed=seed_pair)

    angle = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([1, 0], tf.int32),
        minval=-15.0,
        maxval=15.0,
        dtype=tf.float32,
    ) * (np.pi / 180.0)
    try:
        image = tf.image.rotate(image, angles=angle, interpolation="BILINEAR")
    except Exception:
        pass

    dx = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([2, 0], tf.int32),
        minval=-0.08,
        maxval=0.08,
        dtype=tf.float32,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([3, 0], tf.int32),
        minval=-0.08,
        maxval=0.08,
        dtype=tf.float32,
    )
    shift_x = tf.cast(dx * tf.cast(tf.shape(image)[1], tf.float32), tf.int32)
    shift_y = tf.cast(dy * tf.cast(tf.shape(image)[0], tf.float32), tf.int32)
    image = tf.roll(image, shift=[shift_y, shift_x], axis=[0, 1])

    scale = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([4, 0], tf.int32),
        minval=0.90,
        maxval=1.10,
        dtype=tf.float32,
    )
    new_size = tf.cast(scale * tf.cast(IMG_SIZE, tf.float32), tf.int32)
    new_size = tf.maximum(1, new_size)
    image_rs = tf.image.resize(image, [new_size, new_size], method="bilinear")
    image_rs = tf.image.resize_with_crop_or_pad(image_rs, IMG_SIZE, IMG_SIZE)

    shear = tf.random.stateless_uniform(
        [],
        seed=seed_pair + tf.constant([5, 0], tf.int32),
        minval=-0.08,
        maxval=0.08,
        dtype=tf.float32,
    )
    try:
        transform = tf.stack([1.0, shear, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0], axis=0)
        transform = tf.reshape(transform, [1, 8])
        image4 = tf.expand_dims(image_rs, 0)
        image4 = tf.raw_ops.ImageProjectiveTransformV3(
            images=image4,
            transforms=transform,
            output_shape=tf.constant([IMG_SIZE, IMG_SIZE], tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )
        image = tf.squeeze(image4, 0)
    except Exception:
        image = image_rs

    return image


_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _decode_image(jpeg_bytes):
    img = tf.image.decode_jpeg(jpeg_bytes, channels=3)
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear")
    img = tf.cast(img, tf.float32)  # [0,255]
    return img


def _preprocess(img):
    return preprocess_input(img)


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No train TFRecords found."
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(test_tfrecs) > 0, "No test TFRecords found."

val_frac = 0.10
n_files = len(train_tfrecs)
n_val = max(1, int(round(n_files * val_frac)))

rng = np.random.default_rng(SEED)
shuffled = train_tfrecs.copy()
rng.shuffle(shuffled)
val_files = shuffled[:n_val]
trn_files = shuffled[n_val:]

if DEBUG:
    trn_files = trn_files[:2]
    val_files = val_files[:1]


def _parse_train_bytes(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TRAIN)
    jpeg = ex["image"]
    label = tf.cast(ex["target"], tf.int32)
    return jpeg, label


def _parse_test_bytes(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_TEST)
    jpeg = ex["image"]
    image_name = ex["image_name"]
    return jpeg, image_name


@tf.function
def _decode_only_train(jpeg, label):
    img = _decode_image(jpeg)
    return img, label


@tf.function
def _decode_only_test(jpeg, name):
    img = _decode_image(jpeg)
    return img, name


@tf.function
def _train_aug_preproc_from_img(index, img, label):
    seed_pair = tf.stack([tf.cast(SEED, tf.int32), tf.cast(index, tf.int32)], axis=0)
    img = _augment_stateless(img, seed_pair)
    img = _preprocess(img)
    label_oh = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, label_oh


@tf.function
def _valid_preproc_from_img(img, label):
    img = _preprocess(img)
    label_oh = tf.one_hot(label, depth=num_classes, dtype=tf.float32)
    return img, label_oh


@tf.function
def _test_preproc_from_img(img, name):
    img = _preprocess(img)
    return img, name


def _estimate_split_counts_from_csv(n_total, n_train_files, n_val_files):
    n_files_total = n_train_files + n_val_files
    n_train = int(round(n_total * (n_train_files / n_files_total)))
    n_val = n_total - n_train
    n_train = max(1, n_train)
    n_val = max(1, n_val)
    return n_train, n_val




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True
try:
    _DATA_OPTIONS.autotune.enabled = True
except Exception:
    pass
try:
    _DATA_OPTIONS.experimental_optimization.map_parallelization = True
    _DATA_OPTIONS.experimental_optimization.parallel_batch = True
    _DATA_OPTIONS.experimental_optimization.map_fusion = True
except Exception:
    pass


def _get_compression(files):
    return None


def make_train_ds(files, training=True):
    comp = _get_compression(files)
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        compression_type=comp,
        deterministic=True,
    )
    ds = ds.with_options(_DATA_OPTIONS)

    if training:
        ds = ds.map(_parse_train_bytes, num_parallel_calls=AUTOTUNE)
        ds = ds.map(_decode_only_train, num_parallel_calls=AUTOTUNE)
        ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.repeat()
        ds = ds.enumerate()
        ds = ds.map(
            lambda i, il: _train_aug_preproc_from_img(i, il[0], il[1]),
            num_parallel_calls=AUTOTUNE,
        )
    else:
        ds = ds.map(_parse_train_bytes, num_parallel_calls=AUTOTUNE)
        ds = ds.map(_decode_only_train, num_parallel_calls=AUTOTUNE)
        ds = ds.map(_valid_preproc_from_img, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(trn_files, training=True)
valid_ds = make_train_ds(val_files, training=False)

n_total = len(train_df)
n_train, n_val = _estimate_split_counts_from_csv(
    n_total, len(trn_files), len(val_files)
)
steps_per_epoch = int(np.ceil(n_train / BATCH_SIZE))
validation_steps = int(np.ceil(n_val / BATCH_SIZE))

if DEBUG:
    steps_per_epoch = min(steps_per_epoch, 50)
    validation_steps = min(validation_steps, 20)

inp = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inp, outputs=out)

my_model.compile(
    optimizer=Adam(learning_rate=2e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

my_model.summary()

history = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/249745432.py in <cell line: 0>()
     53 
     54 
---> 55 train_ds = make_train_ds(trn_files, training=True)
     56 valid_ds = make_train_ds(val_files, training=False)
     57 

/tmp/ipykernel_11/249745432.py in make_train_ds(files, training)
     23 def make_train_ds(files, training=True):
     24     comp = _get_compression(files)
---> 25     ds = tf.data.TFRecordDataset(
     26         files,
     27         num_parallel_reads=AUTOTUNE,

TypeError: TFRecordDatasetV2.__init__() got an unexpected keyword argument 'deterministic'

## === cell 2
def make_test_ds(files, batch_size=64):
    comp = _get_compression(files)
    ds = tf.data.TFRecordDataset(
        files,
        num_parallel_reads=AUTOTUNE,
        compression_type=comp,
        deterministic=True,
    )
    ds = ds.with_options(_DATA_OPTIONS)

    ds = ds.map(_parse_test_bytes, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_decode_only_test, num_parallel_calls=AUTOTUNE)
    ds = ds.map(_test_preproc_from_img, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_tfrecs, batch_size=64)

test_names_list = []
for _, names in test_ds:
    test_names_list.append(names.numpy())
test_names = np.concatenate(test_names_list).astype("U")
print("Extracted test_names:", test_names.shape)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2687422388.py in <cell line: 0>()
     20 
     21 
---> 22 test_ds = make_test_ds(test_tfrecs, batch_size=64)
     23 
     24 # Collect names in one pass (same order as predictions) deterministically.

/tmp/ipykernel_11/2687422388.py in make_test_ds(files, batch_size)
      3 def make_test_ds(files, batch_size=64):
      4     comp = _get_compression(files)
----> 5     ds = tf.data.TFRecordDataset(
      6         files,
      7         num_parallel_reads=AUTOTUNE,

TypeError: TFRecordDatasetV2.__init__() got an unexpected keyword argument 'deterministic'

## === cell 3
pred_proba = my_model.predict(test_ds, verbose=0)
pred_test_labels = np.argmax(pred_proba, axis=-1).astype(int)

assert (
    test_names.shape[0] == pred_test_labels.shape[0]
), f"Name/pred count mismatch: {test_names.shape[0]} vs {pred_test_labels.shape[0]}"

final_csv = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})

sample_sub = pd.read_csv(SAMPLE_SUB)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")
assert final_csv["label"].isna().sum() == 0, "Some test images missing predictions."

final_csv["label"] = final_csv["label"].astype(int)
final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3606673447.py in <cell line: 0>()
----> 1 pred_proba = my_model.predict(test_ds, verbose=0)
      2 pred_test_labels = np.argmax(pred_proba, axis=-1).astype(int)
      3 
      4 assert (
      5     test_names.shape[0] == pred_test_labels.shape[0]

NameError: name 'my_model' is not defined

## === cell 4
final_csv.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1842027079.py in <cell line: 0>()
----> 1 final_csv.head()

NameError: name 'final_csv' is not defined
