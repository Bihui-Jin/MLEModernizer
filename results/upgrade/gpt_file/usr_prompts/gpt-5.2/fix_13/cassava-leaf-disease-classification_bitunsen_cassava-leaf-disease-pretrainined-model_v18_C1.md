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

0.8584164400120883

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import random
import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

assert os.path.isdir(BASE_DIR), f"BASE_DIR not found: {BASE_DIR}"
assert os.path.isdir(TRAIN_DIR), f"TRAIN_DIR not found: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"TEST_DIR not found: {TEST_DIR}"



## === cell 2
from PIL import Image

import tensorflow as tf
import keras
from keras import layers

keras.backend.clear_session()
np.random.seed(42)
random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
print(json.dumps(map_classes, indent=2))

label_list = sorted([int(k) for k in map_classes.keys()])
NUM_CLASSES = len(label_list)
assert NUM_CLASSES == 5, f"Expected 5 classes, got {NUM_CLASSES}"



## === cell 4
pass



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16

PRE_TRAINED_MODEL = "../input/unionmodelv05/Cassava_Best_UnitedModel_V05.hdf5"
print("Pretrained model exists?:", os.path.exists(PRE_TRAINED_MODEL))

RESAMPLE = getattr(Image, "Resampling", Image).LANCZOS



## === cell 6
train_df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
train_df["label"] = train_df["label"].astype(int)
train_df["filepath"] = (TRAIN_DIR + train_df["image_id"].astype(str)).astype(str)

fps = train_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
train_df = train_df.loc[exists_mask].reset_index(drop=True)

print(
    "Train rows:",
    len(train_df),
    "Unique labels:",
    sorted(train_df["label"].unique().tolist()),
)



## === cell 7
sample_sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_df = sample_sub[["image_id"]].copy()
test_df["filepath"] = (TEST_DIR + test_df["image_id"].astype(str)).astype(str)

fps = test_df["filepath"].to_numpy()
exists_mask = np.fromiter(
    (os.path.exists(p) for p in fps), dtype=bool, count=fps.shape[0]
)
test_df = test_df.loc[exists_mask].reset_index(drop=True)

print("Test rows:", len(test_df))




## === cell 8
@tf.function
def decode_and_resize(path, label=None, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0

    if training:
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        img = tf.image.random_brightness(img, max_delta=0.12)
        img = tf.image.random_contrast(img, lower=0.85, upper=1.15)

    if label is None:
        return img
    return img, tf.cast(label, tf.int32)


def stratified_split_df(df, label_col="label", test_size=0.2, seed=42):
    rng = np.random.RandomState(seed)
    train_idx = []
    valid_idx = []
    for lab, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_valid = int(np.floor(len(idx) * test_size))
        valid_idx.extend(idx[:n_valid].tolist())
        train_idx.extend(idx[n_valid:].tolist())
    return df.loc[train_idx].reset_index(drop=True), df.loc[valid_idx].reset_index(
        drop=True
    )


train_part, valid_part = stratified_split_df(
    train_df, label_col="label", test_size=0.2, seed=42
)

x_tr = train_part["filepath"].values
y_tr = train_part["label"].values
x_va = valid_part["filepath"].values
y_va = valid_part["label"].values

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_distribute.auto_shard_policy = (
        tf.data.experimental.AutoShardPolicy.OFF
    )
except Exception:
    pass


@tf.function
def _decode_and_augment(p, y):
    img, y = decode_and_resize(p, y, training=True)
    return img, y


@tf.function
def _decode_noaug(p, y):
    img, y = decode_and_resize(p, y, training=False)
    return img, y


TFREC_TRAIN_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TFREC_TEST_DIR = os.path.join(BASE_DIR, "test_tfrecords")
assert os.path.isdir(TFREC_TRAIN_DIR), f"train_tfrecords not found: {TFREC_TRAIN_DIR}"
assert os.path.isdir(TFREC_TEST_DIR), f"test_tfrecords not found: {TFREC_TEST_DIR}"

train_tfrecs = sorted(
    [
        os.path.join(TFREC_TRAIN_DIR, f)
        for f in os.listdir(TFREC_TRAIN_DIR)
        if f.endswith(".tfrec")
    ]
)
test_tfrecs = sorted(
    [
        os.path.join(TFREC_TEST_DIR, f)
        for f in os.listdir(TFREC_TEST_DIR)
        if f.endswith(".tfrec")
    ]
)
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"

tr_id_set = tf.constant(train_part["image_id"].astype(str).values)
va_id_set = tf.constant(valid_part["image_id"].astype(str).values)
tr_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        tr_id_set, tf.ones_like(tr_id_set, dtype=tf.int64)
    ),
    default_value=tf.constant(0, dtype=tf.int64),
)
va_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        va_id_set, tf.ones_like(va_id_set, dtype=tf.int64)
    ),
    default_value=tf.constant(0, dtype=tf.int64),
)

FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}
FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_train_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TRAIN)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return x["image_id"], img, tf.cast(x["label"], tf.int32)


@tf.function
def _parse_test_example(ex):
    x = tf.io.parse_single_example(ex, FEATURES_TEST)
    img = tf.image.decode_jpeg(x["image"], channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return x["image_id"], img


@tf.function
def _augment_img(img, y):
    img = tf.image.random_flip_left_right(img)
    img = tf.image.random_flip_up_down(img)
    img = tf.image.random_brightness(img, max_delta=0.12)
    img = tf.image.random_contrast(img, lower=0.85, upper=1.15)
    return img, y


ds_all = tf.data.TFRecordDataset(
    train_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(options)
ds_all = ds_all.map(_parse_train_example, num_parallel_calls=AUTOTUNE)

ds_train = ds_all.filter(lambda image_id, img, y: tr_table.lookup(image_id) > 0)
ds_valid = ds_all.filter(lambda image_id, img, y: va_table.lookup(image_id) > 0)

ds_train = ds_train.map(lambda image_id, img, y: (img, y), num_parallel_calls=AUTOTUNE)
ds_valid = ds_valid.map(lambda image_id, img, y: (img, y), num_parallel_calls=AUTOTUNE)

ds_train = ds_train.cache()
ds_train = ds_train.shuffle(2048, seed=42, reshuffle_each_iteration=True)
ds_train = ds_train.map(_augment_img, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_valid = ds_valid.cache()
ds_valid = ds_valid.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

ds_test_tfr = tf.data.TFRecordDataset(
    test_tfrecs, num_parallel_reads=AUTOTUNE
).with_options(options)
ds_test_tfr = ds_test_tfr.map(_parse_test_example, num_parallel_calls=AUTOTUNE)
ds_test_tfr = ds_test_tfr.cache()

test_order_ids = test_df["image_id"].astype(str).values
test_pos = tf.range(len(test_order_ids), dtype=tf.int32)
test_order_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(tf.constant(test_order_ids), test_pos),
    default_value=tf.constant(-1, dtype=tf.int32),
)



## === cell 9
inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = layers.MaxPooling2D()(x)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.25)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
EPOCHS = 8
history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1044257117.py in <cell line: 0>()
      1 EPOCHS = 8
----> 2 history = model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, verbose=2)
      3 

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::MemoryCacheImpl::ParallelMapV2::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_1988]

## === cell 11
GLOBAL_SEED = 42
N_AUG = 5
N_VIEWS = 1 + N_AUG


@tf.function
def _tta_views_tf(img, idx):
    img = tf.convert_to_tensor(img, tf.float32)  # [H,W,3]
    views0 = tf.expand_dims(img, axis=0)  # [1,H,W,3]

    js = tf.range(N_AUG, dtype=tf.int32)  # [5]
    idx = tf.cast(idx, tf.int32)
    seed_second = idx * 1000 + js  # [5]
    seed_base = tf.stack(
        [tf.fill([N_AUG], tf.cast(GLOBAL_SEED, tf.int32)), seed_second], axis=1
    )  # [5,2]

    out = tf.broadcast_to(img, [N_AUG, IMG_HEIGHT, IMG_WIDTH, 3])  # [5,H,W,3]

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([1, 11], tf.int32)
    )
    do = r < 0.5
    out = tf.where(do[:, None, None, None], tf.reverse(out, axis=[2]), out)

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([2, 22], tf.int32)
    )
    do = r < 0.2
    out = tf.where(do[:, None, None, None], tf.reverse(out, axis=[1]), out)

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([3, 33], tf.int32)
    )
    do = r < 0.5
    delta = tf.random.stateless_uniform(
        [N_AUG],
        seed=seed_base + tf.constant([4, 44], tf.int32),
        minval=-0.12,
        maxval=0.12,
    )
    out = tf.where(
        do[:, None, None, None],
        tf.clip_by_value(out + delta[:, None, None, None], 0.0, 1.0),
        out,
    )

    r = tf.random.stateless_uniform(
        [N_AUG], seed=seed_base + tf.constant([5, 55], tf.int32)
    )
    do = r < 0.5
    c = tf.random.stateless_uniform(
        [N_AUG],
        seed=seed_base + tf.constant([6, 66], tf.int32),
        minval=0.85,
        maxval=1.15,
    )
    mean = tf.reduce_mean(out, axis=[1, 2], keepdims=True)  # [5,1,1,3]
    out_contrast = tf.clip_by_value(
        (out - mean) * c[:, None, None, None] + mean, 0.0, 1.0
    )
    out = tf.where(do[:, None, None, None], out_contrast, out)

    return tf.concat([views0, out], axis=0)  # [6,H,W,3]


@tf.function
def _tta_views_batch(imgs, idxs):
    imgs = tf.convert_to_tensor(imgs, tf.float32)
    idxs = tf.cast(idxs, tf.int32)
    b = tf.shape(imgs)[0]

    views0 = imgs[:, None, :, :, :]  # [B,1,H,W,3]

    js = tf.range(N_AUG, dtype=tf.int32)[None, :]  # [1,5]
    idxs2 = idxs[:, None]  # [B,1]
    seed_second = idxs2 * 1000 + js  # [B,5]

    seed_base = tf.stack(
        [
            tf.fill([b, N_AUG], tf.cast(GLOBAL_SEED, tf.int32)),
            seed_second,
        ],
        axis=2,
    )  # [B,5,2]

    out = tf.broadcast_to(imgs[:, None, :, :, :], [b, N_AUG, IMG_HEIGHT, IMG_WIDTH, 3])

    r = tf.random.stateless_uniform(
        [b, N_AUG], seed=seed_base + tf.constant([1, 11], tf.int32)
    )
    do = r < 0.5
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[3]), out)

    r = tf.random.stateless_uniform(
        [b, N_AUG], seed=seed_base + tf.constant([2, 22], tf.int32)
    )
    do = r < 0.2
    out = tf.where(do[:, :, None, None, None], tf.reverse(out, axis=[2]), out)

    r = tf.random.stateless_uniform(
        [b, N_AUG], seed=seed_base + tf.constant([3, 33], tf.int32)
    )
    do = r < 0.5
    delta = tf.random.stateless_uniform(
        [b, N_AUG],
        seed=seed_base + tf.constant([4, 44], tf.int32),
        minval=-0.12,
        maxval=0.12,
    )
    out = tf.where(
        do[:, :, None, None, None],
        tf.clip_by_value(out + delta[:, :, None, None, None], 0.0, 1.0),
        out,
    )

    r = tf.random.stateless_uniform(
        [b, N_AUG], seed=seed_base + tf.constant([5, 55], tf.int32)
    )
    do = r < 0.5
    c = tf.random.stateless_uniform(
        [b, N_AUG],
        seed=seed_base + tf.constant([6, 66], tf.int32),
        minval=0.85,
        maxval=1.15,
    )
    mean = tf.reduce_mean(out, axis=[2, 3], keepdims=True)  # [B,5,1,1,3]
    out_contrast = tf.clip_by_value(
        (out - mean) * c[:, :, None, None, None] + mean, 0.0, 1.0
    )
    out = tf.where(do[:, :, None, None, None], out_contrast, out)

    views = tf.concat([views0, out], axis=1)  # [B,6,H,W,3]
    return tf.reshape(views, [b * N_VIEWS, IMG_HEIGHT, IMG_WIDTH, 3])


def _collect_test_from_tfrecord(ds, n_expected):
    positions = []
    ids = []
    imgs = []
    for image_id, img in ds:
        pos = test_order_table.lookup(image_id).numpy()
        if pos >= 0:
            positions.append(int(pos))
            ids.append(image_id.numpy().decode("utf-8"))
            imgs.append(img.numpy())
    assert len(ids) == n_expected, (len(ids), n_expected)
    order = np.argsort(np.asarray(positions, dtype=np.int32))
    ids = [ids[i] for i in order]
    imgs = np.stack([imgs[i] for i in order], axis=0).astype(np.float32, copy=False)
    return ids, imgs


pred_ids, test_imgs_np = _collect_test_from_tfrecord(ds_test_tfr, len(test_df))

ds_test_imgs = tf.data.Dataset.from_tensor_slices(test_imgs_np).with_options(options)
ds_test_imgs = ds_test_imgs.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
ds_test_imgs = ds_test_imgs.enumerate()  # (batch_idx, img_batch)


@tf.function
def _batch_img_to_views(batch_idx, imgs):
    b = tf.shape(imgs)[0]
    start = tf.cast(batch_idx, tf.int32) * tf.cast(batch_size, tf.int32)
    idxs = start + tf.range(b, dtype=tf.int32)  # global per-image indices
    views_flat = _tta_views_batch(imgs, idxs)  # [B*N_VIEWS,H,W,3]
    return views_flat


ds_views_flat = ds_test_imgs.map(
    lambda bi, imgs: _batch_img_to_views(bi, imgs), num_parallel_calls=AUTOTUNE
)
ds_views_flat = ds_views_flat.prefetch(AUTOTUNE)

preds_flat = model.predict(ds_views_flat, verbose=0)

n_images = len(pred_ids)
assert preds_flat.shape[0] == n_images * N_VIEWS, (preds_flat.shape, n_images, N_VIEWS)

preds = preds_flat.reshape(n_images, N_VIEWS, NUM_CLASSES)
preds_mean = preds.mean(axis=1)
pred_labels = preds_mean.argmax(axis=1).astype(np.int32, copy=False).tolist()

submission = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})
submission = sample_sub[["image_id"]].merge(submission, on="image_id", how="left")
assert submission["label"].notnull().all(), "Some test image_ids were not predicted."
submission["label"] = submission["label"].astype(int)

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(submission.head())
print(submission.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/4085711172.py in <cell line: 0>()
    155 
    156 
--> 157 pred_ids, test_imgs_np = _collect_test_from_tfrecord(ds_test_tfr, len(test_df))
    158 
    159 # Keep the exact same TTA + predict semantics, but feed from in-memory array (fast).

/tmp/ipykernel_11/4085711172.py in _collect_test_from_tfrecord(ds, n_expected)
    142     ids = []
    143     imgs = []
--> 144     for image_id, img in ds:
    145         pos = test_order_table.lookup(image_id).numpy()
    146         if pos >= 0:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:19 transformation with iterator: Iterator::Root::Prefetch::MemoryCacheImpl::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{function_node __inference__parse_test_example_255}}{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 12
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image_id", "label"], chk.columns
assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
chk.head(3)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3560446106.py in <cell line: 0>()
----> 1 chk = pd.read_csv("submission.csv")
      2 assert list(chk.columns) == ["image_id", "label"], chk.columns
      3 assert len(chk) == len(sample_sub), (len(chk), len(sample_sub))
      4 chk.head(3)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'submission.csv'
