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

0.8756421879721971

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

import numpy as np
import pandas as pd



## === cell 1
import json
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"

with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))



## === cell 3
label_list = [int(key) for key in map_classes.keys()]
label_list



## === cell 4
input_files = os.listdir(os.path.join(BASE_DIR, "train_images"))
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 300
IMG_WIDTH = 300
batch_size = 16
PRE_TRAINED_MODEL = "../input/unionmodelv03/Cassava_Best_UnitedModel_V03.hdf5"



## === cell 6
pass



## === cell 7
pass



## === cell 8
pass



## === cell 9
test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
test_samples = test_df.shape[0]
test_samples



## === cell 10
AUTOTUNE = tf.data.AUTOTUNE

_FEATURES_LABELED_FLEX = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
    "image_id": tf.io.VarLenFeature(tf.string),  # optional
    "id": tf.io.VarLenFeature(tf.string),  # optional
}
_FEATURES_TEST_FLEX = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.VarLenFeature(tf.string),  # optional
    "id": tf.io.VarLenFeature(tf.string),  # optional
}


def _decode_and_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [IMG_HEIGHT, IMG_WIDTH],
        method=tf.image.ResizeMethod.LANCZOS3,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _get_string_feature_flex(parsed_ex, key):
    """Return dense scalar string for a VarLenFeature, or empty string if missing."""
    v = parsed_ex.get(key, None)
    if v is None:
        return tf.constant("", tf.string)
    dense = tf.sparse.to_dense(v, default_value="")
    dense = tf.reshape(dense, [-1])
    return tf.cond(
        tf.size(dense) > 0, lambda: dense[0], lambda: tf.constant("", tf.string)
    )


def _get_image_id_from_parsed(parsed_ex):
    image_id = _get_string_feature_flex(parsed_ex, "image_id")
    _id = _get_string_feature_flex(parsed_ex, "id")
    return tf.cond(tf.strings.length(image_id) > 0, lambda: image_id, lambda: _id)


def tfa_image_rotate(image, angle):
    h = tf.cast(tf.shape(image)[0], tf.float32)
    w = tf.cast(tf.shape(image)[1], tf.float32)
    cy = (h - 1.0) / 2.0
    cx = (w - 1.0) / 2.0
    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)
    a0 = cos_a
    a1 = -sin_a
    a2 = cx - cos_a * cx + sin_a * cy
    b0 = sin_a
    b1 = cos_a
    b2 = cy - sin_a * cx - cos_a * cy
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])
    transform = tf.expand_dims(transform, 0)
    img4 = tf.expand_dims(image, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.shape(image)[:2],
        interpolation="BILINEAR",
        fill_mode="CONSTANT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


@tf.function
def _train_map_from_example(example):
    ex = tf.io.parse_single_example(example, _FEATURES_LABELED_FLEX)
    image_id = _get_image_id_from_parsed(ex)

    img = _decode_and_resize_from_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)

    hash_id = tf.cast(tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1), tf.int32)
    key = tf.stack([tf.constant(SEED, tf.int32), hash_id])
    rnd = tf.random.stateless_uniform([], seed=key, dtype=tf.float32)
    do_aug = rnd > 0.5

    def _aug_fn(x):
        x = tf.image.stateless_random_flip_left_right(
            x, seed=key + tf.constant([1, 0], tf.int32)
        )
        x = tf.image.stateless_random_flip_up_down(
            x, seed=key + tf.constant([2, 0], tf.int32)
        )
        x = tf.image.stateless_random_brightness(
            x, max_delta=0.2, seed=key + tf.constant([3, 0], tf.int32)
        )
        x = tf.image.stateless_random_contrast(
            x, lower=0.8, upper=1.2, seed=key + tf.constant([4, 0], tf.int32)
        )
        angle = tf.random.stateless_uniform(
            [], seed=key + tf.constant([5, 0], tf.int32), minval=-15.0, maxval=15.0
        ) * (np.pi / 180.0)
        x = tfa_image_rotate(x, angle)
        return tf.clip_by_value(x, 0.0, 1.0)

    img = tf.cond(do_aug, lambda: _aug_fn(img), lambda: img)
    return img, label


@tf.function
def _val_map_from_example(example):
    ex = tf.io.parse_single_example(example, _FEATURES_LABELED_FLEX)
    img = _decode_and_resize_from_bytes(ex["image"])
    label = tf.cast(ex["label"], tf.int32)
    return img, label


@tf.function
def _test_map_from_example(example):
    ex = tf.io.parse_single_example(example, _FEATURES_TEST_FLEX)
    image_id = _get_image_id_from_parsed(ex)
    img = _decode_and_resize_from_bytes(ex["image"])
    return image_id, img


TRAIN_TFREC_DIR = os.path.join(BASE_DIR, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_DIR, "test_tfrecords")

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

test_ds = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.map(
    _test_map_from_example, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.cache()
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)



## === cell 11
train_csv_path = os.path.join(BASE_DIR, "train.csv")
train_df = pd.read_csv(train_csv_path)

train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
val_frac = 0.1
val_size = int(len(train_df) * val_frac)
val_df = train_df.iloc[:val_size].reset_index(drop=True)
trn_df = train_df.iloc[val_size:].reset_index(drop=True)

num_classes = 5

val_ids = tf.constant(val_df["image_id"].values, dtype=tf.string)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids, tf.ones_like(val_ids, dtype=tf.int32)),
    default_value=0,
)


@tf.function
def _is_val_example(example):
    ex = tf.io.parse_single_example(example, _FEATURES_LABELED_FLEX)
    image_id = _get_image_id_from_parsed(ex)
    return tf.equal(val_table.lookup(image_id), 1)


@tf.function
def _is_train_example(example):
    ex = tf.io.parse_single_example(example, _FEATURES_LABELED_FLEX)
    image_id = _get_image_id_from_parsed(ex)
    return tf.not_equal(val_table.lookup(image_id), 1)


all_train_ds = tf.data.TFRecordDataset(train_tfrecs, num_parallel_reads=AUTOTUNE)

SHUFFLE_BUFFER = 8192

train_ds = (
    all_train_ds.filter(_is_train_example)
    .shuffle(buffer_size=SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True)
    .map(_train_map_from_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    all_train_ds.filter(_is_val_example)
    .map(_val_map_from_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 12
from tensorflow.keras import layers, models

base = tf.keras.applications.EfficientNetB0(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base.trainable = False  # keep fast; avoids large training cost

inputs = tf.keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
outputs = layers.Dense(num_classes, activation="softmax")(x)
model = models.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

model.summary()



## === cell 13
EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)

test_image_ids = []
test_pred_probs = []

for batch_ids, batch_imgs in test_ds:
    probs = model.predict_on_batch(batch_imgs)
    test_image_ids.append(batch_ids.numpy())
    test_pred_probs.append(probs)

test_image_ids = np.concatenate(test_image_ids, axis=0).astype("U")
pred_probs = np.concatenate(test_pred_probs, axis=0)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

sample_sub_path = os.path.join(BASE_DIR, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

pred_map = dict(zip(test_image_ids, pred_labels))
sample_sub["label"] = sample_sub["image_id"].map(pred_map)

if sample_sub["label"].isna().any():
    fallback = int(train_df["label"].mode().iloc[0])
    sample_sub["label"] = sample_sub["label"].fillna(fallback)

sample_sub["label"] = sample_sub["label"].astype(int)
sample_sub.to_csv("submission.csv", index=False)
print(sample_sub.head(3))
print("Wrote submission.csv with shape:", sample_sub.shape)
print(
    "Missing predictions filled:",
    int(sample_sub["image_id"].map(pred_map).isna().sum()),
)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3858897076.py in <cell line: 0>()
      1 EPOCHS = 3
      2 
----> 3 history = model.fit(
      4     train_ds,
      5     validation_data=val_ds,

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
Error in user-defined function passed to FilterDataset:8 transformation with iterator: Iterator::Root::Prefetch::MapAndBatch::Shuffle::Filter: Feature: label (data type: int64) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_16866]
