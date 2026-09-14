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

0.703838017527954

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.15807) has done: 'I fix the initial TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the dependency on a missing external `.h5` file by instantiating the same kind of Keras classifier locally (EfficientNet backbone + softmax head) so the notebook runs end-to-end. Finally, I make the test generator robust (correct file paths, ensure prediction length matches test rows) and always write a valid `submission.csv` with the required `image_id,label` columns.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_TRAIN_CSV = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "../input/cassava-leaf-disease-classification/train.csv",
    "../data/cassava-leaf-disease-classification/train.csv",
]
CANDIDATE_TRAIN_IMG_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "../data/cassava-leaf-disease-classification/train_images",
]
CANDIDATE_TRAIN_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
]

train_csv_path = None
for p in CANDIDATE_TRAIN_CSV:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not find train.csv. Tried:\n" + "\n".join(CANDIDATE_TRAIN_CSV)
    )

train_img_dir = None
for d in CANDIDATE_TRAIN_IMG_DIRS:
    if os.path.isdir(d):
        train_img_dir = d
        break
if train_img_dir is None:
    raise FileNotFoundError(
        "Could not find train_images dir. Tried:\n"
        + "\n".join(CANDIDATE_TRAIN_IMG_DIRS)
    )

train_tfrec_files = []
for pat in CANDIDATE_TRAIN_TFREC_GLOBS:
    train_tfrec_files = sorted(glob.glob(pat))
    if len(train_tfrec_files) > 0:
        break

train_df = pd.read_csv(train_csv_path)
if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)

print("Train rows:", len(train_df), "train_img_dir:", train_img_dir)
print("Found train tfrecords:", len(train_tfrec_files))



## === cell 2
NUM_CLASSES = 5
IMG_SIZE = (300, 300)

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = keras.layers.Dropout(0.2)(base.output)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

my_model.summary()



## === cell 3
val_frac = 0.1
train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:n_val].copy()
trn_df = train_df_shuf.iloc[n_val:].copy()

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

batch_size = 32 if not DEBUG else 16
EPOCHS = 2 if DEBUG else 3

steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))

AUTOTUNE = tf.data.AUTOTUNE

tf_data_opts = tf.data.Options()
tf_data_opts.deterministic = True

_FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64),
}


def _decode_and_resize(image_bytes):
    img = tf.io.decode_jpeg(image_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _augment(img, seed):
    seed1 = tf.stack([seed, 0])
    seed2 = tf.stack([seed, 1])
    seed3 = tf.stack([seed, 2])
    seed4 = tf.stack([seed, 3])

    img = tf.image.stateless_random_flip_left_right(img, seed=seed1)

    angle = tf.random.stateless_uniform([], seed=seed2, minval=-15.0, maxval=15.0) * (
        np.pi / 180.0
    )
    img = tfa_image_rotate(img, angle)

    max_dx = 0.05 * float(IMG_SIZE[1])
    max_dy = 0.05 * float(IMG_SIZE[0])
    dx = tf.random.stateless_uniform([], seed=seed3, minval=-max_dx, maxval=max_dx)
    dy = tf.random.stateless_uniform([], seed=seed4, minval=-max_dy, maxval=max_dy)
    img = tfa_image_translate(img, [dx, dy])

    zoom = tf.random.stateless_uniform(
        [], seed=tf.stack([seed, 4]), minval=0.9, maxval=1.1
    )
    img = tf.cond(
        zoom < 1.0,
        lambda: _zoom_in(img, zoom),
        lambda: _zoom_out(img, zoom),
    )
    return img


def _projective_transform(img, transform):
    img4 = tf.expand_dims(img, 0)
    transform = tf.expand_dims(transform, 0)
    out = tf.raw_ops.ImageProjectiveTransformV3(
        images=img4,
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    return tf.squeeze(out, 0)


def tfa_image_translate(img, translations):
    dx, dy = translations[0], translations[1]
    transform = tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0])
    return _projective_transform(img, transform)


def tfa_image_rotate(img, angle):
    h = float(IMG_SIZE[0])
    w = float(IMG_SIZE[1])
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    a0 = cos_a
    a1 = -sin_a
    a3 = sin_a
    a4 = cos_a
    a2 = cx - a0 * cx - a1 * cy
    a5 = cy - a3 * cx - a4 * cy
    transform = tf.stack([a0, a1, a2, a3, a4, a5, 0.0, 0.0])
    return _projective_transform(img, transform)


def _zoom_in(img, zoom):
    crop_frac = zoom  # keep zoom fraction of area per dimension
    new_h = tf.cast(tf.round(crop_frac * IMG_SIZE[0]), tf.int32)
    new_w = tf.cast(tf.round(crop_frac * IMG_SIZE[1]), tf.int32)
    cropped = tf.image.random_crop(img, size=[new_h, new_w, 3], seed=SEED)
    return tf.image.resize(cropped, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)


def _zoom_out(img, zoom):
    pad_frac = 1.0 / zoom
    new_h = tf.cast(tf.round(pad_frac * IMG_SIZE[0]), tf.int32)
    new_w = tf.cast(tf.round(pad_frac * IMG_SIZE[1]), tf.int32)
    resized = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR
    )
    padded = tf.image.resize_with_pad(resized, IMG_SIZE[0], IMG_SIZE[1])
    return padded


def _parse_train_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION)
    img = _decode_and_resize(ex["image"])
    label = tf.cast(
        ex["label"], tf.float32
    )  # keep dtype compatible with original generator
    return img, label, ex["image_id"]


trn_ids = tf.constant(trn_df["image_id"].values)
val_ids = tf.constant(val_df["image_id"].values)
trn_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(trn_ids, tf.ones_like(trn_ids, dtype=tf.int32)),
    default_value=0,
)
val_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(val_ids, tf.ones_like(val_ids, dtype=tf.int32)),
    default_value=0,
)


def _is_in_trn(img, label, image_id):
    return tf.equal(trn_table.lookup(image_id), 1)


def _is_in_val(img, label, image_id):
    return tf.equal(val_table.lookup(image_id), 1)


def _strip_id(img, label, image_id):
    return img, label


def _make_train_ds_from_tfrecs(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(tf_data_opts)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(_is_in_trn)
    ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)

    def _aug_map(img, label, image_id):
        sid = tf.strings.to_hash_bucket_fast(image_id, 2**31 - 1)
        img = _augment(img, tf.cast(sid, tf.int32))
        img = preprocess_fn(img)
        return img, label

    ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds_from_tfrecs(files):
    ds = tf.data.TFRecordDataset(files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(tf_data_opts)
    ds = ds.map(_parse_train_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.filter(_is_in_val)

    def _val_map(img, label, image_id):
        img = preprocess_fn(img)
        return img, label

    ds = ds.map(_val_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _as_prefetch_dataset(gen):
    output_signature = (
        tf.TensorSpec(shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
    )
    ds = tf.data.Dataset.from_generator(lambda: gen, output_signature=output_signature)
    return ds.prefetch(tf.data.AUTOTUNE)


if len(train_tfrec_files) > 0:
    train_ds = _make_train_ds_from_tfrecs(train_tfrec_files)
    val_ds = _make_val_ds_from_tfrecs(train_tfrec_files)
else:
    train_idg = ImageDataGenerator(
        preprocessing_function=preprocess_fn,
        rotation_range=15,
        width_shift_range=0.05,
        height_shift_range=0.05,
        zoom_range=0.1,
        horizontal_flip=True,
    )
    val_idg = ImageDataGenerator(preprocessing_function=preprocess_fn)

    train_gen = train_idg.flow_from_dataframe(
        dataframe=trn_df,
        directory=train_img_dir,
        x_col="image_id",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=True,
        seed=SEED,
    )
    val_gen = val_idg.flow_from_dataframe(
        dataframe=val_df,
        directory=train_img_dir,
        x_col="image_id",
        y_col="label",
        target_size=IMG_SIZE,
        batch_size=batch_size,
        class_mode="raw",
        shuffle=False,
        seed=SEED,
    )
    train_ds = _as_prefetch_dataset(train_gen)
    val_ds = _as_prefetch_dataset(val_gen)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/382435471.py in <cell line: 0>()
    265     val_ds = _as_prefetch_dataset(val_gen)
    266 
--> 267 history = my_model.fit(
    268     train_ds,
    269     validation_data=val_ds,

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
Error in user-defined function passed to ParallelMapDatasetV2:3 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2::Shuffle::Filter::ParallelMapV2: Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_79433]

## === cell 4
CANDIDATE_TEST_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
]
test_tfrec_files = []
for pat in CANDIDATE_TEST_TFREC_GLOBS:
    test_tfrec_files = sorted(glob.glob(pat))
    if len(test_tfrec_files) > 0:
        break

_FEATURE_DESCRIPTION_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_id": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION_TEST)
    img = _decode_and_resize(ex["image"])
    img = preprocess_fn(img)
    return img, ex["image_id"]


if len(test_tfrec_files) > 0:
    test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.with_options(tf_data_opts)
    test_ds = test_ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.batch(64).prefetch(AUTOTUNE)

    test_ids = []
    for _, bid in test_ds.unbatch().map(lambda x, y: y):
        test_ids.append(bid.numpy().decode("utf-8"))
    df_test = pd.DataFrame({"image_id": test_ids})
else:
    CANDIDATE_TEST_GLOBS = [
        "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
        "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
        "../input/cassava-leaf-disease-classification/test_images/*.jpg",
        "../data/cassava-leaf-disease-classification/test_images/*.jpg",
    ]
    test_images = []
    for pat in CANDIDATE_TEST_GLOBS:
        test_images = glob.glob(pat)
        if len(test_images) > 0:
            break

    if len(test_images) == 0:
        raise FileNotFoundError(
            "Could not find test images. Tried:\n" + "\n".join(CANDIDATE_TEST_GLOBS)
        )

    df_test = pd.DataFrame({"path": test_images})
    df_test["image_id"] = df_test["path"].astype(str).str.split("/").str[-1]

    def make_test_gen(batch_size=64):
        my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_fn)
        test_gen = my_test_idg.flow_from_dataframe(
            dataframe=df_test,
            x_col="path",
            y_col=None,
            batch_size=batch_size,
            seed=SEED,
            shuffle=False,
            class_mode=None,
            target_size=IMG_SIZE,
        )
        return test_gen

    def make_test_ds(batch_size=64):
        gen = make_test_gen(batch_size=batch_size)
        output_signature = tf.TensorSpec(
            shape=(None, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32
        )
        ds = tf.data.Dataset.from_generator(
            lambda: gen, output_signature=output_signature
        )
        return gen, ds.prefetch(tf.data.AUTOTUNE)

    test_gen, test_ds = make_test_ds(batch_size=64)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/1974987704.py in <cell line: 0>()
     36     # Collect image_ids once (cheap) to build submission rows without globbing JPEG paths.
     37     test_ids = []
---> 38     for _, bid in test_ds.unbatch().map(lambda x, y: y):
     39         test_ids.append(bid.numpy().decode("utf-8"))
     40     df_test = pd.DataFrame({"image_id": test_ids})

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 5
if isinstance(test_ds.element_spec, tuple) and len(test_ds.element_spec) == 2:
    pred_test = my_model.predict(
        test_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE), verbose=1
    )
else:
    steps = int(np.ceil(len(df_test) / 64))
    pred_test = my_model.predict(test_ds, steps=steps, verbose=1)

pred_test = pred_test[: len(df_test)]
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_csv = df_test[["image_id"]].copy()
final_csv["label"] = pred_test_labels

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/211101954.py in <cell line: 0>()
      2 if isinstance(test_ds.element_spec, tuple) and len(test_ds.element_spec) == 2:
      3     # test_ds yields (images, image_id) batches -> predict on images only
----> 4     pred_test = my_model.predict(
      5         test_ds.map(lambda x, y: x, num_parallel_calls=AUTOTUNE), verbose=1
      6     )

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Feature: image_id (data type: string) is required but could not be found.
	 [[{{node ParseSingleExample/ParseExample/ParseExampleV2}}]] [Op:IteratorGetNext] name: 

## === cell 6
CANDIDATE_SAMPLE = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "../data/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in CANDIDATE_SAMPLE:
    if os.path.exists(p):
        sample_path = p
        break

if sample_path is not None:
    sample = pd.read_csv(sample_path)
    print("sample_submission rows:", len(sample), "submission rows:", len(final_csv))

    aligned = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    aligned["label"] = aligned["label"].fillna(0).astype(int)
    aligned.to_csv("submission.csv", index=False)
    final_csv = aligned
    print("Aligned submission to sample_submission order and rewrote submission.csv")

print(final_csv.head())
print("Final submission rows:", len(final_csv))
print("Saved: submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3884894868.py in <cell line: 0>()
     13 if sample_path is not None:
     14     sample = pd.read_csv(sample_path)
---> 15     print("sample_submission rows:", len(sample), "submission rows:", len(final_csv))
     16 
     17     aligned = sample[["image_id"]].merge(final_csv, on="image_id", how="left")

NameError: name 'final_csv' is not defined
