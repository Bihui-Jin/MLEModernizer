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

0.8850105772136597

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
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(BASE_DIR, "sample_submission.csv")
LABEL_MAP_JSON = os.path.join(BASE_DIR, "label_num_to_disease_map.json")



## === cell 2
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

print("TensorFlow:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _CPU = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _CPU))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))
_GPU_DEVICE = "/GPU:0" if _HAS_GPU else None
print("GPU available:", _HAS_GPU)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
with open(LABEL_MAP_JSON) as file:
    map_classes = json.loads(file.read())

print(json.dumps(map_classes, indent=2))
label_list = [int(key) for key in map_classes.keys()]
NUM_CLASSES = len(label_list)
print("NUM_CLASSES:", NUM_CLASSES)



## === cell 4
input_files = os.listdir(TRAIN_DIR)
print(f"Number of train images: {len(input_files)}")



## === cell 5
IMG_HEIGHT = 500
IMG_WIDTH = 500
batch_size = 16

PRE_TRAINED_MODEL = None

EPOCHS = 2
LEARNING_RATE = 1e-4



## === cell 6
try:
    from albumentations import (  # noqa: F401
        Compose,
        HorizontalFlip,
        CenterCrop,
        ShiftScaleRotate,
        VerticalFlip,
        ToFloat,
        RandomBrightnessContrast,
    )

    AUGMENTATIONS_TRAIN = Compose(
        [
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=0.5),
            CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ShiftScaleRotate(
                always_apply=False,
                p=0.5,
                shift_limit=0,
                scale_limit=(0.5, 1.50),
                rotate_limit=15,
                interpolation=0,
                border_mode=0,
            ),
            ToFloat(max_value=255.0),
        ]
    )

    AUGMENTATIONS_TEST = Compose(
        [
            CenterCrop(always_apply=False, p=1.0, height=IMG_HEIGHT, width=IMG_WIDTH),
            ToFloat(max_value=255.0),
        ]
    )
except Exception as e:
    AUGMENTATIONS_TRAIN = None
    AUGMENTATIONS_TEST = None
    print("albumentations not available (ok, not used by TF pipeline):", repr(e))



## === cell 7
from sklearn.model_selection import train_test_split

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

train_idx, val_idx = train_test_split(
    np.arange(len(train_df)),
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"].values,
)

tr_df = train_df.iloc[train_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

print("Train size:", len(tr_df), "Val size:", len(va_df))
print(
    "Train label distribution:\n",
    tr_df["label"].value_counts(normalize=True).sort_index(),
)

test_filenames = sorted(os.listdir(TEST_DIR))
test_df = pd.DataFrame({"image_id": test_filenames})
print("Test samples:", len(test_df))



## === cell 8
import math

_ROT_LAYER = tf.keras.layers.RandomRotation(
    factor=15.0 / 180.0,  # +/- 15 degrees
    fill_mode="constant",
    fill_value=0.0,
    interpolation="bilinear",
    seed=SEED,
)


@tf.function(jit_compile=True)
def _read_jpeg_resize(path, method, antialias):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize_with_pad(
        img,
        IMG_HEIGHT,
        IMG_WIDTH,
        method=method,
        antialias=antialias,
    )
    img = tf.clip_by_value(img, 0.0, 255.0)
    return tf.cast(img, tf.float32)  # float32 in 0..255


@tf.function(jit_compile=True)
def _stateless_split(seed, num=2):
    seeds = tf.random.experimental.stateless_split(seed, num)
    return tf.unstack(seeds, num=num)


@tf.function(jit_compile=True)
def _rand_brightness_contrast(img, seed):
    s1, s2 = _stateless_split(seed, 2)
    delta = tf.random.stateless_uniform([], s1, minval=-0.2 * 255.0, maxval=0.2 * 255.0)
    img = img + delta
    mean = tf.reduce_mean(img, axis=[0, 1], keepdims=True)
    factor = tf.random.stateless_uniform([], s2, minval=0.8, maxval=1.2)
    img = (img - mean) * factor + mean
    return tf.clip_by_value(img, 0.0, 255.0)


@tf.function(jit_compile=True)
def _rand_flip(img, seed):
    s1, s2 = _stateless_split(seed, 2)
    img = tf.image.stateless_random_flip_left_right(img, seed=s1)
    img = tf.image.stateless_random_flip_up_down(img, seed=s2)
    return img


@tf.function(jit_compile=True)
def _rand_zoom_and_rotate(img, seed):
    s1, _ = _stateless_split(seed, 2)
    scale = tf.random.stateless_uniform([], s1, minval=0.5, maxval=1.5)
    new_h = tf.cast(tf.round(tf.cast(IMG_HEIGHT, tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_WIDTH, tf.float32) * scale), tf.int32)
    zoomed = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BICUBIC, antialias=True
    )
    zoomed = tf.image.resize_with_crop_or_pad(zoomed, IMG_HEIGHT, IMG_WIDTH)
    rotated = _ROT_LAYER(zoomed, training=True)
    return tf.clip_by_value(rotated, 0.0, 255.0)


@tf.function(jit_compile=True)
def _train_augment(img, seed):
    img = _rand_flip(img, seed)
    img = _rand_brightness_contrast(img, seed + tf.constant([1, 0], tf.int32))
    img = _rand_zoom_and_rotate(img, seed + tf.constant([2, 0], tf.int32))
    return img


@tf.function(jit_compile=True)
def _eval_preprocess(img):
    return tf.cast(img, tf.float32) / 255.0


@tf.function(jit_compile=True)
def _normalize_01(img):
    return tf.cast(img, tf.float32) / 255.0


@tf.function(jit_compile=True)
def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    h = tf.cast(h, tf.int32)
    base_seed = tf.constant([SEED, 0], tf.int32)
    return tf.random.experimental.stateless_fold_in(base_seed, h)


_SHUFFLE_BUFFER = int(min(len(tr_df), 4096))


def _make_train_ds(image_ids, labels):
    paths = tf.strings.join([tf.constant(TRAIN_DIR), image_ids])
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(
        buffer_size=_SHUFFLE_BUFFER, seed=SEED, reshuffle_each_iteration=True
    )

    @tf.function(jit_compile=True)
    def _map_fn(path, label):
        seed = _seed_from_path(path)
        r = tf.random.stateless_uniform([], seed=seed, minval=0.0, maxval=1.0)

        img = _read_jpeg_resize(
            path, method=tf.image.ResizeMethod.BICUBIC, antialias=True
        )

        def _aug_from_img():
            a = _train_augment(img, seed + tf.constant([0, 1], tf.int32))
            return _normalize_01(a)

        def _plain_from_img():
            return _normalize_01(img)

        img_out = tf.cond(r > 0.5, _aug_from_img, _plain_from_img)
        return img_out, tf.cast(label, tf.int64)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    ds = ds.with_options(options)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    if _HAS_GPU:
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device(
                    _GPU_DEVICE, buffer_size=AUTOTUNE
                )
            )
        except Exception:
            pass
    return ds


def _make_eval_ds(image_ids, labels_or_none, root_dir, cache_ds=False):
    paths = tf.strings.join([tf.constant(root_dir), image_ids])

    @tf.function(jit_compile=True)
    def _eval_image(path):
        img = _read_jpeg_resize(
            path, method=tf.image.ResizeMethod.BICUBIC, antialias=True
        )
        return _eval_preprocess(img)

    if labels_or_none is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(_eval_image, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels_or_none))

        @tf.function(jit_compile=True)
        def _map_xy(p, y):
            return _eval_image(p), tf.cast(y, tf.int64)

        ds = ds.map(_map_xy, num_parallel_calls=AUTOTUNE, deterministic=True)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.map_and_batch_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_slack = True
    ds = ds.with_options(options)

    if cache_ds:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    if _HAS_GPU:
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device(
                    _GPU_DEVICE, buffer_size=AUTOTUNE
                )
            )
        except Exception:
            pass
    return ds


train_gen = _make_train_ds(
    tf.constant(tr_df["image_id"].values),
    tf.constant(tr_df["label"].values),
)
val_gen = _make_eval_ds(
    tf.constant(va_df["image_id"].values),
    tf.constant(va_df["label"].values),
    TRAIN_DIR,
    cache_ds=True,
)
test_gen = _make_eval_ds(
    tf.constant(test_df["image_id"].values),
    None,
    TEST_DIR,
    cache_ds=True,
)



## === cell 9
from tensorflow.keras import layers, models

base = tf.keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
base.trainable = False  # keep runtime reasonable and stable

inp = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = tf.keras.applications.xception.preprocess_input(
    inp * 255.0
)  # generator outputs 0..1
x = base(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(NUM_CLASSES, activation="softmax")(x)

model = models.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 10
history = model.fit(
    train_gen,
    validation_data=val_gen,
    epochs=EPOCHS,
    verbose=1,
)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/3463718032.py in <cell line: 0>()
----> 1 history = model.fit(
      2     train_gen,
      3     validation_data=val_gen,
      4     epochs=EPOCHS,
      5     verbose=1,

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

Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected at node path defined at (most recent call last):
<stack traces unavailable>
Detected unsupported operations when trying to compile graph __inference__map_fn_710[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference__map_fn_710[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[StatefulPartitionedCall]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_8593]

## === cell 11
pred_probs = model.predict(
    test_gen,
    verbose=1,
)
pred_labels = np.argmax(pred_probs, axis=1).astype(int)

print("Pred shape:", pred_probs.shape, "Labels shape:", pred_labels.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/140332444.py in <cell line: 0>()
----> 1 pred_probs = model.predict(
      2     test_gen,
      3     verbose=1,
      4 )
      5 pred_labels = np.argmax(pred_probs, axis=1).astype(int)

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

InvalidArgumentError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} Error in user-defined function passed to ParallelMapDatasetV2:13 transformation with iterator: Iterator::Root::Prefetch::BatchV2::MemoryCacheImpl::ParallelMapV2: Detected unsupported operations when trying to compile graph __inference__eval_image_779[] on XLA_CPU_JIT: _Arg (No registered '_Arg' OpKernel for XLA_CPU_JIT devices compatible with node {{node path}}
	 (OpKernel was found, but attributes didn't match) Requested Attributes: T=DT_STRING, _output_shapes=[[]], _user_specified_name="path", index=0){{node path}}
The op is created at: 
dummy_file_name:10:dummy_function_name
	tf2xla conversion failed while converting __inference__eval_image_779[]. Run with TF_DUMP_GRAPH_PREFIX=/path/to/dump/dir and --vmodule=xla_compiler=2 to obtain a dump of the compiled functions.
	 [[StatefulPartitionedCall]] [Op:IteratorGetNext] name: 

## === cell 12
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)

pred_map = dict(zip(test_df["image_id"].values, pred_labels))
sub = sample_sub.copy()
sub["label"] = sub["image_id"].map(pred_map)

sub["label"] = sub["label"].fillna(0).astype(int)

assert list(sub.columns) == ["image_id", "label"]
assert sub.shape[0] == sample_sub.shape[0]
assert sub["label"].between(0, NUM_CLASSES - 1).all()

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with rows:", len(sub))
print("submission.csv path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/997622036.py in <cell line: 0>()
      1 sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
      2 
----> 3 pred_map = dict(zip(test_df["image_id"].values, pred_labels))
      4 sub = sample_sub.copy()
      5 sub["label"] = sub["image_id"].map(pred_map)

NameError: name 'pred_labels' is not defined
