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

0.5660320338470837

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by (1) decoding/resizing and caching ~18.7k 300×300 images in memory and (2) running heavy augmentation (projective rotation) and validation every epoch, which forces full-dataset passes multiple times. I keep the exact model and training stages, but switch the input pipeline to read from the provided TFRecords (same images/labels) with deterministic, parallel parsing to cut I/O/CPU overhead significantly. I also remove in-memory `.cache()` of the full decoded dataset (it’s expensive to build and can thrash memory) and replace it with TFRecord streaming + `prefetch`, while keeping determinism and the same augmentation logic. Finally, I keep evaluation semantics but reduce validation overhead by setting `validation_freq`/`steps_per_epoch`? (No—those would change semantics), so instead I preserve full validation while making it faster via TFRecord + parallelism and avoiding redundant path-building.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("KERAS_HOME", "/kaggle/working/.keras")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

if os.path.exists("/kaggle/input/cassava-leaf-disease-classification"):
    BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
elif os.path.exists("../input/cassava-leaf-disease-classification"):
    BASE_INPUT = "../input/cassava-leaf-disease-classification"
elif os.path.exists("/kaggle/data/cassava-leaf-disease-classification"):
    BASE_INPUT = "/kaggle/data/cassava-leaf-disease-classification"
else:
    BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"

TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TRAIN_TFREC_DIR = os.path.join(BASE_INPUT, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_INPUT, "test_tfrecords")

print("BASE_INPUT:", BASE_INPUT)
print("TensorFlow:", tf.__version__)
print("TRAIN_CSV exists:", os.path.exists(TRAIN_CSV))
print("SAMPLE_SUB exists:", os.path.exists(SAMPLE_SUB))
print("TRAIN_TFREC_DIR exists:", os.path.exists(TRAIN_TFREC_DIR))
print("TEST_TFREC_DIR exists:", os.path.exists(TEST_TFREC_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_df = pd.read_csv(TRAIN_CSV)
num_classes = int(train_df["label"].nunique())

train_df = train_df.copy()
train_df["label"] = train_df["label"].astype(int)

IMG_SIZE = (300, 300)
BATCH_SIZE = 32
EPOCHS = 2  # keep runtime within budget while still yielding non-random predictions
VAL_SPLIT = 0.10

train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))

assert len(train_tfrecs) > 0, f"No train tfrecords found in {TRAIN_TFREC_DIR}"
assert len(test_tfrecs) > 0, f"No test tfrecords found in {TEST_TFREC_DIR}"

n = len(train_df)
val_n = int(round(n * VAL_SPLIT))

rng = np.random.RandomState(SEED)
perm_files = rng.permutation(len(train_tfrecs))
val_file_count = max(1, int(round(len(train_tfrecs) * VAL_SPLIT)))
val_files = [train_tfrecs[i] for i in perm_files[:val_file_count]]
trn_files = [train_tfrecs[i] for i in perm_files[val_file_count:]]
if len(trn_files) == 0:
    trn_files, val_files = train_tfrecs[:-1], train_tfrecs[-1:]

_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "label": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
}


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _augment(img, seed_pair):
    seed = tf.stack([seed_pair[0], seed_pair[1]])

    r = tf.random.stateless_uniform(
        [5], seed=seed, minval=0.0, maxval=1.0, dtype=tf.float32
    )

    img = tf.cond(r[0] < 0.5, lambda: tf.image.flip_left_right(img), lambda: img)

    max_dx = tf.cast(tf.round(IMG_SIZE[1] * 0.05), tf.int32)
    max_dy = tf.cast(tf.round(IMG_SIZE[0] * 0.05), tf.int32)

    dx = (
        tf.cast(tf.floor(r[1] * tf.cast(2 * max_dx + 1, tf.float32)), tf.int32) - max_dx
    )
    dy = (
        tf.cast(tf.floor(r[2] * tf.cast(2 * max_dy + 1, tf.float32)), tf.int32) - max_dy
    )

    pad_x = max_dx
    pad_y = max_dy
    img_padded = tf.image.pad_to_bounding_box(
        img, pad_y, pad_x, IMG_SIZE[0] + 2 * pad_y, IMG_SIZE[1] + 2 * pad_x
    )
    img = tf.image.crop_to_bounding_box(
        img_padded, pad_y + dy, pad_x + dx, IMG_SIZE[0], IMG_SIZE[1]
    )

    scale = 0.9 + r[3] * (1.1 - 0.9)
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * scale), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * scale), tf.int32)
    img2 = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    img = img2

    angle = (-10.0 + r[4] * (20.0)) * (np.pi / 180.0)

    c = tf.cast(IMG_SIZE[1], tf.float32) / 2.0
    rr = tf.cast(IMG_SIZE[0], tf.float32) / 2.0
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a
    a2 = c - a0 * c - a1 * rr
    b2 = rr - b0 * c - b1 * rr
    transform = tf.stack([a0, a1, a2, b0, b1, b2, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE[0], IMG_SIZE[1]], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
    return img


@tf.function
def _train_map_fn(img, label, idx_in_epoch):
    seed_pair = (tf.constant(SEED, tf.int32), tf.cast(idx_in_epoch, tf.int32))
    img = _augment(img, seed_pair)
    return img, label


@tf.function
def _valid_map_fn(img, label):
    return img, label


options = tf.data.Options()
options.deterministic = True


def _make_train_valid_datasets():
    def _decode_map_batched(serialized_batch):
        ex = tf.io.parse_example(serialized_batch, _FEATURES_TRAIN)
        img_bytes = ex["image"]
        label = tf.cast(ex["label"], tf.int32)
        target = tf.cast(ex["target"], tf.int32)
        y = tf.where(label >= 0, label, target)

        imgs = tf.map_fn(
            _decode_and_resize_from_bytes,
            img_bytes,
            fn_output_signature=tf.float32,
            parallel_iterations=AUTOTUNE,
        )
        return imgs, y

    train_ser = tf.data.TFRecordDataset(
        trn_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    valid_ser = tf.data.TFRecordDataset(
        val_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)

    train_ser = train_ser.shuffle(
        buffer_size=min(8192, max(1, n - val_n)),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    train_ds_local = train_ser.batch(BATCH_SIZE, drop_remainder=False).map(
        _decode_map_batched, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    train_ds_local = train_ds_local.enumerate()

    def _augment_batch(batch_idx, data):
        imgs, labels = data
        bsz = tf.shape(imgs)[0]
        idxs = tf.range(bsz, dtype=tf.int64)
        global_idx = tf.cast(batch_idx, tf.int64) * tf.cast(BATCH_SIZE, tf.int64) + idxs

        def _aug_one(args):
            img, lab, gidx = args
            return _train_map_fn(img, lab, tf.cast(gidx, tf.int32))

        out_imgs, out_labels = tf.map_fn(
            _aug_one,
            (imgs, labels, global_idx),
            fn_output_signature=(tf.float32, tf.int32),
            parallel_iterations=AUTOTUNE,
        )
        return out_imgs, out_labels

    train_ds_local = train_ds_local.map(
        _augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    train_ds_local = train_ds_local.prefetch(AUTOTUNE)

    valid_ds_local = (
        valid_ser.batch(BATCH_SIZE, drop_remainder=False)
        .map(_decode_map_batched, num_parallel_calls=AUTOTUNE, deterministic=True)
        .map(_valid_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    )
    valid_ds_local = valid_ds_local.cache().prefetch(AUTOTUNE)

    return train_ds_local, valid_ds_local


train_ds, valid_ds = _make_train_valid_datasets()

inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
base = EfficientNetB3(include_top=False, weights="imagenet", input_tensor=inputs)
x = GlobalAveragePooling2D()(base.output)
outputs = Dense(num_classes, activation="softmax")(x)
my_model = Model(inputs=inputs, outputs=outputs)

base.trainable = False
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

_ = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)

base.trainable = True
my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
_ = my_model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=1,
    verbose=1,
)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1747763347.py in <cell line: 0>()
    191 
    192 
--> 193 train_ds, valid_ds = _make_train_valid_datasets()
    194 
    195 inputs = Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

/tmp/ipykernel_11/1747763347.py in _make_train_valid_datasets()
    152     )
    153 
--> 154     train_ds_local = train_ser.batch(BATCH_SIZE, drop_remainder=False).map(
    155         _decode_map_batched, num_parallel_calls=AUTOTUNE, deterministic=True
    156     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_filer62ve60_.py in tf___decode_map_batched(serialized_batch)
     13                 target = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(ex)['target'], ag__.ld(tf).int32), None, fscope)
     14                 y = ag__.converted_call(ag__.ld(tf).where, (ag__.ld(label) >= 0, ag__.ld(label), ag__.ld(target)), None, fscope)
---> 15                 imgs = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.ld(_decode_and_resize_from_bytes), ag__.ld(img_bytes)), dict(fn_output_signature=ag__.ld(tf).float32, parallel_iterations=ag__.ld(AUTOTUNE)), fscope)
     16                 try:
     17                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    457 
    458   if kwargs is not None:
--> 459     return f(*args, **kwargs)
    460   return f(*args)
    461 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    430     raise TypeError("'body' must be callable.")
    431   if parallel_iterations < 1:
--> 432     raise TypeError("'parallel_iterations' must be a positive integer.")
    433 
    434   loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

TypeError: in user code:

    File "/tmp/ipykernel_11/1747763347.py", line 133, in _decode_map_batched  *
        imgs = tf.map_fn(

    TypeError: 'parallel_iterations' must be a positive integer.


## === cell 2
sample_df = pd.read_csv(SAMPLE_SUB)
df_test = sample_df[["image_id"]].copy()

_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _parse_test_batch(serialized_batch):
    ex = tf.io.parse_example(serialized_batch, _FEATURES_TEST)
    imgs = tf.map_fn(
        _decode_and_resize_from_bytes,
        ex["image"],
        fn_output_signature=tf.float32,
        parallel_iterations=AUTOTUNE,
    )
    return imgs, ex["image_name"]


test_ds = tf.data.TFRecordDataset(test_tfrecs, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.with_options(options)
test_ds = (
    test_ds.batch(128, drop_remainder=False)
    .map(_parse_test_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
    .prefetch(AUTOTUNE)
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2379901750.py in <cell line: 0>()
     24 test_ds = (
     25     test_ds.batch(128, drop_remainder=False)
---> 26     .map(_parse_test_batch, num_parallel_calls=AUTOTUNE, deterministic=True)
     27     .prefetch(AUTOTUNE)
     28 )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_fileclfzpc4_.py in tf___parse_test_batch(serialized_batch)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_example, (ag__.ld(serialized_batch), ag__.ld(_FEATURES_TEST)), None, fscope)
---> 11                 imgs = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.ld(_decode_and_resize_from_bytes), ag__.ld(ex)['image']), dict(fn_output_signature=ag__.ld(tf).float32, parallel_iterations=ag__.ld(AUTOTUNE)), fscope)
     12                 try:
     13                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_11/2379901750.py", line 13, in _parse_test_batch  *
        imgs = tf.map_fn(

    TypeError: 'parallel_iterations' must be a positive integer.


## === cell 3
all_names = []
all_preds = []
for batch_imgs, batch_names in test_ds:
    batch_pred = my_model(batch_imgs, training=False)
    all_names.append(batch_names)
    all_preds.append(batch_pred)

test_names = tf.concat(all_names, axis=0).numpy().astype("U")
pred_test = tf.concat(all_preds, axis=0).numpy()
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

pred_df = pd.DataFrame({"image_id": test_names, "label": pred_test_labels})
final_csv = df_test.merge(pred_df, on="image_id", how="left")

if final_csv["label"].isna().any():
    sample_df = pd.read_csv(SAMPLE_SUB)
    sample_df = sample_df.copy()
    sample_df["path"] = [
        f"{TEST_IMG_DIR.rstrip('/')}/{fn}"
        for fn in sample_df["image_id"].astype(str).tolist()
    ]

    if len(sample_df) and (not os.path.exists(sample_df["path"].iloc[0])):
        alt_dir = os.path.join(TEST_IMG_DIR, "test_images")
        sample_df["path"] = [
            f"{alt_dir.rstrip('/')}/{fn}"
            for fn in sample_df["image_id"].astype(str).tolist()
        ]

    test_paths = sample_df["path"].values

    @tf.function
    def _decode_and_resize(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([IMG_SIZE[0], IMG_SIZE[1], 3])
        return img

    @tf.function
    def _test_map_fn(path):
        return _decode_and_resize(path)

    test_ds_fallback = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds_fallback = test_ds_fallback.map(
        _test_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds_fallback = test_ds_fallback.with_options(options)
    test_ds_fallback = test_ds_fallback.batch(128, drop_remainder=False).prefetch(
        AUTOTUNE
    )

    pred_test = my_model.predict(test_ds_fallback, verbose=1)
    pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

    final_csv = pd.DataFrame(
        {"image_id": sample_df["image_id"].values, "label": pred_test_labels}
    )

final_csv["label"] = final_csv["label"].astype(int)

assert (
    final_csv.shape[0] == sample_df.shape[0]
), "Submission row count does not match sample_submission."
assert list(final_csv.columns) == [
    "image_id",
    "label",
], "Submission columns are incorrect."

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2340595476.py in <cell line: 0>()
      1 all_names = []
      2 all_preds = []
----> 3 for batch_imgs, batch_names in test_ds:
      4     batch_pred = my_model(batch_imgs, training=False)
      5     all_names.append(batch_names)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __iter__(self)
    323       self._disallow_iteration()
    324 
--> 325     first_dim = self._get_first_dim()
    326     return _TensorIterator(self, first_dim)
    327 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in _get_first_dim(self)
    331       raise TypeError("Cannot iterate over a tensor with unknown shape.")
    332     if not shape:
--> 333       raise TypeError("Cannot iterate over a scalar tensor.")
    334     if shape[0] is None:
    335       raise TypeError(

TypeError: Cannot iterate over a scalar tensor.
