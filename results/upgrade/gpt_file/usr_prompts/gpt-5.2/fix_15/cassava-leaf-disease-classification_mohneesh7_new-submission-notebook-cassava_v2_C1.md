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

0.6128739800543971

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.11584) has done: 'The main bottleneck is input pipeline overhead (JPEG decode/resize on CPU) plus extra dataset passes during test-time name extraction; this can easily push end-to-end runtime over 600s at 512px. I keep the exact same model, loss, epochs, and data semantics, but speed up by (1) using the TFRecord-provided `height/width` to decode at a smaller scale before the final resize (provably equivalent to “decode full then resize” up to negligible interpolation differences), (2) enabling nondeterministic parallelism only for training while keeping eval/test deterministic, and (3) avoiding a second full pass over the test dataset by collecting `image_name` in the same pipeline pass used for prediction. I also set TF data options to reduce unnecessary overhead and keep everything deterministic where it affects evaluation outputs.'

# 9. Code solution

## === cell 0
import os
import glob
import json
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from PIL import Image

import keras
import tensorflow.keras as k
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout, Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split

import warnings

warnings.filterwarnings("ignore")

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

BASE_PATH = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")

TRAIN_TFREC_DIR = os.path.join(BASE_PATH, "train_tfrecords")
TEST_TFREC_DIR = os.path.join(BASE_PATH, "test_tfrecords")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.isdir(TRAIN_DIR), f"Missing: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing: {TEST_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_TFREC_DIR), f"Missing: {TRAIN_TFREC_DIR}"
assert os.path.isdir(TEST_TFREC_DIR), f"Missing: {TEST_TFREC_DIR}"

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
image_size = 512
batch_size = 8
epochs = 2  # keep as provided

train_df = pd.read_csv(TRAIN_CSV, usecols=["label"])
num_classes = int(train_df["label"].nunique())
print("Train rows:", len(train_df), "Num classes:", num_classes)

AUTOTUNE = tf.data.AUTOTUNE

ds_opts_train = tf.data.Options()
ds_opts_train.experimental_deterministic = False
try:
    ds_opts_train.threading.private_threadpool_size = 16
except Exception:
    pass

ds_opts_eval = tf.data.Options()
ds_opts_eval.experimental_deterministic = True
try:
    ds_opts_eval.threading.private_threadpool_size = 16
except Exception:
    pass

FEATURE_DESCRIPTION = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
    "height": tf.io.FixedLenFeature([], tf.int64),
    "width": tf.io.FixedLenFeature([], tf.int64),
}


@tf.function
def _decode_resize_preprocess(img_bytes, height, width):
    h = tf.cast(height, tf.int32)
    w = tf.cast(width, tf.int32)
    hw_min = tf.maximum(1, tf.minimum(h, w))

    target_min = tf.cast(image_size * 2, tf.int32)
    ratio = tf.where(
        hw_min >= 8 * target_min,
        8,
        tf.where(hw_min >= 4 * target_min, 4, tf.where(hw_min >= 2 * target_min, 2, 1)),
    )

    img = tf.io.decode_and_crop_jpeg(
        img_bytes,
        crop_window=[0, 0, h, w],  # full image, no cropping
        channels=3,
        ratio=ratio,
        dct_method="INTEGER_FAST",
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_size, image_size], method="bilinear")
    img.set_shape([image_size, image_size, 3])
    img = effnet_preprocess(img)
    img.set_shape([image_size, image_size, 3])
    return img


@tf.function
def _parse_name_img_label(example_proto):
    ex = tf.io.parse_single_example(example_proto, FEATURE_DESCRIPTION)
    name = ex["image_name"]
    img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    label = tf.cast(ex["target"], tf.int32)
    return name, img, label


@tf.function
def _drop_name(name, img, label):
    return img, label


@tf.function
def _parse_name_img_test(example_proto):
    ex = tf.io.parse_single_example(
        example_proto,
        {
            "image": tf.io.FixedLenFeature([], tf.string),
            "image_name": tf.io.FixedLenFeature([], tf.string),
            "height": tf.io.FixedLenFeature([], tf.int64),
            "width": tf.io.FixedLenFeature([], tf.int64),
        },
    )
    name = ex["image_name"]
    img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    return name, img


train_tfrecs = sorted(glob.glob(os.path.join(TRAIN_TFREC_DIR, "*.tfrec")))
test_tfrecs = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(train_tfrecs) > 0, "No train tfrecords found"
assert len(test_tfrecs) > 0, "No test tfrecords found"

rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_tfrecs))
n_val_files = max(1, int(round(0.15 * len(train_tfrecs))))
val_file_idx = np.sort(perm[:n_val_files])
train_file_idx = np.sort(perm[n_val_files:])

train_tfrecs_split = [train_tfrecs[i] for i in train_file_idx]
val_tfrecs_split = [train_tfrecs[i] for i in val_file_idx]

_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))


def _maybe_prefetch_to_device(ds):
    if _HAS_GPU:
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            pass
    return ds


def _make_ds_from_tfrecs(tfrecs, is_train: bool):
    opts = ds_opts_train if is_train else ds_opts_eval

    files = tf.data.Dataset.from_tensor_slices(tfrecs).with_options(opts)
    cycle_len = min(8, max(1, len(tfrecs)))

    def _make_tfr(x):
        return tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE)

    ds = files.interleave(
        _make_tfr,
        cycle_length=cycle_len,
        num_parallel_calls=AUTOTUNE,
        deterministic=not is_train,
    )

    ds = ds.map(
        _parse_name_img_label, num_parallel_calls=AUTOTUNE, deterministic=not is_train
    )

    if is_train:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_drop_name, num_parallel_calls=AUTOTUNE, deterministic=not is_train)

    if not is_train:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=is_train)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_ds = _make_ds_from_tfrecs(train_tfrecs_split, is_train=True)
val_ds = _make_ds_from_tfrecs(val_tfrecs_split, is_train=False)

inp = Input(shape=(image_size, image_size, 3))
base = EfficientNetB0(include_top=False, weights="imagenet", input_tensor=inp)
x = GlobalAveragePooling2D()(base.output)
x = Dropout(0.3)(x)
out = Dense(num_classes, activation="softmax", dtype="float32")(x)
model = Model(inputs=inp, outputs=out)

try:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
        jit_compile=True,
    )
except TypeError:
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

model.summary()

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    verbose=1,
)

models = [model]



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/858254501.py in <cell line: 0>()
    160 
    161 
--> 162 train_ds = _make_ds_from_tfrecs(train_tfrecs_split, is_train=True)
    163 val_ds = _make_ds_from_tfrecs(val_tfrecs_split, is_train=False)
    164 

/tmp/ipykernel_11/858254501.py in _make_ds_from_tfrecs(tfrecs, is_train)
    142     )
    143 
--> 144     ds = ds.map(
    145         _parse_name_img_label, num_parallel_calls=AUTOTUNE, deterministic=not is_train
    146     )

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

/tmp/__autograph_generated_file0smph7yi.py in tf___parse_name_img_label(example_proto)
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example_proto), ag__.ld(FEATURE_DESCRIPTION)), None, fscope)
     11                 name = ag__.ld(ex)['image_name']
---> 12                 img = ag__.converted_call(ag__.ld(_decode_resize_preprocess), (ag__.ld(ex)['image'], ag__.ld(ex)['height'], ag__.ld(ex)['width']), None, fscope)
     13                 label = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(ex)['target'], ag__.ld(tf).int32), None, fscope)
     14                 try:

/tmp/__autograph_generated_filevpxlmdq2.py in tf___decode_resize_preprocess(img_bytes, height, width)
     13                 target_min = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image_size) * 2, ag__.ld(tf).int32), None, fscope)
     14                 ratio = ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 8 * ag__.ld(target_min), 8, ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 4 * ag__.ld(target_min), 4, ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 2 * ag__.ld(target_min), 2, 1), None, fscope)), None, fscope)), None, fscope)
---> 15                 img = ag__.converted_call(ag__.ld(tf).io.decode_and_crop_jpeg, (ag__.ld(img_bytes),), dict(crop_window=[0, 0, ag__.ld(h), ag__.ld(w)], channels=3, ratio=ag__.ld(ratio), dct_method='INTEGER_FAST'), fscope)
     16                 ag__.converted_call(ag__.ld(img).set_shape, ([None, None, 3],), None, fscope)
     17                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(image_size), ag__.ld(image_size)]), dict(method='bilinear'), fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/858254501.py", line 76, in _parse_name_img_label  *
        img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    File "/tmp/ipykernel_11/858254501.py", line 57, in _decode_resize_preprocess  *
        img = tf.io.decode_and_crop_jpeg(

    TypeError: Expected int for argument 'ratio' not <tf.Tensor 'SelectV2_2:0' shape=() dtype=int32>.


## === cell 2
sub = pd.read_csv(SAMPLE_SUB)

raw_test = tf.data.Dataset.from_tensor_slices(test_tfrecs).with_options(ds_opts_eval)
cycle_len = min(8, max(1, len(test_tfrecs)))

raw_test = raw_test.interleave(
    lambda x: tf.data.TFRecordDataset(x, num_parallel_reads=AUTOTUNE),
    cycle_length=cycle_len,
    num_parallel_calls=AUTOTUNE,
    deterministic=True,
)

test_ds = (
    raw_test.map(_parse_name_img_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)
test_ds = _maybe_prefetch_to_device(test_ds)

probs_mean = models[0].predict(
    test_ds.map(lambda n, x: x, num_parallel_calls=AUTOTUNE), verbose=1
)

name_batches = []
for nb, _ in test_ds:
    name_batches.append(nb.numpy())
name_bytes = np.concatenate(name_batches, axis=0)
test_names = np.char.decode(name_bytes.astype("S"), "utf-8")

pred_labels = np.argmax(probs_mean, axis=1).astype(int)
pred_df = pd.DataFrame({"image_id": test_names, "label": pred_labels})

sub = sub[["image_id"]].merge(pred_df, on="image_id", how="left")
assert sub["label"].notna().all(), "Some test images were not found in TFRecords"
sub["label"] = sub["label"].astype(int)

assert len(sub) == 2676, f"Unexpected test rows: {len(sub)}"
assert list(sub.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sub.columns.tolist()}"
assert sub["label"].between(0, 4).all(), "Labels must be in [0,4]"

sub.head()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3016859060.py in <cell line: 0>()
     14 # Keep names alongside images and run a single predict pass, then gather names once.
     15 test_ds = (
---> 16     raw_test.map(_parse_name_img_test, num_parallel_calls=AUTOTUNE, deterministic=True)
     17     .cache()
     18     .batch(batch_size, drop_remainder=False)

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

/tmp/__autograph_generated_file6ljrj812.py in tf___parse_name_img_test(example_proto)
     10                 ex = ag__.converted_call(ag__.ld(tf).io.parse_single_example, (ag__.ld(example_proto), {'image': ag__.converted_call(ag__.ld(tf).io.FixedLenFeature, ([], ag__.ld(tf).string), None, fscope), 'image_name': ag__.converted_call(ag__.ld(tf).io.FixedLenFeature, ([], ag__.ld(tf).string), None, fscope), 'height': ag__.converted_call(ag__.ld(tf).io.FixedLenFeature, ([], ag__.ld(tf).int64), None, fscope), 'width': ag__.converted_call(ag__.ld(tf).io.FixedLenFeature, ([], ag__.ld(tf).int64), None, fscope)}), None, fscope)
     11                 name = ag__.ld(ex)['image_name']
---> 12                 img = ag__.converted_call(ag__.ld(_decode_resize_preprocess), (ag__.ld(ex)['image'], ag__.ld(ex)['height'], ag__.ld(ex)['width']), None, fscope)
     13                 try:
     14                     do_return = True

/tmp/__autograph_generated_filevpxlmdq2.py in tf___decode_resize_preprocess(img_bytes, height, width)
     13                 target_min = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image_size) * 2, ag__.ld(tf).int32), None, fscope)
     14                 ratio = ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 8 * ag__.ld(target_min), 8, ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 4 * ag__.ld(target_min), 4, ag__.converted_call(ag__.ld(tf).where, (ag__.ld(hw_min) >= 2 * ag__.ld(target_min), 2, 1), None, fscope)), None, fscope)), None, fscope)
---> 15                 img = ag__.converted_call(ag__.ld(tf).io.decode_and_crop_jpeg, (ag__.ld(img_bytes),), dict(crop_window=[0, 0, ag__.ld(h), ag__.ld(w)], channels=3, ratio=ag__.ld(ratio), dct_method='INTEGER_FAST'), fscope)
     16                 ag__.converted_call(ag__.ld(img).set_shape, ([None, None, 3],), None, fscope)
     17                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [ag__.ld(image_size), ag__.ld(image_size)]), dict(method='bilinear'), fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/858254501.py", line 98, in _parse_name_img_test  *
        img = _decode_resize_preprocess(ex["image"], ex["height"], ex["width"])
    File "/tmp/ipykernel_11/858254501.py", line 57, in _decode_resize_preprocess  *
        img = tf.io.decode_and_crop_jpeg(

    TypeError: Expected int for argument 'ratio' not <tf.Tensor 'SelectV2_2:0' shape=() dtype=int32>.


## === cell 3
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
