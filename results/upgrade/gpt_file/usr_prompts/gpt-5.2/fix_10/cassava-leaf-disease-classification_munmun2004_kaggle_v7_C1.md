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

0.1403747355696585

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

import json
import random
import warnings
import math
import glob
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalAveragePooling2D, Dense, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import EfficientNetB3

import matplotlib.pyplot as plt

warnings.filterwarnings("ignore")

print("TensorFlow:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

try:
    pass
except Exception as e:
    print("JIT toggle skipped:", repr(e))

try:
    pass
except Exception as e:
    print("Determinism toggle skipped:", repr(e))

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Could not set threading:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
work_dir = "../input/cassava-leaf-disease-classification/"
train_path = os.path.join(work_dir, "train_images")
test_path = os.path.join(work_dir, "test_images")
train_tfrec_dir = os.path.join(work_dir, "train_tfrecords")
test_tfrec_dir = os.path.join(work_dir, "test_tfrecords")

print("work_dir exists:", os.path.exists(work_dir))
print("train_path exists:", os.path.exists(train_path))
print("test_path exists:", os.path.exists(test_path))
print("train_tfrec_dir exists:", os.path.exists(train_tfrec_dir))
print("test_tfrec_dir exists:", os.path.exists(test_tfrec_dir))




## === cell 2
def seed_everything(seed=0):
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


seed = 21
seed_everything(seed)



## === cell 3
data = pd.read_csv(os.path.join(work_dir, "train.csv"))
print("train.csv shape:", data.shape)
print(data["label"].value_counts())



## === cell 4
with open(os.path.join(work_dir, "label_num_to_disease_map.json"), "r") as f:
    real_labels = json.load(f)
real_labels = {int(k): v for k, v in real_labels.items()}

data["class_name"] = data["label"].map(real_labels)
print(data[["image_id", "label", "class_name"]].head())



## === cell 5
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    data,
    test_size=0.05,
    random_state=42,
    stratify=data["label"],
)

print("train/val sizes:", train_df.shape, val_df.shape)



## === cell 6
IMG_SIZE = 456
size = (IMG_SIZE, IMG_SIZE)
n_CLASS = 5
BATCH_SIZE = 15
EPOCHS = 20

AUTOTUNE = tf.data.AUTOTUNE

class_names_sorted = sorted(data["class_name"].unique().tolist())
class_name_to_idx = {name: i for i, name in enumerate(class_names_sorted)}
idx_to_class_name = {i: name for name, i in class_name_to_idx.items()}
print("Class mapping (class_name -> index):", class_name_to_idx)

train_tfrec_files = sorted(glob.glob(os.path.join(train_tfrec_dir, "*.tfrec")))
test_tfrec_files = sorted(glob.glob(os.path.join(test_tfrec_dir, "*.tfrec")))
print("Found train tfrecords:", len(train_tfrec_files))
print("Found test tfrecords:", len(test_tfrec_files))




## === cell 7
def _effnet_preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


@tf.function
def _augment(img, seed_pair):
    offsets = tf.constant(
        [[0, 0], [1, 0], [2, 0], [3, 0], [4, 0], [5, 0], [6, 0]], dtype=tf.int32
    )
    seeds = seed_pair[None, :] + offsets  # shape [7,2]

    img = tf.image.stateless_random_flip_left_right(img, seed=seeds[0])
    img = tf.image.stateless_random_flip_up_down(img, seed=seeds[1])

    angle = tf.random.stateless_uniform(
        [], seed=seeds[2], minval=-40.0, maxval=40.0
    ) * (math.pi / 180.0)
    scale = tf.random.stateless_uniform([], seed=seeds[3], minval=0.8, maxval=1.2)

    tx = tf.random.stateless_uniform(
        [], seed=seeds[4], minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=seeds[5], minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_SIZE, tf.float32)
    shear = tf.random.stateless_uniform([], seed=seeds[6], minval=-0.2, maxval=0.2)

    cx = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0
    cy = (tf.cast(IMG_SIZE, tf.float32) - 1.0) / 2.0

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    a0 = scale * (cos_a + shear * sin_a)
    a1 = scale * (-sin_a + shear * cos_a)
    b0 = scale * (sin_a)
    b1 = scale * (cos_a)

    det = a0 * b1 - a1 * b0
    inv_a0 = b1 / det
    inv_a1 = -a1 / det
    inv_b0 = -b0 / det
    inv_b1 = a0 / det

    t_x = cx + tx - (a0 * cx + a1 * cy)
    t_y = cy + ty - (b0 * cx + b1 * cy)

    c = -(inv_a0 * t_x + inv_a1 * t_y)
    f = -(inv_b0 * t_x + inv_b1 * t_y)

    transform = tf.stack([inv_a0, inv_a1, c, inv_b0, inv_b1, f, 0.0, 0.0])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="NEAREST",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


TFREC_FEATURES_TRAIN = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}
TFREC_FEATURES_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


@tf.function
def _decode_resize_float(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, size, method=tf.image.ResizeMethod.NEAREST_NEIGHBOR)
    return tf.cast(img, tf.float32)


@tf.function
def _parse_train_decode_only(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_TRAIN)
    img = _decode_resize_float(ex["image"])
    y = tf.cast(ex["target"], tf.int32)
    return img, y


@tf.function
def _train_aug_and_preproc(idx, img, y):
    s = tf.stack([tf.cast(seed, tf.int32), tf.cast(idx, tf.int32)], axis=0)
    img = _augment(img, s)
    img = _effnet_preprocess(img)
    y = tf.one_hot(y, depth=n_CLASS, dtype=tf.float32)
    return img, y


@tf.function
def _val_decode_and_preproc(path, y):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_float(img_bytes)
    img = _effnet_preprocess(img)
    y = tf.one_hot(tf.cast(y, tf.int32), depth=n_CLASS, dtype=tf.float32)
    return img, y


@tf.function
def _parse_and_test_preproc(example_proto):
    ex = tf.io.parse_single_example(example_proto, TFREC_FEATURES_TEST)
    img = _decode_resize_float(ex["image"])
    img = _effnet_preprocess(img)
    return img, ex["image_name"]


def _dataset_options():
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.autotune.enabled = True
    return options


def _with_index_scan(ds):
    def _scan_fn(state, elem):
        new_state = state + 1
        return new_state, (state, elem)

    ds = ds.scan(tf.constant(0, dtype=tf.int64), _scan_fn)
    return ds.map(
        lambda _state, idx_elem: idx_elem,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )


def make_train_dataset_from_tfrecords(tfrec_files):
    ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(_dataset_options())
    ds = ds.shuffle(buffer_size=4096, seed=42, reshuffle_each_iteration=True)

    ds = ds.map(
        _parse_train_decode_only, num_parallel_calls=AUTOTUNE, deterministic=True
    )

    ds = (
        ds.cache()
    )  # cache decoded+resized tensors; augmentation still varies per epoch

    ds = _with_index_scan(ds)  # element structure: (idx, (img, y_int))

    ds = ds.map(
        lambda idx, img_y: _train_aug_and_preproc(idx, img_y[0], img_y[1]),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_dataset_from_jpegs(df, directory):
    paths = (directory.rstrip("/") + "/" + df["image_id"].astype(str)).values
    labels = df["label"].astype(np.int32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_dataset_options())
    ds = ds.map(
        _val_decode_and_preproc, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_dataset_from_tfrecords(train_tfrec_files)
val_ds = make_val_dataset_from_jpegs(val_df, train_path)

STEP_SIZE_TRAIN = int(math.ceil(len(train_df) / BATCH_SIZE))
STEP_SIZE_VAL = int(math.ceil(len(val_df) / BATCH_SIZE))
print("STEP_SIZE_TRAIN:", STEP_SIZE_TRAIN, "STEP_SIZE_VAL:", STEP_SIZE_VAL)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/2207331759.py in <cell line: 0>()
    175 
    176 
--> 177 train_ds = make_train_dataset_from_tfrecords(train_tfrec_files)
    178 val_ds = make_val_dataset_from_jpegs(val_df, train_path)
    179 

/tmp/ipykernel_55/2207331759.py in make_train_dataset_from_tfrecords(tfrec_files)
    149 
    150     # Bugfix: correctly unpack nested element; previous code did img_y[0]/[1] on a scalar.
--> 151     ds = ds.map(
    152         lambda idx, img_y: _train_aug_and_preproc(idx, img_y[0], img_y[1]),
    153         num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filekfnm6xau.py in <lambda>(idx, img_y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda idx, img_y: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_aug_and_preproc, (idx, img_y[0], img_y[1]), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filekfnm6xau.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda idx, img_y: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_aug_and_preproc, (idx, img_y[0], img_y[1]), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in _create_c_op(graph, node_def, inputs, control_inputs, op_def, extract_traceback)
   1054   except errors.InvalidArgumentError as e:
   1055     # Convert to ValueError for backwards compatibility.
-> 1056     raise ValueError(e.message)
   1057 
   1058   # Record the current Python stack trace as the creating stacktrace of this

ValueError: in user code:

    File "/tmp/ipykernel_55/2207331759.py", line 152, in None  *
        lambda idx, img_y: _train_aug_and_preproc(idx, img_y[0], img_y[1])

    ValueError: Attempting to slice scalar input. for '{{node strided_slice}} = StridedSlice[Index=DT_INT32, T=DT_INT32, begin_mask=0, ellipsis_mask=0, end_mask=0, new_axis_mask=0, shrink_axis_mask=1](args_1, strided_slice/stack, strided_slice/stack_1, strided_slice/stack_2)' with input shapes: [], [1], [1], [1] and with computed input tensors: input[3] = <1>.


## === cell 8
def create_model():
    model = Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(IMG_SIZE, IMG_SIZE, 3),
            include_top=False,
            weights="imagenet",
        )
    )
    model.add(GlobalAveragePooling2D())
    model.add(
        Dense(
            256,
            activation="relu",
            bias_regularizer=tf.keras.regularizers.L1L2(l1=0.01, l2=0.001),
        )
    )
    model.add(Dropout(0.5))
    model.add(Dense(n_CLASS, activation="softmax"))
    return model


leaf_model = create_model()
leaf_model.summary()



## === cell 9
loss = tf.keras.losses.CategoricalCrossentropy(
    from_logits=False,
    label_smoothing=0.0001,
    name="categorical_crossentropy",
)

leaf_model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss=loss,
    metrics=["categorical_accuracy"],
)

model_save = ModelCheckpoint(
    filepath="Cassava_best.weights.h5",
    save_best_only=True,
    save_weights_only=True,
    monitor="val_loss",
    mode="min",
    verbose=1,
)

early_stop = EarlyStopping(
    monitor="val_loss",
    min_delta=0.001,
    patience=5,
    mode="min",
    verbose=1,
    restore_best_weights=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.3,
    patience=2,
    min_delta=0.001,
    mode="min",
    verbose=1,
)



## === cell 10
history = leaf_model.fit(
    train_ds,
    steps_per_epoch=STEP_SIZE_TRAIN,
    validation_data=val_ds,
    validation_steps=STEP_SIZE_VAL,
    epochs=EPOCHS,
    callbacks=[model_save, early_stop, reduce_lr],
    verbose=1,
)

if os.path.exists("Cassava_best.weights.h5"):
    leaf_model.load_weights("Cassava_best.weights.h5")

loaded_model = leaf_model



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1754087583.py in <cell line: 0>()
      1 history = leaf_model.fit(
----> 2     train_ds,
      3     steps_per_epoch=STEP_SIZE_TRAIN,
      4     validation_data=val_ds,
      5     validation_steps=STEP_SIZE_VAL,

NameError: name 'train_ds' is not defined

## === cell 11
plt.figure(figsize=(8, 4))
plt.plot(history.epoch, history.history.get("loss", []), "-o", label="training_loss")
plt.plot(
    history.epoch, history.history.get("val_loss", []), "-o", label="validation_loss"
)
plt.legend()
plt.xlim(left=0)
plt.xlabel("epochs")
plt.ylabel("loss")
plt.show()

plt.figure(figsize=(8, 4))
plt.plot(
    history.epoch,
    history.history.get("categorical_accuracy", []),
    "-o",
    label="train_accuracy",
)
plt.plot(
    history.epoch,
    history.history.get("val_categorical_accuracy", []),
    "-o",
    label="validation_accuracy",
)
plt.legend()
plt.xlim(left=0)
plt.xlabel("epochs")
plt.ylabel("accuracy")
plt.show()



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/402929057.py in <cell line: 0>()
      1 plt.figure(figsize=(8, 4))
----> 2 plt.plot(history.epoch, history.history.get("loss", []), "-o", label="training_loss")
      3 plt.plot(
      4     history.epoch, history.history.get("val_loss", []), "-o", label="validation_loss"
      5 )

NameError: name 'history' is not defined

## === cell 12
from sklearn.metrics import confusion_matrix, classification_report

Y_pred = loaded_model.predict(val_ds, verbose=1)
y_pred = np.argmax(Y_pred, axis=1)

y_true = val_df["label"].astype(int).values
print("Confusion Matrix")
print(confusion_matrix(y_true, y_pred))

target_names = [idx_to_class_name[i] for i in range(n_CLASS)]
print(classification_report(y_true, y_pred, target_names=target_names))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2632920348.py in <cell line: 0>()
      1 from sklearn.metrics import confusion_matrix, classification_report
      2 
----> 3 Y_pred = loaded_model.predict(val_ds, verbose=1)
      4 y_pred = np.argmax(Y_pred, axis=1)
      5 

NameError: name 'loaded_model' is not defined

## === cell 13
ss = pd.read_csv(os.path.join(work_dir, "sample_submission.csv"))

test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_ds = test_ds.with_options(_dataset_options())
test_ds = test_ds.map(
    _parse_and_test_preproc, num_parallel_calls=AUTOTUNE, deterministic=True
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

all_names_chunks = []
preds_chunks = []
for xb, nb in test_ds:
    pb = loaded_model(xb, training=False).numpy()
    all_names_chunks.append(nb.numpy())
    preds_chunks.append(np.argmax(pb, axis=1).astype(np.int64))

all_names = np.concatenate(all_names_chunks, axis=0).astype("U")
preds = np.concatenate(preds_chunks, axis=0).astype(int)

submission = pd.DataFrame({"image_id": all_names, "label": preds})

submission = ss[["image_id"]].merge(submission, on="image_id", how="left")
if submission["label"].isna().any():
    submission["label"] = submission["label"].fillna(0)

submission["label"] = submission["label"].astype(int)
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv with shape:", submission.shape)
print(submission.head())
print("submission.csv exists:", os.path.exists("submission.csv"))

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1272490270.py in <cell line: 0>()
     11 preds_chunks = []
     12 for xb, nb in test_ds:
---> 13     pb = loaded_model(xb, training=False).numpy()
     14     all_names_chunks.append(nb.numpy())
     15     preds_chunks.append(np.argmax(pb, axis=1).astype(np.int64))

NameError: name 'loaded_model' is not defined
