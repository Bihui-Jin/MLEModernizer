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

0.8779087337564219

# 6. Current score

0.14163

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.14163) has done: 'Main bottlenecks are the expensive per-image augmentation (projective transform with Python-level helper functions) executed every epoch, plus unnecessary `cache()` for training (which forces storing all decoded 224x224x3 float images in RAM and can slow/pressure memory). The core model and training loop stay identical, but we (1) precompute labels via vectorized mapping, (2) compile the tf.data pipeline with `@tf.function` and move constant tensors outside the hot path, (3) remove training cache while keeping validation cache (equivalent semantics, faster), and (4) add `steps_per_execution` to reduce Python overhead during `fit` without changing the math. We also avoid the expensive `enumerate()`/seed building by using `Dataset.random()` with a fixed seed and stateless ops, keeping determinism. These changes preserve accuracy while cutting input pipeline overhead substantially to fit the 600s budget.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{BASE_PATH}/train.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
LABEL_MAP_PATH = f"{BASE_PATH}/label_num_to_disease_map.json"

label_to_disease = pd.read_json(LABEL_MAP_PATH, typ="series")
train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=SEED,
)

classes = sorted(train_csv["disease"].unique().tolist())
class_indices = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

print("Classes found (tf.data):", class_indices)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

IMG_H = tf.constant(IMG_SIZE[0], tf.int32)
IMG_W = tf.constant(IMG_SIZE[1], tf.int32)
OUT_SHAPE = tf.stack([IMG_H, IMG_W])
CX = tf.constant((IMG_SIZE[1] - 1) / 2.0, tf.float32)
CY = tf.constant((IMG_SIZE[0] - 1) / 2.0, tf.float32)
PI_OVER_180 = tf.constant(np.pi / 180.0, tf.float32)
SEED_I64 = tf.constant(SEED, tf.int64)


@tf.function
def _read_and_decode(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment_like_imagedatagenerator(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=(seed2[0], seed2[1] + 1))
    img = tf.image.stateless_random_flip_up_down(img, seed=(seed2[0], seed2[1] + 2))

    angle = (
        tf.random.stateless_uniform(
            [], seed=(seed2[0], seed2[1] + 3), minval=-45.0, maxval=45.0
        )
        * PI_OVER_180
    )

    tx = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 4), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_H, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 5), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_W, tf.float32)

    shear = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 6), minval=-0.2, maxval=0.2
    )

    zoom = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 7), minval=0.8, maxval=1.2
    )

    cos_a = tf.math.cos(angle) / zoom
    sin_a = tf.math.sin(angle) / zoom

    shear_m = tf.reshape(
        tf.stack([1.0, tf.math.tan(shear), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        (3, 3),
    )
    rot_m = tf.reshape(
        tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0]),
        (3, 3),
    )

    m = tf.linalg.matmul(rot_m, shear_m)

    a0, a1, a2, b0, b1, b2, c0, c1, _ = tf.unstack(tf.reshape(m, (-1,)))
    a2 = a2 + CX - (a0 * CX + a1 * CY) - ty
    b2 = b2 + CY - (b0 * CX + b1 * CY) - tx

    t = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=t,
        output_shape=OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


@tf.function
def _preprocess(img):
    return preprocess_input(img)


def make_dataset(df, training):
    paths = df["path"].to_numpy()

    y = df["disease"].map(class_indices).to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    options = tf.data.Options()
    options.experimental_deterministic = True  # preserve determinism
    options.autotune.enabled = True
    ds = ds.with_options(options)

    @tf.function
    def _decode(path, label):
        img = _read_and_decode(path)
        return img, label

    ds = ds.map(_decode, num_parallel_calls=AUTOTUNE)

    if not training:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)

        rng = tf.data.Dataset.random(seed=SEED, rerandomize_each_iteration=True)
        ds = tf.data.Dataset.zip((ds, rng))

        @tf.function
        def _aug(pair, r):
            img, label = pair
            r_i64 = tf.cast(r * tf.cast(tf.int64.max, tf.float32), tf.int64)
            seed2 = tf.stack([SEED_I64, r_i64], axis=0)
            img = _augment_like_imagedatagenerator(img, seed2)
            return img, label

        ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)

    @tf.function
    def _prep(img, label):
        img = _preprocess(img)
        label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
        return img, label_oh

    ds = ds.map(_prep, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train, training=True)
valid_ds = make_dataset(valid, training=False)

steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3144994684.py in <cell line: 0>()
    178 
    179 
--> 180 train_ds = make_dataset(train, training=True)
    181 valid_ds = make_dataset(valid, training=False)
    182 

/tmp/ipykernel_11/3144994684.py in make_dataset(df, training)
    164             return img, label
    165 
--> 166         ds = ds.map(_aug, num_parallel_calls=AUTOTUNE)
    167 
    168     @tf.function

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

/tmp/__autograph_generated_filerwp9wwj0.py in tf___aug(pair, r)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img, label = ag__.ld(pair)
---> 11                 r_i64 = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(r) * ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(tf).int64.max, ag__.ld(tf).float32), None, fscope), ag__.ld(tf).int64), None, fscope)
     12                 seed2 = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(SEED_I64), ag__.ld(r_i64)],), dict(axis=0), fscope)
     13                 img = ag__.converted_call(ag__.ld(_augment_like_imagedatagenerator), (ag__.ld(img), ag__.ld(seed2)), None, fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/3144994684.py", line 161, in _aug  *
        r_i64 = tf.cast(r * tf.cast(tf.int64.max, tf.float32), tf.int64)

    TypeError: Input 'y' of 'Mul' Op has type float32 that does not match type int64 of argument 'x'.


## === cell 2
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=steps_per_epoch,  # executes an epoch as a single tf.function call
)

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/612835021.py in <cell line: 0>()
     15     loss="categorical_crossentropy",
     16     metrics=["accuracy"],
---> 17     steps_per_execution=steps_per_epoch,  # executes an epoch as a single tf.function call
     18 )
     19 

NameError: name 'steps_per_epoch' is not defined

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"]

test_paths = sample_sub["path"].values

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def _decode_test(path):
    img = _read_and_decode(path)
    img = _preprocess(img)
    return img


test_ds = (
    test_ds.map(_decode_test, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(
    test_ds,
    verbose=1,
)

inv_class_indices = {v: k for k, v in class_indices.items()}  # idx -> disease_str
disease_to_labelid = {
    str(v): int(k) for k, v in label_to_disease.to_dict().items()
}  # disease_str -> label_id

pred_class_idx = np.argmax(probs, axis=1).astype(int)
pred_disease = [inv_class_indices[i] for i in pred_class_idx]
pred_labels = np.array([disease_to_labelid[str(d)] for d in pred_disease], dtype=int)

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": pred_labels,
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print(
    "Label distribution:\n",
    submission_df["label"].value_counts(dropna=False).sort_index(),
)
