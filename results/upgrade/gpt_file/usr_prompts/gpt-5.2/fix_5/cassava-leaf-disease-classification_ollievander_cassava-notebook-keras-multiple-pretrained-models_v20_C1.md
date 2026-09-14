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

0.8445149592021759

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

INPUT_DIR = "../input/cassava-leaf-disease-classification/"
OUTPUT_DIR = "./"
os.makedirs(OUTPUT_DIR, exist_ok=True)

TRAIN_PATH = os.path.join(INPUT_DIR, "train_images")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

print("INPUT_DIR exists:", os.path.exists(INPUT_DIR))
print("TRAIN_PATH exists:", os.path.exists(TRAIN_PATH))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 1
import json
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.optimizers import Adam

AUTOTUNE = tf.data.AUTOTUNE

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
print(train.shape)
train.head()



## === cell 3
with open(os.path.join(INPUT_DIR, "label_num_to_disease_map.json")) as f:
    classes = json.load(f)

classes



## === cell 4
train["class"] = train["label"].apply(lambda x: classes[str(x)])
train[["image_id", "label", "class"]].head()



## === cell 5
print("Class distribution:\n", train["class"].value_counts())



## === cell 6
train["path"] = train["image_id"].apply(lambda x: os.path.join(TRAIN_PATH, str(x)))
train = train.astype({"image_id": "str", "label": "str", "class": "str", "path": "str"})


def stratified_split_df(df, label_col, test_size=0.05, seed=100):
    rng = np.random.RandomState(seed)
    parts = []
    val_parts = []
    for label, g in df.groupby(label_col):
        idx = g.index.values.copy()
        rng.shuffle(idx)
        n_val = int(np.ceil(len(idx) * test_size))
        val_idx = idx[:n_val]
        train_idx = idx[n_val:]
        val_parts.append(df.loc[val_idx])
        parts.append(df.loc[train_idx])
    train_df = (
        pd.concat(parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    val_df = (
        pd.concat(val_parts, axis=0)
        .sample(frac=1.0, random_state=seed)
        .reset_index(drop=True)
    )
    return train_df, val_df


train, val = stratified_split_df(train, "label", test_size=0.05, seed=100)

print("train:", train.shape, "val:", val.shape)
train.head()



## === cell 7
IMG_SIZE = (512, 512)
NUM_CLASSES = train["label"].nunique()

batch_size = 16

class_names = sorted(train["label"].unique().tolist())
class_to_index = {c: i for i, c in enumerate(class_names)}
index_to_class = {i: c for c, i in class_to_index.items()}

train_paths = train["path"].values
train_labels = train["label"].map(class_to_index).astype(np.int32).values

val_paths = val["path"].values
val_labels = val["label"].map(class_to_index).astype(np.int32).values


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # matches JPEG dataset
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0  # matches rescale=1/255
    return img


ROT_RAD = 40.0 * np.pi / 180.0
SHEAR = 0.1
ZOOM_RANGE = 0.2
SHIFT_FRAC = 0.1


def _augment(img, seed2):
    s0 = seed2
    img = tf.image.stateless_random_flip_left_right(img, seed=s0)
    img = tf.image.stateless_random_rotation(
        img,
        seed=s0 + tf.constant([1, 0], tf.int32),
        factor=ROT_RAD / (2.0 * np.pi),
        fill_mode="nearest",
    )
    z = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([2, 0], tf.int32),
        minval=1.0 - ZOOM_RANGE,
        maxval=1.0 + ZOOM_RANGE,
    )
    sh = tf.random.stateless_uniform(
        [], seed=s0 + tf.constant([3, 0], tf.int32), minval=-SHEAR, maxval=SHEAR
    )
    tx = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([4, 0], tf.int32),
        minval=-SHIFT_FRAC,
        maxval=SHIFT_FRAC,
    ) * float(IMG_SIZE[1])
    ty = tf.random.stateless_uniform(
        [],
        seed=s0 + tf.constant([5, 0], tf.int32),
        minval=-SHIFT_FRAC,
        maxval=SHIFT_FRAC,
    ) * float(IMG_SIZE[0])

    cx = (IMG_SIZE[1] - 1) / 2.0
    cy = (IMG_SIZE[0] - 1) / 2.0

    def matmul(a, b):
        return tf.linalg.matmul(a, b)

    T1 = tf.convert_to_tensor(
        [[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]], tf.float32
    )
    T2 = tf.convert_to_tensor(
        [[1.0, 0.0, cx + tx], [0.0, 1.0, cy + ty], [0.0, 0.0, 1.0]], tf.float32
    )
    Zm = tf.convert_to_tensor(
        [[z, 0.0, 0.0], [0.0, z, 0.0], [0.0, 0.0, 1.0]], tf.float32
    )
    Shm = tf.convert_to_tensor(
        [[1.0, sh, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]], tf.float32
    )

    M = matmul(T2, matmul(Shm, matmul(Zm, T1)))

    Minv = tf.linalg.inv(M)
    a0 = Minv[0, 0]
    a1 = Minv[0, 1]
    a2 = Minv[0, 2]
    b0 = Minv[1, 0]
    b1 = Minv[1, 1]
    b2 = Minv[1, 2]
    c0 = Minv[2, 0]
    c1 = Minv[2, 1]
    transform = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1], axis=0)

    img = tf.keras.ops.image.transform(
        img,
        transform=transform,
        interpolation="bilinear",
        fill_mode="nearest",
        fill_value=0.0,
        output_shape=IMG_SIZE,
    )
    return img


def _make_ds(paths, labels=None, training=False, cache=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    if training:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=True
        )

    if labels is None:

        def map_fn(path, idx):
            img = _decode_resize(path)
            if training:
                img = _augment(
                    img, tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
                )
            return img

        ds = ds.enumerate()
        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:

        def map_fn(path, y, idx):
            img = _decode_resize(path)
            if training:
                img = _augment(
                    img, tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
                )
            y = tf.one_hot(y, depth=NUM_CLASSES, dtype=tf.float32)
            return img, y

        ds = ds.enumerate()
        ds = ds.map(
            lambda idx, xy: map_fn(xy[0], xy[1], idx),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    if cache:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


datagen = _make_ds(train_paths, train_labels, training=True, cache=False)
val_datagen = _make_ds(val_paths, val_labels, training=False, cache=True)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1039257009.py in <cell line: 0>()
    171 
    172 
--> 173 datagen = _make_ds(train_paths, train_labels, training=True, cache=False)
    174 val_datagen = _make_ds(val_paths, val_labels, training=False, cache=True)
    175 

/tmp/ipykernel_11/1039257009.py in _make_ds(paths, labels, training, cache)
    158 
    159         ds = ds.enumerate()
--> 160         ds = ds.map(
    161             lambda idx, xy: map_fn(xy[0], xy[1], idx),
    162             num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filekn163f9t.py in <lambda>(idx, xy)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda idx, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(map_fn, (xy[0], xy[1], idx), None, lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filekn163f9t.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda idx, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(map_fn, (xy[0], xy[1], idx), None, lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filexkaux8cf.py in tf__map_fn(path, y, idx)
     25                     nonlocal img
     26                     pass
---> 27                 ag__.if_stmt(ag__.ld(training), if_body, else_body, get_state, set_state, ('img',), 1)
     28                 y = ag__.converted_call(ag__.ld(tf).one_hot, (ag__.ld(y),), dict(depth=ag__.ld(NUM_CLASSES), dtype=ag__.ld(tf).float32), fscope)
     29                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filexkaux8cf.py in if_body()
     20                 def if_body():
     21                     nonlocal img
---> 22                     img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope), ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(idx), ag__.ld(tf).int32), None, fscope)],), None, fscope)), None, fscope)
     23 
     24                 def else_body():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_files7mzzk6i.py in tf___augment(img, seed2)
     10                 s0 = ag__.ld(seed2)
     11                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(s0)), fscope)
---> 12                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_rotation, (ag__.ld(img),), dict(seed=ag__.ld(s0) + ag__.converted_call(ag__.ld(tf).constant, ([1, 0], ag__.ld(tf).int32), None, fscope), factor=ag__.ld(ROT_RAD) / (2.0 * ag__.ld(np).pi), fill_mode='nearest'), fscope)
     13                 z = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(s0) + ag__.converted_call(ag__.ld(tf).constant, ([2, 0], ag__.ld(tf).int32), None, fscope), minval=1.0 - ag__.ld(ZOOM_RANGE), maxval=1.0 + ag__.ld(ZOOM_RANGE)), fscope)
     14                 sh = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([],), dict(seed=ag__.ld(s0) + ag__.converted_call(ag__.ld(tf).constant, ([3, 0], ag__.ld(tf).int32), None, fscope), minval=-ag__.ld(SHEAR), maxval=ag__.ld(SHEAR)), fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1039257009.py", line 161, in None  *
        lambda idx, xy: map_fn(xy[0], xy[1], idx)
    File "/tmp/ipykernel_11/1039257009.py", line 153, in map_fn  *
        img = _augment(
    File "/tmp/ipykernel_11/1039257009.py", line 47, in _augment  *
        img = tf.image.stateless_random_rotation(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'stateless_random_rotation'


## === cell 8
train_steps = int(np.ceil(len(train_paths) / batch_size))
val_steps = int(np.ceil(len(val_paths) / batch_size))
print("Train batches:", train_steps, "Val batches:", val_steps)
print("Class indices (generator):", class_to_index)



## === cell 9
sample_sub = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))
print(sample_sub.shape)
sample_sub.head()



## === cell 10
test_images = sample_sub["image_id"].astype(str).tolist()
df_test = pd.DataFrame({"image_id": test_images})
df_test["path"] = df_test["image_id"].apply(lambda x: os.path.join(TEST_PATH, str(x)))

missing = (~df_test["path"].apply(os.path.exists)).sum()
print("Missing test files:", int(missing))
df_test.head()



## === cell 11
test_paths = df_test["path"].values
test_gen2 = _make_ds(test_paths, labels=None, training=False, cache=True)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2482691998.py in <cell line: 0>()
      1 # --- Speed: build cached/prefetched tf.data test pipeline to accelerate predict (no label, keep order).
      2 test_paths = df_test["path"].values
----> 3 test_gen2 = _make_ds(test_paths, labels=None, training=False, cache=True)
      4 

/tmp/ipykernel_11/1039257009.py in _make_ds(paths, labels, training, cache)
    145 
    146         ds = ds.enumerate()
--> 147         ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    148     else:
    149 

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

/tmp/__autograph_generated_file9yo8u4_g.py in tf__map_fn(path, idx)
      9                 do_return = False
     10                 retval_ = ag__.UndefinedReturnValue()
---> 11                 img = ag__.converted_call(ag__.ld(_decode_resize), (ag__.ld(path),), None, fscope)
     12 
     13                 def get_state():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file9rvjqwim.py in tf___decode_resize(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(img),), dict(channels=3), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(IMG_SIZE)), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    329   if conversion.is_in_allowlist_cache(f, options):
    330     logging.log(2, 'Allowlisted %s: from cache', f)
--> 331     return _call_unconverted(f, args, kwargs, options, False)
    332 
    333   if ag_ctx.control_status_ctx().status == ag_ctx.Status.DISABLED:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/io_ops.py in read_file(filename, name)
    132     A tensor of dtype "string", with the file contents.
    133   """
--> 134   return gen_io_ops.read_file(filename, name)
    135 
    136 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_io_ops.py in read_file(filename, name)
    586       pass  # Add nodes to the TensorFlow graph.
    587   # Add nodes to the TensorFlow graph.
--> 588   _, _, _op, _outputs = _op_def_library._apply_op_helper(
    589         "ReadFile", filename=filename, name=name)
    590   _result = _outputs[:]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _apply_op_helper(op_type_name, name, **keywords)
    776   with g.as_default(), ops.name_scope(name) as scope:
    777     if fallback:
--> 778       _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map,
    779                              keywords, default_type_attr_map, attrs, inputs,
    780                              input_types)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/op_def_library.py in _ExtractInputsAndAttrs(op_type_name, op_def, allowed_list_attr_map, keywords, default_type_attr_map, attrs, inputs, input_types)
    576                   (input_name, op_type_name, observed))
    577         if input_arg.type != types_pb2.DT_INVALID:
--> 578           raise TypeError(f"{prefix} expected type of "
    579                           f"{dtypes.as_dtype(input_arg.type).name}.")
    580         else:

TypeError: in user code:

    File "/tmp/ipykernel_11/1039257009.py", line 139, in map_fn  *
        img = _decode_resize(path)
    File "/tmp/ipykernel_11/1039257009.py", line 24, in _decode_resize  *
        img = tf.io.read_file(path)

    TypeError: Input 'filename' of 'ReadFile' Op has type int64 that does not match expected type of string.


## === cell 12
num_classes = NUM_CLASSES

inputs = tf.keras.Input(shape=(512, 512, 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)
model.summary()

history = model.fit(
    datagen,
    validation_data=val_datagen,
    epochs=3,
    verbose=1,
)

model2 = None  # keep variable for downstream ensemble code compatibility



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3924021819.py in <cell line: 0>()
     20 
     21 history = model.fit(
---> 22     datagen,
     23     validation_data=val_datagen,
     24     epochs=3,

NameError: name 'datagen' is not defined

## === cell 13
preds = []
tta = 3  # keep TTA semantics but limit to keep runtime safe

for i in range(tta):
    p1 = model.predict(test_gen2, verbose=0)
    if model2 is not None:
        p2 = model2.predict(test_gen2, verbose=0)
        preds.append(p1 + p2)
    else:
        preds.append(p1)

predbis = np.mean(preds, axis=0)

idx_to_class = {v: k for k, v in class_to_index.items()}  # index -> string label
pred_idx = np.argmax(predbis, axis=-1).astype(int)
predictions = np.array([int(idx_to_class[i]) for i in pred_idx], dtype=int)

print("predbis shape:", predbis.shape, "predictions shape:", predictions.shape)
print("Unique predicted labels:", np.unique(predictions))



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2987237782.py in <cell line: 0>()
      5 # --- Reusing the cached test dataset preserves ordering and improves throughput.
      6 for i in range(tta):
----> 7     p1 = model.predict(test_gen2, verbose=0)
      8     if model2 is not None:
      9         p2 = model2.predict(test_gen2, verbose=0)

NameError: name 'test_gen2' is not defined

## === cell 14
submission = pd.DataFrame({"image_id": test_images, "label": predictions})

assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission"
assert list(submission.columns) == ["image_id", "label"], "Wrong submission columns"

sub_path = os.path.join(OUTPUT_DIR, "submission.csv")
submission.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
submission.head()



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2282794080.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_images, "label": predictions})
      2 
      3 assert (
      4     submission.shape[0] == sample_sub.shape[0]
      5 ), "Row count mismatch vs sample_submission"

NameError: name 'predictions' is not defined

## === cell 15
submission.tail()

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1163902920.py in <cell line: 0>()
----> 1 submission.tail()

NameError: name 'submission' is not defined
