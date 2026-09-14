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

2.7

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

0.8856149894227864

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'The changes keep the exact model architecture and training schedule but speed up data loading and reduce the number of steps per epoch by using a larger batch size and parallel workers. `ImageDataGenerator` now loads images with 4 processes, and both training and prediction use a batch size of 64, which roughly halves the iteration count while preserving the same learning dynamics. All other logic, labels, augmentations, and fine‑tuning stages remain unchanged, so the final predictions stay identical apart from negligible floating‑point differences.'
- What this solution (achieved 0.11024) has done: 'The changes enable TensorFlow to use multiple CPU threads and parallel data loading, which removes the main bottleneck of image preprocessing and model training without altering the model architecture, epochs, or augmentation logic. By configuring `workers` and `use_multiprocessing` in `model.fit`, the data generator runs in parallel, significantly speeding up each epoch while keeping the same training behavior. Threading settings are also applied early to let TensorFlow fully utilize available cores.'
- What this solution (achieved 0.61099) has done: 'The changes increase the data pipeline and training throughput: the batch size is doubled (128 → more images per step) and multi‑process data loading is enabled in `model.fit` (workers = 4, `use_multiprocessing=True`). This cuts the number of optimization steps roughly in half while keeping the same number of epochs, model architecture, and accuracy‑related logic unchanged, allowing the whole run to finish well under the 600‑second limit.'
- What this solution (achieved 0.11024) has done: 'The changes add conditional protobuf installation to avoid unnecessary re‑install time, enable multi‑process data loading for the ImageDataGenerator pipelines, and increase the number of workers used during model fitting. These adjustments keep the exact model architecture, training schedule, and evaluation logic intact while eliminating the main I/O bottleneck, allowing the script to complete well within the 600‑second limit.'

# 9. Code solution

## === cell 0
import subprocess, sys, os

try:
    import google.protobuf  # noqa: F401
    from pkg_resources import get_distribution

    if get_distribution("protobuf").version != "3.20.3":
        raise ImportError
except Exception:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "protobuf==3.20.3", "--quiet"]
    )

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras import layers, models, optimizers
    from tensorflow.keras.applications import EfficientNetB0
except Exception as e:
    raise ImportError(f"TensorFlow import failed: {e}")

tf.random.set_seed(42)
np.random.seed(42)

tf.config.threading.set_intra_op_parallelism_threads(8)
tf.config.threading.set_inter_op_parallelism_threads(8)




## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")
TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "sample_submission.csv")  # only for column order

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(int)

train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)
split_idx = int(0.9 * len(train_df))
train_split = train_df.iloc[:split_idx]
val_split = train_df.iloc[split_idx:]


def build_dataset(df, augment=False):
    paths = tf.strings.join([TRAIN_IMG_DIR, "/", df["image_id"]])
    labels = tf.convert_to_tensor(df["label"].values, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_image(path, label):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [224, 224])
        img = img / 255.0
        return img, tf.one_hot(label, 5)

    ds = ds.map(_load_image, num_parallel_calls=tf.data.AUTOTUNE)

    if augment:
        ds = ds.map(
            lambda x, y: (tf.image.random_flip_left_right(x), y),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda x, y: (
                tf.keras.preprocessing.image.random_rotation(
                    x, 20, row_axis=0, col_axis=1, channel_axis=2
                ),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda x, y: (
                tf.keras.preprocessing.image.random_shift(
                    x, 0.1, 0.1, row_axis=0, col_axis=1, channel_axis=2
                ),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.map(
            lambda x, y: (
                tf.keras.preprocessing.image.random_zoom(
                    x, (0.9, 1.1), row_axis=0, col_axis=1, channel_axis=2
                ),
                y,
            ),
            num_parallel_calls=tf.data.AUTOTUNE,
        )

    ds = ds.shuffle(1024).batch(64).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = build_dataset(train_split, augment=True)
val_dataset = build_dataset(val_split, augment=False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2317360874.py in <cell line: 0>()
     71 
     72 
---> 73 train_dataset = build_dataset(train_split, augment=True)
     74 val_dataset = build_dataset(val_split, augment=False)
     75 

/tmp/ipykernel_11/2317360874.py in build_dataset(df, augment)
     39             num_parallel_calls=tf.data.AUTOTUNE,
     40         )
---> 41         ds = ds.map(
     42             lambda x, y: (
     43                 tf.keras.preprocessing.image.random_rotation(

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

/tmp/__autograph_generated_filewroagw9h.py in <lambda>(x, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(tf.keras.preprocessing.image.random_rotation, (x, 20), dict(row_axis=0, col_axis=1, channel_axis=2), lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filewroagw9h.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda x, y: ag__.with_function_scope(lambda lscope: (ag__.converted_call(tf.keras.preprocessing.image.random_rotation, (x, 20), dict(row_axis=0, col_axis=1, channel_axis=2), lscope), y), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

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

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in random_rotation(x, rg, row_axis, col_axis, channel_axis, fill_mode, cval, interpolation_order)
   1561     """DEPRECATED."""
   1562     theta = np.random.uniform(-rg, rg)
-> 1563     x = apply_affine_transform(
   1564         x,
   1565         theta=theta,

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py in apply_affine_transform(x, theta, tx, ty, shear, zx, zy, row_axis, col_axis, channel_axis, fill_mode, cval, order)
   1860             transform_matrix, h, w
   1861         )
-> 1862         x = np.rollaxis(x, channel_axis, 0)
   1863 
   1864         # Matrix construction assumes that coordinates are x, y (in that order).

/usr/local/lib/python3.11/dist-packages/numpy/core/numeric.py in rollaxis(a, axis, start)
   1325     axes.remove(axis)
   1326     axes.insert(start, axis)
-> 1327     return a.transpose(axes)
   1328 
   1329 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor.py in __getattr__(self, name)
    253                 "tolist", "data"}:
    254       # TODO(wangpeng): Export the enable_numpy_behavior knob
--> 255       raise AttributeError(
    256           f"{type(self).__name__} object has no attribute '{name}'. " + """
    257         If you are looking for numpy-related methods, please run the following:

AttributeError: in user code:

    File "/tmp/ipykernel_11/2317360874.py", line 42, in None  *
        )
    File "/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py", line 1563, in random_rotation  **
        x = apply_affine_transform(
    File "/usr/local/lib/python3.11/dist-packages/keras/src/legacy/preprocessing/image.py", line 1862, in apply_affine_transform
        x = np.rollaxis(x, channel_axis, 0)
    File "/usr/local/lib/python3.11/dist-packages/numpy/core/numeric.py", line 1327, in rollaxis
        return a.transpose(axes)

    AttributeError: SymbolicTensor object has no attribute 'transpose'. 
            If you are looking for numpy-related methods, please run the following:
            tf.experimental.numpy.experimental_enable_numpy_behavior()
          


## === cell 2
base_model = EfficientNetB0(
    weights="imagenet", include_top=False, input_shape=(224, 224, 3)
)
base_model.trainable = False  # freeze for initial phase

inputs = layers.Input(shape=(224, 224, 3))
x = layers.RandomFlip(mode="horizontal")(inputs)
x = layers.RandomRotation(factor=0.1111)(x)  # 20 degrees ≈ 0.111 rad
x = layers.RandomZoom(height_factor=0.1, width_factor=0.1)(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(5, activation="softmax")(x)

model = models.Model(inputs, outputs)

model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset,
    epochs=25,
    validation_data=val_dataset,
    verbose=1,
)

base_model.trainable = True
model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/495205177.py in <cell line: 0>()
     25 # Phase‑1 training (freeze base)
     26 model.fit(
---> 27     train_dataset,
     28     epochs=25,
     29     validation_data=val_dataset,

NameError: name 'train_dataset' is not defined

## === cell 3
test_df = pd.read_csv(TEST_CSV)[["image_id"]].copy()
test_paths = tf.strings.join([TEST_IMG_DIR, "/", test_df["image_id"]])


def _load_test_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [224, 224])
    img = img / 255.0
    return img


test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = test_dataset.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(64).prefetch(tf.data.AUTOTUNE)

preds = model.predict(test_dataset, verbose=1)
pred_labels_idx = np.argmax(preds, axis=1)

idx_to_label = {
    v: k for k, v in train_dataset.element_spec[1].numpy().astype(int).tolist()
}
pred_labels = pred_labels_idx.astype(int).tolist()




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/264297976.py in <cell line: 0>()
     21 # Map back to original label integers (same as training indices)
     22 idx_to_label = {
---> 23     v: k for k, v in train_dataset.element_spec[1].numpy().astype(int).tolist()
     24 }
     25 # Since we used one‑hot with order 0‑4, idx == label

NameError: name 'train_dataset' is not defined

## === cell 4
submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/826213746.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"image_id": test_df["image_id"], "label": pred_labels})
      2 submission_path = "/kaggle/working/submission.csv"
      3 submission.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

NameError: name 'pred_labels' is not defined
