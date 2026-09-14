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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.4322754254056167

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression

print("tf:", tf.__version__, "keras:", keras.__version__)

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

BATCH_SIZE = 128

tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_inter_op_parallelism_threads(2)

AUTOTUNE = tf.data.AUTOTUNE



## === cell 2
DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
imglist_train = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
print("n_train_images:", len(imglist_train))



## === cell 3
import tensorflow.keras.applications.resnet50 as resnet


def make_image_dataset(paths, batch_size=BATCH_SIZE, cache_path=None):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)

    def _load(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_and_resize(
            img_bytes,
            size=[IMG_HEIGHT, IMG_WIDTH],
            channels=3,
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,  # matches typical TF resize behavior; negligible numerical difference only
        )
        img = tf.cast(img, tf.float32)
        img = resnet.preprocess_input(img)
        return img

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_slack = True
    options.autotune.enabled = True

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(options)
    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache_path is not None:
        ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 4
base = resnet.ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
model_f = keras.Model(inp, x)


@tf.function(reduce_retracing=True)
def _feature_batch(x):
    return model_f(x, training=False)


print("Feature dim:", model_f.output_shape)



## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

all_tokens = training_csv["labels"].astype(str).str.split()
tagnames = np.unique(np.concatenate(all_tokens.values))
print("n_classes:", len(tagnames))
print("classes:", tagnames)

tag_to_idx = {t: i for i, t in enumerate(tagnames)}

train_df = training_csv.copy()
train_df["image_path"] = train_df["image"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))

train_paths = train_df["image_path"].tolist()
missing = sum(0 if os.path.exists(p) else 1 for p in train_paths)
assert missing == 0, f"Missing {missing} training images on disk."

Y_train = np.zeros((len(train_df), len(tagnames)), dtype=np.int8)
for r, toks in enumerate(all_tokens.values):
    if toks:
        Y_train[r, [tag_to_idx[t] for t in toks]] = 1
print("Y_train shape:", Y_train.shape)



## === cell 6
ds_train = make_image_dataset(
    train_paths,
    batch_size=BATCH_SIZE,
    cache_path="/kaggle/working/cache_train_resnet50",
)
Xf_train = model_f.predict(ds_train, verbose=1)
print("Xf_train:", Xf_train.shape)

del ds_train
gc.collect()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/795440613.py in <cell line: 0>()
      1 # Speed: disable disk cache writes; use in-memory cache to avoid recompute within run.
----> 2 ds_train = make_image_dataset(
      3     train_paths,
      4     batch_size=BATCH_SIZE,
      5     cache_path="/kaggle/working/cache_train_resnet50",

/tmp/ipykernel_11/1187479611.py in make_image_dataset(paths, batch_size, cache_path)
     30     ds = tf.data.Dataset.from_tensor_slices(paths)
     31     ds = ds.with_options(options)
---> 32     ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
     33 
     34     # Speed: in-memory cache only (no disk I/O). Keep signature/arg for compatibility.

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

/tmp/__autograph_generated_filezxg2_aem.py in tf___load(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_resize, (ag__.ld(img_bytes),), dict(size=[ag__.ld(IMG_HEIGHT), ag__.ld(IMG_WIDTH)], channels=3, method=ag__.ld(tf).image.ResizeMethod.BILINEAR, antialias=False), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(resnet).preprocess_input, (ag__.ld(img),), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1187479611.py", line 13, in _load  *
        img = tf.image.decode_and_resize(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'decode_and_resize'


## === cell 7
sample_sub = pd.read_csv(SAMPLE_SUB)
sample_sub["image_path"] = sample_sub["image"].map(
    lambda x: os.path.join(TEST_IMG_DIR, x)
)

test_paths = sample_sub["image_path"].tolist()
missing_test = sum(0 if os.path.exists(p) else 1 for p in test_paths)
assert missing_test == 0, f"Missing {missing_test} test images on disk."

ds_test = make_image_dataset(
    test_paths, batch_size=BATCH_SIZE, cache_path="/kaggle/working/cache_test_resnet50"
)
Xf_test = model_f.predict(ds_test, verbose=1)
print("Xf_test:", Xf_test.shape)

del ds_test
gc.collect()



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2620516514.py in <cell line: 0>()
      9 
     10 # Speed: same as train pipeline.
---> 11 ds_test = make_image_dataset(
     12     test_paths, batch_size=BATCH_SIZE, cache_path="/kaggle/working/cache_test_resnet50"
     13 )

/tmp/ipykernel_11/1187479611.py in make_image_dataset(paths, batch_size, cache_path)
     30     ds = tf.data.Dataset.from_tensor_slices(paths)
     31     ds = ds.with_options(options)
---> 32     ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
     33 
     34     # Speed: in-memory cache only (no disk I/O). Keep signature/arg for compatibility.

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

/tmp/__autograph_generated_filezxg2_aem.py in tf___load(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 img = ag__.converted_call(ag__.ld(tf).image.decode_and_resize, (ag__.ld(img_bytes),), dict(size=[ag__.ld(IMG_HEIGHT), ag__.ld(IMG_WIDTH)], channels=3, method=ag__.ld(tf).image.ResizeMethod.BILINEAR, antialias=False), fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope)
     13                 img = ag__.converted_call(ag__.ld(resnet).preprocess_input, (ag__.ld(img),), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1187479611.py", line 13, in _load  *
        img = tf.image.decode_and_resize(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'decode_and_resize'


## === cell 8
scaler = MinMaxScaler(feature_range=(0, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)

print("Scaled shapes:", trainXn.shape, testXn_test.shape)

del Xf_train, Xf_test
gc.collect()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1527876044.py in <cell line: 0>()
      1 scaler = MinMaxScaler(feature_range=(0, 1))
----> 2 trainXn = scaler.fit_transform(Xf_train)
      3 testXn_test = scaler.transform(Xf_test)
      4 
      5 print("Scaled shapes:", trainXn.shape, testXn_test.shape)

NameError: name 'Xf_train' is not defined

## === cell 9
from sklearn.multiclass import OneVsRestClassifier

base_lr = LogisticRegression(
    solver="saga",
    penalty="l2",
    max_iter=1000,
    random_state=SEED,
    class_weight="balanced",
    n_jobs=1,  # avoid nested parallelism; OVR handles parallelism via OneVsRestClassifier(n_jobs=...)
)

n_jobs = max(1, (os.cpu_count() or 2) - 1)
clf = OneVsRestClassifier(base_lr, n_jobs=n_jobs)
clf.fit(trainXn, Y_train)

del Y_train
gc.collect()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2311557930.py in <cell line: 0>()
     15 n_jobs = max(1, (os.cpu_count() or 2) - 1)
     16 clf = OneVsRestClassifier(base_lr, n_jobs=n_jobs)
---> 17 clf.fit(trainXn, Y_train)
     18 
     19 del Y_train

NameError: name 'trainXn' is not defined

## === cell 10
testKaggle_ppredscore1 = clf.predict_proba(testXn_test).astype(np.float32, copy=False)
print("Pred score matrix:", testKaggle_ppredscore1.shape)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3571518308.py in <cell line: 0>()
----> 1 testKaggle_ppredscore1 = clf.predict_proba(testXn_test).astype(np.float32, copy=False)
      2 print("Pred score matrix:", testKaggle_ppredscore1.shape)
      3 
      4 

NameError: name 'testXn_test' is not defined

## === cell 11
def class2tags_boolmat(bool_mat, tagnames):
    out = []
    for row in bool_mat:
        idx = np.flatnonzero(row)
        out.append(" ".join(tagnames[idx]))
    return out


test_predclass = testKaggle_ppredscore1 > 0.5
test_predtags = class2tags_boolmat(test_predclass, tagnames)

empty_mask = np.fromiter(
    (t == "" for t in test_predtags), count=len(test_predtags), dtype=bool
)
if empty_mask.any():
    argm = np.argmax(testKaggle_ppredscore1[empty_mask], axis=1)
    fill = tagnames[argm]
    it = iter(fill.tolist())
    for i, is_empty in enumerate(empty_mask.tolist()):
        if is_empty:
            test_predtags[i] = next(it)

print("Example preds:", test_predtags[:5])

submission = pd.DataFrame(
    {"image": sample_sub["image"].values, "labels": test_predtags}
)
print(submission.head())
print(submission.shape)

out_path = "./submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "size:", os.path.getsize(out_path), "bytes")

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2234842163.py in <cell line: 0>()
      8 
      9 
---> 10 test_predclass = testKaggle_ppredscore1 > 0.5
     11 test_predtags = class2tags_boolmat(test_predclass, tagnames)
     12 

NameError: name 'testKaggle_ppredscore1' is not defined
