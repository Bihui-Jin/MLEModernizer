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

0.2095106186518933

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, random, math, re
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "sample_submission.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, test.shape)



## === cell 2
AUTO = tf.data.AUTOTUNE



## === cell 3
pass



## === cell 4
import pathlib



## === cell 5
train_dir = os.path.join(path, "train_images")
test_dir = os.path.join(path, "test_images")

train_images = train["image"].to_numpy()
test_images = sub["image"].to_numpy()

train_paths = [os.path.join(train_dir, fn) for fn in train_images]
test_paths = [os.path.join(test_dir, fn) for fn in test_images]

assert len(train_paths) == len(train)
assert len(test_paths) == len(sub)



## === cell 6
all_classes = sorted({c for s in train["labels"].astype(str).values for c in s.split()})
print("Classes:", all_classes)



## === cell 7
labels_split = train["labels"].astype(str).str.get_dummies(sep=" ")
labels_split = labels_split.reindex(columns=all_classes, fill_value=0).astype(
    np.float32
)

new_train = labels_split.copy()
new_train.insert(0, "image", train["image"].values)



## === cell 8
new_train



## === cell 9
IMAGE_SIZE = (512, 512)
_IMAGE_SIZE_T = tf.constant([IMAGE_SIZE[0], IMAGE_SIZE[1]], dtype=tf.int32)


@tf.function(
    reduce_retracing=True,
    input_signature=[
        tf.TensorSpec([], tf.string),
        tf.TensorSpec([None], tf.float32),  # label vector or dummy
        tf.TensorSpec([], tf.bool),  # has_label
    ],
)
def _decode_image_center_square(filename, label, has_label):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(
        bits, channels=3, dct_method="INTEGER_FAST"
    )  # uint8 [H,W,3]
    image = tf.image.convert_image_dtype(image, tf.float32)  # float32 in [0,1]

    shape = tf.shape(image)
    h = tf.cast(shape[0], tf.float32)
    w = tf.cast(shape[1], tf.float32)
    side = tf.minimum(h, w)

    y1 = (h - side) * 0.5 / h
    x1 = (w - side) * 0.5 / w
    y2 = y1 + (side / h)
    x2 = x1 + (side / w)

    boxes = tf.stack([y1, x1, y2, x2])[tf.newaxis, ...]  # [1,4]
    box_ind = tf.constant([0], dtype=tf.int32)

    image4 = image[tf.newaxis, ...]
    cropped = tf.image.crop_and_resize(
        image4, boxes=boxes, box_ind=box_ind, crop_size=_IMAGE_SIZE_T, method="bilinear"
    )[0]

    return tf.cond(has_label, lambda: (cropped, label), lambda: cropped)


def decode_image(filename, label=None, image_size=(512, 512)):
    if label is None:
        dummy = tf.zeros([1], tf.float32)
        return _decode_image_center_square(filename, dummy, tf.constant(False))
    else:
        return _decode_image_center_square(filename, label, tf.constant(True))




## === cell 10
test_paths[:5], len(test_paths)



## === cell 11
_has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
BATCH_SIZE = 32 if _has_gpu else 16




## === cell 12
def _configure_dataset(ds: tf.data.Dataset, deterministic: bool) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.deterministic = deterministic
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.threading.private_threadpool_size = 16
        opts.threading.max_intra_op_parallelism = 1
    except Exception:
        pass
    try:
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return ds.with_options(opts)


def map_batch_prefetch(
    ds,
    map_fn,
    batch_size,
    deterministic=True,
    cache=None,
    shuffle=None,
    seed=None,
    repeat=False,
):
    if shuffle is not None:
        ds = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(map_fn, num_parallel_calls=AUTO, deterministic=deterministic)

    if cache:
        ds = ds.cache(cache)  # cache decoded/resized tensors to disk for reuse

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    ds = _configure_dataset(ds, deterministic=deterministic)
    return ds




## === cell 13
cache_dir = "/kaggle/working/tf_cache_pp2021"
os.makedirs(cache_dir, exist_ok=True)
test_cache_path = os.path.join(cache_dir, f"test_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.cache")

test_paths_tf = tf.constant(np.asarray(test_paths, dtype=np.str_))
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths_tf)
test_dataset = map_batch_prefetch(
    test_dataset,
    lambda f: decode_image(f, None, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=test_cache_path,
    repeat=False,
)
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2747601744.py in <cell line: 0>()
      7 test_paths_tf = tf.constant(np.asarray(test_paths, dtype=np.str_))
      8 test_dataset = tf.data.Dataset.from_tensor_slices(test_paths_tf)
----> 9 test_dataset = map_batch_prefetch(
     10     test_dataset,
     11     lambda f: decode_image(f, None, image_size=IMAGE_SIZE),

/tmp/ipykernel_10/2865628134.py in map_batch_prefetch(ds, map_fn, batch_size, deterministic, cache, shuffle, seed, repeat)
     32     if shuffle is not None:
     33         ds = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
---> 34     ds = ds.map(map_fn, num_parallel_calls=AUTO, deterministic=deterministic)
     35 
     36     if cache:

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

/tmp/__autograph_generated_filerysa99h8.py in <lambda>(f)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (f, None), dict(image_size=IMAGE_SIZE), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filerysa99h8.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (f, None), dict(image_size=IMAGE_SIZE), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileniiag57e.py in tf__decode_image(filename, label, image_size)
     35                         raise
     36                 dummy = ag__.Undefined('dummy')
---> 37                 ag__.if_stmt(ag__.ld(label) is None, if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     38                 return fscope.ret(retval_, do_return)
     39         return tf__decode_image

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

/tmp/__autograph_generated_fileniiag57e.py in if_body()
     21                     try:
     22                         do_return = True
---> 23                         retval_ = ag__.converted_call(ag__.ld(_decode_image_center_square), (ag__.ld(filename), ag__.ld(dummy), ag__.converted_call(ag__.ld(tf).constant, (False,), None, fscope)), None, fscope)
     24                     except:
     25                         do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    375 
    376   if not options.user_requested and conversion.is_allowlisted(f):
--> 377     return _call_unconverted(f, args, kwargs, options)
    378 
    379   # internal_convert_user_code is for example turned off when issuing a dynamic

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filej1e8qnwq.py in tf___decode_image_center_square(filename, label, has_label)
     22                 box_ind = ag__.converted_call(ag__.ld(tf).constant, ([0],), dict(dtype=ag__.ld(tf).int32), fscope)
     23                 image4 = ag__.ld(image)[ag__.ld(tf).newaxis, ...]
---> 24                 cropped = ag__.converted_call(ag__.ld(tf).image.crop_and_resize, (ag__.ld(image4),), dict(boxes=ag__.ld(boxes), box_ind=ag__.ld(box_ind), crop_size=ag__.ld(_IMAGE_SIZE_T), method='bilinear'), fscope)[0]
     25                 try:
     26                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_10/2747601744.py", line 11, in None  *
        lambda f: decode_image(f, None, image_size=IMAGE_SIZE)
    File "/tmp/ipykernel_10/3309276119.py", line 53, in decode_image  *
        return _decode_image_center_square(filename, dummy, tf.constant(False))
    File "/tmp/ipykernel_10/3309276119.py", line 40, in _decode_image_center_square  *
        cropped = tf.image.crop_and_resize(

    TypeError: Got an unexpected keyword argument 'box_ind'


## === cell 14
import tensorflow as tf
from tensorflow import keras



## === cell 15
NUM_CLASSES = len(all_classes)

idx = np.arange(len(new_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

train_paths_np = np.asarray(train_paths, dtype=np.str_)
x_train = train_paths_np[tr_idx]
y_train = new_train.iloc[tr_idx][all_classes].values.astype(np.float32)

x_val = train_paths_np[va_idx]
y_val = new_train.iloc[va_idx][all_classes].values.astype(np.float32)

train_cache_path = os.path.join(
    cache_dir, f"train_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.cache"
)
val_cache_path = os.path.join(cache_dir, f"val_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}.cache")

train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), y_train))
train_ds = map_batch_prefetch(
    train_ds,
    lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,  # preserve deterministic training order given shuffle seed
    cache=train_cache_path,
    shuffle=2048,
    seed=SEED,
    repeat=True,  # steps_per_epoch controls epoch length
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

val_ds = tf.data.Dataset.from_tensor_slices((tf.constant(x_val), y_val))
val_ds = map_batch_prefetch(
    val_ds,
    lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=val_cache_path,
    shuffle=None,
    seed=None,
    repeat=True,  # validation_steps controls length
)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()

steps_per_epoch = int(math.ceil(len(x_train) / BATCH_SIZE))
validation_steps = int(math.ceil(len(x_val) / BATCH_SIZE))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_10/1984787839.py in <cell line: 0>()
     20 
     21 train_ds = tf.data.Dataset.from_tensor_slices((tf.constant(x_train), y_train))
---> 22 train_ds = map_batch_prefetch(
     23     train_ds,
     24     lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE),

/tmp/ipykernel_10/2865628134.py in map_batch_prefetch(ds, map_fn, batch_size, deterministic, cache, shuffle, seed, repeat)
     32     if shuffle is not None:
     33         ds = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
---> 34     ds = ds.map(map_fn, num_parallel_calls=AUTO, deterministic=deterministic)
     35 
     36     if cache:

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

/tmp/__autograph_generated_filer5map2em.py in <lambda>(f, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f, y: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (f, y), dict(image_size=IMAGE_SIZE), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filer5map2em.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda f, y: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (f, y), dict(image_size=IMAGE_SIZE), lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

/tmp/__autograph_generated_fileniiag57e.py in tf__decode_image(filename, label, image_size)
     35                         raise
     36                 dummy = ag__.Undefined('dummy')
---> 37                 ag__.if_stmt(ag__.ld(label) is None, if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     38                 return fscope.ret(retval_, do_return)
     39         return tf__decode_image

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

/tmp/__autograph_generated_fileniiag57e.py in else_body()
     30                     try:
     31                         do_return = True
---> 32                         retval_ = ag__.converted_call(ag__.ld(_decode_image_center_square), (ag__.ld(filename), ag__.ld(label), ag__.converted_call(ag__.ld(tf).constant, (True,), None, fscope)), None, fscope)
     33                     except:
     34                         do_return = False

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filej1e8qnwq.py in tf___decode_image_center_square(filename, label, has_label)
     22                 box_ind = ag__.converted_call(ag__.ld(tf).constant, ([0],), dict(dtype=ag__.ld(tf).int32), fscope)
     23                 image4 = ag__.ld(image)[ag__.ld(tf).newaxis, ...]
---> 24                 cropped = ag__.converted_call(ag__.ld(tf).image.crop_and_resize, (ag__.ld(image4),), dict(boxes=ag__.ld(boxes), box_ind=ag__.ld(box_ind), crop_size=ag__.ld(_IMAGE_SIZE_T), method='bilinear'), fscope)[0]
     25                 try:
     26                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_10/1984787839.py", line 24, in None  *
        lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE)
    File "/tmp/ipykernel_10/3309276119.py", line 55, in decode_image  *
        return _decode_image_center_square(filename, label, tf.constant(True))
    File "/tmp/ipykernel_10/3309276119.py", line 40, in _decode_image_center_square  *
        cropped = tf.image.crop_and_resize(

    TypeError: Got an unexpected keyword argument 'box_ind'


## === cell 16
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4107178584.py in <cell line: 0>()
      1 EPOCHS = 2
----> 2 history = model.fit(
      3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'model' is not defined

## === cell 17
test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
probs = model.predict(test_dataset, steps=test_steps, verbose=1)
probs.shape



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3017906610.py in <cell line: 0>()
      1 test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
----> 2 probs = model.predict(test_dataset, steps=test_steps, verbose=1)
      3 probs.shape
      4 

NameError: name 'model' is not defined

## === cell 18
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

default_thr = 0.30
threshold_by_class = {
    "scab": 0.20,
    "frog_eye_leaf_spot": 0.30,
    "complex": 0.15,
    "rust": 0.30,
    "powdery_mildew": 0.35,
    "healthy": 0.50,  # only chosen when nothing else triggers; higher reduces false "healthy".
}
thr = np.array(
    [threshold_by_class.get(c, default_thr) for c in all_classes], dtype=np.float32
)

probs_np = np.asarray(probs, dtype=np.float32)

healthy_idx = class_to_idx.get("healthy", None)
non_healthy_mask = np.ones(NUM_CLASSES, dtype=bool)
if healthy_idx is not None:
    non_healthy_mask[healthy_idx] = False

hits_mat = probs_np[:, non_healthy_mask] > thr[non_healthy_mask]

hit_counts = hits_mat.sum(axis=1)
is_complex = hit_counts >= 3
is_healthy = hit_counts == 0  # only if nothing else triggers

non_healthy_classes = np.array([c for c in all_classes if c != "healthy"], dtype=object)

pred_string = []
for i in range(probs_np.shape[0]):
    if is_complex[i]:
        pred_string.append("complex")
    elif is_healthy[i]:
        pred_string.append("healthy")
    else:
        pred_string.append(" ".join(non_healthy_classes[hits_mat[i]].tolist()))

submission = sub.copy()
submission["labels"] = pred_string

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/4069826687.py in <cell line: 0>()
     15 )
     16 
---> 17 probs_np = np.asarray(probs, dtype=np.float32)
     18 
     19 healthy_idx = class_to_idx.get("healthy", None)

NameError: name 'probs' is not defined
