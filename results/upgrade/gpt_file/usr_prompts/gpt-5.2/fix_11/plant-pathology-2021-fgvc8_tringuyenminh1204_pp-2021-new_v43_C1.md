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

0.7448199445983383

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os, re, math, random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow imported successfully")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE, "train.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE, "train_images")
TEST_IMG_DIR = os.path.join(BASE, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

print(train_df.head())
print(sub_df.head())
print("train rows:", len(train_df), "test rows:", len(sub_df))



## === cell 2
IMAGE_SIZE = (512, 512)

IMAGE_SIZE0 = tf.constant(IMAGE_SIZE[0], tf.int32)
IMAGE_SIZE1 = tf.constant(IMAGE_SIZE[1], tf.int32)


@tf.function
def _decode_and_resize(bits, image_size0, image_size1):
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(
        image, (image_size0, image_size1), method=tf.image.ResizeMethod.BILINEAR
    )
    image = tf.ensure_shape(image, (image_size0, image_size1, 3))
    return image


def decode_image(filename, label=None, image_size=IMAGE_SIZE):
    bits = tf.io.read_file(filename)
    image = _decode_and_resize(bits, IMAGE_SIZE0, IMAGE_SIZE1)
    if label is None:
        return image
    return image, label




## === cell 3
BATCH_SIZE = 32
AUTO = tf.data.AUTOTUNE
SEED = 1337

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(
        max(2, (os.cpu_count() or 2) // 2)
    )
except Exception as e:
    print("Threading config skipped:", repr(e))

print("Global XLA JIT: not explicitly enabled (model jit_compile=True is used)")



## === cell 4
test_images = sub_df["image"].astype(str).tolist()
IMAGE_PATHS = [os.path.join(TEST_IMG_DIR, fn) for fn in test_images]

sample_check = IMAGE_PATHS[: min(10, len(IMAGE_PATHS))]
missing = [p for p in sample_check if not os.path.exists(p)]
print("Missing test images (sample check):", len(missing))



## === cell 5
data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

try:
    data_opts.threading.private_threadpool_size = min(32, max(8, (os.cpu_count() or 8)))
    data_opts.threading.max_intra_op_parallelism = 1
except Exception:
    pass

try:
    data_opts.experimental_optimization.map_and_batch_fusion = True
    data_opts.experimental_optimization.parallel_batch = True
    data_opts.experimental_optimization.autotune_buffers = True
except Exception:
    pass

try:
    data_opts.experimental_slack = True
except Exception:
    pass

test_dataset = (
    tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
    .with_options(data_opts)
    .map(lambda fn: decode_image(fn), num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031965684.py in <cell line: 0>()
     25     tf.data.Dataset.from_tensor_slices(IMAGE_PATHS)
     26     .with_options(data_opts)
---> 27     .map(lambda fn: decode_image(fn), num_parallel_calls=AUTO, deterministic=True)
     28     .batch(BATCH_SIZE, drop_remainder=False)
     29     .prefetch(AUTO)

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

/tmp/__autograph_generated_file9brjdnzi.py in <lambda>(fn)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda fn: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (fn,), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file9brjdnzi.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda fn: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (fn,), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filew5l4d7_a.py in tf__decode_image(filename, label, image_size)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
---> 11                 image = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(bits), ag__.ld(IMAGE_SIZE0), ag__.ld(IMAGE_SIZE1)), None, fscope)
     12 
     13                 def get_state():

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

/tmp/__autograph_generated_filec4fxaejm.py in tf___decode_and_resize(bits, image_size0, image_size1)
     11                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope) / 255.0
     12                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), (ag__.ld(image_size0), ag__.ld(image_size1))), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
---> 13                 image = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(image), (ag__.ld(image_size0), ag__.ld(image_size1), 3)), None, fscope)
     14                 try:
     15                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_11/4031965684.py", line 27, in None  *
        lambda fn: decode_image(fn)
    File "/tmp/ipykernel_11/2815323740.py", line 20, in decode_image  *
        image = _decode_and_resize(bits, IMAGE_SIZE0, IMAGE_SIZE1)
    File "/tmp/ipykernel_11/2815323740.py", line 14, in _decode_and_resize  *
        image = tf.ensure_shape(image, (image_size0, image_size1, 3))

    TypeError: Dimension value must be integer or None or have an __index__ method, got value '<tf.Tensor 'image_size0:0' shape=() dtype=int32>' with type '<class 'tensorflow.python.framework.ops.SymbolicTensor'>'


## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}


def labels_to_vec(label_str):
    vec = np.zeros(len(CLASSES), dtype=np.float32)
    toks = str(label_str).split()
    for tok in toks:
        j = class_to_idx.get(tok)
        if j is not None:
            vec[j] = 1.0
    return vec


train_paths = [
    os.path.join(TRAIN_IMG_DIR, fn) for fn in train_df["image"].astype(str).tolist()
]
y = np.stack([labels_to_vec(s) for s in train_df["labels"].tolist()], axis=0)

print("Train paths:", len(train_paths), "y shape:", y.shape)



## === cell 7
idx = np.arange(len(train_paths))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
n_val = int(len(idx) * val_frac)
val_idx = idx[:n_val]
trn_idx = idx[n_val:]

trn_paths = np.asarray(train_paths, dtype=object)[trn_idx].tolist()
val_paths = np.asarray(train_paths, dtype=object)[val_idx].tolist()
y_trn = y[trn_idx]
y_val = y[val_idx]


def make_ds(paths, labels=None, training=False, cache_name=None):
    if labels is not None:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(
            lambda fn, lab: decode_image(fn, lab),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda fn: decode_image(fn), num_parallel_calls=AUTO, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_name:
        ds = ds.cache(cache_name)
    else:
        ds = ds.cache()

    ds = ds.prefetch(AUTO)
    return ds.with_options(data_opts)


train_ds = make_ds(
    trn_paths,
    y_trn,
    training=True,
    cache_name="/kaggle/working/cache_train_paths_labels.tfdata",
)
val_ds = make_ds(
    val_paths,
    y_val,
    training=False,
    cache_name="/kaggle/working/cache_val_paths_labels.tfdata",
)

steps_per_epoch = int(math.ceil(len(trn_paths) / BATCH_SIZE))
validation_steps = int(math.ceil(len(val_paths) / BATCH_SIZE))
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2197824034.py in <cell line: 0>()
     43 
     44 
---> 45 train_ds = make_ds(
     46     trn_paths,
     47     y_trn,

/tmp/ipykernel_11/2197824034.py in make_ds(paths, labels, training, cache_name)
     19         if training:
     20             ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
---> 21         ds = ds.map(
     22             lambda fn, lab: decode_image(fn, lab),
     23             num_parallel_calls=AUTO,

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

/tmp/__autograph_generated_filelvs7pnh0.py in <lambda>(fn, lab)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda fn, lab: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (fn, lab), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filelvs7pnh0.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda fn, lab: ag__.with_function_scope(lambda lscope: ag__.converted_call(decode_image, (fn, lab), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filew5l4d7_a.py in tf__decode_image(filename, label, image_size)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
---> 11                 image = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(bits), ag__.ld(IMAGE_SIZE0), ag__.ld(IMAGE_SIZE1)), None, fscope)
     12 
     13                 def get_state():

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

/tmp/__autograph_generated_filec4fxaejm.py in tf___decode_and_resize(bits, image_size0, image_size1)
     11                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope) / 255.0
     12                 image = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(image), (ag__.ld(image_size0), ag__.ld(image_size1))), dict(method=ag__.ld(tf).image.ResizeMethod.BILINEAR), fscope)
---> 13                 image = ag__.converted_call(ag__.ld(tf).ensure_shape, (ag__.ld(image), (ag__.ld(image_size0), ag__.ld(image_size1), 3)), None, fscope)
     14                 try:
     15                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_11/2197824034.py", line 22, in None  *
        lambda fn, lab: decode_image(fn, lab)
    File "/tmp/ipykernel_11/2815323740.py", line 20, in decode_image  *
        image = _decode_and_resize(bits, IMAGE_SIZE0, IMAGE_SIZE1)
    File "/tmp/ipykernel_11/2815323740.py", line 14, in _decode_and_resize  *
        image = tf.ensure_shape(image, (image_size0, image_size1, 3))

    TypeError: Dimension value must be integer or None or have an __index__ method, got value '<tf.Tensor 'image_size0:0' shape=() dtype=int32>' with type '<class 'tensorflow.python.framework.ops.SymbolicTensor'>'


## === cell 8
base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3),
    pooling="avg",
)
inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = inputs
x = keras.applications.resnet50.preprocess_input(x * 255.0)
x = base(x, training=False)
outputs = layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    jit_compile=True,
    steps_per_execution=16,
)

model.summary()



## === cell 9
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4107178584.py in <cell line: 0>()
      1 EPOCHS = 2
      2 history = model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 10
probs = model.predict(test_dataset, verbose=1)
temp_probs = probs
print("probs shape:", probs.shape)

name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    6: "healthy",
}

threshold = {0: 0.5, 1: 0.5, 2: 0.5, 3: 0.5, 4: 0.5}
threshold2 = {0: 0.3, 1: 0.3, 2: 0.3, 3: 0.3, 4: 0.3}

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
thr2 = np.array([threshold2[i] for i in range(5)], dtype=np.float32)

above_thr = temp_probs[:, :5] > thr[None, :]
above_thr2 = temp_probs[:, :5] > thr2[None, :]
cnt2 = above_thr2.sum(axis=1)

pred_string = []
for row_idx in range(temp_probs.shape[0]):
    hits = np.nonzero(above_thr[row_idx])[0].tolist()
    labels = [name[i] for i in hits]

    if cnt2[row_idx] > 2:
        if 2 not in hits:
            labels.append("complex")

    if not labels:
        pred_string.append(name[6])
    else:
        pred_string.append(" ".join(labels))

print("Example preds:", pred_string[:10])

if len(pred_string) != len(test_images):
    out_df = pd.DataFrame(
        {"image": test_images[: len(pred_string)], "labels": pred_string}
    )
else:
    out_df = pd.DataFrame({"image": test_images, "labels": pred_string})

out_df.to_csv("submission.csv", index=False)
print(out_df.head())
print("Wrote submission.csv with rows:", len(out_df))

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/345995280.py in <cell line: 0>()
----> 1 probs = model.predict(test_dataset, verbose=1)
      2 temp_probs = probs
      3 print("probs shape:", probs.shape)
      4 
      5 name = {

NameError: name 'test_dataset' is not defined
