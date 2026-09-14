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

0.7699722991689769

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.2461) has done: 'I remove the failing `tensorflow_addons` import (it triggers the protobuf `MessageFactory.GetPrototype` error in this environment) since it isn’t used by your pipeline. I also fix the missing external model file path by building the same type of lightweight MobileNetV2-based model inside the notebook and running inference end-to-end, so a submission CSV is always produced. Finally, I correct the label post-processing bugs (`==` vs `=`, chained assignment, and the logic that accidentally always triggers) and ensure predictions are aligned with class names from `train.csv`, producing space-delimited labels in the required format.'
- What this solution (achieved 0.289) has done: 'I fix the TensorFlow import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in this environment. Then I fix the training `ValueError: None values not supported` by making the training/validation generators provide labels directly (instead of `y_col=None`) and by passing `valid_generator` normally to `validation_data`. Finally, to move the score upward toward your target without changing the core model/training approach, I train for a few epochs (still lightweight MobileNetV2) and use a slightly more standard multilabel threshold (0.5) with a safe fallback to at least one label.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by slow, single-threaded image loading/resizing inside a Python `Sequence`, plus 512×512 inputs which are expensive for MobileNetV2; we cannot change the latter without changing core logic, so we must remove Python/I/O overhead. I replace the custom `Sequence` with an equivalent `tf.data` pipeline that uses parallel JPEG decode/resize, caching (in-memory) for train/valid, and prefetching, while preserving the same train/valid split, normalization, batching, and shuffle semantics. I also enable deterministic TF execution and set `steps_per_execution` to reduce Python→TF call overhead without changing the training loop semantics. Prediction is likewise switched to `tf.data` with parallelism and prefetch to speed up inference.'
- What this solution (achieved 0.24507) has done: 'The timeout is dominated by repeatedly decoding/resizing large 512×512 JPEGs on-the-fly and by caching the entire training/validation image pipelines in memory (which can thrash/slow down) rather than using an on-disk cache. I keep the exact same model, loss, epochs, and thresholding logic, but make input processing faster and more stable by (1) switching to `decode_and_crop_jpeg` + `central_crop` to reduce decode work before resizing, (2) enabling `tf.data` optimizations like `map` parallelism, `prefetch`, and non-blocking pipeline options while keeping determinism, and (3) changing caching from in-memory to on-disk cache files to avoid RAM pressure and repeated preprocessing across epochs. These changes preserve semantics (still deterministic center-crop + resize + normalization) while reducing wall time and avoiding memory bottlenecks. I also remove a small amount of overhead in shuffle buffer sizing and ensure dataset options are applied consistently.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
print("TF:", tf.__version__)

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
BASE_DIR = "../input/plant-pathology-2021-fgvc8"
if not os.path.exists(os.path.join(BASE_DIR, "train.csv")):
    BASE_DIR = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head())



## === cell 2
h_target = 512
w_target = 512
batch_size = 32

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

print("n_classes:", n_classes)
print("classes:", class_names)



## === cell 3
train_df = train.copy()

train_df["labels_list"] = train_df["labels"].str.split()
Y = mlb.transform(train_df["labels_list"].values).astype("float32")

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_images = train_df.loc[trn_idx, "image"].values
valid_images = train_df.loc[val_idx, "image"].values
Y_train = Y[trn_idx]
Y_valid = Y[val_idx]


@tf.function
def _load_and_preprocess(path, target_size):
    img_bytes = tf.io.read_file(path)

    def _decode_crop():
        return tf.io.decode_and_crop_jpeg(
            img_bytes,
            crop_window=[0.05, 0.05, 0.90, 0.90],
            channels=3,
            fancy_upscaling=False,
        )

    def _decode_full():
        return tf.io.decode_jpeg(img_bytes, channels=3, fancy_upscaling=False)

    img = tf.cond(
        tf.strings.length(img_bytes)
        > 0,  # always true for valid files; keeps graph simple
        _decode_crop,
        _decode_full,
    )
    img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([target_size[0], target_size[1], 3])
    return img


def _build_image_ds(
    image_names,
    y,
    image_dir,
    batch_size,
    target_size,
    shuffle,
    seed,
    cache_mode,  # None | "memory" | "disk"
    cache_path=None,
):
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    base = tf.constant(image_dir + os.sep, dtype=tf.string)
    paths = tf.strings.join([base, image_names])

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: _load_and_preprocess(p, target_size),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        ds = ds.map(
            lambda p, label: (_load_and_preprocess(p, target_size), label),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_mode == "memory":
        ds = ds.cache()
    elif cache_mode == "disk":
        assert cache_path is not None
        ds = ds.cache(cache_path)

    if shuffle:
        n = int(image_names.shape[0]) if image_names.shape.rank == 1 else 2048
        ds = ds.shuffle(
            buffer_size=min(n, 2048), seed=seed, reshuffle_each_iteration=True
        )

    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    try:
        options.autotune.enabled = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.prefetch(AUTOTUNE)
    return ds


CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)

train_ds = _build_image_ds(
    train_images,
    Y_train,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=True,
    seed=SEED,
    cache_mode="disk",
    cache_path=os.path.join(CACHE_DIR, "train.cache"),
)
valid_ds = _build_image_ds(
    valid_images,
    Y_valid,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache_mode="disk",
    cache_path=os.path.join(CACHE_DIR, "valid.cache"),
)
test_ds = _build_image_ds(
    submissions["image"].values,
    None,
    TEST_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache_mode=None,
    cache_path=None,
)

print(
    "Train samples:",
    len(train_images),
    "Valid samples:",
    len(valid_images),
    "Test samples:",
    len(submissions),
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1188545730.py in <cell line: 0>()
    123 os.makedirs(CACHE_DIR, exist_ok=True)
    124 
--> 125 train_ds = _build_image_ds(
    126     train_images,
    127     Y_train,

/tmp/ipykernel_11/1188545730.py in _build_image_ds(image_names, y, image_dir, batch_size, target_size, shuffle, seed, cache_mode, cache_path)
     77         y = tf.convert_to_tensor(y, dtype=tf.float32)
     78         ds = tf.data.Dataset.from_tensor_slices((paths, y))
---> 79         ds = ds.map(
     80             lambda p, label: (_load_and_preprocess(p, target_size), label),
     81             num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_fileinct9kcs.py in <lambda>(p, label)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda p, label: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_load_and_preprocess, (p, target_size), None, lscope), label), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileinct9kcs.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda p, label: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_load_and_preprocess, (p, target_size), None, lscope), label), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

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

/tmp/__autograph_generated_file61_w0p28.py in tf___load_and_preprocess(path, target_size)
     35                             raise
     36                         return fscope_2.ret(retval__2, do_return_2)
---> 37                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).strings.length, (ag__.ld(img_bytes),), None, fscope) > 0, ag__.ld(_decode_crop), ag__.ld(_decode_full)), None, fscope)
     38                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(target_size)), dict(method=ag__.ld(tf).image.ResizeMethod.AREA), fscope)
     39                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

/tmp/__autograph_generated_file61_w0p28.py in _decode_crop()
     20                         except:
     21                             do_return_1 = False
---> 22                             raise
     23                         return fscope_1.ret(retval__1, do_return_1)
     24 

TypeError: in user code:

    File "/tmp/ipykernel_11/1188545730.py", line 80, in None  *
        lambda p, label: (_load_and_preprocess(p, target_size), label)
    File "/tmp/ipykernel_11/1188545730.py", line 41, in _load_and_preprocess  *
        img = tf.cond(
    File "/tmp/__autograph_generated_file61_w0p28.py", line 22, in _decode_crop
        raise

    TypeError: Expected int32 passed to parameter 'crop_window' of op 'DecodeAndCropJpeg', got [0.05, 0.05, 0.9, 0.9] of type 'list' instead. Error: Expected int32, but got 0.05 of type 'float'.


## === cell 4
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=8,
)

EPOCHS = 3

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    verbose=1,
)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344697777.py in <cell line: 0>()
     19 
     20 history = model.fit(
---> 21     train_ds,
     22     epochs=EPOCHS,
     23     validation_data=valid_ds,

NameError: name 'train_ds' is not defined

## === cell 5
valid_probs = model.predict(valid_ds, verbose=1)
y_true = Y_valid[: valid_probs.shape[0]].astype(np.int32)

grid = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], dtype=np.float32)
best_thr = np.full((n_classes,), 0.5, dtype=np.float32)

P = valid_probs.astype(np.float32)  # (N, C)
T = y_true.astype(np.int32)  # (N, C)

pred_all = (P[None, :, :] >= grid[:, None, None]).astype(np.int32)

tp = (pred_all & T[None, :, :]).sum(axis=1).astype(np.float32)  # (G, C)
fp = (pred_all & (1 - T[None, :, :])).sum(axis=1).astype(np.float32)  # (G, C)
fn = ((1 - pred_all) & T[None, :, :]).sum(axis=1).astype(np.float32)  # (G, C)

f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-9)  # (G, C)
best_idx = np.argmax(f1, axis=0)  # (C,)
best_thr = grid[best_idx].astype(np.float32)

print("Thresholds (first 10):", best_thr[:10])



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/806040865.py in <cell line: 0>()
----> 1 valid_probs = model.predict(valid_ds, verbose=1)
      2 y_true = Y_valid[: valid_probs.shape[0]].astype(np.int32)
      3 
      4 grid = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], dtype=np.float32)
      5 best_thr = np.full((n_classes,), 0.5, dtype=np.float32)

NameError: name 'valid_ds' is not defined

## === cell 6
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])

preds = preds.astype(np.float32)
chosen_mask = preds >= best_thr[None, :]  # (N, C)
top_idx = np.argmax(preds, axis=1).astype(np.int32)

none_chosen = ~chosen_mask.any(axis=1)
chosen_mask[none_chosen, :] = False
chosen_mask[none_chosen, top_idx[none_chosen]] = True

if "healthy" in class_names:
    healthy_idx = int(class_names.index("healthy"))
    has_healthy = chosen_mask[:, healthy_idx]
    has_other = chosen_mask.sum(axis=1) > 1
    drop_healthy = has_healthy & has_other
    chosen_mask[drop_healthy, healthy_idx] = False
    empty_after = ~chosen_mask.any(axis=1)
    chosen_mask[empty_after, healthy_idx] = True

idx_to_name = np.array(class_names, dtype=object)
row_idx, col_idx = np.nonzero(chosen_mask)
groups = np.split(
    col_idx, np.cumsum(np.bincount(row_idx, minlength=chosen_mask.shape[0]))[:-1]
)
pred_labels = [" ".join(idx_to_name[g].tolist()) if len(g) else "" for g in groups]

submissions = submissions.copy()
submissions["labels"] = pred_labels
print(submissions.head())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/173389731.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 print("preds shape:", preds.shape)
      3 print(preds[:2])
      4 
      5 preds = preds.astype(np.float32)

NameError: name 'test_ds' is not defined

## === cell 7
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.head(10))
