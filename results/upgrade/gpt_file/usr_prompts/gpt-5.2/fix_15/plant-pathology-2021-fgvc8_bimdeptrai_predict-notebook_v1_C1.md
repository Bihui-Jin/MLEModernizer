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

0.1578947368421052

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.71756) has done: 'Main bottlenecks are input pipeline cost (JPEG decode/resize + expensive custom projective rotation) and per-epoch dataset re-creation/shuffle overhead; the model itself is lightweight. I keep the exact same model, loss, epochs, and augmentation semantics, but make the tf.data pipeline faster by (1) caching decoded+resized images (pre-augmentation) to avoid repeating JPEG decode/resize every epoch, (2) using `shuffle(..., reshuffle_each_iteration=False)` with a deterministic per-example seed that depends on `(epoch, index)` so augmentation still changes each epoch while keeping determinism, and (3) removing Python-level overhead in label splitting and submission label formatting via vectorized/compiled operations. These changes are equivalent in outputs up to negligible float diffs (same ops, same augmentation distributions), but cut redundant CPU work so it finishes within the 600s limit.'
- What this solution (achieved 0.71756) has done: 'The main timeout driver is the CPU-heavy training pipeline: per-step projective rotation/resize zoom inside `tf.data` plus deterministic execution forces slow single-threaded-ish kernels and high Python/graph overhead. To keep the exact same model and training semantics, the optimizations focus on (1) making the dataset pipeline fully graph-compiled and cheaper per element (cache decoded images in-memory instead of on-disk, fuse/enforce static shapes, and precompute constants), (2) reducing input-pipeline contention by setting `drop_remainder` consistently and avoiding repeated shape inference, and (3) ensuring we don’t do any redundant work at predict time. None of these change the architecture, loss, number of steps, augmentations used, or thresholds; they only remove overhead while preserving deterministic behavior.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

train.head(), submissions.head(), train.shape, submissions.shape




## === cell 2
label_split = train["labels"].str.split(" ")

classes = sorted({lab for labs in label_split for lab in labs})
class_to_idx = {c: i for i, c in enumerate(classes)}

y = np.zeros((len(train), len(classes)), dtype=np.float32)
for i, labs in enumerate(label_split):
    for lab in labs:
        y[i, class_to_idx[lab]] = 1.0

labels_df = pd.DataFrame(y, columns=classes)
labels_df.head(), len(classes), classes




## === cell 3
train_df = train.copy()
for c in classes:
    train_df[c] = labels_df[c].values

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

tr_df = train_df.iloc[tr_idx].reset_index(drop=True)
va_df = train_df.iloc[val_idx].reset_index(drop=True)

tr_df.shape, va_df.shape




## === cell 4
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

AUTOTUNE = tf.data.AUTOTUNE

IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
OUT_SHAPE = (IMG_SIZE[0], IMG_SIZE[1])
PI = tf.constant(np.pi, tf.float32)


def _build_paths_and_labels(df, img_dir, with_labels: bool):
    img_dir = str(img_dir).rstrip("/")
    paths = np.array(
        [f"{img_dir}/{img}" for img in df["image"].astype(str).values], dtype=str
    )
    if with_labels:
        labels = df[classes].values.astype(np.float32, copy=False)
        return paths, labels
    return paths


tr_paths, tr_y = _build_paths_and_labels(tr_df, TRAIN_IMG_DIR, with_labels=True)
va_paths, va_y = _build_paths_and_labels(va_df, TRAIN_IMG_DIR, with_labels=True)
te_paths = _build_paths_and_labels(submissions, TEST_IMG_DIR, with_labels=False)

tr_paths_tf = tf.constant(tr_paths)
va_paths_tf = tf.constant(va_paths)
te_paths_tf = tf.constant(te_paths)

tr_y_tf = tf.constant(tr_y)
va_y_tf = tf.constant(va_y)


@tf.function(reduce_retracing=True)
def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img = tf.ensure_shape(img, (IMG_SIZE[0], IMG_SIZE[1], 3))
    return img


@tf.function(reduce_retracing=True)
def _stateless_uniform(shape, seed):
    return tf.random.stateless_uniform(
        shape=shape, seed=seed, minval=0.0, maxval=1.0, dtype=tf.float32
    )


@tf.function(reduce_retracing=True)
def _augment_batch_vectorized(imgs, epoch):
    bsz = tf.shape(imgs)[0]
    i = tf.range(bsz, dtype=tf.int32)
    seeds = tf.stack(
        [tf.fill([bsz], tf.cast(SEED, tf.int32) + tf.cast(epoch, tf.int32)), i], axis=1
    )

    r = _stateless_uniform([bsz], seeds)  # per-image
    do_flip = r < 0.5
    imgs = tf.where(do_flip[:, None, None, None], tf.image.flip_left_right(imgs), imgs)

    angle = (
        _stateless_uniform([bsz], seeds + tf.constant([11, 17], tf.int32)) * 2.0 - 1.0
    ) * (15.0 * PI / 180.0)
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    h = tf.cast(IMG_H, tf.float32)
    w = tf.cast(IMG_W, tf.float32)
    cx = (w - 1.0) / 2.0
    cy = (h - 1.0) / 2.0

    a0 = cos_a
    a1 = sin_a
    a2 = (1.0 - cos_a) * cx - sin_a * cy
    b0 = -sin_a
    b1 = cos_a
    b2 = sin_a * cx + (1.0 - cos_a) * cy

    transforms = tf.stack(
        [
            a0,
            a1,
            a2,
            b0,
            b1,
            b2,
            tf.zeros([bsz], tf.float32),
            tf.zeros([bsz], tf.float32),
        ],
        axis=1,
    )

    imgs = tf.raw_ops.ImageProjectiveTransformV3(
        images=imgs,
        transforms=transforms,
        output_shape=tf.constant([IMG_H, IMG_W], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    imgs = tf.ensure_shape(imgs, (None, IMG_SIZE[0], IMG_SIZE[1], 3))

    z = (_stateless_uniform([bsz], seeds + tf.constant([23, 5], tf.int32)) * 0.2) + 0.9
    new_h = tf.cast(tf.round(z * tf.cast(IMG_H, tf.float32)), tf.int32)
    new_w = tf.cast(tf.round(z * tf.cast(IMG_W, tf.float32)), tf.int32)

    def _resize_one(args):
        img, nh, nw = args
        img2 = tf.image.resize(
            img, [nh, nw], method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img2 = tf.image.resize_with_crop_or_pad(img2, IMG_H, IMG_W)
        img2 = tf.ensure_shape(img2, (IMG_SIZE[0], IMG_SIZE[1], 3))
        return img2

    imgs = tf.map_fn(
        _resize_one,
        (imgs, new_h, new_w),
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )
    imgs = tf.ensure_shape(imgs, (None, IMG_SIZE[0], IMG_SIZE[1], 3))
    return imgs


epoch_var = tf.Variable(0, dtype=tf.int32, trainable=False)


def _make_train_ds_cached(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )

    cache_path = os.path.join("/kaggle/working", "train_decode_cache.tf-data")
    ds = ds.cache(cache_path)

    ds = ds.shuffle(
        buffer_size=len(tr_df), seed=SEED, reshuffle_each_iteration=False
    ).repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=True)

    ds = ds.map(
        lambda imgs, ys: (_augment_batch_vectorized(imgs, epoch_var), ys),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    ).prefetch(AUTOTUNE)

    return ds


def _make_eval_ds(paths, labels=None, drop_remainder=True):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            _decode_resize_rescale, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        return ds
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.map(
        lambda p, y: (_decode_resize_rescale(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_train_ds_cached(tr_paths_tf, tr_y_tf)
val_ds = _make_eval_ds(va_paths_tf, va_y_tf, drop_remainder=True)
test_ds = _make_eval_ds(te_paths_tf, labels=None)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/124877890.py in <cell line: 0>()
    190 
    191 
--> 192 train_ds = _make_train_ds_cached(tr_paths_tf, tr_y_tf)
    193 val_ds = _make_eval_ds(va_paths_tf, va_y_tf, drop_remainder=True)
    194 test_ds = _make_eval_ds(te_paths_tf, labels=None)

/tmp/ipykernel_11/124877890.py in _make_train_ds_cached(paths, labels)
    161     ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    162 
--> 163     ds = ds.map(
    164         lambda imgs, ys: (_augment_batch_vectorized(imgs, epoch_var), ys),
    165         num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filenxs2uxg7.py in <lambda>(imgs, ys)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs, ys: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment_batch_vectorized, (imgs, epoch_var), None, lscope), ys), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filenxs2uxg7.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda imgs, ys: ag__.with_function_scope(lambda lscope: (ag__.converted_call(_augment_batch_vectorized, (imgs, epoch_var), None, lscope), ys), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

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

/tmp/__autograph_generated_filegwpclezq.py in tf___augment_batch_vectorized(imgs, epoch)
     11                 i = ag__.converted_call(ag__.ld(tf).range, (ag__.ld(bsz),), dict(dtype=ag__.ld(tf).int32), fscope)
     12                 seeds = ag__.converted_call(ag__.ld(tf).stack, ([ag__.converted_call(ag__.ld(tf).fill, ([ag__.ld(bsz)], ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(SEED), ag__.ld(tf).int32), None, fscope) + ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(epoch), ag__.ld(tf).int32), None, fscope)), None, fscope), ag__.ld(i)],), dict(axis=1), fscope)
---> 13                 r = ag__.converted_call(ag__.ld(_stateless_uniform), ([ag__.ld(bsz)], ag__.ld(seeds)), None, fscope)
     14                 do_flip = ag__.ld(r) < 0.5
     15                 imgs = ag__.converted_call(ag__.ld(tf).where, (ag__.ld(do_flip)[:, None, None, None], ag__.converted_call(ag__.ld(tf).image.flip_left_right, (ag__.ld(imgs),), None, fscope), ag__.ld(imgs)), None, fscope)

/tmp/__autograph_generated_filei299od1r.py in tf___stateless_uniform(shape, seed)
     13                 except:
     14                     do_return = False
---> 15                     raise
     16                 return fscope.ret(retval_, do_return)
     17         return tf___stateless_uniform

ValueError: in user code:

    File "/tmp/ipykernel_11/124877890.py", line 164, in None  *
        lambda imgs, ys: (_augment_batch_vectorized(imgs, epoch_var), ys)
    File "/tmp/ipykernel_11/124877890.py", line 67, in _augment_batch_vectorized  *
        r = _stateless_uniform([bsz], seeds)  # per-image
    File "/tmp/ipykernel_11/124877890.py", line 50, in _stateless_uniform  *
        shape=shape, seed=seed, minval=0.0, maxval=1.0, dtype=tf.float32

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](seed)' with input shapes: [32,2].


## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), include_top=False, weights="imagenet"
)
base.trainable = False  # keep fast and stable

inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inp, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(classes), activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 6
EPOCHS = 3

steps_per_epoch = len(tr_df) // BATCH_SIZE
validation_steps = max(1, len(va_df) // BATCH_SIZE)


class _EpochVarCallback(tf.keras.callbacks.Callback):
    def on_epoch_begin(self, epoch, logs=None):
        epoch_var.assign(epoch)


history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
    callbacks=[_EpochVarCallback()],
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1711355158.py in <cell line: 0>()
     11 
     12 history = model.fit(
---> 13     train_ds,
     14     validation_data=val_ds,
     15     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 7
test_steps = int(np.ceil(len(submissions) / BATCH_SIZE))

preds = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
preds = preds[: len(submissions)]  # safety
preds.shape, preds[:2]




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1925113472.py in <cell line: 0>()
      2 
      3 preds = model.predict(
----> 4     test_ds,
      5     steps=test_steps,
      6     verbose=1,

NameError: name 'test_ds' is not defined

## === cell 8
thresh = 0.5

mask = preds >= thresh
top_idx = np.argmax(preds, axis=1).astype(np.int32)

class_arr = np.asarray(classes, dtype=object)

pred_labels = np.empty(mask.shape[0], dtype=object)
rows_with_any = mask.any(axis=1)
for i in np.flatnonzero(rows_with_any):
    cols = np.flatnonzero(mask[i])
    pred_labels[i] = " ".join(class_arr[cols].tolist())
for i in np.flatnonzero(~rows_with_any):
    pred_labels[i] = class_arr[int(top_idx[i])]

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels.tolist()

submissions_out.to_csv("submission.csv", index=False)
submissions_out.head(), submissions_out.shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2476984897.py in <cell line: 0>()
      1 thresh = 0.5
      2 
----> 3 mask = preds >= thresh
      4 top_idx = np.argmax(preds, axis=1).astype(np.int32)
      5 

NameError: name 'preds' is not defined

## === cell 9
assert os.path.exists("submission.csv")
assert list(submissions_out.columns) == ["image", "labels"]
assert len(submissions_out) == len(submissions)
print("Wrote submission.csv with", len(submissions_out), "rows")
print(submissions_out.sample(5, random_state=SEED))

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/977763030.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 assert list(submissions_out.columns) == ["image", "labels"]
      3 assert len(submissions_out) == len(submissions)
      4 print("Wrote submission.csv with", len(submissions_out), "rows")
      5 print(submissions_out.sample(5, random_state=SEED))

AssertionError:
