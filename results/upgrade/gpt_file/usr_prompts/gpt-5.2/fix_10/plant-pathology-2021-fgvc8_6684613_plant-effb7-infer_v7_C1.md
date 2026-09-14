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

0.78016620498615

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

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())

os.environ.setdefault("PYTHONHASHSEED", "42")
tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception as _:
    pass

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
def auto_select_accelerator():
    """
    TPU if available, otherwise default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not available, using default strategy. Reason:", repr(e))
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice: 600

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num classes (unique label strings):", n_labels)
print("First 10 classes:", class_name[:10])




## === cell 3
strategy = auto_select_accelerator()

BATCH_SIZE = 64

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")
sample_sub = pd.read_csv(sample_sub_path)

test_df = sample_sub[["image"]].copy()
assert test_df["image"].nunique() == len(
    test_df
), "Duplicate images in sample submission?"

print("Test rows:", len(test_df))
print("Test dir exists:", os.path.isdir(test_dir))




## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input

test_paths = tf.constant(
    [os.path.join(test_dir, fn) for fn in test_df["image"].tolist()]
)

AUTOTUNE = tf.data.AUTOTUNE

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.deterministic = True


@tf.function
def _decode_resize_preprocess(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    return img


@tf.function
def _tta_stack(img, tta):
    img0 = img
    img1 = tf.image.flip_up_down(img)
    img2 = tf.image.flip_left_right(img)
    img3 = tf.image.flip_up_down(img2)
    base4 = tf.stack([img0, img1, img2, img3], axis=0)  # [4,H,W,3]

    tta = tf.cast(tta, tf.int32)
    idx = tf.math.mod(tf.range(tta, dtype=tf.int32), 4)  # [tta]
    return tf.gather(base4, idx, axis=0)


def make_base_test_ds_cached():
    ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(_DATA_OPTIONS)
    ds = ds.map(
        _decode_resize_preprocess, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.cache()  # in-memory cache; avoids expensive filesystem I/O
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_tta_batched_ds(base_ds, tta):
    tta = int(tta)
    ds = base_ds.with_options(_DATA_OPTIONS)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        lambda x: tf.map_fn(
            lambda img: _tta_stack(img, tf.constant(tta, tf.int32)),
            x,
            fn_output_signature=tf.TensorSpec(
                shape=(tta, None, None, 3), dtype=tf.float32
            ),
            parallel_iterations=AUTOTUNE,
        ),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential(
        [
            base,
            GlobalMaxPooling2D(),
            Dense(n_labels, activation="softmax"),
        ]
    )

    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=4e-4),
        metrics=["accuracy"],
    )

model.summary()




## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights not found:", weights_path)
    print(
        "Proceeding with randomly initialized model to generate a valid submission.csv."
    )

try:
    model.run_eagerly = False
except Exception:
    pass




## === cell 7
TTA = 6

base_ds = make_base_test_ds_cached()
tta_batched_ds = make_tta_batched_ds(base_ds, TTA)

pred_chunks = []
for x in tta_batched_ds:
    b = tf.shape(x)[0]
    x_flat = tf.reshape(x, [b * TTA, im_size, im_size, 3])
    p = model(x_flat, training=False)  # [B*TTA, n_labels]
    p = tf.reshape(p, [b, TTA, n_labels])
    pred_chunks.append(p)

p_all = tf.concat(pred_chunks, axis=0).numpy()  # [n_test, TTA, n_labels]

n_test = len(test_df)
if p_all.shape[0] != n_test or p_all.shape[1] != TTA:
    raise RuntimeError(
        f"TTA prediction shape mismatch: got {p_all.shape}, expected ({n_test},{TTA},{n_labels})"
    )

pred = p_all.mean(axis=1)  # [n_test, n_labels]
argpred = np.argmax(pred, axis=1)

if len(argpred) != len(test_df):
    raise RuntimeError(
        f"Prediction length mismatch: got {len(argpred)} preds, expected {len(test_df)}"
    )

test_df["labels"] = argpred.astype(np.int32)
test_df["labels"] = np.take(
    np.array(class_name, dtype=object), test_df["labels"].values
)

submission = test_df[["image", "labels"]].copy()

assert list(submission.columns) == ["image", "labels"]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
print("Submission path:", os.path.abspath("submission.csv"))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3971314641.py in <cell line: 0>()
      2 
      3 base_ds = make_base_test_ds_cached()
----> 4 tta_batched_ds = make_tta_batched_ds(base_ds, TTA)
      5 
      6 # Speed-only change: run one forward pass per batch over flattened [batch*TTA,...],

/tmp/ipykernel_11/2621312115.py in make_tta_batched_ds(base_ds, tta)
     59     ds = base_ds.with_options(_DATA_OPTIONS)
     60     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
---> 61     ds = ds.map(
     62         lambda x: tf.map_fn(
     63             lambda img: _tta_stack(img, tf.constant(tta, tf.int32)),

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

/tmp/__autograph_generated_filerzy09xd5.py in <lambda>(x)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda x: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.map_fn, (ag__.autograph_artifact(lambda img: ag__.converted_call(_tta_stack, (img, ag__.converted_call(tf.constant, (tta, tf.int32), None, lscope)), None, lscope)), x), dict(fn_output_signature=ag__.converted_call(tf.TensorSpec, (), dict(shape=(tta, None, None, 3), dtype=tf.float32), lscope), parallel_iterations=AUTOTUNE), lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filerzy09xd5.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda x: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.map_fn, (ag__.autograph_artifact(lambda img: ag__.converted_call(_tta_stack, (img, ag__.converted_call(tf.constant, (tta, tf.int32), None, lscope)), None, lscope)), x), dict(fn_output_signature=ag__.converted_call(tf.TensorSpec, (), dict(shape=(tta, None, None, 3), dtype=tf.float32), lscope), parallel_iterations=AUTOTUNE), lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    658                   'in a future version' if date is None else
    659                   ('after %s' % date), instructions)
--> 660       return func(*args, **kwargs)
    661 
    662     doc = _add_deprecated_arg_value_notice_to_docstring(func.__doc__, date,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn_v2(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    635   if fn_output_signature is None:
    636     fn_output_signature = dtype
--> 637   return map_fn(
    638       fn=fn,
    639       elems=elems,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/deprecation.py in new_func(*args, **kwargs)
    586                 'in a future version' if date is None else ('after %s' % date),
    587                 instructions)
--> 588       return func(*args, **kwargs)
    589 
    590     doc = _add_deprecated_arg_notice_to_docstring(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/map_fn.py in map_fn(fn, elems, dtype, parallel_iterations, back_prop, swap_memory, infer_shape, name, fn_output_signature)
    495       return (i + 1, tas)
    496 
--> 497     _, r_a = while_loop.while_loop(
    498         lambda i, _: i < n,
    499         compute, (i, result_batchable_ta),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/while_loop.py in while_loop(cond, body, loop_vars, shape_invariants, parallel_iterations, back_prop, swap_memory, name, maximum_iterations, return_same_structure)
    430     raise TypeError("'body' must be callable.")
    431   if parallel_iterations < 1:
--> 432     raise TypeError("'parallel_iterations' must be a positive integer.")
    433 
    434   loop_vars = variable_utils.convert_variables_to_tensors(loop_vars)

TypeError: in user code:

    File "/tmp/ipykernel_11/2621312115.py", line 62, in None  *
        )

    TypeError: 'parallel_iterations' must be a positive integer.
