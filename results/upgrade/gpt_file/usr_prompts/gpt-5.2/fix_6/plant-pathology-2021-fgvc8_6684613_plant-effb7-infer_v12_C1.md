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

0.7849676823638048

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf

tf.keras.backend.clear_session()
try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy (GPU/CPU).
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

df["labels"] = df["labels"].astype(str)

all_tokens = set()
for s in df["labels"].values:
    for t in str(s).split():
        if t.strip():
            all_tokens.add(t.strip())

class_name = sorted(all_tokens)
n_labels = len(class_name)

print("classes:", class_name)
print("n_classes:", n_labels)



## === cell 3
strategy = auto_select_accelerator()

BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "64"))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

print("n_test_images:", len(test_df))
print("example test image:", test_df["image"].iloc[0] if len(test_df) else None)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

DATA_OPTS = tf.data.Options()
try:
    DATA_OPTS.experimental_deterministic = True
except Exception:
    pass


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)  # uint8
    img = tf.image.resize(img, [im_size, im_size], method="bilinear", antialias=False)
    img = tf.cast(img, tf.float32)  # EfficientNet preprocess expects float
    return img


def _preprocess_effnet(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


test_paths = tf.constant([os.path.join(test_dir, f) for f in test_df["image"].tolist()])

base_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(DATA_OPTS)
    .map(_decode_resize, num_parallel_calls=AUTOTUNE)
    .map(_preprocess_effnet, num_parallel_calls=AUTOTUNE)
    .cache()  # cached preprocessed images; avoids repeated decode/resize across TTA
)


@tf.function
def _tta_augment_batch(images, seed2):
    b = tf.shape(images)[0]
    seed0 = tf.fill([b], tf.cast(SEED, tf.int32))
    seed1 = tf.range(seed2, seed2 + b, dtype=tf.int32)
    seeds = tf.stack([seed0, seed1], axis=1)  # [b, 2]

    images = tf.image.stateless_random_flip_left_right(images, seed=seeds)
    images = tf.image.stateless_random_flip_up_down(
        images, seed=seeds + tf.constant([1, 1], tf.int32)
    )

    deltas = tf.random.stateless_uniform(
        [b, 1, 1, 1],
        seed=seeds + tf.constant([2, 2], tf.int32),
        minval=-0.2,
        maxval=0.2,
        dtype=images.dtype,
    )
    images = images + deltas
    return images


def make_pred_ds(tta_idx):
    ds = base_ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(
        lambda x: _tta_augment_batch(x, tf.cast(tta_idx * 100000 + 7, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalMaxPooling2D

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights=None,
        include_top=False,
        input_shape=(im_size, im_size, 3),
    )

    model = Sequential()
    model.add(base)
    model.add(GlobalMaxPooling2D())
    model.add(Dense(n_labels, activation="softmax"))

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
    print("WARNING: Weights file not found at:", weights_path)
    print(
        "Proceeding with untrained weights (submission will be low-scoring but valid)."
    )



## === cell 7
TTA = 6
pred_sum = None

for tta_idx in range(TTA):
    ds = make_pred_ds(tta_idx)
    p = model.predict(ds, verbose=1)
    if pred_sum is None:
        pred_sum = p
    else:
        pred_sum += p

pred = pred_sum / float(TTA)

class_name_arr = np.asarray(class_name, dtype=object)

top1 = np.argmax(pred, axis=1)
top1_prob = pred[np.arange(pred.shape[0]), top1]

REL = float(os.environ.get("REL_THR", "0.55"))  # include if p >= REL * p_top1

thr = (REL * top1_prob)[:, None]
keep_mask = pred >= thr
keep_mask[np.arange(pred.shape[0]), top1] = True  # ensure top1 kept

labels_out = []
for i in range(pred.shape[0]):
    idx = np.flatnonzero(keep_mask[i])
    idx = idx[np.argsort(-pred[i, idx])]
    lab = " ".join(class_name_arr[idx].tolist()).strip()
    if not lab:
        lab = "healthy"
    labels_out.append(lab)

test_df["labels"] = labels_out

submission_path = "submission.csv"
test_df[["image", "labels"]].to_csv(submission_path, index=False)
print("Wrote:", submission_path)
print(test_df.head())
print("Submission rows:", len(test_df))
print("Submission columns:", list(test_df[["image", "labels"]].columns))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3580473140.py in <cell line: 0>()
      4 # Speed: keep prediction loop identical, but with the faster vectorized TTA pipeline.
      5 for tta_idx in range(TTA):
----> 6     ds = make_pred_ds(tta_idx)
      7     p = model.predict(ds, verbose=1)
      8     if pred_sum is None:

/tmp/ipykernel_11/3816098702.py in make_pred_ds(tta_idx)
     61 def make_pred_ds(tta_idx):
     62     ds = base_ds.batch(BATCH_SIZE, drop_remainder=False)
---> 63     ds = ds.map(
     64         lambda x: _tta_augment_batch(x, tf.cast(tta_idx * 100000 + 7, tf.int32)),
     65         num_parallel_calls=AUTOTUNE,

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

/tmp/__autograph_generated_filenmqwdgem.py in <lambda>(x)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda x: ag__.with_function_scope(lambda lscope: ag__.converted_call(_tta_augment_batch, (x, ag__.converted_call(tf.cast, (tta_idx * 100000 + 7, tf.int32), None, lscope)), None, lscope), 'lscope', ag__.STD)
      7         return tf__lam
      8     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filenmqwdgem.py in <lambda>(lscope)
      4 
      5     def inner_factory(ag__):
----> 6         tf__lam = lambda x: ag__.with_function_scope(lambda lscope: ag__.converted_call(_tta_augment_batch, (x, ag__.converted_call(tf.cast, (tta_idx * 100000 + 7, tf.int32), None, lscope)), None, lscope), 'lscope', ag__.STD)
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

/tmp/__autograph_generated_file3k4x7xit.py in tf___tta_augment_batch(images, seed2)
     12                 seed1 = ag__.converted_call(ag__.ld(tf).range, (ag__.ld(seed2), ag__.ld(seed2) + ag__.ld(b)), dict(dtype=ag__.ld(tf).int32), fscope)
     13                 seeds = ag__.converted_call(ag__.ld(tf).stack, ([ag__.ld(seed0), ag__.ld(seed1)],), dict(axis=1), fscope)
---> 14                 images = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(images),), dict(seed=ag__.ld(seeds)), fscope)
     15                 images = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(images),), dict(seed=ag__.ld(seeds) + ag__.converted_call(ag__.ld(tf).constant, ([1, 1], ag__.ld(tf).int32), None, fscope)), fscope)
     16                 deltas = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.ld(b), 1, 1, 1],), dict(seed=ag__.ld(seeds) + ag__.converted_call(ag__.ld(tf).constant, ([2, 2], ag__.ld(tf).int32), None, fscope), minval=-0.2, maxval=0.2, dtype=ag__.ld(images).dtype), fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/3816098702.py", line 64, in None  *
        lambda x: _tta_augment_batch(x, tf.cast(tta_idx * 100000 + 7, tf.int32))
    File "/tmp/ipykernel_11/3816098702.py", line 45, in _tta_augment_batch  *
        images = tf.image.stateless_random_flip_left_right(images, seed=seeds)

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT32](stack)' with input shapes: [?,2].
