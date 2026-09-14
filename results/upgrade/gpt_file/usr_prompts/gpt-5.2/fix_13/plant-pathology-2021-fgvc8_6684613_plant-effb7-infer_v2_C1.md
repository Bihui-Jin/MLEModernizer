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

0.7559372114496771

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

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

print("TF version:", tf.__version__)
print("Num GPUs available:", len(tf.config.list_physical_devices("GPU")))

tf.random.set_seed(42)
np.random.seed(42)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_tensor_float_32_execution(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy.
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
        print("Running on default strategy (CPU/GPU).")
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num unique label-strings (as used by this model):", n_labels)
print("Example label-strings:", class_name[:10])




## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
sample_sub_path = os.path.join(load_dir, "sample_submission.csv")

sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image"]].copy()

try:
    test_images_in_dir = set(tf.io.gfile.listdir(test_dir))
except Exception:
    test_images_in_dir = None

if test_images_in_dir is not None:
    missing = [img for img in test_df["image"].values if img not in test_images_in_dir]
    if missing:
        first_missing = missing[0]
        print(
            "Warning: at least 1 image from sample_submission not found in test_images folder "
            f"(e.g., {first_missing}). Falling back to folder listing."
        )
        test_df = pd.DataFrame({"image": sorted(test_images_in_dir)})
else:
    first_missing = None
    for img in test_df["image"].tolist():
        if not tf.io.gfile.exists(os.path.join(test_dir, img)):
            first_missing = img
            break
    if first_missing is not None:
        print(
            "Warning: at least 1 image from sample_submission not found in test_images folder "
            f"(e.g., {first_missing}). Falling back to folder listing."
        )
        test_df = pd.DataFrame({"image": sorted(tf.io.gfile.listdir(test_dir))})

print("Test images:", len(test_df))




## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [im_size, im_size], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    return img


def _preprocess_base(img):
    img = img * (1.0 / 255.0)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    return img


def _apply_tta(img, idx, tta_round):
    seed_h = tf.stack([tf.cast(tta_round, tf.int32), tf.cast(idx, tf.int32)])
    seed_v = tf.stack([tf.cast(tta_round + 1337, tf.int32), tf.cast(idx, tf.int32)])
    img = tf.image.stateless_random_flip_left_right(img, seed=seed_h)
    img = tf.image.stateless_random_flip_up_down(img, seed=seed_v)
    return img


def make_test_tta_ds(images, tta=5):
    images_t = tf.convert_to_tensor(images, dtype=tf.string)
    n = tf.shape(images_t)[0]
    paths = tf.strings.join([tf.constant(test_dir, tf.string), images_t])

    base_ds = tf.data.Dataset.from_tensor_slices((paths, tf.range(n, dtype=tf.int32)))

    opts = tf.data.Options()
    opts.experimental_deterministic = False  # safe: stateless RNG uses explicit seeds
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.autotune = True
        opts.experimental_optimization.map_vectorization.enabled = True
        opts.experimental_optimization.map_parallelization = True
    except Exception:
        pass
    base_ds = base_ds.with_options(opts)

    @tf.function
    def _load_and_preprocess(p, i):
        img = _preprocess_base(_read_decode_resize(p))
        return img, i

    base_ds = base_ds.batch(BATCH_SIZE, drop_remainder=False)
    base_ds = base_ds.map(
        lambda p, i: (
            tf.map_fn(_read_decode_resize, p, fn_output_signature=tf.float32),
            i,
        ),
        num_parallel_calls=AUTOTUNE,
    )
    base_ds = base_ds.map(
        lambda img_b, i_b: (_preprocess_base(img_b), i_b),
        num_parallel_calls=AUTOTUNE,
    )

    base_ds = base_ds.cache()
    base_ds = base_ds.prefetch(AUTOTUNE)

    tta_rounds = tf.range(tta, dtype=tf.int32)

    def _add_round(img_b, idx_b):
        bsz = tf.shape(idx_b)[0]
        return tf.data.Dataset.from_tensor_slices((img_b, idx_b)).unbatch()

    ds = base_ds.flat_map(_add_round)

    ds = ds.repeat(tta)

    ds = ds.enumerate()

    @tf.function
    def _tta_map(pos, data):
        img, idx = data
        r = tf.cast(tf.math.floormod(pos, tta), tf.int32)
        img = _apply_tta(img, idx, r)
        return img

    ds = ds.map(_tta_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_images_list = test_df["image"].tolist()
n_test = len(test_images_list)




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
weights_path = "/kaggle/input/modelplant1/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print(
        f"Warning: weights file not found at {weights_path}. Proceeding with random weights (score will be poor)."
    )




## === cell 7
import math

TTA = 5

tta_ds = make_test_tta_ds(test_images_list, tta=TTA)

n_pred = n_test * TTA

preds_all = model.predict(tta_ds, verbose=0)

preds_all = preds_all[:n_pred]
if preds_all.shape[0] != n_pred:
    raise RuntimeError(
        f"Prediction count mismatch: got {preds_all.shape[0]}, expected {n_pred}"
    )

preds_all = preds_all.reshape((n_test, TTA, n_labels))
pred = preds_all.mean(axis=1)

argpred = np.argmax(pred, axis=1)

test_df = test_df.copy()
test_df["labels"] = [class_name[i] for i in argpred]

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/228231958.py in <cell line: 0>()
      3 TTA = 5
      4 
----> 5 tta_ds = make_test_tta_ds(test_images_list, tta=TTA)
      6 
      7 n_pred = n_test * TTA

/tmp/ipykernel_11/828515163.py in make_test_tta_ds(images, tta)
     81         return tf.data.Dataset.from_tensor_slices((img_b, idx_b)).unbatch()
     82 
---> 83     ds = base_ds.flat_map(_add_round)
     84 
     85     ds = ds.repeat(tta)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in flat_map(self, map_func, name)
   2387     # pylint: disable=g-import-not-at-top,protected-access
   2388     from tensorflow.python.data.ops import flat_map_op
-> 2389     return flat_map_op._flat_map(self, map_func, name=name)
   2390     # pylint: enable=g-import-not-at-top,protected-access
   2391 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/flat_map_op.py in _flat_map(input_dataset, map_func, name)
     22 def _flat_map(input_dataset, map_func, name=None):  # pylint: disable=unused-private-name
     23   """See `Dataset.flat_map()` for details."""
---> 24   return _FlatMapDataset(input_dataset, map_func, name)
     25 
     26 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/flat_map_op.py in __init__(self, input_dataset, map_func, name)
     31 
     32     self._input_dataset = input_dataset
---> 33     self._map_func = structured_function.StructuredFunctionWrapper(
     34         map_func, self._transformation_name(), dataset=input_dataset)
     35     if not isinstance(self._map_func.output_structure, dataset_ops.DatasetSpec):

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

/tmp/__autograph_generated_file6cb6y_ii.py in tf___add_round(img_b, idx_b)
     11                 try:
     12                     do_return = True
---> 13                     retval_ = ag__.converted_call(ag__.converted_call(ag__.ld(tf).data.Dataset.from_tensor_slices, ((ag__.ld(img_b), ag__.ld(idx_b)),), None, fscope).unbatch, (), None, fscope)
     14                 except:
     15                     do_return = False

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in unbatch(self, name)
   2981     # pylint: disable=g-import-not-at-top,protected-access
   2982     from tensorflow.python.data.ops import unbatch_op
-> 2983     return unbatch_op._unbatch(self, name=name)
   2984     # pylint: enable=g-import-not-at-top,protected-access
   2985 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/unbatch_op.py in _unbatch(input_dataset, name)
     24   """See `Dataset.unbatch()` for details."""
     25   normalized_dataset = dataset_ops.normalize_to_dense(input_dataset)
---> 26   return _UnbatchDataset(normalized_dataset, name=name)
     27 
     28 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/unbatch_op.py in __init__(self, input_dataset, name)
     34     flat_shapes = input_dataset._flat_shapes  # pylint: disable=protected-access
     35     if any(s.ndims == 0 for s in flat_shapes):
---> 36       raise ValueError("Cannot unbatch an input with scalar components.")
     37     known_batch_dim = tensor_shape.Dimension(None)
     38     for s in flat_shapes:

ValueError: in user code:

    File "/tmp/ipykernel_11/828515163.py", line 81, in _add_round  *
        return tf.data.Dataset.from_tensor_slices((img_b, idx_b)).unbatch()

    ValueError: Cannot unbatch an input with scalar components.
