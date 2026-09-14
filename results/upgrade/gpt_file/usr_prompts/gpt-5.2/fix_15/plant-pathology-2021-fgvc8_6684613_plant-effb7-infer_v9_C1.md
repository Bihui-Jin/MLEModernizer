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

0.7906925207756237

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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

print("TensorFlow:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def auto_select_accelerator():
    """
    TPU if available, else default strategy. Kept as-is logically, just made exception broader.
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
im_size = IMSIZES[7]  # 600 as in original code

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
df = pd.read_csv(os.path.join(load_dir, "train.csv"))

df["labels"] = df["labels"].astype(str)
class_name = df.labels.unique().tolist()
n_labels = len(class_name)

print("Num classes:", n_labels)
print("First 10 classes:", class_name[:10])



## === cell 3
strategy = auto_select_accelerator()
BATCH_SIZE = 32

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_df = pd.DataFrame({"image": sorted(os.listdir(test_dir))})

print("Test images:", len(test_df))
test_df.head()



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_and_cast(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)

    img = tf.image.resize(
        img,
        (im_size, im_size),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    return tf.cast(img, tf.float32)


def _preprocess(img):
    return tf.keras.applications.efficientnet.preprocess_input(img)


options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass
try:
    options.experimental_optimization.autotune_buffers = True
except Exception:
    pass

test_files = tf.constant(test_df["image"].values)
test_paths = tf.strings.join([tf.constant(test_dir), test_files])

base_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
base_ds = base_ds.map(_decode_resize_and_cast, num_parallel_calls=AUTOTUNE)
base_ds = base_ds.cache()
base_ds = base_ds.prefetch(AUTOTUNE)



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
        jit_compile=False,
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if tf.io.gfile.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights not found:", weights_path)
    print("Proceeding with untrained model to produce a valid submission.csv.")



## === cell 7
import math

TTA = 6


@tf.function
def _augment_batched(seeds, imgs):
    def _fold_in_one(seed_i, data):
        return tf.random.experimental.stateless_fold_in(seed_i, tf.cast(data, tf.int64))

    s1 = tf.map_fn(lambda si: _fold_in_one(si, 1), seeds, fn_output_signature=tf.int64)
    s2 = tf.map_fn(lambda si: _fold_in_one(si, 2), seeds, fn_output_signature=tf.int64)
    s3 = tf.map_fn(lambda si: _fold_in_one(si, 3), seeds, fn_output_signature=tf.int64)
    s4 = tf.map_fn(lambda si: _fold_in_one(si, 4), seeds, fn_output_signature=tf.int64)

    x = tf.image.stateless_random_flip_left_right(imgs, seed=s1)
    x = tf.image.stateless_random_flip_up_down(x, seed=s2)

    scale = tf.random.stateless_uniform(
        [tf.shape(x)[0]],
        seed=s3,
        minval=0.9,
        maxval=1.1,
    )

    new_sizes = tf.cast(tf.round(scale * tf.cast(im_size, tf.float32)), tf.int32)

    def _resize_one(args):
        img_i, ns = args
        y = tf.image.resize(
            img_i,
            (ns, ns),
            method=tf.image.ResizeMethod.BILINEAR,
            antialias=False,
        )
        y = tf.image.resize_with_crop_or_pad(y, im_size, im_size)
        return y

    x = tf.map_fn(
        _resize_one,
        (x, new_sizes),
        fn_output_signature=tf.float32,
        parallel_iterations=32,
    )

    factor = tf.random.stateless_uniform(
        [tf.shape(x)[0]],
        seed=s4,
        minval=0.9,
        maxval=1.1,
    )
    factor = tf.reshape(factor, [-1, 1, 1, 1])
    x = tf.clip_by_value(x * factor, 0.0, 255.0)

    x = _preprocess(x)
    return x


def make_tta_dataset(base, tta, n_items):
    n_items = int(n_items)
    total = n_items * int(tta)

    image_idx = tf.range(total, dtype=tf.int64) % tf.cast(n_items, tf.int64)
    tta_idx = tf.range(total, dtype=tf.int64) // tf.cast(n_items, tf.int64)

    seeds = tf.stack(
        [
            tf.fill([total], tf.cast(SEED, tf.int64)),
            tta_idx * tf.cast(1000003, tf.int64) + image_idx,
        ],
        axis=1,
    )

    ds_img = base.repeat(tta)
    ds_seed = tf.data.Dataset.from_tensor_slices(seeds)
    ds = tf.data.Dataset.zip((ds_seed, ds_img)).with_options(options)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.map(_augment_batched, num_parallel_calls=AUTOTUNE)

    try:
        ds = ds.apply(tf.data.experimental.prefetch_to_device("/device:GPU:0"))
    except Exception:
        ds = ds.prefetch(AUTOTUNE)
    return ds


n_test = len(test_df)
tta_ds = make_tta_dataset(base_ds, TTA, n_test)

steps = int(math.ceil((n_test * TTA) / BATCH_SIZE))
pred_all = model.predict(tta_ds, steps=steps, verbose=1)

pred_all = pred_all[: (n_test * TTA)]
pred_all = pred_all.reshape((TTA, n_test, n_labels))
pred = pred_all.mean(axis=0)

argpred = np.argmax(pred, axis=1)

class_name_arr = np.array(class_name, dtype=object)
test_df["labels"] = class_name_arr[argpred]

submission = test_df[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Saved submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1575309142.py in <cell line: 0>()
     89 
     90 n_test = len(test_df)
---> 91 tta_ds = make_tta_dataset(base_ds, TTA, n_test)
     92 
     93 steps = int(math.ceil((n_test * TTA) / BATCH_SIZE))

/tmp/ipykernel_11/1575309142.py in make_tta_dataset(base, tta, n_items)
     79 
     80     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
---> 81     ds = ds.map(_augment_batched, num_parallel_calls=AUTOTUNE)
     82 
     83     try:

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/traceback_utils.py in error_handler(*args, **kwargs)
    151     except Exception as e:
    152       filtered_tb = _process_traceback_frames(e.__traceback__)
--> 153       raise e.with_traceback(filtered_tb) from None
    154     finally:
    155       del filtered_tb

/tmp/__autograph_generated_filel_uhnhxy.py in tf___augment_batched(seeds, imgs)
     25                 s3 = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.autograph_artifact(lambda si: ag__.converted_call(ag__.ld(_fold_in_one), (ag__.ld(si), 3), None, fscope)), ag__.ld(seeds)), dict(fn_output_signature=ag__.ld(tf).int64), fscope)
     26                 s4 = ag__.converted_call(ag__.ld(tf).map_fn, (ag__.autograph_artifact(lambda si: ag__.converted_call(ag__.ld(_fold_in_one), (ag__.ld(si), 4), None, fscope)), ag__.ld(seeds)), dict(fn_output_signature=ag__.ld(tf).int64), fscope)
---> 27                 x = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(imgs),), dict(seed=ag__.ld(s1)), fscope)
     28                 x = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(x),), dict(seed=ag__.ld(s2)), fscope)
     29                 scale = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, ([ag__.converted_call(ag__.ld(tf).shape, (ag__.ld(x),), None, fscope)[0]],), dict(seed=ag__.ld(s3), minval=0.9, maxval=1.1), fscope)

ValueError: in user code:

    File "/tmp/ipykernel_11/1575309142.py", line 18, in _augment_batched  *
        x = tf.image.stateless_random_flip_left_right(imgs, seed=s1)

    ValueError: Shape must be rank 1 but is rank 2 for '{{node stateless_random_flip_left_right/stateless_random_uniform/StatelessRandomGetKeyCounter}} = StatelessRandomGetKeyCounter[Tseed=DT_INT64](map/TensorArrayV2Stack/TensorListStack)' with input shapes: [?,2].
