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

0.7833979686057254

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math
import random
import numpy as np
import pandas as pd
import tensorflow as tf

print("TensorFlow:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("Skipping explicit XLA JIT / op determinism toggles for compatibility.")

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
    TPU if available, else default strategy (GPU/CPU).
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception as e:
        strategy = tf.distribute.get_strategy()
        print("TPU not found, using default strategy. Reason:", type(e).__name__)
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy




## === cell 2
IMSIZES = (224, 240, 260, 300, 380, 456, 528, 600)
im_size = IMSIZES[7]  # keep original choice (600)

load_dir = "/kaggle/input/plant-pathology-2021-fgvc8/"
train_csv_path = os.path.join(load_dir, "train.csv")
df = pd.read_csv(train_csv_path)

df["labels"] = df["labels"].astype(str)
all_tokens = sorted(
    {tok for s in df["labels"].values for tok in s.split() if tok.strip()}
)
class_name = all_tokens
n_labels = len(class_name)

print("Classes:", class_name)
print("n_labels:", n_labels)



## === cell 3
strategy = auto_select_accelerator()

DEFAULT_BATCH_SIZE = 64
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", DEFAULT_BATCH_SIZE))

test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
test_images = sorted(os.listdir(test_dir))
test_df = pd.DataFrame({"image": test_images})
print("Test images:", len(test_df))
print("BATCH_SIZE:", BATCH_SIZE)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

test_paths = np.array(
    [os.path.join(test_dir, fn) for fn in test_df["image"].values], dtype=object
)


def make_base_test_dataset(file_paths):
    ds = tf.data.Dataset.from_tensor_slices(
        tf.convert_to_tensor(file_paths, dtype=tf.string)
    )
    ds = ds.enumerate()  # (idx, path)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    @tf.function
    def _decode_only(idx, path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(
            img, (im_size, im_size), method=tf.image.ResizeMethod.BILINEAR
        )
        img = tf.cast(img, tf.float32)
        return tf.cast(idx, tf.int32), tf.ensure_shape(img, (im_size, im_size, 3))

    ds = ds.map(_decode_only, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset_from_base(base_ds, tta_idx):
    tta_idx = tf.cast(tta_idx, tf.int32)

    @tf.function
    def _map(idx, img):
        base_seed = tf.stack(
            [tf.cast(SEED, tf.int32) + 1000 * tta_idx, tf.cast(idx, tf.int32)], axis=0
        )

        img_aug = tf.image.stateless_random_flip_left_right(
            img, seed=base_seed + tf.constant([1, 0], tf.int32)
        )
        img_aug = tf.image.stateless_random_flip_up_down(
            img_aug, seed=base_seed + tf.constant([2, 0], tf.int32)
        )

        img_zoom = tf.image.stateless_random_zoom(
            img_aug,
            (0.9, 1.1),
            seed=base_seed + tf.constant([3, 0], tf.int32),
            fill_mode="constant",
        )

        img_zoom = tf.keras.applications.efficientnet.preprocess_input(img_zoom)
        return tf.ensure_shape(img_zoom, (im_size, im_size, 3))

    ds = base_ds.map(_map, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 5
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import GlobalMaxPooling2D, Dense

with strategy.scope():
    base = tf.keras.applications.EfficientNetB7(
        weights="imagenet",
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
        jit_compile=True,
    )

model.summary()



## === cell 6
weights_path = "/kaggle/input/model222/bestmodel_tpu_aug.h5"
if os.path.exists(weights_path):
    model.load_weights(weights_path)
    print("Loaded weights:", weights_path)
else:
    print("WARNING: Weights file not found:", weights_path)
    print(
        "Proceeding with ImageNet-pretrained EfficientNetB7 base + random classification head."
    )



## === cell 7
TTA = 6
N = len(test_df)

base_ds = make_base_test_dataset(test_paths)


@tf.function
def _forward(batch):
    return model(batch, training=False)


pred_sum = np.zeros((N, n_labels), dtype=np.float32)

for tta_idx in range(TTA):
    ds = make_test_dataset_from_base(base_ds, tta_idx)

    offset = 0
    for batch in ds:
        probs = _forward(batch)
        probs_np = probs.numpy().astype(np.float32, copy=False)
        bsz = probs_np.shape[0]
        pred_sum[offset : offset + bsz] += probs_np
        offset += bsz

pred = pred_sum / float(TTA)

top1 = np.argmax(pred, axis=1)
test_df["labels"] = [class_name[i] for i in top1]

sub_path = "submission.csv"
test_df[["image", "labels"]].to_csv(sub_path, index=False)
print("Wrote", sub_path)
print(test_df.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2159488869.py in <cell line: 0>()
     16 
     17 for tta_idx in range(TTA):
---> 18     ds = make_test_dataset_from_base(base_ds, tta_idx)
     19 
     20     # Deterministic accumulation in the same order as dataset iteration

/tmp/ipykernel_11/2154979379.py in make_test_dataset_from_base(base_ds, tta_idx)
     71         return tf.ensure_shape(img_zoom, (im_size, im_size, 3))
     72 
---> 73     ds = base_ds.map(_map, num_parallel_calls=AUTOTUNE)
     74     ds = ds.batch(BATCH_SIZE, drop_remainder=False)
     75     ds = ds.prefetch(AUTOTUNE)

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

/tmp/__autograph_generated_fileap934fct.py in tf___map(idx, img)
     12                 img_aug = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_left_right, (ag__.ld(img),), dict(seed=ag__.ld(base_seed) + ag__.converted_call(ag__.ld(tf).constant, ([1, 0], ag__.ld(tf).int32), None, fscope)), fscope)
     13                 img_aug = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(img_aug),), dict(seed=ag__.ld(base_seed) + ag__.converted_call(ag__.ld(tf).constant, ([2, 0], ag__.ld(tf).int32), None, fscope)), fscope)
---> 14                 img_zoom = ag__.converted_call(ag__.ld(tf).image.stateless_random_zoom, (ag__.ld(img_aug), (0.9, 1.1)), dict(seed=ag__.ld(base_seed) + ag__.converted_call(ag__.ld(tf).constant, ([3, 0], ag__.ld(tf).int32), None, fscope), fill_mode='constant'), fscope)
     15                 img_zoom = ag__.converted_call(ag__.ld(tf).keras.applications.efficientnet.preprocess_input, (ag__.ld(img_zoom),), None, fscope)
     16                 try:

AttributeError: in user code:

    File "/tmp/ipykernel_11/2154979379.py", line 63, in _map  *
        img_zoom = tf.image.stateless_random_zoom(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'stateless_random_zoom'


## === cell 8
pred

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2686397780.py in <cell line: 0>()
----> 1 pred

NameError: name 'pred' is not defined
