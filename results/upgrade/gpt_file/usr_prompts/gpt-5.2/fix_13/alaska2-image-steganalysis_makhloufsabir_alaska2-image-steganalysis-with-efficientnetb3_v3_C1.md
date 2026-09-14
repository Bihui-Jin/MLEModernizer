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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.696

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.57737) has done: 'Your timeout is dominated by decoding/resizing huge 512×512 JPEGs for tens of thousands of samples plus a heavy EfficientNetB3 forward/backward pass; the single biggest win without changing the model/training semantics is to eliminate redundant image decoding/resize work. I keep the exact architecture, loss, optimizer, epochs, and the same deterministic stratified training subset, but I (1) cache the decoded+resized training dataset (in RAM if possible, otherwise to a local cache file) and (2) set explicit `steps_per_execution` to reduce Python overhead per step while keeping identical math. I also ensure TF data pipelines avoid unnecessary Python/object overhead and keep determinism settings intact. These changes are provably equivalent (same samples, same preprocessing, same training loop) but remove repeated expensive I/O/CPU work that causes the timeout.'
- What this solution (achieved 0.59275) has done: 'The immediate blocker is a TensorFlow import-time crash caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x (`MessageFactory.GetPrototype` removal). I fix this by pinning protobuf behavior in-process before importing TensorFlow (pure compatibility fix; score-neutral) and by forcing Python protobuf implementation as a fallback. I also make the submission alignment robust by sorting test filenames to match `sample_submission.csv` Id order (prevents silent misalignment that can hurt AUC). Core model, preprocessing, and training loop remain unchanged.'
- What this solution (achieved 0.57213) has done: 'I fix the TensorFlow import crash caused by the protobuf 6.x API change by forcing the pure-Python protobuf implementation early and (crucially) setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION=3`, which is the compatible setting for current protobuf. I also add a safe fallback that re-imports TensorFlow only after those env vars are in place, so the notebook runs end-to-end reliably. To nudge score upward without changing the model/training semantics, I correct the test prediction alignment by ensuring predictions are generated in the exact `sample_submission.csv` Id order (and verify lengths), which prevents silent ordering issues that can significantly hurt AUC. Everything else (architecture, preprocessing, subset, optimizer, epochs, and training loop) is preserved.'
- What this solution (achieved 0.58387) has done: 'I fix the TensorFlow/protobuf import crash by setting the required environment variables *before* any TensorFlow-related import happens, and by ensuring they are force-set (not `setdefault`) in case Kaggle’s environment already defines incompatible values. This is a pure compatibility fix and does not change your model, data, or training semantics. I also add a minimal safety check to fall back to CPU if TensorFlow still fails to import (so the notebook produces a valid submission instead of dying). The rest of the pipeline (data ordering, subset logic, EfficientNetB3 model, training loop, and submission writing) is preserved.'
- What this solution (achieved 0.58623) has done: 'I fix the TensorFlow import crash by forcing a protobuf version compatible with TF 2.18 at runtime (protobuf 6.x removed `MessageFactory.GetPrototype`), without changing your model/training logic. Concretely, I install `protobuf==3.20.3` inside the notebook before importing TensorFlow (common Kaggle-safe workaround), then keep your determinism settings and the rest of the pipeline intact. This should make the notebook run end-to-end reliably and write a valid `submission.csv` with `Id,Label`. No score-tuning changes are introduced beyond fixing the broken runtime (your data ordering/alignment is already correct).'
- What this solution (achieved 0.57409) has done: 'Your current score (0.58623) is below the target (0.696), so we should make a small, legitimate improvement without changing the core model/training design. The biggest low-risk gain here is to align preprocessing with EfficientNetB3’s expected `preprocess_input` (your pipeline currently feeds raw `[0,1]` floats), which typically improves ranking/AUC without altering architecture, loss, optimizer, or training loop. I also remove the redundant final sigmoid stack (`Dense(1)` then `Dense(sigmoid)`) by keeping the same output semantics via a single sigmoid head (this is still the same binary classifier setup, but reduces an unnecessary affine layer that can hurt calibration/ranking). Everything else (data split/subset, caching, epochs, steps, submission ordering) remains unchanged and it still writes a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"]
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"

import numpy as np
import pandas as pd

SEED = 10
os.environ["PYTHONHASHSEED"] = str(SEED)

import tensorflow as tf
import tensorflow.keras.layers as l
from tensorflow.keras.regularizers import l2
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF version:", tf.__version__)



## === cell 1
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # may raise if no TPU
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
    print("Running on TPU:", tpu.master())
except Exception as e:
    gpus = tf.config.list_physical_devices("GPU")
    if len(gpus) > 1:
        strategy = tf.distribute.MirroredStrategy()
        print("Running on multi-GPU, replicas:", strategy.num_replicas_in_sync)
    else:
        strategy = tf.distribute.get_strategy()
        print("Running on default strategy. GPUs:", len(gpus))
    print("TPU not used due to:", repr(e))



## === cell 2
AUTO = tf.data.AUTOTUNE

BASE_PATH = "/kaggle/input/alaska2-image-steganalysis"
sample = pd.read_csv(f"{BASE_PATH}/sample_submission.csv")

BATCH_SIZE = 32 * strategy.num_replicas_in_sync
EPOCHS = 1

print("Batch size:", BATCH_SIZE, "Epochs:", EPOCHS)
print(sample.head())




## === cell 3
def list_paths_sorted(folder):
    with os.scandir(folder) as it:
        names = [e.name for e in it if e.is_file()]
    names.sort()
    return names


def build_paths(folder_name):
    folder = os.path.join(BASE_PATH, folder_name)
    names = list_paths_sorted(folder)
    paths = [os.path.join(folder, n) for n in names]
    return names, paths


test_names, test_paths = build_paths("Test")
id_to_path = dict(zip(test_names, test_paths))
missing = set(sample["Id"]) - set(id_to_path.keys())
if len(missing) > 0:
    raise ValueError(
        f"Some Ids from sample_submission.csv are missing in Test folder: {list(sorted(missing))[:5]} ..."
    )
X_test = [id_to_path[_id] for _id in sample["Id"].tolist()]

train_parts = []
for cate in ["JUNIWARD", "JMiPOD", "Cover", "UERD"]:
    _, paths = build_paths(cate)
    label = 0 if cate == "Cover" else 1
    train_parts.append((paths, np.full(len(paths), label, dtype=np.int32)))

X = np.concatenate([np.asarray(p, dtype=object) for p, _ in train_parts])
y = np.concatenate([lab for _, lab in train_parts])

print("Training set distribution:")
unique, counts = np.unique(y, return_counts=True)
print(dict(zip(unique.tolist(), counts.tolist())))
print("Train size:", X.shape[0], "Test size:", len(X_test))



## === cell 4
X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)

X_test = np.asarray(X_test, dtype=object)

print("Train/Val/Test sizes:", X_train.shape[0], X_val.shape[0], X_test.shape[0])



## === cell 5
IMG_SIZE = (512, 512)

_preprocess = tf.keras.applications.efficientnet.preprocess_input


@tf.function(reduce_retracing=True)
def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_ACCURATE")
    image = tf.image.rgb_to_yuv(image)
    y_chan = image[..., :1]  # luminance
    image = tf.concat([y_chan, y_chan, y_chan], axis=-1)

    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, image_size)
    image.set_shape([image_size[0], image_size[1], 3])
    image = _preprocess(image)  # EfficientNetB3 preprocessing
    if label is None:
        return image
    return image, tf.cast(label, tf.float32)




## === cell 6
MAX_TRAIN_SAMPLES = 20000  # deterministic stratified subset


def stratified_subset(X_arr, y_arr, n_total, seed=SEED):
    if n_total is None or n_total >= len(X_arr):
        return X_arr, y_arr
    rng = np.random.RandomState(seed)
    idx0 = np.where(y_arr == 0)[0]
    idx1 = np.where(y_arr == 1)[0]
    rng.shuffle(idx0)
    rng.shuffle(idx1)
    n0 = int(round(n_total * (len(idx0) / len(y_arr))))
    n0 = max(1, min(len(idx0), n0))
    n1 = n_total - n0
    n1 = max(1, min(len(idx1), n1))
    while (n0 + n1) < n_total:
        if n1 < len(idx1):
            n1 += 1
        elif n0 < len(idx0):
            n0 += 1
        else:
            break
    sel = np.concatenate([idx0[:n0], idx1[:n1]])
    rng.shuffle(sel)
    return X_arr[sel], y_arr[sel]


X_train, y_train = stratified_subset(X_train, y_train, MAX_TRAIN_SAMPLES, seed=SEED)
print(
    "After subset - Train size:",
    X_train.shape[0],
    "Val size:",
    X_val.shape[0],
    "Test size:",
    X_test.shape[0],
)



## === cell 7
options = tf.data.Options()
options.experimental_deterministic = True

TRAIN_CACHE_PATH = "/kaggle/working/train_cache.tfcache"

train_dataset = (
    tf.data.Dataset.from_tensor_slices((X_train, y_train))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache(TRAIN_CACHE_PATH)  # cache decoded+resized+preprocessed tensors once
    .shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
    .repeat()
    .batch(BATCH_SIZE, drop_remainder=True)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((X_val, y_val))
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(X_test)
    .with_options(options)
    .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4127457478.py in <cell line: 0>()
      7     tf.data.Dataset.from_tensor_slices((X_train, y_train))
      8     .with_options(options)
----> 9     .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
     10     .cache(TRAIN_CACHE_PATH)  # cache decoded+resized+preprocessed tensors once
     11     .shuffle(4096, seed=SEED, reshuffle_each_iteration=True)

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

/tmp/__autograph_generated_file4ht4l77w.py in tf__decode_image(filename, label, image_size)
     10                 bits = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(filename),), None, fscope)
     11                 image = ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.ld(bits),), dict(channels=3, dct_method='INTEGER_ACCURATE'), fscope)
---> 12                 image = ag__.converted_call(ag__.ld(tf).image.rgb_to_yuv, (ag__.ld(image),), None, fscope)
     13                 y_chan = ag__.ld(image)[..., :1]
     14                 image = ag__.converted_call(ag__.ld(tf).concat, ([ag__.ld(y_chan), ag__.ld(y_chan), ag__.ld(y_chan)],), dict(axis=-1), fscope)

TypeError: in user code:

    File "/tmp/ipykernel_11/2989058204.py", line 13, in decode_image  *
        image = tf.image.rgb_to_yuv(image)

    TypeError: Expected uint8, but got 0.299 of type 'float'.


## === cell 8
with strategy.scope():
    backbone = tf.keras.applications.EfficientNetB3(
        input_shape=(512, 512, 3),
        include_top=False,
        weights="imagenet",
    )
    model = tf.keras.Sequential(
        [
            backbone,
            l.GlobalAveragePooling2D(),
            l.Dense(
                1024,
                activation="relu",
                kernel_regularizer=l2(0.001),
                bias_regularizer=l2(0.001),
            ),
            l.Dropout(0.4),
            l.BatchNormalization(),
            l.BatchNormalization(),
            l.Activation("relu"),
            l.Dropout(0.4),
            l.Dense(1, activation="sigmoid"),
        ]
    )

    opt = Adam(learning_rate=0.001, beta_1=0.9, beta_2=0.999, amsgrad=False)

    model.compile(
        optimizer=opt,
        loss="binary_crossentropy",
        metrics=["accuracy"],
        steps_per_execution=16,
    )

model.summary()



## === cell 9
STEPS_PER_EPOCH = max(1, X_train.shape[0] // BATCH_SIZE)

history = model.fit(
    train_dataset,
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    validation_data=valid_dataset,
    verbose=1,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2129639451.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_dataset,
      5     steps_per_epoch=STEPS_PER_EPOCH,
      6     epochs=EPOCHS,

NameError: name 'train_dataset' is not defined

## === cell 10
model.save("Mymodel.h5")



## === cell 11
pred = model.predict(test_dataset, verbose=1)
pred = np.asarray(pred).reshape(-1)

n_test = sample.shape[0]
if pred.shape[0] < n_test:
    raise RuntimeError(f"Got only {pred.shape[0]} predictions for {n_test} test rows.")
pred = pred[:n_test]

print("Pred shape:", pred.shape, "Test rows:", n_test)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/454050059.py in <cell line: 0>()
----> 1 pred = model.predict(test_dataset, verbose=1)
      2 pred = np.asarray(pred).reshape(-1)
      3 
      4 n_test = sample.shape[0]
      5 if pred.shape[0] < n_test:

NameError: name 'test_dataset' is not defined

## === cell 12
submission = sample.copy()
submission["Label"] = pred

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)

print("Wrote:", submission_path)
print(submission.head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3493506287.py in <cell line: 0>()
      1 submission = sample.copy()
----> 2 submission["Label"] = pred
      3 
      4 submission_path = "submission.csv"
      5 submission.to_csv(submission_path, index=False)

NameError: name 'pred' is not defined
