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
Identify hotels from images.

## Metric
Mean Average Precision @ 5 (MAP@5)

## Submission Format
For each image in the test set, you must predict a space-delimited list of hotel IDs that could match that image. The first ID should be the most relevant one and the last the least relevant one. The file should contain a header and have the following format:

```
image,hotel_id
99e91ad5f2870678.jpg,36363 53586 18807 64314 60181
b5cc62ab665591a9.jpg,36363 53586 18807 64314 60181
d5664a972d5a644b.jpg,36363 53586 18807 64314 60181
```

## Dataset
**train.csv** - The training set metadata.

- `image` - The image ID.

- `chain` - An ID code for the hotel chain. A `chain` of zero (0) indicates that the hotel is either not part of a chain or the chain is not known. This field is not available for the test set. The number of hotels per chain varies widely.

- `hotel_id` - The hotel ID. The target class.

- `timestamp` - When the image was taken. Provided for the training set only.

**sample_submission.csv** - A sample submission file in the correct format.

- `image` The image ID

- `hotel_id` The hotel ID. The target class.

**train_images** - The training set contains 97000+ images from around 7700 hotels from across the globe. All of the images for each hotel chain are in a dedicated subfolder for that chain.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 13,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
            train/
                train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        input/
            description.md (120 lines)
            sample_submission.csv (9757 lines)
            sample_submission.csv.zip (106.8 kB)
            test.zip (160 Bytes)
            test_images.zip (2.6 GB)
            train.csv (87799 lines)
            train.csv.zip (1.9 MB)
            train.zip (162 Bytes)
            train_images.zip (23.5 GB)
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
            test/
                test/
                    test/
            test_images/
                ccc436fc41bf402f.jpg (72.8 kB)
                fb9d48b39c614c32.jpg (91.3 kB)
                ... and 9754 other files
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
            train/
                train/
                    train/
            train_images/
                0/
                    b5bd0a0a2de05bb5.jpg (73.8 kB)
                    c242bcf0719f9d61.jpg (71.9 kB)
                    ... and 18211 other files
                1/
                    a7ad6a44813b77c8.jpg (81.1 kB)
                    9b89db65b496490d.jpg (630.0 kB)
                    ... and 1116 other files
                ... and 87 other folders
        working/
            hotel-id-2021-fgvc8/
                description.md (120 lines)
                sample_submission.csv (9757 lines)
                ... and 7 other files
                hotel-id-2021-fgvc8/
                test/
                    test/
                test_images/
                    ccc436fc41bf402f.jpg (72.8 kB)
                    fb9d48b39c614c32.jpg (91.3 kB)
                    ... and 9754 other files
                    test_images/
                train/
                    train/
                train_images/
                    0/
                        b5bd0a0a2de05bb5.jpg (73.8 kB)
                        c242bcf0719f9d61.jpg (71.9 kB)
                        ... and 18211 other files
                    1/
                        a7ad6a44813b77c8.jpg (81.1 kB)
                        9b89db65b496490d.jpg (630.0 kB)
                        ... and 1116 other files
                    ... and 87 other folders
```

-> data/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> data/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> data/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> input/hotel-id-2021-fgvc8/sample_submission.csv has 9756 rows and 2 columns.
The columns are: image, hotel_id

-> input/hotel-id-2021-fgvc8/train.csv has 87798 rows and 4 columns.
The columns are: image, chain, hotel_id, timestamp

-> (stopped after 10 files for performance)

# 5. Target score

0.0463500884061631

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.00186) has done: 'Main bottlenecks are (1) the per-path existence check using `tf.map_fn(... .numpy() ...)` which forces slow Python execution and (2) caching full decoded/resized train/val/test image tensors to disk, which adds huge IO/serialization overhead and can easily dominate runtime. I replace the existence check with a fast vectorized `tf.io.gfile.exists` loop in Python (no TF graph/`map_fn`) and keep only lightweight caching (file-path lists) while relying on parallel decode/resize + prefetch for throughput. I also make the input pipeline more efficient but equivalent by using `num_parallel_calls=AUTO`, `deterministic=True`, adding `prefetch(AUTO)` before expensive stages where appropriate, and avoiding redundant dataset materialization; model/training/prediction logic and semantics remain unchanged.'
- What this solution (achieved 0.00245) has done: 'I fix the crash at import time caused by an incompatibility between TensorFlow 2.18 and protobuf 6.x by forcing protobuf to use the pure-Python implementation before importing TensorFlow. Then I keep your training/inference logic the same, but make one minimal, score-improving change: switch ResNet50 to ImageNet pretrained weights (still frozen) so predictions aren’t effectively random, which should move MAP@5 substantially toward your target. I also make the train/val split happen after dropping missing image paths to avoid mismatched indexing, and ensure the submission is aligned to `sample_submission.csv` and always written as `submission.csv`. All other architecture, loss, and loops remain unchanged.'
- What this solution (achieved 0.00245) has done: 'We fix the import-time crash by setting both protobuf environment variables early enough and by making the TensorFlow import robust to the protobuf 6.x / TF 2.18 incompatibility. Then we fix a path bug where `DIR="../input/hotel-id-2021-fgvc8"` may not exist in this environment by auto-selecting the first existing dataset root from the provided paths (without changing the rest of the pipeline). Finally, to improve MAP@5 toward your target with minimal semantic change, we ensure test images are predicted in the exact `sample_submission.csv` order (instead of filesystem glob order), preventing misalignment that can severely depress the score.'
- What this solution (achieved 0.00245) has done: 'You’re hitting a TensorFlow import crash caused by the protobuf 6.x runtime API change (`MessageFactory.GetPrototype`), so the pipeline never reaches training/inference or writes `submission.csv`. I fix this by forcing TensorFlow to use the C++ protobuf implementation (instead of the pure-Python one) and by importing TensorFlow first (before importing `tensorflow_hub/tfds` etc.), which resolves this specific error in TF 2.18 + protobuf 6 in Kaggle environments. After that, I keep your model/training/prediction logic the same, but ensure the test image paths are constructed from `sample_submission.csv` order (already done) and that any missing test files still produce valid 5-id strings so a valid `submission.csv` is always written. These changes are execution-unblocking and score-positive (the model can actually run), without altering your core architecture/training semantics.'
- What this solution (achieved 0.00245) has done: 'I fix the TensorFlow import crash by setting protobuf environment variables before any TensorFlow-related import so TF 2.18 can work with protobuf 6.x in this environment. Then I keep your model/training/prediction logic unchanged, but make the pipeline robust by ensuring the dataset path resolution is stable and the test predictions remain aligned to `sample_submission.csv` order (which strongly affects MAP@5). Finally, I keep the submission generation identical but ensure it always writes a valid `submission.csv` with the required columns even if some test files are missing.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed even after forcing pure-Python protobuf runtime."
    ) from e

from sklearn import preprocessing

print("TF:", tf.__version__)
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Determinism setting skipped:", repr(e))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def intialize_accel(hardware):
    """
    input:
    str: GPU or TPU for hardware accelerator

    output:
    strategy -- used later for model definition and fitting
    """
    if hardware == "TPU":
        try:
            resolver = tf.distribute.cluster_resolver.TPUClusterResolver()
            tf.config.experimental_connect_to_cluster(resolver)
            tf.tpu.experimental.initialize_tpu_system(resolver)
            strategy = tf.distribute.TPUStrategy(resolver)
            print("TPU Initialized")
            print("TPU Units:", strategy.num_replicas_in_sync)
            return strategy
        except Exception as e:
            print("TPU Initialization Failed:", repr(e))
            print("Falling back to default strategy.")
            return tf.distribute.get_strategy()

    elif hardware == "GPU":
        print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))
        try:
            strategy = tf.distribute.MirroredStrategy()
        except Exception as e:
            print("MirroredStrategy init failed:", repr(e))
            strategy = tf.distribute.get_strategy()
        return strategy

    print("Unknown hardware option; using default strategy.")
    return tf.distribute.get_strategy()


strategy = intialize_accel("TPU")
AUTO = tf.data.AUTOTUNE



## === cell 2
CANDIDATE_DIRS = [
    "../input/hotel-id-2021-fgvc8",
    "/kaggle/input/hotel-id-2021-fgvc8",
    "/kaggle/data/hotel-id-2021-fgvc8",
    "../input",
    "/kaggle/input",
]
DIR = None
for d in CANDIDATE_DIRS:
    if os.path.exists(d) and os.path.isdir(d):
        if os.path.basename(d) == "hotel-id-2021-fgvc8":
            DIR = d
            break
        if os.path.exists(os.path.join(d, "hotel-id-2021-fgvc8")):
            DIR = os.path.join(d, "hotel-id-2021-fgvc8")
            break

if DIR is None:
    raise FileNotFoundError(
        "Could not locate dataset directory. Tried: " + ", ".join(CANDIDATE_DIRS)
    )

Train_PATH = os.path.join(DIR, "train_images")
Test_PATH = os.path.join(DIR, "test_images")

train_csv_path = os.path.join(DIR, "train.csv")
sample_sub_path = os.path.join(DIR, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)
train_df = train_df.drop_duplicates(subset=["image"]).reset_index(drop=True)

print("Using DIR:", DIR)
print("Number of unique hotel chains: ", train_df.chain.nunique())
print("Number of unique hotels: ", train_df.hotel_id.nunique())
print("Number of Training Samples: ", train_df.shape[0])
print(train_df.head())



## === cell 3
Classes = train_df.hotel_id.nunique()
Channels = 3
size = (200, 200)



## === cell 4
le = preprocessing.LabelEncoder()
train_df["label"] = le.fit_transform(train_df["hotel_id"])

train_df["image_path"] = (
    Train_PATH
    + os.sep
    + train_df["chain"].astype(str).to_numpy()
    + os.sep
    + train_df["image"].to_numpy()
)

paths_list = train_df["image_path"].tolist()
exists_mask_np = np.fromiter(
    (tf.io.gfile.exists(p) for p in paths_list), dtype=bool, count=len(paths_list)
)
missing = int((~exists_mask_np).sum())
if missing:
    print(f"Warning: {missing} train image paths missing; dropping them.")
train_df = train_df.loc[exists_mask_np].reset_index(drop=True)

print(train_df[["hotel_id", "label", "image_path"]].head())

Split = int(0.9 * train_df.shape[0])




## === cell 5
def image_proces(path, labels):
    """
    Reads, decodes, resizes and scales image to [0,1].
    """

    def _read_decode(p):
        data = tf.io.read_file(p)
        img = tf.io.decode_jpeg(data, channels=3, dct_method="INTEGER_FAST")
        return img

    img = tf.cond(
        tf.io.gfile.exists(path),
        lambda: _read_decode(path),
        lambda: tf.zeros([size[0], size[1], 3], dtype=tf.uint8),
    )
    img = tf.image.resize(img, size)
    img = tf.cast(img, tf.float32) / 255.0
    return img, labels


_DS_OPTIONS = tf.data.Options()
_DS_OPTIONS.deterministic = True  # keep stable order for reproducibility
try:
    _DS_OPTIONS.threading.private_threadpool_size = 16
except Exception:
    pass


def import_image(paths, labels, cache=False, cache_path=None):
    dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
    dataset = dataset.with_options(_DS_OPTIONS)
    dataset = dataset.map(image_proces, num_parallel_calls=AUTO)
    if cache:
        dataset = (
            dataset.cache(cache_path) if cache_path is not None else dataset.cache()
        )
    return dataset


def data_augment(image, labels):
    image = tf.image.random_brightness(image, max_delta=0.1)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    return image, labels


sample = pd.read_csv(sample_sub_path)
sample_images = sample["image"].tolist()

Paths_Test = [os.path.join(Test_PATH, img) for img in sample_images]

exists_test = np.fromiter(
    (tf.io.gfile.exists(p) for p in Paths_Test), dtype=bool, count=len(Paths_Test)
)
if not exists_test.all():
    missing_test = int((~exists_test).sum())
    print(
        f"Warning: {missing_test} test image paths missing; will still create submission."
    )
print("Test images (from sample_submission):", len(Paths_Test))

dataset_Test = import_image(
    Paths_Test,
    np.arange(len(Paths_Test), dtype=np.int32),
    cache=False,
)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1715509530.py in <cell line: 0>()
     61 print("Test images (from sample_submission):", len(Paths_Test))
     62 
---> 63 dataset_Test = import_image(
     64     Paths_Test,
     65     np.arange(len(Paths_Test), dtype=np.int32),

/tmp/ipykernel_55/1715509530.py in import_image(paths, labels, cache, cache_path)
     32     dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
     33     dataset = dataset.with_options(_DS_OPTIONS)
---> 34     dataset = dataset.map(image_proces, num_parallel_calls=AUTO)
     35     if cache:
     36         dataset = (

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

/tmp/__autograph_generated_filehhsqtufk.py in tf__image_proces(path, labels)
     26                             raise
     27                         return fscope_1.ret(retval__1, do_return_1)
---> 28                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).io.gfile.exists, (ag__.ld(path),), None, fscope), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(_read_decode), (ag__.ld(path),), None, fscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).zeros, ([ag__.ld(size)[0], ag__.ld(size)[1], 3],), dict(dtype=ag__.ld(tf).uint8), fscope))), None, fscope)
     29                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(size)), None, fscope)
     30                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/lib/io/file_io.py in file_exists_v2(path)
    288   """
    289   try:
--> 290     _pywrap_file_io.FileExists(compat.path_to_bytes(path))
    291   except errors.NotFoundError:
    292     return False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/compat.py in path_to_bytes(path)
    209   if hasattr(path, '__fspath__'):
    210     path = path.__fspath__()
--> 211   return as_bytes(path)
    212 
    213 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/compat.py in as_bytes(bytes_or_text, encoding)
     79     return bytes_or_text
     80   else:
---> 81     raise TypeError('Expected binary or unicode string, got %r' %
     82                     (bytes_or_text,))
     83 

TypeError: in user code:

    File "/tmp/ipykernel_55/1715509530.py", line 13, in image_proces  *
        img = tf.cond(

    TypeError: Expected binary or unicode string, got <tf.Tensor 'args_0:0' shape=() dtype=string>


## === cell 6
print("Number of Test Samples:", int(dataset_Test.cardinality().numpy()))



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2587899372.py in <cell line: 0>()
----> 1 print("Number of Test Samples:", int(dataset_Test.cardinality().numpy()))
      2 

NameError: name 'dataset_Test' is not defined

## === cell 7
paths = train_df["image_path"].values
labels = train_df["label"].values.astype(np.int32)

train_paths, val_paths = paths[:Split], paths[Split:]
train_labels, val_labels = labels[:Split], labels[Split:]

BATCH_SIZE = 32

train_dataset_base = import_image(
    train_paths,
    train_labels,
    cache=False,
)
train_dataset = train_dataset_base.map(data_augment, num_parallel_calls=AUTO)
train_dataset = train_dataset.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

val_dataset = import_image(
    val_paths,
    val_labels,
    cache=False,
)
val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)

print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Val batches:", tf.data.experimental.cardinality(val_dataset).numpy())




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3356196781.py in <cell line: 0>()
      7 BATCH_SIZE = 32
      8 
----> 9 train_dataset_base = import_image(
     10     train_paths,
     11     train_labels,

/tmp/ipykernel_55/1715509530.py in import_image(paths, labels, cache, cache_path)
     32     dataset = tf.data.Dataset.from_tensor_slices((paths, labels))
     33     dataset = dataset.with_options(_DS_OPTIONS)
---> 34     dataset = dataset.map(image_proces, num_parallel_calls=AUTO)
     35     if cache:
     36         dataset = (

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

/tmp/__autograph_generated_filehhsqtufk.py in tf__image_proces(path, labels)
     26                             raise
     27                         return fscope_1.ret(retval__1, do_return_1)
---> 28                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).io.gfile.exists, (ag__.ld(path),), None, fscope), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(_read_decode), (ag__.ld(path),), None, fscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).zeros, ([ag__.ld(size)[0], ag__.ld(size)[1], 3],), dict(dtype=ag__.ld(tf).uint8), fscope))), None, fscope)
     29                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), ag__.ld(size)), None, fscope)
     30                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

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

/usr/local/lib/python3.11/dist-packages/tensorflow/python/lib/io/file_io.py in file_exists_v2(path)
    288   """
    289   try:
--> 290     _pywrap_file_io.FileExists(compat.path_to_bytes(path))
    291   except errors.NotFoundError:
    292     return False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/compat.py in path_to_bytes(path)
    209   if hasattr(path, '__fspath__'):
    210     path = path.__fspath__()
--> 211   return as_bytes(path)
    212 
    213 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/util/compat.py in as_bytes(bytes_or_text, encoding)
     79     return bytes_or_text
     80   else:
---> 81     raise TypeError('Expected binary or unicode string, got %r' %
     82                     (bytes_or_text,))
     83 

TypeError: in user code:

    File "/tmp/ipykernel_55/1715509530.py", line 13, in image_proces  *
        img = tf.cond(

    TypeError: Expected binary or unicode string, got <tf.Tensor 'args_0:0' shape=() dtype=string>


## === cell 8
def create_model(Base, input_shape):
    inputs = tf.keras.Input(shape=tuple(input_shape))

    norm = tf.keras.layers.Normalization()
    x = norm(inputs)

    x = Base(x, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(50, activation="relu", dtype="float32")(x)
    x = tf.keras.layers.BatchNormalization()(x)
    outputs = tf.keras.layers.Dense(Classes, activation="softmax", dtype="float32")(x)

    model = tf.keras.Model(inputs, outputs)
    model._norm_layer = norm  # keep reference for later adapt
    return model




## === cell 9
def compile_model(model, lr):
    optimizer = tf.keras.optimizers.Adam(learning_rate=lr)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False)
    metrics = [tf.keras.metrics.SparseCategoricalAccuracy(name="accuracy")]

    model.compile(
        optimizer=optimizer, loss=loss, metrics=metrics, steps_per_execution=8
    )
    return model




## === cell 10
EPOCHS = 1
VERBOSE = 1

reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_accuracy",
    factor=0.1,
    patience=3,
    mode="max",
    min_delta=0.0001,
    verbose=1,
)

checkpoint_filepath = "./best_model.weights.h5"
model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_accuracy",
    mode="max",
    save_best_only=True,
    verbose=1,
)

callback = tf.keras.callbacks.EarlyStopping(
    monitor="val_accuracy", patience=10, mode="max", min_delta=0.0001, verbose=1
)



## === cell 11
input_shape = [200, 200, Channels]

with strategy.scope():
    Base = tf.keras.applications.ResNet50(
        weights="imagenet", include_top=False, input_shape=tuple(input_shape)
    )
    Base.trainable = False
    model = create_model(Base, input_shape)
    model = compile_model(model, lr=0.001)

try:
    adapt_ds = (
        train_dataset_base.map(lambda x, y: x, num_parallel_calls=AUTO)
        .take(512)
        .prefetch(AUTO)
    )
    model._norm_layer.adapt(adapt_ds)
    print("Normalization layer adapted.")
except Exception as e:
    print("Normalization adapt skipped due to:", repr(e))

print("Fitting")
History = model.fit(
    train_dataset,
    epochs=EPOCHS,
    callbacks=[reduce_lr, model_checkpoint_callback, callback],
    validation_data=val_dataset,
    verbose=VERBOSE,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3560007624.py in <cell line: 0>()
     22 print("Fitting")
     23 History = model.fit(
---> 24     train_dataset,
     25     epochs=EPOCHS,
     26     callbacks=[reduce_lr, model_checkpoint_callback, callback],

NameError: name 'train_dataset' is not defined

## === cell 12
if os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_filepath)
    print("Loaded best weights from:", checkpoint_filepath)
else:
    print("Best weights file not found; using last epoch weights.")

best_model = model



## === cell 13
dataset_Test_batched = dataset_Test.batch(32, drop_remainder=False).prefetch(AUTO)
predictions = best_model.predict(dataset_Test_batched, verbose=1)
print("Pred shape:", predictions.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1878213060.py in <cell line: 0>()
----> 1 dataset_Test_batched = dataset_Test.batch(32, drop_remainder=False).prefetch(AUTO)
      2 predictions = best_model.predict(dataset_Test_batched, verbose=1)
      3 print("Pred shape:", predictions.shape)
      4 

NameError: name 'dataset_Test' is not defined

## === cell 14
topk = 5
k = topk
part = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
row_idx = np.arange(predictions.shape[0])[:, None]
part_scores = predictions[row_idx, part]
order_within = np.argsort(-part_scores, axis=1)
topk_idx = part[row_idx, order_within]

topk_hotel_ids = le.inverse_transform(topk_idx.reshape(-1)).reshape(-1, topk)

images_names = [os.path.basename(p) for p in Paths_Test]
pred_strings = [" ".join(map(str, row)) for row in topk_hotel_ids]

submission = pd.DataFrame({"image": images_names, "hotel_id": pred_strings})

sample = pd.read_csv(sample_sub_path)
submission = sample[["image"]].merge(submission, on="image", how="left")
if submission["hotel_id"].isna().any():
    fallback_ids = le.inverse_transform(np.arange(min(topk, Classes))).tolist()
    if len(fallback_ids) < topk:
        fallback_ids = (fallback_ids * (topk // len(fallback_ids) + 1))[:topk]
    fallback = " ".join(map(str, fallback_ids))
    submission["hotel_id"] = submission["hotel_id"].fillna(fallback)

print(submission.head())
print("Submission rows:", len(submission))

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3657097791.py in <cell line: 0>()
      1 topk = 5
      2 k = topk
----> 3 part = np.argpartition(-predictions, kth=k - 1, axis=1)[:, :k]
      4 row_idx = np.arange(predictions.shape[0])[:, None]
      5 part_scores = predictions[row_idx, part]

NameError: name 'predictions' is not defined
