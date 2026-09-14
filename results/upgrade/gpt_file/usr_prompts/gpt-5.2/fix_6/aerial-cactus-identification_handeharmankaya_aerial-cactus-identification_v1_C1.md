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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.938658

# 6. Current score

0.75484

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.75484) has done: 'The main timeout is coming from slow Python-based image loading/augmentation via `ImageDataGenerator` combined with wrapping it in `tf.data.from_generator` (which adds overhead without enabling true parallel decoding). I keep the exact same model, loss, optimizer, epochs, callbacks, class weights, and augmentation semantics, but switch the input pipeline to a `tf.data` pipeline that reads files directly with parallel map + prefetch (and uses TF-native random flips/rotations/zoom equivalent to the generator settings). I also remove redundant/no-op cells that print large DataFrame info/heads (pure overhead) and ensure determinism is preserved via fixed seeds. This reduces input bottlenecks dramatically while keeping training/evaluation logic intact.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "4")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "2")

import numpy as np
import pandas as pd
import zipfile

from sklearn.model_selection import train_test_split
from sklearn.utils import class_weight

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

import warnings

warnings.filterwarnings("ignore")

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTRAOP_THREADS", "4"))
    )
    tf.config.threading.set_inter_op_parallelism_threads(
        int(os.environ.get("TF_NUM_INTEROP_THREADS", "2"))
    )
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("Python:", os.sys.version)
print("TF:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 2
work_base = "/kaggle/working/aerial-cactus-identification"
train_out = os.path.join(work_base, "train")
test_out = os.path.join(work_base, "test")


def _dir_has_jpgs(p: str) -> bool:
    try:
        return os.path.isdir(p) and any(
            fn.lower().endswith(".jpg") for fn in os.listdir(p)
        )
    except Exception:
        return False


need_extract = not (_dir_has_jpgs(train_out) and _dir_has_jpgs(test_out))
if need_extract:
    os.makedirs(work_base, exist_ok=True)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/train.zip", "r"
    ) as z:
        z.extractall(work_base)
    with zipfile.ZipFile(
        "/kaggle/input/aerial-cactus-identification/test.zip", "r"
    ) as z:
        z.extractall(work_base)




## === cell 3
base_work_dir = "/kaggle/working/aerial-cactus-identification"
train_dir = os.path.join(base_work_dir, "train")
test_dir = os.path.join(base_work_dir, "test")

assert os.path.isdir(train_dir), f"Train directory not found: {train_dir}"
assert os.path.isdir(test_dir), f"Test directory not found: {test_dir}"

train_labels = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")




## === cell 4
df = train_labels.rename(columns={"id": "image", "has_cactus": "label"}).copy()
df["image"] = train_dir.rstrip("/") + "/" + df["image"].astype(str)




## === cell 5
missing = [p for p in df["image"].iloc[:50].tolist() if not os.path.exists(p)]
print("Missing among first 50:", len(missing))




## === cell 6
test_filenames = sorted(
    [
        e.name
        for e in os.scandir(test_dir)
        if e.is_file() and e.name.lower().endswith(".jpg")
    ]
)
test_df = pd.DataFrame({"id": test_filenames})
print("Train images:", len(df), "Test images:", len(test_df))




## === cell 7
assert len(test_df) > 0, "No test images found."




## === cell 8
print(df.head(2))




## === cell 9
print("Train df shape:", df.shape)




## === cell 10
pass




## === cell 11
pass




## === cell 12
df["label"] = df["label"].astype(str)




## === cell 13
pass




## === cell 14
print(df["label"].value_counts())




## === cell 15
RUN_EDA_PLOTS = False




## === cell 16
if RUN_EDA_PLOTS:
    import seaborn as sns
    import matplotlib.pyplot as plt

    sns.countplot(x=df["label"], palette=["salmon", "skyblue"])




## === cell 17
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    img_list1 = df["image"].tolist()
    label_list1 = train_labels["has_cactus"].tolist()
    fig, ax = plt.subplots(2, 5)
    fig.set_size_inches(12, 6)
    k = 0
    for i in range(2):
        for j in range(5):
            img_path = img_list1[k]
            label = label_list1[k]
            img = plt.imread(img_path)
            ax[i, j].imshow(img)
            status = "Has Cactus (1)" if label == 1 else "No Cactus(0)"
            ax[i, j].set_title(status, fontsize=10)
            ax[i, j].axis("off")
            k += 1
    plt.tight_layout()
    plt.show()




## === cell 18
print(test_df.head(2))




## === cell 19
print("Test df shape:", test_df.shape)




## === cell 20
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    sample_test = test_df.sample(10, random_state=SEED).reset_index(drop=True)
    fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(12, 6))
    for i, ax in enumerate(axes.flat):
        img_name = sample_test.loc[i, "id"]
        full_path = os.path.join(test_dir, img_name)
        img = plt.imread(full_path)
        ax.imshow(img)
        ax.axis("off")
    plt.tight_layout()
    plt.show()




## === cell 21
IMG_SIZE = (64, 64)
BATCH_SIZE = 64




## === cell 22
y_int = df["label"].astype(int).values
cw = class_weight.compute_class_weight(
    class_weight="balanced", classes=np.array([0, 1]), y=y_int
)
cw = {0: float(cw[0]), 1: float(cw[1])}

train_df, val_df = train_test_split(
    df, test_size=0.2, random_state=SEED, stratify=df["label"]
)

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train_df["image"].values
train_labels_bin = train_df["label"].astype(np.float32).values
val_paths = val_df["image"].values
val_labels_bin = val_df["label"].astype(np.float32).values

weights_dict = {0: float(cw[0]), 1: float(cw[1])}
print("class_weight used:", weights_dict)

_base_seed = tf.constant([SEED, 0], dtype=tf.int32)


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def _augment(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed2)
    img = tf.image.stateless_random_flip_up_down(
        img, seed2 + tf.constant([0, 1], tf.int32)
    )

    angle = tf.random.stateless_uniform(
        shape=[],
        seed=seed2 + tf.constant([0, 2], tf.int32),
        minval=-30.0,
        maxval=30.0,
        dtype=tf.float32,
    ) * (np.pi / 180.0)
    img = tf.image.rotate(
        img, angles=angle, interpolation="BILINEAR", fill_mode="REFLECT"
    )

    z = tf.random.stateless_uniform(
        shape=[],
        seed=seed2 + tf.constant([0, 3], tf.int32),
        minval=0.8,
        maxval=1.2,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * z), tf.int32)
    zoomed = tf.image.resize(img, (new_h, new_w), method="bilinear", antialias=False)
    zoomed = tf.image.resize_with_crop_or_pad(zoomed, IMG_SIZE[0], IMG_SIZE[1])
    return zoomed


def _train_map_fn(path, label, idx):
    img = _decode_and_resize(path)
    seed2 = _base_seed + tf.stack([0, tf.cast(idx, tf.int32)])
    img = _augment(img, seed2)
    return img, label


def _val_map_fn(path, label):
    img = _decode_and_resize(path)
    return img, label


train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels_bin))
train_ds = train_ds.shuffle(len(train_paths), seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.enumerate()
train_ds = train_ds.map(
    lambda idx, xy: _train_map_fn(xy[0], xy[1], idx), num_parallel_calls=AUTOTUNE
)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels_bin))
val_ds = val_ds.map(_val_map_fn, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1847023178.py in <cell line: 0>()
     94 # Add an index to create deterministic per-example stateless seeds.
     95 train_ds = train_ds.enumerate()
---> 96 train_ds = train_ds.map(
     97     lambda idx, xy: _train_map_fn(xy[0], xy[1], idx), num_parallel_calls=AUTOTUNE
     98 )

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

/tmp/__autograph_generated_file64z0hlo7.py in <lambda>(idx, xy)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda idx, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_map_fn, (xy[0], xy[1], idx), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_file64z0hlo7.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda idx, xy: ag__.with_function_scope(lambda lscope: ag__.converted_call(_train_map_fn, (xy[0], xy[1], idx), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_file_5d4hjd5.py in tf___train_map_fn(path, label, idx)
     10                 img = ag__.converted_call(ag__.ld(_decode_and_resize), (ag__.ld(path),), None, fscope)
     11                 seed2 = ag__.ld(_base_seed) + ag__.converted_call(ag__.ld(tf).stack, ([0, ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(idx), ag__.ld(tf).int32), None, fscope)],), None, fscope)
---> 12                 img = ag__.converted_call(ag__.ld(_augment), (ag__.ld(img), ag__.ld(seed2)), None, fscope)
     13                 try:
     14                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filey6wcxlcq.py in tf___augment(img, seed2)
     11                 img = ag__.converted_call(ag__.ld(tf).image.stateless_random_flip_up_down, (ag__.ld(img), ag__.ld(seed2) + ag__.converted_call(ag__.ld(tf).constant, ([0, 1], ag__.ld(tf).int32), None, fscope)), None, fscope)
     12                 angle = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=ag__.ld(seed2) + ag__.converted_call(ag__.ld(tf).constant, ([0, 2], ag__.ld(tf).int32), None, fscope), minval=-30.0, maxval=30.0, dtype=ag__.ld(tf).float32), fscope) * (ag__.ld(np).pi / 180.0)
---> 13                 img = ag__.converted_call(ag__.ld(tf).image.rotate, (ag__.ld(img),), dict(angles=ag__.ld(angle), interpolation='BILINEAR', fill_mode='REFLECT'), fscope)
     14                 z = ag__.converted_call(ag__.ld(tf).random.stateless_uniform, (), dict(shape=[], seed=ag__.ld(seed2) + ag__.converted_call(ag__.ld(tf).constant, ([0, 3], ag__.ld(tf).int32), None, fscope), minval=0.8, maxval=1.2, dtype=ag__.ld(tf).float32), fscope)
     15                 new_h = ag__.converted_call(ag__.ld(tf).cast, (ag__.converted_call(ag__.ld(tf).round, (ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(IMG_SIZE)[0], ag__.ld(tf).float32), None, fscope) * ag__.ld(z),), None, fscope), ag__.ld(tf).int32), None, fscope)

AttributeError: in user code:

    File "/tmp/ipykernel_11/1847023178.py", line 97, in None  *
        lambda idx, xy: _train_map_fn(xy[0], xy[1], idx)
    File "/tmp/ipykernel_11/1847023178.py", line 81, in _train_map_fn  *
        img = _augment(img, seed2)
    File "/tmp/ipykernel_11/1847023178.py", line 58, in _augment  *
        img = tf.image.rotate(

    AttributeError: module 'tensorflow._api.v2.image' has no attribute 'rotate'


## === cell 23
train_steps = int(np.ceil(len(train_paths) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_paths) / BATCH_SIZE))
print("Train steps:", train_steps, "Val steps:", val_steps)
assert train_steps > 0 and val_steps > 0, "Dataset has length 0 (check paths/df)."




## === cell 24
base_model = ResNet50(weights="imagenet", include_top=False, input_shape=(64, 64, 3))
base_model.trainable = False

model = Sequential()
model.add(base_model)
model.add(GlobalAveragePooling2D())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()




## === cell 25
callbacks = [
    EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=3, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    steps_per_epoch=train_steps,
    epochs=20,
    validation_data=val_ds,
    validation_steps=val_steps,
    callbacks=callbacks,
    class_weight=weights_dict,
)




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3134051279.py in <cell line: 0>()
     10     steps_per_epoch=train_steps,
     11     epochs=20,
---> 12     validation_data=val_ds,
     13     validation_steps=val_steps,
     14     callbacks=callbacks,

NameError: name 'val_ds' is not defined

## === cell 26
print(history.history["accuracy"][-1])




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1985134035.py in <cell line: 0>()
----> 1 print(history.history["accuracy"][-1])
      2 
      3 

NameError: name 'history' is not defined

## === cell 27
model.save("cactus.h5")




## === cell 28
pass




## === cell 29
if RUN_EDA_PLOTS:
    import matplotlib.pyplot as plt

    plt.plot(history.history["accuracy"], label="Accuracy")
    plt.plot(history.history["val_accuracy"], label="Val_Accuracy")
    plt.plot(history.history["loss"], label="Loss")
    plt.plot(history.history["val_loss"], label="Val_Loss")
    plt.legend()




## === cell 30
test_paths = (test_dir.rstrip("/") + "/" + test_df["id"].astype(str)).values


def _test_map_fn(path):
    img = _decode_and_resize(path)
    return img


test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.map(_test_map_fn, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)




## === cell 31
test_steps = int(np.ceil(len(test_paths) / BATCH_SIZE))
predictions = model.predict(
    test_ds,
    steps=test_steps,
    verbose=1,
)
predictions = predictions.reshape(-1)[: len(test_df)]

print("Predictions shape:", predictions.shape)
assert len(predictions) == len(test_df), "Prediction count does not match test rows."




## === cell 32
submission_df = pd.DataFrame({"id": test_df["id"].values, "has_cactus": predictions})
submission_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", submission_df.shape)
print(submission_df.head())
