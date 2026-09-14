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
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Target score

0.9374697534977384

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
from pathlib import Path

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)


def find_hcd_root(base="/kaggle/input"):
    base = Path(base)
    candidates = [
        base / "histopathologic-cancer-detection",
        base / "competitions" / "histopathologic-cancer-detection",
    ]
    for c in candidates:
        if (c / "train_labels.csv").exists() and (c / "sample_submission.csv").exists():
            return c

    for p in base.rglob("train_labels.csv"):
        if p.name == "train_labels.csv":
            root = p.parent
            if (root / "sample_submission.csv").exists():
                return root

    raise FileNotFoundError(
        "Could not locate histopathologic-cancer-detection dataset under /kaggle/input. "
        "Expected train_labels.csv and sample_submission.csv."
    )


DATA_ROOT = find_hcd_root("/kaggle/input")
TRAIN_CSV = DATA_ROOT / "train_labels.csv"
SAMPLE_SUB_CSV = DATA_ROOT / "sample_submission.csv"

if (DATA_ROOT / "train").is_dir() and (DATA_ROOT / "test").is_dir():
    TRAIN_DIR = DATA_ROOT / "train"
    TEST_DIR = DATA_ROOT / "test"
elif (DATA_ROOT / "histopathologic-cancer-detection" / "train").is_dir():
    TRAIN_DIR = DATA_ROOT / "histopathologic-cancer-detection" / "train"
    TEST_DIR = DATA_ROOT / "histopathologic-cancer-detection" / "test"
else:
    train_dirs = [p for p in DATA_ROOT.rglob("train") if p.is_dir()]
    test_dirs = [p for p in DATA_ROOT.rglob("test") if p.is_dir()]
    TRAIN_DIR = train_dirs[0] if train_dirs else None
    TEST_DIR = test_dirs[0] if test_dirs else None
    if TRAIN_DIR is None or TEST_DIR is None:
        raise FileNotFoundError(
            "Could not locate train/ and test/ directories for images."
        )

print("DATA_ROOT:", str(DATA_ROOT))
print("TRAIN_CSV:", str(TRAIN_CSV))
print("SAMPLE_SUB_CSV:", str(SAMPLE_SUB_CSV))
print("TRAIN_DIR:", str(TRAIN_DIR))
print("TEST_DIR:", str(TEST_DIR))



## === cell 1
from pathlib import Path

for d in [TRAIN_DIR, TEST_DIR]:
    d = Path(d)
    try:
        p = next(d.glob("*.tif"))
        print(f"Example .tif in {d}:", str(p))
    except StopIteration:
        print("No .tif found in", str(d))



## === cell 2
train_labels = pd.read_csv(TRAIN_CSV)
print("shape:", train_labels.shape)
print(train_labels.head())
print("Missing values:\n", train_labels.isnull().sum())
print(f"Sum of duplicated labels: {train_labels.duplicated().sum()}.")

train_labels["train_filepath"] = (
    str(TRAIN_DIR) + "/" + train_labels["id"].astype(str) + ".tif"
)

missing = 0
for p in train_labels["train_filepath"].head(20):
    if not Path(p).exists():
        missing += 1
print("Missing among first 20 train images:", missing)



## === cell 3
print("EDA plot skipped to save time.")



## === cell 4
print("Sample visualization skipped to save time.")



## === cell 5
import warnings

warnings.filterwarnings("ignore")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf

print("TensorFlow version:", tf.__version__)
print("GPU available:", tf.config.list_physical_devices("GPU"))

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    import tensorflow_io as tfio  # available on Kaggle for this competition typically

    _HAS_TFIO = True
except Exception:
    _HAS_TFIO = False


def _decode_tif_tfio(path):
    img_bytes = tf.io.read_file(path)
    image = tfio.experimental.image.decode_tiff(
        img_bytes
    )  # uint8/uint16, shape [H,W,C]
    image = tf.cast(image, tf.float32)
    image = tf.cond(
        tf.reduce_max(image) > 255.0,
        lambda: image / 65535.0,
        lambda: image / 255.0,
    )
    c = tf.shape(image)[-1]
    image = tf.cond(
        tf.equal(c, 1),
        lambda: tf.image.grayscale_to_rgb(image),
        lambda: image[..., :3],
    )
    image = tf.ensure_shape(image, [96, 96, 3])
    return image


def _decode_tif_pil(path):
    from PIL import Image

    def _read(p):
        p = p.decode("utf-8")
        img = Image.open(p)
        img = img.convert("RGB")
        arr = np.asarray(img, dtype=np.float32) / 255.0
        return arr

    image = tf.numpy_function(_read, [path], Tout=tf.float32)
    image.set_shape([96, 96, 3])
    return image


def decode_tif(path):
    if _HAS_TFIO:
        return _decode_tif_tfio(path)
    return _decode_tif_pil(path)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
from sklearn.model_selection import train_test_split

train_df, val_df = train_test_split(
    train_labels,
    test_size=0.2,
    stratify=train_labels["label"],
    random_state=SEED,
)


def load_image(path, label):
    image = decode_tif(path)
    label = tf.cast(label, tf.float32)
    return image, label


BATCH_SIZE = 64

options = tf.data.Options()
options.deterministic = False

train_dataset = tf.data.Dataset.from_tensor_slices(
    (train_df["train_filepath"].values, train_df["label"].values)
).with_options(options)
train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
train_dataset = train_dataset.cache()
train_dataset = (
    train_dataset.shuffle(1000, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(tf.data.AUTOTUNE)
)

validation_dataset = tf.data.Dataset.from_tensor_slices(
    (val_df["train_filepath"].values, val_df["label"].values)
).with_options(options)
validation_dataset = validation_dataset.map(
    load_image, num_parallel_calls=tf.data.AUTOTUNE
)
validation_dataset = validation_dataset.cache()
validation_dataset = validation_dataset.batch(
    BATCH_SIZE, drop_remainder=False
).prefetch(tf.data.AUTOTUNE)

print("Data Pre-processing complete..")
print("Train batches:", tf.data.experimental.cardinality(train_dataset).numpy())
print("Val batches:", tf.data.experimental.cardinality(validation_dataset).numpy())



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/1357808495.py in <cell line: 0>()
     23     (train_df["train_filepath"].values, train_df["label"].values)
     24 ).with_options(options)
---> 25 train_dataset = train_dataset.map(load_image, num_parallel_calls=tf.data.AUTOTUNE)
     26 train_dataset = train_dataset.cache()
     27 train_dataset = (

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

/tmp/__autograph_generated_fileqgex5eh1.py in tf__load_image(path, label)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 image = ag__.converted_call(ag__.ld(decode_tif), (ag__.ld(path),), None, fscope)
     11                 label = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(label), ag__.ld(tf).float32), None, fscope)
     12                 try:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filefe1kx04a.py in tf__decode_tif(path)
     33                         do_return = False
     34                         raise
---> 35                 ag__.if_stmt(ag__.ld(_HAS_TFIO), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     36                 return fscope.ret(retval_, do_return)
     37         return tf__decode_tif

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filefe1kx04a.py in if_body()
     20                     try:
     21                         do_return = True
---> 22                         retval_ = ag__.converted_call(ag__.ld(_decode_tif_tfio), (ag__.ld(path),), None, fscope)
     23                     except:
     24                         do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filef1sxi4eo.py in tf___decode_tif_tfio(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 image = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope)
     13                 image = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).reduce_max, (ag__.ld(image),), None, fscope) > 255.0, ag__.autograph_artifact(lambda: ag__.ld(image) / 65535.0), ag__.autograph_artifact(lambda: ag__.ld(image) / 255.0)), None, fscope)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    432     if is_autograph_strict_conversion_mode():
    433       raise
--> 434     return _fall_back_unconverted(f, args, kwargs, options, e)
    435 
    436   with StackTraceMapper(converted_f), tf_stack.CurrentModuleFilter():

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _fall_back_unconverted(f, args, kwargs, options, exc)
    483     logging.warning(warning_template, f, file_bug_message, exc)
    484 
--> 485   return _call_unconverted(f, args, kwargs, options)
    486 
    487 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in _call_unconverted(f, args, kwargs, options, update_cache)
    458   if kwargs is not None:
    459     return f(*args, **kwargs)
--> 460   return f(*args)
    461 
    462 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py in decode_tiff(contents, index, name)
     85       A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
     86     """
---> 87     return core_ops.io_decode_tiff(contents, index, name=name)
     88 
     89 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/1357808495.py", line 12, in load_image  *
        image = decode_tif(path)
    File "/tmp/ipykernel_11/3854905979.py", line 70, in decode_tif  *
        return _decode_tif_tfio(path)
    File "/tmp/ipykernel_11/3854905979.py", line 31, in _decode_tif_tfio  *
        image = tfio.experimental.image.decode_tiff(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff  **
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']


## === cell 7
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    Activation,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.metrics import AUC
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

model_best = Sequential(
    [
        Conv2D(32, kernel_size=5, strides=1, padding="same", input_shape=(96, 96, 3)),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(64, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(128, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(256, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Conv2D(512, kernel_size=3, strides=1, padding="same"),
        BatchNormalization(),
        Activation("relu"),
        MaxPooling2D(pool_size=2, strides=2),
        Flatten(),
        Dropout(0.5),
        Dense(256, activation="relu"),
        Dropout(0.3),
        Dense(1, activation="sigmoid"),
    ]
)

model_best.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy", AUC(name="auroc")],
)

checkpoint = ModelCheckpoint(
    "model_best.keras", monitor="val_auroc", save_best_only=True, mode="max", verbose=0
)
early_stopping = EarlyStopping(
    monitor="val_auroc", patience=3, mode="max", restore_best_weights=True, verbose=1
)

num_epochs = 10

history_best_model = model_best.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=num_epochs,
    callbacks=[early_stopping, checkpoint],
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3047627941.py in <cell line: 0>()
     60 history_best_model = model_best.fit(
     61     train_dataset,
---> 62     validation_data=validation_dataset,
     63     epochs=num_epochs,
     64     callbacks=[early_stopping, checkpoint],

NameError: name 'validation_dataset' is not defined

## === cell 8
print("Diagnostics plot skipped to save time.")



## === cell 9
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_df = sample_sub.copy()
test_df["test_filepath"] = str(TEST_DIR) + "/" + test_df["id"].astype(str) + ".tif"


def load_image_test(path):
    return decode_tif(path)


options = tf.data.Options()
options.deterministic = False

test_dataset = tf.data.Dataset.from_tensor_slices(
    test_df["test_filepath"].values
).with_options(options)
test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
    tf.data.AUTOTUNE
)

preds = model_best.predict(test_dataset, verbose=1).reshape(-1)
test_df["label"] = preds.astype(np.float32)

submission_path = "/kaggle/working/submission.csv"
test_df[["id", "label"]].to_csv(submission_path, index=False)

print("Saved submission:", submission_path)
print(test_df.head())
print("Submission shape:", test_df[["id", "label"]].shape)
assert submission_path.endswith(".csv")
assert list(test_df[["id", "label"]].columns) == ["id", "label"]
assert len(test_df) == len(sample_sub)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2024343345.py in <cell line: 0>()
     14     test_df["test_filepath"].values
     15 ).with_options(options)
---> 16 test_dataset = test_dataset.map(load_image_test, num_parallel_calls=tf.data.AUTOTUNE)
     17 test_dataset = test_dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
     18     tf.data.AUTOTUNE

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

/tmp/__autograph_generated_filegxck7p7_.py in tf__load_image_test(path)
     10                 try:
     11                     do_return = True
---> 12                     retval_ = ag__.converted_call(ag__.ld(decode_tif), (ag__.ld(path),), None, fscope)
     13                 except:
     14                     do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filefe1kx04a.py in tf__decode_tif(path)
     33                         do_return = False
     34                         raise
---> 35                 ag__.if_stmt(ag__.ld(_HAS_TFIO), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     36                 return fscope.ret(retval_, do_return)
     37         return tf__decode_tif

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1215     _tf_if_stmt(cond, body, orelse, get_state, set_state, symbol_names, nouts)
   1216   else:
-> 1217     _py_if_stmt(cond, body, orelse)
   1218 
   1219 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/operators/control_flow.py in _py_if_stmt(cond, body, orelse)
   1268 def _py_if_stmt(cond, body, orelse):
   1269   """Overload of if_stmt that executes a Python if statement."""
-> 1270   return body() if cond else orelse()

/tmp/__autograph_generated_filefe1kx04a.py in if_body()
     20                     try:
     21                         do_return = True
---> 22                         retval_ = ag__.converted_call(ag__.ld(_decode_tif_tfio), (ag__.ld(path),), None, fscope)
     23                     except:
     24                         do_return = False

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filef1sxi4eo.py in tf___decode_tif_tfio(path)
      9                 retval_ = ag__.UndefinedReturnValue()
     10                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope)
---> 11                 image = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     12                 image = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(image), ag__.ld(tf).float32), None, fscope)
     13                 image = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).reduce_max, (ag__.ld(image),), None, fscope) > 255.0, ag__.autograph_artifact(lambda: ag__.ld(image) / 65535.0), ag__.autograph_artifact(lambda: ag__.ld(image) / 255.0)), None, fscope)

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

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py in decode_tiff(contents, index, name)
     85       A `Tensor` of type `uint8` and shape of `[height, width, 4]` (RGBA).
     86     """
---> 87     return core_ops.io_decode_tiff(contents, index, name=name)
     88 
     89 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in __getattr__(self, attrb)
     86 
     87     def __getattr__(self, attrb):
---> 88         return getattr(self._load(), attrb)
     89 
     90     def __dir__(self):

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load(self)
     82     def _load(self):
     83         if self._mod is None:
---> 84             self._mod = _load_library(self._library)
     85         return self._mod
     86 

/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py in _load_library(filename, lib)
     67         except (tf.errors.NotFoundError, OSError) as e:
     68             errs.append(str(e))
---> 69     raise NotImplementedError(
     70         "unable to open file: "
     71         + f"{filename}, from paths: {filenames}\ncaused by: {errs}"

NotImplementedError: in user code:

    File "/tmp/ipykernel_11/2024343345.py", line 7, in load_image_test  *
        return decode_tif(path)
    File "/tmp/ipykernel_11/3854905979.py", line 70, in decode_tif  *
        return _decode_tif_tfio(path)
    File "/tmp/ipykernel_11/3854905979.py", line 31, in _decode_tif_tfio  *
        image = tfio.experimental.image.decode_tiff(
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/experimental/image_ops.py", line 87, in decode_tiff  **
        return core_ops.io_decode_tiff(contents, index, name=name)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 88, in __getattr__
        return getattr(self._load(), attrb)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 84, in _load
        self._mod = _load_library(self._library)
    File "/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/__init__.py", line 69, in _load_library
        raise NotImplementedError(

    NotImplementedError: unable to open file: libtensorflow_io.so, from paths: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so']
    caused by: ['/usr/local/lib/python3.11/dist-packages/tensorflow_io/python/ops/libtensorflow_io.so: undefined symbol: _ZN3tsl8str_util9LowercaseB5cxx11ESt17basic_string_viewIcSt11char_traitsIcEE']
