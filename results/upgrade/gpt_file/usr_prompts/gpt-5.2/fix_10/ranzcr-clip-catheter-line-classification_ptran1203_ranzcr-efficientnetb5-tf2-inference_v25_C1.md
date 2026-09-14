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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.9406085412308705

# 6. Current score

0.49399

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.49997) has done: 'I fix the runtime error by removing the incorrect `.numpy()` call on the output of `predict_on_batch` (it already returns a NumPy array in TF2). Then I ensure the script always writes `submission.csv` even if something goes wrong by validating shapes and clipping predictions to `[0,1]` for a robust, valid probability submission. I also correct a harmless but wrong `label_list` assignment (it currently pulls from TFRecord feature keys, not label columns) to avoid confusion without changing core behavior. Finally, I keep the exact submission columns/order from `sample_submission.csv` so the output passes the format assertions.'
- What this solution (achieved 0.49399) has done: 'The timeout is dominated by decoding/resizing ~27k training JPEGs for two epochs plus AUC computation each step; the model itself is heavy (EfficientNetB5). To stay within 600s without changing the core logic, the main fixes are: (1) remove expensive per-file `os.path.exists` checks by filtering with a deterministic TensorFlow file-exists filter inside the dataset pipeline, (2) avoid on-disk `ds.cache("...")` for the large training set (it often becomes slower than recomputing due to disk I/O), and (3) fuse the test UID extraction and prediction into a single pass to avoid iterating the test dataset twice. All changes preserve the same architecture, preprocessing, loss, training loop, epochs, and metric semantics; they only remove redundant work and slow I/O patterns.'

# 9. Code solution

## === cell 0
import os, sys, subprocess
import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit


def _ensure_compatible_protobuf():
    try:
        import google.protobuf  # noqa: F401

        ver = getattr(google.protobuf, "__version__", "0")
        major = int(ver.split(".")[0])
        if major < 5:
            pass
    except Exception as e:
        print("Warning: protobuf compatibility check failed:", e)


_ensure_compatible_protobuf()

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception as e:
    print("Warning: could not enable XLA JIT:", e)

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

W = H = 338
BATCH_SIZE = 16
autotune = tf.data.AUTOTUNE  # equivalent to experimental.AUTOTUNE

DATA_DIR = "../input/ranzcr-clip-catheter-line-classification"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
test_dir = os.path.join(DATA_DIR, "test")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

target_cols_full = [
    "ETT - Abnormal",
    "ETT - Borderline",
    "ETT - Normal",
    "NGT - Abnormal",
    "NGT - Borderline",
    "NGT - Incompletely Imaged",
    "NGT - Normal",
    "CVC - Abnormal",
    "CVC - Borderline",
    "CVC - Normal",
    "Swan Ganz Catheter Present",
]


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
sample_sub = pd.read_csv(sample_sub_path)
sub_cols = [c for c in sample_sub.columns if c != "StudyInstanceUID"]

train_df = pd.read_csv(train_csv_path)
train_label_cols = [c for c in target_cols_full if c in train_df.columns]

if len(train_label_cols) == 0:
    raise ValueError("No target label columns found in train.csv; cannot train.")

N_CLASSES = len(train_label_cols)

print("Submission columns (from sample_submission.csv):", sub_cols)
print("Training label columns (from train.csv):", train_label_cols)
print("N_CLASSES for model output:", N_CLASSES)


## === cell 2
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
groups = train_df["PatientID"].values
idx_train, idx_val = next(gss.split(train_df, groups=groups))

df_tr = train_df.iloc[idx_train].reset_index(drop=True)
df_va = train_df.iloc[idx_val].reset_index(drop=True)

train_dir = os.path.join(DATA_DIR, "train")


def _path_from_uid(uid):
    return os.path.join(train_dir, f"{uid}.jpg")


df_tr["path"] = df_tr["StudyInstanceUID"].map(_path_from_uid)
df_va["path"] = df_va["StudyInstanceUID"].map(_path_from_uid)

print("Train/Val sizes (before tf.data existence filtering):", len(df_tr), len(df_va))


## === cell 3
preprocess_input = tf.keras.applications.efficientnet.preprocess_input


def _load_jpg_with_label(path, y):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (H, W))  # keep identical resize semantics
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # critical for EfficientNet
    y = tf.cast(y, tf.float32)
    return img, y


_ds_options = tf.data.Options()
_ds_options.experimental_deterministic = True
_ds_options.experimental_optimization.apply_default_optimizations = True
try:
    _ds_options.experimental_optimization.map_parallelization = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)


def _make_ds(df, training, cache_id=None):
    paths = df["path"].values
    y = df[train_label_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    ds = ds.with_options(_ds_options)

    ds = ds.filter(lambda p, yy: tf.io.gfile.exists(p))

    if training:
        ds = ds.shuffle(min(len(df), 4096), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load_jpg_with_label, num_parallel_calls=autotune)

    if cache_id is not None and cache_id == "test":
        ds = ds.cache(os.path.join(CACHE_DIR, f"{cache_id}.cache"))

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.prefetch(autotune)
    return ds


ds_tr = _make_ds(df_tr, training=True, cache_id=None)
ds_va = _make_ds(df_va, training=False, cache_id=None)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3210719350.py in <cell line: 0>()
     52 
     53 
---> 54 ds_tr = _make_ds(df_tr, training=True, cache_id=None)
     55 ds_va = _make_ds(df_va, training=False, cache_id=None)
     56 

/tmp/ipykernel_55/3210719350.py in _make_ds(df, training, cache_id)
     33     # CHANGE (timeout): filter missing files inside tf.data (fast, parallelizable) instead of Python exists checks.
     34     # Correctness preserved: only removes records that cannot be read; previously removed by os.path.exists anyway.
---> 35     ds = ds.filter(lambda p, yy: tf.io.gfile.exists(p))
     36 
     37     if training:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in filter(self, predicate, name)
   2561     # pylint: disable=g-import-not-at-top,protected-access
   2562     from tensorflow.python.data.ops import filter_op
-> 2563     return filter_op._filter(self, predicate, name)
   2564     # pylint: enable=g-import-not-at-top,protected-access
   2565 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in _filter(input_dataset, predicate, name)
     23 
     24 def _filter(input_dataset, predicate, name=None):  # pylint: disable=redefined-builtin
---> 25   return _FilterDataset(input_dataset, predicate, name=name)
     26 
     27 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/filter_op.py in __init__(self, input_dataset, predicate, use_legacy_function, name)
     36     """See `Dataset.filter` for details."""
     37     self._input_dataset = input_dataset
---> 38     wrapped_func = structured_function.StructuredFunctionWrapper(
     39         predicate,
     40         self._transformation_name(),

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

/tmp/__autograph_generated_fileeamawkfe.py in <lambda>(p, yy)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, yy: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.io.gfile.exists, (p,), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_fileeamawkfe.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, yy: ag__.with_function_scope(lambda lscope: ag__.converted_call(tf.io.gfile.exists, (p,), None, lscope), 'lscope', ag__.STD)
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

    File "/tmp/ipykernel_55/3210719350.py", line 35, in None  *
        lambda p, yy: tf.io.gfile.exists(p)

    TypeError: Expected binary or unicode string, got <tf.Tensor 'args_0:0' shape=() dtype=string>


## === cell 4
def get_model(
    base_model,
    baseline_weight=None,
    init_weight=None,
    lr=0.001,
    optimizer=tf.optimizers.Adam,
):
    base_model = base_model(
        include_top=False, input_shape=(W, H, 3), pooling="avg", weights=baseline_weight
    )
    base_out = base_model.output
    out = tf.keras.layers.Dropout(0.3)(base_out)
    out = tf.keras.layers.Dense(N_CLASSES, activation="sigmoid")(out)
    model = tf.keras.models.Model(inputs=base_model.input, outputs=out)

    model.compile(
        optimizer=optimizer(learning_rate=lr),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(multi_label=True, num_labels=N_CLASSES)],
        jit_compile=True,
    )
    if init_weight:
        try:
            model.load_weights(init_weight)
            print(f"Weight loaded from {init_weight}")
        except Exception as e:
            print(f"Load weight from {init_weight} failed, {e}")
    return model


base_model_fn = tf.keras.applications.EfficientNetB5
model = get_model(base_model_fn, baseline_weight="imagenet", init_weight=None)


## === cell 5
EPOCHS = 2

history = model.fit(
    ds_tr,
    validation_data=ds_va,
    epochs=EPOCHS,
    verbose=1,
)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3589937083.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     ds_tr,
      5     validation_data=ds_va,
      6     epochs=EPOCHS,

NameError: name 'ds_tr' is not defined

## === cell 6
test_files = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
test_files = sorted(test_files)


def _load_test_jpg(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (H, W))
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)  # critical for EfficientNet
    uid = tf.strings.regex_replace(tf.strings.split(path, os.sep)[-1], r"\.jpg$", "")
    return img, uid


test_data = tf.data.Dataset.from_tensor_slices(test_files).with_options(_ds_options)
test_data = test_data.map(_load_test_jpg, num_parallel_calls=autotune)

test_data = test_data.cache(os.path.join(CACHE_DIR, "test.cache"))

test_data = test_data.batch(BATCH_SIZE, drop_remainder=False)
test_data = test_data.apply(tf.data.experimental.ignore_errors())
test_data = test_data.prefetch(autotune)


## === cell 7
uids_list = []
preds_list = []

for x_batch, uid_batch in test_data:
    preds_batch = model(x_batch, training=False)
    preds_list.append(preds_batch.numpy())
    uids_list.append(uid_batch.numpy().astype("U"))

uids = np.concatenate(uids_list, axis=0).tolist()
preds = np.concatenate(preds_list, axis=0)
preds = np.clip(preds, 0.0, 1.0)

pred_map = {c: preds[:, i] for i, c in enumerate(train_label_cols)}

test_df = pd.DataFrame({"StudyInstanceUID": uids})
for c in sub_cols:
    if c in pred_map:
        test_df[c] = pred_map[c]
    else:
        test_df[c] = 0.5

test_df = (
    test_df.set_index("StudyInstanceUID")
    .reindex(sample_sub["StudyInstanceUID"])
    .reset_index()
)

test_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", test_df.shape)
test_df.head()


## === cell 8
assert os.path.exists("submission.csv")
out = pd.read_csv("submission.csv")
assert list(out.columns) == list(sample_sub.columns), (
    out.columns.tolist(),
    sample_sub.columns.tolist(),
)
assert len(out) == len(sample_sub)
out.describe(include="all")
