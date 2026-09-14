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

3.7

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
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        input/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 3 other files
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

-> working/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
Here is some information about the columns:
has_cactus (float64) has 1 unique values: [0.5]
id (object) has 2000 unique values. Some example values: ['09034a34de0e2015a8a28dfe18f423f6.jpg', '44b974fd955c4cd306c7dbae154646b9.jpg', '37cd593f40ba13982e7587a613a9a789.jpg', 'c892c865a5706d3072ba44beafbf2d1c.jpg']

-> working/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
Here is some information about the columns:
has_cactus (int64) has 2 unique values: [1, 0]
id (object) has 2000 unique values. Some example values: ['2de8f189f1dce439766637e75df0ee27.jpg', 'f94b18b4c6850fc44b54f5fbe3c5b0c6.jpg', 'fb0624c0ebc01643a8f4318537099078.jpg', '00b4dfbb267109b5f0d0dde365fa6161.jpg']

# 5. Target score

0.7577

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fixed the protobuf incompatibility, filtered out any missing train images so the dataset can be built, added a fallback that uses the sample‑submission IDs when the test folder is absent, and guarded the plotting code so it only runs if training succeeded. These changes let the script run end‑to‑end and produce a valid `submission.csv` while keeping the original model architecture and training logic.'
- What this solution (achieved 0.5) has done: 'Implemented robust path discovery to locate the dataset regardless of the execution environment, added safety checks for missing images, and ensured `test_files` is always defined. Fixed the training‑validation split error by correctly handling the dataset location. Kept the original VGG19‑based architecture, but after the initial 20‑epoch training we fine‑tune the whole base model with a lower learning rate for a few additional epochs to improve validation AUC. The script now reliably creates a proper `submission.csv` and produces a modest score increase toward the target.'
- What this solution (achieved 0.5) has done: 'I make the training‐image lookup tolerant of missing “.jpg” files by also trying “.png”, so the dataset is not emptied out. Then I pass class‑weights to the fit call (to help the imbalanced cactus labels) and raise the fine‑tuning epochs a bit, which should lift the AUC toward the target while keeping the original VGG19‑based model unchanged. Finally I keep the existing fallback for a missing test folder and ensure the submission CSV is written.'
- What this solution (achieved 0.5) has done: 'Implemented robust image path handling and flexible image decoding to ensure the training set is correctly populated and both JPEG/PNG files are processed without being filtered out. This eliminates the empty‑dataset error, enables proper train/validation splitting, and improves prediction quality, moving the AUC score closer to the target. All other logic, model architecture, and training procedures remain unchanged.'
- What this solution (achieved 0.5) has done: 'The fix filters out any missing training images before splitting, preventing `ReadFile` errors and allowing the model to train, which raise the validation AUC above the baseline 0.5. No core logic or architecture changes are made.'
- What this solution (achieved 0.5) has done: 'Implemented robust image‑file resolution in **cell 0**.  
For each id the code now attempts the original filename, then falls back to the same name with the opposite extension (`.jpg` ↔ `.png`). Rows whose image cannot be located are dropped, preventing an empty training set and allowing the subsequent split, model training, and prediction steps to run. This fix restores the end‑to‑end pipeline and enables a realistic AUC improvement toward the target score.'
- What this solution (achieved 0.5) has done: 'I make the image‑path handling tolerant by skipping the strict existence filter and by providing a safe fallback (a zero‑filled image) when a file is missing. This removes the empty‑metadata error, lets the train/validation split run, and enables the model to train on the available images, which should raise the AUC from the baseline 0.5 toward the target.'
- What this solution (achieved 0.5) has done: 'Implemented robust image loading using `tf.numpy_function` to avoid TensorFlow’s string‑tensor issue, added the same safe loader for test data, and included an AUC metric in model compilation (keeps core architecture unchanged). These fixes ensure the dataset pipelines run without errors, the model trains, and a valid `submission.csv` is written, moving the validation AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I added lightweight image augmentations (random flips, brightness, and contrast) to the training pipeline and modestly extended the fine‑tuning stage (more epochs with a lower learning rate). These changes keep the original VGG19‑based architecture untouched while providing a small boost in validation AUC, moving the score closer to the target.'
- What this solution (achieved 0.5) has done: 'Implemented faster TensorFlow data pipelines by replacing Python‑level image loading (`tf.numpy_function`) with pure TF graph operations, adding caching, and using `tf.cond` for missing files. These changes eliminate costly Python callbacks during every epoch while preserving the exact model architecture, training loop, and evaluation logic.'

# 9. Code solution

## === cell 0
import os, sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

try:
    from google.protobuf import message_factory as mf

    if not hasattr(mf.MessageFactory, "GetPrototype"):

        def _GetPrototype(self, descriptor):
            return descriptor._concrete_class

        mf.MessageFactory.GetPrototype = _GetPrototype
except Exception:
    pass  # protobuf optional; TensorFlow will raise later if needed

import pandas as pd, numpy as np, tensorflow as tf, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

candidates = [
    os.path.abspath(
        os.path.join(os.getcwd(), "..", "input", "aerial-cactus-identification")
    ),
    "/kaggle/input/aerial-cactus-identification",
    "/working/aerial-cactus-identification",
    "/kaggle/working/aerial-cactus-identification",
    os.path.abspath(os.path.join(os.getcwd(), "aerial-cactus-identification")),
]

BASE_PATH = None
for p in candidates:
    if os.path.isdir(p) and os.path.isfile(os.path.join(p, "train.csv")):
        BASE_PATH = p
        break

if BASE_PATH is None:
    raise FileNotFoundError("Could not locate the competition data directory.")

TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

meta_data = pd.read_csv(TRAIN_CSV)
meta_data["has_cactus"] = meta_data["has_cactus"].astype(np.float32)

meta_data["filepath"] = meta_data["id"].apply(
    lambda img_id: os.path.join(TRAIN_DIR, img_id)
)




## === cell 1
train_df, val_df = train_test_split(
    meta_data,
    test_size=0.1,
    random_state=42,
    stratify=meta_data["has_cactus"],
)


def make_dataset(df, shuffle=True, augment=False):
    """Create an efficient tf.data pipeline.
    Replaces the original Python‑based image loader with a pure‑TF implementation
    and adds caching to avoid redundant disk reads across epochs."""
    paths = df["filepath"].values
    labels = df["has_cactus"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _load_image(path):
        img = tf.cond(
            tf.io.gfile.exists(path),
            lambda: tf.image.decode_jpeg(tf.io.read_file(path), channels=3),
            lambda: tf.zeros([32, 32, 3], tf.uint8),
        )
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([32, 32, 3])
        return img

    def _process(path, label):
        img = _load_image(path)
        return img, label

    ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # Cache decoded images for all subsequent epochs

    if augment:

        def _augment(img, label):
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
            img = tf.image.random_brightness(img, max_delta=0.1)
            img = tf.image.random_contrast(img, lower=0.9, upper=1.1)
            return img, label

        ds = ds.map(_augment, num_parallel_calls=tf.data.AUTOTUNE)

    if shuffle:
        ds = ds.shuffle(buffer_size=len(df), seed=42)
    ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = make_dataset(train_df, shuffle=True, augment=True)
val_dataset = make_dataset(val_df, shuffle=False, augment=False)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3161055115.py in <cell line: 0>()
     52 
     53 
---> 54 train_dataset = make_dataset(train_df, shuffle=True, augment=True)
     55 val_dataset = make_dataset(val_df, shuffle=False, augment=False)
     56 

/tmp/ipykernel_11/3161055115.py in make_dataset(df, shuffle, augment)
     32         return img, label
     33 
---> 34     ds = ds.map(_process, num_parallel_calls=tf.data.AUTOTUNE)
     35     ds = ds.cache()  # Cache decoded images for all subsequent epochs
     36 

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

/tmp/__autograph_generated_filexz3v852q.py in tf___process(path, label)
      9                 do_return = False
     10                 retval_ = ag__.UndefinedReturnValue()
---> 11                 img = ag__.converted_call(ag__.ld(_load_image), (ag__.ld(path),), None, fscope)
     12                 try:
     13                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filek4ubbz4p.py in tf___load_image(path)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 img = ag__.converted_call(ag__.ld(tf).cond, (ag__.converted_call(ag__.ld(tf).io.gfile.exists, (ag__.ld(path),), None, fscope), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).image.decode_jpeg, (ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(path),), None, fscope),), dict(channels=3), fscope)), ag__.autograph_artifact(lambda: ag__.converted_call(ag__.ld(tf).zeros, ([32, 32, 3], ag__.ld(tf).uint8), None, fscope))), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [32, 32]), None, fscope)
     12                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

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

    File "/tmp/ipykernel_11/3161055115.py", line 31, in _process  *
        img = _load_image(path)
    File "/tmp/ipykernel_11/3161055115.py", line 20, in _load_image  *
        img = tf.cond(

    TypeError: Expected binary or unicode string, got <tf.Tensor 'args_0:0' shape=() dtype=string>


## === cell 2
from tensorflow.keras.applications import VGG19
from tensorflow.keras import layers, models, optimizers, metrics

base_model = VGG19(input_shape=(32, 32, 3), include_top=False, weights="imagenet")
base_model.trainable = False  # initial frozen training

x = base_model.output
x = layers.Flatten()(x)
x = layers.Dense(1024, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(512, activation="relu")(x)
x = layers.Dropout(0.2)(x)
x = layers.Dense(256, activation="relu")(x)
x = layers.Dropout(0.2)(x)
output = layers.Dense(1, activation="sigmoid")(x)

model = models.Model(inputs=base_model.input, outputs=output)
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=1e-4),
    metrics=["accuracy", metrics.AUC(name="auc")],
)

model.summary()

labels = train_df["has_cactus"].values
neg = np.sum(labels == 0)
pos = np.sum(labels == 1)
total = len(labels)
weight_for_0 = (1 / neg) * (total / 2.0) if neg > 0 else 1.0
weight_for_1 = (1 / pos) * (total / 2.0) if pos > 0 else 1.0
class_weight = {0: weight_for_0, 1: weight_for_1}

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=20,
    class_weight=class_weight,
    verbose=1,
)

base_model.trainable = True
model.compile(
    loss="binary_crossentropy",
    optimizer=optimizers.Adam(learning_rate=5e-6),
    metrics=["accuracy", metrics.AUC(name="auc")],
)

lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(
    monitor="val_auc", factor=0.5, patience=3, verbose=1, min_lr=1e-8
)

fine_history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=40,  # extended fine‑tuning
    class_weight=class_weight,
    callbacks=[lr_reduce],
    verbose=1,
)

if "history" in locals():
    for key in fine_history.history:
        history.history.setdefault(key, []).extend(fine_history.history[key])

final_auc = history.history["val_auc"][-1]
print(f"\nFinal validation AUC: {final_auc:.5f}")




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3881592508.py in <cell line: 0>()
     33 
     34 history = model.fit(
---> 35     train_dataset,
     36     validation_data=val_dataset,
     37     epochs=20,

NameError: name 'train_dataset' is not defined

## === cell 3
if os.path.isdir(TEST_DIR):
    test_files = sorted(
        [f for f in os.listdir(TEST_DIR) if f.lower().endswith((".jpg", ".png"))]
    )
    test_paths = [os.path.join(TEST_DIR, f) for f in test_files]

    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

    def _load_test_image(path):
        img = tf.cond(
            tf.io.gfile.exists(path),
            lambda: tf.image.decode_jpeg(tf.io.read_file(path), channels=3),
            lambda: tf.zeros([32, 32, 3], tf.uint8),
        )
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        img.set_shape([32, 32, 3])
        return img

    test_ds = test_ds.map(_load_test_image, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.cache()
    test_ds = test_ds.batch(32).prefetch(tf.data.AUTOTUNE)

    preds = model.predict(test_ds).flatten()
else:
    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    sample_sub = pd.read_csv(sample_sub_path)
    test_files = sample_sub["id"].tolist()
    preds = np.full(len(test_files), 0.5, dtype=np.float32)  # fallback baseline




## === cell 4
submission = pd.DataFrame({"id": test_files, "has_cactus": preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")




## === cell 5
if "history" in globals():
    plt.figure(figsize=(8, 4))
    plt.plot(history.history["accuracy"], label="Train Acc")
    plt.plot(history.history["val_accuracy"], label="Val Acc")
    plt.title("Training vs Validation Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    plt.show()
