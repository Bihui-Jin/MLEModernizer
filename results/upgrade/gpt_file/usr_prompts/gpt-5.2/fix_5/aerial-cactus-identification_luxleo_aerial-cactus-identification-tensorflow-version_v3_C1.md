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

3.10

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

0.9909

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the missing `train/` and `test/` folder issue by extracting the zip files into a known working directory and then consistently referencing those absolute paths everywhere (counting, plotting, tf.data loaders, and prediction). I also fix the TensorFlow dataset shape error by explicitly setting the tensor shape after `tf.py_function`, which removes the “unknown TensorShape” crash during `model.fit`. The protobuf `MessageFactory.GetPrototype` crash be avoided by using a simple, stable `tf.keras.utils.set_random_seed` setup and removing the notebook-only `%matplotlib inline` magic that can trigger environment-dependent issues. Finally, I ensure predictions are flattened to 1D and written to a valid `submission.csv` with the exact required columns.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for filename in filenames[:5]:
        print(os.path.join(dirname, filename))



## === cell 1
import pandas as pd

PATH = "/kaggle/input/aerial-cactus-identification/"
labels = pd.read_csv(PATH + "train.csv")
submissions = pd.read_csv(PATH + "sample_submission.csv")
labels.head()



## === cell 2
import matplotlib as mpl
import matplotlib.pyplot as plt

mpl.rc("font", size=15)
plt.figure(figsize=(7, 7))

label = ["Has catus", "Hasn't cactus"]
plt.pie(labels["has_cactus"].value_counts(), labels=label, autopct="%.1f%%")
plt.show()



## === cell 3
from zipfile import ZipFile as zf
import os

WORK_DIR = "/kaggle/working/aerial-cactus-identification-extracted"
os.makedirs(WORK_DIR, exist_ok=True)

EXTRACT_ROOT = os.path.join(WORK_DIR, "extracted")
os.makedirs(EXTRACT_ROOT, exist_ok=True)

with zf(PATH + "train.zip") as zipper:
    zipper.extractall(EXTRACT_ROOT)

with zf(PATH + "test.zip") as zipper:
    zipper.extractall(EXTRACT_ROOT)


def _resolve_image_dir(work_dir: str, split: str) -> str:
    """
    Search for a directory named `split` that contains jpg files.
    Handles common Kaggle zip structures (flat, nested competition folder, double-nested).
    """
    candidates = [
        os.path.join(work_dir, split),
        os.path.join(work_dir, "aerial-cactus-identification", split),
        os.path.join(
            work_dir,
            "aerial-cactus-identification",
            "aerial-cactus-identification",
            split,
        ),
    ]
    for c in candidates:
        if os.path.isdir(c) and any(f.lower().endswith(".jpg") for f in os.listdir(c)):
            return c

    best = None
    for root, dirs, files in os.walk(work_dir):
        if os.path.basename(root) == split and any(
            f.lower().endswith(".jpg") for f in files
        ):
            best = root
            break
        if root.count(os.sep) - work_dir.count(os.sep) >= 8:
            dirs[:] = []
    if best is not None:
        return best

    raise FileNotFoundError(
        f"Could not find extracted '{split}' directory under {work_dir}. Checked: {candidates}"
    )


TRAIN_DIR = _resolve_image_dir(EXTRACT_ROOT, "train")
TEST_DIR = _resolve_image_dir(EXTRACT_ROOT, "test")

print("Extracted to:", EXTRACT_ROOT)
print("Resolved TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.isdir(TRAIN_DIR))
print("Resolved TEST_DIR:", TEST_DIR, "exists:", os.path.isdir(TEST_DIR))



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1037540932.py in <cell line: 0>()
     55 
     56 
---> 57 TRAIN_DIR = _resolve_image_dir(EXTRACT_ROOT, "train")
     58 TEST_DIR = _resolve_image_dir(EXTRACT_ROOT, "test")
     59 

/tmp/ipykernel_11/1037540932.py in _resolve_image_dir(work_dir, split)
     50         return best
     51 
---> 52     raise FileNotFoundError(
     53         f"Could not find extracted '{split}' directory under {work_dir}. Checked: {candidates}"
     54     )

FileNotFoundError: Could not find extracted 'train' directory under /kaggle/working/aerial-cactus-identification-extracted/extracted. Checked: ['/kaggle/working/aerial-cactus-identification-extracted/extracted/train', '/kaggle/working/aerial-cactus-identification-extracted/extracted/aerial-cactus-identification/train', '/kaggle/working/aerial-cactus-identification-extracted/extracted/aerial-cactus-identification/aerial-cactus-identification/train']

## === cell 4
import os

n_t = len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])
n_test = len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
print(n_t, n_test, sep="\t")



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/17476455.py in <cell line: 0>()
      1 import os
      2 
----> 3 n_t = len([f for f in os.listdir(TRAIN_DIR) if f.lower().endswith(".jpg")])
      4 n_test = len([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
      5 print(n_t, n_test, sep="\t")

NameError: name 'TRAIN_DIR' is not defined

## === cell 5
import cv2
import matplotlib as mpl
import matplotlib.pyplot as plt
import os

mpl.rc("font", size=7)
plt.figure(figsize=(15, 6))

cac_img_name = labels.loc[labels["has_cactus"] == 1, "id"].iloc[-12:].tolist()

for idx, img_name in enumerate(cac_img_name):
    img_path = os.path.join(TRAIN_DIR, img_name)
    img = cv2.imread(img_path)
    if img is None:
        continue
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    ax = plt.subplot(2, 6, idx + 1)
    ax.imshow(img)
    ax.axis("off")
plt.show()



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/309833543.py in <cell line: 0>()
     10 
     11 for idx, img_name in enumerate(cac_img_name):
---> 12     img_path = os.path.join(TRAIN_DIR, img_name)
     13     img = cv2.imread(img_path)
     14     if img is None:

NameError: name 'TRAIN_DIR' is not defined

## === cell 6
from sklearn.model_selection import train_test_split

train, val = train_test_split(
    labels, test_size=0.1, stratify=labels["has_cactus"], random_state=50
)
print(train.shape)



## === cell 7
import os
import tensorflow as tf
import numpy as np

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

tf.keras.utils.set_random_seed(50)

IMG_SHAPE = (32, 32, 3)
IMG_SIZE = (32, 32)

AUTOTUNE = tf.data.AUTOTUNE


def _read_decode_resize(path: tf.Tensor) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape(IMG_SHAPE)
    return img


def _make_train_example(img_id, y):
    path = tf.strings.join([TRAIN_DIR, "/", img_id])
    img = _read_decode_resize(path)
    y = tf.cast(y, tf.float32)
    y.set_shape(())
    return img, y


def _make_val_example(img_id, y):
    path = tf.strings.join([TRAIN_DIR, "/", img_id])
    img = _read_decode_resize(path)
    y = tf.cast(y, tf.float32)
    y.set_shape(())
    return img, y


ds_train = tf.data.Dataset.from_tensor_slices(
    (train["id"].values.astype(str), train["has_cactus"].values)
).map(_make_train_example, num_parallel_calls=AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices(
    (val["id"].values.astype(str), val["has_cactus"].values)
).map(_make_val_example, num_parallel_calls=AUTOTUNE)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 8
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras import Input
from tensorflow.keras import metrics


def model_1(shape):
    input = Input(shape=shape)
    conv_activation = "relu"
    x = Conv2D(32, 3, padding="same", activation=conv_activation)(input)
    x = MaxPooling2D(2)(x)
    x = Conv2D(64, 3, padding="same", activation=conv_activation)(x)
    x = MaxPooling2D(2)(x)
    x = Flatten()(x)
    x = Dense(1, activation="sigmoid")(x)
    model = Model(input, x)
    return model


model = model_1((32, 32, 3))
model.compile(
    loss="binary_crossentropy", optimizer="adam", metrics=[metrics.binary_accuracy]
)

print("done")



## === cell 9
from tensorflow.keras.utils import plot_model

try:
    plot_model(model, show_shapes=True)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
batch_size = 32
epochs = 10

ds_train_batched = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
ds_val_batched = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)

hist = model.fit(ds_train_batched, validation_data=ds_val_batched, epochs=epochs)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3836326264.py in <cell line: 0>()
      2 epochs = 10
      3 
----> 4 ds_train_batched = ds_train.batch(batch_size).prefetch(tf.data.AUTOTUNE)
      5 ds_val_batched = ds_val.batch(batch_size).prefetch(tf.data.AUTOTUNE)
      6 

NameError: name 'ds_train' is not defined

## === cell 11
import pandas as pd
import tensorflow as tf

test_labels = pd.read_csv(PATH + "sample_submission.csv")


def _make_test_example(img_id):
    path = tf.strings.join([TEST_DIR, "/", img_id])
    img = _read_decode_resize(path)
    return img


ds_test = tf.data.Dataset.from_tensor_slices(test_labels["id"].values.astype(str)).map(
    _make_test_example, num_parallel_calls=tf.data.AUTOTUNE
)

ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

preds = model.predict(ds_test, verbose=0)
preds.shape



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2811230262.py in <cell line: 0>()
     11 
     12 
---> 13 ds_test = tf.data.Dataset.from_tensor_slices(test_labels["id"].values.astype(str)).map(
     14     _make_test_example, num_parallel_calls=tf.data.AUTOTUNE
     15 )

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

/tmp/__autograph_generated_filep4xic5qb.py in tf___make_test_example(img_id)
      8                 do_return = False
      9                 retval_ = ag__.UndefinedReturnValue()
---> 10                 path = ag__.converted_call(ag__.ld(tf).strings.join, ([ag__.ld(TEST_DIR), '/', ag__.ld(img_id)],), None, fscope)
     11                 img = ag__.converted_call(ag__.ld(_read_decode_resize), (ag__.ld(path),), None, fscope)
     12                 try:

NameError: in user code:

    File "/tmp/ipykernel_11/2811230262.py", line 8, in _make_test_example  *
        path = tf.strings.join([TEST_DIR, "/", img_id])

    NameError: name 'TEST_DIR' is not defined


## === cell 12
submissions = pd.read_csv(PATH + "sample_submission.csv")
submissions["has_cactus"] = preds.reshape(-1)
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
print(submissions.head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4029433764.py in <cell line: 0>()
      1 submissions = pd.read_csv(PATH + "sample_submission.csv")
----> 2 submissions["has_cactus"] = preds.reshape(-1)
      3 submissions.to_csv("submission.csv", index=False)
      4 print("Wrote submission.csv with shape:", submissions.shape)
      5 print(submissions.head())

NameError: name 'preds' is not defined

## === cell 13
import shutil
import os

if os.path.isdir(WORK_DIR):
    shutil.rmtree(WORK_DIR)
print("Cleanup done.")
