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

3.12

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
pillow==11.3.0
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

0.4964

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow_io as tfio
from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from PIL import Image
from sklearn.metrics import roc_auc_score
from tensorflow.keras.models import load_model

tf.get_logger().setLevel("ERROR")
np.random.seed(0)
tf.random.set_seed(0)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
input_dir = "/kaggle/input/histopathologic-cancer-detection"

sample_path = os.path.join(input_dir, "sample_submission.csv")
train_labels_path = os.path.join(input_dir, "train_labels.csv")
train_dir = os.path.join(input_dir, "train") + "/"
test_dir = os.path.join(input_dir, "test") + "/"

sample_data = pd.read_csv(sample_path)
train_data = pd.read_csv(train_labels_path)

(train_dir, test_dir, sample_data.shape, train_data.shape)




## === cell 2
def print_short_summary(name, data):
    """
    Prints data head, shape and info.
    Args:
        name (str): name of dataset
        data (dataframe): dataset in a pd.DataFrame format
    """
    print(name)
    print("\n1. Data head:")
    print(data.head())
    print("\n2. Data shape: {}".format(data.shape))
    print("\n3. Data info:")
    data.info()


def print_number_files(dirpath):
    print("{}: {} files".format(dirpath, len(os.listdir(dirpath))))




## === cell 3
print_short_summary("Train data", train_data)



## === cell 4
print_short_summary("Sample data", sample_data)



## === cell 5
print_number_files(train_dir)



## === cell 6
print_number_files(test_dir)



## === cell 7
plt.figure(figsize=(16, 9))
tmp = train_data["label"].value_counts()
sns.barplot(y=["No Cancer", "Cancer"], x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Label")
plt.title("Number of records per label")
plt.show()




## === cell 8
def get_images_to_plot(file_names):
    """
    Returns list of images
    Args:
        file_names: list of filenames
    Returns:
        list of image objects
    """
    return [Image.open(f) for f in file_names]


def get_image_label(dirname, data, labels, n=5):
    dict_img = {}
    for l in labels:
        indexes = data["label"] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp["id"] + ".tif"
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp

    return dict_img




## === cell 9
img_path = train_dir + train_data["id"][0] + ".tif"
img = Image.open(img_path)
print("Original image size: {}".format(img.size))



## === cell 10
data = get_image_label(train_dir, train_data, [0, 1])



## === cell 11
fig, axes = plt.subplots(nrows=2, ncols=5, figsize=(16, 9))

labels = ["No Cancer", "Cancer"]
for i in range(10):
    row = i // 5
    col = i % 5
    axes[row, col].imshow(data[row][col])
    axes[row, col].set_title(labels[row])
    axes[row, col].axis("off")

plt.tight_layout()
plt.show()



## === cell 12
SAMPLE_SIZE = 0.2
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]
cancer = cancer[: int(SAMPLE_SIZE * len(cancer))]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=0
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=0).reset_index(
    drop=True
)

balanced_train_data.shape



## === cell 13
image_paths = train_dir + balanced_train_data["id"] + ".tif"
image_paths = image_paths.values
labels = balanced_train_data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths, labels, test_size=0.25, shuffle=True, random_state=0
)

(X_train.shape, X_test.shape)




## === cell 14
def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images using TensorFlow.
    Decode TIFF, resize to 32x32px, scale pixels to [0,1].
    Ensure 4 channels (RGBA) to match the existing model architecture.
    """
    img_bytes = tf.io.read_file(image_path)

    img = tfio.experimental.image.decode_tiff(img_bytes)  # typically (96,96,3)
    img = tf.cast(img, tf.float32) / 255.0

    img = tf.image.resize(img, [32, 32], method="bilinear", antialias=True)

    alpha = tf.ones([32, 32, 1], dtype=img.dtype)
    img = tf.concat([img, alpha], axis=-1)

    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size, buffer_size, training=True):
    """
    Create a TensorFlow dataset from image paths (and optional labels).
    If training=True, shuffle; if not, keep deterministic order for submission alignment.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, tuple):
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
        )

    if training:
        dataset = dataset.shuffle(
            buffer_size=buffer_size, seed=0, reshuffle_each_iteration=True
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(AUTOTUNE)
    return dataset




## === cell 15
BATCH_SIZE = 64
TRAIN_BUFFER_SIZE = X_train.shape[0]
TEST_BUFFER_SIZE = X_test.shape[0]

train_dataset = get_prefetched_data(
    (X_train, y_train), BATCH_SIZE, TRAIN_BUFFER_SIZE, training=True
)
test_dataset = get_prefetched_data(
    (X_test, y_test), BATCH_SIZE, TEST_BUFFER_SIZE, training=False
)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/346395148.py in <cell line: 0>()
      3 TEST_BUFFER_SIZE = X_test.shape[0]
      4 
----> 5 train_dataset = get_prefetched_data(
      6     (X_train, y_train), BATCH_SIZE, TRAIN_BUFFER_SIZE, training=True
      7 )

/tmp/ipykernel_11/2934225250.py in get_prefetched_data(data, batch_size, buffer_size, training)
     32     if isinstance(data, tuple):
     33         dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
---> 34         dataset = dataset.map(get_decoded_image, num_parallel_calls=AUTOTUNE)
     35     else:
     36         dataset = tf.data.Dataset.from_tensor_slices(data)

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

/tmp/__autograph_generated_filenqrrwx69.py in tf__get_decoded_image(image_path, label)
     14                 retval_ = ag__.UndefinedReturnValue()
     15                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(image_path),), None, fscope)
---> 16                 img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     17                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0
     18                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [32, 32]), dict(method='bilinear', antialias=True), fscope)

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

    File "/tmp/ipykernel_11/2934225250.py", line 11, in get_decoded_image  *
        img = tfio.experimental.image.decode_tiff(img_bytes)  # typically (96,96,3)
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


## === cell 16
def roc_auc_score_(y_true, y_pred):
    """
    Calculate ROC AUC score using sklearn built-in function.
    Used in a model.compile as a custom metric.
    """
    y_true = tf.cast(tf.reshape(y_true, [-1]), tf.int32)
    y_pred = tf.cast(tf.reshape(y_pred, [-1]), tf.float32)

    def _sk_auc(t, p):
        return np.float64(roc_auc_score(t, p))

    out = tf.py_function(_sk_auc, (y_true, y_pred), Tout=tf.float64)
    out.set_shape([])  # critical for Keras metric reduction
    return out




## === cell 17
model_base = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 18
model_drop_bn = models.Sequential(
    [
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(1, activation="sigmoid"),
    ]
)



## === cell 19
model_tuned = models.Sequential(
    [
        layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
        layers.BatchNormalization(),
        layers.MaxPooling2D((4, 4)),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.35),
        layers.Dense(1, activation="sigmoid"),
    ]
)




## === cell 20
def plot_model_scores(scores):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, "b", label="Train score")
    plt.plot(epochs, test_scores, "r", label="Test score")
    plt.title("Train and test ROC AUC scores")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC Score")
    plt.legend()
    plt.grid(True)
    plt.show()


def get_model_results(model_name, model):
    """
    Return tuple of runtime, train and test scores.
    Compile, fit and save model along the way.
    """
    st = time.time()
    model.compile(
        optimizer="adam", loss="binary_crossentropy", metrics=[roc_auc_score_]
    )
    history = model.fit(
        train_dataset, epochs=5, validation_data=test_dataset, verbose=2
    )
    runtime = time.time() - st

    model.save(f"{model_name}.keras")

    train_scores = history.history["roc_auc_score_"]
    test_scores = history.history["val_roc_auc_score_"]
    del model

    return (runtime, (train_scores, test_scores))




## === cell 21
runtime_base, scores_base = get_model_results("base", model_base)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/351769605.py in <cell line: 0>()
----> 1 runtime_base, scores_base = get_model_results("base", model_base)
      2 

/tmp/ipykernel_11/3901848745.py in get_model_results(model_name, model)
     27     )
     28     history = model.fit(
---> 29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )
     31     runtime = time.time() - st

NameError: name 'train_dataset' is not defined

## === cell 22
plot_model_scores(scores_base)



## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3689712647.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base)
      2 

NameError: name 'scores_base' is not defined

## === cell 23
runtime_drop_bn, scores_drop_bn = get_model_results("drop_bn", model_drop_bn)



## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2928168416.py in <cell line: 0>()
----> 1 runtime_drop_bn, scores_drop_bn = get_model_results("drop_bn", model_drop_bn)
      2 

/tmp/ipykernel_11/3901848745.py in get_model_results(model_name, model)
     27     )
     28     history = model.fit(
---> 29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )
     31     runtime = time.time() - st

NameError: name 'train_dataset' is not defined

## === cell 24
plot_model_scores(scores_drop_bn)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3214938728.py in <cell line: 0>()
----> 1 plot_model_scores(scores_drop_bn)
      2 

NameError: name 'scores_drop_bn' is not defined

## === cell 25
runtime_tuned, scores_tuned = get_model_results("tuned", model_tuned)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1937299287.py in <cell line: 0>()
----> 1 runtime_tuned, scores_tuned = get_model_results("tuned", model_tuned)
      2 

/tmp/ipykernel_11/3901848745.py in get_model_results(model_name, model)
     27     )
     28     history = model.fit(
---> 29         train_dataset, epochs=5, validation_data=test_dataset, verbose=2
     30     )
     31     runtime = time.time() - st

NameError: name 'train_dataset' is not defined

## === cell 26
plot_model_scores(scores_tuned)



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/400259038.py in <cell line: 0>()
----> 1 plot_model_scores(scores_tuned)
      2 

NameError: name 'scores_tuned' is not defined

## === cell 27
table = [
    {
        "model": "Base",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_base,
        "train_roc_auc_score": scores_base[0][-1],
        "test_roc_auc_score": scores_base[1][-1],
    },
    {
        "model": "Drop and BN",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_drop_bn,
        "train_roc_auc_score": scores_drop_bn[0][-1],
        "test_roc_auc_score": scores_drop_bn[1][-1],
    },
    {
        "model": "Tuned",
        "sample_size": SAMPLE_SIZE,
        "runtime": runtime_tuned,
        "train_roc_auc_score": scores_tuned[0][-1],
        "test_roc_auc_score": scores_tuned[1][-1],
    },
]

pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime"], ascending=[False, True]
)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3643122313.py in <cell line: 0>()
      3         "model": "Base",
      4         "sample_size": SAMPLE_SIZE,
----> 5         "runtime": runtime_base,
      6         "train_roc_auc_score": scores_base[0][-1],
      7         "test_roc_auc_score": scores_base[1][-1],

NameError: name 'runtime_base' is not defined

## === cell 28
model_tuned_20 = load_model(
    "tuned.keras", custom_objects={"roc_auc_score_": roc_auc_score_}
)



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/4126659689.py in <cell line: 0>()
      1 # Load saved tuned model with custom metric parameter
----> 2 model_tuned_20 = load_model(
      3     "tuned.keras", custom_objects={"roc_auc_score_": roc_auc_score_}
      4 )
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    198         )
    199     elif str(filepath).endswith(".keras"):
--> 200         raise ValueError(
    201             f"File not found: filepath={filepath}. "
    202             "Please ensure the file is an accessible `.keras` "

ValueError: File not found: filepath=tuned.keras. Please ensure the file is an accessible `.keras` zip file.

## === cell 29
submis_files = sorted([f for f in os.listdir(test_dir) if f.endswith(".tif")])
submis_data = np.array([os.path.join(test_dir, f) for f in submis_files])

BATCH_SIZE = 64
SUBMIS_BUFFER_SIZE = submis_data.shape[0]

submis_dataset = get_prefetched_data(
    submis_data, BATCH_SIZE, SUBMIS_BUFFER_SIZE, training=False
)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/2107538734.py in <cell line: 0>()
      5 SUBMIS_BUFFER_SIZE = submis_data.shape[0]
      6 
----> 7 submis_dataset = get_prefetched_data(
      8     submis_data, BATCH_SIZE, SUBMIS_BUFFER_SIZE, training=False
      9 )

/tmp/ipykernel_11/2934225250.py in get_prefetched_data(data, batch_size, buffer_size, training)
     35     else:
     36         dataset = tf.data.Dataset.from_tensor_slices(data)
---> 37         dataset = dataset.map(
     38             lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
     39         )

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

/tmp/__autograph_generated_filey_5ohe00.py in <lambda>(p)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p: ag__.with_function_scope(lambda lscope: ag__.converted_call(get_decoded_image, (p, None), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filey_5ohe00.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p: ag__.with_function_scope(lambda lscope: ag__.converted_call(get_decoded_image, (p, None), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filenqrrwx69.py in tf__get_decoded_image(image_path, label)
     14                 retval_ = ag__.UndefinedReturnValue()
     15                 img_bytes = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(image_path),), None, fscope)
---> 16                 img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img_bytes),), None, fscope)
     17                 img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0
     18                 img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [32, 32]), dict(method='bilinear', antialias=True), fscope)

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

    File "/tmp/ipykernel_11/2934225250.py", line 38, in None  *
        lambda p: get_decoded_image(p, None)
    File "/tmp/ipykernel_11/2934225250.py", line 11, in get_decoded_image  *
        img = tfio.experimental.image.decode_tiff(img_bytes)  # typically (96,96,3)
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


## === cell 30
result_20 = model_tuned_20.predict(submis_dataset, verbose=0).reshape(-1)
result_20 = np.clip(result_20.astype(np.float32), 0.0, 1.0)
result_20[:5], result_20.shape



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3864113907.py in <cell line: 0>()
----> 1 result_20 = model_tuned_20.predict(submis_dataset, verbose=0).reshape(-1)
      2 result_20 = np.clip(result_20.astype(np.float32), 0.0, 1.0)
      3 result_20[:5], result_20.shape
      4 

NameError: name 'model_tuned_20' is not defined

## === cell 31
id_ = np.char.replace(np.array(submis_files), ".tif", "")
label_ = result_20

table = pd.DataFrame({"id": id_, "label": label_})
table.head()



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1632498466.py in <cell line: 0>()
      1 id_ = np.char.replace(np.array(submis_files), ".tif", "")
----> 2 label_ = result_20
      3 
      4 table = pd.DataFrame({"id": id_, "label": label_})
      5 table.head()

NameError: name 'result_20' is not defined

## === cell 32
table.to_csv("submission_20.csv", index=False)
print("Wrote submission_20.csv with shape:", table.shape)
print("Columns:", list(table.columns))
print("Path exists:", os.path.exists("submission_20.csv"))
print("First rows:\n", table.head())

## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2853535095.py in <cell line: 0>()
----> 1 table.to_csv("submission_20.csv", index=False)
      2 print("Wrote submission_20.csv with shape:", table.shape)
      3 print("Columns:", list(table.columns))
      4 print("Path exists:", os.path.exists("submission_20.csv"))
      5 print("First rows:\n", table.head())

NameError: name 'table' is not defined
