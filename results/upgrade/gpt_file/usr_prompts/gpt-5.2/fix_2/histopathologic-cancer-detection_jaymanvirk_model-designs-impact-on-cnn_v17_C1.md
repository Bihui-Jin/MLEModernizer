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

0.7367

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the two root blockers: the protobuf/tensorflow-io import crash and the brittle file discovery that reads non-CSV files. Then I make dataset creation robust for both (paths, labels) and (paths-only) cases, and ensure model training actually runs and the saved model exists before loading. Finally, I generate probabilities (not rounded 0/1) for `label` to match the ROC-AUC metric and write a valid `submission.csv` with the required columns and order.'

# 9. Code solution

## === cell 0
import os
import time
import numpy as np

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import tensorflow as tf

try:
    import tensorflow_io as tfio

    _HAS_TFIO = True
except Exception as e:
    print(
        "Warning: tensorflow_io failed to import; falling back to PIL loader. Error:",
        repr(e),
    )
    tfio = None
    _HAS_TFIO = False

from sklearn.utils import resample
from sklearn.model_selection import train_test_split

from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model

SEED = 0
np.random.seed(SEED)
tf.random.set_seed(SEED)



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

assert os.path.exists(sample_path), f"Missing: {sample_path}"
assert os.path.exists(train_labels_path), f"Missing: {train_labels_path}"
assert os.path.isdir(train_dir), f"Missing: {train_dir}"
assert os.path.isdir(test_dir), f"Missing: {test_dir}"

sample_data = pd.read_csv(sample_path)
train_data = pd.read_csv(train_labels_path)

list_l = [sample_path, train_labels_path, test_dir, train_dir]
list_l



## === cell 2
del list_l




## === cell 3
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




## === cell 4
print_short_summary("Train data", train_data)



## === cell 5
print_short_summary("Sample data", sample_data)



## === cell 6
print_number_files(train_dir)



## === cell 7
print_number_files(test_dir)



## === cell 8
del print_short_summary, print_number_files



## === cell 9
plt.figure(figsize=(16, 9))
tmp = train_data["label"].value_counts()
sns.barplot(y=["No Cancer", "Cancer"], x=tmp.values, orient="h")
plt.xlabel("Number of records")
plt.ylabel("Label")
plt.title("Number of records per label")
plt.show()



## === cell 10
del tmp




## === cell 11
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
    """
    Return dictionary with label-imagepath
    Args:
        dirname: name of the directory
        data: dataset of file names
        labels: list of labels
        n (opt): number of images per label
    Returns:
        dict_img: dictionary with label-imagepath pairs
    """
    dict_img = {}
    for l in labels:
        indexes = data["label"] == l
        tmp = data[indexes][:n]
        tmp = dirname + tmp["id"] + ".tif"
        tmp = tmp.values
        tmp = get_images_to_plot(tmp)
        dict_img[l] = tmp

    return dict_img




## === cell 12
img_path = train_dir + train_data["id"][0] + ".tif"
img = Image.open(img_path)
print("Original image size: {}".format(img.size))



## === cell 13
data = get_image_label(train_dir, train_data, [0, 1])



## === cell 14
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



## === cell 15
del get_images_to_plot, get_image_label, img_path, img
del data, fig, axes, labels, row, col



## === cell 16
no_cancer = train_data[train_data["label"] == 0]
cancer = train_data[train_data["label"] == 1]

no_cancer_downsampled = resample(
    no_cancer, replace=False, n_samples=len(cancer), random_state=SEED
)

balanced_train_data = pd.concat([no_cancer_downsampled, cancer])
balanced_train_data = balanced_train_data.sample(frac=1, random_state=SEED).reset_index(
    drop=True
)



## === cell 17
del no_cancer, cancer, no_cancer_downsampled



## === cell 18
image_paths = train_dir + balanced_train_data["id"] + ".tif"
image_paths = image_paths.values

labels = balanced_train_data["label"].values

X_train, X_test, y_train, y_test = train_test_split(
    image_paths,
    labels,
    test_size=0.25,
    shuffle=True,
    random_state=SEED,
    stratify=labels,
)



## === cell 19
del image_paths, labels




## === cell 20
def _decode_tiff_to_tensor(image_path):
    """Decode .tif into float32 tensor in [0,1], resized to 32x32, with 4 channels."""
    if _HAS_TFIO:
        img = tf.io.read_file(image_path)
        img = tfio.experimental.image.decode_tiff(
            img
        )  # usually returns [H,W,C], often C=4
        img = tf.image.resize(img, [32, 32])
        img = tf.cast(img, tf.float32) / 255.0
        c = tf.shape(img)[-1]
        img = tf.cond(
            tf.equal(c, 4),
            lambda: img,
            lambda: tf.cond(
                tf.equal(c, 3),
                lambda: tf.concat([img, tf.ones_like(img[..., :1])], axis=-1),
                lambda: tf.image.grayscale_to_rgb(img)[
                    ..., :3
                ],  # will be 3, then add alpha below
            ),
        )
        c2 = tf.shape(img)[-1]
        img = tf.cond(
            tf.equal(c2, 4),
            lambda: img,
            lambda: tf.concat([img[..., :3], tf.ones_like(img[..., :1])], axis=-1),
        )
        return img

    def _py_load(p):
        p = p.decode("utf-8")
        im = Image.open(p).convert("RGBA").resize((32, 32))
        arr = np.asarray(im, dtype=np.float32) / 255.0  # (32,32,4)
        return arr

    img = tf.py_function(func=_py_load, inp=[image_path], Tout=tf.float32)
    img.set_shape((32, 32, 4))
    return img


def get_decoded_image(image_path, label=None):
    """
    Load and preprocess images.
    Decode image with 4 channels RGBA.
    Resize image to 32x32px.
    Scale pixels from 0 to 1.
    """
    img = _decode_tiff_to_tensor(image_path)
    return img if label is None else (img, label)


def get_prefetched_data(data, batch_size):
    """
    Create a TensorFlow dataset from image paths and labels or paths-only.
    """
    AUTOTUNE = tf.data.AUTOTUNE

    if isinstance(data, (tuple, list)) and len(data) == 2:
        dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
        dataset = dataset.map(
            lambda p, y: get_decoded_image(p, y), num_parallel_calls=AUTOTUNE
        )
    else:
        dataset = tf.data.Dataset.from_tensor_slices(data)
        dataset = dataset.map(
            lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
        )

    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(buffer_size=AUTOTUNE)
    return dataset




## === cell 21
BATCH_SIZE = 128

train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE)
test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE)



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/3055837797.py in <cell line: 0>()
      1 BATCH_SIZE = 128
      2 
----> 3 train_dataset = get_prefetched_data((X_train, y_train), BATCH_SIZE)
      4 test_dataset = get_prefetched_data((X_test, y_test), BATCH_SIZE)
      5 

/tmp/ipykernel_11/1125112701.py in get_prefetched_data(data, batch_size)
     62     if isinstance(data, (tuple, list)) and len(data) == 2:
     63         dataset = tf.data.Dataset.from_tensor_slices((data[0], data[1]))
---> 64         dataset = dataset.map(
     65             lambda p, y: get_decoded_image(p, y), num_parallel_calls=AUTOTUNE
     66         )

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

/tmp/__autograph_generated_filex64p2gfr.py in <lambda>(p, y)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y: ag__.with_function_scope(lambda lscope: ag__.converted_call(get_decoded_image, (p, y), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filex64p2gfr.py in <lambda>(lscope)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p, y: ag__.with_function_scope(lambda lscope: ag__.converted_call(get_decoded_image, (p, y), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filep0t3tg30.py in tf__get_decoded_image(image_path, label)
     14                 do_return = False
     15                 retval_ = ag__.UndefinedReturnValue()
---> 16                 img = ag__.converted_call(ag__.ld(_decode_tiff_to_tensor), (ag__.ld(image_path),), None, fscope)
     17                 try:
     18                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filets5awszj.py in tf___decode_tiff_to_tensor(image_path)
     64                 c = ag__.Undefined('c')
     65                 _py_load = ag__.Undefined('_py_load')
---> 66                 ag__.if_stmt(ag__.ld(_HAS_TFIO), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     67                 return fscope.ret(retval_, do_return)
     68         return tf___decode_tiff_to_tensor

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

/tmp/__autograph_generated_filets5awszj.py in if_body()
     20                     nonlocal do_return, retval_
     21                     img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(image_path),), None, fscope)
---> 22                     img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img),), None, fscope)
     23                     img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [32, 32]), None, fscope)
     24                     img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

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

    File "/tmp/ipykernel_11/1125112701.py", line 65, in None  *
        lambda p, y: get_decoded_image(p, y)
    File "/tmp/ipykernel_11/1125112701.py", line 51, in get_decoded_image  *
        img = _decode_tiff_to_tensor(image_path)
    File "/tmp/ipykernel_11/1125112701.py", line 5, in _decode_tiff_to_tensor  *
        img = tfio.experimental.image.decode_tiff(
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


## === cell 22
del X_train, y_train, X_test, y_test




## === cell 23
def get_model_base():
    """
    Return base model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 24
def get_model_base_deep():
    """
    Return deeper model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Conv2D(32, (3, 3), activation="relu"),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 25
def get_model_base_wide():
    """
    Return wider model architecture
    """
    model_drop_bn = models.Sequential(
        [
            layers.Conv2D(64, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model_drop_bn




## === cell 26
def get_model_base_maxpool():
    """
    Return maxpool model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.MaxPooling2D((2, 2), strides=(2, 2)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 27
def get_model_base_dropout():
    """
    Return dropout model architecture
    """
    model = models.Sequential(
        [
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=(32, 32, 4)),
            layers.Flatten(),
            layers.Dense(32, activation="relu"),
            layers.Dropout(0.25),
            layers.Dense(1, activation="sigmoid"),
        ]
    )

    return model




## === cell 28
def get_compiled_model(func):
    """
    Create model to be trained with a multi-GPU strategy.
    """
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        strategy = tf.distribute.MirroredStrategy()
        print("Number of devices: {}".format(strategy.num_replicas_in_sync))
    else:
        strategy = tf.distribute.OneDeviceStrategy(device="/cpu:0")
        print("No GPU available, falling back to CPU.")

    with strategy.scope():
        compiled_model = func()
        compiled_model.compile(
            optimizer=tf.keras.optimizers.Adam(),
            loss=tf.keras.losses.BinaryCrossentropy(),
            metrics=[tf.keras.metrics.AUC(name="auc")],
        )

    return compiled_model




## === cell 29
def plot_model_scores(scores, model_name):
    """
    Plot train and test ROC AUC scores of a model by epoch
    """
    train_scores, test_scores = scores
    epochs = range(1, len(train_scores) + 1)

    plt.figure(figsize=(16, 9))
    plt.plot(epochs, train_scores, label="Train score")
    plt.plot(epochs, test_scores, label="Test score")
    plt.title("Train and test ROC AUC scores of the {}".format(model_name))
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
    m = get_compiled_model(model)

    st = time.time()
    history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
    runtime = time.time() - st

    m.save("{}.h5".format(model_name))

    train_scores = history.history["auc"]
    test_scores = history.history["val_auc"]

    tf.keras.backend.clear_session()

    return (runtime, (train_scores, test_scores))




## === cell 30
runtime_base, scores_base = get_model_results("model_base", get_model_base)



## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3774139332.py in <cell line: 0>()
----> 1 runtime_base, scores_base = get_model_results("model_base", get_model_base)
      2 

/tmp/ipykernel_11/2773462807.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

NameError: name 'train_dataset' is not defined

## === cell 31
plot_model_scores(scores_base, "base model")



## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2757851445.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base, "base model")
      2 

NameError: name 'scores_base' is not defined

## === cell 32
runtime_base_deep, scores_base_deep = get_model_results(
    "model_base_deep", get_model_base_deep
)



## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1501219057.py in <cell line: 0>()
----> 1 runtime_base_deep, scores_base_deep = get_model_results(
      2     "model_base_deep", get_model_base_deep
      3 )
      4 

/tmp/ipykernel_11/2773462807.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

NameError: name 'train_dataset' is not defined

## === cell 33
plot_model_scores(scores_base_deep, "base + additional layers model")



## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435582006.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_deep, "base + additional layers model")
      2 

NameError: name 'scores_base_deep' is not defined

## === cell 34
runtime_base_wide, scores_base_wide = get_model_results(
    "model_base_wide", get_model_base_wide
)



## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/389434075.py in <cell line: 0>()
----> 1 runtime_base_wide, scores_base_wide = get_model_results(
      2     "model_base_wide", get_model_base_wide
      3 )
      4 

/tmp/ipykernel_11/2773462807.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

NameError: name 'train_dataset' is not defined

## === cell 35
plot_model_scores(scores_base_wide, "base + wider layers model")



## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/178024283.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_wide, "base + wider layers model")
      2 

NameError: name 'scores_base_wide' is not defined

## === cell 36
runtime_base_maxpool, scores_base_maxpool = get_model_results(
    "model_base_maxpool", get_model_base_maxpool
)



## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3139894174.py in <cell line: 0>()
----> 1 runtime_base_maxpool, scores_base_maxpool = get_model_results(
      2     "model_base_maxpool", get_model_base_maxpool
      3 )
      4 

/tmp/ipykernel_11/2773462807.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

NameError: name 'train_dataset' is not defined

## === cell 37
plot_model_scores(scores_base_maxpool, "base + max pooling model")



## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3164274992.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_maxpool, "base + max pooling model")
      2 

NameError: name 'scores_base_maxpool' is not defined

## === cell 38
runtime_base_dropout, scores_base_dropout = get_model_results(
    "model_base_dropout", get_model_base_dropout
)



## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1796779764.py in <cell line: 0>()
----> 1 runtime_base_dropout, scores_base_dropout = get_model_results(
      2     "model_base_dropout", get_model_base_dropout
      3 )
      4 

/tmp/ipykernel_11/2773462807.py in get_model_results(model_name, model)
     25 
     26     st = time.time()
---> 27     history = m.fit(train_dataset, epochs=5, validation_data=test_dataset, verbose=2)
     28     runtime = time.time() - st
     29 

NameError: name 'train_dataset' is not defined

## === cell 39
plot_model_scores(scores_base_dropout, "base + dropout model")



## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/110224612.py in <cell line: 0>()
----> 1 plot_model_scores(scores_base_dropout, "base + dropout model")
      2 

NameError: name 'scores_base_dropout' is not defined

## === cell 40
results = [
    ("Base", runtime_base, scores_base),
    ("Base + Add. layers", runtime_base_deep, scores_base_deep),
    ("Base + Wider layers", runtime_base_wide, scores_base_wide),
    ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),
    ("Base + Dropout", runtime_base_dropout, scores_base_dropout),
]
table = []
for i in range(len(results)):
    tmp = {
        "model": results[i][0],
        "runtime (sec)": results[i][1],
        "train_roc_auc_score": results[i][2][0][-1],
        "test_roc_auc_score": results[i][2][1][-1],
    }
    table.append(tmp)

pd.DataFrame(table).sort_values(
    by=["test_roc_auc_score", "runtime (sec)"], ascending=[False, True]
)



## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2027199582.py in <cell line: 0>()
      1 results = [
----> 2     ("Base", runtime_base, scores_base),
      3     ("Base + Add. layers", runtime_base_deep, scores_base_deep),
      4     ("Base + Wider layers", runtime_base_wide, scores_base_wide),
      5     ("Base + Max pooling", runtime_base_maxpool, scores_base_maxpool),

NameError: name 'runtime_base' is not defined

## === cell 41
del tmp, table



## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2435447492.py in <cell line: 0>()
      1 # Keep datasets and paths for submission step; only delete local loop vars.
----> 2 del tmp, table
      3 

NameError: name 'tmp' is not defined

## === cell 42
model = load_model("model_base_deep.h5", compile=False)



## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/259727692.py in <cell line: 0>()
      1 # Load top performed model (as in original notebook intent).
      2 # If you prefer a different one, adjust the filename only.
----> 3 model = load_model("model_base_deep.h5", compile=False)
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model_base_deep.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 43
submis_data = test_dir + sample_data["id"] + ".tif"
submis_data = submis_data.values

submis_dataset = get_prefetched_data(submis_data, BATCH_SIZE)



## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NotImplementedError                       Traceback (most recent call last)
/tmp/ipykernel_11/1189194055.py in <cell line: 0>()
      2 submis_data = submis_data.values
      3 
----> 4 submis_dataset = get_prefetched_data(submis_data, BATCH_SIZE)
      5 

/tmp/ipykernel_11/1125112701.py in get_prefetched_data(data, batch_size)
     67     else:
     68         dataset = tf.data.Dataset.from_tensor_slices(data)
---> 69         dataset = dataset.map(
     70             lambda p: get_decoded_image(p, None), num_parallel_calls=AUTOTUNE
     71         )

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

/tmp/__autograph_generated_filei7k23us4.py in <lambda>(p)
      3 
      4     def inner_factory(ag__):
----> 5         tf__lam = lambda p: ag__.with_function_scope(lambda lscope: ag__.converted_call(get_decoded_image, (p, None), None, lscope), 'lscope', ag__.STD)
      6         return tf__lam
      7     return inner_factory

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/function_wrappers.py in with_function_scope(thunk, scope_name, options)
    111   """Inline version of the FunctionScope context manager."""
    112   with FunctionScope('lambda_', scope_name, options) as scope:
--> 113     return thunk(scope)

/tmp/__autograph_generated_filei7k23us4.py in <lambda>(lscope)
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

/tmp/__autograph_generated_filep0t3tg30.py in tf__get_decoded_image(image_path, label)
     14                 do_return = False
     15                 retval_ = ag__.UndefinedReturnValue()
---> 16                 img = ag__.converted_call(ag__.ld(_decode_tiff_to_tensor), (ag__.ld(image_path),), None, fscope)
     17                 try:
     18                     do_return = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    439         result = converted_f(*effective_args, **kwargs)
    440       else:
--> 441         result = converted_f(*effective_args)
    442     except Exception as e:
    443       _attach_error_metadata(e, converted_f)

/tmp/__autograph_generated_filets5awszj.py in tf___decode_tiff_to_tensor(image_path)
     64                 c = ag__.Undefined('c')
     65                 _py_load = ag__.Undefined('_py_load')
---> 66                 ag__.if_stmt(ag__.ld(_HAS_TFIO), if_body, else_body, get_state, set_state, ('do_return', 'retval_'), 2)
     67                 return fscope.ret(retval_, do_return)
     68         return tf___decode_tiff_to_tensor

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

/tmp/__autograph_generated_filets5awszj.py in if_body()
     20                     nonlocal do_return, retval_
     21                     img = ag__.converted_call(ag__.ld(tf).io.read_file, (ag__.ld(image_path),), None, fscope)
---> 22                     img = ag__.converted_call(ag__.ld(tfio).experimental.image.decode_tiff, (ag__.ld(img),), None, fscope)
     23                     img = ag__.converted_call(ag__.ld(tf).image.resize, (ag__.ld(img), [32, 32]), None, fscope)
     24                     img = ag__.converted_call(ag__.ld(tf).cast, (ag__.ld(img), ag__.ld(tf).float32), None, fscope) / 255.0

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

    File "/tmp/ipykernel_11/1125112701.py", line 70, in None  *
        lambda p: get_decoded_image(p, None)
    File "/tmp/ipykernel_11/1125112701.py", line 51, in get_decoded_image  *
        img = _decode_tiff_to_tensor(image_path)
    File "/tmp/ipykernel_11/1125112701.py", line 5, in _decode_tiff_to_tensor  *
        img = tfio.experimental.image.decode_tiff(
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


## === cell 44
results = model.predict(submis_dataset, verbose=1)



## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/763274262.py in <cell line: 0>()
----> 1 results = model.predict(submis_dataset, verbose=1)
      2 

NameError: name 'model' is not defined

## === cell 45
sample_data["label"] = np.ravel(results).astype(np.float32)



## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1850854202.py in <cell line: 0>()
      1 # SCORE FIX: competition expects probabilities; rounding to 0/1 hurts ROC-AUC.
----> 2 sample_data["label"] = np.ravel(results).astype(np.float32)
      3 

NameError: name 'results' is not defined

## === cell 46
sample_data.head()



## === cell 47
submission = sample_data[["id", "label"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)



## === cell 48
del submis_data, submis_dataset, submission

## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1827531951.py in <cell line: 0>()
----> 1 del submis_data, submis_dataset, submission

NameError: name 'submis_dataset' is not defined
