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

No external packages required in the script and installed.

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

0.9457521634886392

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import warnings

warnings.simplefilter("ignore")

import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow.keras import models, layers
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.applications import Xception
from tensorflow.keras.optimizers import Adam

tf.get_logger().setLevel("ERROR")
print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass



## === cell 2
WORK_DIR = "../input/ranzcr-clip-catheter-line-classification"
print("WORK_DIR exists:", os.path.exists(WORK_DIR))
print("WORK_DIR sample:", os.listdir(WORK_DIR)[:10])



## === cell 3
train_dir = os.path.join(WORK_DIR, "train")
test_dir = os.path.join(WORK_DIR, "test")
print("Train dir exists:", os.path.exists(train_dir))
print("Test dir exists:", os.path.exists(test_dir))



## === cell 4
train = pd.read_csv(os.path.join(WORK_DIR, "train.csv"))
ss = pd.read_csv(os.path.join(WORK_DIR, "sample_submission.csv"))

label_cols = [c for c in train.columns if c not in ["StudyInstanceUID", "PatientID"]]
print("Number of label columns:", len(label_cols))
print("Labels:\n", label_cols)

train_images = (WORK_DIR + "/train/" + train["StudyInstanceUID"] + ".jpg").astype(str)
test_images = (WORK_DIR + "/test/" + ss["StudyInstanceUID"] + ".jpg").astype(str)

labels = train[label_cols].values.astype(np.float32)

train_annot = pd.read_csv(os.path.join(WORK_DIR, "train_annotations.csv"))
train.head()



## === cell 5
print("Skipping label count plots for runtime.")



## === cell 6
BATCH_SIZE = 8
EPOCHS = 3  # unchanged

TARGET_SIZE = 299

print(
    "Configured BATCH_SIZE:", BATCH_SIZE, "EPOCHS:", EPOCHS, "TARGET_SIZE:", TARGET_SIZE
)



## === cell 7
from tensorflow.keras.applications.xception import (
    preprocess_input as xception_preprocess,
)


@tf.function(reduce_retracing=True)
def _decode_xception(path, target_h, target_w):
    file_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(file_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.image.resize(
        img,
        (target_h, target_w),
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img.set_shape([None, None, 3])
    img = xception_preprocess(img)
    img.set_shape([target_h, target_w, 3])
    return img


@tf.function(reduce_retracing=True)
def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    return tf.stack([tf.cast(SEED, tf.int32), tf.cast(h, tf.int32)], axis=0)


@tf.function(reduce_retracing=True)
def _augment_from_path(path, img):
    s = _seed_from_path(path)

    img2 = tf.image.stateless_random_flip_left_right(
        img, seed=s + tf.constant([0, 10], tf.int32)
    )
    img2 = tf.image.stateless_random_flip_up_down(
        img2, seed=s + tf.constant([0, 11], tf.int32)
    )

    r = tf.random.stateless_uniform(
        [],
        seed=s + tf.constant([0, 12], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img2 = tf.image.rot90(img2, k=r)

    u0 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 13], tf.int32))
    img2 = tf.cond(
        u0 >= 0.6,
        lambda: tf.image.stateless_random_saturation(
            img2, lower=0.85, upper=1.15, seed=s + tf.constant([0, 33], tf.int32)
        ),
        lambda: img2,
    )

    u1 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 14], tf.int32))
    img2 = tf.cond(
        u1 >= 0.6,
        lambda: tf.image.stateless_random_contrast(
            img2, lower=0.85, upper=1.15, seed=s + tf.constant([0, 44], tf.int32)
        ),
        lambda: img2,
    )

    u2 = tf.random.stateless_uniform([], seed=s + tf.constant([0, 15], tf.int32))
    img2 = tf.cond(
        u2 >= 0.4,
        lambda: tf.image.stateless_random_brightness(
            img2, max_delta=0.1, seed=s + tf.constant([0, 55], tf.int32)
        ),
        lambda: img2,
    )
    return img2


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    augment=True,
    repeat=True,
    shuffle=1024,
    cache_decoded=False,
    cache_decoded_path=None,
):
    AUTO = tf.data.AUTOTUNE
    with_labels = labels is not None

    opts = tf.data.Options()
    opts.experimental_deterministic = False
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    th = tf.constant(TARGET_SIZE, tf.int32)
    tw = tf.constant(TARGET_SIZE, tf.int32)

    if with_labels:
        dset = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)
        if shuffle:
            dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

        @tf.function(reduce_retracing=True)
        def _decode_keep_path(p, y):
            img = _decode_xception(p, th, tw)
            return p, img, y

        dset = dset.map(_decode_keep_path, num_parallel_calls=AUTO, deterministic=False)

        if cache_decoded:
            dset = (
                dset.cache(cache_decoded_path) if cache_decoded_path else dset.cache()
            )

        if repeat:
            dset = dset.repeat()

        if augment:

            @tf.function(reduce_retracing=True)
            def _aug(p, img, y):
                return _augment_from_path(p, img), y

            dset = dset.map(_aug, num_parallel_calls=AUTO, deterministic=False)

        @tf.function(reduce_retracing=True)
        def _drop_path(p, img, y):
            return img, y

        dset = dset.map(_drop_path, num_parallel_calls=AUTO, deterministic=False)

    else:
        dset = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
        if shuffle:
            dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

        @tf.function(reduce_retracing=True)
        def _decode_keep_path(p):
            img = _decode_xception(p, th, tw)
            return p, img

        dset = dset.map(_decode_keep_path, num_parallel_calls=AUTO, deterministic=False)

        if cache_decoded:
            dset = (
                dset.cache(cache_decoded_path) if cache_decoded_path else dset.cache()
            )

        if repeat:
            dset = dset.repeat()

        if augment:

            @tf.function(reduce_retracing=True)
            def _aug(p, img):
                return p, _augment_from_path(p, img)

            dset = dset.map(_aug, num_parallel_calls=AUTO, deterministic=False)

        @tf.function(reduce_retracing=True)
        def _drop_path(p, img):
            return img

        dset = dset.map(_drop_path, num_parallel_calls=AUTO, deterministic=False)

    dset = dset.apply(tf.data.experimental.ignore_errors())
    dset = dset.batch(bsize, drop_remainder=False)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 8
x_train, x_valid, y_train, y_valid = train_test_split(
    train_images.values, labels, test_size=0.15, random_state=SEED, shuffle=True
)

STEPS_PER_EPOCH = int(np.ceil(len(x_train) / BATCH_SIZE))
VALIDATION_STEPS = int(np.ceil(len(x_valid) / BATCH_SIZE))
print("Train size:", len(x_train), "Valid size:", len(x_valid))
print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)

train_ds = build_dataset(
    x_train,
    y_train,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=2048,
    augment=True,
    cache_decoded=False,
    cache_decoded_path=None,
)

valid_cache_path = os.path.join("/kaggle/working", "valid_decoded_cache")
valid_ds = build_dataset(
    x_valid,
    y_valid,
    bsize=BATCH_SIZE,
    repeat=True,
    shuffle=False,
    augment=False,
    cache_decoded=True,
    cache_decoded_path=valid_cache_path,
)

test_cache_path = os.path.join("/kaggle/working", "test_decoded_cache")
test_ds = build_dataset(
    test_images.values,
    labels=None,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache_decoded=True,
    cache_decoded_path=test_cache_path,
)

print(train_ds, valid_ds, test_ds)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/919444290.py in <cell line: 0>()
      8 print("STEPS_PER_EPOCH:", STEPS_PER_EPOCH, "VALIDATION_STEPS:", VALIDATION_STEPS)
      9 
---> 10 train_ds = build_dataset(
     11     x_train,
     12     y_train,

/tmp/ipykernel_11/2926121076.py in build_dataset(paths, labels, bsize, augment, repeat, shuffle, cache_decoded, cache_decoded_path)
    123             return p, img, y
    124 
--> 125         dset = dset.map(_decode_keep_path, num_parallel_calls=AUTO, deterministic=False)
    126 
    127         if cache_decoded:

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

/tmp/__autograph_generated_file3cltj27f.py in tf___decode_keep_path(p, y)
     10                 do_return = False
     11                 retval_ = ag__.UndefinedReturnValue()
---> 12                 img = ag__.converted_call(ag__.ld(_decode_xception), (ag__.ld(p), ag__.ld(th), ag__.ld(tw)), None, fscope)
     13                 try:
     14                     do_return = True

/tmp/__autograph_generated_filesw9hnkp6.py in tf___decode_xception(path, target_h, target_w)
     14                 ag__.converted_call(ag__.ld(img).set_shape, ([None, None, 3],), None, fscope)
     15                 img = ag__.converted_call(ag__.ld(xception_preprocess), (ag__.ld(img),), None, fscope)
---> 16                 ag__.converted_call(ag__.ld(img).set_shape, ([ag__.ld(target_h), ag__.ld(target_w), 3],), None, fscope)
     17                 try:
     18                     do_return = True

TypeError: in user code:

    File "/tmp/ipykernel_11/2926121076.py", line 122, in _decode_keep_path  *
        img = _decode_xception(p, th, tw)
    File "/tmp/ipykernel_11/2926121076.py", line 22, in _decode_xception  *
        img.set_shape([target_h, target_w, 3])

    TypeError: Dimension value must be integer or None or have an __index__ method, got value '<tf.Tensor 'target_h:0' shape=() dtype=int32>' with type '<class 'tensorflow.python.framework.ops.SymbolicTensor'>'


## === cell 9
inputs = layers.Input(shape=(TARGET_SIZE, TARGET_SIZE, 3))
base = Xception(
    include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
)
x = base.output
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(len(label_cols), activation="sigmoid")(x)
model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    jit_compile=True,
    steps_per_execution=32,
)

model.summary()



## === cell 10
ckpt_path = "xception_ranzcr.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_loss",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
    ReduceLROnPlateau(
        monitor="val_loss", factor=0.5, patience=1, min_lr=1e-6, verbose=1
    ),
]

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=STEPS_PER_EPOCH,
    validation_steps=VALIDATION_STEPS,
    callbacks=callbacks,
    verbose=2,
)

if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3661602042.py in <cell line: 0>()
     14 
     15 history = model.fit(
---> 16     train_ds,
     17     validation_data=valid_ds,
     18     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 11
pred = model.predict(test_ds, verbose=1)
print("Pred shape:", pred.shape)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1137278307.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds, verbose=1)
      2 print("Pred shape:", pred.shape)
      3 

NameError: name 'test_ds' is not defined

## === cell 12
sub = ss.copy()

for c in label_cols:
    if c not in sub.columns:
        sub[c] = 0.0

sub = sub[["StudyInstanceUID"] + label_cols]

sub.loc[:, label_cols] = pred.astype(np.float32)

sub[label_cols] = sub[label_cols].clip(0.0, 1.0)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
sub.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2819097816.py in <cell line: 0>()
      7 sub = sub[["StudyInstanceUID"] + label_cols]
      8 
----> 9 sub.loc[:, label_cols] = pred.astype(np.float32)
     10 
     11 sub[label_cols] = sub[label_cols].clip(0.0, 1.0)

NameError: name 'pred' is not defined

## === cell 13
assert os.path.exists("submission.csv")
check = pd.read_csv("submission.csv")
assert check.shape[0] == ss.shape[0]
assert check.columns[0] == "StudyInstanceUID"
for c in label_cols:
    assert c in check.columns
print("Submission looks valid.")

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/1168335203.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 check = pd.read_csv("submission.csv")
      3 assert check.shape[0] == ss.shape[0]
      4 assert check.columns[0] == "StudyInstanceUID"
      5 for c in label_cols:

AssertionError:
