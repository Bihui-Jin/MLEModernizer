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

0.9427667068533316

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

import random
import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

tf.config.threading.set_intra_op_parallelism_threads(0)
tf.config.threading.set_inter_op_parallelism_threads(0)

try:
    for _gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(_gpu, True)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_56/3983520389.py in <cell line: 0>()
      8 import numpy as np
      9 import pandas as pd
---> 10 import tensorflow as tf
     11 from sklearn.model_selection import train_test_split
     12 

/usr/local/lib/python3.11/dist-packages/tensorflow/__init__.py in <module>
     47 _tf2.enable()
     48 
---> 49 from tensorflow._api.v2 import __internal__
     50 from tensorflow._api.v2 import __operators__
     51 from tensorflow._api.v2 import audio

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow._api.v2.__internal__ import autograph
      9 from tensorflow._api.v2.__internal__ import decorator
     10 from tensorflow._api.v2.__internal__ import dispatch

/usr/local/lib/python3.11/dist-packages/tensorflow/_api/v2/__internal__/autograph/__init__.py in <module>
      6 import sys as _sys
      7 
----> 8 from tensorflow.python.autograph.core.ag_ctx import control_status_ctx # line: 34
      9 from tensorflow.python.autograph.impl.api import tf_convert # line: 493

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/core/ag_ctx.py in <module>
     19 import threading
     20 
---> 21 from tensorflow.python.autograph.utils import ag_logging
     22 from tensorflow.python.util.tf_export import tf_export
     23 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/__init__.py in <module>
     15 """Utility module that contains APIs usable in the generated code."""
     16 
---> 17 from tensorflow.python.autograph.utils.context_managers import control_dependency_on_returns
     18 from tensorflow.python.autograph.utils.misc import alias_tensors
     19 from tensorflow.python.autograph.utils.tensor_list import dynamic_list_append

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/utils/context_managers.py in <module>
     17 import contextlib
     18 
---> 19 from tensorflow.python.framework import ops
     20 from tensorflow.python.ops import tensor_array_ops
     21 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in <module>
     31 
     32 from google.protobuf import message
---> 33 from tensorflow.core.framework import attr_value_pb2
     34 from tensorflow.core.framework import full_type_pb2
     35 from tensorflow.core.framework import function_pb2

/usr/local/lib/python3.11/dist-packages/tensorflow/core/framework/attr_value_pb2.py in <module>
      3 # source: tensorflow/core/framework/attr_value.proto
      4 """Generated protocol buffer code."""
----> 5 from google.protobuf.internal import builder as _builder
      6 from google.protobuf import descriptor as _descriptor
      7 from google.protobuf import descriptor_pool as _descriptor_pool

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/builder.py in <module>
     16 
     17 from google.protobuf.internal import enum_type_wrapper
---> 18 from google.protobuf.internal import python_message
     19 from google.protobuf import message as _message
     20 from google.protobuf import reflection as _reflection

/usr/local/lib/python3.11/dist-packages/google/protobuf/internal/python_message.py in <module>
     36 import weakref
     37 
---> 38 from google.protobuf import descriptor as descriptor_mod
     39 from google.protobuf import message as message_mod
     40 from google.protobuf import text_format

/usr/local/lib/python3.11/dist-packages/google/protobuf/descriptor.py in <module>
     27   # TODO: Remove this import after fix api_implementation
     28   if _message is None:
---> 29     from google.protobuf.pyext import _message
     30   _USE_C_DESCRIPTORS = True
     31 

ImportError: cannot import name '_message' from 'google.protobuf.pyext' (/usr/local/lib/python3.11/dist-packages/google/protobuf/pyext/__init__.py)

## === cell 1
IMAGE_SIZE = 512




## === cell 2
def auto_select_accelerator():
    """
    TPU if available; otherwise default strategy (GPU/CPU).
    """
    try:
        tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
        tf.config.experimental_connect_to_cluster(tpu)
        tf.tpu.experimental.initialize_tpu_system(tpu)
        strategy = tf.distribute.TPUStrategy(tpu)
        print("Running on TPU:", tpu.master())
    except Exception:
        strategy = tf.distribute.get_strategy()
    print(f"Running on {strategy.num_replicas_in_sync} replicas")
    return strategy


def build_decoder(with_labels=True, target_size=(IMAGE_SIZE, IMAGE_SIZE), ext="jpg"):
    @tf.function
    def decode(path):
        file_bytes = tf.io.read_file(path)
        if ext == "png":
            img = tf.image.decode_png(file_bytes, channels=3)
        elif ext in ["jpg", "jpeg"]:
            img = tf.io.decode_jpeg(file_bytes, channels=3, dct_method="INTEGER_FAST")
        else:
            raise ValueError("Image extension not supported")

        img = tf.cast(img, tf.float32) / 255.0
        img = tf.image.resize(img, target_size, antialias=False)
        return img

    @tf.function
    def decode_with_labels(path, label):
        return decode(path), label

    return decode_with_labels if with_labels else decode


def build_augmenter(with_labels=True):
    @tf.function
    def augment(img):
        img = tf.image.random_flip_left_right(img)
        img = tf.image.random_flip_up_down(img)
        return img

    @tf.function
    def augment_with_labels(img, label):
        return augment(img), label

    return augment_with_labels if with_labels else augment


def _ensure_cache_dir(cache_dir: str):
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)


def build_dataset(
    paths,
    labels=None,
    bsize=32,
    cache=True,
    decode_fn=None,
    augment_fn=None,
    augment=True,
    repeat=False,
    shuffle=1024,
    cache_dir="",
    drop_remainder=False,
    ignore_errors=False,
):
    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass
    try:
        options.experimental_optimization.map_and_batch_fusion = True
    except Exception:
        pass
    try:
        options.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    try:
        options.experimental_optimization.autotune_buffers = True
    except Exception:
        pass

    if decode_fn is None:
        decode_fn = build_decoder(labels is not None)

    if augment_fn is None:
        augment_fn = build_augmenter(labels is not None)

    AUTO = tf.data.AUTOTUNE
    slices = paths if labels is None else (paths, labels)

    dset = tf.data.Dataset.from_tensor_slices(slices).with_options(options)

    try:
        dset = dset.apply(
            tf.data.experimental.prefetch_to_device("/cpu:0", buffer_size=AUTO)
        )
    except Exception:
        pass

    dset = dset.map(decode_fn, num_parallel_calls=AUTO)

    if ignore_errors:
        dset = dset.apply(tf.data.experimental.ignore_errors())

    if cache:
        if cache_dir:
            _ensure_cache_dir(cache_dir)
            cache_path = os.path.join(cache_dir, "tfdata_cache")
            dset = dset.cache(cache_path)
        else:
            dset = dset.cache()

    if shuffle:
        dset = dset.shuffle(shuffle, seed=SEED, reshuffle_each_iteration=True)

    if repeat:
        dset = dset.repeat()

    if augment:
        dset = dset.map(augment_fn, num_parallel_calls=AUTO)

    dset = dset.batch(bsize, drop_remainder=drop_remainder)
    dset = dset.prefetch(AUTO)
    return dset




## === cell 3
COMPETITION_NAME = "ranzcr-clip-catheter-line-classification"
load_dir = f"/kaggle/input/{COMPETITION_NAME}/"

strategy = auto_select_accelerator()
BATCH_SIZE = strategy.num_replicas_in_sync * 16

train_df = pd.read_csv(os.path.join(load_dir, "train.csv"))
sample_sub = pd.read_csv(os.path.join(load_dir, "sample_submission.csv"))

ALL_LABEL_COLS = [
    c for c in train_df.columns if c not in ["StudyInstanceUID", "PatientID"]
]
ALL_LABEL_COLS = list(ALL_LABEL_COLS)

print("Number of train label columns:", len(ALL_LABEL_COLS))
print("Train labels:", ALL_LABEL_COLS)
print("Sample submission label columns:", list(sample_sub.columns))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2663366089.py in auto_select_accelerator()
      5     try:
----> 6         tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      7         tf.config.experimental_connect_to_cluster(tpu)

NameError: name 'tf' is not defined

During handling of the above exception, another exception occurred:

NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4257821802.py in <cell line: 0>()
      2 load_dir = f"/kaggle/input/{COMPETITION_NAME}/"
      3 
----> 4 strategy = auto_select_accelerator()
      5 BATCH_SIZE = strategy.num_replicas_in_sync * 16
      6 

/tmp/ipykernel_56/2663366089.py in auto_select_accelerator()
     10         print("Running on TPU:", tpu.master())
     11     except Exception:
---> 12         strategy = tf.distribute.get_strategy()
     13     print(f"Running on {strategy.num_replicas_in_sync} replicas")
     14     return strategy

NameError: name 'tf' is not defined

## === cell 4
train_paths_all = (
    load_dir + "train/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
).to_numpy(dtype=object)
y_all = train_df[ALL_LABEL_COLS].to_numpy(dtype=np.float32)

x_tr, x_va, y_tr, y_va = train_test_split(
    train_paths_all, y_all, test_size=0.1, random_state=SEED, shuffle=True
)

train_decoder = build_decoder(with_labels=True, target_size=(IMAGE_SIZE, IMAGE_SIZE))
valid_decoder = build_decoder(with_labels=True, target_size=(IMAGE_SIZE, IMAGE_SIZE))

SHUFFLE_BUF = 1024

dtrain = build_dataset(
    x_tr,
    y_tr,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=SHUFFLE_BUF,
    augment=True,
    cache=False,
    cache_dir="",
    decode_fn=train_decoder,
    drop_remainder=True,
    ignore_errors=False,
)

dvalid = build_dataset(
    x_va,
    y_va,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir="/kaggle/working/tfdata_cache_valid",
    decode_fn=valid_decoder,
    drop_remainder=False,
    ignore_errors=False,
)

steps_per_epoch = int(np.ceil(len(x_tr) / BATCH_SIZE))
valid_steps = int(np.ceil(len(x_va) / BATCH_SIZE))

print("Train size:", len(x_tr), "Valid size:", len(x_va))
print(
    "BATCH_SIZE:",
    BATCH_SIZE,
    "steps_per_epoch:",
    steps_per_epoch,
    "valid_steps:",
    valid_steps,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2840515829.py in <cell line: 0>()
      1 train_paths_all = (
----> 2     load_dir + "train/" + train_df["StudyInstanceUID"].astype(str) + ".jpg"
      3 ).to_numpy(dtype=object)
      4 y_all = train_df[ALL_LABEL_COLS].to_numpy(dtype=np.float32)
      5 

NameError: name 'train_df' is not defined

## === cell 5
with strategy.scope():
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),
        pooling="avg",
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(len(ALL_LABEL_COLS), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
        metrics=[],
        jit_compile=False,
        steps_per_execution=64,
    )

EPOCHS = 2
history = model.fit(
    dtrain,
    validation_data=dvalid,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=valid_steps,
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/1545572943.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     base = tf.keras.applications.EfficientNetB0(
      3         include_top=False,
      4         weights="imagenet",
      5         input_shape=(IMAGE_SIZE, IMAGE_SIZE, 3),

NameError: name 'strategy' is not defined

## === cell 6
test_uids = sample_sub["StudyInstanceUID"].astype(str)
test_paths = (load_dir + "test/" + test_uids + ".jpg").to_numpy(dtype=object)

test_decoder = build_decoder(with_labels=False, target_size=(IMAGE_SIZE, IMAGE_SIZE))

dtest = build_dataset(
    test_paths,
    bsize=BATCH_SIZE,
    repeat=False,
    shuffle=False,
    augment=False,
    cache=True,
    cache_dir="/kaggle/working/tfdata_cache_test",
    decode_fn=test_decoder,
    drop_remainder=False,
    ignore_errors=False,
)

pred = model.predict(dtest, verbose=1)
pred = np.clip(pred, 0.0, 1.0)

pred_full = pd.DataFrame(pred, columns=ALL_LABEL_COLS)
sub_df = pd.DataFrame({"StudyInstanceUID": test_uids})

required_label_cols = [c for c in sample_sub.columns if c != "StudyInstanceUID"]
missing = [c for c in required_label_cols if c not in pred_full.columns]
if missing:
    raise ValueError(
        f"sample_submission expects columns not present in training labels: {missing}. "
        f"Train labels: {ALL_LABEL_COLS}"
    )

sub_df[required_label_cols] = pred_full[required_label_cols].to_numpy(dtype=np.float32)
sub_df = sub_df[["StudyInstanceUID"] + required_label_cols]

sub_path = "submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", sub_df.shape)
print("Submission columns:", list(sub_df.columns))
print(sub_df.head())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_56/4211466506.py in <cell line: 0>()
----> 1 test_uids = sample_sub["StudyInstanceUID"].astype(str)
      2 test_paths = (load_dir + "test/" + test_uids + ".jpg").to_numpy(dtype=object)
      3 
      4 test_decoder = build_decoder(with_labels=False, target_size=(IMAGE_SIZE, IMAGE_SIZE))
      5 

NameError: name 'sample_sub' is not defined
