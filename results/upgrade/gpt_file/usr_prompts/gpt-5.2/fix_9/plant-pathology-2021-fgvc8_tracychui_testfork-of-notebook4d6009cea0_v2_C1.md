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
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Target score

0.3428175702413932

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import gc
import numpy as np
import pandas as pd

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "cpp"

os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K
import tensorflow.keras.applications.resnet50 as resnet

from sklearn import preprocessing
from sklearn.linear_model import LogisticRegression

K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 123
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/2565007910.py in <cell line: 0>()
     12 os.environ.setdefault("TF_XLA_FLAGS", "--tf_xla_auto_jit=2")
     13 
---> 14 import tensorflow as tf
     15 from tensorflow import keras
     16 from tensorflow.keras import backend as K

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
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"




## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_DIR path."




## === cell 3
training_csv = pd.read_csv(TRAIN_CSV)
training_class = np.array([])
for labels in pd.unique(training_csv["labels"]):
    training_class = np.append(training_class, labels.split())
tagnames = np.unique(training_class)
print("n_train:", len(training_csv), "n_tags:", len(tagnames))
print("tags:", tagnames)




## === cell 4
def labels_to_multihot(labels_series, tagnames):
    from sklearn.preprocessing import MultiLabelBinarizer

    mlb = MultiLabelBinarizer(classes=list(tagnames))
    split = labels_series.fillna("").astype(str).str.split().tolist()
    Y = mlb.fit_transform(split).astype(np.int8, copy=False)
    return Y


Y = labels_to_multihot(training_csv["labels"], tagnames)




## === cell 5
train_paths = [os.path.join(TRAIN_DIR, fn) for fn in training_csv["image"].tolist()]

print("n_train_after_filter:", len(train_paths), "Y shape:", Y.shape)




## === cell 6
base = resnet.ResNet50(
    weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
x = base(inp, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
feat_model = keras.Model(inp, x)
feat_dim = feat_model.output_shape[-1]
print("feature dim:", feat_dim)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1755181096.py in <cell line: 0>()
----> 1 base = resnet.ResNet50(
      2     weights="imagenet", include_top=False, input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
      3 )
      4 inp = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
      5 x = base(inp, training=False)

NameError: name 'resnet' is not defined

## === cell 7
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


def extract_features(paths, batch_size=32):
    paths_tf = tf.constant(paths)
    ds = tf.data.Dataset.from_tensor_slices(paths_tf)

    opts = tf.data.Options()
    try:
        opts.experimental_deterministic = True
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
    except Exception:
        pass
    ds = ds.with_options(opts)

    ds = ds.map(
        _decode_resize_preprocess,
        num_parallel_calls=tf.data.AUTOTUNE,
        deterministic=True,
    )

    ds = ds.cache()

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    feats = feat_model.predict(ds, verbose=0)
    feats = np.asarray(feats, dtype=np.float32)
    return feats




## === cell 8
Xf_train = extract_features(train_paths, batch_size=192)
print("Xf_train:", Xf_train.shape)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/926274294.py in <cell line: 0>()
----> 1 Xf_train = extract_features(train_paths, batch_size=192)
      2 print("Xf_train:", Xf_train.shape)
      3 
      4 

/tmp/ipykernel_11/3349306256.py in extract_features(paths, batch_size)
     16 
     17 def extract_features(paths, batch_size=32):
---> 18     paths_tf = tf.constant(paths)
     19     ds = tf.data.Dataset.from_tensor_slices(paths_tf)
     20 

NameError: name 'tf' is not defined

## === cell 9
Xf_test = extract_features(imglist_test, batch_size=256)
print("Xf_test:", Xf_test.shape)

gc.collect()




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4088614335.py in <cell line: 0>()
----> 1 Xf_test = extract_features(imglist_test, batch_size=256)
      2 print("Xf_test:", Xf_test.shape)
      3 
      4 gc.collect()
      5 

/tmp/ipykernel_11/3349306256.py in extract_features(paths, batch_size)
     16 
     17 def extract_features(paths, batch_size=32):
---> 18     paths_tf = tf.constant(paths)
     19     ds = tf.data.Dataset.from_tensor_slices(paths_tf)
     20 

NameError: name 'tf' is not defined

## === cell 10
scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
trainXn = scaler.fit_transform(Xf_train)
testXn_test = scaler.transform(Xf_test)
print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/490810005.py in <cell line: 0>()
----> 1 scaler = preprocessing.MinMaxScaler(feature_range=(-1, 1))
      2 trainXn = scaler.fit_transform(Xf_train)
      3 testXn_test = scaler.transform(Xf_test)
      4 print("trainXn:", trainXn.shape, "testXn_test:", testXn_test.shape)
      5 

NameError: name 'preprocessing' is not defined

## === cell 11
is_const = np.array(
    [(Y[:, i].sum() == 0) or (Y[:, i].sum() == len(Y)) for i in range(len(tagnames))],
    dtype=bool,
)
nonconst_idx = np.where(~is_const)[0]

tagmodels1 = {t: None for t in tagnames}

multi_lr = None
if len(nonconst_idx) > 0:
    multi_lr = LogisticRegression(
        solver="lbfgs",
        max_iter=1000,
        class_weight="balanced",
        random_state=SEED,
        n_jobs=-1,
    )
    multi_lr.fit(trainXn, Y[:, nonconst_idx])

print(
    "trained tag models:",
    int(len(nonconst_idx)),
    "/",
    len(tagnames),
)




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3345548739.py in <cell line: 0>()
      9 multi_lr = None
     10 if len(nonconst_idx) > 0:
---> 11     multi_lr = LogisticRegression(
     12         solver="lbfgs",
     13         max_iter=1000,

NameError: name 'LogisticRegression' is not defined

## === cell 12
testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)

for i in np.where(is_const)[0]:
    y = Y[:, i]
    const = 1.0 if y.sum() == len(y) else 0.0
    testKaggle_ppredscore1[:, i] = const

if multi_lr is not None:
    prob = multi_lr.predict_proba(testXn_test)
    if isinstance(prob, list):
        pos_probs = np.column_stack([p[:, 1] for p in prob]).astype(
            np.float32, copy=False
        )
    else:
        pos_probs = prob[:, :, 1].astype(np.float32, copy=False)
    testKaggle_ppredscore1[:, nonconst_idx] = pos_probs

print("pred score matrix:", testKaggle_ppredscore1.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3122423244.py in <cell line: 0>()
----> 1 testKaggle_ppredscore1 = np.zeros((len(testXn_test), len(tagnames)), dtype=np.float32)
      2 
      3 for i in np.where(is_const)[0]:
      4     y = Y[:, i]
      5     const = 1.0 if y.sum() == len(y) else 0.0

NameError: name 'testXn_test' is not defined

## === cell 13
def class2tags(classes, tagnames):
    tagnames = np.asarray(tagnames)
    return [" ".join(tagnames[row]) for row in classes]




## === cell 14
test_predclass = testKaggle_ppredscore1 > 0.5
test_predtags = class2tags(test_predclass, tagnames)

for i in range(len(test_predtags)):
    if test_predtags[i] == "":
        test_predtags[i] = "healthy"

print("example preds:", test_predtags[:5])




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3635302910.py in <cell line: 0>()
----> 1 test_predclass = testKaggle_ppredscore1 > 0.5
      2 test_predtags = class2tags(test_predclass, tagnames)
      3 
      4 for i in range(len(test_predtags)):
      5     if test_predtags[i] == "":

NameError: name 'testKaggle_ppredscore1' is not defined

## === cell 15
df1 = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})
df2 = pd.DataFrame({"labels": test_predtags})
submission = pd.concat([df1, df2], axis=1)

sample = pd.read_csv(SAMPLE_SUB)
assert list(sample.columns) == ["image", "labels"]
assert list(submission.columns) == ["image", "labels"]
assert len(submission) == len(
    sample
), f"Submission rows {len(submission)} != sample rows {len(sample)}"

submission.head()




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/386754042.py in <cell line: 0>()
      1 df1 = pd.DataFrame({"image": [os.path.basename(p) for p in imglist_test]})
----> 2 df2 = pd.DataFrame({"labels": test_predtags})
      3 submission = pd.concat([df1, df2], axis=1)
      4 
      5 sample = pd.read_csv(SAMPLE_SUB)

NameError: name 'test_predtags' is not defined

## === cell 16
submission.to_csv("./submission.csv", index=False)
print("Wrote ./submission.csv with shape:", submission.shape)
print(submission.head())

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3759179968.py in <cell line: 0>()
----> 1 submission.to_csv("./submission.csv", index=False)
      2 print("Wrote ./submission.csv with shape:", submission.shape)
      3 print(submission.head())

NameError: name 'submission' is not defined
