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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.14

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.8909467796476843

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Model

from sklearn.model_selection import train_test_split

tf.config.run_functions_eagerly(False)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/aptos2019-blindness-detection"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
TEST_CSV = os.path.join(DATA_DIR, "test.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train_df = pd.read_csv(TRAIN_CSV)
test_df = pd.read_csv(TEST_CSV)

train_df["path"] = TRAIN_IMG_DIR + "/" + train_df["id_code"].astype(str) + ".png"
test_df["path"] = TEST_IMG_DIR + "/" + test_df["id_code"].astype(str) + ".png"

assert {"id_code", "diagnosis"}.issubset(train_df.columns)
assert {"id_code"}.issubset(test_df.columns)
print("Train rows:", len(train_df), "Test rows:", len(test_df))
train_df.head()




## === cell 2
IMG_SIZE = 300
BATCH_SIZE = 16  # unchanged
AUTOTUNE = tf.data.AUTOTUNE

_TFDATA_OPTS_TRAIN = tf.data.Options()
_TFDATA_OPTS_TRAIN.deterministic = True
try:
    _TFDATA_OPTS_TRAIN.experimental_optimization.apply_default_optimizations = True
    _TFDATA_OPTS_TRAIN.experimental_optimization.map_and_batch_fusion = True
    _TFDATA_OPTS_TRAIN.experimental_optimization.parallel_batch = True
    _TFDATA_OPTS_TRAIN.experimental_optimization.autotune_buffers = True
except Exception:
    pass

_TFDATA_OPTS_EVAL = tf.data.Options()
_TFDATA_OPTS_EVAL.deterministic = True
try:
    _TFDATA_OPTS_EVAL.experimental_optimization.apply_default_optimizations = True
    _TFDATA_OPTS_EVAL.experimental_optimization.map_and_batch_fusion = True
    _TFDATA_OPTS_EVAL.experimental_optimization.parallel_batch = True
    _TFDATA_OPTS_EVAL.experimental_optimization.autotune_buffers = True
except Exception:
    pass


@tf.function
def decode_and_resize(path):
    image_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    img = tf.image.resize(
        img,
        [IMG_SIZE, IMG_SIZE],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function
def _augment(img, label):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, label


def _build_image_table(paths_np: np.ndarray):
    keys = tf.constant(paths_np, dtype=tf.string)

    ds = tf.data.Dataset.from_tensor_slices(keys).map(
        decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    values = list(ds.as_numpy_iterator())  # list of (H,W,3) float32 arrays
    values = tf.constant(np.stack(values, axis=0), dtype=tf.float32)

    table = tf.lookup.StaticHashTable(
        initializer=tf.lookup.KeyValueTensorInitializer(keys=keys, values=values),
        default_value=tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32),
    )
    return table


def make_train_ds_with_table(df, table, training=True):
    paths = df["path"].to_numpy()
    labels = df["diagnosis"].to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_TFDATA_OPTS_TRAIN if training else _TFDATA_OPTS_EVAL)

    @tf.function
    def _lookup(path, label):
        return table.lookup(path), label

    ds = ds.map(_lookup, num_parallel_calls=AUTOTUNE, deterministic=True)

    if training:
        ds = ds.shuffle(min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_with_table(df, table):
    paths = df["path"].to_numpy()
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(_TFDATA_OPTS_EVAL)

    @tf.function
    def _lookup(path):
        return table.lookup(path)

    ds = ds.map(_lookup, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_split, val_split = train_test_split(
    train_df, test_size=0.15, random_state=SEED, stratify=train_df["diagnosis"]
)

train_table = _build_image_table(train_split["path"].to_numpy())
val_table = _build_image_table(val_split["path"].to_numpy())
test_table = _build_image_table(test_df["path"].to_numpy())

train_ds = make_train_ds_with_table(train_split, train_table, training=True)
val_ds = make_train_ds_with_table(val_split, val_table, training=False)
test_ds = make_test_ds_with_table(test_df, test_table)

train_steps = int(np.ceil(len(train_split) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_split) / BATCH_SIZE))
test_steps = int(np.ceil(len(test_df) / BATCH_SIZE))

print("Datasets ready.")
print("train_steps:", train_steps, "val_steps:", val_steps, "test_steps:", test_steps)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_shape.py in merge_with(self, other)
   1040       try:
-> 1041         self.assert_same_rank(other)
   1042         new_dims = [

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_shape.py in assert_same_rank(self, other)
   1093       if self.rank != other.rank:
-> 1094         raise ValueError("Shapes %s and %s must have the same rank" %
   1095                          (self, other))

ValueError: Shapes (300, 300, 3) and () must have the same rank

During handling of the above exception, another exception occurred:

ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3234795620.py in <cell line: 0>()
    113 
    114 # Build split-specific in-memory tables (keeps peak RAM bounded vs caching full train+val together twice).
--> 115 train_table = _build_image_table(train_split["path"].to_numpy())
    116 val_table = _build_image_table(val_split["path"].to_numpy())
    117 test_table = _build_image_table(test_df["path"].to_numpy())

/tmp/ipykernel_11/3234795620.py in _build_image_table(paths_np)
     63     values = tf.constant(np.stack(values, axis=0), dtype=tf.float32)
     64 
---> 65     table = tf.lookup.StaticHashTable(
     66         initializer=tf.lookup.KeyValueTensorInitializer(keys=keys, values=values),
     67         default_value=tf.zeros([IMG_SIZE, IMG_SIZE, 3], dtype=tf.float32),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in __call__(cls, *args, **kwargs)
    101       previous_getter = _make_getter(getter, previous_getter)
    102 
--> 103     return previous_getter(*args, **kwargs)
    104 
    105 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in <lambda>(*a, **kw)
     96       return obj
     97 
---> 98     previous_getter = lambda *a, **kw: default_resource_creator(None, *a, **kw)
     99     resource_creator_stack = ops.get_default_graph()._resource_creator_stack
    100     for getter in resource_creator_stack[cls._resource_type()]:

/usr/local/lib/python3.11/dist-packages/tensorflow/python/trackable/resource.py in default_resource_creator(next_creator, *a, **kw)
     93       assert next_creator is None
     94       obj = cls.__new__(cls, *a, **kw)
---> 95       obj.__init__(*a, **kw)
     96       return obj
     97 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, initializer, default_value, name, experimental_is_anonymous)
    351     self._name = name or "hash_table"
    352     self._table_name = None
--> 353     super(StaticHashTable, self).__init__(default_value, initializer)
    354     self._value_shape = self._default_value.get_shape()
    355 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/lookup_ops.py in __init__(self, default_value, initializer)
    195     self._default_value = ops.convert_to_tensor(
    196         default_value, dtype=self._value_dtype)
--> 197     self._default_value.get_shape().merge_with(tensor_shape.TensorShape([]))
    198     if isinstance(initializer, trackable_base.Trackable):
    199       self._initializer = self._track_trackable(initializer, "_initializer")

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/tensor_shape.py in merge_with(self, other)
   1046         return TensorShape(new_dims)
   1047       except ValueError:
-> 1048         raise ValueError("Shapes %s and %s are not compatible" % (self, other))
   1049 
   1050   def __add__(self, other):

ValueError: Shapes (300, 300, 3) and () are not compatible

## === cell 3
base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
x = base.output
x = GlobalAveragePooling2D()(x)
x = Dropout(0.3)(x)
outputs = Dense(5, activation="softmax")(x)
model = Model(inputs=base.input, outputs=outputs)

base.trainable = False

model.compile(
    optimizer=Adam(1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

model.summary()




## === cell 4
EPOCHS_HEAD = 3
history1 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_HEAD,
    steps_per_epoch=train_steps,  # unchanged semantics: full pass
    validation_steps=val_steps,  # unchanged semantics: full pass
    verbose=1,
)

base.trainable = True
model.compile(
    optimizer=Adam(1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)

EPOCHS_FT = 2
history2 = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS_FT,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4015163078.py in <cell line: 0>()
      1 EPOCHS_HEAD = 3
      2 history1 = model.fit(
----> 3     train_ds,
      4     validation_data=val_ds,
      5     epochs=EPOCHS_HEAD,

NameError: name 'train_ds' is not defined

## === cell 5
print("Predicting...")
probs = model.predict(test_ds, steps=test_steps, verbose=1)
preds = np.argmax(probs, axis=1).astype(int)

submission = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": preds})
submission["diagnosis"] = submission["diagnosis"].astype(int)
submission.to_csv("submission.csv", index=False)

print("Saved submission.csv")
print(submission.head())
print("Submission shape:", submission.shape)
assert list(submission.columns) == ["id_code", "diagnosis"]
assert len(submission) == len(test_df)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3509557709.py in <cell line: 0>()
      1 print("Predicting...")
----> 2 probs = model.predict(test_ds, steps=test_steps, verbose=1)
      3 preds = np.argmax(probs, axis=1).astype(int)
      4 
      5 submission = pd.DataFrame({"id_code": test_df["id_code"].values, "diagnosis": preds})

NameError: name 'test_ds' is not defined
