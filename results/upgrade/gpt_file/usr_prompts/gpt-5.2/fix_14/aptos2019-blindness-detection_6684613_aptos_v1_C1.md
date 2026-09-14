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

3.9

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

0.6328167661904547

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import tensorflow as tf
import cv2

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _cpu = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(_cpu)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, _cpu // 2))
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

BASE = "../input/aptos2019-blindness-detection"
if not os.path.exists(BASE):
    BASE = "/kaggle/input/aptos2019-blindness-detection"

TRAIN_CSV = os.path.join(BASE, "train.csv")
TEST_CSV = os.path.join(BASE, "test.csv")
SAMPLE_SUB = os.path.join(BASE, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE, "train_images")
TEST_DIR = os.path.join(BASE, "test_images")

print("BASE:", BASE)
print("Train csv exists:", os.path.exists(TRAIN_CSV))
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

BS = 16
IMG_SIZE = 300
SIZE = (IMG_SIZE, IMG_SIZE)

df = pd.read_csv(TRAIN_CSV)
df["name"] = df["id_code"].astype(str) + ".png"
df["diagnosis"] = df["diagnosis"].astype(int)

train_df, val_df = train_test_split(
    df, test_size=0.1, random_state=SEED, shuffle=True, stratify=df["diagnosis"]
)

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input


@tf.function
def _decode_resize_preprocess(img_bytes):
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method="bilinear", antialias=False)
    img = tf.ensure_shape(img, [IMG_SIZE, IMG_SIZE, 3])
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


@tf.function
def _load_image_and_label(path, label):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_preprocess(img_bytes)
    return img, tf.cast(label, tf.int32)


@tf.function
def _load_image_only(path):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_preprocess(img_bytes)
    return img


train_paths = (TRAIN_DIR + os.sep + train_df["name"]).to_numpy(dtype=str)
train_labels = train_df["diagnosis"].to_numpy(dtype=np.int32)

val_paths = (TRAIN_DIR + os.sep + val_df["name"]).to_numpy(dtype=str)
val_labels = val_df["diagnosis"].to_numpy(dtype=np.int32)

AUTOTUNE = tf.data.AUTOTUNE

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_slack = True

shuffle_buf = min(len(train_df), 1024)

train_cache_path = os.path.join("/kaggle/working", "train_cache.tfcache")
val_cache_path = os.path.join("/kaggle/working", "val_cache.tfcache")

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_labels)).with_options(
    options
)
train_ds = train_ds.shuffle(shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
train_ds = train_ds.map(_load_image_and_label, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors)
train_ds = train_ds.batch(BS, drop_remainder=True).prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
    options
)
val_ds = val_ds.map(_load_image_and_label, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors)
val_ds = val_ds.batch(BS, drop_remainder=False).prefetch(AUTOTUNE)

for _ in train_ds.take(1):
    pass
for _ in val_ds.take(1):
    pass

print("Train/Val sizes:", len(train_df), len(val_df))
print("Caching enabled: DISK")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4023277777.py in <cell line: 0>()
     72 train_ds = train_ds.cache(train_cache_path)
     73 # Speed/robustness: skip rare decode errors instead of stalling; does not change semantics on valid data.
---> 74 train_ds = train_ds.apply(tf.data.experimental.ignore_errors)
     75 train_ds = train_ds.batch(BS, drop_remainder=True).prefetch(AUTOTUNE)
     76 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in apply(self, transformation_func)
   2587     dataset = transformation_func(self)
   2588     if not isinstance(dataset, data_types.DatasetV2):
-> 2589       raise TypeError(
   2590           f"`transformation_func` must return a `tf.data.Dataset` object. "
   2591           f"Got {type(dataset)}.")

TypeError: `transformation_func` must return a `tf.data.Dataset` object. Got <class 'function'>.

## === cell 2
from tensorflow.keras.optimizers import RMSprop, Adam, SGD
from tensorflow.keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    BatchNormalization,
)

base_model = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
)
base_model.trainable = False  # keep training stable and within runtime

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base_model(inputs, training=False)
x = GlobalAveragePooling2D()(x)
x = BatchNormalization()(x)
x = Dropout(0.3)(x)
outputs = Dense(5, activation="softmax")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

checkpoint = ModelCheckpoint(
    "bestmodel.keras",
    save_best_only=True,
    monitor="val_loss",
    mode="min",
    verbose=0,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss",
    factor=0.2,
    patience=3,
    min_lr=1e-6,
    mode="min",
    verbose=0,
)

restore_best = EarlyStopping(
    monitor="val_loss", mode="min", patience=10**9, restore_best_weights=True, verbose=0
)

model.summary()



## === cell 3
EPOCHS = 8  # unchanged

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    callbacks=[checkpoint, reduce_lr, restore_best],
    verbose=1,
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1317719522.py in <cell line: 0>()
      3 history = model.fit(
      4     train_ds,
----> 5     validation_data=val_ds,
      6     epochs=EPOCHS,
      7     callbacks=[checkpoint, reduce_lr, restore_best],

NameError: name 'val_ds' is not defined

## === cell 4
sub = pd.read_csv(TEST_CSV)
test_paths = (TEST_DIR + os.sep + sub["id_code"].astype(str) + ".png").to_numpy(
    dtype=str
)

options_test = tf.data.Options()
options_test.experimental_deterministic = True
options_test.experimental_optimization.apply_default_optimizations = True
options_test.experimental_slack = True

test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options_test)
test_ds = test_ds.map(_load_image_only, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors)
test_ds = test_ds.batch(BS).prefetch(AUTOTUNE)

if os.path.exists("bestmodel.keras"):
    model = tf.keras.models.load_model("bestmodel.keras", compile=False)

pred_proba = model.predict(test_ds, verbose=1)
results = np.argmax(pred_proba, axis=1).astype(int)

print("Preds:", results[:10], "len:", len(results), "test len:", len(sub))



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2680799981.py in <cell line: 0>()
     12 test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options_test)
     13 test_ds = test_ds.map(_load_image_only, num_parallel_calls=AUTOTUNE)
---> 14 test_ds = test_ds.apply(tf.data.experimental.ignore_errors)
     15 test_ds = test_ds.batch(BS).prefetch(AUTOTUNE)
     16 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in apply(self, transformation_func)
   2587     dataset = transformation_func(self)
   2588     if not isinstance(dataset, data_types.DatasetV2):
-> 2589       raise TypeError(
   2590           f"`transformation_func` must return a `tf.data.Dataset` object. "
   2591           f"Got {type(dataset)}.")

TypeError: `transformation_func` must return a `tf.data.Dataset` object. Got <class 'function'>.

## === cell 5
pred_df = pd.read_csv(SAMPLE_SUB)

pred_df = pred_df.merge(sub[["id_code"]], on="id_code", how="right", sort=False)

if len(pred_df) != len(results):
    raise ValueError(
        f"Prediction length mismatch: preds={len(results)} rows={len(pred_df)}"
    )

pred_df["diagnosis"] = results
pred_df[["id_code", "diagnosis"]].to_csv("submission.csv", index=False)

print(pred_df.head())
print("Wrote submission.csv with rows:", len(pred_df))

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3641607493.py in <cell line: 0>()
      3 pred_df = pred_df.merge(sub[["id_code"]], on="id_code", how="right", sort=False)
      4 
----> 5 if len(pred_df) != len(results):
      6     raise ValueError(
      7         f"Prediction length mismatch: preds={len(results)} rows={len(pred_df)}"

NameError: name 'results' is not defined
