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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8318223028105167

# 6. Current score

0.1278

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11809) has done: 'The timeout is dominated by slow Python-side image loading/augmentation in `ImageDataGenerator` plus extra EDA/plotting cells that do disk I/O and rendering but don’t affect training or predictions. I keep the exact same model, loss, optimizer, epochs, and training loop semantics, but replace the generators with a deterministic `tf.data` pipeline that performs identical preprocessing and geometric augmentations on the GPU/TF runtime with caching/prefetching. I also remove (skip) all exploratory plotting/image display work so the notebook only does what’s required to train, validate, and write `submission.csv`. Finally, I keep determinism/seed settings and avoid any approximations (no fewer epochs, no early stopping changes, no mixed precision).'
- What this solution (achieved 0.11809) has done: 'The timeout is dominated by expensive per-image augmentation using `ImageProjectiveTransformV3` plus unnecessary `cache()` of decoded images in memory, and by retracing overhead from non-jitted map functions. I keep the exact same model, loss, epochs, and augmentation math/semantics, but make the input pipeline faster by (1) using `tf.io.decode_jpeg(..., dct_method="INTEGER_FAST")` (equivalent decode), (2) removing the large in-memory cache on the training pipeline (to avoid memory pressure/slowdowns), (3) compiling the heavy augmentation+preprocess map with `@tf.function(jit_compile=True)` and ensuring static shapes to reduce Python overhead, and (4) keeping determinism/seeding intact. Validation/test caching remains (small enough and used repeatedly) to preserve speed without changing results.'
- What this solution (achieved 0.11584) has done: 'The timeout is dominated by expensive per-image CPU decoding/resizing plus the custom projective-transform augmentation running every epoch for ~15k images, and the input pipeline currently redoes decode+resize work each epoch without caching. I keep the exact model, loss, optimizer, epochs, and augmentation math, but restructure the tf.data pipeline to cache the deterministic decode+resize stage once, then apply shuffle/augment on top so augmentation still changes each epoch while eliminating repeated JPEG decode/resize cost. I also remove unnecessary string-hash work from the hot path by precomputing per-sample stateless RNG seeds (deterministic and unique per image) once and passing them through the dataset, and enable TF data map parallelism + prefetch deterministically. These changes preserve evaluation semantics and determinism while substantially reducing wall time.'
- What this solution (achieved 0.1278) has done: 'The timeout is dominated by (1) repeatedly scanning and decoding every TFRecord just to decide whether each file belongs to train vs valid, and (2) Python-side per-batch prediction loops for the test set. I keep the exact model/training logic, but replace the expensive TFRecord membership scan with a deterministic file-level split that matches the dataset’s fixed ordering (each tfrec contains a contiguous block of images) while preserving the same 80/20 stratified split at the sample level. I also remove the Python prediction loop by using `model.predict(...)` and return names via a parallel `name_ds` pipeline, which is equivalent but faster. Finally, I add dataset caching (in-memory) for validation/test to avoid re-decoding across epochs/evaluation without changing semantics.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
if os.environ.get("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "").lower() == "python":
    del os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"]

import json
import math
import numpy as np
import pandas as pd
import tensorflow as tf

try:
    import matplotlib.pyplot as plt
except Exception:
    plt = None

try:
    import seaborn as sns

    sns.set()
except Exception:
    sns = None

from tensorflow import keras
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras import layers, models
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.set_soft_device_placement(True)
except Exception:
    pass

try:
    import multiprocessing

    _CPU = multiprocessing.cpu_count()
    tf.config.threading.set_intra_op_parallelism_threads(_CPU)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/cassava-leaf-disease-classification"



## === cell 2
train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")
train_tfrecords_dir = os.path.join(path, "train_tfrecords")
test_tfrecords_dir = os.path.join(path, "test_tfrecords")



## === cell 3
with open(
    "../input/cassava-leaf-disease-classification/label_num_to_disease_map.json"
) as file:
    classes = json.loads(file.read())

print(json.dumps(classes, indent=4))



## === cell 4
df_train = pd.read_csv(os.path.join(path, "train.csv"))
df_train.head()



## === cell 5
df_train["class"] = df_train["label"].map({int(i): c for i, c in classes.items()})
df_train.head()



## === cell 6
_ = df_train["class"].value_counts()




## === cell 7
def plot(images, labels, predictions=None):
    pass




## === cell 8
pass



## === cell 9
pass



## === cell 10
pass



## === cell 11
pass



## === cell 12
pass



## === cell 13
pass



## === cell 14
pass



## === cell 15
pass



## === cell 16
pass



## === cell 17
train = df_train.astype({"label": str})
from sklearn.model_selection import train_test_split

train, valid = train_test_split(
    train, test_size=0.2, random_state=SEED, stratify=train["label"]
)



## === cell 18
img_size = 300
size = (img_size, img_size)
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

class_names = [classes[str(i)] for i in range(5)]
class_to_index = {name: i for i, name in enumerate(class_names)}

IMG_SIZE_F = tf.constant(float(img_size), tf.float32)
PI_OVER_180 = tf.constant(math.pi / 180.0, tf.float32)
CX = tf.constant((img_size - 1) / 2.0, tf.float32)
CY = tf.constant((img_size - 1) / 2.0, tf.float32)

_DATA_OPTIONS = tf.data.Options()
_DATA_OPTIONS.experimental_deterministic = True  # preserve deterministic semantics
try:
    _DATA_OPTIONS.autotune.enabled = True
except Exception:
    pass

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
    "target": tf.io.FixedLenFeature([], tf.int64),
}

_NAME_ONLY_FEATURE = {
    "image_name": tf.io.FixedLenFeature([], tf.string),
}


def _apply_ds_options(ds):
    return ds.with_options(_DATA_OPTIONS)


def _parse_tfrecord(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.io.decode_jpeg(
        ex["image"], channels=3, dct_method="INTEGER_FAST", try_recover_truncated=True
    )
    img = tf.image.resize(
        img, [img_size, img_size], method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
    )
    img = tf.cast(img, tf.float32)
    img.set_shape([img_size, img_size, 3])
    label = tf.cast(ex["target"], tf.int32)
    image_name = ex["image_name"]
    return img, label, image_name


def _parse_name_only(example_proto):
    ex = tf.io.parse_single_example(example_proto, _NAME_ONLY_FEATURE)
    return ex["image_name"]


def _preprocess(img):
    return preprocess_input(img)


def _one_hot(label_idx):
    return tf.one_hot(label_idx, depth=5, dtype=tf.float32)


def _augment(img, seed):
    seeds = tf.random.experimental.stateless_split(seed, 7)
    s1, s2, s3, s4, s5, s6, s7 = (
        seeds[0],
        seeds[1],
        seeds[2],
        seeds[3],
        seeds[4],
        seeds[5],
        seeds[6],
    )

    img = tf.image.stateless_random_flip_left_right(img, seed=s1)
    img = tf.image.stateless_random_flip_up_down(img, seed=s2)

    angle = (
        tf.random.stateless_uniform([], seed=s3, minval=-45.0, maxval=45.0)
        * PI_OVER_180
    )
    zoom = tf.random.stateless_uniform([], seed=s4, minval=0.8, maxval=1.2)

    tx = tf.random.stateless_uniform([], seed=s5, minval=-0.2, maxval=0.2) * IMG_SIZE_F
    ty = tf.random.stateless_uniform([], seed=s6, minval=-0.2, maxval=0.2) * IMG_SIZE_F
    shear = tf.random.stateless_uniform([], seed=s7, minval=-0.2, maxval=0.2)

    cos_a = tf.cos(angle)
    sin_a = tf.sin(angle)

    r00 = cos_a
    r01 = -sin_a
    r10 = sin_a
    r11 = cos_a

    sh = tf.tan(shear)
    s00 = 1.0
    s01 = sh
    s10 = 0.0
    s11 = 1.0

    z00 = 1.0 / zoom
    z11 = 1.0 / zoom

    a00 = z00 * (s00 * r00 + s01 * r10)
    a01 = z00 * (s00 * r01 + s01 * r11)
    b00 = z11 * (s10 * r00 + s11 * r10)
    b01 = z11 * (s10 * r01 + s11 * r11)

    a02 = CX - a00 * CX - a01 * CY - tx
    b02 = CY - b00 * CX - b01 * CY - ty

    transform = tf.stack([a00, a01, a02, b00, b01, b02, 0.0, 0.0])[tf.newaxis, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=[img_size, img_size],
        interpolation="NEAREST",
        fill_mode="REFLECT",
        fill_value=0.0,
    )[0]
    img.set_shape([img_size, img_size, 3])
    return img


@tf.function(jit_compile=False)
def _train_map_compiled(img, y, idx):
    seed = tf.stack([tf.cast(SEED, tf.int32), tf.cast(idx, tf.int32)])
    img = _augment(img, seed)
    img = _preprocess(img)
    y = _one_hot(y)
    return img, y


@tf.function(jit_compile=False)
def _valid_map_compiled(img, y):
    img = _preprocess(img)
    y = _one_hot(y)
    return img, y


def _list_tfrec_files(directory, prefix):
    pattern = os.path.join(directory, f"{prefix}*.tfrec")
    files = tf.io.gfile.glob(pattern)
    files = sorted(files)
    return files


_TRAIN_TFREC_FILES = _list_tfrec_files(train_tfrecords_dir, "ld_train")

valid_name_set = set(valid["image_id"].astype(str).tolist())


def _split_tfrec_files_by_index_mask(tfrec_files, valid_mask, records_per_file=1338):
    n = len(valid_mask)
    n_files = len(tfrec_files)
    file_ids = np.arange(n, dtype=np.int32) // records_per_file
    file_ids = np.clip(file_ids, 0, n_files - 1)
    valid_counts = np.bincount(file_ids[valid_mask], minlength=n_files)
    total_counts = np.bincount(file_ids, minlength=n_files)
    is_valid_file = valid_counts > (total_counts // 2)
    train_files = [fp for fp, v in zip(tfrec_files, is_valid_file) if not v]
    valid_files = [fp for fp, v in zip(tfrec_files, is_valid_file) if v]
    return train_files, valid_files


valid_mask = df_train.index.isin(valid.index).to_numpy(dtype=bool)

_TRAIN_FILES, _VALID_FILES = _split_tfrec_files_by_index_mask(
    _TRAIN_TFREC_FILES, valid_mask, records_per_file=1338
)


def make_train_ds(_df_unused):
    ds = tf.data.TFRecordDataset(_TRAIN_FILES, num_parallel_reads=AUTOTUNE)
    ds = _apply_ds_options(ds)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)
    ds = ds.map(lambda img, y, name: (img, y), num_parallel_calls=AUTOTUNE)
    shuffle_buf = int(min(len(train), 4096))
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.enumerate()
    ds = ds.map(
        lambda idx, data: _train_map_compiled(data[0], data[1], idx),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return _apply_ds_options(ds)


def make_valid_ds(_df_unused):
    ds = tf.data.TFRecordDataset(_VALID_FILES, num_parallel_reads=AUTOTUNE)
    ds = _apply_ds_options(ds)
    ds = ds.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)
    ds = ds.map(
        lambda img, y, name: _valid_map_compiled(img, y), num_parallel_calls=AUTOTUNE
    )
    ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return _apply_ds_options(ds)


train_ds = make_train_ds(train)
valid_ds = make_valid_ds(valid)



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3941999507.py in <cell line: 0>()
    177 # Build a deterministic valid mask aligned to the original train.csv row order.
    178 # train_test_split returns subsets (shuffled), but we can recover membership by index.
--> 179 valid_mask = df_train.index.isin(valid.index).to_numpy(dtype=bool)
    180 
    181 _TRAIN_FILES, _VALID_FILES = _split_tfrec_files_by_index_mask(

AttributeError: 'numpy.ndarray' object has no attribute 'to_numpy'

## === cell 19
step_size_train = int(math.ceil(len(train) / BATCH_SIZE))
step_size_valid = int(math.ceil(len(valid) / BATCH_SIZE))
step_size_train, step_size_valid




## === cell 20
def modelTransf():
    model = models.Sequential()
    model.add(
        EfficientNetB3(
            input_shape=(img_size, img_size, 3), include_top=False, weights="imagenet"
        )
    )
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dense(256, activation="relu"))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(5, activation="softmax"))
    return model




## === cell 21
model = modelTransf()



## === cell 22
model.summary()



## === cell 23
model.compile(
    loss="categorical_crossentropy",
    optimizer=Adam(learning_rate=0.001),
    metrics=["accuracy"],
)



## === cell 24
early_stopping = EarlyStopping(
    monitor="val_loss", patience=10, mode="min", restore_best_weights=True
)

checkpoint = ModelCheckpoint(
    "modelB3.keras", monitor="val_loss", verbose=1, mode="min", save_best_only=True
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.2, patience=10, min_lr=0.001, mode="min", verbose=1
)



## === cell 25
history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=30,
    steps_per_epoch=step_size_train,
    validation_steps=step_size_valid,
    callbacks=[early_stopping, checkpoint, reduce_lr],
)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2105928782.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_ds,
      3     validation_data=valid_ds,
      4     epochs=30,
      5     steps_per_epoch=step_size_train,

NameError: name 'train_ds' is not defined

## === cell 26
best_model_path = "modelB3.keras"
if os.path.exists(best_model_path):
    model_trained = keras.models.load_model(best_model_path)
else:
    model_trained = model



## === cell 27
from sklearn.metrics import accuracy_score

val_probs = model_trained.predict(valid_ds, verbose=0)
val_pred = np.argmax(val_probs, axis=1)[: len(valid)]
val_true = valid["class"].map(class_to_index).values
print("Validation accuracy:", accuracy_score(val_true, val_pred))



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1068294254.py in <cell line: 0>()
      2 
      3 # --- SPEED FIX (equivalent): use model.predict directly (efficient internal loop) on cached valid_ds.
----> 4 val_probs = model_trained.predict(valid_ds, verbose=0)
      5 val_pred = np.argmax(val_probs, axis=1)[: len(valid)]
      6 val_true = valid["class"].map(class_to_index).values

NameError: name 'valid_ds' is not defined

## === cell 28
submission_file = pd.read_csv(
    "../input/cassava-leaf-disease-classification/sample_submission.csv"
)
submission_file.head()



## === cell 29
test_df = submission_file.copy()

test_tfrec_files = sorted(
    tf.io.gfile.glob(os.path.join(test_tfrecords_dir, "ld_test*.tfrec"))
)
test_raw = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
test_raw = _apply_ds_options(test_raw)
test_parsed = test_raw.map(_parse_tfrecord, num_parallel_calls=AUTOTUNE)


@tf.function(jit_compile=False)
def _test_img_map(img, y, name):
    img = _preprocess(img)
    return img


@tf.function(jit_compile=False)
def _test_name_map(img, y, name):
    return name


test_img_ds = test_parsed.map(_test_img_map, num_parallel_calls=AUTOTUNE).cache()
test_img_ds = test_img_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
test_img_ds = _apply_ds_options(test_img_ds)

test_name_ds = test_parsed.map(_test_name_map, num_parallel_calls=AUTOTUNE).cache()
test_name_ds = test_name_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
test_name_ds = _apply_ds_options(test_name_ds)



## === cell 30
test_probs = model_trained.predict(test_img_ds, verbose=0)
test_pred = np.argmax(test_probs, axis=1)[: len(test_df)].astype(np.int64)

names = np.concatenate([nb.numpy() for nb in test_name_ds], axis=0).astype(str)[
    : len(test_df)
]

pred_df = pd.DataFrame({"image_id": names, "label": test_pred.astype(np.int64)})
test_pred_ordered = (
    test_df[["image_id"]]
    .merge(pred_df, on="image_id", how="left", sort=False)["label"]
    .to_numpy(dtype=np.int64)
)

len(test_pred_ordered), len(test_df)



## === cell 31
submission = submission_file.copy()
submission["label"] = test_pred_ordered.astype(int)
submission.head()



## === cell 32
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Saved at:", os.path.abspath("submission.csv"))
