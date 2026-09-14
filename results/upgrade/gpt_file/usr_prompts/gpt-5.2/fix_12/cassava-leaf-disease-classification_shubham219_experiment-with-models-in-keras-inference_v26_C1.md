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

0.703838017527954

# 6. Current score

0.13976

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.15807) has done: 'I fix the initial TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before importing TensorFlow, which avoids the `MessageFactory.GetPrototype` error in many Kaggle images. Then I remove the dependency on a missing external `.h5` file by instantiating the same kind of Keras classifier locally (EfficientNet backbone + softmax head) so the notebook runs end-to-end. Finally, I make the test generator robust (correct file paths, ensure prediction length matches test rows) and always write a valid `submission.csv` with the required `image_id,label` columns.'
- What this solution (achieved 0.13976) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in many Kaggle runtimes. Then I fix the TFRecord parsing logic: the cassava TFRecords don’t contain `image_id`, so I remove it from the required feature spec and avoid train/val filtering by `image_id` when using TFRecords (while preserving the same model/training loop via a simple `validation_split` on the dataframe generator fallback). Finally, I make test inference always produce a correctly aligned `submission.csv` by using `sample_submission.csv` for ordering and predicting from either TFRecords (without `image_id`) or the JPEG directory, ensuring the file is always written.'
- What this solution (achieved 0.13976) has done: 'The timeout is dominated by slow input pipelines: `ImageDataGenerator.flow_from_dataframe` + wrapping it with `tf.data.Dataset.from_generator` forces Python to sit in the training loop and starves the model, and you also force the slow pure-Python protobuf implementation. I keep the same model, loss, optimizer, epochs, and the same data augmentation semantics, but move training/validation loading to a fully-TensorFlow `tf.data` pipeline that does decode/resize/preprocess + the exact same augmentations on-device with parallel map and prefetch. For test, I keep TFRecords but optimize decoding (fused jpeg decode+resize) and batching/prefetch. These changes remove Python from the hot path and typically cut epoch time dramatically while preserving the algorithm and accuracy.'
- What this solution (achieved 0.13976) has done: 'The timeout is dominated by JPEG file I/O + expensive per-image augmentation in the `tf.data` pipeline while training EfficientNetB3 for multiple epochs. I keep the exact same model and training semantics, but make the input pipeline cheaper and more parallel by (1) precomputing image paths once (removing pandas `apply`/lambda overhead), (2) enabling `tf.data` map/batch fusion and running augmentation in parallel with `num_parallel_calls=AUTOTUNE`, (3) adding `prefetch_to_device('/GPU:0')` when a GPU is present to overlap host decoding with device compute, and (4) using `cache()` only for the *validation* dataset (no randomness) to eliminate repeated decode/resize across epochs without changing results. These changes are equivalent in outputs (same preprocessing/augmentation/model/training steps), but reduce input overhead enough to fit under 600s.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import glob
import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from tensorflow.keras.applications import EfficientNetB3

SEED = 42
DEBUG = False

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_TRAIN_CSV = [
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    "/kaggle/data/cassava-leaf-disease-classification/train.csv",
    "../input/cassava-leaf-disease-classification/train.csv",
    "../data/cassava-leaf-disease-classification/train.csv",
]
CANDIDATE_TRAIN_IMG_DIRS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/data/cassava-leaf-disease-classification/train_images",
    "../input/cassava-leaf-disease-classification/train_images",
    "../data/cassava-leaf-disease-classification/train_images",
]
CANDIDATE_TRAIN_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/train_tfrecords/*.tfrec",
]

train_csv_path = None
for p in CANDIDATE_TRAIN_CSV:
    if os.path.exists(p):
        train_csv_path = p
        break
if train_csv_path is None:
    raise FileNotFoundError(
        "Could not find train.csv. Tried:\n" + "\n".join(CANDIDATE_TRAIN_CSV)
    )

train_img_dir = None
for d in CANDIDATE_TRAIN_IMG_DIRS:
    if os.path.isdir(d):
        train_img_dir = d
        break
if train_img_dir is None:
    raise FileNotFoundError(
        "Could not find train_images dir. Tried:\n"
        + "\n".join(CANDIDATE_TRAIN_IMG_DIRS)
    )

train_tfrec_files = []
for pat in CANDIDATE_TRAIN_TFREC_GLOBS:
    train_tfrec_files = sorted(glob.glob(pat))
    if len(train_tfrec_files) > 0:
        break

train_df = pd.read_csv(train_csv_path)
if DEBUG:
    train_df = train_df.sample(2000, random_state=SEED).reset_index(drop=True)

train_df["label"] = train_df["label"].astype(int)
train_df["image_id"] = train_df["image_id"].astype(str)

print("Train rows:", len(train_df), "train_img_dir:", train_img_dir)
print("Found train tfrecords:", len(train_tfrec_files))




## === cell 2
NUM_CLASSES = 5
IMG_SIZE = (300, 300)

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = keras.layers.Dropout(0.2)(base.output)
out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs=base.input, outputs=out)

my_model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["sparse_categorical_accuracy"],
)

my_model.summary()




## === cell 3
val_frac = 0.1
train_df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df_shuf) * val_frac)
val_df = train_df_shuf.iloc[:n_val].copy()
trn_df = train_df_shuf.iloc[n_val:].copy()

preprocess_fn = tf.keras.applications.efficientnet.preprocess_input

batch_size = 32 if not DEBUG else 16
EPOCHS = 2 if DEBUG else 3

steps_per_epoch = int(np.ceil(len(trn_df) / batch_size))
validation_steps = int(np.ceil(len(val_df) / batch_size))

AUTOTUNE = tf.data.AUTOTUNE

tf_data_opts = tf.data.Options()
tf_data_opts.deterministic = True

try:
    tf_data_opts.experimental_optimization.map_fusion = True
    tf_data_opts.experimental_optimization.map_parallelization = True
    tf_data_opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

use_tfrecs_for_train = False

try:
    import tensorflow_addons as tfa  # may or may not exist in the Kaggle image

    _HAS_TFA = True
except Exception:
    tfa = None
    _HAS_TFA = False


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


@tf.function
def _maybe_rotate(img, angle_radians):
    if _HAS_TFA:
        return tfa.image.rotate(
            img, angle_radians, interpolation="BILINEAR", fill_mode="reflect"
        )
    return img


@tf.function
def _augment_like_idg(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)

    angle = tf.random.uniform([], -15.0, 15.0, seed=SEED) * (np.pi / 180.0)
    img = _maybe_rotate(img, angle)

    h = tf.shape(img)[0]
    w = tf.shape(img)[1]
    pad_h = tf.cast(tf.cast(h, tf.float32) * 0.05, tf.int32)
    pad_w = tf.cast(tf.cast(w, tf.float32) * 0.05, tf.int32)
    img = tf.pad(img, [[pad_h, pad_h], [pad_w, pad_w], [0, 0]], mode="REFLECT")
    img = tf.image.random_crop(img, size=[h, w, 3], seed=SEED)

    scale = tf.random.uniform([], 0.9, 1.1, seed=SEED)
    new_h = tf.cast(tf.cast(h, tf.float32) * scale, tf.int32)
    new_w = tf.cast(tf.cast(w, tf.float32) * scale, tf.int32)
    img2 = tf.image.resize(img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR)
    img2 = tf.image.resize_with_crop_or_pad(img2, h, w)
    return img2


_HAS_GPU = bool(tf.config.list_physical_devices("GPU"))


def _make_path_ds(df, shuffle, augment, cache_ds):
    image_ids = df["image_id"].astype(str).to_numpy()
    paths = np.char.add(np.char.add(train_img_dir, os.sep), image_ids).astype("U")
    labels = df["label"].astype(np.int32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if shuffle:
        ds = ds.shuffle(len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.with_options(tf_data_opts)

    def _load(path, label):
        img = _decode_resize_preprocess(path)
        if augment:
            img = _augment_like_idg(img)
        return img, tf.cast(label, tf.int32)

    ds = ds.map(_load, num_parallel_calls=AUTOTUNE, deterministic=True)
    if cache_ds:
        ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)

    if _HAS_GPU:
        try:
            ds = ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            ds = ds.prefetch(AUTOTUNE)
    else:
        ds = ds.prefetch(AUTOTUNE)
    return ds


if use_tfrecs_for_train:
    raise RuntimeError(
        "TFRecord training disabled because train TFRecords lack labels."
    )
else:
    train_ds = _make_path_ds(trn_df, shuffle=True, augment=True, cache_ds=False)
    val_ds = _make_path_ds(val_df, shuffle=False, augment=False, cache_ds=True)

history = my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2817259040.py in <cell line: 0>()
    121     )
    122 else:
--> 123     train_ds = _make_path_ds(trn_df, shuffle=True, augment=True, cache_ds=False)
    124     val_ds = _make_path_ds(val_df, shuffle=False, augment=False, cache_ds=True)
    125 

/tmp/ipykernel_11/2817259040.py in _make_path_ds(df, shuffle, augment, cache_ds)
     87 def _make_path_ds(df, shuffle, augment, cache_ds):
     88     image_ids = df["image_id"].astype(str).to_numpy()
---> 89     paths = np.char.add(np.char.add(train_img_dir, os.sep), image_ids).astype("U")
     90     labels = df["label"].astype(np.int32).to_numpy()
     91 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U63' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 4
CANDIDATE_TEST_TFREC_GLOBS = [
    "/kaggle/input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "/kaggle/data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../input/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
    "../data/cassava-leaf-disease-classification/test_tfrecords/*.tfrec",
]
test_tfrec_files = []
for pat in CANDIDATE_TEST_TFREC_GLOBS:
    test_tfrec_files = sorted(glob.glob(pat))
    if len(test_tfrec_files) > 0:
        break


def _decode_and_resize(image_bytes):
    img = tf.image.decode_jpeg(image_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


_FEATURE_DESCRIPTION_TEST = {
    "image": tf.io.FixedLenFeature([], tf.string),
}


def _parse_test_example(serialized):
    ex = tf.io.parse_single_example(serialized, _FEATURE_DESCRIPTION_TEST)
    img = _decode_and_resize(ex["image"])
    img = preprocess_fn(img)
    return img


if len(test_tfrec_files) > 0:
    test_ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
    test_ds = test_ds.with_options(tf_data_opts)
    test_ds = test_ds.map(
        _parse_test_example, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    test_ds = test_ds.batch(64, drop_remainder=False)
    if _HAS_GPU:
        try:
            test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            test_ds = test_ds.prefetch(AUTOTUNE)
    else:
        test_ds = test_ds.prefetch(AUTOTUNE)
    df_test = None  # will be taken from sample_submission
else:
    CANDIDATE_TEST_GLOBS = [
        "/kaggle/input/cassava-leaf-disease-classification/test_images/*.jpg",
        "/kaggle/data/cassava-leaf-disease-classification/test_images/*.jpg",
        "../input/cassava-leaf-disease-classification/test_images/*.jpg",
        "../data/cassava-leaf-disease-classification/test_images/*.jpg",
    ]
    test_images = []
    for pat in CANDIDATE_TEST_GLOBS:
        test_images = glob.glob(pat)
        if len(test_images) > 0:
            break

    if len(test_images) == 0:
        raise FileNotFoundError(
            "Could not find test images. Tried:\n" + "\n".join(CANDIDATE_TEST_GLOBS)
        )

    df_test = pd.DataFrame({"path": test_images})
    df_test["image_id"] = df_test["path"].astype(str).str.split("/").str[-1]

    test_paths = df_test["path"].astype(str).values
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.with_options(tf_data_opts)

    def _load_test(path):
        return _decode_resize_preprocess(path)

    test_ds = test_ds.map(_load_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    test_ds = test_ds.batch(64, drop_remainder=False)
    if _HAS_GPU:
        try:
            test_ds = test_ds.apply(tf.data.experimental.prefetch_to_device("/GPU:0"))
        except Exception:
            test_ds = test_ds.prefetch(AUTOTUNE)
    else:
        test_ds = test_ds.prefetch(AUTOTUNE)




## === cell 5
CANDIDATE_SAMPLE = [
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
    "../input/cassava-leaf-disease-classification/sample_submission.csv",
    "../data/cassava-leaf-disease-classification/sample_submission.csv",
]
sample_path = None
for p in CANDIDATE_SAMPLE:
    if os.path.exists(p):
        sample_path = p
        break
if sample_path is None:
    raise FileNotFoundError(
        "Could not find sample_submission.csv. Tried:\n" + "\n".join(CANDIDATE_SAMPLE)
    )

sample = pd.read_csv(sample_path)
sample["image_id"] = sample["image_id"].astype(str)
print("sample_submission rows:", len(sample))

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(np.int64)

if df_test is None:
    n = len(sample)
    if len(pred_test_labels) != n:
        if len(pred_test_labels) > n:
            pred_test_labels = pred_test_labels[:n]
        else:
            pred_test_labels = np.pad(
                pred_test_labels, (0, n - len(pred_test_labels)), constant_values=0
            )
    final_csv = sample[["image_id"]].copy()
    final_csv["label"] = pred_test_labels.astype(int)
else:
    tmp = df_test[["image_id"]].copy()
    tmp["label"] = pred_test_labels[: len(tmp)].astype(int)
    final_csv = sample[["image_id"]].merge(tmp, on="image_id", how="left")
    final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)
print(final_csv.head())
print("Wrote submission.csv with", len(final_csv), "rows")
print("Saved: submission.csv")
