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

3.13

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

0.8779087337564219

# 6. Current score

0.14425

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14163) has done: 'Main bottlenecks are the expensive per-image augmentation (projective transform with Python-level helper functions) executed every epoch, plus unnecessary `cache()` for training (which forces storing all decoded 224x224x3 float images in RAM and can slow/pressure memory). The core model and training loop stay identical, but we (1) precompute labels via vectorized mapping, (2) compile the tf.data pipeline with `@tf.function` and move constant tensors outside the hot path, (3) remove training cache while keeping validation cache (equivalent semantics, faster), and (4) add `steps_per_execution` to reduce Python overhead during `fit` without changing the math. We also avoid the expensive `enumerate()`/seed building by using `Dataset.random()` with a fixed seed and stateless ops, keeping determinism. These changes preserve accuracy while cutting input pipeline overhead substantially to fit the 600s budget.'
- What this solution (achieved 0.14425) has done: 'The timeout is dominated by expensive CPU-side image augmentation (projective transforms) and by caching full decoded images to disk for train/valid/test, which creates huge I/O and serialization overhead. I keep the exact model, epochs, loss, and augmentation math, but restructure the `tf.data` pipeline to (1) cache only compressed JPEG bytes (much smaller and faster to write/read), (2) decode+resize from cached bytes each epoch, and (3) enable dataset optimizations and non-deterministic parallelism safely without changing results. I also avoid writing/reading multiple cache shards by using a single cache file per split and keep determinism seeds intact while allowing faster pipeline scheduling.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ["PYTHONHASHSEED"] = str(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

BASE_PATH = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV_PATH = f"{BASE_PATH}/train.csv"
TRAIN_IMG_DIR = f"{BASE_PATH}/train_images"
SAMPLE_SUB_PATH = f"{BASE_PATH}/sample_submission.csv"
TEST_IMG_DIR = f"{BASE_PATH}/test_images"
LABEL_MAP_PATH = f"{BASE_PATH}/label_num_to_disease_map.json"

label_to_disease = pd.read_json(LABEL_MAP_PATH, typ="series")
train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_csv["disease"] = train_csv["label"].map(label_to_disease).astype(str)
train_csv["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_csv["image_id"]
train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv,
    test_size=0.2,
    stratify=train_csv["label"],
    random_state=SEED,
)

classes = sorted(train_csv["disease"].unique().tolist())
class_indices = {c: i for i, c in enumerate(classes)}
NUM_CLASSES = len(classes)

print("Classes found (tf.data):", class_indices)

IMG_SIZE = (224, 224)
BATCH_SIZE = 32

IMG_H = tf.constant(IMG_SIZE[0], tf.int32)
IMG_W = tf.constant(IMG_SIZE[1], tf.int32)
OUT_SHAPE = tf.stack([IMG_H, IMG_W])
CX = tf.constant((IMG_SIZE[1] - 1) / 2.0, tf.float32)
CY = tf.constant((IMG_SIZE[0] - 1) / 2.0, tf.float32)
PI_OVER_180 = tf.constant(np.pi / 180.0, tf.float32)
SEED_I64 = tf.constant(SEED, tf.int64)

CACHE_DIR = "/kaggle/working/tfds_cache"
tf.io.gfile.makedirs(CACHE_DIR)

TRAIN_BYTES_CACHE = os.path.join(CACHE_DIR, "train_jpeg_bytes.cache")
VALID_BYTES_CACHE = os.path.join(CACHE_DIR, "valid_jpeg_bytes.cache")
TEST_BYTES_CACHE = os.path.join(CACHE_DIR, "test_jpeg_bytes.cache")


@tf.function
def _seed_from_path(path):
    h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
    h = tf.cast(h, tf.int64) ^ SEED_I64
    return tf.stack([h * 1315423911, h * 2654435761], axis=0)


@tf.function
def _read_jpeg_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_and_resize(jpeg_bytes):
    img = tf.io.decode_jpeg(jpeg_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(img, (IMG_H, IMG_W), method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


@tf.function
def _augment_like_imagedatagenerator(img, seed2):
    img = tf.image.stateless_random_flip_left_right(img, seed=(seed2[0], seed2[1] + 1))
    img = tf.image.stateless_random_flip_up_down(img, seed=(seed2[0], seed2[1] + 2))

    angle = (
        tf.random.stateless_uniform(
            [], seed=(seed2[0], seed2[1] + 3), minval=-45.0, maxval=45.0
        )
        * PI_OVER_180
    )

    tx = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 4), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_H, tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 5), minval=-0.2, maxval=0.2
    ) * tf.cast(IMG_W, tf.float32)

    shear = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 6), minval=-0.2, maxval=0.2
    )

    zoom = tf.random.stateless_uniform(
        [], seed=(seed2[0], seed2[1] + 7), minval=0.8, maxval=1.2
    )

    cos_a = tf.math.cos(angle) / zoom
    sin_a = tf.math.sin(angle) / zoom

    shear_m = tf.reshape(
        tf.stack([1.0, tf.math.tan(shear), 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]),
        (3, 3),
    )
    rot_m = tf.reshape(
        tf.stack([cos_a, -sin_a, 0.0, sin_a, cos_a, 0.0, 0.0, 0.0, 1.0]),
        (3, 3),
    )

    m = tf.linalg.matmul(rot_m, shear_m)

    a0, a1, a2, b0, b1, b2, c0, c1, _ = tf.unstack(tf.reshape(m, (-1,)))
    a2 = a2 + CX - (a0 * CX + a1 * CY) - ty
    b2 = b2 + CY - (b0 * CX + b1 * CY) - tx

    t = tf.stack([a0, a1, a2, b0, b1, b2, c0, c1])[None, :]

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=t,
        output_shape=OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]
    return img


@tf.function
def _preprocess(img):
    return preprocess_input(img)


@tf.function
def _path_to_jpeg_bytes(path):
    return _read_jpeg_bytes(path)


@tf.function
def _jpeg_bytes_to_base_img(jpeg_bytes):
    return _decode_and_resize(jpeg_bytes)


@tf.function
def _bytes_to_xy(jpeg_bytes, path, label, training):
    base_img = _jpeg_bytes_to_base_img(jpeg_bytes)
    img = base_img
    if training:
        seed2 = _seed_from_path(path)
        img = _augment_like_imagedatagenerator(img, seed2)
    img = _preprocess(img)
    label_oh = tf.one_hot(label, NUM_CLASSES, dtype=tf.float32)
    return img, label_oh


def make_dataset(df, training, bytes_cache_path):
    paths = df["path"].to_numpy(dtype=object)
    y = df["disease"].map(class_indices).to_numpy(dtype=np.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    options = tf.data.Options()
    options.experimental_deterministic = False
    options.autotune.enabled = True
    options.threading.private_threadpool_size = 0
    options.experimental_optimization.map_fusion = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.parallel_filter = True
    ds = ds.with_options(options)

    if training:
        shuffle_buf = min(len(df), 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    ds = ds.map(
        lambda p, l: (p, l, _path_to_jpeg_bytes(p)),
        num_parallel_calls=AUTOTUNE,
    ).cache(bytes_cache_path)

    ds = ds.map(
        lambda p, l, jb: _bytes_to_xy(jb, p, l, training),
        num_parallel_calls=AUTOTUNE,
    )

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(train, training=True, bytes_cache_path=TRAIN_BYTES_CACHE)
valid_ds = make_dataset(valid, training=False, bytes_cache_path=VALID_BYTES_CACHE)

steps_per_epoch = (len(train) + BATCH_SIZE - 1) // BATCH_SIZE
validation_steps = (len(valid) + BATCH_SIZE - 1) // BATCH_SIZE




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3293787103.py in <cell line: 0>()
    203 
    204 
--> 205 train_ds = make_dataset(train, training=True, bytes_cache_path=TRAIN_BYTES_CACHE)
    206 valid_ds = make_dataset(valid, training=False, bytes_cache_path=VALID_BYTES_CACHE)
    207 

/tmp/ipykernel_11/3293787103.py in make_dataset(df, training, bytes_cache_path)
    177     options.experimental_optimization.map_parallelization = True
    178     options.experimental_optimization.parallel_batch = True
--> 179     options.experimental_optimization.parallel_filter = True
    180     ds = ds.with_options(options)
    181 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property parallel_filter on OptimizationOptions.

## === cell 2
base_model = EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
)

x = base_model.output
x = GlobalAveragePooling2D()(x)
preds = Dense(NUM_CLASSES, activation="softmax")(x)
model = Model(inputs=base_model.input, outputs=preds)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=steps_per_epoch,  # executes an epoch as a single tf.function call
)

EPOCHS = 3

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2027049416.py in <cell line: 0>()
     14     loss="categorical_crossentropy",
     15     metrics=["accuracy"],
---> 16     steps_per_execution=steps_per_epoch,  # executes an epoch as a single tf.function call
     17 )
     18 

NameError: name 'steps_per_epoch' is not defined

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub["path"] = TEST_IMG_DIR.rstrip("/") + "/" + sample_sub["image_id"]

test_paths = sample_sub["path"].to_numpy(dtype=object)
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


@tf.function
def _bytes_to_x(jpeg_bytes):
    base_img = _jpeg_bytes_to_base_img(jpeg_bytes)
    return _preprocess(base_img)


test_ds = (
    test_ds.map(_path_to_jpeg_bytes, num_parallel_calls=AUTOTUNE)
    .cache(TEST_BYTES_CACHE)
    .map(_bytes_to_x, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE)
    .prefetch(AUTOTUNE)
)

probs = model.predict(
    test_ds,
    verbose=1,
)

inv_class_indices = {v: k for k, v in class_indices.items()}  # idx -> disease_str
disease_to_labelid = {str(v): int(k) for k, v in label_to_disease.to_dict().items()}

pred_class_idx = np.argmax(probs, axis=1).astype(int)
pred_disease = [inv_class_indices[i] for i in pred_class_idx]
pred_labels = np.array([disease_to_labelid[str(d)] for d in pred_disease], dtype=int)

submission_df = pd.DataFrame(
    {
        "image_id": sample_sub["image_id"].values,
        "label": pred_labels,
    }
)

submission_path = "/kaggle/working/submission.csv"
submission_df.to_csv(submission_path, index=False)
print("Submission file created:", submission_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
print(
    "Label distribution:\n",
    submission_df["label"].value_counts(dropna=False).sort_index(),
)
