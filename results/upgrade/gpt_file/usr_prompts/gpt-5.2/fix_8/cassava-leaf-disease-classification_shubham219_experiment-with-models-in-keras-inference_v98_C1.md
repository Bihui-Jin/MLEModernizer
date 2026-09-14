# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.11

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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl
from tensorflow.keras.models import Model

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

CANDIDATE_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/input/cassava-leaf-disease-classification",
]
DATA_ROOT = next((p for p in CANDIDATE_ROOTS if os.path.exists(p)), None)
if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )

TEST_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
TRAIN_DIR = os.path.join(DATA_ROOT, "train_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")

print("DATA_ROOT =", DATA_ROOT)
print("Num test images (glob):", len(glob.glob(TEST_GLOB)))
print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train images dir exists:", os.path.exists(TRAIN_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))




## === cell 1
NUM_CLASSES = 5
IMG_SIZE = (256, 256)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = tfl.Dense(NUM_CLASSES, activation="softmax")(base.output)
my_model = Model(inputs=base.input, outputs=x)

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

print("Model built. Output shape:", my_model.output_shape)




## === cell 2
train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image_id"].map(lambda x: os.path.join(TRAIN_DIR, x))

if train_df["path"].isna().any():
    raise ValueError("Some training paths are NaN; check train.csv/image_id parsing.")

preprocess = tf.keras.applications.resnet50.preprocess_input

BATCH_SIZE = 32
VAL_SPLIT = 0.1

AUTOTUNE = tf.data.AUTOTUNE

paths = train_df["path"].astype(str).values
labels = train_df["label"].astype(np.int32).values

n = len(paths)
indices = np.arange(n, dtype=np.int32)

rng = np.random.RandomState(SEED)
rng.shuffle(indices)

val_n = int(round(n * VAL_SPLIT))
val_idx = indices[:val_n]
train_idx = indices[val_n:]

train_paths = paths[train_idx]
train_labels = labels[train_idx]
val_paths = paths[val_idx]
val_labels = labels[val_idx]

rng_aug = np.random.RandomState(SEED)
train_n = len(train_paths)

flip_lr = rng_aug.rand(train_n) < 0.5

angle_deg = rng_aug.uniform(-15.0, 15.0, size=train_n).astype(np.float32)
angle_rad = angle_deg * (np.pi / 180.0)

max_dx = 0.05 * float(IMG_SIZE[1])
max_dy = 0.05 * float(IMG_SIZE[0])
dx = rng_aug.uniform(-max_dx, max_dx, size=train_n).astype(np.float32)
dy = rng_aug.uniform(-max_dy, max_dy, size=train_n).astype(np.float32)

zoom = rng_aug.uniform(0.9, 1.1, size=train_n).astype(np.float32)

train_flip = flip_lr.astype(np.bool_)
train_angle = angle_rad.astype(np.float32)
train_dx = dx.astype(np.float32)
train_dy = dy.astype(np.float32)
train_zoom = zoom.astype(np.float32)

IMG_H = IMG_SIZE[0]
IMG_W = IMG_SIZE[1]
OUT_SHAPE_T = tf.constant([IMG_H, IMG_W], dtype=tf.int32)
CX = tf.constant((IMG_W - 1) / 2.0, dtype=tf.float32)
CY = tf.constant((IMG_H - 1) / 2.0, dtype=tf.float32)


@tf.function(jit_compile=False)
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    return img


@tf.function(jit_compile=False)
def _apply_projective(img, transform):
    return tf.raw_ops.ImageProjectiveTransformV3(
        images=img[None, ...],
        transforms=transform,
        output_shape=OUT_SHAPE_T,
        interpolation="BILINEAR",
        fill_mode="NEAREST",
        fill_value=0.0,
    )[0]


@tf.function(jit_compile=False)
def _augment_with_params(img, flip, angle, dx, dy, zoom):
    img = tf.cond(flip, lambda: tf.image.flip_left_right(img), lambda: img)

    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)

    r00 = cos_a
    r01 = -sin_a
    r02 = CX - cos_a * CX + sin_a * CY
    r10 = sin_a
    r11 = cos_a
    r12 = CY - sin_a * CX - cos_a * CY

    t00 = 1.0
    t01 = 0.0
    t02 = dx
    t10 = 0.0
    t11 = 1.0
    t12 = dy

    inv_zoom = 1.0 / zoom
    z00 = inv_zoom
    z01 = 0.0
    z02 = CX - z00 * CX
    z10 = 0.0
    z11 = inv_zoom
    z12 = CY - z11 * CY

    m00 = z00 * (t00 * r00 + t01 * r10) + z01 * (t10 * r00 + t11 * r10)
    m01 = z00 * (t00 * r01 + t01 * r11) + z01 * (t10 * r01 + t11 * r11)
    m02 = (
        z00 * (t00 * r02 + t01 * r12 + t02) + z01 * (t10 * r02 + t11 * r12 + t12) + z02
    )

    m10 = z10 * (t00 * r00 + t01 * r10) + z11 * (t10 * r00 + t11 * r10)
    m11 = z10 * (t00 * r01 + t01 * r11) + z11 * (t10 * r01 + t11 * r11)
    m12 = (
        z10 * (t00 * r02 + t01 * r12 + t02) + z11 * (t10 * r02 + t11 * r12 + t12) + z12
    )

    transform = tf.stack([m00, m01, m02, m10, m11, m12, 0.0, 0.0])[None, :]
    img = _apply_projective(img, transform)
    return img


DS_OPTIONS = tf.data.Options()
DS_OPTIONS.experimental_deterministic = True  # preserve deterministic behavior
try:
    DS_OPTIONS.threading.private_threadpool_size = 0
except Exception:
    pass


def _make_ds(
    paths_arr,
    labels_arr=None,
    training=False,
    flip_arr=None,
    angle_arr=None,
    dx_arr=None,
    dy_arr=None,
    zoom_arr=None,
):
    if labels_arr is None:
        ds = tf.data.Dataset.from_tensor_slices(paths_arr)

        def map_fn(p):
            img = _decode_and_resize(p)
            img = preprocess(img)
            return img

        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.batch(16, drop_remainder=False)
        ds = ds.prefetch(AUTOTUNE)
        ds = ds.with_options(DS_OPTIONS)
        return ds

    if training:
        if flip_arr is None:
            raise ValueError("Training dataset requires augmentation parameter arrays.")
        ds = tf.data.Dataset.from_tensor_slices(
            (paths_arr, labels_arr, flip_arr, angle_arr, dx_arr, dy_arr, zoom_arr)
        )
        ds = ds.shuffle(
            buffer_size=min(len(paths_arr), 8192),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

        def map_fn(p, y, flip, angle, dx, dy, zoom):
            img = _decode_and_resize(p)
            img = _augment_with_params(img, flip, angle, dx, dy, zoom)
            img = preprocess(img)
            return img, y

        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths_arr, labels_arr))

        def map_fn(p, y):
            img = _decode_and_resize(p)
            img = preprocess(img)
            return img, y

        ds = ds.map(map_fn, num_parallel_calls=AUTOTUNE)
        ds = ds.cache()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.with_options(DS_OPTIONS)
    return ds


train_ds = _make_ds(
    train_paths,
    train_labels,
    training=True,
    flip_arr=train_flip,
    angle_arr=train_angle,
    dx_arr=train_dx,
    dy_arr=train_dy,
    zoom_arr=train_zoom,
)
val_ds = _make_ds(val_paths, val_labels, training=False)

base.trainable = False
my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=2,
    verbose=1,
)

base.trainable = True
for layer in base.layers[:-30]:
    layer.trainable = False

my_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

my_model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=1,
    verbose=1,
)




## === cell 3
test_images = glob.glob(TEST_GLOB)
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found at: {TEST_GLOB}")

df_test = pd.DataFrame(test_images, columns=["path"])

test_paths = df_test["path"].astype(str).values
test_ds = _make_ds(test_paths, labels_arr=None, training=False)

pred_test = my_model.predict(test_ds, verbose=1)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test.copy()
final_submission["image_id"] = (
    final_submission["path"].str.replace("\\", "/").str.split("/").str[-1]
)
final_submission["label"] = pred_test_labels.astype(int)

final_csv = final_submission[["image_id", "label"]]

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        final_csv["label"] = final_csv["label"].fillna(0).astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 4
final_csv.head()
