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

# 5. Target score

0.7263523723179208

# 6. Current score

0.32212

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11173) has done: 'I remove the incompatible `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash under Python 3.11, since it isn’t used by the solution. Because the referenced external weight file path doesn’t exist in this environment, I keep the same ResNet50-based core logic but instantiate a standard ImageNet-pretrained ResNet50 classifier (so `my_model` is defined and predictions can run). I also fix the test image path to the actual `/kaggle/input/...` location and add lightweight safety checks to ensure we always write a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 0.25448) has done: 'I remove the protobuf-implementation override that’s triggering the `MessageFactory.GetPrototype` crash, since it’s not needed for this script in Kaggle’s TF build. Then I fix the ResNet50 ImageNet fallback so it can actually load weights by using the required `(224, 224, 3)` input/target size when `include_top=True`. Finally, I keep the rest of the prediction/submission logic intact but ensure the generator target size matches the chosen model input size so `my_model` is defined and inference completes, producing a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.32212) has done: 'The timeout is most likely coming from expensive per-image augmentation using `ImageProjectiveTransformV3` plus disk caching overhead (writing/reading large `.cache` files), and from non-optimal tf.data pipeline ordering that prevents overlap of CPU decode/augment with GPU/TPU compute. I keep the exact model/training logic intact, but make the input pipeline faster by (1) switching caches to in-memory (no disk serialization), (2) fusing decode+resize into a single parallel map and then applying augmentation, and (3) enabling safe tf.data optimizations (`map_and_batch`, parallelism, prefetch) while preserving determinism/seeds. I also ensure `predict()` uses the dataset’s batch size (no separate hardcoded 32) and avoid unnecessary pandas work during submission creation.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import math
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as tfl  # kept since originally imported (even if unused)
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
)  # kept since originally imported
import matplotlib.pyplot as plt  # kept since originally imported (even if unused)

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
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

print("TF:", tf.__version__)
print("Keras:", tf.keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

weight_path = "/kaggle/input/resnet50-trasnfer-learning-on-tpu/cassava_base.h5"

if os.path.exists(weight_path):
    my_model = tf.keras.models.load_model(weight_path, compile=False)
    model_mode = "cassava_5class"
    target_size = (256, 256)
else:
    target_size = (224, 224)
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights="imagenet",
        input_shape=(target_size[0], target_size[1], 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inputs=base.input, outputs=out)
    model_mode = "trained_5class_from_imagenet_backbone"

print("model_mode:", model_mode, "| target_size:", target_size)




## === cell 2
if not os.path.exists(TRAIN_CSV):
    TRAIN_CSV = "/kaggle/input/train.csv"
train_df = pd.read_csv(TRAIN_CSV)

train_df["path"] = TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)

if (not os.path.exists(TRAIN_IMG_DIR)) or (
    len(train_df) and (not os.path.exists(train_df["path"].iloc[0]))
):
    TRAIN_IMG_DIR = "../input/cassava-leaf-disease-classification/train_images"
    train_df["path"] = (
        TRAIN_IMG_DIR.rstrip("/") + "/" + train_df["image_id"].astype(str)
    )

preprocess = tf.keras.applications.resnet50.preprocess_input

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

df_tr = train_df.iloc[tr_idx].copy()
df_val = train_df.iloc[val_idx].copy()

BATCH = 32
AUTOTUNE = tf.data.AUTOTUNE
IMG_H, IMG_W = target_size

_ROT_DEG = 10.0
_W_SHIFT = 0.05
_H_SHIFT = 0.05
_ZOOM = 0.1

IMG_H_F = tf.constant(float(IMG_H), dtype=tf.float32)
IMG_W_F = tf.constant(float(IMG_W), dtype=tf.float32)
CX = tf.constant((float(IMG_W) - 1.0) / 2.0, dtype=tf.float32)
CY = tf.constant((float(IMG_H) - 1.0) / 2.0, dtype=tf.float32)
PI_OVER_180 = tf.constant(math.pi / 180.0, dtype=tf.float32)
OUT_SHAPE = tf.constant([IMG_H, IMG_W], dtype=tf.int32)
FILL_VALUE = tf.constant(0.0, dtype=tf.float32)
ZERO = tf.constant(0.0, dtype=tf.float32)


@tf.function
def _decode_resize_img(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_H, IMG_W], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


@tf.function
def _rand_affine_like_idg(img):
    angle = tf.random.uniform([], -_ROT_DEG, _ROT_DEG, dtype=tf.float32) * PI_OVER_180
    tx = tf.random.uniform([], -_H_SHIFT * IMG_H_F, _H_SHIFT * IMG_H_F)
    ty = tf.random.uniform([], -_W_SHIFT * IMG_W_F, _W_SHIFT * IMG_W_F)
    zoom = tf.random.uniform([], 1.0 - _ZOOM, 1.0 + _ZOOM, dtype=tf.float32)

    cos_a = tf.cos(angle) * zoom
    sin_a = tf.sin(angle) * zoom

    a0 = cos_a
    a1 = -sin_a
    b0 = sin_a
    b1 = cos_a

    a2 = CX - a0 * CX - a1 * CY + ty
    b2 = CY - b0 * CX - b1 * CY + tx

    transform = tf.stack([a0, a1, a2, b0, b1, b2, ZERO, ZERO])[tf.newaxis, :]
    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=img[tf.newaxis, ...],
        transforms=transform,
        output_shape=OUT_SHAPE,
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=FILL_VALUE,
    )[0]
    img = tf.image.random_flip_left_right(img)
    img.set_shape([IMG_H, IMG_W, 3])
    return img


tr_paths = df_tr["path"].values.astype(str)
tr_labels = df_tr["label"].values.astype(np.int32)
val_paths = df_val["path"].values.astype(str)
val_labels = df_val["label"].values.astype(np.int32)

SHUF_BUF = int(min(len(tr_paths), 4096))

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.experimental_optimization.map_parallelization = True
opts.experimental_optimization.parallel_batch = True
opts.experimental_optimization.apply_default_optimizations = True
opts.experimental_optimization.autotune_buffers = True


@tf.function
def _decode_only_map(path, label):
    return _decode_resize_img(path), tf.cast(label, tf.int32)


@tf.function
def _aug_preprocess(img, label):
    img = _rand_affine_like_idg(img)
    img = preprocess(img)
    return img, label


@tf.function
def _preprocess_only(img, label):
    return preprocess(img), label


train_base = tf.data.Dataset.from_tensor_slices((tr_paths, tr_labels)).with_options(
    opts
)
train_base = train_base.shuffle(
    buffer_size=SHUF_BUF, seed=SEED, reshuffle_each_iteration=True
)

train_dec = train_base.map(_decode_only_map, num_parallel_calls=AUTOTUNE).cache()

train_ds = (
    train_dec.map(_aug_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_base = tf.data.Dataset.from_tensor_slices((val_paths, val_labels)).with_options(
    opts
)
val_dec = val_base.map(_decode_only_map, num_parallel_calls=AUTOTUNE).cache()

val_ds = (
    val_dec.map(_preprocess_only, num_parallel_calls=AUTOTUNE)
    .batch(BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3702468254.py in <cell line: 0>()
    103 # Speed fix: allow graph-level dataset optimizations while keeping determinism enabled above.
    104 opts.experimental_optimization.apply_default_optimizations = True
--> 105 opts.experimental_optimization.autotune_buffers = True
    106 
    107 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 3
if model_mode != "cassava_5class":
    for layer in my_model.layers:
        if isinstance(layer, tf.keras.Model):
            layer.trainable = False
    for layer in my_model.layers:
        if "resnet50" in layer.name:
            layer.trainable = False

    my_model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    EPOCHS = 3

    my_model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        verbose=1,
    )




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3766625861.py in <cell line: 0>()
     16 
     17     my_model.fit(
---> 18         train_ds,
     19         validation_data=val_ds,
     20         epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 4
test_images = tf.io.gfile.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
if len(test_images) == 0:
    test_images = tf.io.gfile.glob(
        "../input/cassava-leaf-disease-classification/test_images/*.jpg"
    )

df_test = pd.DataFrame({"path": test_images})
if df_test.empty:
    raise RuntimeError("No test images found. Check the test_images path.")

test_paths = df_test["path"].values.astype(str)

test_base = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
test_dec = test_base.map(_decode_resize_img, num_parallel_calls=AUTOTUNE).cache()


@tf.function
def _test_preprocess(img):
    return preprocess(img)


test_ds = (
    test_dec.map(_test_preprocess, num_parallel_calls=AUTOTUNE)
    .batch(BATCH, drop_remainder=False)
    .prefetch(AUTOTUNE)
)




## === cell 5
pred = my_model.predict(
    test_ds,
    verbose=1,
)

pred_labels = np.argmax(pred, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.rsplit("/", n=1).str[-1]
final_submission["label"] = pred_labels
final_csv = final_submission[["image_id", "label"]]

if os.path.exists(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
else:
    sample = pd.read_csv("/kaggle/input/sample_submission.csv")

final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")

if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label)

final_csv["label"] = final_csv["label"].astype(int)

assert len(final_csv) == len(
    sample
), "Submission row count mismatch vs sample_submission.csv"
assert list(final_csv.columns) == ["image_id", "label"], "Submission columns mismatch"

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)




## === cell 6
final_csv.head()
