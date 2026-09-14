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

0.6668177697189483

# 6. Current score

0.15845

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.06839) has done: 'Main bottlenecks are (1) the expensive Python-based protobuf implementation forced via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, (2) rebuilding and re-reading the entire `tf.data` pipeline from disk for each TTA pass, and (3) non-vectorized existence checks over ~2.6k files. I switch protobuf back to the default C++ implementation (keeping behavior identical), build the base decoded dataset once and reuse it across TTA passes (so JPEG decode/resize happens only once), and replace the Python loop file check with a vectorized `tf.io.gfile.glob`-based set membership check. TTA randomness/semantics and the model call pattern remain the same (still one non-TTA pass + `TTA_PASSES` augmented passes), but runtime drops significantly because disk decode/resize is no longer repeated.'
- What this solution (achieved 0.15845) has done: 'The timeout is dominated by expensive image decoding/resizing repeated across 4 full prediction passes (1 base + 3 TTA), plus a per-element TTA pipeline that uses random ops with a fixed seed (so it doesn’t actually create different augmentations across passes) while still paying the full augmentation cost each time. I keep identical model/prediction semantics, but remove the redundant TTA loop by computing the TTA-transformed dataset exactly once and reusing that single deterministic prediction for all TTA passes (mathematically identical given the fixed seed). I also eliminate an O(N) set-diff filesystem check that scans the whole directory and replace it with a direct existence check on just the referenced files (same correctness, less overhead). Finally, I keep the same batching/prefetching logic but avoid `repeat()` and extra iterator construction that can add overhead and unpredictability.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.threading.set_inter_op_parallelism_threads(0)
    tf.config.threading.set_intra_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
preferred_weight_path = "../input/model-v14/model_v0.25.h5"

candidate_paths = []
if os.path.exists(preferred_weight_path):
    candidate_paths = [preferred_weight_path]
else:
    likely_paths = [
        "../input",
        "/kaggle/input",
    ]
    for base in likely_paths:
        if os.path.isdir(base):
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*.h5")))
            candidate_paths += sorted(glob.glob(os.path.join(base, "*", "*", "*.h5")))

if candidate_paths:
    weight_path = candidate_paths[0]
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False)

    in_shape = my_model.input_shape
    if isinstance(in_shape, list):
        in_shape = in_shape[0]
    if (
        in_shape is None
        or len(in_shape) != 4
        or in_shape[1] is None
        or in_shape[2] is None
    ):
        MODEL_INPUT_SIZE = (300, 300)
    else:
        MODEL_INPUT_SIZE = (int(in_shape[1]), int(in_shape[2]))
    NEED_PREPROCESS = False  # keep original behavior for trained .h5
    print("Model input size:", MODEL_INPUT_SIZE)
else:
    print(
        "No .h5 model found under /kaggle/input or ../input. Using EfficientNetB3(ImageNet) fallback."
    )
    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB3
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    MODEL_INPUT_SIZE = (300, 300)
    base = EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=(MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3),
        pooling="avg",
    )
    x = layers.Dense(5, activation="softmax")(base.output)
    my_model = Model(inputs=base.input, outputs=x)
    NEED_PREPROCESS = True

    _ = my_model(
        tf.zeros((1, MODEL_INPUT_SIZE[0], MODEL_INPUT_SIZE[1], 3), dtype=tf.float32)
    )
    print("Fallback model built:", my_model.name)




## === cell 2
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
if not os.path.exists(sample_sub_path):
    sample_sub_path = (
        "../input/cassava-leaf-disease-classification/sample_submission.csv"
    )

sample_sub = pd.read_csv(sample_sub_path)

test_img_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
if not os.path.isdir(test_img_dir):
    test_img_dir = "../input/cassava-leaf-disease-classification/test_images"

df_test = pd.DataFrame({"image_id": sample_sub["image_id"].astype(str).values})
df_test["path"] = test_img_dir.rstrip("/") + "/" + df_test["image_id"].values

paths_np = df_test["path"].to_numpy()

missing = [p for p in paths_np if not tf.io.gfile.exists(p)]
if missing:
    raise FileNotFoundError(
        f"{len(missing)} test images listed in sample_submission.csv were not found in {test_img_dir}. "
        f"Example missing: {missing[0]}"
    )

if NEED_PREPROCESS:
    from tensorflow.keras.applications.efficientnet import (
        preprocess_input as effnet_preprocess,
    )

    def _preprocess_if_needed(x):
        return effnet_preprocess(x)

else:

    def _preprocess_if_needed(x):
        return x


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    img = _preprocess_if_needed(img)
    return img


@tf.function(reduce_retracing=True)
def _apply_tta(img):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    h = MODEL_INPUT_SIZE[0]
    w = MODEL_INPUT_SIZE[1]
    pad_h = tf.cast(tf.round(0.05 * tf.cast(h, tf.float32)), tf.int32)
    pad_w = tf.cast(tf.round(0.05 * tf.cast(w, tf.float32)), tf.int32)
    img = tf.image.resize_with_crop_or_pad(img, h + 2 * pad_h, w + 2 * pad_w)
    img = tf.image.random_crop(img, size=[h, w, 3], seed=SEED)

    zoom = tf.random.uniform([], minval=0.90, maxval=1.0, seed=SEED)
    crop_h = tf.cast(tf.round(zoom * tf.cast(h, tf.float32)), tf.int32)
    crop_w = tf.cast(tf.round(zoom * tf.cast(w, tf.float32)), tf.int32)
    img = tf.image.random_crop(img, size=[crop_h, crop_w, 3], seed=SEED)
    img = tf.image.resize(img, MODEL_INPUT_SIZE, method=tf.image.ResizeMethod.BILINEAR)

    angle = tf.random.uniform([], minval=-10.0, maxval=10.0, seed=SEED) * (
        np.pi / 180.0
    )
    cos_a = tf.math.cos(angle)
    sin_a = tf.math.sin(angle)
    tx = 0.0
    ty = 0.0
    transform = tf.stack([cos_a, -sin_a, tx, sin_a, cos_a, ty, 0.0, 0.0])[tf.newaxis, :]
    try:
        img = tf.raw_ops.ImageProjectiveTransformV3(
            images=img[tf.newaxis, ...],
            transforms=transform,
            output_shape=tf.constant([h, w], dtype=tf.int32),
            interpolation="BILINEAR",
            fill_mode="REFLECT",
            fill_value=0.0,
        )[0]
    except Exception:
        pass

    return img


def make_base_test_ds():
    ds = tf.data.Dataset.from_tensor_slices(paths_np)
    ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds_from_base(base_ds, batch_size=64, tta=False):
    ds = base_ds
    if tta:
        ds = ds.map(_apply_tta, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 3
BATCH_SIZE = 128
TTA_PASSES = 3

base_test_ds = make_base_test_ds()

n_test = len(df_test)
steps = int(np.ceil(n_test / BATCH_SIZE))

test_ds = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=False)
pred_sum = my_model.predict(test_ds, verbose=1)
pred_sum = np.asarray(pred_sum, dtype=np.float64)

tta_ds_once = make_test_ds_from_base(base_test_ds, batch_size=BATCH_SIZE, tta=True)
pred_tta = my_model.predict(tta_ds_once, verbose=0)
pred_tta = np.asarray(pred_tta, dtype=np.float64)
pred_sum += pred_tta * TTA_PASSES

pred_mean = pred_sum / (TTA_PASSES + 1)
pred_test_labels = np.argmax(pred_mean, axis=-1).astype(int)

final_csv = pd.DataFrame(
    {"image_id": df_test["image_id"].values, "label": pred_test_labels}
)

if len(final_csv) != len(sample_sub):
    raise ValueError(
        f"Submission length mismatch: got {len(final_csv)} rows, expected {len(sample_sub)}"
    )
if list(final_csv.columns) != ["image_id", "label"]:
    raise ValueError(f"Submission columns incorrect: {final_csv.columns.tolist()}")

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())




## === cell 4
sub = pd.read_csv("submission.csv")
assert sub.shape[0] == sample_sub.shape[0]
assert sub.columns.tolist() == ["image_id", "label"]
assert sub["image_id"].iloc[0] == sample_sub["image_id"].iloc[0]
print("submission.csv OK:", sub.shape)
print(sub["label"].value_counts().sort_index())
