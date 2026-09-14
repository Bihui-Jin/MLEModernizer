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

0.684950135992747

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'The timeout is dominated by repeated disk JPEG decoding and expensive CPU-side augmentations done 5 times via `ImageDataGenerator`, plus extra overhead from using the pure-Python protobuf implementation. I keep the exact same TTA semantics (5 augmented prediction passes, same augment parameters, same input size/model) but switch inference to an equivalent `tf.data` pipeline that caches decoded/resized images once and applies the same random transforms inside TensorFlow for each pass, using parallel map/prefetch to maximize throughput. I also remove the protobuf “python” fallback (which is much slower) and avoid per-row pandas `apply` in favor of vectorized basename extraction. These changes are performance-only and preserve the algorithmic logic and evaluation behavior (negligible FP differences only).'
- What this solution (achieved 0.11099) has done: 'I fix the TensorFlow import crash by removing the forced protobuf “cpp” implementation environment variables, which are incompatible with the available `google.protobuf` build in this runtime and prevent any code from executing. Then I keep your exact inference core logic (same model loading, same 5-pass TTA averaging, same resize and augmentation semantics) but make `tf` available again so later cells run. Finally, I ensure the script always writes `submission.csv` with the required columns and row count checks, so you reliably get a valid Kaggle submission file end-to-end.'
- What this solution (achieved 0.11024) has done: 'The crash happens before any modeling code runs because TensorFlow’s protobuf dependency is incompatible with the protobuf version in this Kaggle runtime, triggering `MessageFactory.GetPrototype` errors on import. The minimal robust fix is to force protobuf to use the pure-Python implementation *before* importing TensorFlow, which avoids the failing C++ binding path in this environment. The rest of your pipeline (model load, 5-pass TTA, tf.data caching, submission writing) is kept identical, so the score should recover toward your target by allowing the intended model inference to actually execute. I also add a tiny safety fallback to ensure `label` is an `int` column in the final CSV.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    "../data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
]

DATA_ROOT = None
for p in CANDIDATE_INPUT_ROOTS:
    if os.path.isdir(p):
        DATA_ROOT = p
        break

if DATA_ROOT is None:
    raise FileNotFoundError(
        "Could not find cassava-leaf-disease-classification dataset directory in expected locations."
    )

TEST_GLOB = os.path.join(DATA_ROOT, "test_images", "*.jpg")
test_images = sorted(glob.glob(TEST_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found with glob: {TEST_GLOB}")

df_test = pd.DataFrame({"path": test_images})
print("DATA_ROOT:", DATA_ROOT)
print("Num test images:", len(df_test))



## === cell 2
CANDIDATE_MODEL_PATHS = [
    "/kaggle/input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "../input/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "/kaggle/data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
    "../data/experiment-with-models-using-keras-with-updates/model_v0.25.h5",
]

weight_path = None
for p in CANDIDATE_MODEL_PATHS:
    if os.path.exists(p):
        weight_path = p
        break

custom_objects = {}
try:
    custom_objects["swish"] = tf.nn.swish
except Exception:
    pass


def build_fallback_model(input_shape=(512, 512, 3), num_classes=5):
    inputs = tf.keras.Input(shape=input_shape)
    x = tf.keras.layers.Rescaling(1.0 / 255.0)(inputs)
    x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPool2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="softmax")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


if weight_path is not None:
    print("Loading model from:", weight_path)
    my_model = load_model(weight_path, compile=False, custom_objects=custom_objects)
else:
    print(
        "WARNING: model_v0.25.h5 not found in expected locations; "
        "using a fallback untrained CNN to allow submission generation."
    )
    my_model = build_fallback_model()



## === cell 3
AUTO = tf.data.AUTOTUNE
TARGET_SIZE = (512, 512)

_ROTATION_RANGE_DEG = 90.0
_BRIGHTNESS_RANGE = (0.2, 0.4)
_HFLIP = True
_VFLIP = True

_seed_base = tf.constant([SEED, 0], dtype=tf.int32)

paths_np = df_test["path"].to_numpy()


def _decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, TARGET_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _rotate_compat(img, angle_rad):
    if hasattr(tf.image, "rotate"):
        try:
            return tf.image.rotate(img, angles=angle_rad, interpolation="BILINEAR")
        except Exception:
            return tf.image.rotate(
                img, angles=tf.reshape(angle_rad, [1]), interpolation="BILINEAR"
            )
    else:
        return img


def _augment(img, seed):
    factor = tf.random.stateless_uniform(
        [], seed=seed, minval=_BRIGHTNESS_RANGE[0], maxval=_BRIGHTNESS_RANGE[1]
    )
    img = img * factor

    seed1 = seed + tf.constant([1, 0], tf.int32)
    if _HFLIP:
        do = tf.random.stateless_uniform([], seed=seed1) < 0.5
        img = tf.cond(do, lambda: tf.image.flip_left_right(img), lambda: img)

    seed2 = seed + tf.constant([2, 0], tf.int32)
    if _VFLIP:
        do = tf.random.stateless_uniform([], seed=seed2) < 0.5
        img = tf.cond(do, lambda: tf.image.flip_up_down(img), lambda: img)

    seed3 = seed + tf.constant([3, 0], tf.int32)
    angle = tf.random.stateless_uniform(
        [], seed=seed3, minval=-_ROTATION_RANGE_DEG, maxval=_ROTATION_RANGE_DEG
    )
    angle_rad = angle * (np.pi / 180.0)

    if hasattr(tf.image, "rotate"):
        img = _rotate_compat(img, angle_rad)
    else:
        k = tf.random.stateless_uniform(
            [], seed=seed3, minval=0, maxval=4, dtype=tf.int32
        )
        img = tf.image.rot90(img, k=k)

    return img


base_ds = tf.data.Dataset.from_tensor_slices(paths_np)
base_ds = base_ds.map(_decode_resize, num_parallel_calls=AUTO)
base_ds = base_ds.cache()
base_ds = base_ds.prefetch(AUTO)


def make_test_ds_tta(batch_size=128, tta_pass=0):
    def add_index(img, idx):
        seed = (
            _seed_base
            + tf.constant([tta_pass, 0], tf.int32)
            + tf.cast(tf.stack([0, idx]), tf.int32)
        )
        img = _augment(img, seed)
        return img

    ds = tf.data.Dataset.zip((base_ds, tf.data.Dataset.range(len(paths_np))))
    ds = ds.map(add_index, num_parallel_calls=AUTO)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTO)
    return ds


pred_list = []
for t in range(5):
    test_ds = make_test_ds_tta(batch_size=128, tta_pass=t)
    pred = my_model.predict(test_ds, verbose=1)
    pred_list.append(pred)

pred_test = np.mean(np.stack(pred_list, axis=0), axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = (
    pd.Series(paths_np).str.rsplit(os.sep, n=1).str[-1].to_numpy()
)
final_submission["label"] = pred_test_labels.astype(np.int64)

final_csv = final_submission[["image_id", "label"]]

if final_csv.shape[0] != len(test_images):
    raise RuntimeError("Submission row count does not match number of test images.")
if list(final_csv.columns) != ["image_id", "label"]:
    raise RuntimeError(
        "Submission columns are incorrect; expected ['image_id','label']."
    )

final_csv.to_csv("submission.csv", index=False)



## === cell 4
print(final_csv.head())
print(f"\nWrote submission.csv with {len(final_csv)} rows.")
print("submission.csv exists:", os.path.exists("submission.csv"))
print(
    "submission.csv size (bytes):",
    os.path.getsize("submission.csv") if os.path.exists("submission.csv") else None,
)
