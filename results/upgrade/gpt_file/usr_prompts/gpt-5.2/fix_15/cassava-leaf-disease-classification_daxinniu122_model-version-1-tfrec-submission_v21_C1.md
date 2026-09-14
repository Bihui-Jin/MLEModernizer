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

0.8671804170444243

# 6. Current score

0.12818

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.42975) has done: 'The timeout is dominated by decoding/resizing each image three times and by running three separate `model.predict()` calls per batch, each with its own overhead and input pipeline synchronization. I keep the exact same ensemble logic/weights but refactor the input pipeline to decode once and branch into three resized tensors, ensure caching/prefetching is optimal, and replace per-batch `model.predict()` with direct `model(x, training=False)` to avoid Keras predict overhead while preserving identical inference semantics. I also force static shapes and keep everything in TensorFlow until the final `argmax`, minimizing host<->device transfers and Python overhead. File paths, models, weights, and output formatting remain unchanged.'
- What this solution (achieved 0.13901) has done: 'The crash happens before any modeling code runs due to an incompatibility between TensorFlow and the forced pure-Python protobuf implementation (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`), which triggers the `MessageFactory.GetPrototype` AttributeError. I remove that override (and instead force the faster default C++ protobuf) so TensorFlow can import cleanly in Kaggle’s environment. I keep the ensemble/inference logic identical, only adding a safe fallback to avoid downloading ImageNet weights (offline) by using `weights=None` if needed, ensuring the notebook always completes and writes `submission.csv`. These changes are primarily to unblock execution; once it runs, score depend on whether the external `../input/f-models/*.h5` are actually present.'
- What this solution (achieved 0.06016) has done: 'The timeout is dominated by expensive, repeated data handling in the TFRecord branch: it fully materializes all resized images into RAM and then rebuilds three datasets, adding large CPU/GPU transfer overhead before inference even starts. I keep the exact same ensemble logic/weights/models and prediction semantics, but stream TFRecords directly through the three models without ever storing the full test set in memory. I also avoid `model.predict()` overhead by using a single `@tf.function` inference step and preallocated NumPy output, while keeping determinism and identical argmax-on-weighted-sum behavior. Finally, I ensure the pipeline uses efficient `map/batch/prefetch` with `drop_remainder=False` and no redundant `.prefetch()` chains.'
- What this solution (achieved 0.12818) has done: 'We need to fix the TensorFlow import crash caused by an incompatible protobuf runtime (`MessageFactory.GetPrototype` missing). The minimal, Kaggle-safe fix is to force the pure-Python protobuf implementation *before* importing TensorFlow, which avoids the C++ protobuf / generated-code mismatch that triggers this specific AttributeError in some environments. I keep the rest of the ensemble/inference logic unchanged, only adjusting the environment variables and adding a small compatibility fallback for `tf.keras` vs standalone `keras` import to ensure the script runs end-to-end. This should also improve the score versus the current 0.06016 by ensuring the intended fine-tuned `.h5` models can actually be loaded and used instead of falling back to random-weight models.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

print("TF version:", tf.__version__)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train_images"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
TRAIN_TFREC_DIR = f"{DATA_DIR}/train_tfrecords"
TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)


def _infer_hw(m, fallback=512):
    try:
        shp = m.input_shape  # (None, H, W, 3)
        h = int(shp[1]) if shp is not None and shp[1] is not None else fallback
        w = int(shp[2]) if shp is not None and shp[2] is not None else fallback
        return h, w
    except Exception:
        return fallback, fallback


def _load_h5_if_exists(path):
    if tf.io.gfile.exists(path):
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(f"WARNING: Failed to load model at {path}: {type(e).__name__}: {e}")
            return None
    else:
        print(f"WARNING: Model file not found: {path}")
        return None


NUM_CLASSES = 5

F_MODELS_CANDIDATE_DIRS = [
    "../input/f-models",  # original
    "/kaggle/input/f-models",
    "/kaggle/input",
]

MODEL_FILES = {
    "m1": "ResNet50_f.h5",
    "m2": "VGG19_f.h5",
    "m3": "MobileNetV3L_f.h5",
}


def _find_model_path(fname: str):
    for d in F_MODELS_CANDIDATE_DIRS:
        p = os.path.join(d, fname)
        if tf.io.gfile.exists(p):
            return p
    try:
        for sub in tf.io.gfile.listdir("/kaggle/input"):
            p = os.path.join("/kaggle/input", sub, fname)
            if tf.io.gfile.exists(p):
                return p
    except Exception:
        pass
    return os.path.join(F_MODELS_CANDIDATE_DIRS[0], fname)


m1_path = _find_model_path(MODEL_FILES["m1"])
m2_path = _find_model_path(MODEL_FILES["m2"])
m3_path = _find_model_path(MODEL_FILES["m3"])

model1 = _load_h5_if_exists(m1_path)
model2 = _load_h5_if_exists(m2_path)
model3 = _load_h5_if_exists(m3_path)


def _safe_app_model(app_ctor, input_size, weights="imagenet"):
    inp = keras.Input(shape=(input_size, input_size, 3))
    try:
        base = app_ctor(
            include_top=False, weights=weights, input_tensor=inp, pooling="avg"
        )
    except Exception as e:
        print(
            f"WARNING: Could not load {app_ctor.__name__} with weights='{weights}': {type(e).__name__}: {e}"
        )
        base = app_ctor(
            include_top=False, weights=None, input_tensor=inp, pooling="avg"
        )
    x = base.output
    out = keras.layers.Dense(NUM_CLASSES, activation="softmax")(x)
    return keras.Model(inp, out)


if (model1 is None) or (model2 is None) or (model3 is None):
    print(
        "WARNING: One or more fine-tuned .h5 models are unavailable. "
        "Building fallback ImageNet models to produce a valid submission."
    )
    if model1 is None:
        model1 = _safe_app_model(keras.applications.ResNet50, 224, weights="imagenet")
    if model2 is None:
        model2 = _safe_app_model(keras.applications.VGG19, 224, weights="imagenet")
    if model3 is None:
        model3 = _safe_app_model(
            keras.applications.MobileNetV3Large, 224, weights="imagenet"
        )

norm_constant = 0.87 + 0.91 + 0.6
alpha_1 = 0.91 / norm_constant
alpha_2 = 0.87 / norm_constant
alpha_3 = 0.6 / norm_constant

H1, W1 = _infer_hw(model1, 512)
H2, W2 = _infer_hw(model2, 512)
H3, W3 = _infer_hw(model3, 512)

print("Resolved model paths:", m1_path, m2_path, m3_path)
print("Input sizes:", (H1, W1), (H2, W2), (H3, W3))
print("Test images dir exists:", tf.io.gfile.exists(TEST_IMG_DIR))
print("Sample submission rows:", len(sample_sub))

if not tf.io.gfile.exists(TEST_IMG_DIR):
    raise FileNotFoundError(f"TEST_IMG_DIR not found: {TEST_IMG_DIR}")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

BATCH_SIZE = 64
AUTOTUNE = tf.data.AUTOTUNE

image_ids = sample_sub["image_id"].astype(str).values
test_paths = np.array([os.path.join(TEST_IMG_DIR, i) for i in image_ids], dtype=object)

for p in test_paths[:5]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(f"Missing test image file: {p}")


@tf.function
def _decode_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.cast(img, tf.float32)
    img.set_shape([None, None, 3])
    return img


SAME_HW = (H1 == H2) and (H1 == H3) and (W1 == W2) and (W1 == W3)

if SAME_HW:

    @tf.function
    def _resize_three(img):
        img1 = tf.image.resize(img, (H1, W1))
        img1.set_shape([H1, W1, 3])
        return (img1, img1, img1)

else:

    @tf.function
    def _resize_three(img):
        img1 = tf.image.resize(img, (H1, W1))
        img2 = tf.image.resize(img, (H2, W2))
        img3 = tf.image.resize(img, (H3, W3))
        img1.set_shape([H1, W1, 3])
        img2.set_shape([H2, W2, 3])
        img3.set_shape([H3, W3, 3])
        return (img1, img2, img3)


def _list_tfrecords(dir_path):
    if not tf.io.gfile.exists(dir_path):
        return []
    files = tf.io.gfile.glob(os.path.join(dir_path, "*.tfrec"))
    return sorted(files)


test_tfrec_files = _list_tfrecords(TEST_TFREC_DIR)

alpha_1_tf = tf.constant(alpha_1, dtype=tf.float32)
alpha_2_tf = tf.constant(alpha_2, dtype=tf.float32)
alpha_3_tf = tf.constant(alpha_3, dtype=tf.float32)


@tf.function
def _ensemble_predict_step(b_img1, b_img2, b_img3):
    p1 = model1(b_img1, training=False) * alpha_1_tf
    p2 = model2(b_img2, training=False) * alpha_2_tf
    p3 = model3(b_img3, training=False) * alpha_3_tf
    p = p1 + p2 + p3
    return tf.argmax(p, axis=1, output_type=tf.int64)


if test_tfrec_files:
    feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }

    @tf.function
    def _parse_test(ex):
        x = tf.io.parse_single_example(ex, feature_description)
        img = tf.image.decode_jpeg(x["image"], channels=3)
        img = tf.cast(img, tf.float32)
        img.set_shape([None, None, 3])
        return x["image_name"], img

    ds = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=AUTOTUNE)
    ds = ds.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(
        lambda name, img: (name, _resize_three(img)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    n = len(image_ids)
    preds_out = np.empty((n,), dtype=np.int64)

    id_to_idx = {img_id: i for i, img_id in enumerate(image_ids)}

    filled = 0
    for b_names, b_imgs in ds:
        b_img1, b_img2, b_img3 = b_imgs
        b_preds = _ensemble_predict_step(b_img1, b_img2, b_img3).numpy()

        b_names_np = b_names.numpy()
        for j in range(b_names_np.shape[0]):
            name = b_names_np[j].decode("utf-8")
            preds_out[id_to_idx[name]] = int(b_preds[j])
        filled += int(b_names_np.shape[0])

    if filled != n:
        raise RuntimeError(
            f"TFRecord prediction count mismatch: got {filled}, expected {n}"
        )

    preds = preds_out.tolist()

else:
    ds = tf.data.Dataset.from_tensor_slices(test_paths)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_resize_three, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    n = len(image_ids)
    preds_out = np.empty((n,), dtype=np.int64)
    offset = 0
    for b_imgs in ds:
        b_img1, b_img2, b_img3 = b_imgs
        b_preds = _ensemble_predict_step(b_img1, b_img2, b_img3).numpy()
        bs = b_preds.shape[0]
        preds_out[offset : offset + bs] = b_preds
        offset += bs

    if offset != n:
        raise RuntimeError(
            f"Image-file prediction count mismatch: got {offset}, expected {n}"
        )

    preds = preds_out.tolist()

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].astype(str).values, "label": preds}
)
out_path = "/kaggle/working/submission.csv"
my_submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(my_submission.head())
print("Rows:", len(my_submission), "Cols:", list(my_submission.columns))
assert out_path.endswith(".csv")
assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(sample_sub)
assert my_submission["label"].between(0, 4).all()
