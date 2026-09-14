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

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)
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
def _ensemble_predict(b_img1, b_img2, b_img3):
    p1 = model1(b_img1, training=False) * alpha_1_tf
    p2 = model2(b_img2, training=False) * alpha_2_tf
    p3 = model3(b_img3, training=False) * alpha_3_tf
    p = p1 + p2 + p3
    return tf.argmax(p, axis=1, output_type=tf.int64)


def _predict_probs_fast(model, ds_images, steps: int):
    return model.predict(ds_images, steps=steps, verbose=0)


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

    all_names = []
    all_imgs1 = []
    all_imgs2 = []
    all_imgs3 = []
    for b_names, b_imgs in ds:
        b_img1, b_img2, b_img3 = b_imgs
        all_names.append(b_names.numpy())
        all_imgs1.append(b_img1)
        all_imgs2.append(b_img2)
        all_imgs3.append(b_img3)

    all_names = np.concatenate(all_names, axis=0).astype("S")

    ds1 = (
        tf.data.Dataset.from_tensor_slices(tf.concat(all_imgs1, axis=0))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    ds2 = (
        tf.data.Dataset.from_tensor_slices(tf.concat(all_imgs2, axis=0))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )
    ds3 = (
        tf.data.Dataset.from_tensor_slices(tf.concat(all_imgs3, axis=0))
        .batch(BATCH_SIZE)
        .prefetch(AUTOTUNE)
    )

    n = all_names.shape[0]
    steps = (n + BATCH_SIZE - 1) // BATCH_SIZE

    p1 = _predict_probs_fast(model1, ds1, steps=steps) * alpha_1
    p2 = _predict_probs_fast(model2, ds2, steps=steps) * alpha_2
    p3 = _predict_probs_fast(model3, ds3, steps=steps) * alpha_3
    all_preds = np.argmax(p1 + p2 + p3, axis=1).astype(np.int64)

    name_to_pred = {n.decode("utf-8"): int(p) for n, p in zip(all_names, all_preds)}
    preds = [name_to_pred[i] for i in image_ids]
else:
    ds = tf.data.Dataset.from_tensor_slices(test_paths)
    ds = ds.map(_decode_image, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(_resize_three, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

    ds1 = ds.map(
        lambda a, b, c: a, num_parallel_calls=AUTOTUNE, deterministic=True
    ).prefetch(AUTOTUNE)
    ds2 = ds.map(
        lambda a, b, c: b, num_parallel_calls=AUTOTUNE, deterministic=True
    ).prefetch(AUTOTUNE)
    ds3 = ds.map(
        lambda a, b, c: c, num_parallel_calls=AUTOTUNE, deterministic=True
    ).prefetch(AUTOTUNE)

    n = len(image_ids)
    steps = (n + BATCH_SIZE - 1) // BATCH_SIZE

    p1 = _predict_probs_fast(model1, ds1, steps=steps) * alpha_1
    p2 = _predict_probs_fast(model2, ds2, steps=steps) * alpha_2
    p3 = _predict_probs_fast(model3, ds3, steps=steps) * alpha_3
    preds = np.argmax(p1 + p2 + p3, axis=1).astype(np.int64).tolist()

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
