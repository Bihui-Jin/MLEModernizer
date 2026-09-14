# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.8851616802659413

# 6. Current score

0.5426

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.64387) has done: 'I fix the runtime failure that prevents the model from loading by switching protobuf to the pure-Python implementation *before* importing TensorFlow/tf_keras/keras, which addresses the `MessageFactory.GetPrototype` error in this environment. I also make the model path robust by falling back to running an in-notebook small CNN if the external `../input/my-model/...` file is not present, so the notebook always produces a valid `submission.csv`. To nudge accuracy upward (since you currently have no score at all), the fallback model train on the provided `train_images` with a simple stratified split and standard normalization, while preserving the original “predict then argmax” inference semantics. Finally, I ensure the submission format matches `sample_submission.csv` exactly and that the output file has a `.csv` suffix.'
- What this solution (achieved 0.63976) has done: 'I fix the protobuf/TensorFlow import ordering bug that’s still triggering `MessageFactory.GetPrototype` by moving the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting to the very first cell, before any TensorFlow/tf_keras (or anything that may transitively load protobuf) is imported. I also switch `BASE_DIR` to the correct Kaggle dataset mount (`/kaggle/input/...`) so the notebook can reliably find the images/CSVs in this environment. To improve the score toward your target with minimal core-logic disruption, I keep the same small CNN and training loop but train a bit longer (still a simple fit loop) and add lightweight in-model augmentation layers (doesn’t change inference semantics: still `predict -> argmax`). The script still always write a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.69096) has done: 'I fix the two runtime blockers that prevent any submission from being produced: (1) the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by ensuring protobuf uses the pure-Python implementation before TensorFlow is imported, and (2) the TFRecord parsing crash by replacing the nonexistent `tf.strings.decode` with correct UTF-8 decoding via `tf.io.decode_raw`/`tf.strings.unicode_encode`-free approach (here: `tf.strings.as_string`/`tf.io.decode_raw` isn’t appropriate; instead we treat `image_name` as a bytes string tensor and decode it later in Python, removing the decode op from the graph). These changes keep your model/training core logic intact (same CNN, same fit loop, same predict→argmax semantics) while making the pipeline run end-to-end. I also add a small safety check to ensure TFRecord feature keys match what’s actually present, so label attachment works reliably and the final `submission.csv` is always written with the exact required columns.'
- What this solution (achieved 0.58146) has done: 'I fix the pathing bug that causes all the `FileNotFoundError`s by auto-detecting the correct dataset root (your current `BASE_DIR` points to a non-existent nested folder). Then I keep your existing fallback-training + TFRecord inference logic intact, only making it robust to the two common TFRecord schemas for this competition (`image_name` vs `image_id`, and optional `target`). Finally, I ensure the pipeline always reaches the CSV write step and produces a valid `submission.csv` with exactly `image_id,label` aligned to `sample_submission.csv`.'
- What this solution (achieved 0.5725) has done: 'Your current score (0.58146) is far below the target (0.88516), so we should make small, low-risk changes that legitimately improve generalization without changing the overall approach (still TFRecord → normalize/resize → small CNN → fit → predict→argmax). The biggest gap is that the fallback CNN is being trained from scratch for only 12 epochs on a hard 5-way image task; a minimal but meaningful improvement is to (1) use a lightweight pretrained backbone (transfer learning) while keeping the same “single-model, single fit loop” semantics, and (2) fix a subtle label lookup mismatch risk by normalizing TFRecord `image_id` strings (some TFRecords store bare IDs without “.jpg”). These changes should move accuracy upward substantially toward your target while staying within Kaggle constraints and still producing a valid `submission.csv` end-to-end. I keep the same TFRecord input pipeline, normalization, augmentation, and argmax post-processing, only swapping the feature extractor inside the same model definition.'
- What this solution (achieved 0.51009) has done: 'Your current score (0.5725) is far below the target (0.8852), so we should make the smallest changes that legitimately improve generalization while keeping the same overall pipeline (TFRecords → normalize/resize → MobileNetV2 backbone → softmax → predict→argmax). The biggest low-risk gain here is to use the correct MobileNetV2 preprocessing (instead of mean/std normalization) and to feed the backbone the tensor after augmentation (right now augmentation is computed but not actually used). I keep the same model family, same single fit loop, same loss/metric, and same submission construction, only adjusting preprocessing and the one wiring bug so training/inference are aligned with the pretrained backbone. These changes are directly aimed at improving accuracy without changing the evaluation semantics or introducing extra training tricks.'
- What this solution (achieved 0.5426) has done: 'Your current score (0.51009) is far below the target (0.88516), so we should make a small, legitimate generalization improvement without changing the overall pipeline (TFRecords → resize/normalize → MobileNetV2 → softmax → argmax → submission). The biggest low-risk issue is that you keep the ImageNet backbone fully frozen; unfreezing only the top portion of MobileNetV2 for a short fine-tuning phase (same model, same loss, same fit loop) typically gives a large boost on this dataset. I also make sure the augmentation is only active during training (not during validation/test) by running the model normally (Keras handles this), and I keep preprocessing consistent with MobileNetV2. The rest of your TFRecord parsing and submission ordering stays identical to preserve evaluation semantics and minimize code changes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401

        pb_version = getattr(google.protobuf, "__version__", None)
    except Exception:
        pb_version = None

    need_install = False
    if pb_version is None:
        need_install = True
    else:
        try:
            major = int(str(pb_version).split(".")[0])
            if major >= 5:
                need_install = True
        except Exception:
            need_install = True

    if need_install:
        try:
            subprocess.check_call(
                [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
            )
        except Exception as e:
            print("Warning: could not pin protobuf<5; proceeding. Error:", repr(e))


_ensure_protobuf_compatible()

import json


def _pick_base_dir():
    candidates = [
        "/kaggle/input/cassava-leaf-disease-classification",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification",
        "/kaggle/data/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
    ]
    required = ["train.csv", "sample_submission.csv", "label_num_to_disease_map.json"]
    for c in candidates:
        if all(os.path.exists(os.path.join(c, r)) for r in required):
            return c
    for c in candidates:
        if os.path.isdir(c):
            for d in os.listdir(c):
                p = os.path.join(c, d)
                if os.path.isdir(p) and all(
                    os.path.exists(os.path.join(p, r)) for r in required
                ):
                    return p
    raise FileNotFoundError(
        "Could not find dataset root. Checked: "
        + ", ".join(candidates)
        + ". Ensure /kaggle/input is mounted with competition data."
    )


BASE_DIR = _pick_base_dir()
print("BASE_DIR:", BASE_DIR)
print("Exists?", os.path.exists(BASE_DIR))
print("train.csv exists?", os.path.exists(os.path.join(BASE_DIR, "train.csv")))
print("train_images exists?", os.path.exists(os.path.join(BASE_DIR, "train_images")))
print(
    "train_tfrecords exists?", os.path.exists(os.path.join(BASE_DIR, "train_tfrecords"))
)



## === cell 1
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.loads(file.read())
    map_classes = {int(k): v for k, v in map_classes.items()}

print(json.dumps(map_classes, indent=4))



## === cell 2
import pandas as pd
import cv2



## === cell 3
train_img_dir = os.path.join(BASE_DIR, "train_images")
if os.path.isdir(train_img_dir):
    input_files = os.listdir(train_img_dir)
    print(f"Number of train images: {len(input_files)}")
else:
    print(
        "train_images directory not found at:",
        train_img_dir,
        "(this is ok if using TFRecords only)",
    )



## === cell 4
img_shapes = {}
if os.path.isdir(train_img_dir):
    for image_name in os.listdir(train_img_dir)[:300]:
        image = cv2.imread(os.path.join(train_img_dir, image_name))
        if image is None:
            continue
        img_shapes[image.shape] = img_shapes.get(image.shape, 0) + 1
print(img_shapes)



## === cell 5
df_train = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
df_train["class_name"] = df_train["label"].map(map_classes)
df_train.head()



## === cell 6
df_train["image_id"] = df_train["image_id"].astype("str")
df_train["label"] = df_train["label"].astype("str")



## === cell 7
df_train["label"].value_counts()



## === cell 8
import numpy as np

import tensorflow as tf
import tf_keras

try:
    import keras  # Keras 3.x (optional)
except Exception as e:
    keras = None
    print(
        "Warning: keras (Keras 3) import failed; will rely on tf_keras. Error:", repr(e)
    )

print("tf version:", tf.__version__)
print("tf_keras version:", getattr(tf_keras, "__version__", "unknown"))

MODEL_PATH = "../input/my-model/Cassava_best_model.h5"
final_model = None
load_errors = []

if os.path.exists(MODEL_PATH):
    try:
        final_model = tf_keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        load_errors.append(("tf_keras.models.load_model", repr(e)))

    if final_model is None and keras is not None:
        try:
            final_model = keras.saving.load_model(MODEL_PATH, compile=False)
        except Exception as e:
            load_errors.append(("keras.saving.load_model", repr(e)))

    if final_model is None:
        print(
            "Warning: Model could not be loaded from external path; will train fallback model.\n"
            + "\n".join([f"- {src}: {err}" for src, err in load_errors])
        )
else:
    print("External model not found at:", MODEL_PATH, "-> will train fallback model.")



## === cell 9
try:
    if final_model is not None:
        try:
            final_model.summary()
        except Exception as e:
            print("Warning: final_model.summary() failed (non-fatal). Error:", repr(e))
except Exception:
    pass


def _infer_hw_from_model(m):
    for attr in ("input_shape", "inputs"):
        try:
            val = getattr(m, attr, None)
        except Exception:
            val = None

        if attr == "input_shape" and isinstance(val, tuple) and len(val) == 4:
            _, h, w, c = val
            return h, w, c

        if attr == "inputs" and val is not None:
            try:
                t = val[0] if isinstance(val, (list, tuple)) else val
                shp = tuple(getattr(t, "shape", ()))
                if len(shp) == 4:
                    _, h, w, c = shp
                    return h, w, c
            except Exception:
                pass

    return None, None, None


H, W, C = (None, None, None)
if final_model is not None:
    H, W, C = _infer_hw_from_model(final_model)

if H is None or W is None:
    H, W = 224, 224  # keep original fallback size
if C not in (1, 3, None):
    print(f"Warning: unexpected channel dimension C={C}; will feed RGB (3 channels).")
    C = 3

print("Using inference input size H,W,C =", H, W, C)



## === cell 10
from sklearn.model_selection import train_test_split

if final_model is None:
    SEED = 42
    tf.random.set_seed(SEED)
    np.random.seed(SEED)

    df = pd.read_csv(os.path.join(BASE_DIR, "train.csv"))
    df["image_id"] = df["image_id"].astype(str)
    df["label"] = df["label"].astype(int)

    counts = df["label"].value_counts().sort_index()
    n_classes = 5
    total = counts.sum()
    class_weight = {
        i: float(total / (n_classes * counts.get(i, 1))) for i in range(n_classes)
    }
    print("class_weight:", class_weight)

    train_df, val_df = train_test_split(
        df,
        test_size=0.1,
        random_state=SEED,
        stratify=df["label"],
    )

    train_ids = tf.constant(train_df["image_id"].tolist(), dtype=tf.string)
    train_labels = tf.constant(train_df["label"].tolist(), dtype=tf.int64)
    val_ids = tf.constant(val_df["image_id"].tolist(), dtype=tf.string)
    val_labels = tf.constant(val_df["label"].tolist(), dtype=tf.int64)

    train_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(train_ids, train_labels),
        default_value=tf.constant(-1, dtype=tf.int64),
    )
    val_table = tf.lookup.StaticHashTable(
        tf.lookup.KeyValueTensorInitializer(val_ids, val_labels),
        default_value=tf.constant(-1, dtype=tf.int64),
    )

    tfrec_train_dir = os.path.join(BASE_DIR, "train_tfrecords")
    tfrec_files = sorted(
        [
            os.path.join(tfrec_train_dir, f)
            for f in os.listdir(tfrec_train_dir)
            if f.endswith(".tfrec")
        ]
    )
    if len(tfrec_files) == 0:
        raise FileNotFoundError(f"No .tfrec files found in {tfrec_train_dir}")

    train_feature_description = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
        "target": tf.io.FixedLenFeature([], tf.int64, default_value=-1),
    }

    def _normalize(img):
        img = tf.cast(img, tf.float32)
        return tf_keras.applications.mobilenet_v2.preprocess_input(img)

    def _parse_example(example_proto):
        ex = tf.io.parse_single_example(example_proto, train_feature_description)
        img = tf.image.decode_jpeg(ex["image"], channels=3)
        img = tf.image.resize(img, [int(H), int(W)], method="bilinear")
        img = _normalize(img)

        image_id = ex["image_name"]
        image_id = tf.where(
            tf.equal(tf.strings.length(image_id), 0), ex["image_id"], image_id
        )
        image_id = tf.strings.strip(image_id)
        image_id = tf.where(
            tf.strings.regex_full_match(image_id, r".*\.jpg$"),
            image_id,
            tf.strings.join([image_id, ".jpg"]),
        )
        return img, image_id

    def _attach_label(image, image_id, table):
        lbl = table.lookup(image_id)
        return image, lbl

    def _filter_labeled(image, label):
        return tf.not_equal(label, tf.constant(-1, dtype=tf.int64))

    def _make_ds_from_tfrecords(table, training):
        ds = tf.data.TFRecordDataset(tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
        ds = ds.map(_parse_example, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.map(
            lambda img, iid: _attach_label(img, iid, table),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds = ds.filter(_filter_labeled)
        if training:
            ds = ds.shuffle(8192, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(32).prefetch(tf.data.AUTOTUNE)
        return ds

    ds_train = _make_ds_from_tfrecords(train_table, training=True)
    ds_val = _make_ds_from_tfrecords(val_table, training=False)

    inputs = tf_keras.layers.Input(shape=(int(H), int(W), 3))
    x = tf_keras.layers.RandomFlip("horizontal")(inputs)
    x = tf_keras.layers.RandomRotation(0.05)(x)
    x = tf_keras.layers.RandomZoom(0.1)(x)

    backbone = tf_keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(int(H), int(W), 3),
        pooling=None,
    )

    backbone.trainable = False

    x = backbone(x, training=False)
    x = tf_keras.layers.GlobalAveragePooling2D()(x)
    x = tf_keras.layers.Dropout(0.2)(x)
    outputs = tf_keras.layers.Dense(5, activation="softmax")(x)

    final_model = tf_keras.Model(inputs, outputs)

    final_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    final_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=8,
        verbose=1,
        class_weight=class_weight,
    )

    backbone.trainable = True
    for layer in backbone.layers:
        if isinstance(layer, tf_keras.layers.BatchNormalization):
            layer.trainable = False
    for layer in backbone.layers[:-30]:
        layer.trainable = False

    final_model.compile(
        optimizer=tf_keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf_keras.losses.SparseCategoricalCrossentropy(),
        metrics=["accuracy"],
    )

    final_model.fit(
        ds_train,
        validation_data=ds_val,
        epochs=4,
        verbose=1,
        class_weight=class_weight,
    )



## === cell 11
sub = pd.read_csv(os.path.join(BASE_DIR, "sample_submission.csv"))
test_images = sub["image_id"].astype(str).tolist()

tfrec_test_dir = os.path.join(BASE_DIR, "test_tfrecords")
test_tfrec_files = sorted(
    [
        os.path.join(tfrec_test_dir, f)
        for f in os.listdir(tfrec_test_dir)
        if f.endswith(".tfrec")
    ]
)
if len(test_tfrec_files) == 0:
    raise FileNotFoundError(f"No .tfrec files found in {tfrec_test_dir}")

test_feature_description = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string, default_value=b""),
    "image_id": tf.io.FixedLenFeature([], tf.string, default_value=b""),
}


def _normalize(img):
    img = tf.cast(img, tf.float32)
    return tf_keras.applications.mobilenet_v2.preprocess_input(img)


def _parse_test_example(example_proto):
    ex = tf.io.parse_single_example(example_proto, test_feature_description)
    img = tf.image.decode_jpeg(ex["image"], channels=3)
    img = tf.image.resize(img, [int(H), int(W)], method="bilinear")
    img = _normalize(img)

    image_id = ex["image_name"]
    image_id = tf.where(
        tf.equal(tf.strings.length(image_id), 0), ex["image_id"], image_id
    )
    image_id = tf.strings.strip(image_id)
    image_id = tf.where(
        tf.strings.regex_full_match(image_id, r".*\.jpg$"),
        image_id,
        tf.strings.join([image_id, ".jpg"]),
    )
    return img, image_id


ds_test = tf.data.TFRecordDataset(test_tfrec_files, num_parallel_reads=tf.data.AUTOTUNE)
ds_test = ds_test.map(_parse_test_example, num_parallel_calls=tf.data.AUTOTUNE)
ds_test = ds_test.batch(32).prefetch(tf.data.AUTOTUNE)

all_ids = []
all_preds = []

for batch_imgs, batch_ids in ds_test:
    probs = final_model.predict(batch_imgs, verbose=0)
    probs = np.asarray(probs)
    pred = np.argmax(probs, axis=-1).astype(int)
    all_preds.append(pred)

    for b in batch_ids.numpy().tolist():
        all_ids.append(
            b.decode("utf-8") if isinstance(b, (bytes, bytearray)) else str(b)
        )

predictions = np.concatenate(all_preds).astype(int).tolist()

print("n_test_images (sample_submission):", len(test_images))
print("n_test_from_tfrecords:", len(all_ids))
print("n_predictions:", len(predictions))

pred_map = {iid: p for iid, p in zip(all_ids, predictions)}

missing = [iid for iid in test_images if iid not in pred_map]
if len(missing) > 0:
    print(
        f"Warning: {len(missing)} test ids missing from tfrecords parsing; filling with 0. Example:",
        missing[:5],
    )
ordered_preds = [int(pred_map.get(iid, 0)) for iid in test_images]

if len(ordered_preds) != len(test_images):
    raise ValueError(
        f"Prediction length mismatch: {len(ordered_preds)} vs {len(test_images)}"
    )

submission = pd.DataFrame({"image_id": test_images, "label": ordered_preds})
submission["image_id"] = submission["image_id"].astype(str)
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
