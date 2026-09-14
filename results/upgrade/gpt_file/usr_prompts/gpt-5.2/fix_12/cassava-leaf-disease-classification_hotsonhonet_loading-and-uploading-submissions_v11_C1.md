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

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG = "../input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "../input/cassavaeffentb7models/content/Models"

import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    import cv2
except Exception:
    cv2 = None

try:
    from matplotlib import pyplot as plt
except Exception:
    plt = None

try:
    _cpu = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu))
    tf.config.threading.set_inter_op_parallelism_threads(min(2, _cpu))
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")




## === cell 1
MODEL_PATH = "../input/ensemble-resnet50-effb0-effb4/resnet50_b0_b4.h5"

NUM_CLASSES = 5
IMG_SIZE = (600, 600)

model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded from:", MODEL_PATH)
else:
    base = keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    x = base.output
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)

    try:
        if os.path.isdir(MODELS_WEIGHTS):
            cand = []
            for f in sorted(os.listdir(MODELS_WEIGHTS)):
                if f.endswith((".h5", ".weights.h5")):
                    cand.append(os.path.join(MODELS_WEIGHTS, f))
            if len(cand) > 0:
                try:
                    model.load_weights(cand[0])
                    print("Loaded fallback weights from:", cand[0])
                except Exception as e:
                    print(
                        "Found weights but could not load (continuing without):",
                        str(e)[:200],
                    )
    except Exception as e:
        print("Weights scan failed (continuing without):", str(e)[:200])

    print("Fallback EfficientNetB7 model built (ImageNet weights).")

print("Model Loading Complete")




## === cell 2
print("Skipping plot_model to save time under 600s timeout.")




## === cell 3
print("Skipping image preview to save time under 600s timeout.")




## === cell 4
ss = pd.read_csv(SAMPLE_CSV)

model_name = (getattr(model, "name", "") or "").lower()
if "efficientnet" in model_name:
    preprocess_fn = keras.applications.efficientnet.preprocess_input
elif "resnet" in model_name:
    preprocess_fn = keras.applications.resnet.preprocess_input
else:
    preprocess_fn = keras.applications.efficientnet.preprocess_input

AUTOTUNE = tf.data.AUTOTUNE
batch_size = 64

test_tfrecord_dir = "../input/cassava-leaf-disease-classification/test_tfrecords"
tfrec_files = []
if os.path.isdir(test_tfrecord_dir):
    tfrec_files = sorted(
        [
            os.path.join(test_tfrecord_dir, f)
            for f in os.listdir(test_tfrecord_dir)
            if f.endswith((".tfrec", ".tfrecord", ".tfrecords"))
        ]
    )

use_tfrecords = len(tfrec_files) > 0
if not use_tfrecords:
    test_dir = "../input/cassava-leaf-disease-classification/test_images"
    images = ss["image_id"].tolist()
    paths = [os.path.join(test_dir, x) for x in images]

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}
_FEATURES_WITH_TARGET = dict(_FEATURES)
_FEATURES_WITH_TARGET["target"] = tf.io.FixedLenFeature([], tf.int64, default_value=-1)


@tf.function
def _decode_resize_preprocess_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # RGB
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_fn(img)
    return img


def _parse_tfrec_image_and_name(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES_WITH_TARGET)
    img = _decode_resize_preprocess_from_bytes(ex["image"])
    name = ex["image_name"]
    return img, name


def _load_decode_resize_preprocess_from_path(path):
    img_bytes = tf.io.read_file(path)
    img = _decode_resize_preprocess_from_bytes(img_bytes)
    return img


options = tf.data.Options()
options.experimental_deterministic = False
try:
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_slack = True
except Exception:
    pass

if use_tfrecords:
    ds_all = tf.data.TFRecordDataset(
        tfrec_files, num_parallel_reads=AUTOTUNE
    ).with_options(options)
    ds_all = ds_all.map(_parse_tfrec_image_and_name, num_parallel_calls=AUTOTUNE)
    ds_all = ds_all.batch(batch_size, drop_remainder=False)
    ds_all = ds_all.prefetch(AUTOTUNE)

    ds_x = ds_all.map(lambda x, n: x, num_parallel_calls=AUTOTUNE)
    ds_n = ds_all.map(lambda x, n: n, num_parallel_calls=AUTOTUNE)

    probs = model.predict(ds_x, verbose=0)
    names = tf.concat([nb for nb in ds_n], axis=0).numpy()
    names = [n.decode("utf-8") for n in names.tolist()]

    pred_df = pd.DataFrame(
        {"image_id": names, "label": np.argmax(probs, axis=1).astype(int)}
    )
    pred_df = pred_df.sort_values("image_id", kind="mergesort")

    my_submission = ss[["image_id"]].merge(pred_df, on="image_id", how="left")
    if my_submission["label"].isna().any():
        my_submission["label"] = my_submission["label"].fillna(0).astype(int)
    else:
        my_submission["label"] = my_submission["label"].astype(int)

else:
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(_load_decode_resize_preprocess_from_path, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    prob = model.predict(ds, verbose=0)
    preds = np.argmax(prob, axis=1).astype(int)
    my_submission = pd.DataFrame({"image_id": ss["image_id"].values, "label": preds})

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)
print(my_submission.head())




## === cell 5
my_submission.head()
