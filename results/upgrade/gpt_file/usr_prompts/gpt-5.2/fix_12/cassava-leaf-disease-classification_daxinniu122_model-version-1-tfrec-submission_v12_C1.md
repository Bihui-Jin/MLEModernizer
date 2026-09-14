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

0.8541855545482019

# 6. Current score

0.61136

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'Main runtime is being wasted by (1) per-batch `model.predict()` overhead inside a Python loop, (2) slow JPEG decode/resize in the input pipeline without caching/deterministic options, and (3) Python-side byte-decoding/list appends. The core logic (same TFRecord parsing, same preprocessing, same ResNet50 model, same argmax predictions, same merge with sample_submission) is preserved, but inference is executed as a single graph `model(test_ds)` pass with `tf.data` optimizations (deterministic, map fusion, caching) and vectorized ID decoding at the end. These changes reduce Python overhead and maximize pipeline throughput while keeping outputs equivalent up to negligible floating-point differences. File paths and submission semantics remain unchanged.'
- What this solution (achieved 0.60762) has done: 'The timeout is dominated by decoding/resizing 2676 images to 512×512 and running ResNet50 inference, plus an extra full dataset pass just to re-collect image ids. I remove the redundant second pass by having the dataset yield deterministic integer indices and then map predictions back to `sample_submission.image_id` directly (same ordering as TFRecords), which preserves evaluation semantics while cutting runtime almost in half. I also switch dataset caching from in-memory (`cache()`) to no-cache (for a single pass it only adds overhead/memory pressure) and enable standard TF data optimizations while keeping deterministic ordering and identical preprocessing/model behavior. Model architecture/inference remains unchanged; only the input pipeline and post-processing are optimized.'
- What this solution (achieved 0.61136) has done: 'We fix two hard runtime blockers: the protobuf/TF incompatibility causing `MessageFactory.GetPrototype` errors and the missing `tf.image.decode_and_resize_jpeg` API (it doesn’t exist in this TF build). To keep core inference logic the same, we replace it with the standard `tf.io.decode_jpeg` + `tf.image.resize` path while preserving dtype/range and `resnet50.preprocess_input`. Since TFRecord order already matches `sample_submission.csv`, we keep deterministic ordering but remove the second dataset pass for IDs and instead directly reuse `sample_sub.image_id`, which also avoids the `test_ds` NameError cascade. Finally, we ensure the script always writes `/kaggle/working/submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
OUT_PATH = "/kaggle/working/submission.csv"

TEST_TFREC_DIR = f"{DATA_DIR}/test_tfrecords"
MODEL_PATH = "/kaggle/input/resnet50-ver-1/ResNet50_ver_1.h5"

IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # inference batch size

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Sample submission path exists:", os.path.exists(SAMPLE_SUB_PATH))
print("Test tfrecord dir exists:", os.path.isdir(TEST_TFREC_DIR))
print("Model path exists:", os.path.exists(MODEL_PATH))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "Unexpected sample_submission columns"

_FEATURES = {
    "image": tf.io.FixedLenFeature([], tf.string),
    "image_name": tf.io.FixedLenFeature([], tf.string),
}

preprocess_input = tf.keras.applications.resnet50.preprocess_input


def _decode_and_preprocess(example_proto):
    ex = tf.io.parse_single_example(example_proto, _FEATURES)
    img = tf.io.decode_jpeg(ex["image"], channels=3)  # uint8 [0..255]
    img = tf.image.resize(img, IMG_SIZE, method="bilinear")  # float32 by default
    img = tf.cast(img, tf.float32)  # ensure float32
    img = preprocess_input(img)  # ResNet50 preprocess expects float32 in RGB 0..255
    image_id = ex["image_name"]
    return img, image_id


def make_test_dataset(tfrecord_paths, batch_size):
    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.map_and_batch_fusion = True

    ds = tf.data.TFRecordDataset(
        tfrecord_paths,
        num_parallel_reads=tf.data.AUTOTUNE,
        buffer_size=64 * 1024 * 1024,
    ).with_options(options)

    ds = ds.map(_decode_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


test_tfrecords = sorted(glob.glob(os.path.join(TEST_TFREC_DIR, "*.tfrec")))
assert len(test_tfrecords) > 0, f"No TFRecords found in {TEST_TFREC_DIR}"

test_ds = make_test_dataset(test_tfrecords, BATCH_SIZE)

print("Found TFRecords:", len(test_tfrecords))
print("Sample submission rows:", len(sample_sub))




## === cell 2
def build_resnet50_head(input_shape=(512, 512, 3), num_classes=5):
    base = tf.keras.applications.ResNet50(
        include_top=False,
        weights=None,
        input_shape=(input_shape[0], input_shape[1], input_shape[2]),
        pooling="avg",
    )
    x = keras.layers.Dropout(0.2)(base.output)
    out = keras.layers.Dense(num_classes, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)
    return model


def load_resnet50_model_safely(model_path: str) -> tf.keras.Model:
    if model_path and os.path.exists(model_path):
        try:
            m = tf.keras.models.load_model(model_path, compile=False)
            print("Loaded model via tf.keras.models.load_model:", model_path)
            return m
        except Exception as e:
            print("load_model failed, attempting weights-only load. Error:", repr(e))
            m = build_resnet50_head(
                input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=5
            )
            m.load_weights(model_path)
            print("Loaded weights into fallback architecture:", model_path)
            return m
    else:
        print(
            f"WARNING: model file not found at {model_path}. Using untrained fallback model so pipeline completes."
        )
        return build_resnet50_head(
            input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3), num_classes=5
        )


model = load_resnet50_model_safely(MODEL_PATH)



## === cell 3
num_test = len(sample_sub)
steps = (num_test + BATCH_SIZE - 1) // BATCH_SIZE

pred_probs = model.predict(test_ds, steps=steps, verbose=0)
pred_probs = np.asarray(pred_probs)[:num_test]
preds = np.argmax(pred_probs, axis=1).astype(np.int32)

id_list = sample_sub["image_id"].astype(str).tolist()
assert len(preds) == num_test, "Prediction count must match sample_submission rows"
assert len(id_list) == num_test, "ID count must match sample_submission rows"

sub_df = pd.DataFrame({"image_id": id_list, "label": preds.astype(int)})

assert len(sub_df) == len(sample_sub), "Submission length mismatch vs sample_submission"
assert sub_df["label"].between(0, 4).all(), "Predicted labels must be in [0,4]"

sub_df.to_csv(OUT_PATH, index=False)

print("Wrote:", OUT_PATH)
print(sub_df.head())
print("Rows:", len(sub_df))
print("Label distribution:\n", sub_df["label"].value_counts().sort_index())
