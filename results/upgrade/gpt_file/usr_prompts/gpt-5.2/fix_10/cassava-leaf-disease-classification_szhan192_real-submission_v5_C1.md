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

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.1013901480809912

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the protobuf/TensorFlow load-time crash that prevents `load_model()` from working by forcing protobuf’s pure-Python implementation early (this addresses the `MessageFactory.GetPrototype` error in this environment). Then I make execution robust so downstream cells don’t fail when an earlier cell errors, and I ensure the script always writes a valid `submission.csv` with the required `image_id,label` columns. I also add a safe fallback path: if the external model file is missing or still can’t be loaded, the code generate a valid baseline submission (majority-class labels) so you always “yield” a Kaggle submission. These changes preserve the core inference logic when the model loads successfully and are focused on unblocking end-to-end execution.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before* TensorFlow (or anything that imports protobuf via TF) is imported; in your current notebook, `import tensorflow as tf` happens in the next cell, so the env var change must be in the same cell and placed first. I also make the model-loading cell robust: if the load still fails, we proceed to a valid baseline submission without raising. Since your current score (0.61099) is far above the target (0.10139) and higher-is-better, I not add any score-improving changes; I keep the baseline fallback label fixed as you already had so behavior stays stable.'
- What this solution (achieved 0.61099) has done: 'I fix the protobuf/TensorFlow import crash by moving the protobuf environment-variable forcing to a cell that runs *before* any TensorFlow/Keras import (your current order still allows TF to touch protobuf before the env var takes effect in this environment). I also keep your existing inference logic and fallback behavior intact, only making the minimal edits needed so the notebook runs end-to-end and always writes a valid `submission.csv`. Since your current score (0.61099) is already far above the target (0.10139) and higher-is-better, I won’t add any score-improving changes—this is strictly a stability/runtime fix.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *before any protobuf/TensorFlow import occurs* (your current cell order allows TF to initialize protobuf before the env var takes effect). I also add a safe fallback to ensure the pipeline always completes and writes a valid `submission.csv` even if the model can’t be loaded. Since your current score (0.61099) is far above the target (0.10139) and higher-is-better, I not make any score-improving changes; the only score-affecting behavior is the existing fallback, which remains a simple constant-label baseline. The core inference logic (load model → per-image predict → argmax) is preserved exactly when the model loads successfully.'
- What this solution (achieved 0.61099) has done: 'I fix the protobuf/TensorFlow crash by forcing protobuf’s pure-Python implementation *before* any TensorFlow import happens (your current ordering still allows TF to hit the broken protobuf path). I also add a small compatibility fallback that patches `MessageFactory.GetPrototype` to `GetMessageClass` when the protobuf version lacks it, which is the specific attribute error you’re seeing. These changes are purely runtime/stability fixes; the model inference logic and submission formatting remain the same. Since your current score (0.61099) is already far above the target (0.10139) and higher-is-better, I not make any score-improving changes.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory  # noqa: F401

    if not hasattr(_message_factory.MessageFactory, "GetPrototype") and hasattr(
        _message_factory.MessageFactory, "GetMessageClass"
    ):
        _message_factory.MessageFactory.GetPrototype = (
            _message_factory.MessageFactory.GetMessageClass
        )
except Exception:
    pass

import numpy as np
import pandas as pd



## === cell 1
import tensorflow as tf
from tensorflow import keras

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
SAMPLE_SUB_PATH = f"{DATA_DIR}/sample_submission.csv"
TEST_IMG_DIR = f"{DATA_DIR}/test_images"
OUT_PATH = "/kaggle/working/submission.csv"

CANDIDATE_MODEL_PATHS = [
    "/kaggle/input/densenet22/densenet_22.h5",
    "/kaggle/input/densenet22/densenet_22.keras",
    "/kaggle/input/densenet-22/densenet_22.h5",
]

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
if "image_id" not in sample_sub.columns:
    raise ValueError("sample_submission.csv must contain image_id column")

MODEL_PATH = None
for p in CANDIDATE_MODEL_PATHS:
    if tf.io.gfile.exists(p):
        MODEL_PATH = p
        break

model = None
model_load_error = None

if MODEL_PATH is not None:
    try:
        model = keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        model_load_error = e
        model = None

print("TensorFlow:", tf.__version__)
print("Found model path:", MODEL_PATH)
if model is None and MODEL_PATH is not None:
    print("WARNING: Model failed to load; will fall back to baseline submission.")
    print("Model load error:", repr(model_load_error))



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
preds = []
TARGET_SIZE = 256

if not tf.io.gfile.exists(TEST_IMG_DIR):
    raise FileNotFoundError(f"TEST_IMG_DIR not found: {TEST_IMG_DIR}")

if model is not None:
    for image_id in sample_sub["image_id"].values:
        img_path = os.path.join(TEST_IMG_DIR, image_id)
        if not tf.io.gfile.exists(img_path):
            raise FileNotFoundError(f"Missing test image: {img_path}")

        img = keras.preprocessing.image.load_img(
            img_path, target_size=(TARGET_SIZE, TARGET_SIZE)
        )
        x = keras.preprocessing.image.img_to_array(img)
        x = np.expand_dims(x, axis=0).astype(np.float32) / 255.0

        prediction = model.predict(x, verbose=0)
        preds.append(int(np.argmax(prediction, axis=1)[0]))
else:
    fallback_label = 3
    preds = [fallback_label] * len(sample_sub)

if len(preds) != len(sample_sub):
    raise RuntimeError(
        f"Prediction length mismatch: preds={len(preds)} vs sample_sub={len(sample_sub)}"
    )

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].values, "label": preds}
)
my_submission.to_csv(OUT_PATH, index=False)

print("Loaded model:", MODEL_PATH if model is not None else None)
print("Wrote:", OUT_PATH)
print(my_submission.head())
print("Rows:", len(my_submission), "Cols:", list(my_submission.columns))
print("Unique labels:", sorted(my_submission["label"].unique().tolist()))
