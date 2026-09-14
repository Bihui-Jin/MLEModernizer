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

0.0959504381988516

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow/TFHub load-time crash that prevents `sample_sub` from being created by removing the unused `tensorflow_hub` import (it’s triggering a protobuf incompatibility in this environment). Then I make model loading more robust by loading without compilation (inference-only) and by providing a safe fallback if any external model file path is missing, so the notebook always produces a valid `submission.csv`. Finally, I also keep the core ensemble logic intact but speed up and stabilize inference by batching test images instead of predicting one-by-one (score-neutral, just avoids timeouts and memory spikes).'
- What this solution (achieved 0.05531) has done: 'The protobuf-related crash happens before your pipeline even reads `sample_submission.csv`, so the first fix is to avoid importing TensorFlow at module import time and instead import it lazily after environment setup (this prevents the `MessageFactory.GetPrototype` error in this Kaggle runtime). I keep your ensemble logic, weights, and preprocessing the same, but add a robust fallback that still writes a valid `submission.csv` if TensorFlow cannot be imported or if models are missing. This should both unblock execution (so you always get a CSV) and, when TF loads successfully, restore real model inference (which should move accuracy upward from the “all zeros” fallback toward your target). I also keep batching as-is for stability and runtime.'
- What this solution (achieved 0.05531) has done: 'We need to fix the TensorFlow import crash (`MessageFactory` protobuf mismatch), because it forces the fallback “all zeros” submission and severely hurts accuracy. The smallest safe fix is to pin TensorFlow’s Python protobuf implementation at process start (before importing TF) by setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, which avoids the failing C++ protobuf path in this environment. I also make the image preprocessing more robust by forcing RGB conversion and using `tf.image.resize` on float tensors, without changing your model/ensemble logic. Everything else (model paths, weights, batching, argmax, submission format/path) stays the same.'
- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow import crash that forces the low-accuracy fallback by setting protobuf-related environment variables early and ensuring they take effect before any TensorFlow import. This is the smallest change that should restore real model inference (and thus move accuracy up toward your target) while preserving your ensemble logic, preprocessing, batching, and argmax submission semantics. I also add a safe “restartless” second attempt to import TensorFlow after setting variables, and keep the existing robust fallbacks so a valid `submission.csv` is always written. No model/training logic is changed—only import/runtime stability is addressed.'
- What this solution (achieved 0.05531) has done: 'We fix the TensorFlow import crash that currently forces the low-accuracy fallback by ensuring the protobuf “python” implementation env var is set *before any protobuf/TensorFlow-related import occurs*, and by defensively forcing the pure-Python protobuf module load before importing TensorFlow. This is the minimal change that should restore real model inference (and thus improve accuracy toward your target) while keeping your ensemble logic, weights, preprocessing, and argmax semantics unchanged. We also keep the robust fallbacks so a valid `submission.csv` is always written even if TF still can’t load. Finally, we keep batching and I/O paths identical for stability and runtime.'
- What this solution (achieved 0.05531) has done: 'We fix the root cause of the TensorFlow import crash (`MessageFactory.GetPrototype` AttributeError) that currently forces the low-accuracy “all zeros” fallback submission. The minimal robust fix is to ensure the pure-Python protobuf implementation is selected before any protobuf/TensorFlow import, and to explicitly import `google.protobuf` early after setting those env vars (so the setting actually takes effect in-process). This should allow TensorFlow to import successfully, enabling your existing ensemble inference path and improving accuracy toward the target without changing model logic, weights, preprocessing, or argmax semantics. We also keep the fallback behavior unchanged so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.05531) has done: 'We fix the TensorFlow import crash that currently triggers the all-zeros fallback (and thus the low score) by pinning protobuf to the pure-Python implementation *before any protobuf/TensorFlow modules load*, and by force-loading the Python protobuf backend early to avoid the `MessageFactory.GetPrototype` failure. This is a minimal, execution-unblocking change that preserves your ensemble logic, model files, preprocessing, batching, and argmax semantics, but should restore real model inference and move accuracy upward toward your target. We also add a safe second TensorFlow import attempt after setting env vars (in case something imported protobuf too early), while keeping the existing robust fallback so a valid `submission.csv` is always produced.'
- What this solution (achieved 0.05531) has done: 'We fix the TensorFlow import crash that forces the all-zeros fallback by ensuring the pure-Python protobuf backend is selected before TensorFlow (and any TF-dependent protobuf stubs) load, using only environment variables (no brittle internal protobuf APIs). We also add a safe monkey-patch fallback for the specific `MessageFactory.GetPrototype` AttributeError by aliasing it to `GetMessageClass` if protobuf still exposes only the newer method in this runtime. These changes are execution-unblocking and should restore your existing model-loading + ensemble inference path (thus improving accuracy toward the target) without changing your model architecture, preprocessing semantics, or argmax submission logic. Finally, we keep the same paths/output and always write a valid `submission.csv`.'
- What this solution (achieved 0.05531) has done: 'We fix the TensorFlow import crash that currently forces the low-accuracy all-zeros fallback by applying the protobuf `MessageFactory.GetPrototype` patch in the correct module (`google.protobuf.message_factory.MessageFactory`) and doing it *before* importing TensorFlow. This is a minimal, execution-unblocking change that preserves your ensemble inference logic, preprocessing, weights, and argmax submission semantics. Once TensorFlow imports successfully, your existing model-loading and batched prediction path run and should increase accuracy toward the target. We also keep the robust fallbacks so a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "3")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

try:
    import google.protobuf.message_factory as _mf  # noqa: F401

    if hasattr(_mf, "MessageFactory"):
        _cls = _mf.MessageFactory
        if not hasattr(_cls, "GetPrototype") and hasattr(_cls, "GetMessageClass"):
            _cls.GetPrototype = _cls.GetMessageClass  # type: ignore[attr-defined]
except Exception as _e:
    print("WARNING: protobuf patch not applied:", _e)

import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = "/kaggle/working/submission.csv"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(
    sample_sub.columns
), "sample_submission.csv must have image_id,label"

image_ids = sample_sub["image_id"].tolist()



## === cell 1
TF_AVAILABLE = True
TF_IMPORT_ERROR = None


def _try_import_tf():
    import tensorflow as tf  # noqa
    from tensorflow import keras  # noqa
    from tensorflow.keras.preprocessing.image import img_to_array  # noqa

    return tf, keras, img_to_array


try:
    tf, keras, img_to_array = _try_import_tf()
    tf.random.set_seed(42)
    np.random.seed(42)
except Exception as e1:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e1)
    print("WARNING: TensorFlow import failed:", TF_IMPORT_ERROR)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
norm_constant = 0.87 + 0.86 + 0.84
alpha_1 = 0.86 / norm_constant
alpha_2 = 0.87 / norm_constant
alpha_3 = 0.84 / norm_constant

MODEL1_PATH = "../input/resnet50-ver-1/ResNet50_ver_1.h5"
MODEL2_PATH = "../input/vgg19-ver-1/VGG19_ver_1_cp2.h5"
MODEL3_PATH = "../input/mobilenetv3large-ver-1/MobileNetV3Large_ver_1_1_cp5.h5"

loaded_models = []

if TF_AVAILABLE:

    def _safe_load_model(path):
        """
        Stability:
        - load_model(..., compile=False) for inference-only.
        - If missing/unloadable, skip.
        """
        if not tf.io.gfile.exists(path):
            print(f"WARNING: model file not found: {path}. This model will be skipped.")
            return None
        try:
            return tf.keras.models.load_model(path, compile=False)
        except Exception as e:
            print(
                f"WARNING: failed to load model at {path}: {e}. This model will be skipped."
            )
            return None

    model1 = _safe_load_model(MODEL1_PATH)
    model2 = _safe_load_model(MODEL2_PATH)
    model3 = _safe_load_model(MODEL3_PATH)

    loaded_models = [(model1, alpha_1), (model2, alpha_2), (model3, alpha_3)]
    loaded_models = [(m, w) for (m, w) in loaded_models if m is not None]

    if len(loaded_models) == 0:
        print(
            "WARNING: No models loaded. Will create a valid but low-accuracy submission with label=0 for all rows."
        )



## === cell 3
BATCH_SIZE = 16  # keep as in your current solution for stability


def write_submission(preds, out_path=OUT_PATH):
    assert len(preds) == len(
        sample_sub
    ), "Predictions length must match sample submission length."
    my_submission = pd.DataFrame(
        {"image_id": sample_sub["image_id"].values, "label": preds}
    )
    my_submission.to_csv(out_path, index=False)
    print(my_submission.head())
    print(f"Wrote submission to: {out_path} (rows={len(my_submission)})")


if (not TF_AVAILABLE) or (len(loaded_models) == 0):
    preds = [0] * len(image_ids)
    write_submission(preds, OUT_PATH)
else:

    def load_and_preprocess(image_id):
        img_path = os.path.join(TEST_IMG_DIR, image_id)
        img = keras.preprocessing.image.load_img(img_path, color_mode="rgb")
        img = img_to_array(img).astype(np.float32) / 255.0
        img = tf.image.resize(img, (512, 512))
        return img

    preds = []
    for start in range(0, len(image_ids), BATCH_SIZE):
        batch_ids = image_ids[start : start + BATCH_SIZE]
        batch_imgs = [load_and_preprocess(iid) for iid in batch_ids]
        batch = tf.stack(batch_imgs, axis=0)

        ensemble = None
        for m, w in loaded_models:
            p = m.predict(batch, verbose=0) * w
            ensemble = p if ensemble is None else (ensemble + p)

        batch_pred = np.argmax(ensemble, axis=1).astype(int).tolist()
        preds.extend(batch_pred)

    write_submission(preds, OUT_PATH)
