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

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

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
try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras.preprocessing.image import img_to_array

    tf.random.set_seed(42)
    np.random.seed(42)
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = str(e)
    print("WARNING: TensorFlow import failed; will write fallback submission.")
    print("TensorFlow import error:", TF_IMPORT_ERROR)



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
        img = keras.preprocessing.image.load_img(img_path)
        img = img_to_array(img) / 255.0
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
