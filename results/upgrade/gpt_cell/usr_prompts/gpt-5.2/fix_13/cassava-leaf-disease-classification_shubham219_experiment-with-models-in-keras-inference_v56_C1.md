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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.794

# 6. Current score

0.06577

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10688) has done: 'Diagnosis: The crash happens because the loaded EfficientNetB0 model expects inputs of shape (224, 224, 3), but the test generator in `make_test_gen()` is producing images resized to (512, 512, 3). This mismatch triggers Keras’ input compatibility check during `my_model.predict()`. The core inference logic is fine; only the generator’s `target_size` needs to match the model’s expected spatial input size.

Patch summary: In cell 4, create the test generator with a `target_size` derived from `my_model.input_shape` (falling back safely to (224, 224) if unavailable). This keeps the same prediction loop and submission construction, while ensuring the batch images match the model’s required dimensions.

Updated cells: cell 4 only.

Compatibility notes for cell k+1: `final_csv` is still created with the same columns and written to `submission.csv`, so cell 5 (`final_csv.head()`) remains compatible.

Assumptions: `my_model.input_shape` is a standard Keras shape tuple like `(None, H, W, 3)` for this image model; if not, the fallback `(224, 224)` is correct for EfficientNetB0 and avoids the crash.'
- What this solution (achieved 0.11622) has done: 'Your current score is far below target, so we should improve accuracy rather than tweak calibration. The biggest issue is that the inference pipeline does not apply the same EfficientNet-specific preprocessing the model was trained with, which typically destroys performance; we add the correct `preprocess_input` via `ImageDataGenerator(preprocessing_function=...)` while keeping the same generator/predict/argmax submission logic. We also ensure the input `target_size` matches the loaded model and that `df_test` is not modified in-place when building the submission (to avoid any accidental leakage/column reuse). These are minimal, metric-aligned changes that should move accuracy substantially toward the 0.794 target without changing the model or training approach.'
- What this solution (achieved 0.08969) has done: 'Your score is still far below the target, so we should improve correctness rather than tune anything. The biggest remaining issue is likely that the loaded model file is an **EfficientNetB3-based model**, but your preprocessing currently uses **EfficientNet (B0) preprocess_input**, which can severely hurt accuracy even though the code runs. I change only the preprocessing function selection to match the loaded model (B3 vs B0) while keeping the same generator/predict/argmax submission logic. I also make the model-file selection more deterministic (prefer the exact expected filename, otherwise prefer any `.h5` that contains “b3”) to reduce the chance you load an unintended checkpoint.'
- What this solution (achieved 0.32623) has done: 'Your score is far below the target, so we should increase accuracy by fixing a likely label–image mismatch and ensuring we’re loading the intended checkpoint consistently. The biggest correctness issue is that your submission is built from a globbed file list, which may not match the exact ordering/content expected by `sample_submission.csv`; instead we should drive inference using `sample_submission.csv`’s `image_id` and map each to its file path, preserving the required row order. We keep your model, generator-based prediction, preprocessing selection, and argmax logic unchanged, only changing how `df_test` is constructed/aligned and making model-file selection prefer the exact expected filename. These minimal changes are directly tied to improving evaluation accuracy while still producing a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'Your current score (0.32623) is far below the target (0.794), so we should focus on fixing likely inference-time correctness issues with minimal changes. The most probable remaining problem is that the loaded `.h5` model uses custom `efficientnet.tfkeras` layers and/or expects `efficientnet.tfkeras.preprocess_input`, but the current preprocessing selection can silently pick the wrong Keras `preprocess_input`, badly hurting accuracy. I (1) load the model with the full `efficientnet.tfkeras` custom objects (when available) to avoid layer mismatches, and (2) force preprocessing to use `efficientnet.tfkeras.preprocess_input` whenever the model file suggests it was trained with that package, while keeping the same generator→predict→argmax submission logic. I also keep the sample_submission-driven ordering you already added so the submission rows align with Kaggle’s expected order.'
- What this solution (achieved 0.26084) has done: 'Your current score (0.61099) is still well below the target (0.794), so we should make a minimal, correctness-focused inference change that typically yields a meaningful accuracy gain without changing model/training core logic. The most likely remaining issue is that the test generator is not using the exact same “center-crop vs resize” geometry the model was trained with; many Cassava EfficientNet checkpoints were trained with a slightly larger resize followed by a center crop to the model’s input size. I keep your generator→predict→argmax pipeline identical, but change only the inference preprocessing to optionally do “resize to (input+32) then center-crop back to input” before `preprocess_input`, while falling back safely to plain resize if that fails. This should move accuracy upward toward the target while remaining within Kaggle constraints and still producing a valid `submission.csv`.'
- What this solution (achieved 0.06577) has done: 'Your score dropped after adding the resize+center-crop preprocessing, which is a common mismatch if the checkpoint was trained with plain resize. To move accuracy back up toward the target with minimal change, I keep your exact generator→predict→argmax→CSV pipeline but remove the optional resize+crop and use only the selected EfficientNet `preprocess_input` on the resized images. I also stop filtering out missing test paths (the sample submission order is the ground truth required order) and instead assert that all expected test files exist so row alignment stays correct. These are small, inference-only correctness fixes that should increase score from 0.26084 toward 0.794.'

# 9. Code solution

## === cell 0
import sys

if "tensorflow" not in sys.modules:
    import subprocess

    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import glob
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from tensorflow import keras

SEED = 42
DEBUG = False



## === cell 1
import os, subprocess, sys

try:
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "../input/kerasapplication/Keras_Applications-1.0.8-py3-none-any.whl",
        ]
    )
    subprocess.check_call(
        [
            sys.executable,
            "-m",
            "pip",
            "install",
            "-q",
            "../input/efficientnet/efficientnet-1.1.1-py3-none-any.whl",
        ]
    )
except Exception:
    pass



## === cell 2
try:
    from efficientnet.tfkeras import EfficientNetB0  # type: ignore
    import efficientnet.tfkeras as efn  # type: ignore

    _HAS_EFN = True
except ModuleNotFoundError:
    from tensorflow.keras.applications import EfficientNetB0

    efn = None
    _HAS_EFN = False

import os
import glob

expected_fname = "model_v0.38.h5"
candidate_paths = []

base_dirs = [
    "../input",
    "/kaggle/input",
    "/kaggle/data/input",
    "/kaggle/data/kaggle/data/input",
    "/kaggle/data/kaggle/data/cassava-leaf-disease-classification",
    "/kaggle/data/cassava-leaf-disease-classification",
]

for base in base_dirs:
    if os.path.isdir(base):
        candidate_paths.extend(
            glob.glob(os.path.join(base, "**", expected_fname), recursive=True)
        )

if not candidate_paths:
    all_h5 = []
    for base in base_dirs:
        if os.path.isdir(base):
            all_h5.extend(glob.glob(os.path.join(base, "**", "*.h5"), recursive=True))
    v038_like = [p for p in all_h5 if "v0.38" in os.path.basename(p).lower()]
    if v038_like:
        candidate_paths = sorted(v038_like)
    else:
        b3_like = [p for p in all_h5 if "b3" in os.path.basename(p).lower()]
        candidate_paths = sorted(b3_like) if b3_like else sorted(all_h5)

if candidate_paths:
    weight_path = candidate_paths[0]

    custom_objects = {}
    if _HAS_EFN:
        custom_objects.update(getattr(efn, "__dict__", {}))
    custom_objects["EfficientNetB0"] = EfficientNetB0

    my_model = load_model(weight_path, custom_objects=custom_objects)
else:
    weight_path = None
    my_model = EfficientNetB0(weights=None, include_top=True, classes=5)



## === cell 3
import os

test_dir = "../input/cassava-leaf-disease-classification/test_images"
sample_path = "../input/cassava-leaf-disease-classification/sample_submission.csv"

sample_sub = pd.read_csv(sample_path)
df_test = sample_sub[["image_id"]].copy()
df_test["path"] = df_test["image_id"].apply(lambda x: os.path.join(test_dir, x))

missing = df_test.loc[~df_test["path"].apply(os.path.exists), "image_id"].tolist()
if missing:
    raise FileNotFoundError(
        f"Missing {len(missing)} test images (first 5): {missing[:5]}"
    )


def _select_preprocess_fn(model, weight_path=None):
    wp = os.path.basename(weight_path).lower() if weight_path else ""
    mn = (getattr(model, "name", "") or "").lower()

    if _HAS_EFN:
        return efn.preprocess_input

    from tensorflow.keras.applications.efficientnet import preprocess_input as fn

    return fn


eff_preprocess_input = _select_preprocess_fn(my_model, weight_path=weight_path)

try:
    _h, _w = my_model.input_shape[1], my_model.input_shape[2]
    _target_size = (
        (int(_h), int(_w)) if _h is not None and _w is not None else (224, 224)
    )
except Exception:
    _target_size = (224, 224)


def make_test_gen(batch_size=64, target_size=_target_size):
    my_test_idg = ImageDataGenerator(preprocessing_function=eff_preprocess_input)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=target_size,
    )
    return test_gen


pred_list = []
for i in range(1):
    test_gen = make_test_gen(batch_size=128)
    pred_test = my_model.predict(test_gen, verbose=True)
    pred_list.append(pred_test.tolist())

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1)

final_submission = df_test[["image_id"]].copy()
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)
final_csv.head()



## === cell 4
final_csv.head()
