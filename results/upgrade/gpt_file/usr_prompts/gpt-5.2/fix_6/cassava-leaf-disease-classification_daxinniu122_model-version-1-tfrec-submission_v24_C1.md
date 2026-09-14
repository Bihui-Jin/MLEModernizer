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

0.8637050468419462

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime crash caused by importing `tensorflow_hub` (it triggers a protobuf incompatibility in this Kaggle environment), since your script doesn’t actually use TF-Hub. Then I make model loading robust by using `compile=False` and providing a safe `custom_objects` dict so `.h5` models saved with TF-Hub/Keras layers can still be deserialized. Finally, I ensure `sample_sub` is always defined (so cell 29 can run) and add a fallback that creates a valid `submission.csv` even if the external model files aren’t present, so you always get a valid CSV output end-to-end.'
- What this solution (achieved 0.05531) has done: 'We fix the immediate runtime crash by avoiding TensorFlow import-time protobuf issues: set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing `tensorflow` and remove any `tensorflow_hub` dependency (your code doesn’t need it for inference). Then we correct the biggest scoring bug: the models were trained with their own preprocessing, but inference currently feeds raw 0–255 RGB, which makes predictions near-random; we apply the appropriate `tf.keras.applications.*.preprocess_input` for each model before calling it (same ensemble logic, just correct inputs). Finally, we make image loading robust (RGB conversion + file existence guard) and still always write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'The crash happens before any modeling because TensorFlow import triggers a protobuf API mismatch (`MessageFactory.GetPrototype`) in this environment. The minimal robust fix is to ensure the runtime uses a compatible protobuf implementation by forcing the pure-Python protobuf backend and (critically) using TensorFlow only after that environment variable is set; additionally, we avoid importing TensorFlow at module import time if it fails, so the notebook can still produce a valid submission. Since your current score is extremely low versus the target, the biggest legitimate accuracy fix is to keep the same ensemble logic but ensure inference preprocessing matches the models’ expected inputs (ResNet50/VGG19 preprocess + correct image size), and to fail loudly if models are missing instead of silently predicting a constant label for most images. The patch below makes TensorFlow import stable, loads the two `.h5` models, applies the correct preprocessing, and always writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05531) has done: 'You’re crashing on TensorFlow import due to an old protobuf runtime in this environment; the minimal reliable fix is to force the pure-Python protobuf backend *and* ensure we import `protobuf` before `tensorflow`, plus clear conflicting protobuf env vars that can trigger the C++ implementation. Once TF imports, we keep your exact ensemble logic, but make the image tensors explicitly `float32` and use `np.expand_dims` instead of `tf.reshape` to avoid any shape/type edge cases and keep preprocessing consistent. Finally, we keep the fallback path but make it explicit and always write `/kaggle/working/submission.csv` with the required columns and row order from `sample_submission.csv`, so you always get a valid submission.'
- What this solution (achieved 0.05531) has done: 'We fix the root runtime issue preventing TensorFlow from importing by upgrading the `protobuf` package to a TensorFlow-compatible version *inside the notebook session* before importing TF (this is the direct cause of the `MessageFactory.GetPrototype` error). Then we keep your exact ensemble/inference logic the same, but remove the now-unnecessary protobuf “python backend” forcing that can actually keep you on the problematic codepath. Finally, we make the model paths and data paths robust (without changing semantics) and still always write `/kaggle/working/submission.csv` with the required columns and row order.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import numpy as np
import pandas as pd

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")
OUT_PATH = "/kaggle/working/submission.csv"

print("Python:", sys.version)
print("DATA_DIR exists:", os.path.exists(DATA_DIR))
print("TEST_IMG_DIR exists:", os.path.exists(TEST_IMG_DIR))
print("SAMPLE_SUB_PATH exists:", os.path.exists(SAMPLE_SUB_PATH))




## === cell 1
def _pip_install(pkg: str):
    print(f"Installing: {pkg}")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", pkg])


_pip_install("protobuf==3.20.3")

import google.protobuf  # noqa: F401
import google.protobuf.__version__ as _pbver  # type: ignore

print("protobuf version:", google.protobuf.__version__)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/2008861956.py in <cell line: 0>()
     11 
     12 import google.protobuf  # noqa: F401
---> 13 import google.protobuf.__version__ as _pbver  # type: ignore
     14 
     15 print("protobuf version:", google.protobuf.__version__)

ModuleNotFoundError: No module named 'google.protobuf.__version__'

## === cell 2
tf = None
keras = None
img_to_array = None

try:
    import tensorflow as tf  # noqa: E402
    from tensorflow import keras  # noqa: E402
    from tensorflow.keras.preprocessing.image import img_to_array  # noqa: E402

    tf.random.set_seed(42)
    np.random.seed(42)
    print("TF version:", tf.__version__)
except Exception as e:
    print(
        "WARNING: TensorFlow failed to import; will fall back to constant predictions."
    )
    print(f"TF import error: {type(e).__name__}: {e}")



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
assert {"image_id", "label"}.issubset(sample_sub.columns)
print("sample_sub shape:", sample_sub.shape)
print(sample_sub.head())




## === cell 4
def safe_load_model(path):
    if tf is None:
        return None
    if not os.path.exists(path):
        print(f"WARNING: Model file not found: {path}")
        return None
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        print(f"WARNING: Failed to load model at {path}: {type(e).__name__}: {e}")
        return None


MODEL_DIR_CANDIDATES = [
    "../input/f-models",
    "/kaggle/input/f-models",
]
MODEL_DIR = None
for d in MODEL_DIR_CANDIDATES:
    if os.path.exists(d):
        MODEL_DIR = d
        break

if MODEL_DIR is None:
    MODEL_DIR = MODEL_DIR_CANDIDATES[0]

model1 = safe_load_model(os.path.join(MODEL_DIR, "ResNet50_f.h5"))
model2 = safe_load_model(os.path.join(MODEL_DIR, "VGG19_f.h5"))
model3 = safe_load_model(
    os.path.join(MODEL_DIR, "MobileNetV3L_f.h5")
)  # kept for parity (unused)

norm_constant = 0.87 + 0.91
alpha_1 = 0.91 / norm_constant
alpha_2 = 0.87 / norm_constant

print(
    "Loaded models:",
    {
        "model1": model1 is not None,
        "model2": model2 is not None,
        "model3": model3 is not None,
    },
)
print("alphas:", alpha_1, alpha_2)



## === cell 5
if tf is not None:
    from tensorflow.keras.applications.resnet50 import (
        preprocess_input as resnet50_preprocess,
    )
    from tensorflow.keras.applications.vgg19 import preprocess_input as vgg19_preprocess

preds = []
fallback_label = 0

target_size = (224, 224)

missing_or_error = 0
used_fallback = 0

for image in sample_sub.image_id.values:
    img_path = os.path.join(TEST_IMG_DIR, image)

    if (tf is None) or (model1 is None) or (model2 is None):
        preds.append(fallback_label)
        used_fallback += 1
        continue

    try:
        if not os.path.exists(img_path):
            missing_or_error += 1
            preds.append(fallback_label)
            used_fallback += 1
            continue

        pil_img = keras.preprocessing.image.load_img(
            img_path, color_mode="rgb", target_size=target_size
        )
        img = img_to_array(pil_img).astype(np.float32)  # range 0..255
        img = np.expand_dims(img, axis=0)  # (1, 224, 224, 3)

        x1 = resnet50_preprocess(img.copy())
        x2 = vgg19_preprocess(img.copy())

        prediction1 = model1.predict(x1, verbose=0) * alpha_1
        prediction2 = model2.predict(x2, verbose=0) * alpha_2
        prediction = (
            prediction1 + prediction2
        )  # + prediction3 (kept commented as original)

        preds.append(int(np.argmax(prediction, axis=1)[0]))
    except Exception as e:
        missing_or_error += 1
        preds.append(fallback_label)
        used_fallback += 1

my_submission = pd.DataFrame(
    {"image_id": sample_sub.image_id.values, "label": np.asarray(preds, dtype=int)}
)

my_submission["image_id"] = my_submission["image_id"].astype(str)
my_submission["label"] = my_submission["label"].astype(int)
my_submission.to_csv(OUT_PATH, index=False)

print(
    f"Wrote submission: {OUT_PATH} shape={my_submission.shape} errors={missing_or_error} fallback_rows={used_fallback}"
)
print(my_submission.head())
print(
    "label value counts:\n",
    my_submission["label"].value_counts(dropna=False).sort_index(),
)
