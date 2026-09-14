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

0.8652160773647628

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.11584) has done: 'I fix the model-loading crash by switching from standalone `keras` to `tensorflow.keras` and forcing protobuf to use the pure-Python implementation, which avoids the `MessageFactory.GetPrototype` error seen in many Kaggle images. I also fix the preprocessing bug (PIL image not converted to RGB/float and missing EfficientNet-style normalization), which currently makes predictions unreliable and can change tensor shapes. Finally, I ensure predictions are produced for every test image in a deterministic order that matches `sample_submission.csv`, so `image_id` and `label` lengths always match and the submission format is valid (`submission.csv`). These changes keep the core “load pretrained model → predict → argmax → write submission” logic intact while making it run end-to-end and improving expected accuracy.'
- What this solution (achieved 0.05531) has done: 'I fix the model-loading crash by avoiding `load_model()` (which triggers the protobuf `MessageFactory.GetPrototype` issue in this environment) and instead rebuilding the same EfficientNetB3 classifier in code, then loading weights from the provided `.hdf5`. I keep the core logic identical (pretrained EfficientNetB3 → predict on test → argmax → write submission) and keep your preprocessing, only switching it to the official `efficientnet.preprocess_input` to match training normalization and improve accuracy toward the target. I also make the file/path handling more robust (fallback paths, existence checks) while keeping the same Kaggle input locations and producing a valid `submission.csv` with the exact required columns.'
- What this solution (achieved 0.05531) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash that prevents TensorFlow/Keras from importing/initializing by forcing the pure-Python protobuf implementation *before* any TensorFlow import, and by clearing any already-imported `google.protobuf` modules to ensure the setting takes effect in this notebook runtime. This should make the model build + `load_weights()` run so the rest of your existing inference logic works unchanged and produces `submission.csv`. I also keep your EfficientNetB3 architecture and preprocessing intact, only adding a small, safe fallback for `preprocess_input` import to avoid version-specific import errors. No training logic or modeling approach is changed—this is purely to unblock execution and restore expected accuracy.'
- What this solution (achieved 0.05531) has done: 'We fix the root-cause crash (`MessageFactory.GetPrototype`) by forcing the pure-Python protobuf implementation *before* any TensorFlow import and restarting protobuf-related modules so the setting actually takes effect in this runtime. Then we keep your core inference logic unchanged (rebuild EfficientNetB3 → load provided weights → preprocess → predict → argmax → write `submission.csv`). I also add a small compatibility fallback for the EfficientNet `preprocess_input` import and guard against missing model files so the notebook always reaches CSV creation. These changes should both unblock execution and bring the score back up toward the target by ensuring the model and preprocessing match what the weights expect.'
- What this solution (achieved 0.05531) has done: 'We fix the root cause of the `MessageFactory.GetPrototype` crash by ensuring protobuf is forced to the pure-Python implementation *before* TensorFlow is imported, and by clearing both `google.protobuf` and `tensorflow` from `sys.modules` so the setting actually takes effect in this runtime. This unblocks the TensorFlow/Keras import, model construction, and `load_weights()` so `model` is defined and downstream inference runs. We keep your core logic intact (EfficientNetB3 → load provided weights → preprocess with EfficientNet `preprocess_input` → predict → argmax → write `submission.csv`). The only other changes are small robustness guards (explicitly disable GPU to avoid slow/fragile CUDA init in Kaggle CPU images, and a clearer check for the weights path) without changing evaluation semantics.'
- What this solution (achieved 0.05531) has done: 'You’re still hitting the protobuf `MessageFactory.GetPrototype` crash during the TensorFlow import/build, so the main fix is to make the “force pure-Python protobuf” setting actually take effect before TensorFlow initializes by also setting `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` at the very start and clearing any already-imported protobuf modules. Then, to ensure the notebook always produces a valid `submission.csv`, I keep your exact model/preprocess/predict→argmax pipeline but add a safe fallback path: if TensorFlow cannot be imported in this environment, it write a deterministic baseline submission (valid format) instead of crashing (this is score-worse but guarantees “Not yielded” becomes a valid submission). No training logic, architecture, loss, or evaluation semantics are changed when TensorFlow successfully loads; the fallback only activates if the environment cannot run TF due to protobuf issues.'

# 9. Code solution

## === cell 0
import os
import sys
import importlib

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")

for m in list(sys.modules.keys()):
    if (
        m.startswith("google.protobuf")
        or m.startswith("protobuf")
        or m.startswith("tensorflow")
        or m.startswith("keras")
    ):
        del sys.modules[m]

import numpy as np
import pandas as pd



## === cell 1
BASE_DIR = "../input/cassava-leaf-disease-classification"
TEST_DIR = f"{BASE_DIR}/test_images/"
SAMPLE_SUB_PATH = f"{BASE_DIR}/sample_submission.csv"

IMG_SIZE = 300
size = (IMG_SIZE, IMG_SIZE)

if not os.path.exists(SAMPLE_SUB_PATH):
    alt_base = "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    if os.path.exists(os.path.join(alt_base, "sample_submission.csv")):
        BASE_DIR = alt_base
        TEST_DIR = f"{BASE_DIR}/test_images/"
        SAMPLE_SUB_PATH = f"{BASE_DIR}/sample_submission.csv"

assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found at {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_DIR), f"test_images dir not found at {TEST_DIR}"



## === cell 2
TF_AVAILABLE = False
TF_IMPORT_ERROR = None

try:
    import tensorflow as tf

    TF_AVAILABLE = True
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = e

if TF_AVAILABLE:
    tf.random.set_seed(123)
    np.random.seed(123)

    from tensorflow.keras import layers, Model
    from tensorflow.keras.applications import EfficientNetB3

    eff_preprocess = None
    _preprocess_errors = []

    for mod_path in [
        "tensorflow.keras.applications.efficientnet",  # v1 (EfficientNetB0-B7)
        "tensorflow.keras.applications.efficientnet_v2",  # fallback only
    ]:
        try:
            eff_preprocess = importlib.import_module(mod_path).preprocess_input
            break
        except Exception as e:
            _preprocess_errors.append((mod_path, repr(e)))

    if eff_preprocess is None:
        raise ImportError(
            "Could not import an EfficientNet preprocess_input. Errors: "
            + "; ".join([f"{m}: {e}" for m, e in _preprocess_errors])
        )

    MODEL_PATH = "../input/effficientnetb3-cassava/best_model.hdf5"
    if not os.path.exists(MODEL_PATH):
        for alt in [
            "../input/efficientnetb3-cassava/best_model.hdf5",
            "../input/effficientnetb3-cassava/best_model.h5",
            "../input/efficientnetb3-cassava/best_model.h5",
        ]:
            if os.path.exists(alt):
                MODEL_PATH = alt
                break

    assert os.path.exists(MODEL_PATH), f"Model weights file not found: {MODEL_PATH}"

    def build_model(img_size=300, num_classes=5):
        inp = layers.Input(shape=(img_size, img_size, 3))
        base = EfficientNetB3(
            include_top=False, weights="imagenet", input_tensor=inp, pooling="avg"
        )
        x = layers.Dropout(0.3)(base.output)
        out = layers.Dense(num_classes, activation="softmax")(x)
        return Model(inp, out)

    model = build_model(IMG_SIZE, 5)
    model.load_weights(MODEL_PATH)



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
if "model" in globals():
    model.summary()



## === cell 4
from PIL import Image


def preprocess_pil(img: Image.Image) -> np.ndarray:
    """
    Keep the same basic flow (PIL -> RGB -> resize -> float32 -> batch dimension),
    but use the official EfficientNet preprocess_input (matches common training pipelines).
    """
    img = img.convert("RGB").resize(size)
    arr = np.asarray(img, dtype=np.float32)  # [0,255]
    arr = eff_preprocess(arr)  # EfficientNet normalization
    arr = np.expand_dims(arr, axis=0)  # (1, H, W, C)
    return arr




## === cell 5
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image_id"].tolist()

missing = [
    img_id
    for img_id in test_images
    if not os.path.exists(os.path.join(TEST_DIR, img_id))
]
if missing:
    raise FileNotFoundError(f"Missing {len(missing)} test images, e.g. {missing[:5]}")

if TF_AVAILABLE and "model" in globals():
    preds = np.empty(len(test_images), dtype=np.int64)

    for i, image_id in enumerate(test_images):
        img_path = os.path.join(TEST_DIR, image_id)
        with Image.open(img_path) as img:
            x = preprocess_pil(img)
        p = model.predict(x, verbose=0)
        preds[i] = int(np.argmax(p, axis=1)[0])
else:
    err_str = (
        repr(TF_IMPORT_ERROR)
        if TF_IMPORT_ERROR is not None
        else "Unknown TF/protobuf issue"
    )
    print(
        "WARNING: TensorFlow/model unavailable; writing baseline submission. Error:",
        err_str,
    )
    preds = np.zeros(len(test_images), dtype=np.int64)



## === cell 6
sub = pd.DataFrame({"image_id": test_images, "label": preds})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Saved submission.csv with", len(sub), "rows")
print("submission.csv columns:", list(sub.columns))
print("Unique predicted labels:", np.unique(preds))
