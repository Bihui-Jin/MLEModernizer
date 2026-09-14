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

3.13

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

0.78649138712602

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the runtime crash that happens before `prediction` is created by addressing the `MessageFactory.GetPrototype` protobuf incompatibility that prevents TensorFlow/Keras model loading in this environment. I add a safe, local workaround by forcing the pure-Python protobuf implementation early (before importing TensorFlow), which avoids that specific AttributeError in many Kaggle images. I also make submission-writing robust by guaranteeing `prediction` is always defined and has the correct length even if model loading still fails, so a valid `submission.csv` is always produced. These changes are minimal and do not alter your preprocessing or inference logic when the model loads successfully.'
- What this solution (achieved 0.05531) has done: 'We fix the protobuf/TensorFlow import crash by forcing the pure-Python protobuf implementation *before any protobuf/tensorflow-related import*, and we additionally guard against this environment variable being set too late by re-execing the process once if needed. Then we correct the model-path handling so `.keras` files are loaded with `load_model`, while `TFSMLayer` is only attempted for actual SavedModel directories (your current fallback is incompatible with a `.keras` file). Finally, we keep your preprocessing/inference logic intact but ensure `prediction` always matches `sample_submission.csv` length and that a valid `submission.csv` is always written.'
- What this solution (achieved 0.05531) has done: 'I fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf implementation **and** clearing any already-imported `google.protobuf` modules before importing TensorFlow, then re-execing once to ensure the setting takes effect. I also make the model path robust by falling back to `sample_submission.csv`-based default predictions if the provided model file isn’t present in this environment (your path `/kaggle/input/newmodel60/...` is very likely missing), while keeping your preprocessing and argmax inference unchanged when the model does load. Finally, I keep the submission-writing logic but ensure `prediction` always aligns with `image_ids` and is an integer label column so a valid `submission.csv` is always produced.'

# 9. Code solution

## === cell 0
import os
import sys

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

if os.environ.get("_PB_REEXEC_DONE", "0") != "1":
    os.environ["_PB_REEXEC_DONE"] = "1"
    os.execv(sys.executable, [sys.executable] + sys.argv)

import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf


def second_model_preprocess(image_np: np.ndarray) -> np.ndarray:
    image = Image.fromarray(image_np.astype("uint8"), "RGB")
    image = image.resize((224, 224))
    image = np.array(image, dtype=np.float32) / 255.0
    image = np.expand_dims(image, axis=0)
    return image


DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_CSV_PATH = os.path.join(DATA_ROOT, "train.csv")

MODEL_PATH = "/kaggle/input/newmodel60/keras/default/1/newModel60.keras"

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
image_ids = sample_sub["image_id"].tolist()

try:
    train_df = pd.read_csv(TRAIN_CSV_PATH)
    fallback_label = int(train_df["label"].value_counts().idxmax())
except Exception as _e:
    fallback_label = 0

model2 = None
use_tfsm_layer = False

if not (os.path.exists(MODEL_PATH) or os.path.isdir(MODEL_PATH)):
    print(f"WARNING: MODEL_PATH not found: {MODEL_PATH}")
    model2 = None
else:
    try:
        model2 = tf.keras.models.load_model(MODEL_PATH, compile=False)
    except Exception as e:
        print("Standard load_model failed.")
        print("Load error:", repr(e))

        if os.path.isdir(MODEL_PATH):
            print("Attempting TFSMLayer fallback (SavedModel directory detected).")
            try:
                model2 = tf.keras.Sequential(
                    [tf.keras.layers.TFSMLayer(MODEL_PATH, call_endpoint="serve")]
                )
                use_tfsm_layer = True
            except Exception as e2:
                print("TFSMLayer fallback failed.")
                print("Fallback error:", repr(e2))
                model2 = None
        else:
            print(
                "Skipping TFSMLayer fallback because MODEL_PATH is not a directory (not a SavedModel)."
            )
            model2 = None

prediction = []
missing = []

if model2 is None:
    print(
        f"WARNING: Model could not be loaded. Writing fallback predictions (most frequent train label = {fallback_label})."
    )
    prediction = [fallback_label] * len(image_ids)
else:
    for image_name in image_ids:
        img_path = os.path.join(TEST_IMG_DIR, image_name)
        if not os.path.exists(img_path):
            missing.append(image_name)
            prediction.append(fallback_label)
            continue

        img = Image.open(img_path).convert("RGB")
        image_np = np.array(img)
        x = second_model_preprocess(image_np)

        probs2 = model2.predict(x, verbose=0)[0]
        prediction.append(int(np.argmax(probs2)))

if missing:
    print(
        f"Warning: {len(missing)} images were missing from {TEST_IMG_DIR}. Example: {missing[:3]}"
    )

print(f"Inference complete. Predictions: {len(prediction)} / {len(image_ids)}")
print(
    f"Model loaded via {'TFSMLayer fallback' if use_tfsm_layer else ('standard load_model' if model2 is not None else 'FAILED')}"
)



## === cell 1
if len(prediction) != len(image_ids):
    print(
        f"WARNING: prediction length mismatch ({len(prediction)} != {len(image_ids)}). Padding/truncating."
    )
    if len(prediction) < len(image_ids):
        prediction = list(prediction) + [
            int(prediction[-1]) if len(prediction) else 0
        ] * (len(image_ids) - len(prediction))
    else:
        prediction = list(prediction)[: len(image_ids)]

submission = pd.DataFrame({"image_id": image_ids, "label": prediction})
submission["label"] = submission["label"].astype(int)

submission.to_csv("submission.csv", index=False)

print(submission.head())
print(f"Saved submission.csv with {len(submission)} rows")
print("submission.csv exists:", os.path.exists("submission.csv"))
