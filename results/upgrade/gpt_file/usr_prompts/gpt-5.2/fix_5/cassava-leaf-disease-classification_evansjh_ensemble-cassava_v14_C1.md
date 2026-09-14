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

0.8964944091870656

# 6. Current score

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I first fix the import/runtime crash caused by `tensorflow_hub` (it’s not needed for the rest of your pipeline once we fall back) and remove the missing `/kaggle/input/cp-model` dependency so the notebook can run in this environment. Then I make model loading robust: only load `.h5` models that actually exist, and if none are available (as in your current environment), fall back to a safe “majority class from train.csv” predictor so a valid `submission.csv` is always produced. Finally, I ensure predictions are generated in the exact `sample_submission.csv` order (required for Kaggle submissions) and that `label` is an integer column.'
- What this solution (achieved 0.61099) has done: 'You’re currently crashing at import time due to an incompatibility between TensorFlow and the protobuf runtime (`MessageFactory.GetPrototype`), so nothing after cell 0 can run. The minimal fix is to force TensorFlow to use the pure-Python protobuf implementation (compatible with newer protobuf) and to import TensorFlow only after that environment variable is set. This change is runtime-only and doesn’t alter your modeling/prediction logic; it just unblocks execution so you can actually load models if present or fall back to majority-class predictions and write a valid `submission.csv`. I also keep the rest of your pipeline intact and ensure the submission is still written in `sample_submission.csv` order with integer labels.'
- What this solution (achieved 0.61099) has done: 'You’re failing immediately on the TensorFlow import due to a protobuf runtime incompatibility; setting the env var alone isn’t sufficient unless it’s applied before Python imports protobuf/TensorFlow, and we should also force the protobuf Python implementation via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION`. I make TensorFlow an optional dependency: try to import it safely, and if it still fails in this environment, continue end-to-end using the existing “majority class” fallback so you always get a valid `submission.csv`. This fix is execution/stability-focused and keeps your core prediction logic identical when models are loadable, while unblocking the pipeline so it can run in Kaggle’s Python 3.13 environment. With no models available here, the score won’t improve (it remain the majority-class baseline), but the notebook reliably produce a valid submission file.'
- What this solution (achieved 0.61099) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf pure-Python implementation is forced before any protobuf/TensorFlow import and by removing already-imported protobuf modules from `sys.modules` to avoid the incompatible C++ runtime being reused. This is a runtime stability fix and preserves your existing prediction logic (load available `.h5` models → vote; otherwise fallback to majority class). I also keep the existing robust fallback so the notebook always finishes and writes `/kaggle/working/submission.csv` in the exact `sample_submission.csv` order. No model/training logic is changed; this only unblocks TensorFlow in the Kaggle Python 3.13 environment so you can actually use any provided models if present, which should improve score toward the target compared to the majority-class baseline.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PYTHONHASHSEED"] = "0"

import sys
import shutil
from collections import Counter

import numpy as np
import pandas as pd

np.random.seed(0)

for k in list(sys.modules.keys()):
    if k == "google" or k.startswith("google."):
        sys.modules.pop(k, None)

TF_AVAILABLE = False
tf = None
load_model = None
load_img = None
img_to_array = None

try:
    import tensorflow as tf  # noqa: F401
    from tensorflow.keras.models import load_model  # noqa: F401
    from tensorflow.keras.preprocessing.image import (
        load_img,
        img_to_array,
    )  # noqa: F401

    tf.random.set_seed(0)
    TF_AVAILABLE = True
    print("TensorFlow import: OK")
except Exception as e:
    TF_AVAILABLE = False
    print("TensorFlow import: FAILED; will run fallback predictor only.")
    print("Import error:", repr(e))



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

OUT_SUB_PATH = "/kaggle/working/submission.csv"

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing: {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

majority_label = int(train_df["label"].value_counts().idxmax())
label_counts = train_df["label"].value_counts().to_dict()
print("Train label distribution:", label_counts)
print("Majority label:", majority_label)



## === cell 2
model_candidates = [
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
        (550, 550),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5",
        (512, 512),
    ),
    (
        "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5",
        (448, 448),
    ),
    (
        "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5",
        (512, 512),
    ),
    (
        "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5",
        (550, 550),
    ),
    (
        "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5",
        (512, 512),
    ),
    ("/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5", (512, 512)),
]

available_models_info = [(p, sz) for (p, sz) in model_candidates if os.path.exists(p)]
print(f"Found {len(available_models_info)} available .h5 model(s).")
for p, sz in available_models_info:
    print(" -", p, "input:", sz)



## === cell 3
models = []
if TF_AVAILABLE and len(available_models_info) > 0:
    for path, input_size in available_models_info:
        try:
            m = load_model(path, compile=False)
            models.append((m, input_size))
        except Exception as e:
            print(f"Warning: failed to load model {path}: {e}")
else:
    if not TF_AVAILABLE:
        print("Skipping model loading because TensorFlow is unavailable.")
    else:
        print(
            "Skipping model loading because no candidate models exist in this environment."
        )

print(f"Successfully loaded {len(models)} model(s).")




## === cell 4
def predict_with_models_for_image(image_path: str, models_with_sizes):
    """
    Core logic preserved: per-model prediction -> majority vote;
    tie-break by average confidence among tied classes.
    """
    model_predictions = []
    confidence_scores = {}

    for model, input_size in models_with_sizes:
        img = load_img(image_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        preds = model.predict(img_array, verbose=0)
        pred_class = int(np.argmax(preds, axis=1)[0])

        conf = float(preds[0][pred_class]) if preds.ndim == 2 else float(np.max(preds))
        model_predictions.append(pred_class)
        confidence_scores.setdefault(pred_class, []).append(conf)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_pred = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
        final_pred = max(
            tied,
            key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls]),
        )

    return int(final_pred)




## === cell 5
image_ids = sample_df["image_id"].tolist()

pred_labels = []
use_models = TF_AVAILABLE and (len(models) > 0)

for image_id in image_ids:
    image_path = os.path.join(TEST_IMG_DIR, image_id)
    if not os.path.exists(image_path):
        pred_labels.append(majority_label)
        continue

    if use_models:
        pred = predict_with_models_for_image(image_path, models)
    else:
        pred = majority_label

    pred_labels.append(int(pred))

submission_df = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission_df["label"] = submission_df["label"].astype(int)

submission_df.to_csv(OUT_SUB_PATH, index=False)
print(f"Wrote submission to: {OUT_SUB_PATH}")
print(submission_df.head())



## === cell 6
assert os.path.exists(OUT_SUB_PATH), "submission.csv was not created."
chk = pd.read_csv(OUT_SUB_PATH)
assert list(chk.columns) == [
    "image_id",
    "label",
], f"Wrong columns: {chk.columns.tolist()}"
assert len(chk) == len(sample_df), f"Row count mismatch: {len(chk)} vs {len(sample_df)}"
assert chk["label"].dtype.kind in (
    "i",
    "u",
), f"Label dtype should be int, got {chk['label'].dtype}"
chk
