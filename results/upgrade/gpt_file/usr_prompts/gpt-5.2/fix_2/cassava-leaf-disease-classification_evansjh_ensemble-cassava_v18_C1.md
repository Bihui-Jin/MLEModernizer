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

0.9013297068600786

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array


print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = f"{BASE}/test_images"
sample_path = f"{BASE}/sample_submission.csv"

assert os.path.exists(test_image_dir), f"Missing test image dir: {test_image_dir}"
assert os.path.exists(sample_path), f"Missing sample submission: {sample_path}"

sample_csv = pd.read_csv(sample_path)
print("sample_submission shape:", sample_csv.shape)
sample_csv.head()



## === cell 2
source_dir = "/kaggle/input/cp-model"
dest_dir = "/kaggle/working/cp-model"
if os.path.exists(source_dir):
    if os.path.exists(dest_dir):
        shutil.rmtree(dest_dir)
    shutil.copytree(source_dir, dest_dir)
    os.environ["TFHUB_CACHE_DIR"] = dest_dir
    print("TFHUB_CACHE_DIR set to:", dest_dir)
else:
    print(
        "Optional TFHub cache dataset not found at",
        source_dir,
        "- continuing without TFHub.",
    )



## === cell 3
model_path_1 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5"
)
model_path_2 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5"
)
model_path_3 = (
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5"
)
model_path_4 = (
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5"
)
model_path_5 = (
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5"
)
model_path_6 = "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5"
model_path_7 = "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5"

classifier = None  # placeholder to keep variable names consistent
model_path_8 = classifier

paths = [
    model_path_1,
    model_path_2,
    model_path_3,
    model_path_4,
    model_path_5,
    model_path_6,
    model_path_7,
]
exists = {p: os.path.exists(p) for p in paths}
print("Model files found:", sum(exists.values()), "of", len(paths))
for p, ok in exists.items():
    if ok:
        print("  OK:", p)
    else:
        print("  MISSING:", p)



## === cell 4
models_info = [
    (model_path_1, (550, 550)),
    (model_path_2, (550, 550)),
    (model_path_3, (299, 299)),  # common for InceptionV3
    (model_path_4, (550, 550)),
    (model_path_5, (550, 550)),
    (model_path_6, (550, 550)),
    (model_path_7, (512, 512)),
]

models = []
for path, input_size in models_info:
    if not os.path.exists(path):
        continue
    try:
        m = load_model(path, compile=False)
        models.append((m, input_size))
        print("Loaded model:", os.path.basename(path), "input_size:", input_size)
    except Exception as e:
        print("Failed to load model:", path, "error:", repr(e))

if classifier is not None:
    models.append((classifier, (224, 224)))

print("Total loaded models:", len(models))
if len(models) == 0:
    raise RuntimeError(
        "No external .h5 models could be loaded from /kaggle/input. "
        "Please attach the datasets that contain the listed model files, or update paths."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/2049591630.py in <cell line: 0>()
     28 print("Total loaded models:", len(models))
     29 if len(models) == 0:
---> 30     raise RuntimeError(
     31         "No external .h5 models could be loaded from /kaggle/input. "
     32         "Please attach the datasets that contain the listed model files, or update paths."

RuntimeError: No external .h5 models could be loaded from /kaggle/input. Please attach the datasets that contain the listed model files, or update paths.

## === cell 5
image_predictions = []

test_ids = set(sample_csv["image_id"].tolist())
available_files = [
    f for f in os.listdir(test_image_dir) if f.endswith((".jpg", ".jpeg", ".png"))
]
available_set = set(available_files)

missing = [
    img_id for img_id in sample_csv["image_id"].tolist() if img_id not in available_set
]
if missing:
    print(
        "Warning: missing",
        len(missing),
        "test images referenced in sample_submission (showing up to 5):",
        missing[:5],
    )

for image_id in sample_csv["image_id"].tolist():
    if image_id not in available_set:
        image_predictions.append({"image_id": image_id, "label": 0})
        continue

    model_predictions = []
    confidence_scores = {}

    img_path = os.path.join(test_image_dir, image_id)

    for model, input_size in models:
        img = load_img(img_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        if callable(model) and not hasattr(model, "predict"):
            preds = model(img_array, training=False)
            preds = tf.convert_to_tensor(preds)
            predicted_class = int(tf.math.argmax(preds, axis=-1).numpy()[0])
            preds_np = preds.numpy()
        else:
            preds_np = model.predict(img_array, verbose=0)
            predicted_class = int(np.argmax(preds_np, axis=1)[0])

        confidence_score = float(preds_np[0][predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        tied_classes = [cls for cls, count in most_common if count == most_common[0][1]]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: sum(confidence_scores[cls]) / len(confidence_scores[cls]),
        )

    image_predictions.append(
        {"image_id": image_id, "label": int(final_predicted_class)}
    )

submission_df = pd.DataFrame(image_predictions)

submission_df = submission_df[["image_id", "label"]]
assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count mismatch vs sample_submission"
assert list(submission_df.columns) == ["image_id", "label"], "Wrong submission columns"

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission_df.shape)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1894103532.py in <cell line: 0>()
     53     class_votes = Counter(model_predictions)
     54     most_common = class_votes.most_common()
---> 55     final_predicted_class = most_common[0][0]
     56 
     57     # Tie-breaker by average confidence (same as original intent; removed unused loop variable)

IndexError: list index out of range

## === cell 6
submission_df.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3942055985.py in <cell line: 0>()
----> 1 submission_df.head()
      2 

NameError: name 'submission_df' is not defined

## === cell 7
print("Label value counts:")
print(submission_df["label"].value_counts().sort_index())
print(
    "Min label:",
    submission_df["label"].min(),
    "Max label:",
    submission_df["label"].max(),
)
assert (
    submission_df["label"].between(0, 4).all()
), "Labels must be in [0,4] for this competition."
submission_df.tail()

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2413095531.py in <cell line: 0>()
      1 # Basic sanity checks for label range
      2 print("Label value counts:")
----> 3 print(submission_df["label"].value_counts().sort_index())
      4 print(
      5     "Min label:",

NameError: name 'submission_df' is not defined
