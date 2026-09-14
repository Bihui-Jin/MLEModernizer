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

os.environ["PYTHONHASHSEED"] = "0"
tf.random.set_seed(0)
np.random.seed(0)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(DATA_DIR, "test_images")
sample_path = os.path.join(DATA_DIR, "sample_submission.csv")

assert os.path.isdir(test_image_dir), f"Missing test images directory: {test_image_dir}"
assert os.path.isfile(sample_path), f"Missing sample submission: {sample_path}"

sample_csv = pd.read_csv(sample_path)
if not {"image_id", "label"}.issubset(sample_csv.columns):
    raise ValueError(
        f"sample_submission.csv must contain image_id,label. Got: {list(sample_csv.columns)}"
    )



## === cell 2
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

models_info_all = [
    (model_path_1, (550, 550)),
    (model_path_6, (512, 512)),
]

existing_models_info = [(p, s) for (p, s) in models_info_all if os.path.isfile(p)]

if len(existing_models_info) == 0:
    h5_candidates = []
    for root, _, files in os.walk("/kaggle/input"):
        for fn in files:
            if fn.lower().endswith(".h5"):
                h5_candidates.append(os.path.join(root, fn))
    preferred = [
        model_path_1,
        model_path_6,
        model_path_2,
        model_path_3,
        model_path_4,
        model_path_5,
        model_path_7,
    ]
    chosen = [p for p in preferred if p in h5_candidates][:2]
    if len(chosen) == 0:
        chosen = h5_candidates[:1]  # at least one model if any exists
    if len(chosen) == 0:
        raise FileNotFoundError(
            "No .h5 models found under /kaggle/input. Please add the model dataset to the notebook."
        )
    existing_models_info = [(chosen[0], (512, 512))] + (
        [(chosen[1], (550, 550))] if len(chosen) > 1 else []
    )

print("Models to load:")
for p, s in existing_models_info:
    print(f" - {p} @ {s}")



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2803108155.py in <cell line: 0>()
     49         chosen = h5_candidates[:1]  # at least one model if any exists
     50     if len(chosen) == 0:
---> 51         raise FileNotFoundError(
     52             "No .h5 models found under /kaggle/input. Please add the model dataset to the notebook."
     53         )

FileNotFoundError: No .h5 models found under /kaggle/input. Please add the model dataset to the notebook.

## === cell 3
models = []
for path, input_size in existing_models_info:
    try:
        m = load_model(path, compile=False)
        models.append((m, input_size))
    except Exception as e:
        raise RuntimeError(f"Failed to load model at {path}: {e}")

print(f"Loaded {len(models)} model(s).")



## === cell 4
image_predictions = []

test_files = [
    fn
    for fn in os.listdir(test_image_dir)
    if fn.lower().endswith((".jpg", ".jpeg", ".png"))
]
test_files.sort()

for image_id in test_files:
    model_predictions = []
    confidence_scores = {}

    img_path = os.path.join(test_image_dir, image_id)

    for model, input_size in models:
        img = load_img(img_path, target_size=input_size)
        img_array = np.expand_dims(img_to_array(img) / 255.0, axis=0)

        preds = model.predict(img_array, verbose=0)
        if isinstance(preds, (list, tuple)):
            preds = preds[0]
        if isinstance(preds, dict):
            preds = list(preds.values())[0]

        preds = np.asarray(preds)
        if preds.ndim == 2:
            p = preds[0]
        else:
            p = preds

        predicted_class = int(np.argmax(p))
        confidence_score = float(p[predicted_class])

        model_predictions.append(predicted_class)
        confidence_scores.setdefault(predicted_class, []).append(confidence_score)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()
    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        top_count = most_common[0][1]
        tied_classes = [cls for cls, count in most_common if count == top_count]
        final_predicted_class = int(
            max(
                tied_classes,
                key=lambda cls: (
                    sum(confidence_scores[cls]) / len(confidence_scores[cls])
                ),
            )
        )

    image_predictions.append({"image_id": image_id, "label": final_predicted_class})

submission_df = pd.DataFrame(image_predictions)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/3602740279.py in <cell line: 0>()
     43     class_votes = Counter(model_predictions)
     44     most_common = class_votes.most_common()
---> 45     final_predicted_class = most_common[0][0]
     46 
     47     # Tie-breaker by average confidence (as in original code)

IndexError: list index out of range

## === cell 5
submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")

if submission_df["label"].isna().any():
    submission_df["label"] = submission_df["label"].fillna(0).astype(int)
else:
    submission_df["label"] = submission_df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print(f"Wrote submission to: {out_path}")
print(submission_df.head())



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/439863893.py in <cell line: 0>()
      1 # Fix: ensure submission matches sample_submission ordering and includes all rows.
      2 # This avoids accidental missing/extra rows if directory listing differs.
----> 3 submission_df = sample_csv[["image_id"]].merge(submission_df, on="image_id", how="left")
      4 
      5 # If any missing predictions (shouldn't happen), fill with a safe default class 0.

NameError: name 'submission_df' is not defined

## === cell 6
print("Submission shape:", submission_df.shape)
print("Unique labels:", sorted(submission_df["label"].unique().tolist()))
assert (
    submission_df.shape[0] == sample_csv.shape[0]
), "Submission row count must match sample_submission.csv"
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission must have columns: image_id,label"

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2552389235.py in <cell line: 0>()
      1 # Display basic sanity checks
----> 2 print("Submission shape:", submission_df.shape)
      3 print("Unique labels:", sorted(submission_df["label"].unique().tolist()))
      4 assert (
      5     submission_df.shape[0] == sample_csv.shape[0]

NameError: name 'submission_df' is not defined
