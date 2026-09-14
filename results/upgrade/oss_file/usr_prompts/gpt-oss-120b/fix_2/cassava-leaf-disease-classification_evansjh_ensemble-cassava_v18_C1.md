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
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from collections import Counter

try:
    import tensorflow_hub as hub
except Exception as e:
    hub = None
    print("tensorflow_hub unavailable or failed to import:", e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUBMISSION = os.path.join(BASE_INPUT, "sample_submission.csv")
SUBMISSION_PATH = "/kaggle/working/submission.csv"

train_df = pd.read_csv(TRAIN_CSV)
majority_label = train_df["label"].mode()[0]



## === cell 2
classifier = None
if hub is not None:
    try:
        model_url = "https://tfhub.dev/google/cropnet/classifier/cassava_disease_V1/2"
        classifier = hub.load(model_url)
        print("TF‑Hub classifier loaded.")
    except Exception as e:
        print("Failed to load TF‑Hub model, will use fallback classifier:", e)

model_paths = [
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/BestModel_3454_8937.h5",
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/best_model_0.37458707.h5",
    "/kaggle/input/combinedmodel3/tensorflow2/default/1/googlenet_inceptionv3.h5",
    "/kaggle/input/bestmodel_550_2/tensorflow2/default/1/BestModel_3577_8940.h5",
    "/kaggle/input/bestmodel_8878/tensorflow2/default/1/BestModel_8878_0358.h5",
    "/kaggle/input/bestmodel_8875/tensorflow2/default/1/BestModel_8875.h5",
    "/kaggle/input/googlenet_512/tensorflow2/default/1/GOOG1.h5",
]

loaded_models = []  # will contain tuples (model, input_size)
for path in model_paths:
    if os.path.exists(path):
        try:
            model = load_model(path)
            input_size = (224, 224)
            loaded_models.append((model, input_size))
            print(f"Loaded model from {path}")
        except Exception as e:
            print(f"Error loading model {path}: {e}")
    else:
        print(f"Model file not found, skipping: {path}")

if classifier is not None:
    loaded_models.append((classifier, (224, 224)))



## === cell 3
image_predictions = []

use_majority_only = len(loaded_models) == 0
if use_majority_only:
    print(
        "No pretrained models available – falling back to majority class prediction for all images."
    )

for image_name in os.listdir(TEST_IMG_DIR):
    if not image_name.lower().endswith((".jpg", ".jpeg", ".png")):
        continue

    if use_majority_only:
        final_pred = majority_label
    else:
        model_preds = []
        confidence_scores = {}

        for model, input_sz in loaded_models:
            img_path = os.path.join(TEST_IMG_DIR, image_name)
            img = load_img(img_path, target_size=input_sz)
            img_arr = np.expand_dims(img_to_array(img) / 255.0, axis=0)

            if (
                hub is not None
                and isinstance(model, hub.Module)
                or hasattr(model, "__call__")
                and model is classifier
            ):
                try:
                    preds = model(img_arr, training=False)
                    preds = tf.nn.softmax(preds, axis=-1).numpy()
                except Exception as e:
                    print(f"Classifier inference error on {image_name}: {e}")
                    preds = np.full((1, 5), 1 / 5)  # uniform fallback
                pred_class = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][pred_class])
            else:
                preds = model.predict(img_arr, verbose=0)
                pred_class = int(np.argmax(preds, axis=1)[0])
                confidence = float(preds[0][pred_class])

            model_preds.append(pred_class)
            confidence_scores.setdefault(pred_class, []).append(confidence)

        vote_counts = Counter(model_preds)
        most_common = vote_counts.most_common()
        final_pred = most_common[0][0]
        if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
            tied = [cls for cls, cnt in most_common if cnt == most_common[0][1]]
            avg_conf = {cls: np.mean(confidence_scores[cls]) for cls in tied}
            final_pred = max(avg_conf, key=avg_conf.get)

    image_predictions.append({"image_id": image_name, "label": final_pred})

submission_df = pd.DataFrame(image_predictions)
submission_df.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}")



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3041161962.py in <cell line: 0>()
     26             if (
     27                 hub is not None
---> 28                 and isinstance(model, hub.Module)
     29                 or hasattr(model, "__call__")
     30                 and model is classifier

AttributeError: module 'tensorflow_hub' has no attribute 'Module'

## === cell 4
submission_df.head()

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1976027514.py in <cell line: 0>()
      1 # Display a few rows of the created submission
----> 2 submission_df.head()

NameError: name 'submission_df' is not defined
