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

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames[:2]:
        print(os.path.join(dirname, filename))



## === cell 1
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
SAMPLE_SUB_PATH = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

import tensorflow as tf
from tensorflow import keras

MODEL_PATH = "/kaggle/input/effficientnetb3-cassava/best_model.hdf5"


def _find_model_path(path: str) -> str:
    if os.path.exists(path):
        return path

    wanted = os.path.basename(path)

    for root, _, files in os.walk("/kaggle/input"):
        if wanted in files:
            return os.path.join(root, wanted)

    candidates = {"best_model.hdf5", "best_model.h5", "model.h5", "model.hdf5"}
    for root, _, files in os.walk("/kaggle/input"):
        for f in files:
            if f in candidates:
                return os.path.join(root, f)

    raise FileNotFoundError(
        f"Could not find model file. Tried basename={wanted} and common alternatives under /kaggle/input."
    )


MODEL_PATH = _find_model_path(MODEL_PATH)
print("Using MODEL_PATH:", MODEL_PATH)

IMG_SIZE = 300
N_CLASSES = 5


def build_efficientnetb3_classifier(img_size=300, n_classes=5):
    inputs = keras.Input(shape=(img_size, img_size, 3))
    base = keras.applications.EfficientNetB3(
        include_top=False,
        weights=None,  # competition-provided weights will be loaded from file
        input_tensor=inputs,
        pooling="avg",
    )
    outputs = keras.layers.Dense(n_classes, activation="softmax")(base.output)
    model = keras.Model(inputs=inputs, outputs=outputs)
    return model


new_model = build_efficientnetb3_classifier(IMG_SIZE, N_CLASSES)

loaded = False
try:
    new_model.load_weights(MODEL_PATH)
    loaded = True
    print("Loaded weights successfully via load_weights().")
except Exception as e:
    print("load_weights failed with:", repr(e))

if not loaded:
    from tensorflow.keras.models import load_model

    new_model = load_model(MODEL_PATH, compile=False)
    print("Loaded full model successfully via load_model() fallback.")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
new_model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3406495079.py in <cell line: 0>()
----> 1 new_model.summary()
      2 

NameError: name 'new_model' is not defined

## === cell 3
try:
    from tensorflow.keras.applications.efficientnet import preprocess_input
except Exception:
    from tensorflow.keras.applications.efficientnet_v2 import preprocess_input

size = (IMG_SIZE, IMG_SIZE)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/182095994.py in <cell line: 0>()
      5     from tensorflow.keras.applications.efficientnet_v2 import preprocess_input
      6 
----> 7 size = (IMG_SIZE, IMG_SIZE)
      8 

NameError: name 'IMG_SIZE' is not defined

## === cell 4
from PIL import Image

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_images = sample_sub["image_id"].tolist()

missing = [
    img_id
    for img_id in test_images
    if not os.path.exists(os.path.join(TEST_DIR, img_id))
]
if missing:
    print(
        f"Warning: {len(missing)} images from sample_submission not found in {TEST_DIR}. "
        "Proceeding by listing directory JPGs and will map back to sample_submission order."
    )
    test_images = sorted(
        [f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")]
    )

preds = []
pred_names = []

for image_name in test_images:
    img_path = os.path.join(TEST_DIR, image_name)
    if not os.path.exists(img_path):
        continue

    img = Image.open(img_path).convert("RGB")
    img = img.resize(size)

    x = np.array(img, dtype=np.float32)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    p = new_model.predict(x, verbose=0)
    preds.append(int(np.argmax(p, axis=1)[0]))
    pred_names.append(image_name)

print("n_test_images_seen:", len(test_images), "n_predicted:", len(preds))

pred_map = dict(zip(pred_names, preds))
fallback_label = (
    int(pd.Series(list(pred_map.values())).mode().iloc[0]) if len(pred_map) else 0
)

final_labels = [
    int(pred_map.get(img_id, fallback_label))
    for img_id in sample_sub["image_id"].tolist()
]

sub = pd.DataFrame({"image_id": sample_sub["image_id"].tolist(), "label": final_labels})
sub = sub[["image_id", "label"]]
sub["label"] = sub["label"].astype(int).clip(0, N_CLASSES - 1)

sub.to_csv("submission.csv", index=False)
print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("Saved to:", os.path.abspath("submission.csv"))

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2212343291.py in <cell line: 0>()
     27 
     28     img = Image.open(img_path).convert("RGB")
---> 29     img = img.resize(size)
     30 
     31     x = np.array(img, dtype=np.float32)

NameError: name 'size' is not defined
