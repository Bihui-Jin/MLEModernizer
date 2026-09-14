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

0.880930794802055

# 6. Current score

0.08819

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.08819) has done: 'I fix the import-time crash by removing the `keras.utils.vis_utils` import (it triggers a protobuf/TF incompatibility in this environment) and use `tf.keras.utils.plot_model` only if available. Then I fix the missing model file by falling back to building a standard `tf.keras.applications` model when the external `.h5` isn’t present, so the notebook always runs end-to-end. Finally, I ensure inference preprocessing matches the chosen backbone (EfficientNetB7) and that prediction code calls `model.predict(img)` with the correct input signature, producing a valid `submission.csv` with `image_id,label`.'

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG = "../input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "../input/cassavaeffentb7models/content/Models"

import os
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    import cv2
except Exception:
    cv2 = None

try:
    from matplotlib import pyplot as plt
except Exception:
    plt = None

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1

MODEL_PATH = "../input/ensemble-resnet50-effb0-effb4/resnet50_b0_b4.h5"

NUM_CLASSES = 5
IMG_SIZE = (300, 300)

model = None
if os.path.exists(MODEL_PATH):
    model = keras.models.load_model(MODEL_PATH, compile=False)
    print("Model loaded from:", MODEL_PATH)
else:
    base = keras.applications.EfficientNetB7(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
        pooling="avg",
    )
    x = base.output
    out = layers.Dense(NUM_CLASSES, activation="softmax")(x)
    model = keras.Model(inputs=base.input, outputs=out)

    try:
        if os.path.isdir(MODELS_WEIGHTS):
            cand = []
            for root, _, files in os.walk(MODELS_WEIGHTS):
                for f in files:
                    if f.endswith((".h5", ".weights.h5")):
                        cand.append(os.path.join(root, f))
            cand = sorted(cand)
            if len(cand) > 0:
                try:
                    model.load_weights(cand[0])
                    print("Loaded fallback weights from:", cand[0])
                except Exception as e:
                    print(
                        "Found weights but could not load (continuing without):",
                        str(e)[:200],
                    )
    except Exception as e:
        print("Weights scan failed (continuing without):", str(e)[:200])

    print("Fallback EfficientNetB7 model built (ImageNet weights).")

print("Model Loading Complete")



## === cell 2
try:
    keras.utils.plot_model(
        model,
        show_shapes=True,
        show_layer_names=True,
        to_file="ensemble_resnet50_effb0_effb4.png",
    )
    print("Model plot saved to ensemble_resnet50_effb0_effb4.png")
except Exception as e:
    print("Skipping plot_model (non-fatal):", str(e)[:200])



## === cell 3
if cv2 is not None and plt is not None and os.path.exists(TEST_IMG):
    img = cv2.imread(TEST_IMG)
    if img is not None:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        print(f"Image Shape : {img.shape}")

        plt.figure(figsize=(8, 6))
        plt.xticks([])
        plt.yticks([])
        plt.imshow(img)
        plt.show()
    else:
        print("Could not read TEST_IMG:", TEST_IMG)
else:
    print("Skipping image preview (cv2/plt missing or TEST_IMG not found).")



## === cell 4

ss = pd.read_csv(SAMPLE_CSV)
test_dir = "../input/cassava-leaf-disease-classification/test_images"

preprocess_fn = None
model_name = (model.name or "").lower()
if "efficientnet" in model_name:
    preprocess_fn = keras.applications.efficientnet.preprocess_input
elif "resnet" in model_name:
    preprocess_fn = keras.applications.resnet.preprocess_input
else:
    preprocess_fn = keras.applications.efficientnet.preprocess_input

preds = []
batch_size = 32

images = ss["image_id"].tolist()
for i in range(0, len(images), batch_size):
    batch_files = images[i : i + batch_size]
    batch = []
    for image_id in batch_files:
        path = os.path.join(test_dir, image_id)
        img = keras.preprocessing.image.load_img(path)
        img = keras.preprocessing.image.img_to_array(img)
        img = tf.keras.preprocessing.image.smart_resize(img, IMG_SIZE)
        batch.append(img)
    batch = np.stack(batch, axis=0).astype(np.float32)
    batch = preprocess_fn(batch)

    try:
        prob = model.predict(batch, verbose=0)
    except Exception:
        prob = model.predict([batch] * 3, verbose=0)

    preds.extend(np.argmax(prob, axis=1).tolist())

my_submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds})
my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 5
my_submission.head()
