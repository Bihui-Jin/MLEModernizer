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

0.8068902991840435

# 6. Current score

0.11584

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.07997) has done: 'I remove the `tensorflow_hub` import that is triggering the protobuf `MessageFactory.GetPrototype` crash in this environment, since it is unused by your pipeline. I also eliminate the notebook-style `!pip install` cell (it won’t run in a pure `.py` Kaggle submission context) and instead rely on `tf.keras.applications.EfficientNetB3` to rebuild the same expected architecture, then load your provided `.h5` weights so inference can proceed. Finally, I fix the missing model definition (`my_model`) and make the test image path robust to the actual Kaggle `/kaggle/input/...` location, ensuring a valid `submission.csv` with the correct `image_id,label` columns is always written.'
- What this solution (achieved 0.11584) has done: 'I remove the import path that triggers the `MessageFactory.GetPrototype` protobuf crash by avoiding Keras/TensorFlow submodules that pull in the problematic dependency chain, and I add a safe fallback so the script can run even if the external `.h5` weights dataset is not attached. I fix the Keras 3 weight-loading failure by using `tf.keras` consistently and (when weights exist) loading them via legacy H5 compatibility; otherwise the model run with random initialization to still generate a valid `submission.csv`. I also ensure the test image discovery and `image_id` extraction are correct and stable, and that the submission is written with the exact required columns. These are minimal changes focused on unblocking execution and improving accuracy when the weights file is available (which is necessary to reach the target score band).'

# 9. Code solution

## === cell 0
import os
import glob
import numpy as np
import pandas as pd

import tensorflow as tf

SEED = 42
DEBUG = False

os.environ["PYTHONHASHSEED"] = str(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT_CANDIDATES = [
    "/kaggle/input/cassava-leaf-disease-classification",
    "../input/cassava-leaf-disease-classification",
]

INPUT_ROOT = None
for p in INPUT_ROOT_CANDIDATES:
    if os.path.exists(p):
        INPUT_ROOT = p
        break

if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not locate cassava-leaf-disease-classification dataset folder in expected locations."
    )

TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test_images")

WEIGHT_PATH_CANDIDATES = [
    "/kaggle/input/experiment-with-models-using-keras-with-updates/fineTuned_v0.38.h5",
    "../input/experiment-with-models-using-keras-with-updates/fineTuned_v0.38.h5",
]

WEIGHT_PATH = None
for wp in WEIGHT_PATH_CANDIDATES:
    if os.path.exists(wp):
        WEIGHT_PATH = wp
        break

if WEIGHT_PATH is None:
    print(
        "WARNING: Could not find fineTuned_v0.38.h5 in expected locations. "
        "Proceeding without loading weights (submission will be valid but accuracy will be poor)."
    )



## === cell 2
layers = tf.keras.layers
Model = tf.keras.Model
EfficientNetB3 = tf.keras.applications.EfficientNetB3


def build_model(input_size=512, n_classes=5):
    inputs = layers.Input(shape=(input_size, input_size, 3))
    base = EfficientNetB3(include_top=False, weights=None, input_tensor=inputs)
    x = layers.GlobalAveragePooling2D()(base.output)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


my_model = build_model(input_size=512, n_classes=5)

if WEIGHT_PATH is not None:
    try:
        my_model.load_weights(WEIGHT_PATH)
        print(f"Loaded weights from: {WEIGHT_PATH}")
    except Exception as e:
        print(f"WARNING: Failed to load weights from {WEIGHT_PATH} due to: {repr(e)}")
        print(
            "Proceeding with randomly initialized weights (submission will be valid but accuracy will be poor)."
        )



## === cell 3
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_images = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
if len(test_images) == 0:
    raise FileNotFoundError(f"No .jpg files found under: {TEST_IMG_DIR}")

df_test = pd.DataFrame({"path": test_images})


def make_test_gen(batch_size=64):
    my_test_idg = ImageDataGenerator()
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=(512, 512),
    )
    return test_gen




## === cell 4
pred_list = []
for i in range(1):
    test_gen = make_test_gen(batch_size=128)
    pred_test = my_model.predict(test_gen, verbose=1)
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].apply(
    lambda p: os.path.basename(p)
)
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]]
final_csv.to_csv("submission.csv", index=False)

if final_csv.shape[0] != len(test_images):
    raise RuntimeError("Submission row count does not match number of test images.")
if final_csv["image_id"].isna().any():
    raise RuntimeError("Found NaN image_id values in submission.")
if final_csv["label"].isna().any():
    raise RuntimeError("Found NaN label values in submission.")
if not os.path.exists("submission.csv"):
    raise RuntimeError("submission.csv was not written as expected.")

print("Wrote submission.csv with shape:", final_csv.shape)



## === cell 5
final_csv.head()
