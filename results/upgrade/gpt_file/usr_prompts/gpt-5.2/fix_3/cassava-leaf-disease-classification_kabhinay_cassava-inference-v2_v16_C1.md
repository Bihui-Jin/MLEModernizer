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

2.7

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

0.8828951344817165

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
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1


def _find_existing_dir(candidates):
    for p in candidates:
        if p and os.path.isdir(p):
            return p
    return None


def _list_subdirs(path):
    if not os.path.isdir(path):
        return []
    out = []
    for name in sorted(os.listdir(path)):
        full = os.path.join(path, name)
        if os.path.isdir(full):
            out.append(full)
    return out




## === cell 2
def load_savedmodel_as_keras_model(savedmodel_dir, input_shape=(448, 448, 3)):
    """
    Loads a TF SavedModel and wraps it as a Keras Model via TFSMLayer.
    Keeps the original computation graph/weights; only changes the loading mechanism.
    """
    layer = keras.layers.TFSMLayer(savedmodel_dir, call_endpoint="serving_default")
    inp = keras.Input(shape=input_shape, name="input_image")
    out = layer(inp)
    if isinstance(out, dict):
        out = out[sorted(out.keys())[0]]
    return keras.Model(inp, out)


DATA_DIR = "../input/cassava-leaf-disease-classification"


def _find_savedmodels(root):
    saved = []
    for dirpath, dirnames, filenames in os.walk(root):
        if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
            saved.append(dirpath)
    return sorted(set(saved))


savedmodel_candidates = _find_savedmodels(DATA_DIR)

if len(savedmodel_candidates) < 2:
    alt_root = "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification"
    savedmodel_candidates += _find_savedmodels(alt_root)
    savedmodel_candidates = sorted(set(savedmodel_candidates))

print("Found SavedModel dirs:", len(savedmodel_candidates))
for p in savedmodel_candidates[:10]:
    print(" -", p)

if len(savedmodel_candidates) < 2:
    raise OSError(
        "Could not find at least 2 SavedModel directories inside the provided input dataset. "
        "Please ensure models are present as TF SavedModel folders containing saved_model.pb."
    )

MODEL_V3_PATH = savedmodel_candidates[0]
MODEL_V4_PATH = savedmodel_candidates[1]

print("Using model_v3:", MODEL_V3_PATH)
print("Using model_v4:", MODEL_V4_PATH)

model_v3 = load_savedmodel_as_keras_model(MODEL_V3_PATH, input_shape=(448, 448, 3))
model_v4 = load_savedmodel_as_keras_model(MODEL_V4_PATH, input_shape=(448, 448, 3))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_11/114966662.py in <cell line: 0>()
     39 
     40 if len(savedmodel_candidates) < 2:
---> 41     raise OSError(
     42         "Could not find at least 2 SavedModel directories inside the provided input dataset. "
     43         "Please ensure models are present as TF SavedModel folders containing saved_model.pb."

OSError: Could not find at least 2 SavedModel directories inside the provided input dataset. Please ensure models are present as TF SavedModel folders containing saved_model.pb.

## === cell 3
def random_crop(img, random_crop_size):
    assert img.shape[2] == 3
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y : (y + dy), x : (x + dx), :]


def crop_generator(batches, crop_length):
    """Take as input a Keras ImageGen (Iterator) and generate random
    crops from the image batches generated by the original iterator.
    """
    while True:
        batch_x = next(batches)
        batch_crops = np.zeros(
            (batch_x.shape[0], crop_length, crop_length, 3), dtype=batch_x.dtype
        )
        for i in range(batch_x.shape[0]):
            batch_crops[i] = random_crop(batch_x[i], (crop_length, crop_length))
        yield batch_crops




## === cell 4
TEST_DIR = os.path.join(DATA_DIR, "test_images")
if not os.path.isdir(TEST_DIR):
    TEST_DIR = os.path.join(
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "test_images",
    )
if not os.path.isdir(TEST_DIR):
    raise OSError("test_images directory not found in expected locations.")

sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")
if not os.path.isfile(sample_sub_path):
    sample_sub_path = os.path.join(
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification",
        "sample_submission.csv",
    )
sample_sub = pd.read_csv(sample_sub_path)
test_df = sample_sub[["image_id"]].copy()

test_datagen = ImageDataGenerator()

test_generator = test_datagen.flow_from_dataframe(
    test_df,
    directory=TEST_DIR,
    x_col="image_id",
    target_size=(448, 448),
    batch_size=1,
    class_mode=None,
    shuffle=False,
)



## === cell 5
pred_v3 = model_v3.predict(test_generator, verbose=1, steps=len(test_df))
pred_v3 = np.asarray(pred_v3)
if pred_v3.ndim == 2 and pred_v3.shape[1] == 6:
    pred_v3 = pred_v3[:, 1:]

test_generator.reset()
pred_v4 = model_v4.predict(test_generator, verbose=1, steps=len(test_df))
pred_v4 = np.asarray(pred_v4)

if pred_v3.ndim != 2 or pred_v4.ndim != 2:
    raise ValueError(
        "Model predictions must be 2D arrays of shape (n_samples, n_classes)."
    )
if pred_v3.shape[0] != len(test_df) or pred_v4.shape[0] != len(test_df):
    raise ValueError("Prediction row count mismatch with test dataframe length.")
if pred_v3.shape[1] != pred_v4.shape[1]:
    raise ValueError(
        "Model class dimension mismatch: {} vs {}".format(
            pred_v3.shape[1], pred_v4.shape[1]
        )
    )



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1453173502.py in <cell line: 0>()
----> 1 pred_v3 = model_v3.predict(test_generator, verbose=1, steps=len(test_df))
      2 pred_v3 = np.asarray(pred_v3)
      3 # Keep original semantics: gambler model might output an extra "reject" channel => drop it.
      4 if pred_v3.ndim == 2 and pred_v3.shape[1] == 6:
      5     pred_v3 = pred_v3[:, 1:]

NameError: name 'model_v3' is not defined

## === cell 6
pred_new = 0.5 * pred_v3 + 0.5 * pred_v4
predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)

submission = pd.DataFrame(
    {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}
)

submission["label"] = submission["label"].astype(int)
submission = submission[["image_id", "label"]]

out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)

print("Wrote:", out_path)
print(submission.head())
print("Rows:", len(submission), "Cols:", submission.columns.tolist())

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1292275855.py in <cell line: 0>()
----> 1 pred_new = 0.5 * pred_v3 + 0.5 * pred_v4
      2 predicted_class_indices_new = np.argmax(pred_new, axis=1).astype(int)
      3 
      4 submission = pd.DataFrame(
      5     {"image_id": test_df["image_id"].values, "label": predicted_class_indices_new}

NameError: name 'pred_v3' is not defined
