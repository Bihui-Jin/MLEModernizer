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

0.8724690238742823

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import json
import numpy as np
import pandas as pd
from PIL import Image
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.models import load_model
from tensorflow.keras.utils import Sequence



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
TRAIN_DIR = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
TEST_DIR = "/kaggle/input/cassava-leaf-disease-classification/test_images/"



## === cell 2
with open(os.path.join(BASE_DIR, "label_num_to_disease_map.json")) as file:
    map_classes = json.load(file)
print(json.dumps(map_classes, indent=2))



## === cell 3
label_list = [int(k) for k in map_classes.keys()]



## === cell 4
IMG_HEIGHT = 400
IMG_WIDTH = 400
BATCH_SIZE = 32
PRE_TRAINED_MODEL = "../input/xceptionv6/Cassava_Best_Xception_Model_V05.hdf5"




## === cell 5
def load_single_image(data_type, image_id):
    """Load and resize a single image."""
    img_path = os.path.join(
        TEST_DIR if data_type == "TEST_DATA" else TRAIN_DIR, image_id
    )
    img = Image.open(img_path).convert("RGB")
    img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
    return np.array(img)




## === cell 6
class SimpleSequence(Sequence):
    """A minimal data sequence for inference (no augmentation)."""

    def __init__(self, ids, batch_size):
        self.ids = ids
        self.batch_size = batch_size

    def __len__(self):
        return int(np.ceil(len(self.ids) / self.batch_size))

    def __getitem__(self, idx):
        batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
        batch_imgs = [load_single_image("TEST_DATA", img_id) for img_id in batch_ids]
        return np.stack(batch_imgs, axis=0)




## === cell 7
try:
    model = load_model(PRE_TRAINED_MODEL)
    print("Loaded pretrained model.")
except Exception as e:
    print(
        f"Pretrained model not found or could not be loaded ({e}); building a lightweight fallback model."
    )
    input_tensor = layers.Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3))
    x = layers.Rescaling(1.0 / 255)(input_tensor)
    x = layers.Conv2D(16, 3, activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    output = layers.Dense(len(label_list), activation="softmax")(x)
    model = models.Model(inputs=input_tensor, outputs=output)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy")
    print("Fallback model created.")



## === cell 8
test_filenames = sorted(os.listdir(TEST_DIR))
test_seq = SimpleSequence(test_filenames, BATCH_SIZE)



## === cell 9
predictions = []
for batch_imgs in test_seq:
    probs = model.predict(batch_imgs, verbose=0)
    preds = np.argmax(probs, axis=1)
    predictions.extend(preds)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
IsADirectoryError                         Traceback (most recent call last)
/tmp/ipykernel_11/2713497908.py in <cell line: 0>()
      1 predictions = []
----> 2 for batch_imgs in test_seq:
      3     probs = model.predict(batch_imgs, verbose=0)
      4     preds = np.argmax(probs, axis=1)
      5     predictions.extend(preds)

/tmp/ipykernel_11/1617655601.py in __getitem__(self, idx)
     11     def __getitem__(self, idx):
     12         batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
---> 13         batch_imgs = [load_single_image("TEST_DATA", img_id) for img_id in batch_ids]
     14         return np.stack(batch_imgs, axis=0)
     15 

/tmp/ipykernel_11/1617655601.py in <listcomp>(.0)
     11     def __getitem__(self, idx):
     12         batch_ids = self.ids[idx * self.batch_size : (idx + 1) * self.batch_size]
---> 13         batch_imgs = [load_single_image("TEST_DATA", img_id) for img_id in batch_ids]
     14         return np.stack(batch_imgs, axis=0)
     15 

/tmp/ipykernel_11/2347000945.py in load_single_image(data_type, image_id)
      4         TEST_DIR if data_type == "TEST_DATA" else TRAIN_DIR, image_id
      5     )
----> 6     img = Image.open(img_path).convert("RGB")
      7     img = img.resize((IMG_HEIGHT, IMG_WIDTH), Image.LANCZOS)
      8     return np.array(img)

/usr/local/lib/python3.11/dist-packages/PIL/Image.py in open(fp, mode, formats)
   3511     if is_path(fp):
   3512         filename = os.fspath(fp)
-> 3513         fp = builtins.open(filename, "rb")
   3514         exclusive_fp = True
   3515     else:

IsADirectoryError: [Errno 21] Is a directory: '/kaggle/input/cassava-leaf-disease-classification/test_images/test_images'

## === cell 10
submission_df = pd.DataFrame({"image_id": test_filenames, "label": predictions})
submission_path = "submission.csv"
submission_df.to_csv(submission_path, index=False)
print(f"Submission written to {submission_path}")

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3008263350.py in <cell line: 0>()
----> 1 submission_df = pd.DataFrame({"image_id": test_filenames, "label": predictions})
      2 submission_path = "submission.csv"
      3 submission_df.to_csv(submission_path, index=False)
      4 print(f"Submission written to {submission_path}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    675         lengths = list(set(raw_lengths))
    676         if len(lengths) > 1:
--> 677             raise ValueError("All arrays must be of the same length")
    678 
    679         if have_dicts:

ValueError: All arrays must be of the same length
