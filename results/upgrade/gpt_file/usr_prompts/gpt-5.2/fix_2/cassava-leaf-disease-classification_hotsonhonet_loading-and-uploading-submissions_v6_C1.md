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

0.1403747355696585

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
TRAIN_IMG_LOC = "../input/cassava-leaf-disease-classification/train_images"
TEST_IMG = "../input/cassava-leaf-disease-classification/test_images/2216849948.jpg"
TRAIN_CSV = "../input/cassava-leaf-disease-classification/train.csv"
SAMPLE_CSV = "../input/cassava-leaf-disease-classification/sample_submission.csv"
MODELS_WEIGHTS = "../input/cassavaeffentb7models/content/Models"

import os
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.utils import load_img, img_to_array
from tensorflow.keras.models import load_model

print("TensorFlow:", tf.__version__)
print("ALL Modules are successfully loaded")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def find_model_path(root_dir: str):
    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"MODELS_WEIGHTS directory not found: {root_dir}")

    candidates = []
    for r, dnames, fnames in os.walk(root_dir):
        for fn in fnames:
            lfn = fn.lower()
            if lfn.endswith(".h5") or lfn.endswith(".keras"):
                candidates.append(os.path.join(r, fn))

    preferred = []
    for c in candidates:
        lc = os.path.basename(c).lower()
        if "best" in lc or "eff" in lc or "b7" in lc or "model" in lc:
            preferred.append(c)

    if preferred:
        preferred.sort(key=lambda p: (len(p), p))
        return preferred[0]

    if candidates:
        candidates.sort(key=lambda p: (len(p), p))
        return candidates[0]

    savedmodel_dirs = []
    for r, dnames, fnames in os.walk(root_dir):
        if "saved_model.pb" in fnames:
            savedmodel_dirs.append(r)
    if savedmodel_dirs:
        savedmodel_dirs.sort(key=lambda p: (len(p), p))
        return savedmodel_dirs[0]

    raise FileNotFoundError(f"No .h5/.keras or SavedModel found under: {root_dir}")


model_path = find_model_path(MODELS_WEIGHTS)
print("Found model at:", model_path)

model = load_model(model_path, compile=False)
print("Model Loading Complete")



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3861258343.py in <cell line: 0>()
     38 
     39 
---> 40 model_path = find_model_path(MODELS_WEIGHTS)
     41 print("Found model at:", model_path)
     42 

/tmp/ipykernel_11/3861258343.py in find_model_path(root_dir)
      2 def find_model_path(root_dir: str):
      3     if not os.path.exists(root_dir):
----> 4         raise FileNotFoundError(f"MODELS_WEIGHTS directory not found: {root_dir}")
      5 
      6     candidates = []

FileNotFoundError: MODELS_WEIGHTS directory not found: ../input/cassavaeffentb7models/content/Models

## === cell 2
model.summary()



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2404564952.py in <cell line: 0>()
      1 # Fix: plot_model import previously caused environment/protobuf issues and is not needed to make a submission.
      2 # Instead, print a brief summary to confirm it loaded.
----> 3 model.summary()
      4 

NameError: name 'model' is not defined

## === cell 3
sub = pd.read_csv(SAMPLE_CSV)
assert {"image_id", "label"}.issubset(
    sub.columns
), f"Unexpected columns in sample submission: {sub.columns.tolist()}"
first_img_id = sub.loc[0, "image_id"]

test_img_dir_candidates = [
    "../input/cassava-leaf-disease-classification/test_images",
    "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
]
TEST_IMG_DIR = None
for cand in test_img_dir_candidates:
    if os.path.isdir(cand):
        TEST_IMG_DIR = cand
        break
if TEST_IMG_DIR is None:
    raise FileNotFoundError(
        f"Could not find test_images directory. Tried: {test_img_dir_candidates}"
    )

sample_path = os.path.join(TEST_IMG_DIR, first_img_id)
print("Using TEST_IMG_DIR:", TEST_IMG_DIR)
print("Example test image exists:", os.path.exists(sample_path), sample_path)



## === cell 4
from tensorflow.keras.applications.efficientnet import preprocess_input

IMG_SIZE = (380, 380)


def load_and_preprocess_image(path, img_size=IMG_SIZE):
    img = load_img(path)  # RGB by default
    img = img_to_array(img)
    try:
        from tensorflow.keras.utils import smart_resize

        img = smart_resize(img, img_size)
    except Exception:
        img = tf.image.resize(img, img_size).numpy()
    img = preprocess_input(img)
    return img


x = load_and_preprocess_image(sample_path)
x = np.expand_dims(x, 0)
print("Single image batch shape:", x.shape)



## === cell 5
preds = model.predict(x, verbose=0)
print("Single-image preds shape:", preds.shape)
print("Single-image predicted label:", int(np.argmax(preds, axis=1)[0]))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/331979018.py in <cell line: 0>()
      1 # Sanity predict on one image
----> 2 preds = model.predict(x, verbose=0)
      3 print("Single-image preds shape:", preds.shape)
      4 print("Single-image predicted label:", int(np.argmax(preds, axis=1)[0]))
      5 

NameError: name 'model' is not defined

## === cell 6
sub = pd.read_csv(SAMPLE_CSV)
image_ids = sub["image_id"].astype(str).tolist()

test_paths = [os.path.join(TEST_IMG_DIR, img_id) for img_id in image_ids]
missing = [p for p in test_paths[:50] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        f"Some test images not found (showing up to 5): {missing[:5]}"
    )

print("Number of test images:", len(test_paths))



## === cell 7
batch_size = 16  # conservative to avoid OOM; does not change model logic

all_preds = []
for i in range(0, len(test_paths), batch_size):
    batch_paths = test_paths[i : i + batch_size]
    batch_imgs = np.stack([load_and_preprocess_image(p) for p in batch_paths], axis=0)
    batch_pred = model.predict(batch_imgs, verbose=0)
    all_preds.append(batch_pred)

all_preds = np.concatenate(all_preds, axis=0)
labels = np.argmax(all_preds, axis=1).astype(int)

submission = pd.DataFrame({"image_id": image_ids, "label": labels})
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2358928164.py in <cell line: 0>()
      7     batch_paths = test_paths[i : i + batch_size]
      8     batch_imgs = np.stack([load_and_preprocess_image(p) for p in batch_paths], axis=0)
----> 9     batch_pred = model.predict(batch_imgs, verbose=0)
     10     all_preds.append(batch_pred)
     11 

NameError: name 'model' is not defined

## === cell 8
print(submission.tail())
print(
    "submission.csv saved in current working directory:",
    os.path.abspath("submission.csv"),
)

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1471928452.py in <cell line: 0>()
      1 # Display tail to confirm formatting
----> 2 print(submission.tail())
      3 print(
      4     "submission.csv saved in current working directory:",
      5     os.path.abspath("submission.csv"),

NameError: name 'submission' is not defined
