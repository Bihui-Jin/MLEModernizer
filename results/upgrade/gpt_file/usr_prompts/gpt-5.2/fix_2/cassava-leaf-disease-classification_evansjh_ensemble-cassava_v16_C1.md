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

0.8974010275007556

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import shutil
from collections import Counter

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
tf.random.set_seed(0)

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
test_image_dir = os.path.join(BASE_INPUT, "test_images")
sample = os.path.join(BASE_INPUT, "sample_submission.csv")
out_path = "/kaggle/working/submission.csv"

if not os.path.exists(sample):
    alt = "/kaggle/input/sample_submission.csv"
    if os.path.exists(alt):
        sample = alt
    else:
        raise FileNotFoundError(f"sample_submission.csv not found at {sample} or {alt}")

if not os.path.exists(test_image_dir):
    alt = "/kaggle/input/test_images"
    if os.path.exists(alt):
        test_image_dir = alt
    else:
        raise FileNotFoundError(
            f"test_images directory not found at {test_image_dir} or {alt}"
        )

print("Using:")
print(" sample:", sample)
print(" test_image_dir:", test_image_dir)
print(" tf version:", tf.__version__)
print(" python:", sys.version.split()[0])



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1

USE_TFHUB = False
classifier = None



## === cell 2
sample_csv = pd.read_csv(sample)
assert list(sample_csv.columns) == [
    "image_id",
    "label",
], "Unexpected sample_submission.csv format"
print("Sample rows:", len(sample_csv))
sample_csv.head()



## === cell 3


def find_h5_models(root="/kaggle/input", max_models=8):
    h5_paths = []
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            if fn.lower().endswith((".h5", ".hdf5")):
                h5_paths.append(os.path.join(dirpath, fn))

    def score(p):
        name = os.path.basename(p).lower()
        s = 0
        if "bestmodel" in name or "best_model" in name:
            s += 3
        if "inception" in name or "goog" in name:
            s += 1
        return (-s, len(p))

    h5_paths = sorted(h5_paths, key=score)
    return h5_paths[:max_models], h5_paths


selected_h5, all_h5 = find_h5_models()
print(f"Found {len(all_h5)} .h5/.hdf5 files under /kaggle/input")
print("Selected for loading:")
for p in selected_h5:
    print(" -", p)



## === cell 4


def get_model_input_size(model):
    ish = model.input_shape
    if (
        isinstance(ish, (list, tuple))
        and len(ish) > 0
        and isinstance(ish[0], (list, tuple))
    ):
        ish = ish[0]
    if ish is None or len(ish) < 4:
        return (224, 224)
    h, w = ish[1], ish[2]
    if h is None or w is None:
        return (224, 224)
    return (int(h), int(w))


models = []
for path in selected_h5:
    try:
        m = load_model(path, compile=False)
        input_size = get_model_input_size(m)
        models.append((m, input_size))
        print(f"Loaded: {os.path.basename(path)}  input_size={input_size}")
    except Exception as e:
        print(f"Skipping model (failed to load) {path}: {type(e).__name__}: {e}")

if len(models) == 0:
    raise RuntimeError(
        "No .h5 models could be loaded from /kaggle/input. "
        "Attach at least one Keras model dataset to this notebook."
    )



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_11/235019189.py in <cell line: 0>()
     33     # Hard fail because we cannot generate meaningful predictions without any model.
     34     # (Still better than producing an invalid submission.)
---> 35     raise RuntimeError(
     36         "No .h5 models could be loaded from /kaggle/input. "
     37         "Attach at least one Keras model dataset to this notebook."

RuntimeError: No .h5 models could be loaded from /kaggle/input. Attach at least one Keras model dataset to this notebook.

## === cell 5

NUM_CLASSES = 5


def predict_one(model, input_size, image_path):
    img = load_img(image_path, target_size=input_size)
    x = np.expand_dims(img_to_array(img) / 255.0, axis=0)

    y = model.predict(x, verbose=0)
    if isinstance(y, tf.Tensor):
        y = y.numpy()
    y = np.asarray(y)

    if y.ndim == 1:
        probs = y
    else:
        probs = y[0]

    pred = int(np.argmax(probs))
    conf = float(probs[pred])
    return pred, conf


image_predictions = []
missing_images = 0

for image_id in sample_csv["image_id"].tolist():
    img_path = os.path.join(test_image_dir, image_id)
    if not os.path.exists(img_path):
        missing_images += 1
        image_predictions.append({"image_id": image_id, "label": 0})
        continue

    model_predictions = []
    confidence_scores = {}  # class -> list of confs

    for model, input_size in models:
        pred_class, conf = predict_one(model, input_size, img_path)

        if pred_class < 0 or pred_class >= NUM_CLASSES:
            pred_class = int(np.clip(pred_class, 0, NUM_CLASSES - 1))

        model_predictions.append(pred_class)
        confidence_scores.setdefault(pred_class, []).append(conf)

    class_votes = Counter(model_predictions)
    most_common = class_votes.most_common()

    final_predicted_class = most_common[0][0]

    if len(most_common) > 1 and most_common[0][1] == most_common[1][1]:
        top_count = most_common[0][1]
        tied_classes = [cls for cls, cnt in most_common if cnt == top_count]
        final_predicted_class = max(
            tied_classes,
            key=lambda cls: (
                sum(confidence_scores.get(cls, [0.0]))
                / max(1, len(confidence_scores.get(cls, [])))
            ),
        )

    image_predictions.append(
        {"image_id": image_id, "label": int(final_predicted_class)}
    )

print("Missing images:", missing_images)
print("Pred rows:", len(image_predictions))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/604077559.py in <cell line: 0>()
     61     most_common = class_votes.most_common()
     62 
---> 63     final_predicted_class = most_common[0][0]
     64 
     65     # Tie-break with average confidence among tied classes (same as original intent, but correct)

IndexError: list index out of range

## === cell 6
submission_df = pd.DataFrame(image_predictions)
submission_df["label"] = submission_df["label"].astype(int)
submission_df = submission_df.merge(
    sample_csv[["image_id"]], on="image_id", how="right"
)

submission_df["label"] = submission_df["label"].fillna(0).astype(int)

assert len(submission_df) == len(sample_csv), "Submission row count mismatch"
submission_df.to_csv(out_path, index=False)
print("Wrote:", out_path)
submission_df.head()



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2750174500.py in <cell line: 0>()
      2 submission_df = pd.DataFrame(image_predictions)
      3 # Ensure same order as sample and correct dtypes
----> 4 submission_df["label"] = submission_df["label"].astype(int)
      5 submission_df = submission_df.merge(
      6     sample_csv[["image_id"]], on="image_id", how="right"

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'label'

## === cell 7
print(submission_df.shape)
print(submission_df.dtypes)
print(submission_df["label"].value_counts().sort_index())
print("Unique image_ids:", submission_df["image_id"].nunique())

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/4186038569.py in <cell line: 0>()
      2 print(submission_df.shape)
      3 print(submission_df.dtypes)
----> 4 print(submission_df["label"].value_counts().sort_index())
      5 print("Unique image_ids:", submission_df["image_id"].nunique())

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/range.py in get_loc(self, key)
    415                 raise KeyError(key) from err
    416         if isinstance(key, Hashable):
--> 417             raise KeyError(key)
    418         self._check_indexing_error(key)
    419         raise KeyError(key)

KeyError: 'label'
