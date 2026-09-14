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

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0

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

0.1403

# 6. Current score

0.61024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.61024) has done: 'I updated the imports to use the standalone keras package (avoiding the TensorFlow‑only import that caused the `MessageFactory` error) and expanded the data‑search paths to include the usual Kaggle input directory (`/kaggle/input/...`). The submission helper now points to the correct location of sample_submission.csv. These minimal fixes let the notebook run end‑to‑end, produce predictions, and write a valid `submission.csv` that should meet the modest target accuracy.'

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
from PIL import Image
import glob
import os
import matplotlib.pyplot as plt
import gc

from keras.models import Sequential
from keras.layers import (
    Dense,
    Dropout,
    Flatten,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.losses import CategoricalCrossentropy
from keras.utils import to_categorical



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
size = (300, 200)


def prepare(data):
    """
    Loads images and labels.
    - For 'train' returns (list_of_arrays, one‑hot labels)
    - For 'test' returns a list of image arrays and the corresponding image_id list
    """
    base_path = os.path.abspath(".")
    possible_roots = [
        os.path.join(
            base_path, "kaggle", "data", "cassava-leaf-disease-classification"
        ),
        os.path.join(base_path, "input", "cassava-leaf-disease-classification"),
        os.path.join(base_path, "data", "cassava-leaf-disease-classification"),
        "/kaggle/input/cassava-leaf-disease-classification",  # typical Kaggle path
    ]
    root = None
    for p in possible_roots:
        if os.path.isdir(p):
            root = p
            break
    if root is None:
        raise FileNotFoundError(
            "Could not locate cassava‑leaf‑disease‑classification data directory."
        )

    if data == "train":
        img_paths = sorted(glob.glob(os.path.join(root, "train_images", "*.jpg")))
        train_df = pd.read_csv(os.path.join(root, "train.csv"))
        img_paths = img_paths[:200]  # small subset for quick runs

        arrays = []
        labels = []
        for img_path in img_paths:
            img_id = os.path.basename(img_path)
            label = train_df.loc[train_df["image_id"] == img_id, "label"].values[0]
            labels.append(label)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            arrays.append(np.array(img))

        y_onehot = to_categorical(labels, num_classes=5)
        return arrays, y_onehot

    elif data == "test":
        img_paths = sorted(glob.glob(os.path.join(root, "test_images", "*.jpg")))
        arrays = []
        ids = []
        for img_path in img_paths:
            img_id = os.path.basename(img_path)
            ids.append(img_id)
            img = Image.open(img_path).convert("RGB")
            img = img.resize(size)
            arrays.append(np.array(img))
        return arrays, ids

    else:
        raise ValueError("data argument must be 'train' or 'test'")




## === cell 2
x_train_list, y_train = prepare("train")




## === cell 3
x_train = np.array(x_train_list, dtype=np.float32) / 255.0  # scale to [0,1]




## === cell 4
gc.collect()




## === cell 5
num_features = 32
num_labels = 5

model = Sequential()
model.add(
    Conv2D(
        num_features,
        kernel_size=(3, 3),
        activation="relu",
        input_shape=(size[0], size[1], 3),
    )
)
model.add(Conv2D(num_features, kernel_size=(3, 3), activation="relu", padding="same"))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.3))
model.add(Flatten())
model.add(Dense(2 * 2 * 2 * num_features, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2 * 2 * num_features, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(2 * num_features, activation="relu"))
model.add(Dropout(0.3))
model.add(Dense(num_labels, activation="softmax"))

model.compile(loss=CategoricalCrossentropy(), optimizer="adam", metrics=["accuracy"])




## === cell 6
model.fit(x_train, y_train, epochs=5, batch_size=16, verbose=2)




## === cell 7
x_test_list, test_ids = prepare("test")
x_test = np.array(x_test_list, dtype=np.float32) / 255.0




## === cell 8
test_pred_probs = model.predict(x_test, batch_size=16, verbose=0)




## === cell 9
def create_submission(probs, ids):
    """
    probs: numpy array (N, 5) of predicted probabilities
    ids: list of image_id strings aligned with probs
    """
    pred_labels = np.argmax(probs, axis=1).astype(int)
    possible_paths = [
        os.path.join(
            os.path.abspath("."),
            "kaggle",
            "data",
            "cassava-leaf-disease-classification",
            "sample_submission.csv",
        ),
        os.path.join(
            os.path.abspath("."),
            "input",
            "cassava-leaf-disease-classification",
            "sample_submission.csv",
        ),
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
    ]
    sample_path = None
    for p in possible_paths:
        if os.path.isfile(p):
            sample_path = p
            break
    if sample_path is None:
        raise FileNotFoundError("sample_submission.csv not found.")
    submission_df = pd.read_csv(sample_path)
    id_to_pred = dict(zip(ids, pred_labels))
    submission_df["label"] = (
        submission_df["image_id"].map(id_to_pred).fillna(0).astype(int)
    )
    submission_path = "submission.csv"
    submission_df.to_csv(submission_path, index=False)
    print(f"Submission written to {submission_path}")
    print(submission_df.head())




## === cell 10
create_submission(test_pred_probs, test_ids)
