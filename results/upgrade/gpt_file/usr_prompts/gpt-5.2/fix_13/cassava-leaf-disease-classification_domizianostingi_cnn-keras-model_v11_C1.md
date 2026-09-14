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

0.61099

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the environment import crash by switching from standalone `keras` to `tf_keras`, which avoids the protobuf `MessageFactory.GetPrototype` error in this Kaggle image setup. I also fix the test image loading bug by reading `sample_submission.csv` and generating predictions for all test images instead of a single hardcoded filename that doesn’t exist. To keep the core model/training logic intact, I only add minimal preprocessing to ensure inputs are float32 and labels have the correct one-hot shape for `categorical_crossentropy`. Finally, I ensure a valid `submission.csv` is written with the required `image_id,label` columns and correct row alignment.'
- What this solution (achieved 0.61099) has done: 'I fix the import crash by switching from `tf_keras` to `tensorflow.keras`, which is the most stable API in this Kaggle environment and avoids the protobuf `MessageFactory.GetPrototype` issue. I keep the model architecture, preprocessing, and training approach identical, but make the import path and loss reference compatible so the notebook runs end-to-end. Because your current score (0.61099) is far above the target (0.1403) and higher-is-better, I won’t make any changes intended to improve performance; the only changes are to restore execution and produce a valid `submission.csv`. The submission writing and alignment with `sample_submission.csv` are preserved.'
- What this solution (achieved 0.61099) has done: 'The crash happens before any training because this Kaggle image has an incompatibility between the bundled `tensorflow`/protobuf stack and importing `tensorflow.keras`. The minimal robust fix is to switch the Keras imports to `tf_keras` (which is installed and avoids the protobuf `MessageFactory.GetPrototype` error here) while keeping the exact same model, loss, and training loop. Since your current score (0.61099) is far above the target (0.1403) and higher-is-better, I’m not making any score-improving changes—only restoring end-to-end execution and submission generation. The data loading and submission alignment with `sample_submission.csv` remain unchanged.'
- What this solution (achieved 0.61099) has done: 'I fix the runtime import crash (`MessageFactory.GetPrototype`) by avoiding the TensorFlow/protobuf code path that triggers it in this Kaggle image: we use the stable `tf_keras` package without importing `tensorflow` at module import time. To keep core modeling/training logic unchanged, the model architecture, loss, optimizer, fit call, preprocessing, and submission generation stay the same. Since your current score (0.61099) is far above the target (0.1403) and higher-is-better, I won’t apply any score-improving changes—only make the minimal compatibility fix so it runs end-to-end and writes a valid `submission.csv`. I also keep determinism seeding via NumPy and (when available) `tf_keras.utils.set_random_seed`.'
- What this solution (achieved 0.61099) has done: 'We fix the crash in the very first cell by avoiding the protobuf code path that `tf_keras` triggers in this Kaggle image (the `MessageFactory.GetPrototype` AttributeError). The smallest robust fix is to switch imports to the standalone `keras` package (Keras 3 is installed here) and keep the exact same Sequential CNN, loss, optimizer, and training/prediction logic. Since your current score (0.61099) is already far above the target (0.1403) and higher-is-better, I not make any performance-improving changes; the goal is only to restore end-to-end execution and a valid `submission.csv`. The submission generation remains aligned to `sample_submission.csv` order and preserves the required columns.'
- What this solution (achieved 0.61099) has done: 'I fix the immediate crash caused by importing standalone `keras` in this Kaggle image (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to the already-installed `tf_keras` API, keeping the same Sequential CNN, loss, optimizer, and training/prediction flow. I also correct the cell numbering to start at 1 (your current script starts at cell 0) so it matches the required format. The rest of the pipeline (loading a small training subset, predicting on all test images in `sample_submission.csv` order, and writing `submission.csv`) remain unchanged and score-neutral aside from negligible backend differences.'
- What this solution (achieved 0.61099) has done: 'We fix the crash in the first cell caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` incompatibility) by switching to the stable `tensorflow.keras` API, while keeping the exact same CNN architecture, loss, optimizer, and training/prediction flow. We also renumber cells to start at 1 to match the required format, without changing execution order. Finally, we keep submission generation aligned to `sample_submission.csv` and ensure `submission.csv` is written with the required columns and row count; no score-improving changes are introduced (your current score is already far above the target, and higher-is-better).'
- What this solution (achieved 0.61099) has done: 'I fix the runtime import crash by avoiding `tensorflow`/`tensorflow.keras`, which is triggering the protobuf `MessageFactory.GetPrototype` error in this environment. To preserve your core CNN architecture, training loop, preprocessing, and submission logic, I switch Keras imports to the already-installed `tf_keras` package and keep the rest of the code the same. Since your current score (0.61099) is far above the target (0.1403) and higher-is-better, I not make any changes intended to improve performance—only ensure the notebook runs end-to-end and writes a valid `submission.csv`. I also renumber cells to start at 1 (your script currently starts at cell 0) to match the required format.'
- What this solution (achieved 0.61099) has done: 'We fix the immediate runtime crash in the first cell caused by importing `tf_keras` (protobuf `MessageFactory.GetPrototype` issue in this environment) by switching to the installed standalone `keras` (Keras 3) API, while keeping the exact same model architecture, loss, optimizer, training call, preprocessing, and submission generation. We also renumber cells to start at 1 as required, without changing execution order. No score-improving changes are introduced (your current score is already far above the target and higher-is-better); this is strictly a stability/compatibility fix to run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the import-time protobuf crash by avoiding the standalone `keras` package (which triggers `MessageFactory.GetPrototype` here) and switching to the already-installed `tf_keras` API while keeping the exact same CNN, loss, optimizer, training call, and prediction logic. I also renumber the cells to start at 1 (your current script starts at cell 0) without changing execution order. Since your current score (0.61099) is far above the target (0.1403) and higher-is-better, I not introduce any performance-improving changes—this is strictly an execution/stability fix that still produces a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the import-time protobuf crash that occurs when importing `tf_keras` in this Kaggle environment by switching the Keras imports to the installed standalone `keras` (Keras 3) API, keeping the exact same model architecture, compilation, training call, and prediction logic. I also renumber the cells to start at 1 (your current script starts at cell 0) to match the required format without changing execution order. Since your current score (0.61099) is far above the target (0.1403) and higher-is-better, I won’t make any changes intended to improve performance; the goal is strictly to run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 0.61099) has done: 'I fix the import-time crash (`MessageFactory.GetPrototype`) by switching from standalone `keras` to the installed `tf_keras`, which avoids the protobuf incompatibility in this environment while keeping the exact same model, loss, and training/prediction flow. I also renumber the cells to start at 1 (your current script starts at cell 0) to match the required format and ensure execution proceeds in order. No changes are made to improve performance (your current score is far above the target and higher-is-better); the goal is stability and producing a valid `submission.csv`. The data loading remains the same, including correct test-set ordering from `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os
import glob
import gc
import numpy as np
import pandas as pd
from PIL import Image

import tf_keras as keras
from tf_keras.models import Sequential
from tf_keras.layers import Dense, Dropout, Flatten
from tf_keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tf_keras.losses import categorical_crossentropy

os.environ["PYTHONHASHSEED"] = "0"
np.random.seed(0)
try:
    keras.utils.set_random_seed(0)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
size = (300, 200)

INPUT_DIR = "../input/cassava-leaf-disease-classification"
TRAIN_IMG_DIR = os.path.join(INPUT_DIR, "train_images")
TEST_IMG_DIR = os.path.join(INPUT_DIR, "test_images")
TRAIN_CSV = os.path.join(INPUT_DIR, "train.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")


def prepare(data, n_train=50):
    """
    Minimal fixes preserved:
    - train: load up to n_train images like original code, but ensure consistent label one-hot shape (5 classes).
    - test: load all test images in the exact order of sample_submission for correct alignment.
    """
    if data == "train":
        frag = sorted(glob.glob(os.path.join(TRAIN_IMG_DIR, "*.jpg")))
        train = pd.read_csv(TRAIN_CSV)

        label_map = dict(zip(train["image_id"].values, train["label"].values))

        arrays = []
        y_ = []

        n = min(n_train, len(frag))
        for i in range(n):
            image_id = os.path.basename(frag[i])
            if image_id not in label_map:
                continue

            y_.append(int(label_map[image_id]))
            image = Image.open(frag[i]).convert("RGB")
            image = image.resize(size)
            image = np.array(image)
            arrays.append(image)

        y_oh = pd.get_dummies(pd.Series(y_), dtype=np.uint8)
        y_oh = y_oh.reindex(columns=[0, 1, 2, 3, 4], fill_value=0)
        y_oh = y_oh.values.astype("float32")

        return arrays, y_oh

    if data == "test":
        sub = pd.read_csv(SAMPLE_SUB)
        arrays = []
        for image_id in sub["image_id"].values:
            img_path = os.path.join(TEST_IMG_DIR, image_id)
            image = Image.open(img_path).convert("RGB")
            image = image.resize(size)
            image = np.array(image)
            arrays.append(image)
        return np.array(arrays)

    raise ValueError("data must be 'train' or 'test'")




## === cell 2
x, y = prepare("train", n_train=50)

x = np.array(x, dtype="float32")
x = x / 255.0

gc.collect()



## === cell 3
num_features = 32
num_labels = 5

model = Sequential()

model.add(
    Conv2D(
        num_features, kernel_size=(3, 3), activation="relu", input_shape=(200, 300, 3)
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

model.compile(loss=categorical_crossentropy, optimizer="adam", metrics=["accuracy"])



## === cell 4
model.fit(x, y, verbose=2)



## === cell 5
test = prepare("test").astype("float32") / 255.0
valore = model.predict(test, verbose=0)




## === cell 6
def sub(valore):
    preds = pd.DataFrame(valore).idxmax(axis=1).astype(int).values

    submission = pd.read_csv(SAMPLE_SUB)
    submission["label"] = preds
    submission.to_csv("submission.csv", index=False)
    return submission


submission = sub(valore)
print(submission.head())
print("Finished! Wrote submission.csv with", len(submission), "rows.")



## === cell 7
assert os.path.exists("submission.csv")
_check = pd.read_csv("submission.csv")
assert list(_check.columns) == ["image_id", "label"]
assert len(_check) == 2676
print("submission.csv is valid.")
