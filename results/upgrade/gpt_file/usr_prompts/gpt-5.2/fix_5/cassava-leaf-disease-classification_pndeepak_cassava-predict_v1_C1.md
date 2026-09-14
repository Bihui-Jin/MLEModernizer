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
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.8768510123904503

# 6. Current score

0.05755

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.13229) has done: 'I fix the runtime crash caused by an incompatible `protobuf` version (the `MessageFactory.GetPrototype` error) by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I keep your exact model architecture/inference logic, but make the image preprocessing deterministic and consistent with EfficientNet (convert to RGB, resize to 300×300, and apply `efficientnet.preprocess_input`) to improve accuracy toward your target without changing the core approach. Finally, I batch predictions for speed and ensure a correctly formatted `submission.csv` is written.'
- What this solution (achieved 0.10052) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* disabling the C++ implementation before any TensorFlow import, which avoids the `MessageFactory.GetPrototype` issue in this environment. Next, I fix the missing weights error by removing the external weights dependency and instead use EfficientNetB0’s built-in `imagenet` weights (same architecture), which should move accuracy substantially toward your target. Finally, I keep your inference logic the same but make sure preprocessing matches EfficientNet expectations and that a correctly formatted `submission.csv` is always written.'
- What this solution (achieved 0.19806) has done: 'I fix the protobuf/TensorFlow import crash by removing the incompatible pure-Python protobuf forcing and instead forcing TensorFlow to use the Python protobuf implementation via `TF_PROTOBUF_IMPLEMENTATION=python` before importing TensorFlow (this avoids the `MessageFactory.GetPrototype` failure under protobuf 6.x). I keep your exact model and inference logic unchanged, only adding a robust fallback that searches the known dataset locations so the notebook runs regardless of whether Kaggle mounts data under `/kaggle/input` or `../input`. Finally, I ensure the submission is always written as `submission.csv` with the required columns and row order matching `sample_submission.csv`.'
- What this solution (achieved 0.05755) has done: 'I fix the TensorFlow/protobuf crash by forcing protobuf to use the pure-Python backend *and* disabling the C++ implementation before TensorFlow is imported, which avoids the `MessageFactory.GetPrototype` error under protobuf 6.x. Then I keep your model architecture and inference logic the same, only ensuring the environment variables are set early enough and adding a small safety check that all test image paths exist (so it fails fast with a clear message instead of crashing mid-loop). This should restore end-to-end execution and produce a valid `submission.csv`. Any score change come only from the code actually running correctly with the same preprocessing/model logic.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_PROTOBUF_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras

tf.random.set_seed(42)
np.random.seed(42)

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
model = keras.models.Sequential()
model.add(
    keras.applications.EfficientNetB0(
        input_shape=(300, 300, 3), weights="imagenet", include_top=False
    )
)
model.add(keras.layers.GlobalAveragePooling2D())
model.add(keras.layers.Dense(5, activation="softmax"))

model.trainable = False




## === cell 2
def find_existing_path(candidates):
    for p in candidates:
        if os.path.exists(p):
            return p
    raise FileNotFoundError(
        "None of the candidate paths exist:\n" + "\n".join(candidates)
    )


sample_csv = find_existing_path(
    [
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/sample_submission.csv",
        "/kaggle/data/sample_submission.csv",
    ]
)

test_dir = find_existing_path(
    [
        "../input/cassava-leaf-disease-classification/test_images/",
        "/kaggle/input/cassava-leaf-disease-classification/test_images/",
        "/kaggle/data/cassava-leaf-disease-classification/test_images/",
        "/kaggle/input/test_images/",
        "/kaggle/data/test_images/",
    ]
)

ss = pd.read_csv(sample_csv)
image_paths = (test_dir.rstrip("/") + "/" + ss["image_id"]).tolist()

missing = [p for p in image_paths[:50] if not os.path.exists(p)]
if missing:
    raise FileNotFoundError(
        "Some test image paths do not exist (showing up to 5):\n"
        + "\n".join(missing[:5])
    )

preprocess = keras.applications.efficientnet.preprocess_input


def load_and_preprocess(path):
    img = keras.preprocessing.image.load_img(
        path, color_mode="rgb", target_size=(300, 300)
    )
    arr = keras.preprocessing.image.img_to_array(img)
    arr = preprocess(arr)
    return arr




## === cell 3
batch_size = 32
n = len(image_paths)
preds = []

for start in range(0, n, batch_size):
    batch_paths = image_paths[start : start + batch_size]
    batch = np.stack([load_and_preprocess(p) for p in batch_paths], axis=0)
    prob = model.predict(batch, verbose=0)
    preds.extend(np.argmax(prob, axis=1).astype(int).tolist())

my_submission = pd.DataFrame({"image_id": ss["image_id"], "label": preds})
my_submission.to_csv("submission.csv", index=False)

assert list(my_submission.columns) == ["image_id", "label"]
assert len(my_submission) == len(ss)
print("Wrote submission.csv with", len(my_submission), "rows")
print(my_submission.head())
