# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.7490178301601692

# 6. Current score

0.11398

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.05531) has done: 'I fix the TensorFlow/Keras import mismatch that triggers the `MessageFactory` / `GetPrototype` crash by using only `tf.keras` (and not mixing in standalone `keras`). Then I restore the missing `EfficientNetB0` symbol and make model creation deterministic and compatible with the provided pretrained weights path by loading weights only if the file exists. Finally, I ensure inference runs end-to-end and writes a valid `submission.csv` with exactly the required columns and test-set row alignment (using `sample_submission.csv` ordering to be safe).'
- What this solution (achieved 0.11622) has done: 'I fix the `MessageFactory/GetPrototype` crash by forcing TensorFlow to use its bundled protobuf implementation before importing TensorFlow (this is a known Kaggle/runtime mismatch issue). Then I keep your model/inference logic the same but switch EfficientNetB0 preprocessing to the correct `tf.keras.applications.efficientnet.preprocess_input`, because feeding only `/255.0` scaled pixels into an EfficientNet trained with standard preprocessing heavily tank accuracy (this should move your score toward the target without changing architecture/training loops). Finally, I keep the sample-submission ordering and ensure the output is a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.05605) has done: 'I fix the TensorFlow/protobuf crash by ensuring the protobuf implementation environment variables are set *before* any TensorFlow-related import and by forcing the pure-Python protobuf backend consistently. I also remove the remaining import pattern that can still trigger the `MessageFactory.GetPrototype` issue in some Kaggle runtimes by relying only on `tf.keras` symbols after importing TensorFlow. The rest of your logic (EfficientNetB0 architecture, weight loading-if-present, preprocessing, batching, argmax labels, and sample-submission ordering) be kept the same to preserve evaluation semantics while unblocking execution and producing a valid `submission.csv`.'
- What this solution (achieved 0.05717) has done: 'We fix the `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf backend *and* importing `google.protobuf` before TensorFlow, which is the remaining ordering issue in your current cell 0. Then we keep your EfficientNetB0 model, weight-loading logic, preprocessing, and inference loop intact, only adjusting imports to avoid any standalone `keras` usage. Finally, we make sure the script reaches the submission-writing step and produces `submission.csv` with the required `image_id,label` columns in the exact sample-submission order.'
- What this solution (achieved 0.11398) has done: 'We fix the `MessageFactory.GetPrototype` crash by pinning protobuf to the pure-Python backend *and* forcing a compatible protobuf version early (this is the typical root cause on Kaggle when TF + protobuf get out of sync). Then we correct the weights path to point to an existing Kaggle input location (your current `../input/weights/...` likely doesn’t exist), so the model can actually load pretrained weights instead of using random weights (this is the main reason your accuracy is near-random and far from the 0.749 target). Finally, we keep your EfficientNetB0 architecture, preprocessing, and inference loop intact and still write a valid `submission.csv` aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ["PYTHONHASHSEED"] = "0"

import sys
import subprocess

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"],
    check=False,
)

import google.protobuf  # noqa: F401

import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(0)
tf.random.set_seed(0)

IMG_SIZE = 512
NB_CHANNELS = 3
BATCH_SIZE = 32

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")




## === cell 1
layers = tf.keras.layers
Sequential = tf.keras.models.Sequential
EfficientNetB0 = tf.keras.applications.EfficientNetB0
preprocess_input = tf.keras.applications.efficientnet.preprocess_input

cnn_base = EfficientNetB0(
    include_top=False,
    weights=None,
    input_shape=(IMG_SIZE, IMG_SIZE, NB_CHANNELS),
)
cnn = Sequential(
    [
        cnn_base,
        layers.GlobalAveragePooling2D(),
        layers.Dense(5, activation="softmax"),
    ]
)
cnn.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer="adam",
    metrics=["accuracy"],
)




## === cell 2
CANDIDATE_WEIGHT_PATHS = [
    "/kaggle/input/weights/EffNetB0_512_8_best_weights.h5",
    "/kaggle/input/cassava-leaf-disease-classification/EffNetB0_512_8_best_weights.h5",
    "/kaggle/input/EffNetB0_512_8_best_weights.h5",
    "../input/weights/EffNetB0_512_8_best_weights.h5",  # keep your original as a fallback
]

found_path = None
for p in CANDIDATE_WEIGHT_PATHS:
    if os.path.exists(p):
        found_path = p
        break

if found_path is None:
    for root, _, files in os.walk("/kaggle/input"):
        if "EffNetB0_512_8_best_weights.h5" in files:
            found_path = os.path.join(root, "EffNetB0_512_8_best_weights.h5")
            break

if found_path is not None:
    cnn.load_weights(found_path)
    print(f"Loaded weights from: {found_path}")
else:
    print(
        "WARNING: Weights file 'EffNetB0_512_8_best_weights.h5' not found under /kaggle/input. "
        "Proceeding with random weights (score will be near-random)."
    )




## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
test_image_ids = sample_sub["image_id"].tolist()

missing = [
    img_id
    for img_id in test_image_ids
    if not os.path.exists(os.path.join(TEST_IMG_DIR, img_id))
]
if len(missing) > 0:
    raise FileNotFoundError(
        f"{len(missing)} test images listed in sample_submission are missing. Example: {missing[0]}"
    )




## === cell 4
predictions = []


def load_and_preprocess(path):
    img = tf.keras.preprocessing.image.load_img(path, target_size=(IMG_SIZE, IMG_SIZE))
    arr = tf.keras.preprocessing.image.img_to_array(img)  # float32 in [0,255]
    arr = preprocess_input(arr)  # EfficientNet standard preprocessing
    return arr


batch = []
batch_ids = []

for img_id in test_image_ids:
    img_path = os.path.join(TEST_IMG_DIR, img_id)
    batch.append(load_and_preprocess(img_path))
    batch_ids.append(img_id)

    if len(batch) == BATCH_SIZE:
        x = np.stack(batch, axis=0)
        probs = cnn.predict(x, verbose=0)
        preds = np.argmax(probs, axis=1).astype(int).tolist()
        predictions.extend(preds)
        batch, batch_ids = [], []

if len(batch) > 0:
    x = np.stack(batch, axis=0)
    probs = cnn.predict(x, verbose=0)
    preds = np.argmax(probs, axis=1).astype(int).tolist()
    predictions.extend(preds)

if len(predictions) != len(test_image_ids):
    raise RuntimeError(
        f"Prediction count mismatch: got {len(predictions)} preds for {len(test_image_ids)} images"
    )

submission = pd.DataFrame({"image_id": test_image_ids, "label": predictions})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with", len(submission), "rows")
