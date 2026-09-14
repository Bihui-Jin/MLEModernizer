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

0.8112722877002115

# 6. Current score

0.31876

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.48169) has done: 'I remove the imports that trigger the protobuf/TF-Hub `MessageFactory` crash and also avoid relying on missing Kaggle inputs (`kerasapplication`, `efficientnet`, and the external `.h5` weight path), which is why `my_model` was never created. To preserve the same “load-a-pretrained-EfficientNet-and-predict” core approach, I build an EfficientNet model from `tf.keras.applications` with ImageNet weights and run deterministic inference on the test images. I also fix the input path to the dataset to match your provided filesystem (`/kaggle/input/...`) and ensure the submission is written as `submission.csv` with the exact required columns. This run end-to-end and produce a valid submission file.'
- What this solution (achieved 0.53363) has done: 'The crash happens before any model code runs because TensorFlow’s protobuf dependency is hitting the known `MessageFactory.GetPrototype` incompatibility in this environment. The smallest reliable fix is to force the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that missing attribute path and lets the rest of your pipeline run unchanged. I’m keeping your core approach (ImageNet EfficientNetB3 frozen backbone + softmax head, deterministic inference, same preprocessing and submission merge) and only adjusting import order/env to unblock execution. This should also improve score vs the current broken/unstable state by ensuring consistent, valid predictions are produced end-to-end.'
- What this solution (achieved 0.10949) has done: 'We fix the protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation before TensorFlow (and any protobuf-using libs) import; this is the minimal reliable unblocking change for this Kaggle environment. To move accuracy toward your target without changing the core “EfficientNetB3 ImageNet backbone + softmax head, no training” approach, we add the standard EfficientNet classification head structure (Dropout + Dense) and use the correct `EfficientNetB3.preprocess_input`, which together typically improves zero-shot transfer performance versus a single dense layer on raw pooled features. We also ensure the test generator uses the same preprocessing function and keep submission formatting/merging unchanged so a valid `submission.csv` is always produced. All other logic (data loading, deterministic inference, argmax labels, and writing CSV) remains the same.'
- What this solution (achieved 0.08445) has done: 'The crash happens before any model code runs due to the known protobuf/TensorFlow incompatibility (`MessageFactory.GetPrototype`). I fix this by forcing the pure-Python protobuf runtime *and* ensuring it’s applied before TensorFlow import, then clearing any pre-imported `google.protobuf` modules that could keep the bad implementation loaded. This is a minimal, execution-unblocking change that preserves your core approach (ImageNet EfficientNetB3 backbone + dropout + dense softmax head, no training, same preprocessing/inference/submission logic). I also keep paths and submission formatting unchanged so `submission.csv` is always produced.'
- What this solution (achieved 0.2657) has done: 'You’re still hitting the protobuf `MessageFactory.GetPrototype` crash because TensorFlow is being imported after protobuf modules have already been loaded with the incompatible implementation; setting env vars alone isn’t reliably taking effect in this environment. I make the smallest execution-unblocking change: force the pure-Python protobuf implementation *and* prevent the C++ one via `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, and do it before any possible protobuf import, without trying to mutate `sys.modules` (which is brittle here). I also align the preprocessing function to the specific EfficientNetB3 variant (`tf.keras.applications.efficientnet.preprocess_input` is fine, but we explicitly use the one tied to EfficientNetB3 to avoid version differences), and keep the rest of your logic (no training, same architecture, same argmax submission) unchanged. This should both run end-to-end and raise the score substantially versus the current 0.08445 which is consistent with a broken/incoherent inference run.'
- What this solution (achieved 0.18161) has done: 'I fix the protobuf/TensorFlow `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *and* removing any already-imported protobuf modules before importing TensorFlow (env vars alone aren’t reliably applied if protobuf was imported earlier). This is the minimal change that unblocks execution while preserving your core “ImageNet EfficientNetB3 backbone + dropout + dense softmax head, no training” approach and the same preprocessing/inference/submission semantics. I also make the EfficientNet preprocessing call match the model family explicitly and ensure the test generator is recreated consistently, but won’t change architecture or training behavior. The script then run end-to-end and always write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.07623) has done: 'You’re crashing before any model code runs due to the known TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`) in this environment, and the current workaround (env vars + clearing `sys.modules`) isn’t reliably taking effect. I make the smallest execution-unblocking change by forcing the pure-Python protobuf implementation at process start (and avoiding the brittle module-popping), then import TensorFlow; this keeps your EfficientNetB3 ImageNet inference approach unchanged. I also align the preprocessing function specifically to EfficientNetB3 (instead of the generic efficientnet module) to avoid version mismatches that can silently hurt accuracy, while keeping the same model/head and argmax submission semantics. The script still write a valid `submission.csv` with `image_id,label`.'
- What this solution (achieved 0.11883) has done: 'I fix the TensorFlow/protobuf `MessageFactory.GetPrototype` crash by forcing the pure-Python protobuf runtime *before anything can import protobuf*, and by isolating the TensorFlow import into a function that runs after the env vars are set (this is the minimal reliable unblock in Kaggle). I also make the preprocessing function match the exact EfficientNetB3 family (`tf.keras.applications.efficientnet.preprocess_input`) and keep your model architecture/inference logic unchanged. Finally, I ensure the test dataframe uses only filenames (not full paths) so `flow_from_dataframe(..., directory=...)` is stable and the merged submission aligns perfectly with `sample_submission.csv`. This should run end-to-end and, because previous 0.07 indicates broken/garbled inference, it should move accuracy back up toward your target without changing the core approach.'
- What this solution (achieved 0.31876) has done: 'The current crash happens before any model code runs due to the TensorFlow/protobuf incompatibility (`MessageFactory.GetPrototype`). The minimal robust fix in Kaggle is to force the pure-Python protobuf runtime *and* prevent the C++ one (`PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python`, without forcing version "2"), plus clear any already-imported `google.protobuf*` modules before importing TensorFlow. Once TensorFlow imports cleanly, your existing EfficientNetB3 ImageNet backbone + dropout + dense softmax head inference pipeline run unchanged and should substantially improve score versus the current broken run. I also keep the submission alignment/merge logic intact and ensure `submission.csv` is written.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import sys
import glob
import numpy as np
import pandas as pd

SEED = 42
DEBUG = False

BASE_INPUT = "/kaggle/input/cassava-leaf-disease-classification"
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
TEST_GLOB = os.path.join(TEST_DIR, "*.jpg")
SAMPLE_SUB_PATH = os.path.join(BASE_INPUT, "sample_submission.csv")

assert os.path.exists(BASE_INPUT), f"Dataset base path not found: {BASE_INPUT}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"sample_submission.csv not found: {SAMPLE_SUB_PATH}"
assert os.path.exists(TEST_DIR), f"test_images dir not found: {TEST_DIR}"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        sys.modules.pop(m, None)


def import_tf():
    import tensorflow as tf  # noqa: F401

    return tf


tf = import_tf()
tf.random.set_seed(SEED)
np.random.seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras import layers, Model

IMG_SIZE = (512, 512)
NUM_CLASSES = 5

base = EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = base(inputs, training=False)
x = layers.Dropout(0.2)(x)
outputs = layers.Dense(NUM_CLASSES, activation="softmax")(x)
my_model = Model(inputs, outputs)

_ = my_model(tf.zeros((1, IMG_SIZE[0], IMG_SIZE[1], 3), dtype=tf.float32))



## === cell 2
test_images = sorted(glob.glob(TEST_GLOB))
if len(test_images) == 0:
    raise FileNotFoundError(f"No test images found with glob: {TEST_GLOB}")

df_test = pd.DataFrame({"image_id": [os.path.basename(p) for p in test_images]})


def make_test_gen(batch_size=64):
    preprocess_input = tf.keras.applications.efficientnet.preprocess_input
    my_test_idg = ImageDataGenerator(preprocessing_function=preprocess_input)
    test_gen = my_test_idg.flow_from_dataframe(
        dataframe=df_test,
        directory=TEST_DIR,
        x_col="image_id",
        y_col=None,
        batch_size=batch_size,
        seed=SEED,
        shuffle=False,
        class_mode=None,
        target_size=IMG_SIZE,
    )
    return test_gen




## === cell 3
pred_list = []

for i in range(1):
    test_gen = make_test_gen(batch_size=128)
    pred_test = my_model.predict(test_gen, verbose=True)
    pred_list.append(pred_test)

pred_test = np.mean(pred_list, axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["label"] = pred_test_labels
final_csv = final_submission[["image_id", "label"]]

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
final_csv = sample_sub[["image_id"]].merge(final_csv, on="image_id", how="left")

if final_csv["label"].isna().any():
    fill_label = int(pd.Series(pred_test_labels).mode().iloc[0])
    final_csv["label"] = final_csv["label"].fillna(fill_label).astype(int)
else:
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)



## === cell 4
final_csv.head()
