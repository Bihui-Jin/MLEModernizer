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

0.8428528256270776

# 6. Current score

0.08259

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.16293) has done: 'I fix the early import crash by forcing TensorFlow to use the Python protobuf implementation (this resolves the `MessageFactory.GetPrototype` AttributeError seen at import time). Next, I remove the hard dependency on an external `.h5` model file that is not present in your environment by switching to a small built-in Keras model (so the notebook runs end-to-end and always writes `submission.csv`). Finally, I keep the same inference/data-loading flow and submission formatting, ensuring the predictions align with `sample_submission.csv` ordering and the output has the required `image_id,label` columns.'
- What this solution (achieved 0.14985) has done: 'I fix the TensorFlow import crash by setting the protobuf environment variables *before* TensorFlow (or anything that imports protobuf via TF) is imported, and I also force the pure‑Python protobuf backend to avoid the `MessageFactory.GetPrototype` error. Next, I correct the fallback model so its preprocessing matches EfficientNet expectations (this is score-improving but still preserves the same overall “single model → predict → argmax” inference semantics). Finally, I ensure the generator produces properly rescaled float images and that the submission aligns exactly to `sample_submission.csv` ordering and writes `submission.csv` successfully.'
- What this solution (achieved 0.08259) has done: 'The crash happens at TensorFlow import time due to an incompatibility between the installed TensorFlow build and the protobuf runtime (the `MessageFactory.GetPrototype` attribute issue). The most reliable minimal fix in Kaggle notebooks is to force the pure-Python protobuf backend *and* disable C-descriptors before anything imports protobuf/TF. After that, your pipeline can run end-to-end unchanged: locate test images, build/load the model, predict, and write a properly formatted `submission.csv`. I’m keeping the modeling/inference semantics the same; the main change is just the import-time environment setup to unblock execution so you can get a valid submission and improve score beyond the “no-TF” failure mode.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_DISABLE_C_DESCRIPTORS", "1")

import glob
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator

SEED = 42
DEBUG = False

tf.random.set_seed(SEED)
np.random.seed(SEED)


def first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


print("Imports OK. TF:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
print(
    "Skipping external wheel installs (not required for inference in this environment)."
)



## === cell 2
weight_path = first_existing(
    [
        "/kaggle/input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
        "../input/model-ensembling-with-k-fold/fineTuned_v0.59.h5",
    ]
)

if weight_path is not None:
    my_model = load_model(weight_path, compile=False)
    print("Loaded model from:", weight_path)
    USING_FALLBACK = False
else:
    print("WARNING: Could not find fineTuned_v0.59.h5 in expected input locations.")
    print("Falling back to a built-in model to generate a valid submission.")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(512, 512, 3),
        pooling="avg",
    )
    inp = tf.keras.Input(shape=(512, 512, 3))
    x = base(inp, training=False)
    out = tf.keras.layers.Dense(5, activation="softmax")(x)
    my_model = tf.keras.Model(inp, out, name="fallback_effnetb0_5class")
    USING_FALLBACK = True
    print("Built fallback model:", my_model.name)



## === cell 3
test_img_dir = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/test_images",
        "../input/cassava-leaf-disease-classification/test_images",
        "/kaggle/data/cassava-leaf-disease-classification/test_images",
        "../data/cassava-leaf-disease-classification/test_images",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/test_images",
    ]
)

if test_img_dir is None:
    raise FileNotFoundError(
        "Could not find test_images directory in expected locations."
    )

test_images = sorted(glob.glob(os.path.join(test_img_dir, "*.jpg")))
if len(test_images) == 0:
    raise RuntimeError(f"No .jpg files found under: {test_img_dir}")

df_test = pd.DataFrame({"path": test_images})


def make_test_gen(batch_size=16):
    if USING_FALLBACK:
        preprocessing_fn = tf.keras.applications.efficientnet.preprocess_input
        my_test_idg = ImageDataGenerator(preprocessing_function=preprocessing_fn)
    else:
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


print("Test images:", len(df_test))



## === cell 4
pred_list = []

for i in range(1):
    test_gen = make_test_gen(batch_size=16)
    pred_test = my_model.predict(test_gen, verbose=1)
    pred_list.append(pred_test)

pred_test = np.mean(np.stack(pred_list, axis=0), axis=0)
pred_test_labels = np.argmax(pred_test, axis=-1).astype(int)

final_submission = df_test.copy()
final_submission["image_id"] = final_submission["path"].str.split("/").str[-1]
final_submission["label"] = pred_test_labels

final_csv = final_submission[["image_id", "label"]].copy()

sample_path = first_existing(
    [
        "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv",
        "../input/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/data/cassava-leaf-disease-classification/sample_submission.csv",
        "../data/cassava-leaf-disease-classification/sample_submission.csv",
        "/kaggle/input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
        "../input/cassava-leaf-disease-classification/cassava-leaf-disease-classification/sample_submission.csv",
    ]
)

if sample_path is not None:
    sample = pd.read_csv(sample_path)
    final_csv = sample[["image_id"]].merge(final_csv, on="image_id", how="left")
    if final_csv["label"].isna().any():
        missing = final_csv[final_csv["label"].isna()]["image_id"].head(5).tolist()
        raise RuntimeError(f"Missing predictions for some test ids, e.g.: {missing}")
    final_csv["label"] = final_csv["label"].astype(int)

final_csv.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final_csv.shape)
print(final_csv.head())



## === cell 5
final_csv.head()
