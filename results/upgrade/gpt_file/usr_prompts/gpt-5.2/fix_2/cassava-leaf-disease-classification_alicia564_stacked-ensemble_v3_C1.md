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

0.8951344817165306

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
from datetime import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf


print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.efficientnet import preprocess_input
from sklearn.preprocessing import LabelEncoder


def load_and_preprocess_image(path, label):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [224, 224])
    image = tf.cast(image, tf.float32)
    image = preprocess_input(image)  # matches ImageDataGenerator preprocessing_function
    return image, label


label_to_disease = pd.read_json(
    "/kaggle/input/cassava-leaf-disease-classification/label_num_to_disease_map.json",
    typ="series",
)
train_csv = pd.read_csv("/kaggle/input/cassava-leaf-disease-classification/train.csv")

train_csv["disease"] = train_csv["label"].map(label_to_disease)
train_csv["path"] = (
    "/kaggle/input/cassava-leaf-disease-classification/train_images/"
    + train_csv["image_id"]
)

le = LabelEncoder()
train_csv["label_encoded"] = le.fit_transform(train_csv["disease"].astype(str))

train_csv["label"] = train_csv["label"].astype(str)

train, valid = train_test_split(
    train_csv, test_size=0.2, stratify=train_csv["label"], random_state=42
)

train_ds = tf.data.Dataset.from_tensor_slices(
    (train["path"].values, train["label_encoded"].values)
)
valid_ds = tf.data.Dataset.from_tensor_slices(
    (valid["path"].values, valid["label_encoded"].values)
)

train_ds = (
    train_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)
valid_ds = (
    valid_ds.map(load_and_preprocess_image, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(32)
    .prefetch(tf.data.AUTOTUNE)
)

datagen_aug = ImageDataGenerator(
    preprocessing_function=preprocess_input,
    rotation_range=45,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode="nearest",
)

train_generator = datagen_aug.flow_from_dataframe(
    dataframe=train,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=True,
    seed=42,
)

datagen_valid = ImageDataGenerator(preprocessing_function=preprocess_input)

valid_generator = datagen_valid.flow_from_dataframe(
    dataframe=valid,
    x_col="path",
    y_col="disease",
    target_size=(224, 224),
    batch_size=32,
    class_mode="categorical",
    shuffle=False,
)

print("Train/Valid sizes:", len(train), len(valid))
print("Num classes:", len(le.classes_))



## === cell 2
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

early_stopping = EarlyStopping(
    monitor="val_loss", patience=3, restore_best_weights=True
)

learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_loss", patience=2, factor=0.5, min_lr=1e-6, verbose=1
)



## === cell 3
from tensorflow.keras.layers import Input, TFSMLayer
from tensorflow.keras.models import Model


def find_saved_model_dir(prefer_keywords):
    """
    Find a SavedModel directory under /kaggle/input that contains saved_model.pb or saved_model.pbtxt.
    prefer_keywords: list of strings to rank paths by.
    """
    candidates = []
    for root, dirs, files in os.walk("/kaggle/input"):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            path_l = root.lower()
            score = 0
            for kw in prefer_keywords:
                if kw.lower() in path_l:
                    score += 10
            if "tensorflow2" in path_l:
                score += 2
            candidates.append((score, root))
    if not candidates:
        raise FileNotFoundError(
            "Could not find any SavedModel (saved_model.pb) under /kaggle/input. "
            "Please add the required model datasets (cropnet/densenet/efficientnet) to the notebook."
        )
    candidates.sort(reverse=True, key=lambda x: x[0])
    return candidates[0][1]


def build_tfsmlayer_model(
    saved_model_dir, input_shape=(224, 224, 3), endpoint="serving_default"
):
    layer = TFSMLayer(saved_model_dir, call_endpoint=endpoint)
    inp = Input(shape=input_shape)
    out = layer(inp)
    return layer, Model(inp, out)


cropnet_dir = find_saved_model_dir(["cropnet", "crop", "cassava"])
densenet_dir = find_saved_model_dir(["densenet"])
efficientnet_dir = find_saved_model_dir(["efficientnet", "b4", "effnet"])

print("Using SavedModel dirs:")
print(" cropnet:", cropnet_dir)
print(" densenet:", densenet_dir)
print(" efficientnet:", efficientnet_dir)

cropnet_layer, cropnet_model = build_tfsmlayer_model(cropnet_dir)
densenet_layer, densenet_model = build_tfsmlayer_model(densenet_dir)
efficientnet_layer, efficientnet_model = build_tfsmlayer_model(efficientnet_dir)

cropnet_weight = 0.65
densenet_weight = 0.25
efficientnet_weight = 0.1



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/223833607.py in <cell line: 0>()
     43 # Attempt to locate the three models.
     44 # NOTE: this preserves core logic (TFSMLayer ensemble) but fixes broken paths.
---> 45 cropnet_dir = find_saved_model_dir(["cropnet", "crop", "cassava"])
     46 densenet_dir = find_saved_model_dir(["densenet"])
     47 efficientnet_dir = find_saved_model_dir(["efficientnet", "b4", "effnet"])

/tmp/ipykernel_11/223833607.py in find_saved_model_dir(prefer_keywords)
     24             candidates.append((score, root))
     25     if not candidates:
---> 26         raise FileNotFoundError(
     27             "Could not find any SavedModel (saved_model.pb) under /kaggle/input. "
     28             "Please add the required model datasets (cropnet/densenet/efficientnet) to the notebook."

FileNotFoundError: Could not find any SavedModel (saved_model.pb) under /kaggle/input. Please add the required model datasets (cropnet/densenet/efficientnet) to the notebook.

## === cell 4
from sklearn.linear_model import LogisticRegression


def to_probs(pred_output):
    """
    TFSMLayer model output can be:
      - dict: {'output_0': array, ...}
      - tensor/ndarray: array
    Return a numpy array of probabilities/logits as (batch, num_classes).
    """
    if isinstance(pred_output, dict):
        if "output_0" in pred_output:
            arr = pred_output["output_0"]
        else:
            arr = next(iter(pred_output.values()))
    else:
        arr = pred_output
    return np.asarray(arr)


soft_voting_features = []
soft_voting_labels = []

for img_batch, label_batch in train_ds:
    img_batch_np = img_batch.numpy()

    cropnet_probs = to_probs(cropnet_model.predict(img_batch_np, verbose=0))
    densenet_probs = to_probs(densenet_model.predict(img_batch_np, verbose=0))
    efficientnet_probs = to_probs(efficientnet_model.predict(img_batch_np, verbose=0))

    soft_voting_probs = (
        cropnet_weight * cropnet_probs
        + densenet_weight * densenet_probs
        + efficientnet_weight * efficientnet_probs
    )

    soft_voting_features.append(soft_voting_probs)
    soft_voting_labels.append(label_batch.numpy())

soft_voting_features = np.concatenate(soft_voting_features, axis=0)
soft_voting_labels = np.concatenate(soft_voting_labels, axis=0)

meta_model = LogisticRegression(max_iter=1000, multi_class="multinomial", n_jobs=-1)
meta_model.fit(soft_voting_features, soft_voting_labels)

print("Meta-model trained on features:", soft_voting_features.shape)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2155620856.py in <cell line: 0>()
     29     img_batch_np = img_batch.numpy()
     30 
---> 31     cropnet_probs = to_probs(cropnet_model.predict(img_batch_np, verbose=0))
     32     densenet_probs = to_probs(densenet_model.predict(img_batch_np, verbose=0))
     33     efficientnet_probs = to_probs(efficientnet_model.predict(img_batch_np, verbose=0))

NameError: name 'cropnet_model' is not defined

## === cell 5
from sklearn.metrics import accuracy_score

meta_model_predictions = meta_model.predict(soft_voting_features)
accuracy = accuracy_score(soft_voting_labels, meta_model_predictions)
print("Meta-model (Logistic Regression) Train Accuracy:", accuracy)



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/166781128.py in <cell line: 0>()
      2 
      3 # Optional quick check on train features (keeps original "check accuracy" intent)
----> 4 meta_model_predictions = meta_model.predict(soft_voting_features)
      5 accuracy = accuracy_score(soft_voting_labels, meta_model_predictions)
      6 print("Meta-model (Logistic Regression) Train Accuracy:", accuracy)

NameError: name 'meta_model' is not defined

## === cell 6
from tensorflow.keras.preprocessing.image import load_img, img_to_array

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
image_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images"
img_size = (224, 224)

predictions = []
image_names = sample_sub["image_id"].tolist()

for filename in image_names:
    img_path = os.path.join(image_dir, filename)
    img = load_img(img_path, target_size=img_size)
    img_array = img_to_array(img).astype(np.float32)
    img_array = preprocess_input(img_array)  # match training preprocessing
    img_array = np.expand_dims(img_array, axis=0)

    cropnet_probs = to_probs(cropnet_model.predict(img_array, verbose=0))
    densenet_probs = to_probs(densenet_model.predict(img_array, verbose=0))
    efficientnet_probs = to_probs(efficientnet_model.predict(img_array, verbose=0))

    soft_voting_probs = (
        cropnet_weight * cropnet_probs
        + densenet_weight * densenet_probs
        + efficientnet_weight * efficientnet_probs
    ).reshape(1, -1)

    pred = int(meta_model.predict(soft_voting_probs)[0])
    predictions.append(pred)

submission_df = pd.DataFrame({"image_id": image_names, "label": predictions})

out_path = "/kaggle/working/submission.csv"
submission_df.to_csv(out_path, index=False)
print("Submission file created:", out_path)
print(submission_df.head())
print("Submission shape:", submission_df.shape)
assert (
    submission_df.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."
assert list(submission_df.columns) == [
    "image_id",
    "label",
], "Submission columns must be image_id,label."

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3425097102.py in <cell line: 0>()
     19     img_array = np.expand_dims(img_array, axis=0)
     20 
---> 21     cropnet_probs = to_probs(cropnet_model.predict(img_array, verbose=0))
     22     densenet_probs = to_probs(densenet_model.predict(img_array, verbose=0))
     23     efficientnet_probs = to_probs(efficientnet_model.predict(img_array, verbose=0))

NameError: name 'cropnet_model' is not defined
