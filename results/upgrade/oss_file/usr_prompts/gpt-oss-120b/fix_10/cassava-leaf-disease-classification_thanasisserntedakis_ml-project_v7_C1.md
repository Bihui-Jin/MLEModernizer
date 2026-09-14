# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
import os, glob, json, shutil, io, multiprocessing
import numpy as np
import pandas as pd
from datetime import datetime
from collections import Counter

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"  # reduce TF logging overhead

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.models import Sequential, model_from_json, load_model
    from tensorflow.keras.layers import (
        Dense,
        Dropout,
        Flatten,
        Conv2D,
        MaxPooling2D,
        GlobalAveragePooling2D,
    )
    from tensorflow.keras import backend as K
    from tensorflow.keras.applications import MobileNetV2
    from tensorflow.keras.optimizers import Adam

    TF_AVAILABLE = True
except Exception as e:
    print("TensorFlow import failed:", e)
    TF_AVAILABLE = False

from sklearn.model_selection import train_test_split




## === cell 1
BASE_PATH = os.path.abspath("../input/cassava-leaf-disease-classification")
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_PATH, "train_images")
TEST_IMG_DIR = os.path.join(BASE_PATH, "test_images")
MODEL_JSON = "../input/resnet-model/model.json"  # original path (may not exist)
MODEL_H5 = "../input/resnet-model/model.h5"  # original path (may not exist)

IMG_SIZE = 224
SIZE = (IMG_SIZE, IMG_SIZE)

BATCH_SIZE = 256

SEED = 42

NUM_WORKERS = max(1, min(4, multiprocessing.cpu_count() or 1))

df_train = pd.read_csv(TRAIN_CSV)
df_train["label"] = df_train["label"].astype(str)
df_train["path"] = df_train["image_id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))

df_train, df_val = train_test_split(
    df_train, test_size=0.1, random_state=SEED, stratify=df_train["label"]
)

if TF_AVAILABLE:
    train_datagen = ImageDataGenerator(
        rescale=1.0 / 255,
        horizontal_flip=True,
        rotation_range=20,
        width_shift_range=0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        zoom_range=0.1,
    )
    val_datagen = ImageDataGenerator(rescale=1.0 / 255)

    train_gen = train_datagen.flow_from_dataframe(
        dataframe=df_train,
        x_col="path",
        y_col="label",
        target_size=SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=SEED,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )

    val_gen = val_datagen.flow_from_dataframe(
        dataframe=df_val,
        x_col="path",
        y_col="label",
        target_size=SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,
        seed=SEED,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )
else:
    train_gen = val_gen = None  # placeholders; not used when TF is unavailable

test_images = glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg"))
df_test = pd.DataFrame({"path": test_images})

if TF_AVAILABLE:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255)
    test_gen = test_datagen.flow_from_dataframe(
        dataframe=df_test,
        x_col="path",
        y_col=None,
        target_size=SIZE,
        batch_size=BATCH_SIZE,
        class_mode=None,
        shuffle=False,
        workers=NUM_WORKERS,
        use_multiprocessing=True,
    )
else:
    test_gen = None  # placeholder




## === cell 2
model = None
if TF_AVAILABLE and os.path.exists(MODEL_JSON) and os.path.exists(MODEL_H5):
    try:
        with open(MODEL_JSON, "r") as jf:
            loaded_model_json = jf.read()
        model = model_from_json(loaded_model_json)
        model.load_weights(MODEL_H5)
        print("Loaded pretrained ResNet model.")
    except Exception as e:
        print("Failed to load pretrained model:", e)
        model = None

if TF_AVAILABLE and model is None:
    if os.path.exists("model_tmp.json") and os.path.exists("model_tmp.h5"):
        try:
            with open("model_tmp.json", "r") as jf:
                loaded_json = jf.read()
            model = model_from_json(loaded_json)
            model.load_weights("model_tmp.h5")
            print("Loaded previously trained MobileNetV2 model from local cache.")
        except Exception as e:
            print("Failed to load cached model:", e)
            model = None

if TF_AVAILABLE and model is None:
    base = MobileNetV2(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )
    base.trainable = False  # freeze base
    model = Sequential(
        [base, GlobalAveragePooling2D(), Dropout(0.2), Dense(5, activation="softmax")]
    )
    model.compile(
        loss="categorical_crossentropy",
        optimizer=Adam(learning_rate=1e-3),
        metrics=["accuracy"],
    )
    print("Created new MobileNetV2 based model.")




## === cell 3
if (
    TF_AVAILABLE
    and model is not None
    and not (os.path.exists("model_tmp.json") and os.path.exists("model_tmp.h5"))
):
    epochs = 5
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=epochs,
        verbose=0,  # suppress per‑batch output to save time
    )
    model_json = model.to_json()
    with open("model_tmp.json", "w") as jf:
        jf.write(model_json)
    model.save_weights("model_tmp.h5")
    print("Model trained and temporary weights saved.")




## === cell 4
if TF_AVAILABLE:
    pred_test = model.predict(test_gen, verbose=0)  # suppress progress output
    pred_test_labels = np.argmax(pred_test, axis=-1)

    final_submission = df_test.copy()
    final_submission["image_id"] = final_submission["path"].apply(
        lambda p: os.path.basename(p)
    )
    final_submission["label"] = pred_test_labels
    submission_csv = final_submission[["image_id", "label"]]
else:
    sample_sub_path = os.path.join(BASE_PATH, "sample_submission.csv")
    if os.path.exists(sample_sub_path):
        submission_csv = pd.read_csv(sample_sub_path)
    else:
        submission_csv = pd.DataFrame(columns=["image_id", "label"])

submission_csv.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written with", len(submission_csv), "rows.")
