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

0.8842550619522515

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I set the protobuf implementation environment variable before importing TensorFlow so the import succeeds, then I remove the unsupported `workers` and `use_multiprocessing` arguments from the Keras `fit` and `predict` calls. These fixes allow the model to train and generate predictions, and the script finally write a correctly‑named `.csv` submission file.'
- What this solution (achieved 0.11024) has done: 'The changes add parallel data loading to all generators and the training loop by specifying multiple workers and enabling multiprocessing, which reduces I/O bottlenecks without altering the model architecture, training epochs, or prediction logic. This keeps the core algorithm identical while speeding up each epoch, allowing the whole script to finish within the 600‑second limit.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import json
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow.keras.preprocessing.image import ImageDataGenerator
    from tensorflow.keras.applications import EfficientNetB0
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import GlobalAveragePooling2D, Dense
    from tensorflow.keras.callbacks import (
        ModelCheckpoint,
        ReduceLROnPlateau,
        EarlyStopping,
    )
except Exception as e:
    tf = None
    print("TensorFlow import failed, using fallback baseline model. Error:", e)

if tf is not None:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 224
BATCH_SIZE = 64
SEED = 42

BASE_INPUT = "../input/cassava-leaf-disease-classification"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_IMG_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_IMG_DIR = os.path.join(BASE_INPUT, "test_images")
SUBMISSION_PATH = "submission.csv"
PRETRAINED_MODEL_PATH = (
    "../input/cassava-challenge-models/Cassava_best_model_effnetb4_512x512.h5"
)

train_df = pd.read_csv(TRAIN_CSV)
train_df["label"] = train_df["label"].astype(str)
train_df["filepath"] = train_df["image_id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x)
)
train_df = train_df[train_df["filepath"].apply(os.path.isfile)].reset_index(drop=True)

most_common_label_str = train_df["label"].mode()[0]
most_common_label_int = int(most_common_label_str)




## === cell 2
from sklearn.model_selection import train_test_split

train_samples, val_samples = train_test_split(
    train_df,
    test_size=0.1,
    stratify=train_df["label"],
    random_state=SEED,
)

train_gen = ImageDataGenerator(
    rescale=1.0 / 255,
    horizontal_flip=True,
    rotation_range=20,
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=0.2,
)

val_gen = ImageDataGenerator(rescale=1.0 / 255)

train_flow = train_gen.flow_from_dataframe(
    train_samples,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=True,
    seed=SEED,
    workers=4,
    use_multiprocessing=True,
)

val_flow = val_gen.flow_from_dataframe(
    val_samples,
    x_col="filepath",
    y_col="label",
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    shuffle=False,
    workers=4,
    use_multiprocessing=True,
)




## === cell 3
def build_model(num_classes=5, img_size=IMG_SIZE):
    base = EfficientNetB0(
        include_top=False, input_shape=(img_size, img_size, 3), weights="imagenet"
    )
    x = GlobalAveragePooling2D()(base.output)
    outputs = Dense(num_classes, activation="softmax")(x)
    model = Model(inputs=base.input, outputs=outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


if tf is not None and os.path.isfile(PRETRAINED_MODEL_PATH):
    model = load_model(PRETRAINED_MODEL_PATH)
else:
    model = build_model() if tf is not None else None

if tf is not None and not os.path.isfile(PRETRAINED_MODEL_PATH):
    checkpoint_cb = ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True,
        mode="max",
        verbose=1,
    )
    reduce_lr_cb = ReduceLROnPlateau(
        monitor="val_accuracy", factor=0.5, patience=2, min_lr=1e-6, verbose=1
    )
    early_stop_cb = EarlyStopping(
        monitor="val_accuracy", patience=5, mode="max", verbose=1
    )

    model.fit(
        train_flow,
        epochs=12,
        validation_data=val_flow,
        callbacks=[checkpoint_cb, reduce_lr_cb, early_stop_cb],
        verbose=2,
        workers=4,
        use_multiprocessing=True,
    )
    if os.path.isfile("best_model.h5"):
        model = load_model("best_model.h5")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3315488442.py in <cell line: 0>()
     35 
     36     # Use multiple workers for faster data feeding during training
---> 37     model.fit(
     38         train_flow,
     39         epochs=12,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 4
test_filenames = sorted(os.listdir(TEST_IMG_DIR))
test_filepaths = [os.path.join(TEST_IMG_DIR, fname) for fname in test_filenames]

test_gen = ImageDataGenerator(rescale=1.0 / 255)
test_flow = test_gen.flow_from_dataframe(
    pd.DataFrame({"filepath": test_filepaths, "image_id": test_filenames}),
    x_col="filepath",
    y_col=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode=None,
    shuffle=False,
    workers=4,
    use_multiprocessing=True,
)

if tf is not None and model is not None:
    pred_probs = model.predict(
        test_flow,
        verbose=0,
    )
    pred_labels = np.argmax(pred_probs, axis=1)
else:
    pred_labels = np.full(
        shape=(len(test_filepaths),), fill_value=most_common_label_int, dtype=int
    )

actual_filenames = [os.path.basename(path) for path in test_flow.filenames]




## === cell 5
submission = pd.DataFrame({"image_id": actual_filenames, "label": pred_labels})
submission.to_csv(SUBMISSION_PATH, index=False)
print(f"Submission written to {SUBMISSION_PATH}, shape: {submission.shape}")
