# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import sys
import random
import subprocess

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "3"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]

try:
    import google.protobuf  # noqa: F401
    from google.protobuf import __version__ as pb_version  # noqa: F401

    major_minor = tuple(int(x) for x in pb_version.split(".")[:2])
    if major_minor >= (4, 21) or major_minor >= (4, 0):
        raise RuntimeError(f"protobuf too new: {pb_version}")
except Exception:
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
    )
    for m in list(sys.modules.keys()):
        if m.startswith("google.protobuf"):
            del sys.modules[m]

import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.optimizers import Adam

print("TF version:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
def processAndWriteDf(df):
    if "image_id" not in df.columns:
        raise ValueError("Expected 'image_id' column in dataframe.")
    df.to_csv("./submission.csv", index=False)
    print("write done -> ./submission.csv")
    return df




## === cell 2
def _resolve_pp2020_base():
    """
    Fix: choose the real existing Kaggle input path (varies by environment).
    Supports both Kaggle's canonical /kaggle/input and the provided /kaggle/data layouts.
    """
    candidates = [
        "/kaggle/input/plant-pathology-2020-fgvc7",
        "/kaggle/data/plant-pathology-2020-fgvc7",
        "/kaggle/input",
        "/kaggle/data",
        "../input/plant-pathology-2020-fgvc7",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c

        nested = os.path.join(c, "plant-pathology-2020-fgvc7")
        if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
            os.path.join(nested, "test.csv")
        ):
            return nested

    raise FileNotFoundError(f"Could not find dataset base dir in: {candidates}")


def getPredictionFromTPUModel():
    """
    Original notebook expected a pre-trained model file that is not present.
    Minimal fix: train a TF/Keras image model on the provided train set and predict on test.
    Keeps the same core approach (ImageDataGenerator + CNN-style model, sigmoid outputs, BCE loss).
    """
    base = _resolve_pp2020_base()
    base_dir = os.path.join(base, "images")
    train_path = os.path.join(base, "train.csv")
    test_path = os.path.join(base, "test.csv")
    sample_path = os.path.join(base, "sample_submission.csv")

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    sample_sub = pd.read_csv(sample_path)

    label_cols = [c for c in sample_sub.columns if c != "image_id"]

    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["image_id"] = train_df["image_id"].astype(str) + ".jpg"
    test_df["image_id"] = test_df["image_id"].astype(str) + ".jpg"

    img_size = (320, 320)

    batch_size = 32

    from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

    train_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input,
        validation_split=0.1,
        rotation_range=12,
        width_shift_range=0.06,
        height_shift_range=0.06,
        zoom_range=0.08,
        horizontal_flip=True,
        fill_mode="nearest",
    )
    valid_datagen = ImageDataGenerator(
        preprocessing_function=preprocess_input, validation_split=0.1
    )
    test_datagen = ImageDataGenerator(preprocessing_function=preprocess_input)

    train_gen = train_datagen.flow_from_dataframe(
        train_df,
        directory=base_dir,
        x_col="image_id",
        y_col=label_cols,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        subset="training",
        shuffle=True,
        seed=SEED,
    )

    valid_gen = valid_datagen.flow_from_dataframe(
        train_df,
        directory=base_dir,
        x_col="image_id",
        y_col=label_cols,
        target_size=img_size,
        batch_size=batch_size,
        class_mode="raw",
        subset="validation",
        shuffle=False,
    )

    test_gen = test_datagen.flow_from_dataframe(
        test_df,
        directory=base_dir,
        x_col="image_id",
        y_col=None,
        target_size=img_size,
        batch_size=batch_size,
        class_mode=None,
        shuffle=False,
    )

    base_model = tf.keras.applications.MobileNetV2(
        input_shape=(img_size[0], img_size[1], 3),
        include_top=False,
        weights="imagenet",
    )
    base_model.trainable = False  # keep minimal and stable within time budget

    inputs = tf.keras.Input(shape=(img_size[0], img_size[1], 3))
    x = base_model(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.25)(x)
    outputs = tf.keras.layers.Dense(len(label_cols), activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)

    model.compile(
        optimizer=Adam(learning_rate=2e-4),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(label_cols), name="auc"
            )
        ],
    )

    callbacks = [
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=2, min_lr=1e-6, verbose=1
        )
    ]

    epochs = 18
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=epochs,
        verbose=1,
        callbacks=callbacks,
    )

    base_model.trainable = True
    for layer in base_model.layers[:-25]:
        layer.trainable = False

    model.compile(
        optimizer=Adam(learning_rate=5e-5),
        loss=tf.keras.losses.BinaryCrossentropy(label_smoothing=0.01),
        metrics=[
            tf.keras.metrics.AUC(
                multi_label=True, num_labels=len(label_cols), name="auc"
            )
        ],
    )
    model.fit(
        train_gen,
        validation_data=valid_gen,
        epochs=6,
        verbose=1,
        callbacks=callbacks,
    )

    p = model.predict(test_gen, verbose=1)
    p = np.clip(p, 0.0, 1.0)

    out = pd.DataFrame(p, columns=label_cols)
    out.insert(0, "image_id", pd.read_csv(test_path)["image_id"].astype(str))  # no .jpg
    out = out[sample_sub.columns.tolist()]  # exact order/columns as required
    return out




## === cell 3
isTPU = True

if isTPU:
    df = getPredictionFromTPUModel()
    print(df.head(5))
    df.to_csv("./submission.csv", index=False)
    print("write done -> ./submission.csv")
else:
    df = pd.read_csv("../input/notebook45bc751087/submission.csv")
    df.head()
    result = processAndWriteDf(df)
    result.head(5)
