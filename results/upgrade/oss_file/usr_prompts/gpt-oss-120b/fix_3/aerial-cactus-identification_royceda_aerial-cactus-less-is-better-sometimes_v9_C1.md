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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.8258

# 6. Current score

0.94509

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.96768) has done: 'We replace the outdated `keras` imports with `tensorflow.keras`, fix the GPU check so it doesn’t abort when a GPU isn’t found, ensure the correct data paths are used, correctly unzip the image archives if needed, define the image generators after importing `ImageDataGenerator`, and finally build a proper submission CSV with both required columns. Minor adjustments such as adding an AUC metric and using a sensible rescaling factor keep the core model unchanged while helping the score move toward the target.'
- What this solution (achieved 0.94509) has done: 'I add an environment variable setting before TensorFlow is imported to avoid the protobuf `MessageFactory` error, and renumber the notebook cells to start from 1 as required. No other logic is changed, preserving the model and training pipeline while ensuring a valid `submission.csv` is created.'

# 9. Code solution

## === cell 0
import os, zipfile, numpy as np, pandas as pd, matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    BatchNormalization,
    AveragePooling2D,
    Dropout,
    Flatten,
    Dense,
)
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

BASE_DIR = "/kaggle/input/aerial-cactus-identification"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train")

if not os.path.isdir(TRAIN_IMG_DIR):
    with zipfile.ZipFile(os.path.join(BASE_DIR, "train.zip"), "r") as z:
        z.extractall("/kaggle/working")
    TRAIN_IMG_DIR = "/kaggle/working/train"
if not os.path.isdir(TEST_IMG_DIR):
    with zipfile.ZipFile(os.path.join(BASE_DIR, "test.zip"), "r") as z:
        z.extractall("/kaggle/working")
    TEST_IMG_DIR = "/kaggle/working/test"

print("TensorFlow version:", tf.__version__)
if tf.test.is_gpu_available():
    print("GPU found:", tf.test.gpu_device_name())
else:
    print("GPU not found – continuing on CPU")




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df = pd.read_csv(TRAIN_CSV)
print("Training samples:", df.shape[0])
df.sample(3)




## === cell 2
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)




## === cell 3
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

val_datagen = ImageDataGenerator(rescale=1.0 / 255)

IMAGE_SIZE = (32, 32)
BATCH_SIZE = 2**10  # 1024

train_generator = train_datagen.flow_from_dataframe(
    dataframe=train_df,
    directory=TRAIN_IMG_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
)

validation_generator = val_datagen.flow_from_dataframe(
    dataframe=validate_df,
    directory=TRAIN_IMG_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=False,
)




## === cell 4
model = Sequential(
    [
        Conv2D(64, (3, 3), activation="relu", padding="same", input_shape=(32, 32, 3)),
        BatchNormalization(),
        AveragePooling2D((2, 2)),
        Conv2D(128, (4, 4), activation="relu", padding="same"),
        BatchNormalization(),
        AveragePooling2D((2, 2)),
        Conv2D(64, (2, 2), activation="relu", padding="same"),
        BatchNormalization(),
        AveragePooling2D((2, 2)),
        Dropout(0.3),
        Conv2D(32, (2, 2), activation="relu", padding="same"),
        BatchNormalization(),
        AveragePooling2D((2, 2)),
        Dropout(0.3),
        Flatten(),
        Dense(64, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

model.compile(
    loss="binary_crossentropy",
    optimizer="adam",
    metrics=["accuracy", tf.keras.metrics.AUC(name="auc")],
)

earlystop = EarlyStopping(patience=3, restore_best_weights=True, monitor="val_auc")
reduce_lr = ReduceLROnPlateau(patience=2, factor=0.5, monitor="val_auc")
callbacks = [earlystop, reduce_lr]




## === cell 5
history = model.fit(
    train_generator,
    epochs=15,
    validation_data=validation_generator,
    callbacks=callbacks,
    verbose=2,
)




## === cell 6
test_files = sorted([f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame({"id": test_files})

test_datagen = ImageDataGenerator(rescale=1.0 / 255)
test_generator = test_datagen.flow_from_dataframe(
    dataframe=test_df,
    directory=TEST_IMG_DIR,
    x_col="id",
    y_col=None,
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    class_mode=None,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

pred = model.predict(test_generator, verbose=0)
test_df["has_cactus"] = pred.ravel()




## === cell 7
submission_path = "/kaggle/working/submission.csv"
test_df.to_csv(submission_path, index=False)
print("Submission saved to:", submission_path)
print(test_df.head())




## === cell 8
print("Files in /kaggle/working:")
print("\n".join(os.listdir("/kaggle/working")))
