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

0.8144

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os

for dirname, _, filenames in os.walk("/kaggle/input/aerial-cactus-identification"):
    for filename in filenames[:10]:
        print(os.path.join(dirname, filename))



## === cell 1
import os

os.system(
    "cp -f /kaggle/input/aerial-cactus-identification/train.csv /kaggle/working/train.csv"
)
os.system(
    "unzip -o /kaggle/input/aerial-cactus-identification/train.zip -d /kaggle/working >/dev/null"
)
os.system(
    "unzip -o /kaggle/input/aerial-cactus-identification/test.zip -d /kaggle/working >/dev/null"
)

print(
    "Unzip/copy done. Working dir contains:",
    sorted(
        [
            p
            for p in os.listdir("/kaggle/working")
            if p in ["train", "test", "train.csv"]
        ]
    ),
)


def _resolve_image_dir(base_dir: str) -> str:
    """
    Return directory that contains .jpg files, handling nested folders.
    """
    direct = base_dir
    nested = os.path.join(base_dir, os.path.basename(base_dir))
    if os.path.isdir(direct) and any(
        fn.lower().endswith(".jpg") for fn in os.listdir(direct)[:50]
    ):
        return direct
    if os.path.isdir(nested) and any(
        fn.lower().endswith(".jpg") for fn in os.listdir(nested)[:50]
    ):
        return nested

    for root, _, files in os.walk(base_dir):
        if any(f.lower().endswith(".jpg") for f in files):
            return root
    raise FileNotFoundError(f"Could not find images under {base_dir}")


TRAIN_IMG_DIR = _resolve_image_dir("/kaggle/working/train")
TEST_IMG_DIR = _resolve_image_dir("/kaggle/working/test")
print("Resolved TRAIN_IMG_DIR:", TRAIN_IMG_DIR)
print("Resolved TEST_IMG_DIR :", TEST_IMG_DIR)



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4076863353.py in <cell line: 0>()
     47 
     48 
---> 49 TRAIN_IMG_DIR = _resolve_image_dir("/kaggle/working/train")
     50 TEST_IMG_DIR = _resolve_image_dir("/kaggle/working/test")
     51 print("Resolved TRAIN_IMG_DIR:", TRAIN_IMG_DIR)

/tmp/ipykernel_11/4076863353.py in _resolve_image_dir(base_dir)
     44         if any(f.lower().endswith(".jpg") for f in files):
     45             return root
---> 46     raise FileNotFoundError(f"Could not find images under {base_dir}")
     47 
     48 

FileNotFoundError: Could not find images under /kaggle/working/train

## === cell 2
import matplotlib.pyplot as plt

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import tensorflow as tf  # noqa: E402

print("tf version:", tf.__version__)
print("Num GPUs Available:", len(tf.config.list_physical_devices("GPU")))



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
df = pd.read_csv("/kaggle/working/train.csv")
print(df.head())
df.has_cactus.value_counts().plot.bar()
plt.show()



## === cell 4
from tensorflow.keras.preprocessing.image import (
    ImageDataGenerator,
    load_img,
)  # noqa: E402

filename = df.id.iloc[10]
print("Example filename:", filename)
image = load_img(os.path.join(TRAIN_IMG_DIR, filename))
plt.imshow(image)
plt.axis("off")
plt.show()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1597080670.py in <cell line: 0>()
      6 filename = df.id.iloc[10]
      7 print("Example filename:", filename)
----> 8 image = load_img(os.path.join(TRAIN_IMG_DIR, filename))
      9 plt.imshow(image)
     10 plt.axis("off")

NameError: name 'TRAIN_IMG_DIR' is not defined

## === cell 5
from sklearn.model_selection import train_test_split

train_df, validate_df = train_test_split(
    df, test_size=0.20, random_state=42, stratify=df["has_cactus"]
)
train_df = train_df.reset_index(drop=True)
validate_df = validate_df.reset_index(drop=True)



## === cell 6
train_datagen = ImageDataGenerator(
    rotation_range=15,
    rescale=1.0 / 255.0,
    zoom_range=0.3,
    horizontal_flip=True,
    vertical_flip=True,
    width_shift_range=0.1,
    height_shift_range=0.1,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)



## === cell 7
IMAGE_SIZE = (32, 32)
INPUT_SHAPE = (32, 32, 3)
BATCH_SIZE = 2**10  # preserve original intent

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
    seed=42,
)

validation_generator = valid_datagen.flow_from_dataframe(
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

import math

steps_per_epoch = max(1, math.ceil(train_generator.n / train_generator.batch_size))
validation_steps = max(
    1, math.ceil(validation_generator.n / validation_generator.batch_size)
)
print(
    "train n/batch/steps:",
    train_generator.n,
    train_generator.batch_size,
    steps_per_epoch,
)
print(
    "valid n/batch/steps:",
    validation_generator.n,
    validation_generator.batch_size,
    validation_steps,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2348343263.py in <cell line: 0>()
      5 train_generator = train_datagen.flow_from_dataframe(
      6     dataframe=train_df,
----> 7     directory=TRAIN_IMG_DIR,
      8     x_col="id",
      9     y_col="has_cactus",

NameError: name 'TRAIN_IMG_DIR' is not defined

## === cell 8
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    Flatten,
    Dense,
    BatchNormalization,
    Dropout,
    AveragePooling2D,
)
from tensorflow.keras.callbacks import EarlyStopping

model = Sequential(
    [
        Conv2D(
            filters=64,
            kernel_size=(2, 2),
            strides=(1, 1),
            activation="relu",
            input_shape=INPUT_SHAPE,
            padding="same",
        ),
        BatchNormalization(),
        AveragePooling2D(pool_size=(2, 2)),
        Dropout(0.2),
        Flatten(),
        Dense(128, activation="relu"),
        Dense(32, activation="relu"),
        Dropout(0.45),
        Dense(1, activation="sigmoid"),
    ]
)

earlystop = EarlyStopping(patience=3, restore_best_weights=True)
model.compile(loss="binary_crossentropy", optimizer="adam", metrics=["accuracy"])
callbacks = [earlystop]
model.summary()



## === cell 9
history = model.fit(
    train_generator,
    epochs=30,
    steps_per_epoch=steps_per_epoch,
    validation_data=validation_generator,
    validation_steps=validation_steps,
    callbacks=callbacks,
    verbose=2,
)



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1627735331.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_generator,
      3     epochs=30,
      4     steps_per_epoch=steps_per_epoch,
      5     validation_data=validation_generator,

NameError: name 'train_generator' is not defined

## === cell 10
pd.DataFrame(history.history).plot()
plt.show()



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1793498889.py in <cell line: 0>()
----> 1 pd.DataFrame(history.history).plot()
      2 plt.show()
      3 

NameError: name 'history' is not defined

## === cell 11
super_train_generator = train_datagen.flow_from_dataframe(
    dataframe=df.reset_index(drop=True),
    directory=TRAIN_IMG_DIR,
    x_col="id",
    y_col="has_cactus",
    target_size=IMAGE_SIZE,
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    class_mode="raw",
    shuffle=True,
    seed=42,
)

super_steps = max(
    1, math.ceil(super_train_generator.n / super_train_generator.batch_size)
)
history2 = model.fit(
    super_train_generator,
    epochs=20,
    steps_per_epoch=super_steps,
    callbacks=callbacks,
    verbose=2,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3423952208.py in <cell line: 0>()
      1 super_train_generator = train_datagen.flow_from_dataframe(
      2     dataframe=df.reset_index(drop=True),
----> 3     directory=TRAIN_IMG_DIR,
      4     x_col="id",
      5     y_col="has_cactus",

NameError: name 'TRAIN_IMG_DIR' is not defined

## === cell 12
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)
test_ids = sample_sub["id"].tolist()

test_df = pd.DataFrame({"id": test_ids})

test_gen = ImageDataGenerator(rescale=1.0 / 255.0)
test_generator = test_gen.flow_from_dataframe(
    test_df,
    directory=TEST_IMG_DIR,
    x_col="id",
    y_col=None,
    class_mode=None,
    target_size=IMAGE_SIZE,
    batch_size=BATCH_SIZE,
    shuffle=False,
)

test_steps = max(1, math.ceil(test_generator.n / test_generator.batch_size))
pred = model.predict(test_generator, steps=test_steps, verbose=0).reshape(-1)[
    : len(test_ids)
]
print("Pred shape:", pred.shape, "min/max:", float(pred.min()), float(pred.max()))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3724298145.py in <cell line: 0>()
      9 test_generator = test_gen.flow_from_dataframe(
     10     test_df,
---> 11     directory=TEST_IMG_DIR,
     12     x_col="id",
     13     y_col=None,

NameError: name 'TEST_IMG_DIR' is not defined

## === cell 13
submission = pd.DataFrame({"id": test_ids, "has_cactus": pred})
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1637752630.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"id": test_ids, "has_cactus": pred})
      2 submission.to_csv("submission.csv", index=False)
      3 print(submission.head())
      4 print("Wrote submission.csv with shape:", submission.shape)
      5 

NameError: name 'pred' is not defined

## === cell 14
assert list(submission.columns) == ["id", "has_cactus"]
assert submission["id"].isna().sum() == 0
assert submission["has_cactus"].isna().sum() == 0
print(submission["has_cactus"].describe())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3868959858.py in <cell line: 0>()
----> 1 assert list(submission.columns) == ["id", "has_cactus"]
      2 assert submission["id"].isna().sum() == 0
      3 assert submission["has_cactus"].isna().sum() == 0
      4 print(submission["has_cactus"].describe())
      5 

NameError: name 'submission' is not defined

## === cell 15
os.system("rm -rf /kaggle/working/train /kaggle/working/test /kaggle/working/train.csv")
print(
    "Cleanup done; submission.csv exists:",
    os.path.exists("/kaggle/working/submission.csv"),
)
