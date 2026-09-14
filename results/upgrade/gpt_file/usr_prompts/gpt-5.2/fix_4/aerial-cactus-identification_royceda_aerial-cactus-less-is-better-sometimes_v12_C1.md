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

0.99792

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.99792) has done: 'I fix the first blocker by making the unzip step place images into the exact `/kaggle/working/train` and `/kaggle/working/test` directories that your `_resolve_image_dir()` expects (the previous unzip created a nested `train/train` and `test/test` layout, causing `FileNotFoundError`). Next, I address the TensorFlow import crash by switching to `tf.keras` in a Kaggle-safe way and removing the protobuf environment override that triggers the `MessageFactory.GetPrototype` issue. After those two fixes, the rest of your pipeline (generators → same CNN → training → prediction) can run unchanged and write a valid `submission.csv` with the required columns. I also add a tiny guard to ensure the submission length matches `sample_submission.csv` exactly.'

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
import zipfile
from pathlib import Path

os.system(
    "cp -f /kaggle/input/aerial-cactus-identification/train.csv /kaggle/working/train.csv"
)

WORKING = Path("/kaggle/working")
TRAIN_DIR = WORKING / "train"
TEST_DIR = WORKING / "test"
TRAIN_DIR.mkdir(parents=True, exist_ok=True)
TEST_DIR.mkdir(parents=True, exist_ok=True)


def unzip_to(zip_path: str, dst_dir: Path):
    with zipfile.ZipFile(zip_path, "r") as z:
        z.extractall(dst_dir)


unzip_to("/kaggle/input/aerial-cactus-identification/train.zip", TRAIN_DIR)
unzip_to("/kaggle/input/aerial-cactus-identification/test.zip", TEST_DIR)

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
print("Train dir example:", list(TRAIN_DIR.glob("*.jpg"))[:3])
print("Test dir example :", list(TEST_DIR.glob("*.jpg"))[:3])


def _resolve_image_dir(base_dir: str) -> str:
    """
    Return directory that contains .jpg files, handling nested folders.
    """
    direct = base_dir
    nested = os.path.join(base_dir, os.path.basename(base_dir))
    if os.path.isdir(direct) and any(
        fn.lower().endswith(".jpg") for fn in os.listdir(direct)[:200]
    ):
        return direct
    if os.path.isdir(nested) and any(
        fn.lower().endswith(".jpg") for fn in os.listdir(nested)[:200]
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



## === cell 2
import matplotlib.pyplot as plt

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



## === cell 10
pd.DataFrame(history.history).plot()
plt.show()



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



## === cell 13
submission = pd.DataFrame({"id": test_ids, "has_cactus": pred.astype(float)})
submission = submission.iloc[: len(test_ids)].copy()
submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)



## === cell 14
assert list(submission.columns) == ["id", "has_cactus"]
assert submission["id"].isna().sum() == 0
assert submission["has_cactus"].isna().sum() == 0
assert submission.shape[0] == len(test_ids)
print(submission["has_cactus"].describe())



## === cell 15
os.system("rm -rf /kaggle/working/train /kaggle/working/test /kaggle/working/train.csv")
print(
    "Cleanup done; submission.csv exists:",
    os.path.exists("/kaggle/working/submission.csv"),
)
