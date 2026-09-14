# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.7

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

# 5. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

import matplotlib.pyplot as plt

from os import listdir
from PIL import Image

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model, load_model
from tensorflow.keras.layers import Conv2D, Dense, Flatten, Input
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from tensorflow.keras.preprocessing.image import ImageDataGenerator, img_to_array
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## === cell 1
CANDIDATE_BASES = [
    "/kaggle/input/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification",
    "/kaggle/input/aerial-cactus-identification/aerial-cactus-identification",
    "/kaggle/data/aerial-cactus-identification/aerial-cactus-identification",
    "../input/aerial-cactus-identification",
    "../input/aerial-cactus-identification/aerial-cactus-identification",
]
DATA_DIR = None
for p in CANDIDATE_BASES:
    if os.path.exists(p):
        if (
            os.path.exists(os.path.join(p, "train.csv"))
            and os.path.isdir(os.path.join(p, "train"))
            and os.path.isdir(os.path.join(p, "test"))
        ):
            DATA_DIR = p
            break

if DATA_DIR is None:
    raise FileNotFoundError(
        "Could not find aerial-cactus-identification dataset directory. Checked: "
        + ", ".join(CANDIDATE_BASES)
    )

TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

print("Using DATA_DIR:", DATA_DIR)
print(
    "Train images:",
    len(os.listdir(TRAIN_DIR)),
    "Test images:",
    len(os.listdir(TEST_DIR)),
)




## === cell 2
def load_images_from_ids(image_dir, ids, target_size=(32, 32)):
    X = np.zeros((len(ids), target_size[0], target_size[1], 3), dtype=np.float32)
    for i, img_id in enumerate(ids):
        fp = os.path.join(image_dir, img_id)
        with Image.open(fp) as im:
            im = im.convert("RGB")
            if im.size != target_size:
                im = im.resize(target_size, resample=Image.BILINEAR)
            X[i] = np.asarray(im, dtype=np.float32)
    return X


train_df = pd.read_csv(TRAIN_CSV)
train_df = train_df.sort_values("id", ascending=True).reset_index(drop=True)

Y_train = train_df["has_cactus"].astype(np.float32).values
train_ids = train_df["id"].values

X_train = load_images_from_ids(TRAIN_DIR, train_ids, target_size=(32, 32))
X_train /= 255.0

sample_sub = pd.read_csv(SAMPLE_SUB)
test_ids = sample_sub["id"].values
X_test = load_images_from_ids(TEST_DIR, test_ids, target_size=(32, 32))
X_test /= 255.0

submission = pd.DataFrame({"id": test_ids})
print(
    "X_train:",
    X_train.shape,
    "Y_train:",
    Y_train.shape,
    "X_test:",
    X_test.shape,
    "submission:",
    submission.shape,
)



## === cell 3
pd.Series(Y_train).value_counts().plot.bar()
plt.show()



## === cell 4
epochs = 10
batch_size = 64



## === cell 5
plt.figure(figsize=(8, 8))
for i in range(0, 15):
    plt.subplot(5, 3, i + 1)
    j = np.random.randint(0, Y_train.shape[0])
    plt.imshow(X_train[j])
    plt.axis("off")
plt.tight_layout()
plt.show()



## === cell 6
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.mobilenet_v2 import MobileNetV2
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.applications.nasnet import NASNetLarge



## === cell 7
learning_rate_reduction = ReduceLROnPlateau(
    monitor="val_accuracy", patience=3, verbose=1, factor=0.7, min_lr=0.00001
)



## === cell 8
datagen = ImageDataGenerator(
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    validation_split=0.1,
)

train_generator = datagen.flow(
    X_train, Y_train, batch_size=batch_size, subset="training", seed=SEED
)
val_generator = datagen.flow(
    X_train, Y_train, batch_size=batch_size, subset="validation", seed=SEED
)




## === cell 9
def buildModel(base_model):
    X = base_model.output
    X = Flatten()(X)
    X = Dense(512, activation="relu", kernel_regularizer="l2")(X)
    X = Dense(1, activation="sigmoid")(X)

    model = Model(inputs=base_model.input, outputs=X)
    return model




## === cell 10
history = []


def fitModel(model, cpoint=False):
    threshold = int(len(model.layers) * 0.8)
    for i in model.layers[:threshold]:
        i.trainable = False
    for i in model.layers[threshold:]:
        i.trainable = True

    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    cb = [learning_rate_reduction]
    if cpoint:
        cb.append(
            ModelCheckpoint(
                filepath="best_model.keras",
                monitor="val_accuracy",
                mode="max",
                save_best_only=True,
                verbose=1,
            )
        )

    history.append(
        model.fit(
            train_generator,
            epochs=epochs,
            steps_per_epoch=int(X_train.shape[0] // batch_size * 1.5),
            validation_data=val_generator,
            validation_steps=int(X_train.shape[0] // batch_size * 0.4),
            callbacks=cb,
            verbose=2,
        )
    )
    return model




## === cell 11
base_model = VGG16(weights="imagenet", include_top=False, input_shape=(32, 32, 3))
model = buildModel(base_model)

epochs = 200
batch_size = 64

train_generator = datagen.flow(
    X_train, Y_train, batch_size=batch_size, subset="training", seed=SEED
)
val_generator = datagen.flow(
    X_train, Y_train, batch_size=batch_size, subset="validation", seed=SEED
)

model = fitModel(model, cpoint=False)



## === cell 12
pred = model.predict(X_test, batch_size=batch_size, verbose=0).reshape(-1)
submission["has_cactus"] = pred.astype(np.float64)

submission["has_cactus"] = submission["has_cactus"].clip(0.0, 1.0)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
