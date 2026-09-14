# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

3.7

# 3. Installed packages

geopandas==0.14.4
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0
tqdm==4.67.1

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

0.9948

# 6. Current score

0.49474

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.49474) has done: 'I fix the environment crash by pinning `protobuf` to a compatible 3.x version before importing TensorFlow/Keras, which resolves the `MessageFactory.GetPrototype` error. Then I update the `ModelCheckpoint` path to use the required `.keras` suffix so callbacks are created and training can run, which also fixes the downstream `callbacks/history` NameErrors. I correct the dataset directory detection so test images are found (your current `test_dir` points to a non-existent nested folder) and make test loading robust by using `tf.keras.utils.image_dataset_from_directory` to avoid manual cv2 loops and the empty-input error. Finally, I ensure a valid `submission.csv` with the exact required columns is written to the working directory.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
)

import numpy as np
import pandas as pd
import tensorflow as tf
import cv2
import matplotlib.pyplot as plt

from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Dense,
    Flatten,
    Dropout,
    BatchNormalization,
)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (
    ReduceLROnPlateau,
    EarlyStopping,
    ModelCheckpoint,
    TensorBoard,
)

seed = 42
np.random.seed(seed)
tf.random.set_seed(seed)

print("TF version:", tf.__version__)
print("Listing ../input:", os.listdir("../input"))



## === cell 1
base_dir = os.path.join("..", "input", "aerial-cactus-identification")
if not os.path.exists(base_dir):
    base_dir = os.path.join(
        "..", "input", "aerial-cactus-identification", "aerial-cactus-identification"
    )

train_csv_path = os.path.join(base_dir, "train.csv")
sample_sub_path = os.path.join(base_dir, "sample_submission.csv")

train_df = pd.read_csv(train_csv_path)

train_dir_candidates = [
    os.path.join(base_dir, "train"),
    os.path.join(base_dir, "train", "train"),
]
test_dir_candidates = [
    os.path.join(base_dir, "test"),
    os.path.join(base_dir, "test", "test"),
]

train_dir = next((p for p in train_dir_candidates if os.path.isdir(p)), None)
test_dir = next((p for p in test_dir_candidates if os.path.isdir(p)), None)

print("base_dir:", base_dir)
print("train_csv_path:", train_csv_path, "exists:", os.path.isfile(train_csv_path))
print(
    "train_dir:", train_dir, "exists:", os.path.isdir(train_dir) if train_dir else None
)
print("test_dir:", test_dir, "exists:", os.path.isdir(test_dir) if test_dir else None)
print(train_df.head())

if train_dir is None or test_dir is None:
    raise FileNotFoundError(
        f"Could not locate train/test directories under base_dir={base_dir}. "
        f"Checked train: {train_dir_candidates}, test: {test_dir_candidates}"
    )



## === cell 2
train_df["has_cactus"] = train_df["has_cactus"].astype(str)

batch_size = 64
train_size = 14000
validation_size = 3500

datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    horizontal_flip=True,
    vertical_flip=False,
    validation_split=0.2,
)

data_args = {
    "dataframe": train_df,
    "directory": train_dir,
    "x_col": "id",
    "y_col": "has_cactus",
    "shuffle": True,
    "target_size": (32, 32),
    "batch_size": batch_size,
    "class_mode": "binary",
}

train_generator = datagen.flow_from_dataframe(**data_args, subset="training")
validation_generator = datagen.flow_from_dataframe(**data_args, subset="validation")



## === cell 3
train_df.head(30)



## === cell 4
sample_path = os.path.join(train_dir, train_df["id"].iloc[0])
print("Sample path:", sample_path, "exists:", os.path.isfile(sample_path))

if os.path.isfile(sample_path):
    img_bgr = cv2.imread(sample_path, cv2.IMREAD_COLOR)
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray_img = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2GRAY)

    kernel = np.ones((3, 3), np.float32) / 9.0
    dst = cv2.filter2D(img_rgb, -1, kernel)

    plt.subplot(121), plt.imshow(gray_img, cmap=plt.cm.gray_r), plt.title("Original")
    plt.xticks([]), plt.yticks([])
    plt.subplot(122), plt.imshow(dst / 255.0), plt.title("Averaging")
    plt.xticks([]), plt.yticks([])
    plt.show()



## === cell 5
model = Sequential()
model.add(Conv2D(128, (3, 3), activation="relu", input_shape=(32, 32, 3)))
model.add(BatchNormalization())
model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.3))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D(2, 2))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=256, activation="relu"))
model.add(Dropout(0.4))
model.add(Dense(units=1, activation="sigmoid"))

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
)
model.summary()



## === cell 6
ckpt_path = "aerial_cactus_detection.keras"

earlystop = EarlyStopping(
    monitor="val_accuracy", patience=15, verbose=1, restore_best_weights=False
)
reducelr = ReduceLROnPlateau(
    monitor="val_accuracy", factor=0.5, patience=3, verbose=1, min_lr=1e-6
)
modelckpt_cb = ModelCheckpoint(
    ckpt_path, monitor="val_accuracy", verbose=1, save_best_only=True, mode="max"
)
tb = TensorBoard()

callbacks = [earlystop, reducelr, modelckpt_cb, tb]



## === cell 7
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    steps_per_epoch=train_size // batch_size,
    validation_steps=validation_size // batch_size,
    epochs=50,
    verbose=1,
    shuffle=True,
    callbacks=callbacks,
)



## === cell 8
epochs = [i for i in range(1, len(history.history["loss"]) + 1)]

plt.plot(epochs, history.history["loss"], color="blue", label="training_loss")
plt.plot(epochs, history.history["val_loss"], color="red", label="validation_loss")
plt.legend(loc="best")
plt.title("loss")
plt.xlabel("epoch")
plt.show()

plt.plot(epochs, history.history["accuracy"], color="blue", label="training_accuracy")
plt.plot(
    epochs, history.history["val_accuracy"], color="red", label="validation_accuracy"
)
plt.legend(loc="best")
plt.title("accuracy")
plt.xlabel("epoch")
plt.show()



## === cell 9
test_df = pd.read_csv(sample_sub_path)
print(test_df.head())

test_ds = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels=None,
    label_mode=None,
    image_size=(32, 32),
    batch_size=256,
    shuffle=False,
)
test_ds = test_ds.map(lambda x: tf.cast(x, tf.float32) / 255.0)

pred = model.predict(test_ds, verbose=1).reshape(-1)

if len(pred) != len(test_df):
    files = []
    for fname in os.listdir(test_dir):
        if fname.lower().endswith(".jpg"):
            files.append(fname)
    files = sorted(files)
    if len(files) != len(test_df):
        raise ValueError(
            f"Mismatch: model predicted {len(pred)} images; sample_submission has {len(test_df)} rows; "
            f"test_dir contains {len(files)} jpg files."
        )
    paths = [os.path.join(test_dir, f) for f in files]
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(path):
        img = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.resize(img, [32, 32], method="bilinear")
        img = tf.cast(img, tf.float32) / 255.0
        return img

    test_ds2 = path_ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE).batch(256)
    pred = model.predict(test_ds2, verbose=1).reshape(-1)

test_df["has_cactus"] = pred.astype(np.float32)

out_path = "submission.csv"
test_df[["id", "has_cactus"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(test_df))
print(test_df.head())
