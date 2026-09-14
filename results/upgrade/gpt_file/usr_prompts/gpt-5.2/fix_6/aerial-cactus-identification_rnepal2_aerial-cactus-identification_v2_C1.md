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
pillow==11.3.0
protobuf==6.33.0
scikit-image==0.25.2
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

0.8526

# 6. Current score

0.97091

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61905) has done: 'I fix the initial import crash caused by an incompatibility between `tf_keras` (and its protobuf usage) and the installed protobuf version by switching the code to use `tensorflow.keras` consistently. Then I fix the path resolution bug that made `read_image()` fall back to a non-existent `/kaggle/working/test` path, ensuring it always loads from the correct `/kaggle/input/aerial-cactus-identification/{train,test}` folders. Finally, I ensure the model is actually created/trained before prediction, and that the written `submission.csv` has exactly the required columns (`id,has_cactus`) and correct row alignment with `sample_submission.csv`.'
- What this solution (achieved 0.97091) has done: 'The timeout is dominated by (1) eager Python loops loading 17.5k images into DataFrames, (2) keeping all images as Python objects (huge overhead), and (3) training NASNetMobile from scratch on CPU without any input pipeline optimizations. The changes below keep the exact same model, loss, optimizer, epochs, and validation semantics, but replace the slow image-loading loops with a cached, parallel `tf.data` pipeline, and feed `model.fit` / `model.predict` from that pipeline. This removes the biggest constant factors (Python/DF object overhead and single-threaded disk decode), reduces memory pressure, and keeps determinism seeds/ops intact.'

# 9. Code solution

## === cell 0
import os, math, time, random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import cv2
from glob import glob
import tensorflow as tf
from sklearn.utils import shuffle

from skimage.io import imread
from skimage import io
from skimage.transform import resize
from skimage import data, color

from PIL import Image as pil_image

import warnings

warnings.filterwarnings("ignore")

from tensorflow.keras import optimizers
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import MaxPooling2D, BatchNormalization, Flatten
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    Activation,
    MaxPool2D,
    AveragePooling2D,
)
from tensorflow.keras.layers import (
    GlobalAveragePooling2D,
    Dense,
    Dropout,
    GlobalMaxPooling2D,
)
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.nasnet import NASNetMobile
from tensorflow.keras.applications.vgg16 import VGG16

from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ["TF_DETERMINISTIC_OPS"] = "1"

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = 32  # in the given original size




## === cell 2
BASE_DIR = "/kaggle/input/aerial-cactus-identification"
TRAIN_DIR = os.path.join(BASE_DIR, "train")
TEST_DIR = os.path.join(BASE_DIR, "test")
TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")

print("BASE_DIR exists:", os.path.exists(BASE_DIR))
print("train images:", len(os.listdir(TRAIN_DIR)))
print("test images:", len(os.listdir(TEST_DIR)))




## === cell 3
train_folder = TRAIN_DIR
test_folder = TEST_DIR
train_df = pd.read_csv(TRAIN_CSV)
train_df.head()




## === cell 4
train_images_path = glob(os.path.join(TRAIN_DIR, "*.jpg"))
test_images_path = glob(os.path.join(TEST_DIR, "*.jpg"))
len(train_images_path), len(test_images_path)




## === cell 5
def expand_path(path):
    if os.path.isfile(path):
        return path

    p_train = os.path.join(TRAIN_DIR, path)
    if os.path.isfile(p_train):
        return p_train

    p_test = os.path.join(TEST_DIR, path)
    if os.path.isfile(p_test):
        return p_test

    raise FileNotFoundError(f"Could not resolve image path: {path}")


def pil_image_load(image):
    image_path = expand_path(image)
    image = pil_image.open(image_path)
    return image.resize((IMG_SIZE, IMG_SIZE))




## === cell 6
train_df.head()




## === cell 7
def read_image(img_path, resized_shape=None):
    img_path = expand_path(img_path)
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)  # BGR uint8
    if img is None:
        raise FileNotFoundError(f"Could not read image: {img_path}")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    if resized_shape:
        if img.shape[0] != resized_shape or img.shape[1] != resized_shape:
            img = cv2.resize(
                img, (resized_shape, resized_shape), interpolation=cv2.INTER_AREA
            )
    else:
        if img.shape[0] != IMG_SIZE or img.shape[1] != IMG_SIZE:
            img = cv2.resize(img, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_AREA)

    return img.astype(np.float32) / 255.0




## === cell 8
train_df = pd.read_csv(TRAIN_CSV)
train_df["filepath"] = train_df["id"].map(lambda x: os.path.join(TRAIN_DIR, x))
train_df.head()




## === cell 9
test_df = pd.DataFrame(columns=["id", "filepath"])
test_df["id"] = sorted([f for f in os.listdir(TEST_DIR) if f.lower().endswith(".jpg")])
test_df["filepath"] = test_df["id"].map(lambda x: os.path.join(TEST_DIR, x))
test_df.head()




## === cell 10
test_df.head()




## === cell 11
DO_PLOT = False

if DO_PLOT:
    random.shuffle(train_images_path)
    fig, ax = plt.subplots(2, 5, figsize=(15, 6))
    fig.suptitle("Some aerial images", fontsize=16)

    df = shuffle(train_df[["id", "has_cactus"]], random_state=SEED)
    for i, item in enumerate(df.values[15:20]):
        image = pil_image.open(expand_path(item[0]))
        ax[0, i].imshow(image)
        ax[0, i].set_title("Has Cactus = %d" % (item[1]))
    ax[0, 0].set_ylabel("train images", size="large")

    for i, path in enumerate(test_images_path[:5]):
        image = pil_image.open(path)
        ax[1, i].imshow(image)
    ax[1, i].imshow(image)
    ax[1, 0].set_ylabel("test images", size="large")




## === cell 12
def CNN():
    model = Sequential()
    model.add(Conv2D(256, (3, 3), strides=(1, 1), input_shape=(IMG_SIZE, IMG_SIZE, 3)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(128, (3, 3), strides=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Conv2D(256, (3, 3), strides=(1, 1)))
    model.add(Activation("relu"))
    model.add(MaxPooling2D((2, 2)))

    model.add(Flatten())
    model.add(Dense(512, activation="relu"))

    model.add(Dense(1, activation="sigmoid"))
    model.compile(
        loss="binary_crossentropy", optimizer=optimizers.RMSprop(), metrics=["accuracy"]
    )
    return model




## === cell 13
def NASNetMoibleClassifier():
    inputs = Input((IMG_SIZE, IMG_SIZE, 3))
    base_model = NASNetMobile(include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3))
    x = base_model(inputs)

    out1 = GlobalMaxPooling2D()(x)
    out2 = GlobalAveragePooling2D()(x)
    out3 = Flatten()(x)

    out = Concatenate(axis=-1)([out1, out2, out3])
    out = Dropout(0.5)(out)

    out = Dense(1, activation="sigmoid")(out)

    model = Model(inputs, out)
    model.compile(
        optimizer=Adam(0.0001), loss="binary_crossentropy", metrics=["accuracy"]
    )
    model.summary()
    return model




## === cell 14
AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # uint8 RGB
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_train_dataset(df, batch_size=128):
    paths = df["filepath"].values
    labels = df["has_cactus"].values.astype(np.float32).reshape(-1, 1)
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.shuffle(buffer_size=len(df), seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(
        lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=AUTOTUNE
    )
    ds = ds.cache()  # caches decoded tensors in memory for multi-epoch training
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_dataset(df, batch_size=256):
    paths = df["filepath"].values
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds




## === cell 15
model = NASNetMoibleClassifier()
model.summary()




## === cell 16
def VGGModel():
    model_vg = VGG16(
        weights="imagenet", include_top=False, input_shape=(IMG_SIZE, IMG_SIZE, 3)
    )

    model = Sequential()
    model.add(model_vg)
    model.add(Flatten())
    model.add(Dense(256))
    model.add(Activation("relu"))
    model.add(Dropout(0.5))
    model.add(Dense(1))
    model.add(Activation("sigmoid"))

    model.compile(
        optimizer=Adam(learning_rate=1e-5),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()
    return model




## === cell 17
def train_batch(train_df):
    x_train = np.stack(train_df["image"].values).astype(np.float32, copy=False)
    y_train = (
        train_df["has_cactus"].values.reshape(-1, 1).astype(np.float32, copy=False)
    )
    return x_train, y_train




## === cell 18
def train_model(model, train_df, epochs=5, verbose=None, batch_size=128):
    begin = time.time()

    checkpoint_path = "best_model.keras"
    checkpointer = ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        verbose=0,
        save_best_only=True,
        mode="max",
    )
    early_stopping = EarlyStopping(
        monitor="val_accuracy",
        verbose=1,
        patience=5,
        mode="max",
        restore_best_weights=False,
    )

    df_shuf = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
    n_val = max(1, int(round(0.05 * len(df_shuf))))
    val_df = df_shuf.iloc[:n_val]
    tr_df = df_shuf.iloc[n_val:]

    train_ds = make_train_dataset(tr_df, batch_size=batch_size)
    val_ds = make_train_dataset(
        val_df, batch_size=batch_size
    )  # shuffle inside; for val we disable shuffle below

    val_paths = val_df["filepath"].values
    val_labels = val_df["has_cactus"].values.astype(np.float32).reshape(-1, 1)
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_labels))
    val_ds = val_ds.map(
        lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=AUTOTUNE
    )
    val_ds = val_ds.cache().batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    model.fit(
        train_ds,
        epochs=epochs,
        verbose=verbose,
        callbacks=[checkpointer, early_stopping],
        validation_data=val_ds,
        shuffle=False,  # dataset already shuffled; keeps semantics while avoiding extra overhead
    )

    elapsed = time.time() - begin
    print("total training time: ", elapsed)
    return model, checkpoint_path




## === cell 19
model, checkpoint_path = train_model(
    model, train_df, epochs=30, verbose=1, batch_size=128
)




## === cell 20
if os.path.exists(checkpoint_path):
    model = tf.keras.models.load_model(checkpoint_path)




## === cell 21
test_ds = make_test_dataset(test_df, batch_size=256)
test_steps = int(math.ceil(len(test_df) / 256))
y_pred = model.predict(test_ds, verbose=0, steps=test_steps).reshape(-1)
y_pred[:5], float(y_pred.min()), float(y_pred.max())




## === cell 22
sample = pd.read_csv(SAMPLE_SUB)

submission = pd.DataFrame({"id": test_df["id"].values})
submission["has_cactus"] = np.clip(y_pred.astype(np.float32), 0.0, 1.0)

submission = sample[["id"]].merge(submission, on="id", how="left")
submission["has_cactus"] = submission["has_cactus"].fillna(0.5).astype(np.float32)
submission = submission[["id", "has_cactus"]]
submission.head()




## === cell 23
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.columns.tolist())
print(submission.head())
