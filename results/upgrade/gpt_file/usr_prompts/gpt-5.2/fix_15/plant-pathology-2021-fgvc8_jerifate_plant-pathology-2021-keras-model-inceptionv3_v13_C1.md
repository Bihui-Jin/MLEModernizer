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
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
missingno==0.5.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split

import matplotlib.pyplot as plt

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

tf.config.run_functions_eagerly(False)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
os.listdir("/kaggle/input/plant-pathology-2021-fgvc8/")



## === cell 2
train_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
test_df = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
print("Dataset Shape: ", train_df.shape)
train_df.head()



## === cell 3
train_images_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
print("Train images dir:", train_images_dir)




## === cell 4
def add_link(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + str(path)




## === cell 5
def add_link_test(path):
    return "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + str(path)




## === cell 6
train_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/" + train_df[
    "image"
].astype(str)
train_df.head()



## === cell 7
test_df["image"] = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/" + test_df[
    "image"
].astype(str)

test_df.head()



## === cell 8
label_counts = train_df["labels"].value_counts()
print("Top-10 label combinations:\n", label_counts.head(10))



## === cell 9
unique_list = np.unique(train_df["labels"])
print(unique_list[:10], "...")
print("Number of unique label combinations:", train_df["labels"].value_counts().count())




## === cell 10
def read_image(path):
    gfile = tf.io.read_file(path)
    image = tf.io.decode_jpeg(gfile, channels=3)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, (224, 224))
    return image




## === cell 11
_image_to_label = dict(zip(train_df["image"].values, train_df["labels"].values))


def get_label(path):
    return_label = _image_to_label.get(path, None)
    print(return_label)
    return [return_label] if return_label is not None else []




## === cell 12
def get_label_image(path):
    label = get_label(path)
    image = read_image(path)
    return label, image




## === cell 13
try:
    pass
except Exception as e:
    print("Skipped sample visualization due to:", repr(e))



## === cell 14
INPUT_SIZE = (224, 224, 3)
BATCH_SIZE = 32

classes = sorted(train_df["labels"].unique().tolist())
CLASSES = len(classes)
print("Number of classes (unique label combinations):", CLASSES)



## === cell 15
train_data, val_data = train_test_split(
    train_df, test_size=0.2, random_state=SEED, shuffle=True
)
print("Train Data Shape: ", train_data.shape)
print("Validation Data Shape: ", val_data.shape)



## === cell 16
from tensorflow.keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(
    rescale=1 / 255.0, width_shift_range=0.3, zoom_range=0.2, horizontal_flip=True
)
test_datagen = ImageDataGenerator(rescale=1 / 255.0)
val_datagen = ImageDataGenerator(rescale=1 / 255.0)



## === cell 17
pre_model = DenseNet121(include_top=False, weights="imagenet", input_shape=INPUT_SIZE)
pre_model.trainable = False



## === cell 18
model = tf.keras.Sequential()
model.add(pre_model)
model.add(layers.Flatten())
model.add(layers.Dense(512, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(256, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(128, activation="relu"))
model.add(layers.Dropout(0.3))
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(CLASSES, activation="softmax"))



## === cell 19
callback = ReduceLROnPlateau(monitor="val_loss", factor=0.01, patience=3, min_lr=1e-5)



## === cell 20
model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

steps_per_epoch = int(np.ceil(len(train_data) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_data) / BATCH_SIZE))

feature_shape = pre_model.output_shape[1:]  # e.g. (7, 7, 1024)
head_input = keras.Input(shape=feature_shape)
x = head_input
for lyr in model.layers[1:]:
    x = lyr(x)
head_model = keras.Model(head_input, x, name="head_model")

head_model.compile(
    optimizer=keras.optimizers.SGD(learning_rate=0.001, momentum=0.9, nesterov=False),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=False,
)

feature_extractor = keras.Model(
    pre_model.input, pre_model.output, name="feature_extractor"
)

AUTOTUNE = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(classes)}
train_label_idx = train_data["labels"].map(class_to_idx).to_numpy(np.int32)
val_label_idx = val_data["labels"].map(class_to_idx).to_numpy(np.int32)


def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img, INPUT_SIZE[:2], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    return img


augmenter = keras.Sequential(
    [
        layers.Rescaling(1.0),  # image already in [0,1]; keep explicit
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.0, width_factor=0.3, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.2, 0.2),
            width_factor=(-0.2, 0.2),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


def _make_ds(paths, training=False, augment=False, cache=False):
    paths = tf.convert_to_tensor(paths, dtype=tf.string)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    if training:
        buffer_size = (
            int(paths.shape[0])
            if paths.shape.rank == 1 and paths.shape[0] is not None
            else 2048
        )
        ds = ds.shuffle(
            buffer_size=buffer_size, seed=SEED, reshuffle_each_iteration=True
        )

    def _load_x(path):
        x = _decode_resize(path)
        if augment:
            x = augmenter(x, training=True)
        return x

    ds = ds.map(_load_x, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache:
        ds = ds.cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths = train_data["image"].to_numpy()
val_paths = val_data["image"].to_numpy()

train_ds_img = _make_ds(train_paths, training=False, augment=False, cache=False)
val_ds_img = _make_ds(val_paths, training=False, augment=False, cache=True)

print("Extracting train bottlenecks...")
train_bneck = feature_extractor.predict(train_ds_img, verbose=1)
print("Extracting val bottlenecks...")
val_bneck = feature_extractor.predict(val_ds_img, verbose=1)

y_train_oh = tf.one_hot(train_label_idx, depth=CLASSES, dtype=tf.float32).numpy()
y_val_oh = tf.one_hot(val_label_idx, depth=CLASSES, dtype=tf.float32).numpy()

history = head_model.fit(
    train_bneck,
    y_train_oh,
    epochs=25,
    batch_size=BATCH_SIZE,
    validation_data=(val_bneck, y_val_oh),
    callbacks=[callback],
    verbose=1,
)



## === cell 21
try:
    pass
except Exception as e:
    print("Skipped training plots due to:", repr(e))



## === cell 22
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
)
submission.head()



## === cell 23
test_paths = test_df["image"].to_numpy()
test_n = len(test_paths)

test_ds_img = _make_ds(test_paths, training=False, augment=False, cache=True)

print("Extracting test bottlenecks...")
test_bneck = feature_extractor.predict(test_ds_img, verbose=1)

preds = head_model.predict(
    test_bneck,
    verbose=1,
    batch_size=BATCH_SIZE,
)

idx_to_class = np.asarray(classes, dtype=object)

test_pred_idx = np.argmax(preds, axis=-1)
test_pred_labels = idx_to_class[test_pred_idx]

submission["image"] = [os.path.basename(p) for p in test_paths]
submission["labels"] = test_pred_labels
submission = submission[["image", "labels"]]

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
