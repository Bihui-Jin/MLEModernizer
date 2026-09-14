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

3.8

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
from __future__ import absolute_import, division, print_function, unicode_literals

import os, gc, math
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras.callbacks import ReduceLROnPlateau, ModelCheckpoint
from matplotlib import pyplot as plt
from keras.utils import to_categorical

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
np.random.seed(42)
tf.random.set_seed(42)

tf.config.optimizer.set_jit(True)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

gc.collect()



## === cell 1
train_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
train_df = train_df.sample(frac=1, random_state=42).reset_index(drop=True)

file_name = train_df["image_id"].to_numpy()
train_y = train_df[["healthy", "multiple_diseases", "rust", "scab"]].to_numpy(
    dtype=np.int64
)

N = len(file_name)
H, W = 273, 410
base_img_dir = "/kaggle/input/plant-pathology-2020-fgvc7/images/"

train_paths = np.char.add(
    np.char.add(base_img_dir, file_name.astype(str)), ".jpg"
).astype(object)

gc.collect()



## === cell 2
training_y = train_y.argmax(axis=1).astype(np.int64)
gc.collect()



## === cell 3
gc.collect()

train_paths_split = train_paths[0:1120]
train_y_split = training_y[0:1120]

val_paths = train_paths[1120:1821]
test_Y = training_y[1120:1821]

gc.collect()



## === cell 4
gc.collect()

backbone = tf.keras.applications.Xception(
    weights="imagenet", include_top=False, input_shape=(273, 410, 3)
)
backbone.trainable = False

model = tf.keras.Sequential(
    [
        backbone,
        tf.keras.layers.GlobalAveragePooling2D(),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.5),
        tf.keras.layers.Dense(512, activation=tf.nn.relu),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(4, activation=tf.nn.softmax),
    ]
)

model.compile(
    optimizer=tf.keras.optimizers.Adamax(),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
    jit_compile=True,
)



## === cell 5
y_binary = to_categorical(
    training_y
)  # kept for parity (not used later, as in original)
y_binary_train = to_categorical(train_y_split)
y_binary_test = to_categorical(test_Y)



## === cell 6
gc.collect()
annealer = ReduceLROnPlateau(
    monitor="accuracy", factor=0.5, patience=5, verbose=1, min_lr=1e-5
)
checkpoint = ModelCheckpoint("model.h5", verbose=1, save_best_only=True)


class myCallback(tf.keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs={}):
        if logs.get("accuracy") > 0.99:
            print("\nReached 99.9% accuracy so cancelling training!")
            self.model.stop_training = True




## === cell 7
gc.collect()

batch_size = 16
steps_per_epoch = math.ceil(len(train_paths_split) / batch_size)

augment = tf.keras.Sequential(
    [
        tf.keras.layers.RandomRotation(factor=1.0, fill_mode="nearest"),
        tf.keras.layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="nearest"
        ),
        tf.keras.layers.RandomZoom(
            height_factor=(-0.2, 0.2), width_factor=(-0.2, 0.2), fill_mode="nearest"
        ),
        tf.keras.layers.RandomFlip(mode="horizontal"),
    ],
    name="augmentation",
)

AUTOTUNE = tf.data.AUTOTUNE

xception_preprocess = tf.keras.applications.xception.preprocess_input


def _load_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [H, W], method=tf.image.ResizeMethod.AREA, antialias=False
    )
    img = tf.cast(img, tf.float32)
    img = xception_preprocess(img)
    return img


def _train_map_from_img(x, y):
    x = augment(x, training=True)
    return x, y


def _val_map_from_img(x, y):
    return x, y


options = tf.data.Options()
options.experimental_deterministic = True

options.experimental_optimization.apply_default_optimizations = True

train_base_ds = (
    tf.data.Dataset.from_tensor_slices(train_paths_split)
    .with_options(options)
    .map(_load_and_resize, num_parallel_calls=AUTOTUNE)
    .cache()
)

val_base_ds = (
    tf.data.Dataset.from_tensor_slices(val_paths)
    .with_options(options)
    .map(_load_and_resize, num_parallel_calls=AUTOTUNE)
    .cache()
)

train_ds = (
    tf.data.Dataset.zip(
        (
            train_base_ds,
            tf.data.Dataset.from_tensor_slices(y_binary_train.astype(np.float32)),
        )
    )
    .with_options(options)
    .shuffle(len(train_paths_split), seed=42, reshuffle_each_iteration=True)
    .map(_train_map_from_img, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = (
    tf.data.Dataset.zip(
        (
            val_base_ds,
            tf.data.Dataset.from_tensor_slices(y_binary_test.astype(np.float32)),
        )
    )
    .with_options(options)
    .map(_val_map_from_img, num_parallel_calls=AUTOTUNE)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    epochs=200,
    callbacks=[annealer, checkpoint],
)



## === cell 8
acc = history.history["accuracy"]
val_acc = history.history.get("val_accuracy", [])
loss = history.history["loss"]
val_loss = history.history.get("val_loss", [])

epochs = range(len(acc))

plt.plot(epochs, acc, "r", label="Training accuracy")
plt.title("Training accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()



## === cell 9
plt.plot(epochs, loss, "r", label="Training Loss")
plt.title("Training Loss")
plt.legend(loc=0)
plt.figure()
plt.show()



## === cell 10
test_df = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")
test_ids = test_df["image_id"].to_numpy()
Nt = len(test_ids)

test_paths = np.char.add(
    np.char.add(base_img_dir, test_ids.astype(str)), ".jpg"
).astype(object)

gc.collect()




## === cell 11
def _test_map(path):
    return _load_and_resize(path)


test_ds = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .with_options(options)
    .map(_test_map, num_parallel_calls=AUTOTUNE)
    .batch(32, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

results = model.predict(test_ds, verbose=0)



## === cell 12
df = pd.DataFrame(results, columns=["healthy", "multiple_diseases", "rust", "scab"])



## === cell 13
df.insert(0, "image_id", test_ids, False)



## === cell 14
df.to_csv("submission.csv", index=False)



## === cell 15
df
