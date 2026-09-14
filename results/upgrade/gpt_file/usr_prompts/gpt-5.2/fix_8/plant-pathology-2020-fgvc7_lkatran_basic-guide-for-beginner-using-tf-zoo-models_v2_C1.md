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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ["PYTHONHASHSEED"] = "0"

from tqdm import tqdm
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
import random

random.seed(0)
np.random.seed(0)

import tensorflow as tf

tf.random.set_seed(0)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE



## === cell 1
submission = pd.read_csv(
    "/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv"
)
train = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/train.csv")
test = pd.read_csv("/kaggle/input/plant-pathology-2020-fgvc7/test.csv")



## === cell 2
_ = train.head()



## === cell 3
_ = train.loc[:, "healthy":"scab"].sum(axis=0) / (train.shape[0] / 100)



## === cell 4
_ = test.head()



## === cell 5
_ = submission.head()



## === cell 6
from tensorflow.keras.utils import load_img, img_to_array


def load_resize_image_float32(image_path, target_size=(224, 224)):
    img = load_img(image_path, target_size=target_size)  # PIL under the hood
    arr = img_to_array(img)  # float32, shape (H,W,3)
    return arr


IMG_SIZE = (224, 224)
IMG_DIR = "/kaggle/input/plant-pathology-2020-fgvc7/images"

train_ids = train["image_id"].astype(str).values
test_ids = test["image_id"].astype(str).values

train_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in train_ids], dtype=object)
test_paths = np.array([f"{IMG_DIR}/{iid}.jpg" for iid in test_ids], dtype=object)

missing = [
    p
    for p in (train_paths[:5].tolist() + test_paths[:5].tolist())
    if not os.path.exists(p)
]
if missing:
    raise FileNotFoundError(f"Could not find image(s), e.g.: {missing[0]}")


def _decode_resize_normalize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return img




## === cell 7
train_label = train.loc[:, "healthy":"scab"]



## === cell 8
train_label = np.asarray(train_label, dtype=np.float32)
train_label = train_label / np.clip(train_label.sum(axis=1, keepdims=True), 1.0, None)



## === cell 9
print("n_train:", len(train_paths))
print("n_test :", len(test_paths))
print("train_label shape:", train_label.shape)



## === cell 10
from tensorflow.keras.preprocessing.image import ImageDataGenerator

datagen = ImageDataGenerator(
    featurewise_center=False,
    samplewise_center=False,
    featurewise_std_normalization=False,
    samplewise_std_normalization=False,
    zca_whitening=False,
    rotation_range=15,
    zoom_range=0.25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    vertical_flip=False,
)



## === cell 11
pass



## === cell 12
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint



## === cell 13
base_model = DenseNet201(
    include_top=False, weights="imagenet", input_shape=(224, 224, 3), pooling="avg"
)

model = Sequential()
model.add(base_model)

model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(128, activation="relu"))
model.add(Dense(4, activation="softmax"))

froze = True  # keep original behavior
if froze is True:
    base_model.trainable = False
else:
    for layer in base_model.layers[: -int(froze)]:
        layer.trainable = False

reduce_learning_rate = ReduceLROnPlateau(
    monitor="categorical_accuracy",
    factor=0.1,
    patience=2,
    cooldown=2,
    min_lr=0.0000001,
    verbose=1,
)
early_stopping = EarlyStopping(monitor="categorical_accuracy", patience=5)

check_point = ModelCheckpoint(
    filepath="resnet_50.h5", monitor="categorical_accuracy", save_best_only=True
)

callbacks = [reduce_learning_rate, early_stopping, check_point]

model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["categorical_accuracy"]
)



## === cell 14
model.summary()



## === cell 15
BATCH_SIZE = 32
steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))

_seed_gen = tf.random.Generator.from_seed(0)


def _augment_with_datagen_tf(img, label):
    seed = _seed_gen.make_seeds(2)[0]  # shape (2,), int64

    def _np_aug(im_np, sd_np):
        s0 = int(sd_np[0])
        s1 = int(sd_np[1])
        sd = (s0 ^ ((s1 & 0xFFFFFFFF) << 1)) & 0xFFFFFFFF
        np.random.seed(sd)
        return datagen.random_transform(im_np).astype(np.float32)

    aug = tf.numpy_function(_np_aug, [img, seed], Tout=tf.float32)
    aug.set_shape((IMG_SIZE[0], IMG_SIZE[1], 3))
    return aug, label


train_ds = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_label))
    .shuffle(buffer_size=len(train_paths), seed=0, reshuffle_each_iteration=True)
    .map(lambda p, y: (_decode_resize_normalize(p), y), num_parallel_calls=AUTOTUNE)
    .map(_augment_with_datagen_tf, num_parallel_calls=AUTOTUNE)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

start = dt.now()
history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=200,
    callbacks=callbacks,
    verbose=1,
)
print(
    "Время работы модели: {}. Количество эпох: {}.".format(
        dt.now() - start, len(history.epoch)
    )
)



## === cell 16
import gc

del train_label
gc.collect()




## === cell 17
def plot_loss(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history["loss"], label="train_loss")
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Loss")
    plt.legend(loc="upper right")
    plt.close()


def plot_acc(his, title):
    epoch = len(his.epoch)
    plt.style.use("ggplot")
    plt.figure()
    plt.plot(
        np.arange(0, epoch),
        his.history.get("categorical_accuracy", []),
        label="categorical_accuracy",
    )
    plt.title(title)
    plt.xlabel("Epoch #")
    plt.ylabel("Accuracy")
    plt.legend(loc="upper right")
    plt.close()




## === cell 18
if "history" in globals():
    plot_loss(history, "Training Dataset")
    plot_acc(history, "Training Dataset")



## === cell 19
ds_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .batch(64)
    .prefetch(AUTOTUNE)
)
y_pred = model.predict(ds_test, verbose=1)
print(y_pred)



## === cell 20
target_cols = [c for c in submission.columns if c != "image_id"]
if y_pred.shape[1] != len(target_cols):
    raise ValueError(
        f"Prediction shape {y_pred.shape} does not match submission targets {len(target_cols)}: {target_cols}"
    )
submission.loc[:, target_cols] = y_pred



## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
