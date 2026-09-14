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

# 5. Target score

0.7953659417082056

# 6. Current score

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.43635) has done: 'The timeout is dominated by Python-side augmentation inside `tf.data` (`tf.numpy_function` + `ImageDataGenerator.random_transform`), which prevents graph optimizations and keeps the input pipeline on the CPU with high per-step overhead. To preserve the exact training semantics while making it fast, I keep the same augmentation logic but switch to Keras’ built-in `ImageDataGenerator.flow(...)`, which performs the same transforms in optimized C/NumPy code and feeds batches directly to `model.fit` without `tf.numpy_function`. I also remove expensive “shuffle the entire dataset buffer” in `tf.data` and avoid repeated dtype conversions by filling arrays as `float32` once and scaling in-place. These changes keep the same model, loss, metrics, callbacks, and augmentation parameters, but drastically reduce per-epoch overhead so the run fits within 600 seconds.'
- What this solution (achieved 0.5) has done: 'The timeout is dominated by (1) slow Python-loop image loading/resizing with PIL and (2) long training caused by feeding a fully materialized NumPy array through `ImageDataGenerator`. To preserve the exact model and training semantics, I keep the same architecture, optimizer, loss, and augmentation policy, but move image decoding/resizing into an efficient, parallel `tf.data` pipeline and keep using the same `ImageDataGenerator` transforms via `tf.numpy_function` (so the augmentation behavior remains the same). I also remove non-essential display cells, add caching/prefetching, and switch prediction to a streaming dataset so we never hold all test images in RAM. These changes reduce wall time substantially without changing what the model learns (only negligible float-level differences are possible).'

# 9. Code solution

## === cell 0
from tqdm import tqdm
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt

import random
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

train_ids = train["image_id"].values
test_ids = test["image_id"].values

train_paths = np.char.add(np.char.add(IMG_DIR + "/", train_ids), ".jpg").astype(str)
test_paths = np.char.add(np.char.add(IMG_DIR + "/", test_ids), ".jpg").astype(str)

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




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1288147508.py in <cell line: 0>()
     16 test_ids = test["image_id"].values
     17 
---> 18 train_paths = np.char.add(np.char.add(IMG_DIR + "/", train_ids), ".jpg").astype(str)
     19 test_paths = np.char.add(np.char.add(IMG_DIR + "/", test_ids), ".jpg").astype(str)
     20 

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U48' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 7
train_label = train.loc[:, "healthy":"scab"]




## === cell 8
train_label = np.asarray(train_label, dtype=np.float32)
train_label = train_label / np.clip(train_label.sum(axis=1, keepdims=True), 1.0, None)




## === cell 9
print("n_train:", len(train_paths))
print("n_test :", len(test_paths))
print("train_label shape:", train_label.shape)




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/3607592153.py in <cell line: 0>()
----> 1 print("n_train:", len(train_paths))
      2 print("n_test :", len(test_paths))
      3 print("train_label shape:", train_label.shape)
      4 
      5 

NameError: name 'train_paths' is not defined

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
        sd = int(sd_np[0]) ^ (int(sd_np[1]) << 1)
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




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/287555571.py in <cell line: 0>()
      2 # transforms via numpy_function. Correctness: augmentation policy stays identical to datagen.
      3 BATCH_SIZE = 32
----> 4 steps_per_epoch = int(np.ceil(len(train_paths) / BATCH_SIZE))
      5 
      6 # Create a deterministic stateless seed stream for per-example augmentation.

NameError: name 'train_paths' is not defined

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
plot_loss(history, "Training Dataset")
plot_acc(history, "Training Dataset")




## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/2796718703.py in <cell line: 0>()
----> 1 plot_loss(history, "Training Dataset")
      2 plot_acc(history, "Training Dataset")
      3 
      4 

NameError: name 'history' is not defined

## === cell 19
ds_test = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
    .batch(64)
    .prefetch(AUTOTUNE)
)
y_pred = model.predict(ds_test, verbose=1)
print(y_pred)




## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1077223321.py in <cell line: 0>()
      1 # Speed: Stream test images with tf.data (no giant NumPy allocation); identical preprocessing.
      2 ds_test = (
----> 3     tf.data.Dataset.from_tensor_slices(test_paths)
      4     .map(_decode_resize_normalize, num_parallel_calls=AUTOTUNE)
      5     .batch(64)

NameError: name 'test_paths' is not defined

## === cell 20
target_cols = [c for c in submission.columns if c != "image_id"]
if y_pred.shape[1] != len(target_cols):
    raise ValueError(
        f"Prediction shape {y_pred.shape} does not match submission targets {len(target_cols)}: {target_cols}"
    )
submission.loc[:, target_cols] = y_pred




## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1936523333.py in <cell line: 0>()
      1 target_cols = [c for c in submission.columns if c != "image_id"]
----> 2 if y_pred.shape[1] != len(target_cols):
      3     raise ValueError(
      4         f"Prediction shape {y_pred.shape} does not match submission targets {len(target_cols)}: {target_cols}"
      5     )

NameError: name 'y_pred' is not defined

## === cell 21
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
