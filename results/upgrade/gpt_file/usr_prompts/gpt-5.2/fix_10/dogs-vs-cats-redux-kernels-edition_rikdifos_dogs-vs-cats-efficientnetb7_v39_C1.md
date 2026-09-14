# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Code solution

## === cell 0
import os
import sys
import warnings

warnings.filterwarnings("ignore")

import cv2, re, random, time, gc
import numpy as np
import pandas as pd

import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.metrics import log_loss, accuracy_score
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.optimizers import RMSprop, Adam

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

SEED = 558
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

plt.rcParams["figure.facecolor"] = "white"
SHOW_PLOTS = False  # keep code but avoid spending time rendering during submission runs

print("TF:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))



## === cell 1
start = time.time()

PATH = "/kaggle/input/dogs-vs-cats-redux-kernels-edition/"

TRAIN_DIR_CAT = os.path.join(PATH, "train", "cat")
TRAIN_DIR_DOG = os.path.join(PATH, "train", "dog")
TEST_DIR = os.path.join(PATH, "test", "unknown")

if not os.path.isdir(TRAIN_DIR_CAT):
    raise FileNotFoundError(f"TRAIN_DIR_CAT not found: {TRAIN_DIR_CAT}")
if not os.path.isdir(TRAIN_DIR_DOG):
    raise FileNotFoundError(f"TRAIN_DIR_DOG not found: {TRAIN_DIR_DOG}")
if not os.path.isdir(TEST_DIR):
    raise FileNotFoundError(f"TEST_DIR not found: {TEST_DIR}")

train_images = []
for d in (TRAIN_DIR_CAT, TRAIN_DIR_DOG):
    with os.scandir(d) as it:
        for e in it:
            name = e.name
            if e.is_file() and name.endswith(".jpg"):
                train_images.append(e.path)

test_images = []
with os.scandir(TEST_DIR) as it:
    for e in it:
        name = e.name
        if e.is_file() and name.endswith(".jpg"):
            test_images.append(e.path)

print("Resolved TRAIN_DIR_CAT:", TRAIN_DIR_CAT)
print("Resolved TRAIN_DIR_DOG:", TRAIN_DIR_DOG)
print("Resolved TEST_DIR     :", TEST_DIR)
print("Found train images:", len(train_images))
print("Found test images :", len(test_images))




## === cell 2
def txt_dig(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [txt_dig(c) for c in re.split(r"(\d+)", os.path.basename(text))]


train_images.sort(key=natural_keys)
test_images.sort(key=natural_keys)

if len(train_images) >= 25000:
    train_images = train_images[0:7500] + train_images[17500:25000]
else:
    train_images = train_images[: min(len(train_images), 15000)]

random.seed(558)
random.shuffle(train_images)

print("Train images after sampling:", len(train_images))



## === cell 3
IMG_WIDTH = 128
IMG_HEIGHT = 128


def _label_from_path_py(p: str) -> int:
    base = os.path.basename(p).lower()
    parent = os.path.basename(os.path.dirname(p)).lower()
    if ("dog" in base) or (parent == "dog"):
        return 1
    if ("cat" in base) or (parent == "cat"):
        return 0
    return -1


y = np.asarray([_label_from_path_py(p) for p in train_images], dtype=np.int32)
valid_mask = y >= 0
if not np.all(valid_mask):
    train_images = [p for p, m in zip(train_images, valid_mask) if m]
    y = y[valid_mask]

print("Train images after label filtering:", len(train_images))
print("Labels shape:", y.shape)

if SHOW_PLOTS and len(y) > 0:
    sns.countplot(x=y)
    plt.show()



## === cell 4
if SHOW_PLOTS:
    import cv2

    random.seed(558)
    plt.figure(figsize=(10, 4), facecolor="white")
    if len(train_images) > 0:
        for idx in range(3):
            ridx = random.randrange(len(train_images))
            im = cv2.imread(train_images[ridx])
            if im is None:
                continue
            im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, 3, idx + 1)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis("off")
    plt.tight_layout()
    plt.show()



## === cell 5
if SHOW_PLOTS:
    import cv2

    plt.figure(figsize=(10, 4), facecolor="white")
    if len(train_images) > 0:
        idxs = [
            min(1024, len(train_images) - 1),
            min(546, len(train_images) - 1),
            min(742, len(train_images) - 1),
        ]
        for j, idx in enumerate(idxs):
            im = cv2.imread(train_images[idx])
            if im is None:
                continue
            im = cv2.resize(im, (IMG_WIDTH, IMG_HEIGHT), interpolation=cv2.INTER_CUBIC)
            plt.subplot(1, 3, j + 1)
            plt.imshow(cv2.cvtColor(im, cv2.COLOR_BGR2RGB))
            plt.axis("off")
        plt.tight_layout()
        plt.show()



## === cell 6
if len(train_images) == 0:
    raise RuntimeError(
        "No training images were found/kept after filtering; cannot proceed."
    )

train_paths, val_paths, y_train, y_val = train_test_split(
    np.asarray(train_images),
    y,
    test_size=0.2,
    random_state=2020,
    stratify=y,
)

print(train_paths.shape, val_paths.shape, y_train.shape, y_val.shape)



## === cell 7
model = models.Sequential()

efnModel = tf.keras.applications.EfficientNetB7(
    weights="imagenet", input_shape=(IMG_WIDTH, IMG_HEIGHT, 3), include_top=False
)

model.add(efnModel)
model.add(layers.GlobalAveragePooling2D())
model.add(layers.Dense(1, activation="sigmoid"))

opt1 = RMSprop(learning_rate=1e-5, decay=1e-6)
opt2 = Adam(learning_rate=1e-4)

model.compile(
    loss="binary_crossentropy",
    optimizer=opt2,
    metrics=["accuracy"],
    jit_compile=True,
)
model.summary()



## === cell 8
BATCH_SIZE = 16

augmenter = keras.Sequential(
    [
        layers.RandomRotation(factor=40.0 / 360.0, seed=SEED),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, seed=SEED, fill_mode="nearest"
        ),
        layers.RandomZoom(
            height_factor=0.2, width_factor=0.2, seed=SEED, fill_mode="nearest"
        ),
        layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augmenter",
)

AUTOTUNE = tf.data.AUTOTUNE


def _decode_resize_scale(path, label=None):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [IMG_HEIGHT, IMG_WIDTH], method=tf.image.ResizeMethod.BICUBIC
    )
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, label


def _aug(xb, yb):
    xb = augmenter(xb, training=True)
    return xb, yb


data_opts = tf.data.Options()
data_opts.deterministic = True

train_cache_path = "/kaggle/working/train_decode_cache.tf-data"
val_cache_path = "/kaggle/working/val_decode_cache.tf-data"
test_cache_path = "/kaggle/working/test_decode_cache.tf-data"

for p in (train_cache_path, val_cache_path, test_cache_path):
    try:
        tf.io.gfile.remove(p)
    except Exception:
        pass

train_ds = tf.data.Dataset.from_tensor_slices((train_paths, y_train)).with_options(
    data_opts
)
train_ds = train_ds.shuffle(
    buffer_size=len(train_paths), seed=SEED, reshuffle_each_iteration=True
)
train_ds = train_ds.map(_decode_resize_scale, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.cache(train_cache_path)
train_ds = train_ds.batch(BATCH_SIZE, drop_remainder=False)
train_ds = train_ds.map(_aug, num_parallel_calls=AUTOTUNE)
train_ds = train_ds.prefetch(AUTOTUNE)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, y_val)).with_options(data_opts)
val_ds = val_ds.map(_decode_resize_scale, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.cache(val_cache_path)
val_ds = val_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

gc.collect()



## === cell 9
earlystop1 = EarlyStopping(patience=5, restore_best_weights=True)
earlystop2 = ReduceLROnPlateau(
    monitor="val_accuracy", min_lr=1e-6, patience=5, mode="max", verbose=1
)


def _steps_from_count(n, batch_size):
    return int((int(n) + int(batch_size) - 1) // int(batch_size))


steps_per_epoch = _steps_from_count(len(train_paths), BATCH_SIZE)
validation_steps = _steps_from_count(len(val_paths), BATCH_SIZE)

history = model.fit(
    train_ds,
    steps_per_epoch=steps_per_epoch,
    epochs=20,
    validation_data=val_ds,
    callbacks=[earlystop1, earlystop2],
    validation_steps=validation_steps,
    verbose=1,
)



## === cell 10
if SHOW_PLOTS:
    plt.rcParams["figure.facecolor"] = "white"
    model_loss = pd.DataFrame(history.history)
    print(model_loss.head())
    ax = model_loss[["accuracy", "val_accuracy"]].plot(ylim=[0, 1])
    ax.figure.show()
    ax2 = model_loss[["loss", "val_loss"]].plot()
    ax2.figure.show()



## === cell 11
val_x_ds = val_ds.map(lambda xb, yb: xb, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

val_preds = model.predict(val_x_ds, steps=validation_steps, verbose=0).ravel()
val_preds = np.clip(val_preds, 1e-7, 1 - 1e-7)
val_preds_class = (val_preds > 0.5).astype(int)

print("Out of Fold Accuracy is {:.5f}".format(accuracy_score(y_val, val_preds_class)))
print("Out of Fold log loss is {:.5f}".format(log_loss(y_val, val_preds)))



## === cell 12
test_ds = tf.data.Dataset.from_tensor_slices(np.asarray(test_images)).with_options(
    data_opts
)
test_ds = test_ds.map(
    lambda p: _decode_resize_scale(p, None), num_parallel_calls=AUTOTUNE
)
test_ds = test_ds.cache(test_cache_path)
test_ds = test_ds.batch(64, drop_remainder=False).prefetch(AUTOTUNE)

test_steps = _steps_from_count(len(test_images), 64)
test_pred = model.predict(test_ds, steps=test_steps, verbose=0).ravel()
test_pred = np.clip(test_pred, 1e-7, 1 - 1e-7)

test_ids = np.fromiter(
    (int(os.path.splitext(os.path.basename(p))[0]) for p in test_images),
    dtype=np.int32,
    count=len(test_images),
)

submission = pd.DataFrame({"id": test_ids, "label": test_pred})
submission = submission.sort_values("id").reset_index(drop=True)

submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Time elapsed: {:.2f} seconds".format(time.time() - start))
print(submission.head())



## === cell 13
import shutil

shutil.rmtree("/kaggle/working/data/", ignore_errors=True)
