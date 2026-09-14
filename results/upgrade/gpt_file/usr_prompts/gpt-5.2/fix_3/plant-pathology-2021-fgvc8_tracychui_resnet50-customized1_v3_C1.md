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

No external packages required in the script and installed.

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
import glob
import random
import gc

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K


K.set_image_data_format("channels_last")
print("keras:", keras.__version__, "tf:", tf.__version__)

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing dir: {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing dir: {TEST_IMG_DIR}"



## === cell 1
IMG_WIDTH = 300
IMG_HEIGHT = 300
NR_CHANNELS = 3
BATCH_SIZE = 16
EPOCHS = 2  # keep small for runtime; preserves core logic (train then predict)



## === cell 2
imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("num test images:", len(imglist_test))
assert len(imglist_test) > 0, "No test images found - check TEST_IMG_DIR"



## === cell 3
import tensorflow.keras.applications.resnet50 as resnet


def make_image_dataset_from_paths(paths, batch_size: int):
    path_ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load_only(path):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        img = tf.cast(img, tf.float32)
        img = resnet.preprocess_input(img)
        return img

    ds = path_ds.map(_load_only, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return ds


test_ds = make_image_dataset_from_paths(imglist_test, BATCH_SIZE)
print("Built test_ds for streaming inference.")



## === cell 4
first_img_bytes = tf.io.read_file(imglist_test[0])
first_img = tf.image.decode_jpeg(first_img_bytes, channels=3)
first_img = tf.image.resize(first_img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
first_img = tf.cast(first_img, tf.float32)
first_img = resnet.preprocess_input(first_img)
first_min = float(tf.reduce_min(first_img).numpy())
first_max = float(tf.reduce_max(first_img).numpy())
print("Example preprocessed pixel stats (min/max):", first_min, first_max)



## === cell 5
training_csv = pd.read_csv(TRAIN_CSV)

training_class = []
for labels in pd.unique(training_csv["labels"]):
    training_class.extend(labels.split())
tagnames = np.unique(np.array(training_class))
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("tagnames:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}


def labels_series_to_multihot(
    labels: pd.Series, tag2idx: dict, num_classes: int
) -> np.ndarray:
    y = np.zeros((len(labels), num_classes), dtype=np.float32)
    for r, s in enumerate(labels.astype(str).values):
        for t in s.split():
            j = tag2idx.get(t, None)
            if j is not None:
                y[r, j] = 1.0
    return y


idx = np.arange(len(training_csv))
np.random.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df = training_csv.iloc[trn_idx].reset_index(drop=True)
val_df = training_csv.iloc[val_idx].reset_index(drop=True)

print("train size:", len(train_df), "val size:", len(val_df))


def make_dataset(df: pd.DataFrame, img_dir: str, batch_size: int, training: bool):
    paths = [os.path.join(img_dir, fn) for fn in df["image"].values]
    y = labels_series_to_multihot(df["labels"], tag2idx, num_classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))

    def _load(path, target):
        img_bytes = tf.io.read_file(path)
        img = tf.image.decode_jpeg(img_bytes, channels=3)
        img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
        img = tf.cast(img, tf.float32)
        img = resnet.preprocess_input(img)
        return img, target

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.cache()  # caches decoded+resized+preprocessed tensors for epoch 2
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    ds = ds.repeat()  # keep identical number of batches via steps_per_epoch
    return ds


train_ds = make_dataset(train_df, TRAIN_IMG_DIR, BATCH_SIZE, training=True)
val_ds = make_dataset(val_df, TRAIN_IMG_DIR, BATCH_SIZE, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / BATCH_SIZE))
validation_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS),
)
base.trainable = False  # feature extractor (consistent with using a pre-trained model)

inputs = keras.Input(shape=(IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model_f = keras.Model(inputs, outputs)

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model_f.summary()

history = model_f.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=2,
)

X_test = model_f.predict(test_ds, verbose=1)
print("X_test:", X_test.shape, X_test.dtype)



## === cell 6
print("Pred sample:", X_test[0])




## === cell 7
def class2tags(classes, tagnames):
    tags = []
    for n in range(classes.shape[0]):
        tmp = []
        for i in range(classes.shape[1]):
            if classes[n, i]:
                tmp.append(tagnames[i])
        if len(tmp) == 0:
            tmp = ["healthy"]
        tags.append(" ".join(tmp))
    return tags




## === cell 8
threshold = 0.3
test_predclass = X_test > threshold
test_predtags = class2tags(test_predclass, tagnames)
print("Example tags:", test_predtags[:5])



## === cell 9
del test_predclass
gc.collect()



## === cell 10
sub = pd.read_csv(SAMPLE_SUB)
sub_images = sub["image"].tolist()

pred_map = {os.path.basename(p): tag for p, tag in zip(imglist_test, test_predtags)}

sub["labels"] = [pred_map.get(img, "healthy") for img in sub_images]

assert list(sub.columns) == ["image", "labels"]
assert len(sub) == len(pd.read_csv(SAMPLE_SUB))

sub.head()



## === cell 11
out_path = "./submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(sub))
print(sub.head(3).to_string(index=False))
