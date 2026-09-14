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
import gc
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

import tensorflow.keras.applications.resnet50 as resnet

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

K.set_image_data_format("channels_last")

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("tf:", tf.__version__, "keras:", keras.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
IMG_WIDTH = 224
IMG_HEIGHT = 224
NR_CHANNELS = 3




## === cell 2
DATA_DIR = "/kaggle/input/plant-pathology-2021-fgvc8"
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TRAIN_CSV_PATH = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_PATH = os.path.join(DATA_DIR, "sample_submission.csv")

imglist_test = sorted(glob.glob(os.path.join(TEST_IMG_DIR, "*.jpg")))
print("n_test_images:", len(imglist_test))
print("first_test_image:", imglist_test[0] if imglist_test else "NONE")




## === cell 3
AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _load_test_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    return img


BATCH_SIZE = 16  # keep same effective batch size used later

data_opts = tf.data.Options()
data_opts.experimental_deterministic = True

if len(imglist_test) > 0:
    test_path_ds = tf.data.Dataset.from_tensor_slices(imglist_test).with_options(
        data_opts
    )
    test_ds = (
        test_path_ds.map(
            _load_test_image, num_parallel_calls=AUTOTUNE, deterministic=True
        )
        .apply(tf.data.experimental.ignore_errors())
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
else:
    test_ds = tf.data.Dataset.from_tensor_slices(
        tf.zeros([0, IMG_HEIGHT, IMG_WIDTH, NR_CHANNELS], dtype=tf.float32)
    ).batch(BATCH_SIZE, drop_remainder=False)

print("test_ds ready; batch_size:", BATCH_SIZE)




## === cell 4
gc.collect()




## === cell 5
training_csv = pd.read_csv(TRAIN_CSV_PATH)

tagnames = np.unique(
    np.concatenate(training_csv["labels"].astype(str).str.split().to_numpy())
)
num_classes = len(tagnames)
print("num_classes:", num_classes)
print("classes:", tagnames)

tag2idx = {t: i for i, t in enumerate(tagnames)}

from sklearn.preprocessing import MultiLabelBinarizer

labels_split = training_csv["labels"].astype(str).str.split()
mlb = MultiLabelBinarizer(classes=tagnames)
Y_all = mlb.fit_transform(labels_split).astype(np.float32)

from sklearn.model_selection import train_test_split

idx = np.arange(len(training_csv))
train_idx, val_idx = train_test_split(
    idx, test_size=0.1, random_state=SEED, shuffle=True
)

train_images = training_csv.loc[train_idx, "image"].values
val_images = training_csv.loc[val_idx, "image"].values
Y_train = Y_all[train_idx]
Y_val = Y_all[val_idx]

CACHE_DIR = "/kaggle/working/tf_cache_pp2021"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE_PATH = os.path.join(
    CACHE_DIR, f"train_{IMG_HEIGHT}x{IMG_WIDTH}_bs{BATCH_SIZE}"
)
VAL_CACHE_PATH = os.path.join(CACHE_DIR, f"val_{IMG_HEIGHT}x{IMG_WIDTH}_bs{BATCH_SIZE}")


@tf.function
def load_train_example_fast(img_name, y):
    img_path = tf.strings.join([TRAIN_IMG_DIR, "/", img_name])
    img_bytes = tf.io.read_file(img_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [IMG_HEIGHT, IMG_WIDTH], method="bilinear")
    img = tf.cast(img, tf.float32)
    img = resnet.preprocess_input(img)
    y = tf.cast(y, tf.float32)
    y.set_shape([num_classes])
    return img, y


train_ds = tf.data.Dataset.from_tensor_slices((train_images, Y_train)).with_options(
    data_opts
)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = (
    train_ds.map(
        load_train_example_fast, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    .apply(tf.data.experimental.ignore_errors())
    .cache(TRAIN_CACHE_PATH)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_images, Y_val)).with_options(data_opts)
val_ds = (
    val_ds.map(load_train_example_fast, num_parallel_calls=AUTOTUNE, deterministic=True)
    .apply(tf.data.experimental.ignore_errors())
    .cache(VAL_CACHE_PATH)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

base = resnet.ResNet50(
    include_top=False, weights="imagenet", input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)
)
x = keras.layers.GlobalAveragePooling2D()(base.output)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model_f = keras.Model(inputs=base.input, outputs=out)

base.trainable = False

model_f.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

history = model_f.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)

X_test = model_f.predict(test_ds, verbose=1)
print("X_test:", X_test.shape, X_test.dtype)




## === cell 6
print("X_test sample:", X_test[0] if len(X_test) else "NONE")




## === cell 7
pass




## === cell 8
def class2tags(classes, tagnames):
    idxs = [np.flatnonzero(row) for row in classes]
    return [
        " ".join(tagnames[i] for i in inds) if len(inds) else "healthy" for inds in idxs
    ]




## === cell 9
test_predclass = X_test > 0.4
test_predtags = class2tags(test_predclass, tagnames)
print("predtags sample:", test_predtags[:5])




## === cell 10
del test_predclass
gc.collect()




## === cell 11
sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
sample_sub_images = sample_sub["image"].tolist()

test_basenames = [os.path.basename(p) for p in imglist_test]
pred_by_image = dict(zip(test_basenames, test_predtags))

final_labels = [pred_by_image.get(img, "healthy") for img in sample_sub_images]

submission = pd.DataFrame({"image": sample_sub_images, "labels": final_labels})
submission.head()




## === cell 12
print("submission shape:", submission.shape)
print(submission.head())




## === cell 13
assert list(submission.columns) == ["image", "labels"]
assert submission["image"].notnull().all()
assert submission["labels"].notnull().all()




## === cell 14
OUT_PATH = "./submission.csv"
submission.to_csv(OUT_PATH, index=False)
print("Wrote:", OUT_PATH)
print(submission.tail())




## === cell 15
assert os.path.exists(OUT_PATH) and OUT_PATH.endswith(".csv")
print("Done.")
