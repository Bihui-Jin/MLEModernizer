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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    for gpu in tf.config.list_physical_devices("GPU"):
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

print("TF:", tf.__version__)
print("GPUs:", tf.config.list_physical_devices("GPU"))




## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB_CSV)

print(train.shape, submissions.shape)
train.head()




## === cell 2
h_target = 256
w_target = 256
batch_size = 32
AUTOTUNE = tf.data.AUTOTUNE

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)
num_classes = len(classes)

print("num_classes:", num_classes)
print("classes:", classes)




## === cell 3
@tf.function(reduce_retracing=True)
def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, h_target, w_target, method="bilinear", antialias=False
    )
    img = tf.image.convert_image_dtype(img, tf.float32)
    return img


@tf.function(reduce_retracing=True)
def _map_labeled(p, y_):
    return decode_and_resize(p), y_


def make_ds(
    paths,
    labels=None,
    training=False,
    repeat=False,
    cache=False,
    cache_path=None,
    drop_remainder=False,
    ignore_errors=False,  # default off; enabling adds overhead and can mask real data issues
):
    det = not training

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=det)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        ds = ds.map(_map_labeled, num_parallel_calls=AUTOTUNE, deterministic=det)

    if ignore_errors:
        ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    if repeat:
        ds = ds.repeat()

    opt = tf.data.Options()
    opt.experimental_deterministic = det
    ds = ds.with_options(opt)

    ds = ds.batch(batch_size, drop_remainder=drop_remainder)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_paths_np = (TRAIN_IMG_DIR + "/" + train["image"].values).astype("U")
train_paths = tf.constant(train_paths_np, dtype=tf.string)
train_labels = tf.constant(y.astype(np.float32))

n = len(train)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_size = int(0.1 * n)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_paths = tf.gather(train_paths, trn_idx)
trn_labels = tf.gather(train_labels, trn_idx)
val_paths = tf.gather(train_paths, val_idx)
val_labels = tf.gather(train_labels, val_idx)

train_ds = make_ds(
    trn_paths,
    trn_labels,
    training=True,
    repeat=True,
    cache=False,
    cache_path=None,
    drop_remainder=True,
    ignore_errors=False,
)

val_ds = make_ds(
    val_paths,
    val_labels,
    training=False,
    repeat=False,
    cache=True,
    cache_path=None,
    drop_remainder=False,
    ignore_errors=False,
)

test_paths_np = (TEST_IMG_DIR + "/" + submissions["image"].values).astype("U")
test_paths = tf.constant(test_paths_np, dtype=tf.string)

test_ds = make_ds(
    test_paths,
    labels=None,
    training=False,
    repeat=False,
    cache=False,
    cache_path=None,
    drop_remainder=False,
    ignore_errors=False,
)

train_steps = int(np.ceil(len(trn_idx) / batch_size))
val_steps = int(np.ceil(len(val_idx) / batch_size))
test_steps = int(np.ceil(len(submissions) / batch_size))

print(
    "train_steps:",
    train_steps,
    "val_steps:",
    val_steps,
    "test_steps:",
    test_steps,
    "batch_size:",
    batch_size,
)




## === cell 4
base = keras.applications.Xception(
    include_top=False, weights="imagenet", input_shape=(h_target, w_target, 3)
)
base.trainable = False  # stabilize & keep runtime under control

inputs = keras.Input(shape=(h_target, w_target, 3))
x = keras.applications.xception.preprocess_input(inputs * 255.0)
x = base(x, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    jit_compile=True,
)

model.summary()




## === cell 5
EPOCHS = 3
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 6
preds = model.predict(test_ds, steps=test_steps, verbose=1)
print("preds shape:", preds.shape)




## === cell 7
thresh = 0.25

preds_np = np.asarray(preds)
above = preds_np >= thresh
argmax_idx = np.argmax(preds_np, axis=1)

try:
    healthy_idx = classes.index("healthy")
except ValueError:
    healthy_idx = None

classes_list = classes
pred_labels = []
append = pred_labels.append

for i in range(preds_np.shape[0]):
    chosen_idx = np.flatnonzero(above[i])
    if chosen_idx.size == 0:
        chosen_idx = np.array([argmax_idx[i]], dtype=np.int64)

    if healthy_idx is not None and chosen_idx.size > 1:
        if (chosen_idx == healthy_idx).any():
            chosen_idx = chosen_idx[chosen_idx != healthy_idx]
            if chosen_idx.size == 0:
                chosen_idx = np.array([healthy_idx], dtype=np.int64)

    append(" ".join(classes_list[j] for j in chosen_idx.tolist()))

submissions = submissions.copy()
submissions["labels"] = pred_labels

submissions.head()




## === cell 8
out_path = "submission.csv"
submissions.to_csv(out_path, index=False)
print("Wrote:", out_path)
print(submissions.shape)
print(submissions.head(10))
