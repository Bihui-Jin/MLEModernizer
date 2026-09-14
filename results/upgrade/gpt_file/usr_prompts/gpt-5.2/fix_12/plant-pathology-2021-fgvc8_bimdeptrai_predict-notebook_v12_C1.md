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
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    _cpu_cnt = os.cpu_count() or 4
    tf.config.threading.set_intra_op_parallelism_threads(min(8, _cpu_cnt))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
h_target = 512
w_target = 512
batch_size = 32

epochs = 3

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("num_classes:", num_classes)
print("classes:", class_names)



## === cell 3
idx = np.arange(len(train))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
val_df = train.iloc[val_idx].reset_index(drop=True).copy()

y_train = y[tr_idx]
y_val = y[val_idx]

train_df["__targets__"] = list(y_train)
val_df["__targets__"] = list(y_val)

print("train/val:", len(train_df), len(val_df))



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _read_file(path):
    return tf.io.read_file(path)


@tf.function(reduce_retracing=True)
def _decode_resize_preprocess(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(img, h_target, w_target, method="bilinear")
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.mobilenet_v2.preprocess_input(img)
    return img


@tf.function(reduce_retracing=True)
def _augment(img, t):
    img = tf.image.random_flip_left_right(img, seed=SEED)
    return img, t


@tf.function(reduce_retracing=True)
def _load_and_preprocess_train(path, t):
    img = _decode_resize_preprocess(_read_file(path))
    img, t = _augment(img, t)
    return img, t


@tf.function(reduce_retracing=True)
def _load_and_preprocess_eval(path, t):
    img = _decode_resize_preprocess(_read_file(path))
    return img, t


def make_ds(df, training):
    img_names = df["image"].to_numpy(dtype=str, copy=False)
    full_paths = np.char.add(TRAIN_IMG_DIR + os.sep, img_names)
    paths = tf.constant(full_paths)

    targets_np = np.stack(df["__targets__"].to_numpy()).astype(np.float32, copy=False)
    targets = tf.constant(targets_np)

    ds = tf.data.Dataset.from_tensor_slices((paths, targets))

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    if training:
        ds = ds.map(
            _load_and_preprocess_train, num_parallel_calls=AUTOTUNE, deterministic=True
        )
    else:
        ds = ds.map(
            _load_and_preprocess_eval, num_parallel_calls=AUTOTUNE, deterministic=True
        )

    if not training:
        ds = ds.cache()

    if training:
        ds = ds.repeat()

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    ds = ds.apply(tf.data.experimental.ignore_errors())

    return ds


train_ds = make_ds(train_df, training=True)
val_ds = make_ds(val_df, training=False)

steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
validation_steps = None



## === cell 5
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
base.trainable = False  # keep training stable/fast

inputs = keras.Input(shape=(h_target, w_target, 3))
x = inputs
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 6
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 7
def make_test_ds(sub_df):
    img_names = sub_df["image"].to_numpy(dtype=str, copy=False)
    full_paths = np.char.add(TEST_IMG_DIR + os.sep, img_names)
    paths = tf.constant(full_paths)

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    @tf.function(reduce_retracing=True)
    def _load_test(path):
        return _decode_resize_preprocess(_read_file(path))

    ds = ds.map(_load_test, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    return ds


test_ds = make_test_ds(submissions)

preds = model.predict(
    test_ds,
    verbose=1,
)
print("preds shape:", preds.shape)



## === cell 8
thresh = 0.4

preds_np = np.asarray(preds)
argmax_idx = preds_np.argmax(axis=1)

chosen_mask = preds_np >= thresh
no_choice = chosen_mask.sum(axis=1) == 0
if np.any(no_choice):
    chosen_mask[no_choice, :] = False
    chosen_mask[no_choice, argmax_idx[no_choice]] = True

if "healthy" in class_names:
    healthy_idx = class_names.index("healthy")
    force_healthy = argmax_idx == healthy_idx
    if np.any(force_healthy):
        chosen_mask[force_healthy, :] = False
        chosen_mask[force_healthy, healthy_idx] = True

class_names_arr = np.array(class_names, dtype=object)
idx_lists = [np.flatnonzero(row) for row in chosen_mask]
pred_labels = [" ".join(class_names_arr[inds].tolist()) for inds in idx_lists]

submissions_out = submissions.copy()
submissions_out["labels"] = pred_labels

submissions_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions_out.shape)
submissions_out.head()



## === cell 9
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(submissions)
print("Submission OK. Example rows:")
print(chk.head(10).to_string(index=False))
