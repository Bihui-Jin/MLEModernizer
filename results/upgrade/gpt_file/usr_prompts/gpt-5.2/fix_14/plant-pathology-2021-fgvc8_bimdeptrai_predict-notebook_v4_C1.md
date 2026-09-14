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
import random
import numpy as np
import pandas as pd

SEED = 42

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow import failed in this environment. "
        "This notebook requires TensorFlow to run."
    ) from e

tf.random.set_seed(SEED)

try:
    import keras as keras  # Keras 3 if available
except Exception:
    keras = tf.keras

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

print("TF version:", tf.__version__)
print("Using keras module:", getattr(keras, "__name__", str(type(keras))))



## === cell 1
train = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 2
submissions = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
submissions.head()



## === cell 3
h_target = 384
w_target = 384
BATCH_SIZE = 16

TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

assert os.path.isdir(TRAIN_DIR), f"Missing train dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing test dir: {TEST_DIR}"



## === cell 4
from sklearn.preprocessing import MultiLabelBinarizer

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
class_names = list(mlb.classes_)
num_classes = len(class_names)

print("Num classes:", num_classes)
print("Classes:", class_names)

train_df = pd.concat([train, pd.DataFrame(y, columns=class_names)], axis=1)
train_df.head()



## === cell 5
AUTOTUNE = tf.data.AUTOTUNE

train_df_local = train_df.copy()
submissions_local = submissions.copy()

n_total = len(train_df_local)
n_val = int(np.ceil(n_total * 0.1))
val_df = train_df_local.iloc[-n_val:].reset_index(drop=True)
trn_df = train_df_local.iloc[: n_total - n_val].reset_index(drop=True)

trn_paths = (TRAIN_DIR + "/" + trn_df["image"].astype(str)).to_numpy()
val_paths = (TRAIN_DIR + "/" + val_df["image"].astype(str)).to_numpy()
tst_paths = (TEST_DIR + "/" + submissions_local["image"].astype(str)).to_numpy()

trn_labels = trn_df[class_names].to_numpy(dtype=np.float32)
val_labels = val_df[class_names].to_numpy(dtype=np.float32)

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.map_vectorization.enabled = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.autotune_buffers = True
except Exception:
    pass
try:
    DATA_OPTS.threading.private_threadpool_size = 16
except Exception:
    pass


def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


augment = keras.Sequential(
    [
        keras.layers.RandomFlip(mode="horizontal_and_vertical", seed=SEED),
        keras.layers.RandomRotation(
            factor=20.0 / 360.0, fill_mode="nearest", seed=SEED
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="nearest",
            seed=SEED,
        ),
    ],
    name="augment",
)


@tf.function
def _augment_batch(x, y):
    return augment(x, training=True), y


def _make_train_ds(paths, labels):
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.with_options(DATA_OPTS)

    shuffle_buf = min(len(paths), 4096)
    ds = ds.shuffle(buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True)

    cache_path = "/kaggle/working/cache_trn_decode_resize"
    ds = ds.map(
        lambda p, y: (_decode_and_resize(p), y),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.map(_augment_batch, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_eval_ds(paths, labels=None, cache=False, cache_path=None):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    ds = ds.with_options(DATA_OPTS)

    if labels is None:
        ds = ds.map(_decode_and_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(
            lambda p, y: (_decode_and_resize(p), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache:
        ds = ds.cache(cache_path)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = _make_train_ds(trn_paths, trn_labels)
valid_generator = _make_eval_ds(
    val_paths,
    val_labels,
    cache=True,
    cache_path="/kaggle/working/cache_val_decode_resize",
)
test_generator = _make_eval_ds(tst_paths, labels=None, cache=False)

print("Train/Val/Test sizes:", len(trn_df), len(val_df), len(submissions_local))



## === cell 6
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
)
base.trainable = False  # keep training light and stable

inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)
outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 7
EPOCHS = 2

history = model.fit(
    train_generator,
    validation_data=valid_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 8
preds = model.predict(
    test_generator,
    verbose=1,
)
print("preds shape:", preds.shape)



## === cell 9
thresh = 0.5
picked_mask = preds >= thresh
has_any = picked_mask.any(axis=1)
argmax_idx = np.argmax(preds, axis=1)

rows_any = np.flatnonzero(has_any)
rows_none = np.flatnonzero(~has_any)

pred_labels = np.empty(preds.shape[0], dtype=object)

if rows_any.size:
    idx_rows, idx_cols = np.where(picked_mask[rows_any])
    splits = np.split(
        idx_cols, np.cumsum(np.bincount(idx_rows, minlength=rows_any.size))[:-1]
    )
    pred_labels[rows_any] = [
        " ".join([class_names[j] for j in cols]) for cols in splits
    ]

if rows_none.size:
    pred_labels[rows_none] = [class_names[int(j)] for j in argmax_idx[rows_none]]

submissions_local["labels"] = pred_labels
submissions_out = submissions_local[["image", "labels"]]
submissions_out.to_csv("submission.csv", index=False)
submissions_out.head()



## === cell 10
print(submissions_out.shape)
print(submissions_out.columns.tolist())
print(submissions_out.iloc[:5])
print("Wrote: submission.csv")
