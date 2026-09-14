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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

try:
    import tensorflow_addons as tfa  # noqa: F401
except Exception as e:
    tfa = None
    print("tensorflow_addons unavailable (safe to ignore):", repr(e))

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    CPU_COUNT = os.cpu_count() or 2
    tf.config.threading.set_intra_op_parallelism_threads(CPU_COUNT)
    tf.config.threading.set_inter_op_parallelism_threads(max(2, CPU_COUNT // 2))
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



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
h_target = 384
w_target = 384
batch_size = 32

CPU_COUNT = os.cpu_count() or 2
GEN_WORKERS = min(8, max(2, CPU_COUNT // 2))
GEN_MAX_QUEUE_SIZE = 4 * GEN_WORKERS



## === cell 3
label_split = train.labels.apply(lambda x: x.split())
mlb = MultiLabelBinarizer()
mlb.fit(label_split)

classes = list(mlb.classes_)
print("Num classes:", len(classes))
print("Classes:", classes)



## === cell 4
AUTOTUNE = tf.data.AUTOTUNE

label_to_index = {c: i for i, c in enumerate(classes)}
num_classes = len(classes)


def _labels_to_multi_hot_py(label_str):
    vec = np.zeros((num_classes,), dtype=np.float32)
    for tok in str(label_str).split():
        j = label_to_index.get(tok)
        if j is not None:
            vec[j] = 1.0
    return vec


def _decode_resize_rescale(image_path):
    img_bytes = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _make_train_valid_datasets(df, validation_split=0.1):
    n = len(df)
    n_valid = int(round(n * validation_split))
    n_train = n - n_valid

    df_train = df.iloc[:n_train].reset_index(drop=True)
    df_valid = df.iloc[n_train:].reset_index(drop=True)

    train_paths = tf.constant(
        [os.path.join(TRAIN_IMG_DIR, f) for f in df_train["image"].values],
        dtype=tf.string,
    )
    train_labels = tf.constant(df_train["labels"].values, dtype=tf.string)

    valid_paths = tf.constant(
        [os.path.join(TRAIN_IMG_DIR, f) for f in df_valid["image"].values],
        dtype=tf.string,
    )
    valid_labels = tf.constant(df_valid["labels"].values, dtype=tf.string)

    def map_train(p, y):
        img = _decode_resize_rescale(p)
        yv = tf.py_function(_labels_to_multi_hot_py, [y], Tout=tf.float32)
        yv.set_shape([num_classes])
        return img, yv

    def map_valid(p, y):
        img = _decode_resize_rescale(p)
        yv = tf.py_function(_labels_to_multi_hot_py, [y], Tout=tf.float32)
        yv.set_shape([num_classes])
        return img, yv

    ds_train = tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    ds_train = ds_train.shuffle(
        buffer_size=min(4096, n_train), seed=SEED, reshuffle_each_iteration=True
    )
    ds_train = ds_train.map(map_train, num_parallel_calls=AUTOTUNE)
    ds_train = ds_train.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    ds_valid = tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    ds_valid = ds_valid.map(map_valid, num_parallel_calls=AUTOTUNE)
    ds_valid = ds_valid.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    steps_per_epoch = int(np.ceil(n_train / batch_size))
    validation_steps = int(np.ceil(n_valid / batch_size))
    return ds_train, ds_valid, steps_per_epoch, validation_steps


def _make_test_dataset(df):
    test_paths = tf.constant(
        [os.path.join(TEST_IMG_DIR, f) for f in df["image"].values], dtype=tf.string
    )
    ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
    ds_test = ds_test.map(_decode_resize_rescale, num_parallel_calls=AUTOTUNE)
    ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    steps = int(np.ceil(len(df) / batch_size))
    return ds_test, steps


train_ds, valid_ds, steps_per_epoch, validation_steps = _make_train_valid_datasets(
    train, validation_split=0.1
)
test_ds, test_steps = _make_test_dataset(submissions)

print(
    "steps_per_epoch:",
    steps_per_epoch,
    "validation_steps:",
    validation_steps,
    "test_steps:",
    test_steps,
)




## === cell 5
def find_first_h5_model(search_root="../input"):
    candidate_paths = [
        os.path.join("../input/plant-pathology-2021-fgvc8", "model.h5"),
        os.path.join("../input/plant-pathology-2021-fgvc8", "model.hdf5"),
        os.path.join("../input", "model.h5"),
        os.path.join("../input", "model.hdf5"),
    ]
    for p in candidate_paths:
        if os.path.exists(p) and os.path.isfile(p):
            return p
    return None


model_path = find_first_h5_model("../input")
print("Discovered model path:", model_path)

model = None
if model_path is not None:
    try:
        model = keras.models.load_model(model_path, compile=False)
        print("Loaded model from:", model_path)
    except Exception as e:
        print(
            "Failed to load discovered model; will train fallback model. Error:",
            repr(e),
        )
        model = None

if model is None:
    inputs = keras.layers.Input(shape=(h_target, w_target, 3))
    x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = keras.layers.MaxPooling2D()(x)
    x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dropout(0.2)(x)
    outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )

    EPOCHS = 2

    model.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        verbose=1,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
    )



## === cell 6
preds = model.predict(
    test_ds,
    verbose=1,
    steps=test_steps,
)
preds = np.asarray(preds)

if preds.ndim == 1:
    preds = preds.reshape(-1, len(classes))
elif (
    preds.ndim == 2
    and preds.shape[1] != len(classes)
    and preds.shape[0] == len(classes)
):
    preds = preds.T

print("preds shape:", preds.shape)
print("num test images:", len(submissions))
assert preds.shape[0] == len(
    submissions
), "Prediction count mismatch with submission rows"



## === cell 7
thresh = 0.2
healthy_idx = classes.index("healthy") if "healthy" in classes else None

preds2 = preds
n = preds2.shape[0]

argmax_all = np.argmax(preds2, axis=1)
max_all = np.max(preds2, axis=1)

picked_mask = preds2 >= thresh  # (n, C)
any_picked = picked_mask.any(axis=1)

rows_none = np.where(~any_picked)[0]
if rows_none.size:
    picked_mask[rows_none, :] = False
    picked_mask[rows_none, argmax_all[rows_none]] = True

if healthy_idx is not None:
    rows_healthy_argmax = np.where(argmax_all == healthy_idx)[0]
    if rows_healthy_argmax.size:
        cond = preds2[rows_healthy_argmax, healthy_idx] >= (
            max_all[rows_healthy_argmax] - 1e-12
        )
        rows_force = rows_healthy_argmax[cond]
        if rows_force.size:
            picked_mask[rows_force, :] = False
            picked_mask[rows_force, healthy_idx] = True

    has_healthy = picked_mask[:, healthy_idx]
    multi = picked_mask.sum(axis=1) > 1
    rows_rm = np.where(has_healthy & multi)[0]
    if rows_rm.size:
        picked_mask[rows_rm, healthy_idx] = False
        emptied = np.where(picked_mask[rows_rm].sum(axis=1) == 0)[0]
        if emptied.size:
            picked_mask[rows_rm[emptied], healthy_idx] = True

out_labels = []
for i in range(n):
    idxs = np.flatnonzero(picked_mask[i]).tolist()
    out_labels.append(" ".join([classes[k] for k in idxs]))

submissions["labels"] = out_labels
submissions["labels"] = submissions["labels"].fillna("").astype(str)
submissions.loc[submissions["labels"].str.strip() == "", "labels"] = "healthy"

submissions = submissions[["image", "labels"]]
submissions.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submissions.shape)
submissions.head()



## === cell 8
assert os.path.exists("submission.csv"), "submission.csv was not created"
chk = pd.read_csv("submission.csv")
print(chk.columns.tolist())
print(chk.head(3))
print("Unique label strings (sample):", chk["labels"].head(10).tolist())
