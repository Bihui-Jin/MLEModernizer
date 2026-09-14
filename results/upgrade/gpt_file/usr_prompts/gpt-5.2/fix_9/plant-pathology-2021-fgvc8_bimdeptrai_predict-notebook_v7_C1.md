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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "1")
os.environ.setdefault("TF_USE_CUDNN_AUTOTUNE", "1")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import random
import numpy as np
import pandas as pd

try:
    import tensorflow as tf
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment (often a protobuf mismatch). "
        "This notebook sets PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python to mitigate. "
        f"Original error: {repr(e)}"
    )

keras = tf.keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)


## === cell 1
TRAIN_CSV = "../input/plant-pathology-2021-fgvc8/train.csv"
SAMPLE_SUB = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TRAIN_DIR = "../input/plant-pathology-2021-fgvc8/train_images"
TEST_DIR = "../input/plant-pathology-2021-fgvc8/test_images"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()


## === cell 2
h_target = 512
w_target = 512


## === cell 3
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(label_split)
classes = list(mlb.classes_)

print("Num classes:", len(classes))
print("Classes:", classes)


## === cell 4
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

BATCH_SIZE = 8
TEST_BATCH_SIZE = 32  # throughput-only; does not change outputs
AUTOTUNE = tf.data.AUTOTUNE

train_df = train.copy()
train_df["labels_list"] = label_split

class_to_idx = {c: i for i, c in enumerate(classes)}


def _build_multihot_matrix(labels_list):
    out = np.zeros((len(labels_list), len(classes)), dtype=np.float32)
    for i, labs in enumerate(labels_list):
        for lab in labs:
            j = class_to_idx.get(lab)
            if j is not None:
                out[i, j] = 1.0
    return out


val_frac = 0.1
n = len(train_df)
n_val = int(np.floor(n * val_frac))
n_train = n - n_val

train_df_train = train_df.iloc[:n_train].reset_index(drop=True)
train_df_val = train_df.iloc[n_train:].reset_index(drop=True)

y_train = _build_multihot_matrix(train_df_train["labels_list"].tolist())
y_val = _build_multihot_matrix(train_df_val["labels_list"].tolist())

train_paths = (TRAIN_DIR + "/" + train_df_train["image"].values).astype(object)
val_paths = (TRAIN_DIR + "/" + train_df_val["image"].values).astype(object)
test_paths = (TEST_DIR + "/" + submissions["image"].values).astype(object)


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST"
    )  # uint8
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = preprocess_input(img)
    img.set_shape([h_target, w_target, 3])
    return img


def _ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True

    opt = opts.experimental_optimization
    try:
        opt.apply_default_optimizations = True
    except Exception:
        pass
    if hasattr(opt, "map_parallelization"):
        try:
            opt.map_parallelization = True
        except Exception:
            pass
    if hasattr(opt, "parallel_batch"):
        try:
            opt.parallel_batch = True
        except Exception:
            pass
    return opts


def _make_ds(
    paths, labels=None, training=False, cache_in_memory=False, batch_size=BATCH_SIZE
):
    opts = _ds_options()

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
        ds = ds.apply(
            tf.data.experimental.map_and_batch(
                _decode_resize_preprocess,
                batch_size=batch_size,
                num_parallel_calls=AUTOTUNE,
                drop_remainder=False,
            )
        )
        if cache_in_memory:
            ds = ds.cache()
        ds = ds.prefetch(AUTOTUNE)
        return ds

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)

    if training:
        shuffle_buf = min(len(paths), 2048)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

    def _map_xy(p, y_):
        return _decode_resize_preprocess(p), y_

    ds = ds.apply(
        tf.data.experimental.map_and_batch(
            _map_xy,
            batch_size=batch_size,
            num_parallel_calls=AUTOTUNE,
            drop_remainder=False,
        )
    )
    if cache_in_memory:
        ds = ds.cache()
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(
    train_paths, y_train, training=True, cache_in_memory=False, batch_size=BATCH_SIZE
)
val_ds = _make_ds(
    val_paths, y_val, training=False, cache_in_memory=False, batch_size=BATCH_SIZE
)
test_ds = _make_ds(
    test_paths,
    labels=None,
    training=False,
    cache_in_memory=False,
    batch_size=TEST_BATCH_SIZE,
)

print(
    "Datasets built:",
    "train batches:",
    tf.data.experimental.cardinality(train_ds).numpy(),
    "val batches:",
    tf.data.experimental.cardinality(val_ds).numpy(),
    "test batches:",
    tf.data.experimental.cardinality(test_ds).numpy(),
)


## === cell 5
model_path = "../input/mobilenetv2-512/mobilenetv2_512.h5"

num_classes = len(classes)


def build_model(num_classes: int):
    base = tf.keras.applications.MobileNetV2(
        input_shape=(h_target, w_target, 3), include_top=False, weights="imagenet"
    )
    base.trainable = False

    inputs = tf.keras.Input(shape=(h_target, w_target, 3))
    x = base(inputs, training=False)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    m = tf.keras.Model(inputs, outputs)

    m.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return m


if tf.io.gfile.exists(model_path):
    model = tf.keras.models.load_model(model_path, compile=False)
    try:
        model.compile(optimizer="adam", loss="binary_crossentropy")
    except Exception:
        pass
    print("Loaded model:", model_path)
else:
    print("Pretrained model not found at:", model_path)
    print(
        "Training fallback enabled to produce a valid submission within the runtime budget."
    )
    model = build_model(num_classes)
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=1,
        verbose=1,
    )


## === cell 6
preds = model.predict(test_ds, verbose=1)
print("Preds shape:", preds.shape)


## === cell 7
thresh = 0.4
healthy_idx = classes.index("healthy") if "healthy" in classes else None

p = preds
chosen = p >= thresh

none_chosen = ~chosen.any(axis=1)
if np.any(none_chosen):
    am = np.argmax(p[none_chosen], axis=1)
    chosen[none_chosen, :] = False
    chosen[none_chosen, am] = True

pred_labels = np.empty((p.shape[0],), dtype=object)

if healthy_idx is not None:
    healthy_mask = p[:, healthy_idx] >= thresh
    pred_labels[healthy_mask] = "healthy"

    other_mask = ~healthy_mask
    other_idx = np.flatnonzero(other_mask)

    cls_arr = np.asarray(classes, dtype=object)
    for i in other_idx:
        idxs = np.flatnonzero(chosen[i])
        if idxs.size:
            pred_labels[i] = " ".join(cls_arr[idxs].tolist())
        else:
            pred_labels[i] = "healthy"
else:
    cls_arr = np.asarray(classes, dtype=object)
    for i in range(p.shape[0]):
        idxs = np.flatnonzero(chosen[i])
        pred_labels[i] = " ".join(cls_arr[idxs].tolist()) if idxs.size else "healthy"

submissions = submissions.copy()
submissions["labels"] = pred_labels.tolist()

submissions.head()


## === cell 8
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.head())


## === cell 9
submissions
