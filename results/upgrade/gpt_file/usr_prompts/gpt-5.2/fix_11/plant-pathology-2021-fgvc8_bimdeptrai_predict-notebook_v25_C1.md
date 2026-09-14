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

import tensorflow as tf
from tensorflow import keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF version:", tf.__version__)



## === cell 1
train = pd.read_csv(
    "../input/plant-pathology-2021-fgvc8/train.csv",
    engine="c",
    usecols=["image", "labels"],
)
train.head()



## === cell 2
submissions = pd.read_csv(
    "../input/plant-pathology-2021-fgvc8/sample_submission.csv",
    engine="c",
    usecols=["image", "labels"],
)
submissions.head()



## === cell 3
h_target = 384
w_target = 384
batch_size = 32



## === cell 4
effnet_preprocess = tf.keras.applications.efficientnet.preprocess_input


def _read_decode_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)  # uint8
    img = tf.image.resize(
        img, (h_target, w_target), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = effnet_preprocess(img)
    return img


def make_test_ds(df, directory, batch_size):
    paths = tf.constant([os.path.join(directory, f) for f in df["image"].values])

    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    ds = ds.map(_read_decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.cache()
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(
    submissions,
    directory="../input/plant-pathology-2021-fgvc8/test_images",
    batch_size=batch_size,
)



## === cell 5
label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
labels_arr = mlb.transform(label_split).astype(np.int8, copy=False)
labels_df = pd.DataFrame(labels_arr, columns=class_names)

print("Num classes:", len(class_names))
print("Classes:", class_names)



## === cell 6
candidate_model_paths = [
    "../input/resnet101-512-to-384/resnet101.h5",
    "/kaggle/input/resnet101-512-to-384/resnet101.h5",
]
model_path = next(
    (p for p in candidate_model_paths if os.path.exists(p)), candidate_model_paths[0]
)

model = None
loaded_external = False
if os.path.exists(model_path):
    model = keras.models.load_model(model_path, compile=False)
    loaded_external = True
    print("Loaded model from:", model_path)
else:
    print("WARNING: Model file not found:", model_path)
    print(
        "Falling back to ImageNet-pretrained EfficientNetB0 with a new classification head."
    )
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(h_target, w_target, 3),
        pooling="avg",
    )
    x = tf.keras.layers.Dropout(0.2)(base.output)
    out = tf.keras.layers.Dense(len(class_names), activation="sigmoid")(x)
    model = tf.keras.Model(inputs=base.input, outputs=out)




## === cell 7
def multilabel_f1_micro(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    tp = tf.reduce_sum(y_true * y_pred)
    fp = tf.reduce_sum((1.0 - y_true) * y_pred)
    fn = tf.reduce_sum(y_true * (1.0 - y_pred))
    f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-7)
    return f1


best_thresh = None


def _augment_train(img, seed):
    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    k = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([1, 0], tf.int32),
        minval=0,
        maxval=4,
        dtype=tf.int32,
    )
    img = tf.image.rot90(img, k=k)

    scale = tf.random.stateless_uniform(
        [],
        seed=seed + tf.constant([2, 0], tf.int32),
        minval=0.9,
        maxval=1.1,
        dtype=tf.float32,
    )
    new_h = tf.cast(tf.round(scale * h_target), tf.int32)
    new_w = tf.cast(tf.round(scale * w_target), tf.int32)
    zoomed = tf.image.resize(img, (new_h, new_w), method=tf.image.ResizeMethod.BILINEAR)
    zoomed = tf.image.resize_with_crop_or_pad(zoomed, h_target, w_target)
    return zoomed


def make_train_val_ds(train_df, val_df, train_y, val_y, directory, batch_size):
    train_paths = tf.constant(
        [os.path.join(directory, f) for f in train_df["image"].values]
    )
    val_paths = tf.constant(
        [os.path.join(directory, f) for f in val_df["image"].values]
    )

    train_y = tf.constant(train_y, dtype=tf.float32)
    val_y = tf.constant(val_y, dtype=tf.float32)

    train_ds = tf.data.Dataset.from_tensor_slices((train_paths, train_y))
    val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y))

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    train_ds = train_ds.with_options(opts)
    val_ds = val_ds.with_options(opts)

    shuffle_buf = int(min(len(train_df), 4096))
    train_ds = train_ds.shuffle(
        buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
    )

    def _train_map(path, y):
        img = _read_decode_resize(path)
        h = tf.strings.to_hash_bucket_fast(path, 2**31 - 1)
        seed = tf.stack(
            [tf.cast(h % (2**31 - 1), tf.int32), tf.cast(SEED, tf.int32)], axis=0
        )
        img = _augment_train(img, seed)
        return img, y

    def _val_map(path, y):
        img = _read_decode_resize(path)
        return img, y

    train_ds = train_ds.map(_train_map, num_parallel_calls=AUTOTUNE)
    train_ds = train_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    val_ds = val_ds.map(_val_map, num_parallel_calls=AUTOTUNE)
    val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

    return train_ds, val_ds


if not loaded_external:
    idx = np.arange(len(train))
    rng = np.random.RandomState(SEED)
    rng.shuffle(idx)
    val_size = int(0.10 * len(idx))
    val_idx = idx[:val_size]
    tr_idx = idx[val_size:]

    train_df = train.iloc[tr_idx].reset_index(drop=True).copy()
    val_df = train.iloc[val_idx].reset_index(drop=True).copy()

    train_y = labels_df.iloc[tr_idx].to_numpy(dtype=np.float32, copy=False)
    val_y = labels_df.iloc[val_idx].to_numpy(dtype=np.float32, copy=False)

    train_ds, val_ds = make_train_val_ds(
        train_df,
        val_df,
        train_y,
        val_y,
        directory="../input/plant-pathology-2021-fgvc8/train_images",
        batch_size=batch_size,
    )

    steps_per_epoch = int(np.ceil(len(train_df) / batch_size))
    validation_steps = int(np.ceil(len(val_df) / batch_size))

    for layer in model.layers:
        layer.trainable = True
    if hasattr(model.layers[0], "layers") or isinstance(
        model.layers[0], tf.keras.Model
    ):
        for lyr in model.layers:
            if isinstance(lyr, tf.keras.layers.Dense):
                lyr.trainable = True
            else:
                lyr.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )

    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=3,
        verbose=1,
    )

    for lyr in model.layers:
        lyr.trainable = True
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[multilabel_f1_micro],
    )
    model.fit(
        train_ds,
        validation_data=val_ds,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        epochs=1,
        verbose=1,
    )

    val_preds = model.predict(val_ds, verbose=1)
    val_true = val_y.astype(np.int8, copy=False)

    grid = np.arange(0.05, 0.60, 0.05, dtype=np.float32)
    yt = val_true.reshape(-1).astype(np.int8, copy=False)
    vp = val_preds.reshape(-1).astype(np.float32, copy=False)

    best_global_t = 0.25
    best_global_f1 = -1.0

    yt1 = yt == 1
    yt0 = ~yt1

    for t in grid:
        yhat1 = vp >= t
        tp = np.sum(yt1 & yhat1)
        fp = np.sum(yt0 & yhat1)
        fn = np.sum(yt1 & (~yhat1))
        denom = 2 * tp + fp + fn
        f1 = (2.0 * tp) / denom if denom > 0 else 0.0
        if f1 > best_global_f1:
            best_global_f1 = f1
            best_global_t = float(t)

    best_thresh = np.full(len(class_names), best_global_t, dtype=np.float32)
    print(
        f"Global threshold calibrated on validation: t={best_global_t:.2f}, microF1={best_global_f1:.5f}"
    )
else:
    best_thresh = np.full(len(class_names), 0.25, dtype=np.float32)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])



## === cell 9
assert preds.shape[0] == len(submissions), (
    f"Prediction count must match submission rows. preds={preds.shape[0]} "
    f"submission={len(submissions)}"
)



## === cell 10
default_thresh = 0.25

P = preds.astype(np.float32, copy=False)

thr = (
    np.full((len(class_names),), default_thresh, dtype=np.float32)
    if best_thresh is None
    else best_thresh.astype(np.float32, copy=False)
)

mask = P >= thr[None, :]
argmax_idx = np.argmax(P, axis=1)

cn = class_names
final_labels = []
append = final_labels.append
for i in range(P.shape[0]):
    chosen_idx = np.flatnonzero(mask[i])
    if chosen_idx.size == 0:
        chosen_idx = np.array([argmax_idx[i]], dtype=np.int64)
    append(" ".join(cn[j] for j in chosen_idx))

sub_out = submissions.copy()
sub_out["labels"] = final_labels

sub_out[["image", "labels"]].to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.head())



## === cell 11
sub_out.head()
