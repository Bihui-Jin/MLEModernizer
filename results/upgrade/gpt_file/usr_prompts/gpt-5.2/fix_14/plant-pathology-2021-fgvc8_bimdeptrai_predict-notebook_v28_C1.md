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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
import tensorflow.keras as keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF:", tf.__version__)
print("Keras:", keras.__version__)



## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()



## === cell 2
label_split = train["labels"].astype(str).str.split()

classes = sorted(pd.unique(label_split.explode()).tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}

tmp = label_split.explode().reset_index()  # columns: ['index', 'labels']
tmp.columns = ["row", "lab"]
tmp = tmp.dropna()

ct = pd.crosstab(tmp["row"], tmp["lab"])
ct = ct.reindex(index=np.arange(len(train)), columns=classes, fill_value=0)

labels_df = ct.astype(np.int8)

print("Classes:", classes)
labels_df.head()



## === cell 3
train_df = train.copy()
for c in classes:
    train_df[c] = labels_df[c].to_numpy()

for c in classes:
    vc = train_df[c].value_counts(normalize=True)
    print(c, dict(vc))



## === cell 4
h_target = 256
w_target = 256
batch_size = 32

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
val_df = train_df.iloc[:n_val].reset_index(drop=True)
trn_df = train_df.iloc[n_val:].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)



## === cell 5
augment_layer = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=10.0 / 360.0, seed=SEED),
        keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, seed=SEED
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=SEED
        ),
        keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augment",
)


def _decode_resize_rescale(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)  # uint8
    img.set_shape([None, None, 3])
    img = tf.image.resize(
        img,
        [h_target, w_target],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(df, directory, training, with_labels=True, cache_id=""):
    """
    SPEED FIX (correctness-preserving):
    - Cache decoded+resized tensors BEFORE shuffle/augmentation to avoid repeating expensive
      decode/resize every epoch; augmentation remains stochastic because it happens AFTER cache.
    - Use a private threadpool to reduce contention/overhead and keep deterministic behavior.
    - Keep shuffle/labels/batching semantics identical.
    """
    img_names = tf.convert_to_tensor(df["image"].astype(str).to_numpy())
    base = tf.constant(directory + os.sep)
    paths = tf.strings.join([base, img_names])

    options = tf.data.Options()
    options.deterministic = True  # keep stable/reproducible order given fixed seed
    options.threading.private_threadpool_size = 16

    if with_labels:
        labels = tf.convert_to_tensor(df[classes].to_numpy(dtype=np.float32))
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.with_options(options)

    if with_labels:

        def _decode_map(p, y_):
            return _decode_resize_rescale(p), y_

        ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)

        ds = ds.cache()

        if training:
            buf = int(min(len(df), 4096))
            ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

            def _aug_map(x, y_):
                x = augment_layer(x, training=True)
                return x, y_

            ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:

        def _decode_map_x(p):
            return _decode_resize_rescale(p)

        ds = ds.map(_decode_map_x, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache()

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    trn_df, TRAIN_DIR, training=True, with_labels=True, cache_id="train_decoded_256"
)
val_ds = make_dataset(
    val_df, TRAIN_DIR, training=False, with_labels=True, cache_id="val_decode"
)
test_ds = make_dataset(
    submissions, TEST_DIR, training=False, with_labels=False, cache_id="test_decode"
)

print("Datasets ready:", train_ds, val_ds, test_ds)



## === cell 6
inputs = keras.Input(shape=(h_target, w_target, 3))

x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)
model.summary()



## === cell 7
epochs = 3  # unchanged
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)



## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
assert preds.shape[1] == len(classes), "Prediction dimension mismatch with classes"



## === cell 9
thresh = {
    "complex": 0.1321,
    "frog_eye_leaf_spot": 0.2455,
    "healthy": 0.2595,
    "powdery_mildew": 0.0867,
    "rust": 0.1282,
    "scab": 0.3155,
}

thr_vec = np.array([thresh.get(c, 0.5) for c in classes], dtype=np.float32)
healthy_idx = classes.index("healthy") if "healthy" in classes else None



## === cell 10
preds_np = np.asarray(preds, dtype=np.float32)
n, k = preds_np.shape

argmax_idx = preds_np.argmax(axis=1)
is_healthy_max = np.zeros(n, dtype=bool)
if healthy_idx is not None:
    is_healthy_max = argmax_idx == healthy_idx

mask = preds_np > thr_vec[None, :]

nohit = (~is_healthy_max) & (mask.sum(axis=1) == 0)
mask2 = mask.copy()
mask2[nohit, :] = False
mask2[nohit, argmax_idx[nohit]] = True

pred_labels = np.empty(n, dtype=object)
pred_labels[is_healthy_max] = "healthy"

healthy_str = "healthy"
needs_join = ~is_healthy_max
needs_join_idx = np.flatnonzero(needs_join)

for i in needs_join_idx.tolist():
    chosen_idx = np.flatnonzero(mask2[i])
    chosen = [classes[j] for j in chosen_idx.tolist()]

    if healthy_str in chosen and len(chosen) > 1:
        chosen = [c for c in chosen if c != healthy_str]
        if len(chosen) == 0:
            chosen = [healthy_str]

    pred_labels[i] = " ".join(chosen)

submissions["labels"] = pred_labels.tolist()
submissions.head()



## === cell 11
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.tail())
