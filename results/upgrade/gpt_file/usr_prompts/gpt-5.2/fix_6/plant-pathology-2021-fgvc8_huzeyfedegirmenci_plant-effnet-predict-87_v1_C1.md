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
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
tf.random.set_seed(SEED)
np.random.seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

BASE_INPUT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_INPUT, "train.csv")
TRAIN_DIR = os.path.join(BASE_INPUT, "train_images")
TEST_DIR = os.path.join(BASE_INPUT, "test_images")
SAMPLE_SUB = os.path.join(BASE_INPUT, "sample_submission.csv")

print("Train CSV exists:", os.path.exists(TRAIN_CSV))
print("Train dir exists:", os.path.exists(TRAIN_DIR))
print("Test dir exists:", os.path.exists(TEST_DIR))
print("Sample submission exists:", os.path.exists(SAMPLE_SUB))

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 1
cats = [
    "scab frog_eye_leaf_spot",
    "frog_eye_leaf_spot",
    "complex",
    "healthy",
    "powdery_mildew",
    "powdery_mildew complex",
    "rust",
    "rust complex",
    "rust frog_eye_leaf_spot",
    "scab",
    "scab frog_eye_leaf_spot complex",
    "frog_eye_leaf_spot complex",
]
cat_to_idx = {c: i for i, c in enumerate(cats)}
idx_to_cat = {i: c for i, c in enumerate(cats)}

train_df = pd.read_csv(TRAIN_CSV)
train_df["path"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_DIR, x))

train_df = train_df[train_df["labels"].isin(cat_to_idx)].reset_index(drop=True)
train_df["y"] = train_df["labels"].map(cat_to_idx).astype(int)

print("Train rows after filtering:", len(train_df))
print("Unique labels in train (mapped):", train_df["labels"].nunique())




## === cell 2
IMG_SIZE = 300
BATCH_SIZE = 16
EPOCHS = 2  # keep small to finish <600s; original relied on a pretrained external .h5

val_frac = 0.1
val_size = int(len(train_df) * val_frac)
train_part = (
    train_df.iloc[:-val_size].reset_index(drop=True)
    if val_size > 0
    else train_df.copy()
)
val_part = (
    train_df.iloc[-val_size:].reset_index(drop=True)
    if val_size > 0
    else train_df.iloc[:0].copy()
)


def _load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


def _make_ds(df, training, cache_key):
    paths = df["path"].values
    labels = df["y"].values
    ds = tf.data.Dataset.from_tensor_slices((paths, labels))

    def _map_fn(p, y):
        x = _load_image(p)
        y = tf.one_hot(y, depth=len(cats))
        return x, y

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE)

    ds = ds.cache(os.path.join(CACHE_DIR, f"{cache_key}.cache"))

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_ds = _make_ds(train_part, training=True, cache_key="train_ds")
val_ds = (
    _make_ds(val_part, training=False, cache_key="val_ds")
    if len(val_part) > 0
    else None
)




## === cell 3
from tensorflow.keras import layers, models


def build_model(img_size=IMG_SIZE, n_classes=len(cats)):
    inputs = layers.Input(shape=(img_size, img_size, 3))
    base = tf.keras.applications.EfficientNetB2(
        include_top=False, weights="imagenet", input_tensor=inputs, pooling="avg"
    )
    base.trainable = False

    x = base.output
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    return model


model = build_model()

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="categorical_crossentropy",
    metrics=[tf.keras.metrics.CategoricalAccuracy(name="acc")],
)

print(model.count_params())




## === cell 4
if val_ds is not None and len(val_part) > 0:
    history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)
else:
    history = model.fit(train_ds, epochs=EPOCHS, verbose=1)




## === cell 5
test_images = sorted(
    [os.path.basename(p) for p in tf.io.gfile.glob(os.path.join(TEST_DIR, "*.jpg"))]
)


def _load_image_tf(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


@tf.function
def _tta_stack_vectorized(x, steps):
    x = tf.ensure_shape(x, (IMG_SIZE, IMG_SIZE, 3))
    steps = tf.cast(steps, tf.int32)

    x_stack = tf.repeat(x[None, ...], repeats=steps, axis=0)  # (steps,H,W,3)

    step_ids = tf.range(steps, dtype=tf.int32)  # (steps,)
    seeds = tf.stack(
        [tf.fill([steps], tf.cast(SEED, tf.int32)), step_ids], axis=1
    )  # (steps,2)
    r = tf.random.stateless_uniform(shape=(steps,), seed=seeds, dtype=tf.float32)

    flip_mask = r >= 0.5  # original: return x if r<0.5 else flip
    flipped = tf.image.flip_left_right(x_stack)
    flip_mask = flip_mask[:, None, None, None]
    return tf.where(flip_mask, flipped, x_stack)


def predict_tta_batch(image_names, steps=10, pred_batch=256):
    n = len(image_names)
    if n == 0:
        return np.empty((0,), dtype=np.int64)

    paths = tf.strings.join([tf.constant(TEST_DIR + "/"), tf.constant(image_names)])

    ds = tf.data.Dataset.from_tensor_slices(paths)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load_image_tf, num_parallel_calls=AUTOTUNE)

    ds = ds.cache(os.path.join(CACHE_DIR, "test_imgs.cache"))

    ds = ds.map(
        lambda x: _tta_stack_vectorized(x, tf.constant(steps, tf.int32)),
        num_parallel_calls=AUTOTUNE,
    )
    ds = ds.unbatch()
    ds = ds.batch(pred_batch).prefetch(AUTOTUNE)

    yhats = model.predict(ds, verbose=0)
    n_classes = yhats.shape[-1]

    yhats = yhats.reshape(n, steps, n_classes).sum(axis=1)
    pred_idx = np.argmax(yhats, axis=1).astype(np.int64)
    return pred_idx


pred_images = test_images
TTA_STEPS = 10

cls_indices = predict_tta_batch(pred_images, steps=TTA_STEPS, pred_batch=256)
pred_labels = [idx_to_cat[int(i)] for i in cls_indices]

print(
    "Predicted",
    len(pred_images),
    "test images. Example:",
    pred_images[0],
    pred_labels[0],
)




## === cell 6
sub_df = pd.DataFrame({"image": pred_images, "labels": pred_labels})

if os.path.exists(SAMPLE_SUB):
    sample_df = pd.read_csv(SAMPLE_SUB)
    sub_df = sample_df[["image"]].merge(sub_df, on="image", how="left")
    sub_df["labels"] = sub_df["labels"].fillna("healthy")

sub_path = "./submission.csv"
sub_df.to_csv(sub_path, index=False)
print("Wrote:", sub_path)
print(sub_df.head())
