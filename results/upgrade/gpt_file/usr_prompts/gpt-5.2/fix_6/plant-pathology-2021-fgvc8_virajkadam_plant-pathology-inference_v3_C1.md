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
import gc
import random
import numpy as np
import pandas as pd
from PIL import Image

import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array



## === cell 1
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
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() or 1))
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 2
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing: {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing: {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_DIR), f"Missing dir: {TRAIN_DIR}"
assert os.path.isdir(TEST_DIR), f"Missing dir: {TEST_DIR}"

img_size = (256, 256)



## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB)
train_df = pd.read_csv(TRAIN_CSV)

assert set(sample_sub.columns) == {"image", "labels"}
assert set(train_df.columns) == {"image", "labels"}

sample_sub.head()



## === cell 4
label_classes = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
label2idx = {c: i for i, c in enumerate(label_classes)}


def encode_labels(label_str: str):
    y = np.zeros(len(label_classes), dtype=np.float32)
    if isinstance(label_str, str) and label_str.strip():
        for tok in label_str.split():
            if tok in label2idx:
                y[label2idx[tok]] = 1.0
    return y


Y = np.stack(train_df["labels"].apply(encode_labels).values)




## === cell 5
def load_image_array(image_path):
    img = load_img(image_path)
    img = img.resize(img_size)
    arr = img_to_array(img).astype(np.float32) / 255.0
    return arr


paths = (TRAIN_DIR + "/" + train_df["image"].astype(str)).values

n = len(paths)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

split = int(0.9 * n)
tr_idx, va_idx = idx[:split], idx[split:]

tr_paths, va_paths = paths[tr_idx], paths[va_idx]
tr_y, va_y = Y[tr_idx], Y[va_idx]


def make_ds(img_paths, y, batch_size=32, training=False, cache=False):
    ds = tf.data.Dataset.from_tensor_slices((img_paths, y))
    if training:
        ds = ds.shuffle(
            min(len(img_paths), 2048), seed=SEED, reshuffle_each_iteration=True
        )

    def _load(p, target):
        img_bytes = tf.io.read_file(p)
        img = tf.io.decode_jpeg(img_bytes, channels=3)  # [H,W,3], uint8
        img = tf.image.central_crop(
            img, central_fraction=1.0
        )  # no-op but keeps semantics explicit

        shape = tf.shape(img)
        h = shape[0]
        w = shape[1]
        side = tf.minimum(h, w)

        offset_h = (h - side) // 2
        offset_w = (w - side) // 2
        img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, side, side)

        img = tf.image.resize(
            img, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img, tf.cast(target, tf.float32)

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if cache:
        cache_dir = "/kaggle/working/tfdata_cache"
        os.makedirs(cache_dir, exist_ok=True)
        tag = "train" if training else "val"
        ds = ds.cache(os.path.join(cache_dir, f"{tag}_256x256.cache"))

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


train_ds = make_ds(tr_paths, tr_y, batch_size=32, training=True, cache=False)
val_ds = make_ds(va_paths, va_y, batch_size=32, training=False, cache=False)




## === cell 6
def build_model():
    inp = layers.Input(shape=(img_size[0], img_size[1], 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(len(label_classes), activation="sigmoid")(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model




## === cell 7
EPOCHS = 4
models_ens = []
for i in range(3):
    tf.keras.backend.clear_session()
    m = build_model()
    m.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=2)
    models_ens.append(m)
    gc.collect()




## === cell 8
def get_label(prediction_prob, thresh=0.35):
    """
    get label for a class that satisfies given threshold
    and return space-delimited labels string as required.
    """
    prediction_prob = np.array(prediction_prob).reshape(-1)
    labels = [
        label_classes[x] for x, prob in enumerate(prediction_prob) if prob >= thresh
    ]
    if len(labels) == 0:
        labels = ["healthy"]
    return " ".join(labels)


def load_images(test_path, image):
    """load image from given path (kept from original, used for test-time inference)"""
    img = load_img(os.path.join(test_path, image))
    img = img.resize(img_size)
    img = img_to_array(img).astype(np.float32)
    img = np.expand_dims(img, axis=0)
    img = img / 255.0
    return img




## === cell 9
def make_test_ds(test_dir, image_names, batch_size=64):
    paths = np.char.add(test_dir + "/", np.asarray(image_names, dtype=str))

    ds = tf.data.Dataset.from_tensor_slices(paths)

    def _load(p):
        img_bytes = tf.io.read_file(p)
        img = tf.io.decode_jpeg(img_bytes, channels=3)

        shape = tf.shape(img)
        h = shape[0]
        w = shape[1]
        side = tf.minimum(h, w)
        offset_h = (h - side) // 2
        offset_w = (w - side) // 2
        img = tf.image.crop_to_bounding_box(img, offset_h, offset_w, side, side)

        img = tf.image.resize(
            img, img_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
        )
        img = tf.cast(img, tf.float32) / 255.0
        return img

    opts = tf.data.Options()
    opts.deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds


@tf.function(reduce_retracing=True)
def _ensemble_predict_batch(batch):
    probs = tf.zeros((tf.shape(batch)[0], len(label_classes)), dtype=tf.float32)
    for m in models_ens:
        probs += m(batch, training=False)
    probs /= tf.cast(len(models_ens), tf.float32)
    return probs


def predict(test_path, threshold):
    """
    Predict on test set, using given threshold.
    Iterate in sample_submission order to guarantee correct alignment.
    """
    test_images = sample_sub["image"].tolist()
    test_ds = make_test_ds(test_path, test_images, batch_size=64)

    n_test = len(test_images)
    all_probs = np.empty((n_test, len(label_classes)), dtype=np.float32)

    offset = 0
    for batch in test_ds:
        probs = _ensemble_predict_batch(batch).numpy()
        bs = probs.shape[0]
        all_probs[offset : offset + bs] = probs
        offset += bs

    labels = [get_label(row, thresh=threshold) for row in all_probs]
    return test_images, labels




## === cell 10
image_ids, labels = predict(TEST_DIR, threshold=0.35)

submission_file = pd.DataFrame({"image": image_ids, "labels": labels})

assert submission_file.shape[0] == sample_sub.shape[0]
assert list(submission_file.columns) == ["image", "labels"]

submission_file.to_csv("submission.csv", index=False)
submission_file.head()



## === cell 11
print("Saved:", os.path.abspath("submission.csv"))
print(submission_file["labels"].value_counts().head(10))
