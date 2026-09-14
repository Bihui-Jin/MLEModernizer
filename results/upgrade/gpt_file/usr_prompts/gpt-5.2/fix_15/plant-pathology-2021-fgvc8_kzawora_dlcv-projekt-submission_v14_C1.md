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
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf

np.random.seed(42)
tf.random.set_seed(42)

BASE_CANDIDATES = [
    Path("/kaggle/input/plant-pathology-2021-fgvc8"),
    Path("/kaggle/data/plant-pathology-2021-fgvc8"),
    Path("../input/plant-pathology-2021-fgvc8"),
]
BASE_DIR = next((p for p in BASE_CANDIDATES if p.exists()), None)
if BASE_DIR is None:
    raise FileNotFoundError(f"Could not find dataset dir in any of: {BASE_CANDIDATES}")

TRAIN_CSV = BASE_DIR / "train.csv"
SAMPLE_SUB = BASE_DIR / "sample_submission.csv"
TRAIN_IMAGES_DIR = BASE_DIR / "train_images"
TEST_IMAGES_DIR = BASE_DIR / "test_images"

print("BASE_DIR:", BASE_DIR)
print("TRAIN_CSV exists:", TRAIN_CSV.exists())
print("SAMPLE_SUB exists:", SAMPLE_SUB.exists())
print("TRAIN_IMAGES_DIR exists:", TRAIN_IMAGES_DIR.exists())
print("TEST_IMAGES_DIR exists:", TEST_IMAGES_DIR.exists())

AUTOTUNE = tf.data.AUTOTUNE
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass



## === cell 1
df_train = pd.read_csv(TRAIN_CSV)
print("Train rows:", len(df_train))
print(df_train.head())

labels = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]
label_to_idx = {l: i for i, l in enumerate(labels)}


def labels_to_vec(s: str):
    s = str(s).strip()
    vec = np.zeros(len(labels), dtype=np.float32)
    if not s:
        return vec
    parts = s.split()
    for p in parts:
        if p in label_to_idx:
            vec[label_to_idx[p]] = 1.0
    return vec


y = np.stack([labels_to_vec(s) for s in df_train["labels"].values], axis=0)
df_train = df_train.copy()
df_train["filepath"] = TRAIN_IMAGES_DIR.as_posix() + "/" + df_train["image"].astype(str)

paths_np = df_train["filepath"].to_numpy(dtype=str)

for p in paths_np[:5]:
    if not tf.io.gfile.exists(p):
        raise FileNotFoundError(
            f"Training image referenced in train.csv does not exist: {p}"
        )

for i, l in enumerate(labels):
    df_train[l] = y[:, i].astype(np.float32)

df_train = df_train.sample(frac=1.0, random_state=42).reset_index(drop=True)
val_frac = 0.1
n_val = int(len(df_train) * val_frac)
df_val = df_train.iloc[:n_val].reset_index(drop=True)
df_trn = df_train.iloc[n_val:].reset_index(drop=True)

print("Train split:", len(df_trn), "Val split:", len(df_val))



## === cell 2
IMG_SIZE = (380, 380)
BATCH_SIZE = 32
EPOCHS = 3

DATA_OPTS = tf.data.Options()
DATA_OPTS.experimental_deterministic = True
try:
    DATA_OPTS.threading.private_threadpool_size = max(8, (os.cpu_count() or 8) - 1)
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.apply_default_optimizations = True
except Exception:
    pass
try:
    DATA_OPTS.experimental_optimization.map_and_batch_fusion = True
except Exception:
    pass

try:
    import tensorflow_addons as tfa  # noqa: F401
except Exception:
    tfa = None


def decode_and_resize(path_or_bytes, is_bytes=False):
    img_bytes = path_or_bytes if is_bytes else tf.io.read_file(path_or_bytes)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) / 255.0  # rescale=1/255
    return img


def augment(img, seed):
    seed = tf.cast(seed, tf.int32)

    img = tf.image.stateless_random_flip_left_right(img, seed=seed)

    angle = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([1, 0], tf.int32), minval=-15.0, maxval=15.0
    ) * (np.pi / 180.0)
    if tfa is not None:
        img = tfa.image.rotate(img, angles=angle, fill_mode="reflect")
    else:
        img = tf.image.rot90(img, k=tf.cast(tf.round(angle / (np.pi / 2.0)), tf.int32))

    tx = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([2, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE[0], tf.float32)
    ty = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([3, 0], tf.int32), minval=-0.05, maxval=0.05
    ) * tf.cast(IMG_SIZE[1], tf.float32)
    if tfa is not None:
        img = tfa.image.translate(img, translations=[ty, tx], fill_mode="reflect")

    z = tf.random.stateless_uniform(
        [], seed=seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )
    new_h = tf.cast(tf.round(tf.cast(IMG_SIZE[0], tf.float32) * z), tf.int32)
    new_w = tf.cast(tf.round(tf.cast(IMG_SIZE[1], tf.float32) * z), tf.int32)
    img2 = tf.image.resize(
        img, [new_h, new_w], method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img2 = tf.image.resize_with_crop_or_pad(img2, IMG_SIZE[0], IMG_SIZE[1])
    img = img2

    return img


def make_train_ds(df, shuffle=True):
    paths = np.asarray(df["filepath"].values, dtype=np.str_)
    ys = np.asarray(df[labels].values, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.with_options(DATA_OPTS)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(len(df), 4096), seed=42, reshuffle_each_iteration=True
        )

    ds = ds.enumerate()  # (index, (path, y))

    @tf.function
    def _load_decode_aug(i, data):
        path, y = data
        img_bytes = tf.io.read_file(path)
        img = decode_and_resize(img_bytes, is_bytes=True)
        seed = tf.stack([tf.constant(42, tf.int32), tf.cast(i, tf.int32)])
        img = augment(img, seed)
        return img, y

    ds = ds.map(_load_decode_aug, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_val_ds(df):
    paths = np.asarray(df["filepath"].values, dtype=np.str_)
    ys = np.asarray(df[labels].values, dtype=np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, ys))
    ds = ds.with_options(DATA_OPTS)

    @tf.function
    def _load_decode(path, y):
        img_bytes = tf.io.read_file(path)
        img = decode_and_resize(img_bytes, is_bytes=True)
        return img, y

    ds = ds.map(_load_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(df_trn, shuffle=True)
val_ds = make_val_ds(df_val)
print("Built train/val datasets.")



## === cell 3
inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(len(labels), activation="sigmoid")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 4
test_paths = np.asarray(
    sorted(tf.io.gfile.glob((TEST_IMAGES_DIR / "*.jpg").as_posix())), dtype=np.str_
)


def make_test_ds(paths):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(DATA_OPTS)

    @tf.function
    def _load_decode(path):
        img_bytes = tf.io.read_file(path)
        img = decode_and_resize(img_bytes, is_bytes=True)
        return img

    ds = ds.map(_load_decode, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(128, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_ds = make_test_ds(test_paths)
print("Test images:", len(test_paths))



## === cell 5
x = model.predict(test_ds, verbose=1)
print("Pred shape:", x.shape)

threshold = 0.7
z = x > threshold

label_arr = np.array(labels, dtype=object)
idxs = [np.flatnonzero(row) for row in z]
predictions_str = [" ".join(label_arr[ii]) if len(ii) else "healthy" for ii in idxs]

filenames = [Path(p).name for p in test_paths]
df_pred = pd.DataFrame({"image": filenames, "labels": predictions_str})

sample = pd.read_csv(SAMPLE_SUB)
df = sample[["image"]].merge(df_pred, on="image", how="left")
df["labels"] = df["labels"].fillna("healthy")

print(df.head())
print("Rows:", len(df), "Unique images:", df["image"].nunique())



## === cell 6
out_path = Path("submission.csv")
df.to_csv(out_path, index=False)
print("Wrote:", out_path.resolve())
print("Submission columns:", list(df.columns))
print("Any missing labels:", df["labels"].isna().sum())
print(
    "Non-string labels:", (df["labels"].apply(lambda v: not isinstance(v, str))).sum()
)
