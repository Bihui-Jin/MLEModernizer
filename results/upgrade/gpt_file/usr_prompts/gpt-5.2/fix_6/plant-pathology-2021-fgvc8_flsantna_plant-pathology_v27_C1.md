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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass


def _find_comp_root():
    candidates = [
        "/kaggle/input/plant-pathology-2021-fgvc8",
        "../input/plant-pathology-2021-fgvc8",
        "./kaggle/input/plant-pathology-2021-fgvc8",
        "/kaggle/data/plant-pathology-2021-fgvc8",
        "../data/plant-pathology-2021-fgvc8",
    ]
    for p in candidates:
        if os.path.isfile(os.path.join(p, "train.csv")):
            return p
    for base in ["/kaggle/input", "../input", "/kaggle/data", "../data"]:
        p = os.path.join(base, "plant-pathology-2021-fgvc8")
        if os.path.isfile(os.path.join(p, "train.csv")):
            return p
    raise FileNotFoundError(
        "Could not locate competition dataset root containing train.csv"
    )


COMP_ROOT = _find_comp_root()
TRAIN_CSV = os.path.join(COMP_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(COMP_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(COMP_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(COMP_ROOT, "test_images")

output_dir = "./"
os.makedirs(output_dir, exist_ok=True)

print("COMP_ROOT:", COMP_ROOT)
print("TRAIN_CSV exists:", os.path.isfile(TRAIN_CSV))
print("TRAIN_IMG_DIR exists:", os.path.isdir(TRAIN_IMG_DIR))
print("TEST_IMG_DIR exists:", os.path.isdir(TEST_IMG_DIR))
print("SAMPLE_SUB exists:", os.path.isfile(SAMPLE_SUB))

image_dims = (300, 300, 3)

data_set = pd.read_csv(TRAIN_CSV)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num train rows:", len(data_set))
print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])



## === cell 1
AUTOTUNE = tf.data.AUTOTUNE
BATCH_SIZE = 32

thr = 0.7

train_df = data_set[["image", "labels"]].copy()
train_df["filepath"] = train_df["image"].map(lambda x: os.path.join(TRAIN_IMG_DIR, x))

train_df = train_df[train_df["filepath"].apply(os.path.isfile)].reset_index(drop=True)

print("Train images found:", len(train_df), "/", len(data_set))

num_classes = len(dataset_labels)
y = (
    train_df["labels"]
    .astype(str)
    .str.get_dummies(sep=" ")
    .reindex(columns=dataset_labels, fill_value=0)
    .to_numpy(dtype=np.float32, copy=False)
)

idx = np.arange(len(train_df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_files = train_df.loc[trn_idx, "filepath"].tolist()
val_files = train_df.loc[val_idx, "filepath"].tolist()
y_trn = y[trn_idx]
y_val = y[val_idx]

print("Train split:", len(trn_files), "Val split:", len(val_files))


@tf.function
def load_and_preprocess(path, target):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, (image_dims[0], image_dims[1]), method="bilinear")
    img = tf.ensure_shape(img, image_dims)
    img = img * 255.0
    return img, target


def make_ds(files, targets, training: bool):
    ds = tf.data.Dataset.from_tensor_slices((files, targets))
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    opts.threading.private_threadpool_size = 0
    opts.threading.max_intra_op_parallelism = 0
    ds = ds.with_options(opts)

    ds = ds.map(load_and_preprocess, num_parallel_calls=AUTOTUNE)
    ds = ds.apply(tf.data.experimental.ignore_errors())

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(trn_files, y_trn, training=True)
val_ds = make_ds(val_files, y_val, training=False)

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=image_dims,
    pooling="avg",
)

inp = tf.keras.Input(shape=image_dims, name="image")
x = base(inp, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

base.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[],
    steps_per_execution=32,
)

EPOCHS_HEAD = 3
hist1 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_HEAD, verbose=1)

base.trainable = True
for layer in base.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss=tf.keras.losses.BinaryCrossentropy(),
    metrics=[],
    steps_per_execution=32,
)

EPOCHS_FT = 2
hist2 = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS_FT, verbose=1)

test_images = sorted(
    [f for f in os.listdir(TEST_IMG_DIR) if f.lower().endswith(".jpg")]
)
print("Num test images:", len(test_images))


@tf.function
def load_img_only(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (image_dims[0], image_dims[1]), method="bilinear")
    img = tf.ensure_shape(img, image_dims)
    img = img * 255.0
    return img


test_paths = [os.path.join(TEST_IMG_DIR, n) for n in test_images]
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)

opts = tf.data.Options()
opts.experimental_deterministic = True
opts.threading.private_threadpool_size = 0
opts.threading.max_intra_op_parallelism = 0
test_ds = test_ds.with_options(opts)

test_ds = test_ds.map(load_img_only, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.apply(tf.data.experimental.ignore_errors())

test_ds = test_ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
probs = np.asarray(probs, dtype=np.float32)

class_names = np.asarray(dataset_labels, dtype=object)
mask = probs > thr

pred_labels = []
for i in range(mask.shape[0]):
    idxs = np.flatnonzero(mask[i])
    if idxs.size == 0:
        pred_labels.append("healthy")
    else:
        pred_labels.append(" ".join(class_names[idxs].tolist()))

submission = pd.DataFrame({"image": test_images, "labels": pred_labels})

if os.path.isfile(SAMPLE_SUB):
    sample = pd.read_csv(SAMPLE_SUB)
    if "image" in sample.columns and len(sample) == len(submission):
        submission = sample[["image"]].merge(submission, on="image", how="left")
        submission["labels"] = submission["labels"].fillna("healthy")

sub_path = os.path.join(output_dir, "submission.csv")
submission.to_csv(sub_path, index=False)
print("Wrote:", sub_path, "rows:", len(submission))
print(submission.head())
