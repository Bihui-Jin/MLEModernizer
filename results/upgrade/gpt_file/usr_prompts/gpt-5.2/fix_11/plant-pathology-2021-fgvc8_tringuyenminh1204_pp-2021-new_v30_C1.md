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
import os, random
import numpy as np
import pandas as pd
import tensorflow as tf

print("tf:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass




## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"
train = pd.read_csv(path + "train.csv")
test = pd.read_csv(path + "sample_submission.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, test.shape)
train.head()




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE




## === cell 3
pass




## === cell 4
import pathlib




## === cell 5
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
test_img_dir = "../input/plant-pathology-2021-fgvc8/test_images"

train_paths = (train_img_dir + "/" + train["image"].astype(str)).tolist()
test_paths = (test_img_dir + "/" + test["image"].astype(str)).tolist()

print("n_train_images (from CSV):", len(train_paths))
print("n_test_images (from sample_submission):", len(test_paths))




## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew", "healthy"]
class_to_idx = {c: i for i, c in enumerate(CLASSES)}
CLASSES




## === cell 7
def labels_to_multihot(label_str: str):
    y = np.zeros(len(CLASSES), dtype=np.float32)
    for token in str(label_str).split():
        if token in class_to_idx:
            y[class_to_idx[token]] = 1.0
    return y


new_train = train[["image", "labels"]].copy()
new_train["target"] = new_train["labels"].apply(labels_to_multihot)
new_train.head()




## === cell 8
new_train




## === cell 9
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(image, image_size, method="bilinear", antialias=False)
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    else:
        label = tf.cast(label, tf.float32)
        return image, label




## === cell 10
test_paths[:5]




## === cell 11
BATCH_SIZE = 64




## === cell 12
pass




## === cell 13
import tensorflow as tf
from tensorflow import keras




## === cell 14
IMG_SIZE = (512, 512)

train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images"
new_train["filepath"] = (train_img_dir + "/" + new_train["image"].astype(str)).values

X_paths = new_train["filepath"].values
Y = np.stack(new_train["target"].values)

idx = np.arange(len(X_paths))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

X_tr, Y_tr = X_paths[tr_idx], Y[tr_idx]
X_va, Y_va = X_paths[va_idx], Y[va_idx]

CACHE_DIR = "/kaggle/working/tf_cache_pp2021"
os.makedirs(CACHE_DIR, exist_ok=True)


def make_ds(paths, labels=None, training=False, cache_in_memory=False, cache_name=None):
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_slack = (
            True  # improves input/compute overlap without changing results
        )
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.apply_default_optimizations = True
    except Exception:
        pass

    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(opts)
        ds = ds.map(
            lambda x: decode_image(x, None, IMG_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(opts)
        ds = ds.map(
            lambda x, y: decode_image(x, y, IMG_SIZE),
            num_parallel_calls=AUTO,
            deterministic=True,
        )

    if cache_name is not None:
        ds = ds.cache(os.path.join(CACHE_DIR, cache_name))
    elif cache_in_memory:
        ds = ds.cache()

    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.batch(BATCH_SIZE, drop_remainder=True)
    else:
        ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_ds(
    X_tr, Y_tr, training=True, cache_in_memory=False, cache_name="train.cache"
)
val_ds = make_ds(
    X_va, Y_va, training=False, cache_in_memory=False, cache_name="val.cache"
)




## === cell 15
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
base.trainable = False  # keep runtime reasonable

inp = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = tf.keras.applications.efficientnet.preprocess_input(inp * 255.0)
x = base(x, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.BinaryCrossentropy(),
    steps_per_execution=32,
)

model.summary()




## === cell 16
EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)




## === cell 17
assert model is not None




## === cell 18
probs = None




## === cell 19
probs_shape = None
probs_shape




## === cell 20
None




## === cell 21
temp_probs = probs




## === cell 22
None




## === cell 23
name = {
    0: "scab",
    1: "frog_eye_leaf_spot",
    2: "complex",
    3: "rust",
    4: "powdery_mildew",
    5: "healthy",
}

threshold = {0: 0.01, 1: 0.01, 2: 0.01, 3: 0.01, 4: 0.01}

subm = sub.copy()

subm["filepath"] = "../input/plant-pathology-2021-fgvc8/test_images/" + subm[
    "image"
].astype(str)
ordered_test_paths = subm["filepath"].values

ordered_test_dataset = make_ds(
    ordered_test_paths,
    labels=None,
    training=False,
    cache_in_memory=False,
    cache_name="test.cache",
)

ordered_probs = model.predict(ordered_test_dataset, verbose=1)

thr = np.array([threshold[i] for i in range(5)], dtype=np.float32)
ordered_probs_5 = ordered_probs[:, :5]
hits_mask = ordered_probs_5 > thr[None, :]

pred_string = []
for row_mask in hits_mask:
    idxs = np.flatnonzero(row_mask)
    if idxs.size:
        s_tokens = [name[int(i)] for i in idxs]
        if idxs.size >= 2 and "complex" not in s_tokens:
            s_tokens.append("complex")
        pred_string.append(" ".join(s_tokens))
    else:
        pred_string.append(name[5])

subm["labels"] = pred_string
subm = subm[["image", "labels"]]
subm.to_csv("submission.csv", index=False)
print(subm.head())
print("Wrote submission.csv with", len(subm), "rows")
