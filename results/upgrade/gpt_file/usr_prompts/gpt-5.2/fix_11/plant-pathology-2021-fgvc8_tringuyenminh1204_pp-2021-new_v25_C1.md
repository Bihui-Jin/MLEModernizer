# Goal

I want you to fix bugs and increase the score toward a target for a Kaggle competition solution. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (big fix and/or evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


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

# 5. Target score

0.2739229653080058

# 6. Current score

0.33303

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.33492) has done: 'The timeout is dominated by repeatedly decoding/resizing large 512×512 images on the CPU during both training and prediction, plus some avoidable Python overhead in label encoding and submission string building. I keep the exact same model, epochs, loss, optimizer, thresholds, and dataset semantics, but speed up the input pipeline by (1) using `tf.image.decode_jpeg/png` instead of generic `decode_image`, (2) enabling dataset caching (memory/disk) so images are decoded/resized once per split, and (3) adding fast, deterministic TF data optimizations (parallel map, prefetch already present, plus removing unused work). I also replace slow Python loops for one-hot creation and prediction string assembly with equivalent vectorized pandas/numpy operations, preserving identical outputs. No approximation, sampling, early stopping, or architectural/training changes are introduced.'
- What this solution (achieved 0.33303) has done: 'Main bottlenecks are (1) resizing/processing 512×512 images (heavy CPU) while training and predicting, (2) suboptimal tf.data pipeline ordering (cache placed after batching/shuffle causing repeated decode/resize work), and (3) Python overhead in label one-hot encoding and path mapping. I keep the same model/epochs/loss and exact evaluation semantics, but make the input pipeline provably equivalent and much faster by caching *decoded+resized* tensors to disk before shuffle (so each image is decoded/resized once), enabling TF graph execution for mapping, and avoiding redundant work (duplicate onehot computation, repeated glob/path operations). I also switch the multi-label onehot creation to a vectorized MultiLabelBinarizer-equivalent using pandas split + numpy indexing (exactly equivalent to the regex contains logic for this dataset’s space-delimited labels). Finally, I remove test `.cache()` in-memory (can blow RAM / stall) and instead rely on prefetch+parallel map while keeping determinism.'
- What this solution (achieved 0.33303) has done: 'I fix the TensorFlow import crash by removing the forced pure-Python protobuf environment variables, which is triggering the `MessageFactory.GetPrototype` AttributeError in Kaggle’s TF/protobuf stack. I keep the model, training loop, datasets, thresholds, and label logic unchanged so the score behavior stays essentially the same (and should remain close to your current score, which is already above the target band). I also make the submission column name exactly match the required format (`labels`), and keep the output filename `submission.csv` to ensure Kaggle accepts it. No architectural/training/threshold changes are introduced—this is a correctness/runtime fix.'

# 9. Code solution

## === cell 0
import os, random, re, math

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd

try:
    import tensorflow as tf
    from tensorflow import keras

    print("TF:", tf.__version__)
    print("Keras:", keras.__version__)
except Exception as e:
    raise RuntimeError(
        "TensorFlow failed to import in this environment. Original error:\n" + str(e)
    )



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_CANDIDATES = [
    "../input/plant-pathology-2021-fgvc8/",
    "/kaggle/input/plant-pathology-2021-fgvc8/",
    "../input/",
    "/kaggle/input/",
]


def _find_competition_base():
    for base in BASE_CANDIDATES:
        if os.path.isdir(base):
            if os.path.isfile(os.path.join(base, "train.csv")) and os.path.isdir(
                os.path.join(base, "train_images")
            ):
                return base if base.endswith("/") else base + "/"
            try:
                for d in os.listdir(base):
                    cand = os.path.join(base, d)
                    if os.path.isdir(cand) and os.path.isfile(
                        os.path.join(cand, "train.csv")
                    ):
                        if os.path.isdir(
                            os.path.join(cand, "train_images")
                        ) and os.path.isdir(os.path.join(cand, "test_images")):
                            return cand + "/"
            except Exception:
                pass
    cand = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/"
    if os.path.isfile(os.path.join(cand, "train.csv")):
        return cand if cand.endswith("/") else cand + "/"
    raise RuntimeError(
        "Could not locate competition data folder under ../input or /kaggle/input"
    )


path = _find_competition_base()
print("Using data path:", path)

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))

print(train.shape, sub.shape)
print(train.head())



## === cell 2
AUTO = tf.data.experimental.AUTOTUNE



## === cell 3
img_path_example = os.path.join(path, "train_images", train.loc[0, "image"])
print("Example image path (not loaded for speed):", img_path_example)



## === cell 4
import pathlib



## === cell 5
train_dir = os.path.join(path, "train_images")
test_dir = os.path.join(path, "test_images")

train_paths = tf.io.gfile.glob(os.path.join(train_dir, "*.jpg"))
train_paths += tf.io.gfile.glob(os.path.join(train_dir, "*.jpeg"))
train_paths += tf.io.gfile.glob(os.path.join(train_dir, "*.png"))
train_paths = sorted(train_paths)

test_paths = tf.io.gfile.glob(os.path.join(test_dir, "*.jpg"))
test_paths += tf.io.gfile.glob(os.path.join(test_dir, "*.jpeg"))
test_paths += tf.io.gfile.glob(os.path.join(test_dir, "*.png"))
test_paths = sorted(test_paths)

print("Found train images:", len(train_paths))
print("Found test images:", len(test_paths))

if len(train_paths) == 0:
    raise RuntimeError(f"No train images found in {train_dir}. Check input path.")
if len(test_paths) == 0:
    raise RuntimeError(f"No test images found in {test_dir}. Check input path.")



## === cell 6
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew", "healthy"]
print("Classes:", CLASSES)




## === cell 7
def multilabel_onehot(series, classes):
    s = series.astype(str).fillna("")
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    y = np.zeros((len(s), len(classes)), dtype=np.float32)
    tokens = s.str.split()
    for i, toks in enumerate(tokens):
        for t in toks:
            j = cls_to_idx.get(t)
            if j is not None:
                y[i, j] = 1.0
    return pd.DataFrame(y, columns=classes, dtype=np.float32)


labels_onehot_features = multilabel_onehot(train["labels"], CLASSES)
new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]
new_train.head()



## === cell 8
new_train




## === cell 9
@tf.function
def decode_image(filename, label=None, image_size=(512, 512)):
    filename = tf.cast(filename, tf.string)
    bits = tf.io.read_file(filename)

    lower = tf.strings.lower(filename)
    is_png = tf.strings.regex_full_match(lower, ".*\\.png")

    def _decode_png():
        return tf.image.decode_png(bits, channels=3)

    def _decode_jpg():
        return tf.image.decode_jpeg(bits, channels=3)

    image = tf.cond(is_png, _decode_png, _decode_jpg)
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(image, image_size)

    if label is None:
        return image
    else:
        label = tf.cast(label, tf.float32)
        return image, label




## === cell 10
test_paths[:5], len(test_paths)



## === cell 11
BATCH_SIZE = 64



## === cell 12
_ds_opts = tf.data.Options()
_ds_opts.experimental_deterministic = True
try:
    _ds_opts.threading.private_threadpool_size = max(8, (os.cpu_count() or 8) - 1)
except Exception:
    pass

test_path_map = {os.path.basename(p): p for p in test_paths}
sub["filepath"] = sub["image"].map(test_path_map)
missing = int(sub["filepath"].isna().sum())
if missing:
    raise RuntimeError(
        f"Missing {missing} test image filepaths. Example missing: "
        f"{sub.loc[sub['filepath'].isna(), 'image'].head(3).tolist()}. "
        f"Check test_images directory and filenames under: {test_dir}"
    )

test_paths_tensor = tf.constant(sub["filepath"].astype(str).values, dtype=tf.string)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths_tensor)
    .with_options(_ds_opts)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 13
from tensorflow import keras as tf_keras  # compatibility alias if referenced downstream



## === cell 14
SEED = 42

train_path_map = {os.path.basename(p): p for p in train_paths}
train["filepath"] = train["image"].map(train_path_map)

train = train[train["filepath"].notna()].reset_index(drop=True)
if len(train) == 0:
    raise RuntimeError(
        f"After mapping images to filepaths, training set is empty. Check train_images directory: {train_dir}"
    )

labels_onehot_features = multilabel_onehot(train["labels"], CLASSES).astype(np.float32)

rng = np.random.default_rng(SEED)
idx = np.arange(len(train))
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

x_tr = train.loc[tr_idx, "filepath"].astype(str).values
y_tr = labels_onehot_features.loc[tr_idx].values.astype(np.float32)
x_va = train.loc[va_idx, "filepath"].astype(str).values
y_va = labels_onehot_features.loc[va_idx].values.astype(np.float32)

x_tr_t = tf.constant(x_tr, dtype=tf.string)
x_va_t = tf.constant(x_va, dtype=tf.string)

cache_dir = "/kaggle/working/tf_cache"
os.makedirs(cache_dir, exist_ok=True)
train_cache = os.path.join(cache_dir, "train_cache")
valid_cache = os.path.join(cache_dir, "valid_cache")

train_dataset = (
    tf.data.Dataset.from_tensor_slices((x_tr_t, y_tr))
    .with_options(_ds_opts)
    .map(lambda f, y: decode_image(f, y), num_parallel_calls=AUTO)
    .cache(train_cache)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((x_va_t, y_va))
    .with_options(_ds_opts)
    .map(lambda f, y: decode_image(f, y), num_parallel_calls=AUTO)
    .cache(valid_cache)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

print("Train/valid sizes:", len(x_tr), len(x_va))



## === cell 15
tf.keras.utils.set_random_seed(SEED)

inputs = keras.Input(shape=(512, 512, 3))
x = keras.layers.Conv2D(16, 3, strides=2, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, strides=2, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(len(CLASSES), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()



## === cell 16
EPOCHS = 2
history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 17
probs = model.predict(test_dataset, verbose=1)



## === cell 18
probs.shape



## === cell 19
probs[:2]



## === cell 20
temp_probs = probs



## === cell 21
temp_probs[:2]



## === cell 22
name = {i: c for i, c in enumerate(CLASSES)}
threshold = {name_idx: 0.15 for name_idx, cls in name.items() if cls != "healthy"}

arr = np.asarray(temp_probs)
healthy_idx = CLASSES.index("healthy")
nonhealthy_idx = [i for i, c in enumerate(CLASSES) if c != "healthy"]

mask = arr[:, nonhealthy_idx] > 0.15
counts = mask.sum(axis=1)

labels_list = []
for r in range(arr.shape[0]):
    picked = [CLASSES[nonhealthy_idx[j]] for j in np.flatnonzero(mask[r])]
    if len(picked) >= 2 and ("complex" not in picked):
        picked.append("complex")
    if len(picked) == 0:
        labels_list.append("healthy")
    else:
        labels_list.append(" ".join(picked))

submission = sub.copy()
if len(labels_list) != len(submission):
    raise ValueError(
        f"Prediction rows ({len(labels_list)}) != submission rows ({len(submission)})"
    )

submission["labels"] = labels_list
submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with rows:", len(submission))
