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

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print(
        "Determinism toggle unavailable in this environment; continuing. Error:",
        repr(e),
    )

try:
    tf.config.optimizer.set_jit(False)  # keep behavior stable; not a speed focus here
except Exception:
    pass

try:
    tf.data.experimental.enable_debug_mode = False  # keep runtime lean (as intended)
except Exception:
    pass

try:
    tf.config.optimizer.set_experimental_options({"tf_io_file_read_cache": True})
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass



## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(path + "train.csv")
sub = pd.read_csv(path + "sample_submission.csv")

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, sub.shape)
train.head()



## === cell 2
CLASSES = ["scab", "frog_eye_leaf_spot", "rust", "complex", "powdery_mildew", "healthy"]
NUM_CLASSES = len(CLASSES)

_CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}

train["filepath"] = train_images_dir + "/" + train["image"].astype(str)

labels_split = train["labels"].astype(str).str.split(" ")
exploded = labels_split.explode()
ct = pd.crosstab(exploded.index, exploded).reindex(columns=CLASSES, fill_value=0)
targets = (ct.to_numpy() > 0).astype(np.float32)
train["target"] = list(targets)

print("Prepared train filepaths/targets.")



## === cell 3
AUTO = tf.data.AUTOTUNE
IMG_SIZE = (512, 512)
BATCH_SIZE = 16  # keep unchanged


def decode_image(filename, label=None, image_size=IMG_SIZE):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.AREA, antialias=False
    )
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label




## === cell 4
idx = np.arange(len(train))
np.random.shuffle(idx)

val_frac = 0.1
val_size = int(len(train) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_df = train.iloc[trn_idx].reset_index(drop=True)
val_df = train.iloc[val_idx].reset_index(drop=True)

x_trn = trn_df["filepath"].values
y_trn = np.stack(trn_df["target"].values)

x_val = val_df["filepath"].values
y_val = np.stack(val_df["target"].values)

print("Train:", x_trn.shape, y_trn.shape, "Val:", x_val.shape, y_val.shape)

cache_dir = "/kaggle/working/tf_cache_pp2021"
os.makedirs(cache_dir, exist_ok=True)


def _map_train(f, y):
    return decode_image(f, y)


def _map_infer(f):
    return decode_image(f, None)


options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True
options.experimental_optimization.parallel_batch = True
options.threading.private_threadpool_size = 0  # let TF tune
options.threading.max_intra_op_parallelism = 0

train_base = tf.data.Dataset.from_tensor_slices((x_trn, y_trn)).with_options(options)
val_base = tf.data.Dataset.from_tensor_slices((x_val, y_val)).with_options(options)

train_ds = (
    train_base.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
val_ds = (
    val_base.map(_map_train, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)



## === cell 5
from tensorflow import keras

base = keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)

base.trainable = False  # keep training light and stable

inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))
x = inputs
x = keras.applications.resnet50.preprocess_input(x * 255.0)
x = base(x, training=False)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

model.summary()



## === cell 6
feat_inputs = keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3), name="feat_image")
feat_x = keras.applications.resnet50.preprocess_input(feat_inputs * 255.0)
feat_outputs = base(feat_x, training=False)  # 2048-d pooled features
feat_extractor = keras.Model(feat_inputs, feat_outputs, name="feat_extractor")

feat_cache_dir = os.path.join(cache_dir, "feat_cache")
os.makedirs(feat_cache_dir, exist_ok=True)
trn_feat_cache_path = os.path.join(feat_cache_dir, "trn_features.cache.npy")
val_feat_cache_path = os.path.join(feat_cache_dir, "val_features.cache.npy")


def _maybe_load_features(npy_path):
    if os.path.exists(npy_path):
        arr = np.load(npy_path, allow_pickle=False, mmap_mode=None)
        return arr
    return None


trn_features = _maybe_load_features(trn_feat_cache_path)
val_features = _maybe_load_features(val_feat_cache_path)

if trn_features is None or val_features is None:
    train_img_only = train_ds.map(
        lambda img, y: img, num_parallel_calls=AUTO, deterministic=True
    )
    val_img_only = val_ds.map(
        lambda img, y: img, num_parallel_calls=AUTO, deterministic=True
    )

    trn_steps = int(np.ceil(len(x_trn) / BATCH_SIZE))
    val_steps = int(np.ceil(len(x_val) / BATCH_SIZE))

    trn_features = feat_extractor.predict(train_img_only, steps=trn_steps, verbose=1)
    val_features = feat_extractor.predict(val_img_only, steps=val_steps, verbose=1)

    np.save(trn_feat_cache_path, trn_features)
    np.save(val_feat_cache_path, val_features)

head_inp = keras.Input(shape=(2048,), name="features")  # ResNet50 avg pool -> 2048
head_out = model.layers[-1](
    head_inp
)  # reuse the same Dense layer instance (shared weights)
head_model = keras.Model(head_inp, head_out, name="head_model")

head_model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=16,
)

train_feat_ds = (
    tf.data.Dataset.from_tensor_slices((trn_features, y_trn))
    .with_options(options)
    .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)
val_feat_ds = (
    tf.data.Dataset.from_tensor_slices((val_features, y_val))
    .with_options(options)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

EPOCHS = 3
history = head_model.fit(
    train_feat_ds,
    validation_data=val_feat_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 7
sub["filepath"] = test_images_dir + "/" + sub["image"].astype(str)
print("Prepared test filepaths.")

options_test = tf.data.Options()
options_test.experimental_deterministic = True
options_test.experimental_optimization.apply_default_optimizations = True
options_test.experimental_optimization.map_parallelization = True
options_test.experimental_optimization.parallel_batch = True
options_test.threading.private_threadpool_size = 0
options_test.threading.max_intra_op_parallelism = 0

test_ds = (
    tf.data.Dataset.from_tensor_slices(sub["filepath"].values)
    .with_options(options_test)
    .map(_map_infer, num_parallel_calls=AUTO, deterministic=True)
    .batch(BATCH_SIZE, drop_remainder=False)
    .prefetch(AUTO)
)

test_feat_cache_path = os.path.join(feat_cache_dir, "test_features.cache.npy")

test_features = _maybe_load_features(test_feat_cache_path)
if test_features is None:
    test_steps = int(np.ceil(len(sub) / BATCH_SIZE))
    test_features = feat_extractor.predict(test_ds, steps=test_steps, verbose=1)
    np.save(test_feat_cache_path, test_features)

probs = head_model.predict(test_features, verbose=1)
print("probs shape:", probs.shape)



## === cell 8
thresholds = {c: 0.5 for c in CLASSES}  # keep unchanged

thr = np.array([thresholds[c] for c in CLASSES], dtype=probs.dtype)
healthy_idx = CLASSES.index("healthy")
nonhealthy_mask = np.ones(NUM_CLASSES, dtype=bool)
nonhealthy_mask[healthy_idx] = False

sel = (probs > thr) & nonhealthy_mask  # (N, C) boolean; healthy is never selected here

idxs = [np.flatnonzero(r).tolist() for r in sel]
pred_string = [
    "healthy" if len(js) == 0 else " ".join(CLASSES[j] for j in js) for js in idxs
]

submission = sub[["image"]].copy()
submission["labels"] = pred_string

submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
