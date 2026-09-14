# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Develop a model to classify paddy leaf images into one of the nine disease categories or normal leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
200001.jpg,normal
200002.jpg,blast
etc.
```

## Dataset
**train.csv** - The training set

- `image_id` - Unique image identifier corresponds to image file names (.jpg) found in the train_images directory.
- `label` - Type of paddy disease, also the target class. There are ten categories, including the normal leaf.
- `variety` - The name of the paddy variety.
- `age` - Age of the paddy in days.

**sample_submission.csv** - Sample submission file.

**train_images** - Training images stored under different sub-directories corresponding to ten target classes. Filename corresponds to the `image_id` column of `train.csv`.

**test_images** - Test set images.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        input/
            description.md (72 lines)
            sample_submission.csv (2603 lines)
            sample_submission.csv.zip (7.4 kB)
            test.zip (160 Bytes)
            test_images.zip (205.2 MB)
            train.csv (7806 lines)
            train.csv.zip (40.1 kB)
            train.zip (162 Bytes)
            train_images.zip (614.5 MB)
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
            test_images/
                102916.jpg (92.1 kB)
                100596.jpg (83.9 kB)
                ... and 2600 other files
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
            train_images/
                bacterial_leaf_blight/
                    109831.jpg (91.1 kB)
                    109785.jpg (81.8 kB)
                    ... and 356 other files
                bacterial_leaf_streak/
                    100394.jpg (99.7 kB)
                    103308.jpg (104.3 kB)
                    ... and 295 other files
                ... and 9 other folders
        working/
            paddy-disease-classification/
                description.md (72 lines)
                sample_submission.csv (2603 lines)
                ... and 7 other files
                paddy-disease-classification/
                test_images/
                    102916.jpg (92.1 kB)
                    100596.jpg (83.9 kB)
                    ... and 2600 other files
                    test_images/
                train_images/
                    bacterial_leaf_blight/
                        109831.jpg (91.1 kB)
                        109785.jpg (81.8 kB)
                        ... and 356 other files
                    bacterial_leaf_streak/
                        100394.jpg (99.7 kB)
                        103308.jpg (104.3 kB)
                        ... and 295 other files
                    ... and 9 other folders
```

-> data/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> data/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> input/paddy-disease-classification/sample_submission.csv has 2602 rows and 2 columns.
The columns are: image_id, label

-> input/paddy-disease-classification/train.csv has 7805 rows and 4 columns.
The columns are: image_id, label, variety, age

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(True)  # XLA JIT (keep as-is)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_ROOT = "/kaggle/input/paddy-disease-classification"
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB = os.path.join(DATA_ROOT, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

for col in ["image_id", "label"]:
    assert col in train_df.columns, f"train.csv missing column: {col}"
for col in ["image_id", "label"]:
    assert col in sample_df.columns, f"sample_submission.csv missing column: {col}"

classes = sorted(train_df["label"].unique().tolist())
num_classes = len(classes)
assert num_classes == 10, f"Expected 10 classes, got {num_classes}"

train_df = train_df.copy()
train_df["filepath"] = (
    TRAIN_IMG_DIR
    + os.sep
    + train_df["label"].astype(str)
    + os.sep
    + train_df["image_id"].astype(str)
)

_paths = train_df["filepath"].to_numpy()
exists_mask = np.array([os.path.exists(p) for p in _paths], dtype=bool)

if (~exists_mask).any():
    train_df = train_df.loc[exists_mask].reset_index(drop=True)

train_df.head()




## === cell 1
from sklearn.model_selection import train_test_split

trn_df, val_df = train_test_split(
    train_df[["filepath", "label"]],
    test_size=0.1,
    random_state=SEED,
    stratify=train_df["label"],
)

IMG_SIZE = 300
BATCH_SIZE = 16

AUTOTUNE = tf.data.AUTOTUNE

class_to_idx = {c: i for i, c in enumerate(classes)}

_label_table = tf.lookup.StaticHashTable(
    tf.lookup.KeyValueTensorInitializer(
        keys=tf.constant(classes),
        values=tf.constant(list(range(len(classes))), dtype=tf.int32),
    ),
    default_value=-1,
)

_preprocess = tf.keras.applications.efficientnet.preprocess_input


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32)
    img = _preprocess(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


@tf.function(reduce_retracing=True)
def _label_to_onehot(label_str):
    label_idx = _label_table.lookup(label_str)
    label = tf.one_hot(label_idx, depth=num_classes, dtype=tf.float32)
    label.set_shape([num_classes])
    return label


@tf.function(reduce_retracing=True)
def _parse_train(path, label_str):
    img = _decode_and_resize(path)
    label = _label_to_onehot(label_str)
    return img, label


@tf.function(reduce_retracing=True)
def _parse_test(path):
    img = _decode_and_resize(path)
    return img


_rand_rot = tf.keras.layers.RandomRotation(
    factor=10.0 / 180.0,  # +/-10 degrees
    fill_mode="reflect",
    interpolation="bilinear",
    seed=SEED,
)


@tf.function(reduce_retracing=True)
def _augment(img, label):
    base_seed = tf.random.uniform([2], maxval=2**31 - 1, dtype=tf.int32, seed=SEED)

    img = tf.image.stateless_random_flip_left_right(img, seed=base_seed)

    img = _rand_rot(tf.expand_dims(img, 0), training=True)
    img = tf.squeeze(img, 0)

    max_dx = 0.05 * IMG_SIZE
    max_dy = 0.05 * IMG_SIZE
    dx = tf.random.stateless_uniform(
        [],
        seed=base_seed + tf.constant([2, 0], tf.int32),
        minval=-max_dx,
        maxval=max_dx,
    )
    dy = tf.random.stateless_uniform(
        [],
        seed=base_seed + tf.constant([3, 0], tf.int32),
        minval=-max_dy,
        maxval=max_dy,
    )

    img = tf.raw_ops.ImageProjectiveTransformV3(
        images=tf.expand_dims(img, 0),
        transforms=tf.expand_dims(
            tf.stack([1.0, 0.0, -dx, 0.0, 1.0, -dy, 0.0, 0.0]), 0
        ),
        output_shape=tf.constant([IMG_SIZE, IMG_SIZE], dtype=tf.int32),
        interpolation="BILINEAR",
        fill_mode="REFLECT",
        fill_value=0.0,
    )
    img = tf.squeeze(img, 0)

    zoom = tf.random.stateless_uniform(
        [], seed=base_seed + tf.constant([4, 0], tf.int32), minval=0.9, maxval=1.1
    )
    new_size = tf.cast(tf.round(IMG_SIZE / zoom), tf.int32)
    new_size = tf.clip_by_value(new_size, int(0.7 * IMG_SIZE), int(1.3 * IMG_SIZE))
    img = tf.image.resize_with_crop_or_pad(img, new_size, new_size)
    img = tf.image.resize(
        img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.BILINEAR
    )

    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img, label


def _ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True  # keep results stable
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.autotune_buffers = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.map_and_batch_fusion = True
        opts.experimental_optimization.noop_elimination = True
        opts.experimental_optimization.filter_fusion = True
    except Exception:
        pass
    return opts


_SHUFFLE_BUFFER = 2048

_TRAIN_CACHE = "/kaggle/working/cache_train_decode"
_VAL_CACHE = "/kaggle/working/cache_val_decode"
_TEST_CACHE = "/kaggle/working/cache_test_decode"

for _p in (_TRAIN_CACHE, _VAL_CACHE, _TEST_CACHE):
    try:
        if tf.io.gfile.exists(_p):
            tf.io.gfile.rmtree(_p)
    except Exception:
        pass


def make_train_ds(df):
    paths = df["filepath"].astype(str).to_numpy()
    labels = df["label"].astype(str).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options())

    ds = ds.shuffle(
        buffer_size=min(len(df), _SHUFFLE_BUFFER),
        seed=SEED,
        reshuffle_each_iteration=True,
    )

    ds = ds.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(_TRAIN_CACHE)

    ds = ds.map(_augment, num_parallel_calls=AUTOTUNE, deterministic=True)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.repeat()
    return ds


def make_val_ds(df):
    paths = df["filepath"].astype(str).to_numpy()
    labels = df["label"].astype(str).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels)).with_options(_ds_options())

    ds = ds.map(_parse_train, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.cache(_VAL_CACHE)

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = ds.repeat()
    return ds


train_data = make_train_ds(trn_df)
val_data = make_val_ds(val_df)

steps_per_epoch = int(np.ceil(len(trn_df) / BATCH_SIZE))
val_steps = int(np.ceil(len(val_df) / BATCH_SIZE))

test_files = sample_df["image_id"].tolist()
test_df = pd.DataFrame(
    {
        "image_id": test_files,
        "filepath": [os.path.join(TEST_IMG_DIR, f) for f in test_files],
    }
)

_test_paths = test_df["filepath"].to_numpy()
test_exists = np.array([os.path.exists(p) for p in _test_paths], dtype=bool)
test_df.loc[~test_exists, "filepath"] = np.nan

test_df.head()




## === cell 2
try:
    from tensorflow.keras import mixed_precision

    mixed_precision.set_global_policy("mixed_float16")
except Exception:
    pass

base = tf.keras.applications.EfficientNetB3(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)

inputs = tf.keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = base(inputs, training=False)
x = tf.keras.layers.Dropout(0.2)(x)

outputs = tf.keras.layers.Dense(num_classes, activation="softmax", dtype="float32")(x)
model = tf.keras.Model(inputs, outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="categorical_crossentropy",
    metrics=["accuracy"],
)

EPOCHS = 6

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 3
test_pred_df = (
    test_df.dropna(subset=["filepath"])
    .reset_index(drop=False)
    .rename(columns={"index": "orig_index"})
)

test_paths = test_pred_df["filepath"].astype(str).to_numpy()
test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
test_ds = test_ds.with_options(_ds_options())

test_ds = test_ds.map(_parse_test, num_parallel_calls=AUTOTUNE, deterministic=True)
test_ds = test_ds.cache(_TEST_CACHE)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=1)
pred_idx = np.argmax(probs, axis=1)
pred_labels = [classes[i] for i in pred_idx]

sub = sample_df.copy()
sub["label"] = "normal"
sub.loc[test_pred_df["orig_index"].values, "label"] = pred_labels

sub = sub[["image_id", "label"]]
assert len(sub) == len(sample_df), "Submission row count mismatch"
sub.head()




## === cell 4
out_path = "submission.csv"
sub.to_csv(out_path, index=False)
print(f"Wrote {out_path} with shape {sub.shape}")
print(sub["label"].value_counts().head())
