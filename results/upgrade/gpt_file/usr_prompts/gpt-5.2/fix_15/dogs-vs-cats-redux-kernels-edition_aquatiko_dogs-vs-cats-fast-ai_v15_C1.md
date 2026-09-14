# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Ensure it runs end-to-end and produces a valid submission file.


# 1. Kaggle task description

## Task
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

0.05806

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import re
import numpy as np
import pandas as pd
import random

random.seed(42)
np.random.seed(42)

print("Listing ../input:")
print(os.listdir("../input"))

PATH = "../input/"
ALT_BASE = os.path.join(PATH, "dogs-vs-cats-redux-kernels-edition")

TRAIN_DIR = os.path.join(PATH, "train")
if not os.path.exists(TRAIN_DIR) and os.path.exists(os.path.join(ALT_BASE, "train")):
    TRAIN_DIR = os.path.join(ALT_BASE, "train")

CAND_TEST_DIRS = [
    os.path.join(PATH, "test", "unknown"),
    os.path.join(ALT_BASE, "test", "unknown"),
    os.path.join(PATH, "test", "test"),
    os.path.join(ALT_BASE, "test", "test"),
]
TEST_DIR = None
for d in CAND_TEST_DIRS:
    if os.path.exists(d):
        TEST_DIR = d
        break

SAMPLE_SUB_CANDS = [
    os.path.join(ALT_BASE, "sample_submission.csv"),
    os.path.join(PATH, "sample_submission.csv"),
]
SAMPLE_SUB_PATH = None
for p in SAMPLE_SUB_CANDS:
    if os.path.exists(p):
        SAMPLE_SUB_PATH = p
        break

print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print(
    "TEST_DIR :",
    TEST_DIR,
    "exists:",
    (TEST_DIR is not None and os.path.exists(TEST_DIR)),
)
print(
    "SAMPLE_SUB:",
    SAMPLE_SUB_PATH,
    "exists:",
    (SAMPLE_SUB_PATH is not None and os.path.exists(SAMPLE_SUB_PATH)),
)

assert os.path.exists(TRAIN_DIR), f"Train dir not found: {TRAIN_DIR}"
assert TEST_DIR is not None and os.path.exists(
    TEST_DIR
), f"Test dir not found in candidates: {CAND_TEST_DIRS}"
assert SAMPLE_SUB_PATH is not None and os.path.exists(
    SAMPLE_SUB_PATH
), f"Sample submission not found in candidates: {SAMPLE_SUB_CANDS}"

sz = 224



## === cell 1
cat_dir = os.path.join(TRAIN_DIR, "cat")
dog_dir = os.path.join(TRAIN_DIR, "dog")
assert os.path.exists(cat_dir) and os.path.exists(
    dog_dir
), "Expected train/cat and train/dog folders."


def list_jpgs(folder):
    out = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".jpg"):
                out.append(e.path)
    out.sort()
    return out


cat_files = list_jpgs(cat_dir)
dog_files = list_jpgs(dog_dir)

train_files = np.array(cat_files + dog_files)
labels = np.array([0] * len(cat_files) + [1] * len(dog_files), dtype=np.float32)

print(
    "Train images:", len(train_files), "Cats:", len(cat_files), "Dogs:", len(dog_files)
)
print("Example:", train_files[-2], "label:", labels[-2])



## === cell 2
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import preprocess_input

print("TensorFlow:", tf.__version__)

try:
    tf.keras.utils.set_random_seed(42)
except Exception:
    tf.random.set_seed(42)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
MAP_PARALLEL_CALLS = AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) - 1)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




## === cell 3
def build_model(img_size=224):
    base = ResNet50(
        include_top=False, weights="imagenet", input_shape=(img_size, img_size, 3)
    )
    base.trainable = False  # mimic pretrained + precompute style (frozen backbone)

    inputs = layers.Input(shape=(img_size, img_size, 3))
    x = preprocess_input(inputs)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)

    model = models.Model(inputs, outputs)
    return model


model = build_model(sz)
model.summary()




## === cell 4
def binary_loss(y, p):
    y = np.asarray(y).astype(np.float64)
    p = np.asarray(p).astype(np.float64)
    eps = 1e-7
    p = np.clip(p, eps, 1 - eps)
    return -np.mean(y * np.log(p) + (1 - y) * np.log(1 - p))




## === cell 5

SZ_T = tf.constant(sz, dtype=tf.int32)


def _decode_resize_image(path, img_size_t):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size_t, img_size_t])
    img = tf.cast(img, tf.float32)
    return img


idx = np.arange(len(train_files))
rng = np.random.RandomState(42)
rng.shuffle(idx)

val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

trn_files, trn_labels = train_files[trn_idx], labels[trn_idx]
val_files, val_labels = train_files[val_idx], labels[val_idx]

batch_size = 32

options = tf.data.Options()
options.experimental_deterministic = True


def _decode_pair(path, y):
    img = _decode_resize_image(path, SZ_T)
    return img, y


def _augment_only(img, y):
    img = tf.image.random_flip_left_right(img)
    return img, y


_feature_extractor = tf.keras.Model(
    inputs=model.input,
    outputs=model.get_layer(
        index=[
            i
            for i, l in enumerate(model.layers)
            if isinstance(l, tf.keras.layers.GlobalAveragePooling2D)
        ][0]
    ).output,
)
_feature_extractor.trainable = False

gap_dim = _feature_extractor.output_shape[-1]
_head_input = layers.Input(shape=(gap_dim,), name="head_input_features")
x = model.layers[-3](_head_input)  # Dense(256, relu)
x = model.layers[-2](x)  # Dropout(0.5)
_head_output = model.layers[-1](x)  # Dense(1, sigmoid)
head_model = tf.keras.Model(_head_input, _head_output, name="head_model_shared_layers")

train_steps = int(np.floor(len(trn_files) / batch_size))
val_steps = int(np.ceil(len(val_files) / batch_size))
print("Train batches:", train_steps, "Val batches:", val_steps)

trn_decode_ds = (
    tf.data.Dataset.from_tensor_slices((trn_files, trn_labels))
    .with_options(options)
    .map(_decode_pair, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

val_decode_ds = (
    tf.data.Dataset.from_tensor_slices((val_files, val_labels))
    .with_options(options)
    .map(_decode_pair, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

trn_features = _feature_extractor.predict(trn_decode_ds.map(lambda x, y: x), verbose=0)
val_features = _feature_extractor.predict(val_decode_ds.map(lambda x, y: x), verbose=0)


def _features_for_augmented_pass(files_np, labels_np):
    ds = (
        tf.data.Dataset.from_tensor_slices((files_np, labels_np))
        .with_options(options)
        .map(_decode_pair, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
        .map(_augment_only, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )
    feats = _feature_extractor.predict(ds.map(lambda x, y: x), verbose=0)
    ys = np.concatenate(
        [y.numpy() for _, y in ds.unbatch().batch(4096)], axis=0
    ).astype(np.float32)
    return feats, ys


trn_features_e1, trn_labels_e1 = _features_for_augmented_pass(trn_files, trn_labels)
trn_features_e2, trn_labels_e2 = _features_for_augmented_pass(trn_files, trn_labels)

trn_features_2ep = np.concatenate([trn_features_e1, trn_features_e2], axis=0)
trn_labels_2ep = np.concatenate([trn_labels_e1, trn_labels_e2], axis=0)

feat_train_ds = (
    tf.data.Dataset.from_tensor_slices((trn_features_2ep, trn_labels_2ep))
    .with_options(options)
    .shuffle(2048, seed=42, reshuffle_each_iteration=False)
    .batch(batch_size, drop_remainder=True)
    .prefetch(AUTOTUNE)
)

feat_val_ds = (
    tf.data.Dataset.from_tensor_slices((val_features, val_labels))
    .with_options(options)
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)



## === cell 6
opt = tf.keras.optimizers.Adam(learning_rate=0.001)
head_model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

history = head_model.fit(
    feat_train_ds,
    validation_data=feat_val_ds,
    epochs=1,
    steps_per_epoch=train_steps * 2,
    validation_steps=int(np.ceil(len(val_files) / batch_size)),
    verbose=2,
)

base_model = None
for layer in model.layers:
    if isinstance(layer, tf.keras.Model) and layer.name.startswith("resnet50"):
        base_model = layer
        break

if base_model is not None:
    base_model.trainable = True
    for l in base_model.layers[:-30]:
        l.trainable = False

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
        metrics=[tf.keras.metrics.AUC(name="auc")],
    )

    train_ds_ft = (
        tf.data.Dataset.from_tensor_slices((trn_files, trn_labels))
        .with_options(options)
        .map(_decode_pair, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
        .shuffle(2048, seed=42, reshuffle_each_iteration=True)
        .map(_augment_only, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
        .batch(batch_size, drop_remainder=True)
        .prefetch(AUTOTUNE)
    )

    val_ds_ft = (
        tf.data.Dataset.from_tensor_slices((val_files, val_labels))
        .with_options(options)
        .map(_decode_pair, num_parallel_calls=MAP_PARALLEL_CALLS, deterministic=True)
        .batch(batch_size, drop_remainder=False)
        .prefetch(AUTOTUNE)
    )

    model.fit(
        train_ds_ft.repeat(),
        validation_data=val_ds_ft,
        epochs=1,
        steps_per_epoch=train_steps,
        validation_steps=val_steps,
        verbose=2,
    )




## === cell 7
def list_jpgs_flat(folder):
    out = []
    with os.scandir(folder) as it:
        for e in it:
            if e.is_file() and e.name.lower().endswith(".jpg"):
                out.append(e.path)
    out.sort()
    return out


test_files = list_jpgs_flat(TEST_DIR)
assert len(test_files) > 0, "No test images found."

print("Found test images:", len(test_files), "from:", TEST_DIR)

test_options = tf.data.Options()
test_options.experimental_deterministic = True

test_ds = (
    tf.data.Dataset.from_tensor_slices(test_files)
    .with_options(test_options)
    .map(
        lambda p: _decode_resize_image(p, SZ_T),
        num_parallel_calls=MAP_PARALLEL_CALLS,
        deterministic=True,
    )
    .batch(batch_size, drop_remainder=False)
    .prefetch(AUTOTUNE)
)

probs = model.predict(test_ds, verbose=0).reshape(-1)
probs = np.clip(probs, 1e-7, 1 - 1e-7)

_id_re = re.compile(r"(\d+)")


def extract_id(path):
    base = os.path.basename(path)
    stem = os.path.splitext(base)[0]
    m = _id_re.fullmatch(stem) or _id_re.search(stem)
    return int(m.group(1)) if m else int(_id_re.search(base).group(1))


ids = list(map(extract_id, test_files))
sub = pd.DataFrame({"id": ids, "label": probs})

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert (
    "id" in sample.columns and "label" in sample.columns
), "Unexpected sample_submission columns."

assert (
    len(sample) >= 10000
), f"sample_submission seems truncated (rows={len(sample)}). Using: {SAMPLE_SUB_PATH}"
assert len(sub) == len(test_files), "Internal mismatch between ids and predictions."

sub = sub.set_index("id").reindex(sample["id"].values).reset_index()

if sub["label"].isna().any():
    missing = int(sub["label"].isna().sum())
    print(
        "Warning: missing predictions for",
        missing,
        "ids; filling with 0.5 to keep a valid submission.",
    )
    sub["label"] = sub["label"].fillna(0.5)

sub["id"] = sub["id"].astype(int)
sub["label"] = sub["label"].astype(float)

print(sub.head())
print(sub.describe())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
assert len(sub) == len(
    sample
), f"Submission rows {len(sub)} != sample_submission rows {len(sample)}"
