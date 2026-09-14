# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

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
TRAIN_DIR = os.path.join(PATH, "train")
TEST_DIR = os.path.join(PATH, "test", "unknown")

ALT_BASE = os.path.join(PATH, "dogs-vs-cats-redux-kernels-edition")
if not os.path.exists(TRAIN_DIR) and os.path.exists(os.path.join(ALT_BASE, "train")):
    TRAIN_DIR = os.path.join(ALT_BASE, "train")
if not os.path.exists(TEST_DIR) and os.path.exists(
    os.path.join(ALT_BASE, "test", "unknown")
):
    TEST_DIR = os.path.join(ALT_BASE, "test", "unknown")

SAMPLE_SUB_PATH = os.path.join(PATH, "sample_submission.csv")
if not os.path.exists(SAMPLE_SUB_PATH) and os.path.exists(
    os.path.join(ALT_BASE, "sample_submission.csv")
):
    SAMPLE_SUB_PATH = os.path.join(ALT_BASE, "sample_submission.csv")

print("TRAIN_DIR:", TRAIN_DIR, "exists:", os.path.exists(TRAIN_DIR))
print("TEST_DIR :", TEST_DIR, "exists:", os.path.exists(TEST_DIR))
print("SAMPLE_SUB:", SAMPLE_SUB_PATH, "exists:", os.path.exists(SAMPLE_SUB_PATH))

assert os.path.exists(TRAIN_DIR), f"Train dir not found: {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Test dir not found: {TEST_DIR}"
assert os.path.exists(
    SAMPLE_SUB_PATH
), f"Sample submission not found: {SAMPLE_SUB_PATH}"

sz = 224



## === cell 1
cat_dir = os.path.join(TRAIN_DIR, "cat")
dog_dir = os.path.join(TRAIN_DIR, "dog")
assert os.path.exists(cat_dir) and os.path.exists(
    dog_dir
), "Expected train/cat and train/dog folders."

cat_files = sorted(
    [
        os.path.join(cat_dir, f)
        for f in os.listdir(cat_dir)
        if f.lower().endswith(".jpg")
    ]
)
dog_files = sorted(
    [
        os.path.join(dog_dir, f)
        for f in os.listdir(dog_dir)
        if f.lower().endswith(".jpg")
    ]
)

train_files = np.array(cat_files + dog_files)
labels = np.array(
    [0] * len(cat_files) + [1] * len(dog_files), dtype=np.float32
)  # 0=cat, 1=dog

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

AUTOTUNE = tf.data.experimental.AUTOTUNE
MAP_PARALLEL_CALLS = max(1, (os.cpu_count() or 2) // 2)

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
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
def decode_and_resize(path, label=None, img_size=224, training=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [img_size, img_size])
    img = tf.cast(img, tf.float32)

    if training:
        img = tf.image.random_flip_left_right(img)

    if label is None:
        return img
    return img, tf.cast(label, tf.float32)


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

train_base = tf.data.Dataset.from_tensor_slices((trn_files, trn_labels)).with_options(
    options
)
train_base = train_base.shuffle(2048, seed=42, reshuffle_each_iteration=True)
train_ds = train_base.map(
    lambda p, y: decode_and_resize(p, y, sz, training=True),
    num_parallel_calls=MAP_PARALLEL_CALLS,
).cache()
train_ds = train_ds.batch(batch_size, drop_remainder=True).prefetch(AUTOTUNE)

val_base = tf.data.Dataset.from_tensor_slices((val_files, val_labels)).with_options(
    options
)
val_ds = val_base.map(
    lambda p, y: decode_and_resize(p, y, sz, training=False),
    num_parallel_calls=MAP_PARALLEL_CALLS,
).cache()
val_ds = val_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

print(
    "Train batches:",
    int(np.floor(len(trn_files) / batch_size)),
    "Val batches:",
    int(np.ceil(len(val_files) / batch_size)),
)



## === cell 6
opt = tf.keras.optimizers.Adam(learning_rate=0.01)
model.compile(
    optimizer=opt,
    loss="binary_crossentropy",
    metrics=[tf.keras.metrics.AUC(name="auc")],
)

history = model.fit(train_ds, validation_data=val_ds, epochs=2, verbose=2)

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
    model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)



## === cell 7
test_files = sorted(
    [
        os.path.join(TEST_DIR, f)
        for f in os.listdir(TEST_DIR)
        if f.lower().endswith(".jpg")
    ]
)
assert len(test_files) > 0, "No test images found."

test_options = tf.data.Options()
test_options.experimental_deterministic = True

test_ds = tf.data.Dataset.from_tensor_slices(test_files).with_options(test_options)
test_ds = test_ds.map(
    lambda p: decode_and_resize(p, None, sz, training=False),
    num_parallel_calls=MAP_PARALLEL_CALLS,
)
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

probs = model.predict(test_ds, verbose=0).reshape(-1)
probs = np.clip(probs, 1e-7, 1 - 1e-7)


def extract_id(path):
    base = os.path.basename(path)
    m = re.search(r"(\d+)", base)
    return int(m.group(1)) if m else base


ids = [extract_id(p) for p in test_files]

sub = pd.DataFrame({"id": ids, "label": probs})

sample = pd.read_csv(SAMPLE_SUB_PATH)
if "id" in sample.columns and len(sample) == len(sub):
    sub = sub.set_index("id").reindex(sample["id"].values).reset_index()

sub["id"] = sub["id"].astype(int)
sub["label"] = sub["label"].astype(float)

print(sub.head())
print(sub.describe())

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0
