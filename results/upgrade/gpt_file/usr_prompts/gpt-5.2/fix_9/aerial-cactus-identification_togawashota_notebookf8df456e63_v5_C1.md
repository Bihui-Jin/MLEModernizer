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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.13

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9940846666666666

# 6. Current score

0.69243

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.69243) has done: 'The timeout is dominated by training EfficientNetB3 for 50 epochs end-to-end; the biggest speed win without changing the model/loop is to avoid backprop through the heavy ImageNet backbone by freezing it (training only the added Dense head), which preserves the exact architecture and training semantics aside from which variables are trainable. Next, the tf.data pipeline is currently doing extra work by enumerating after repeat (creating ever-growing indices) and using `tf.cond` branches; we make augmentation stateless but compute seeds deterministically per-example per-epoch using a fixed per-sample index and epoch counter, and replace `tf.cond` with `tf.where`-style blending to reduce graph overhead while keeping identical augmentation distributions. We also ensure caching happens only for decoded images (not augmented) and use `drop_remainder=True` for the training batches to reduce shape polymorphism overhead (validation/test unchanged). These changes are equivalent in terms of inputs/labels, augmentation randomness distribution, loss, and evaluation, but drastically reduce per-step compute and input overhead to fit in 600 seconds.'

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd

print("Listing /kaggle/input/aerial-cactus-identification:")
base_in = "/kaggle/input/aerial-cactus-identification"
for fn in sorted(os.listdir(base_in))[:15]:
    print(os.path.join(base_in, fn))



## === cell 1
import zipfile

extract_dir = "/kaggle/working"


def _maybe_extract(zip_path, extract_to, expected_dirname):
    expected_path = os.path.join(extract_to, expected_dirname)
    if os.path.isdir(expected_path) and len(os.listdir(expected_path)) > 0:
        print(f"Skipping extraction (already exists): {expected_path}")
        return
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_to)
    print(f"Extracted: {zip_path} -> {extract_to}")


_maybe_extract(
    "/kaggle/input/aerial-cactus-identification/train.zip", extract_dir, "train"
)
_maybe_extract(
    "/kaggle/input/aerial-cactus-identification/test.zip", extract_dir, "test"
)



## === cell 2
print("Working dir top-level:", os.listdir("/kaggle/working")[:20])



## === cell 3
pass




## === cell 4
def resolve_data_dirs(base="/kaggle/working"):
    candidates = [
        (os.path.join(base, "train"), os.path.join(base, "test")),
        (
            os.path.join(base, "aerial-cactus-identification", "train"),
            os.path.join(base, "aerial-cactus-identification", "test"),
        ),
    ]
    for tr, te in candidates:
        if os.path.isdir(tr) and os.path.isdir(te):
            return tr, te
    found_train, found_test = None, None
    for root, dirs, _ in os.walk(base):
        if "train" in dirs and found_train is None:
            found_train = os.path.join(root, "train")
        if "test" in dirs and found_test is None:
            found_test = os.path.join(root, "test")
    if found_train and found_test:
        return found_train, found_test
    raise FileNotFoundError(
        "Could not locate extracted train/test directories under /kaggle/working"
    )


train_dir, test_dir = resolve_data_dirs("/kaggle/working")

train_df = pd.read_csv("/kaggle/input/aerial-cactus-identification/train.csv")
print(train_df.head())
print("Resolved train_dir:", train_dir)
print("Resolved test_dir:", test_dir)




## === cell 5
def count_files(directory):
    return sum(1 for e in os.scandir(directory) if e.is_file())


train_count = count_files(train_dir)
test_count = count_files(test_dir)

print(f"Train images: {train_count}")
print(f"Test images: {test_count}")

assert (
    train_count > 0 and test_count > 0
), "Train/test image folders not found or empty."



## === cell 6
class_ratio = train_df["has_cactus"].value_counts(normalize=True) * 100
print(class_ratio)



## === cell 7
pass



## === cell 8
train_df["has_cactus"] = train_df["has_cactus"].astype("str")



## === cell 9
import random


def custom_preprocessing(image):
    k = random.randint(0, 3)
    image = np.rot90(image, k)

    if random.random() > 0.5:
        image = np.fliplr(image)

    if random.random() > 0.5:
        image = np.flipud(image)

    factor = random.uniform(0.8, 1.2)
    image = np.clip(image.astype(np.float32) * factor, 0.0, 255.0) / 255.0
    return image.astype(np.float32)




## === cell 10
import tensorflow as tf

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Warning: could not set XLA JIT flag:", repr(e))

SEED = 42
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
os.environ.setdefault("PYTHONHASHSEED", str(SEED))

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

df = train_df.copy()
df["_label_int"] = df["has_cactus"].astype(int)

idx = np.arange(len(df))
rng = np.random.default_rng(SEED)
rng.shuffle(idx)
val_size = int(round(0.10 * len(df)))
val_idx = idx[:val_size]
train_idx = idx[val_size:]

train_split = df.iloc[train_idx].reset_index(drop=True)
val_split = df.iloc[val_idx].reset_index(drop=True)

print("Train split:", train_split.shape, "Val split:", val_split.shape)


def _read_image_bytes(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, [32, 32], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)
    return img


def _augment_tf(img, seed_pair):
    seed_pair = tf.cast(seed_pair, tf.int32)

    k = tf.random.stateless_uniform([], seed_pair, minval=0, maxval=4, dtype=tf.int32)
    img = tf.image.rot90(img, k=k)

    seed_lr = seed_pair + tf.constant([1, 0], tf.int32)
    seed_ud = seed_pair + tf.constant([2, 0], tf.int32)
    seed_fac = seed_pair + tf.constant([3, 0], tf.int32)

    do_lr = tf.random.stateless_uniform([], seed_lr) > 0.5
    do_ud = tf.random.stateless_uniform([], seed_ud) > 0.5

    img_lr = tf.image.flip_left_right(img)
    img = tf.where(do_lr, img_lr, img)

    img_ud = tf.image.flip_up_down(img)
    img = tf.where(do_ud, img_ud, img)

    factor = tf.random.stateless_uniform(
        [], seed_fac, minval=0.8, maxval=1.2, dtype=tf.float32
    )
    img = tf.clip_by_value(img * factor, 0.0, 255.0) / 255.0
    img = tf.cast(img, tf.float32)
    img.set_shape([32, 32, 3])
    return img


def make_ds(split_df, training, batch_size):
    ids = tf.constant(split_df["id"].astype(str).to_list(), dtype=tf.string)
    labels = tf.constant(split_df["_label_int"].to_numpy(dtype=np.float32))

    base = tf.constant(train_dir.rstrip("/") + "/", dtype=tf.string)
    paths = tf.strings.join([base, ids])
    n = int(split_df.shape[0])

    ex_idx = tf.range(n, dtype=tf.int32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels, ex_idx))

    def _decode_map(path, label, ex_i):
        img = _read_image_bytes(path)
        img = tf.cast(img, tf.float32)  # keep as 0..255 float32 for augmentation parity
        return img, label, ex_i

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)

    ds = ds.map(_decode_map, num_parallel_calls=tf.data.AUTOTUNE).cache()

    if training:
        shuffle_buf = min(n, 4096)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

        ds = ds.repeat()

        ds = ds.batch(n, drop_remainder=True).enumerate().unbatch()

        def _aug_map(epoch_i, data):
            img, label, ex_i = data
            epoch_i = tf.cast(epoch_i, tf.int32)
            ex_i = tf.cast(ex_i, tf.int32)
            seed_pair = tf.stack([tf.cast(SEED, tf.int32) + epoch_i, ex_i])
            img = _augment_tf(img, seed_pair)
            return img, label

        ds = ds.map(_aug_map, num_parallel_calls=tf.data.AUTOTUNE)
    else:

        def _val_map(img, label, ex_i):
            img = img / 255.0
            img.set_shape([32, 32, 3])
            return img, label

        ds = ds.map(_val_map, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=training).prefetch(tf.data.AUTOTUNE)
    return ds, n


train_dataset, n_train = make_ds(train_split, training=True, batch_size=128)
val_dataset, n_val = make_ds(val_split, training=False, batch_size=64)

steps_per_epoch = (n_train + 128 - 1) // 128
validation_steps = (n_val + 64 - 1) // 64

for xb, yb in train_dataset.take(1):
    print("Train batch:", xb.shape, yb.shape, xb.dtype, yb.dtype)
print("steps_per_epoch:", steps_per_epoch, "validation_steps:", validation_steps)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 11
pass



## === cell 12
sample_sub = pd.read_csv(
    "/kaggle/input/aerial-cactus-identification/sample_submission.csv"
)

test_ids = tf.constant(sample_sub["id"].astype(str).to_list(), dtype=tf.string)
test_base = tf.constant(test_dir.rstrip("/") + "/", dtype=tf.string)
test_paths = tf.strings.join([test_base, test_ids])

test_ds = tf.data.Dataset.from_tensor_slices(test_paths)


def _map_test(path):
    img = _read_image_bytes(path)
    img = img / 255.0
    img.set_shape([32, 32, 3])
    return img


opts = tf.data.Options()
opts.experimental_deterministic = True
test_dataset = (
    test_ds.with_options(opts)
    .map(_map_test, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(64)
    .prefetch(tf.data.AUTOTUNE)
)
assert len(sample_sub) > 0, "Sample submission is empty."
print("Test dataset batches:", int(np.ceil(len(sample_sub) / 64)))



## === cell 13
import tensorflow as tf
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import Adam

efficient_net = EfficientNetB3(
    weights="imagenet",
    input_shape=(32, 32, 3),
    include_top=False,
    pooling="max",
)

efficient_net.trainable = False

model = Sequential()
model.add(efficient_net)
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=120, activation="relu"))
model.add(Dense(units=1, activation="sigmoid"))
model.summary()



## === cell 14
model.compile(
    optimizer=Adam(learning_rate=0.0001),
    loss="binary_crossentropy",
    metrics=["accuracy"],
    steps_per_execution=32,
)



## === cell 15
history = model.fit(
    train_dataset,
    epochs=50,
    validation_data=val_dataset,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2468127752.py in <cell line: 0>()
      1 history = model.fit(
----> 2     train_dataset,
      3     epochs=50,
      4     validation_data=val_dataset,
      5     steps_per_epoch=steps_per_epoch,

NameError: name 'train_dataset' is not defined

## === cell 16
pass



## === cell 17
preds = model.predict(test_dataset, verbose=1)
preds = preds.reshape(-1)
preds = np.clip(preds, 0.0, 1.0)

submission = pd.DataFrame({"id": sample_sub["id"].values, "has_cactus": preds})
print(submission.head(10))
print(submission.shape)
assert (
    submission.shape[0] == sample_sub.shape[0]
), "Row count mismatch vs sample_submission."



## === cell 18
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print(f"Wrote: {out_path}")

print(os.listdir("/kaggle/working")[:30])
print(pd.read_csv(out_path).head())
