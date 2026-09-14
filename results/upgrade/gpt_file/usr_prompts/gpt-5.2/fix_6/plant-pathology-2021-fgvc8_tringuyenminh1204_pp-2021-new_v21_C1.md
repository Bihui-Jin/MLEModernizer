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
import os, random, math, re
import numpy as np
import pandas as pd
import tensorflow as tf

print("TF:", tf.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
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
test = pd.read_csv(path + "sample_submission.csv")
sub = pd.read_csv(path + "sample_submission.csv")

print(train.shape, test.shape)
train.head()




## === cell 2
AUTO = tf.data.AUTOTUNE




## === cell 3
from matplotlib import pyplot as plt

sample_img_path = os.path.join(path, "train_images", train.iloc[0]["image"])
img = plt.imread(sample_img_path)
print(img.shape)
plt.imshow(img)
plt.axis("off")




## === cell 4
import pathlib




## === cell 5
train_dir = os.path.join(path, "train_images")
test_dir = os.path.join(path, "test_images")

train_paths = [os.path.join(train_dir, fn) for fn in train["image"].values]
test_paths = [os.path.join(test_dir, fn) for fn in sub["image"].values]

assert len(train_paths) == len(train)
assert len(test_paths) == len(sub)




## === cell 6
all_classes = sorted({c for s in train["labels"].astype(str).values for c in s.split()})
print("Classes:", all_classes)
all_classes




## === cell 7
class_to_idx = {c: i for i, c in enumerate(all_classes)}
y = np.zeros((len(train), len(all_classes)), dtype=np.float32)

labels_list = train["labels"].astype(str).values
for i, s in enumerate(labels_list):
    toks = s.split()
    if toks:
        js = [class_to_idx[t] for t in toks if t in class_to_idx]
        if js:
            y[i, js] = 1.0

new_train = pd.DataFrame(y, columns=all_classes)
new_train.insert(0, "image", train["image"].values)
new_train.head()




## === cell 8
new_train




## === cell 9
def decode_image(filename, label=None, image_size=(512, 512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3, dct_method="INTEGER_FAST")
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size, method="bilinear", antialias=False)
    if label is None:
        return image
    else:
        return image, label




## === cell 10
test_paths[:5], len(test_paths)




## === cell 11
_has_gpu = len(tf.config.list_physical_devices("GPU")) > 0
BATCH_SIZE = 32 if _has_gpu else 16




## === cell 12
def _configure_dataset(ds: tf.data.Dataset) -> tf.data.Dataset:
    opts = tf.data.Options()
    opts.deterministic = True
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
    try:
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return ds.with_options(opts)


def map_batch_prefetch(
    ds, map_fn, batch_size, deterministic=True, cache=False, shuffle=None, seed=None
):
    if shuffle is not None:
        ds = ds.shuffle(shuffle, seed=seed, reshuffle_each_iteration=True)
    ds = ds.map(map_fn, num_parallel_calls=AUTO, deterministic=deterministic)
    ds = ds.batch(batch_size, drop_remainder=False)
    if cache:
        ds = ds.cache(cache) if isinstance(cache, str) else ds.cache()
    ds = ds.prefetch(AUTO)
    ds = _configure_dataset(ds)
    return ds




## === cell 13
test_dataset = tf.data.Dataset.from_tensor_slices(test_paths)
test_dataset = map_batch_prefetch(
    test_dataset,
    lambda f: decode_image(f, None, image_size=(512, 512)),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=False,
)
test_dataset = test_dataset.apply(tf.data.experimental.ignore_errors())




## === cell 14
import tensorflow as tf
from tensorflow import keras




## === cell 15
IMAGE_SIZE = (512, 512)
NUM_CLASSES = len(all_classes)

idx = np.arange(len(new_train))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(idx))
tr_idx, va_idx = idx[:split], idx[split:]

x_train = [train_paths[i] for i in tr_idx]
y_train = new_train.iloc[tr_idx][all_classes].values.astype(np.float32)

x_val = [train_paths[i] for i in va_idx]
y_val = new_train.iloc[va_idx][all_classes].values.astype(np.float32)

cache_dir = "/kaggle/working/tf_cache_pp2021"
os.makedirs(cache_dir, exist_ok=True)
train_cache_path = os.path.join(
    cache_dir, f"train_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_bs{BATCH_SIZE}.cache"
)
val_cache_path = os.path.join(
    cache_dir, f"val_{IMAGE_SIZE[0]}x{IMAGE_SIZE[1]}_bs{BATCH_SIZE}.cache"
)

train_ds = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_ds = map_batch_prefetch(
    train_ds,
    lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=train_cache_path,
    shuffle=2048,
    seed=SEED,
)
train_ds = train_ds.apply(tf.data.experimental.ignore_errors())

val_ds = tf.data.Dataset.from_tensor_slices((x_val, y_val))
val_ds = map_batch_prefetch(
    val_ds,
    lambda f, y: decode_image(f, y, image_size=IMAGE_SIZE),
    batch_size=BATCH_SIZE,
    deterministic=True,
    cache=val_cache_path,
)
val_ds = val_ds.apply(tf.data.experimental.ignore_errors())

inputs = keras.Input(shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3))
x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)
x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dense(128, activation="relu")(x)
outputs = keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()

steps_per_epoch = int(math.ceil(len(x_train) / BATCH_SIZE))
validation_steps = int(math.ceil(len(x_val) / BATCH_SIZE))




## === cell 16
EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=steps_per_epoch,
    validation_steps=validation_steps,
    verbose=1,
)




## === cell 17
test_steps = int(math.ceil(len(test_paths) / BATCH_SIZE))
probs = model.predict(test_dataset, steps=test_steps, verbose=1)
probs.shape




## === cell 18
class_to_idx = {c: i for i, c in enumerate(all_classes)}
idx_to_class = {i: c for c, i in class_to_idx.items()}

default_thr = 0.30
threshold_by_class = {
    "scab": 0.20,
    "frog_eye_leaf_spot": 0.30,
    "complex": 0.15,
    "rust": 0.30,
    "powdery_mildew": 0.35,
    "healthy": 0.50,  # only chosen when nothing else triggers; higher reduces false "healthy".
}
thr = np.array(
    [threshold_by_class.get(c, default_thr) for c in all_classes], dtype=np.float32
)

probs_np = np.asarray(probs, dtype=np.float32)

healthy_idx = class_to_idx.get("healthy", None)
non_healthy_mask = np.ones(NUM_CLASSES, dtype=bool)
if healthy_idx is not None:
    non_healthy_mask[healthy_idx] = False

hits_mat = probs_np[:, non_healthy_mask] > thr[non_healthy_mask]

hit_counts = hits_mat.sum(axis=1)
is_complex = hit_counts >= 3
is_healthy = hit_counts == 0  # only if nothing else triggers

non_healthy_classes = np.array([c for c in all_classes if c != "healthy"], dtype=object)

pred_string = []
for i in range(probs_np.shape[0]):
    if is_complex[i]:
        pred_string.append("complex")
    elif is_healthy[i]:
        pred_string.append("healthy")
    else:
        pred_string.append(" ".join(non_healthy_classes[hits_mat[i]].tolist()))

submission = sub.copy()
submission["labels"] = pred_string

submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath("submission.csv"))
