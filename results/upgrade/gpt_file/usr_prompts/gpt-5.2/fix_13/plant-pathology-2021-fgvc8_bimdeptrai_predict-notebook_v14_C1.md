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
import random
import numpy as np
import pandas as pd

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE
print("TF:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass




## === cell 1
BASE_DIR = "../input/plant-pathology-2021-fgvc8"
if not os.path.exists(os.path.join(BASE_DIR, "train.csv")):
    BASE_DIR = "../input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8"

TRAIN_CSV = os.path.join(BASE_DIR, "train.csv")
SAMPLE_SUB = os.path.join(BASE_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(BASE_DIR, "train_images")
TEST_IMG_DIR = os.path.join(BASE_DIR, "test_images")

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_IMG_DIR), f"Missing {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_IMG_DIR), f"Missing {TRAIN_IMG_DIR}"

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
print(train.head())




## === cell 2
h_target = 512
w_target = 512
batch_size = 32

label_split = train["labels"].str.split()
mlb = MultiLabelBinarizer().fit(label_split)
class_names = list(mlb.classes_)
n_classes = len(class_names)

print("n_classes:", n_classes)
print("classes:", class_names)




## === cell 3
train_df = train.copy()

train_df["labels_list"] = train_df["labels"].str.split()
Y = mlb.transform(train_df["labels_list"].values).astype("float32")

rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_images = train_df.loc[trn_idx, "image"].values
valid_images = train_df.loc[val_idx, "image"].values
Y_train = Y[trn_idx]
Y_valid = Y[val_idx]


@tf.function
def _load_and_preprocess(path, target_size):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, fancy_upscaling=False)
    img = tf.image.resize(img, target_size, method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) / 255.0
    img.set_shape([None, None, 3])
    return img


def _build_image_ds(
    image_names,
    y,
    image_dir,
    batch_size,
    target_size,
    shuffle,
    seed,
    cache_mode,  # None | "memory"
):
    image_names = tf.convert_to_tensor(image_names, dtype=tf.string)
    base = tf.constant(image_dir + os.sep, dtype=tf.string)
    paths = tf.strings.join([base, image_names])

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(
            lambda p: _load_and_preprocess(p, target_size),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        y = tf.convert_to_tensor(y, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))
        ds = ds.map(
            lambda p, label: (_load_and_preprocess(p, target_size), label),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.apply(tf.data.experimental.ignore_errors())

    if cache_mode == "memory":
        ds = ds.cache()

    if shuffle:
        n = (
            int(paths.shape[0])
            if paths.shape.rank == 0 or paths.shape[0] is None
            else int(paths.shape[0])
        )
        buf = int(min(n if n else 2048, 2048))
        ds = ds.shuffle(buffer_size=buf, seed=seed, reshuffle_each_iteration=True)

    ds = ds.batch(batch_size, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_deterministic = True
    try:
        options.experimental_slack = True
    except Exception:
        pass
    ds = ds.with_options(options)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = _build_image_ds(
    train_images,
    Y_train,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=True,
    seed=SEED,
    cache_mode="memory",
)
valid_ds = _build_image_ds(
    valid_images,
    Y_valid,
    TRAIN_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache_mode="memory",
)
test_ds = _build_image_ds(
    submissions["image"].values,
    None,
    TEST_IMG_DIR,
    batch_size,
    (h_target, w_target),
    shuffle=False,
    seed=SEED,
    cache_mode=None,
)

print(
    "Train samples:",
    len(train_images),
    "Valid samples:",
    len(valid_images),
    "Test samples:",
    len(submissions),
)




## === cell 4
base = tf.keras.applications.MobileNetV2(
    input_shape=(h_target, w_target, 3),
    include_top=False,
    weights="imagenet",
    pooling="avg",
)
inputs = keras.Input(shape=(h_target, w_target, 3))
x = base(inputs, training=False)
outputs = keras.layers.Dense(n_classes, activation="sigmoid")(x)
model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    steps_per_execution=8,
)

EPOCHS = 3

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=valid_ds,
    verbose=1,
)




## === cell 5
valid_probs = model.predict(valid_ds, verbose=1)
y_true = Y_valid[: valid_probs.shape[0]].astype(np.int32)

grid = np.array([0.15, 0.2, 0.25, 0.3, 0.35, 0.4, 0.45, 0.5], dtype=np.float32)
best_thr = np.full((n_classes,), 0.5, dtype=np.float32)

P = valid_probs.astype(np.float32)  # (N, C)
T = y_true.astype(np.int32)  # (N, C)

pred_all = (P[None, :, :] >= grid[:, None, None]).astype(np.int32)

tp = (pred_all & T[None, :, :]).sum(axis=1).astype(np.float32)  # (G, C)
fp = (pred_all & (1 - T[None, :, :])).sum(axis=1).astype(np.float32)  # (G, C)
fn = ((1 - pred_all) & T[None, :, :]).sum(axis=1).astype(np.float32)  # (G, C)

f1 = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-9)  # (G, C)
best_idx = np.argmax(f1, axis=0)  # (C,)
best_thr = grid[best_idx].astype(np.float32)

print("Thresholds (first 10):", best_thr[:10])




## === cell 6
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
print(preds[:2])

preds = preds.astype(np.float32)
chosen_mask = preds >= best_thr[None, :]  # (N, C)
top_idx = np.argmax(preds, axis=1).astype(np.int32)

none_chosen = ~chosen_mask.any(axis=1)
chosen_mask[none_chosen, :] = False
chosen_mask[none_chosen, top_idx[none_chosen]] = True

if "healthy" in class_names:
    healthy_idx = int(class_names.index("healthy"))
    has_healthy = chosen_mask[:, healthy_idx]
    has_other = chosen_mask.sum(axis=1) > 1
    drop_healthy = has_healthy & has_other
    chosen_mask[drop_healthy, healthy_idx] = False
    empty_after = ~chosen_mask.any(axis=1)
    chosen_mask[empty_after, healthy_idx] = True

idx_to_name = np.array(class_names, dtype=object)
row_idx, col_idx = np.nonzero(chosen_mask)
groups = np.split(
    col_idx, np.cumsum(np.bincount(row_idx, minlength=chosen_mask.shape[0]))[:-1]
)
pred_labels = [" ".join(idx_to_name[g].tolist()) if len(g) else "" for g in groups]

submissions = submissions.copy()
submissions["labels"] = pred_labels
print(submissions.head())




## === cell 7
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.head(10))
