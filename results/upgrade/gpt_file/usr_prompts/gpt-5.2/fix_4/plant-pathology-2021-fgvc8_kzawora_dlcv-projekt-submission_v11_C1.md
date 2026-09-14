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


def import_tensorflow_safely():
    try:
        import tensorflow as tf  # noqa: F401

        return tf
    except Exception as e1:
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
        os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
        try:
            import importlib

            if "tensorflow" in globals():
                importlib.reload(globals()["tensorflow"])
            import tensorflow as tf  # noqa: F401

            return tf
        except Exception as e2:
            raise RuntimeError(
                "TensorFlow import failed both normally and with pure-Python protobuf fallback.\n"
                f"First error: {repr(e1)}\nSecond error: {repr(e2)}"
            )


import random
import numpy as np
import pandas as pd
from pathlib import Path

tf = import_tensorflow_safely()

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

CANDIDATE_INPUT_ROOTS = [
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
    "/kaggle/input",
    "/kaggle/data",
]
INPUT_ROOT = None
for root in CANDIDATE_INPUT_ROOTS:
    if os.path.exists(os.path.join(root, "train.csv")) and os.path.exists(
        os.path.join(root, "sample_submission.csv")
    ):
        INPUT_ROOT = root
        break
    nested = os.path.join(root, "plant-pathology-2021-fgvc8")
    if os.path.exists(os.path.join(nested, "train.csv")) and os.path.exists(
        os.path.join(nested, "sample_submission.csv")
    ):
        INPUT_ROOT = nested
        break

if INPUT_ROOT is None:
    raise FileNotFoundError(
        "Could not find train.csv and sample_submission.csv under expected Kaggle input paths. "
        f"Tried: {CANDIDATE_INPUT_ROOTS}"
    )

WORK_ROOT = "/kaggle/working"
TMP_ROOT = "/kaggle/tmp"

train_csv_path = f"{INPUT_ROOT}/train.csv"
sample_sub_path = f"{INPUT_ROOT}/sample_submission.csv"
train_img_dir = f"{INPUT_ROOT}/train_images"
test_img_dir = f"{INPUT_ROOT}/test_images"

assert os.path.exists(train_csv_path), f"Missing train.csv at {train_csv_path}"
assert os.path.exists(
    sample_sub_path
), f"Missing sample_submission.csv at {sample_sub_path}"
assert os.path.isdir(train_img_dir), f"Missing train_images dir at {train_img_dir}"
assert os.path.isdir(test_img_dir), f"Missing test_images dir at {test_img_dir}"

print("Using INPUT_ROOT:", INPUT_ROOT)
print("TensorFlow:", tf.__version__)
print("Train images:", len(list(Path(train_img_dir).glob("*.jpg"))))
print("Test images:", len(list(Path(test_img_dir).glob("*.jpg"))))



## === cell 1
train_df = pd.read_csv(train_csv_path)
sample_sub = pd.read_csv(sample_sub_path)

LABELS = ["complex", "frog_eye_leaf_spot", "powdery_mildew", "rust", "scab"]


def labels_to_multi_hot(label_str: str, label_list):
    s = str(label_str)
    tokens = s.split()
    return np.array([1 if lab in tokens else 0 for lab in label_list], dtype=np.float32)


train_df["filepath"] = train_df["image"].apply(lambda x: str(Path(train_img_dir) / x))
missing = (~train_df["filepath"].apply(lambda p: Path(p).exists())).sum()
assert missing == 0, f"Some train image files are missing: {missing}"

y = np.stack(
    [labels_to_multi_hot(s, LABELS) for s in train_df["labels"].values], axis=0
)

idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
val_frac = 0.1
val_size = int(len(idx) * val_frac)
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

train_paths = train_df.iloc[tr_idx]["filepath"].values
train_y = y[tr_idx]
val_paths = train_df.iloc[val_idx]["filepath"].values
val_y = y[val_idx]

print("Train/Val sizes:", len(train_paths), len(val_paths))
print("Label prevalence (train):", dict(zip(LABELS, train_y.mean(axis=0).round(4))))



## === cell 2
AUTOTUNE = tf.data.AUTOTUNE
TARGET_SIZE = (380, 380)
BATCH_SIZE = 16  # keep runtime reasonable while not changing algorithmic intent


def decode_resize(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, TARGET_SIZE, method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    if label is None:
        return img
    return img, label


def make_ds(paths, labels=None, training=False):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
        ds = ds.map(lambda p: decode_resize(p, None), num_parallel_calls=AUTOTUNE)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
        if training:
            ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=True)
        ds = ds.map(decode_resize, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(train_paths, train_y, training=True)
val_ds = make_ds(val_paths, val_y, training=False)



## === cell 3
inputs = tf.keras.Input(shape=(TARGET_SIZE[0], TARGET_SIZE[1], 3))
x = tf.keras.layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = tf.keras.layers.MaxPooling2D()(x)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2, seed=SEED)(x)
outputs = tf.keras.layers.Dense(len(LABELS), activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

print(model.summary())

EPOCHS = 2
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 4
test_paths = sample_sub["image"].apply(lambda x: str(Path(test_img_dir) / x)).values
exists_mask = np.array([Path(p).exists() for p in test_paths], dtype=bool)
print("Test files found locally:", int(exists_mask.sum()), "/", len(test_paths))

preds = np.zeros((len(test_paths), len(LABELS)), dtype=np.float32)
if exists_mask.any():
    test_ds = make_ds(test_paths[exists_mask], labels=None, training=False)
    preds_exist = model.predict(test_ds, verbose=1)
    preds[exists_mask] = preds_exist.astype(np.float32)

print("Preds shape:", preds.shape)
print("Preds range:", float(np.min(preds)), float(np.max(preds)))



## === cell 5
threshold = 0.4

z = (preds > threshold).astype(np.int32)
predictions = [[LABELS[i] for i, flag in enumerate(row) if flag == 1] for row in z]
predictions_str = [" ".join(p) if len(p) > 0 else "healthy" for p in predictions]

print("Example prediction strings:", predictions_str[:5])

sub = pd.DataFrame({"image": sample_sub["image"].values, "labels": predictions_str})
assert sub.shape[0] == sample_sub.shape[0]
assert list(sub.columns) == ["image", "labels"]

sub.head()



## === cell 6
out_path = f"{WORK_ROOT}/submission.csv"
sub.to_csv(out_path, index=False)
print("Wrote:", out_path)
print("Submission preview:")
print(sub.head(10).to_string(index=False))
print("Submission rows:", len(sub))
