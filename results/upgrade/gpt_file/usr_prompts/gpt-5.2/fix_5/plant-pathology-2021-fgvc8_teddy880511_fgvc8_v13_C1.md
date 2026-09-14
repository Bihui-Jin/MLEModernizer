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
import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_XLA_FLAGS"] = "--tf_xla_auto_jit=2"
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

np.random.seed(42)
tf.random.set_seed(42)
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

DATA_ROOT = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_IMG_DIR = os.path.join(DATA_ROOT, "train_images")
TEST_IMG_DIR = os.path.join(DATA_ROOT, "test_images")
TRAIN_CSV = os.path.join(DATA_ROOT, "train.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_IMG_DIR), f"Missing train images dir: {TRAIN_IMG_DIR}"
assert os.path.exists(TEST_IMG_DIR), f"Missing test images dir: {TEST_IMG_DIR}"
assert os.path.exists(TRAIN_CSV), f"Missing train.csv: {TRAIN_CSV}"
assert os.path.exists(
    SAMPLE_SUB_CSV
), f"Missing sample_submission.csv: {SAMPLE_SUB_CSV}"

IMG_SIZE = 64

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
label_class = [
    "scab",
    "healthy",
    "frog_eye_leaf_spot",
    "cider_apple_rust",
    "complex",
    "powdery_mildew",
    "scab frog_eye_leaf_spot",  # kept to preserve original 7-class setup
]
class_to_idx = {c: i for i, c in enumerate(label_class)}

train_df = pd.read_csv(TRAIN_CSV)
train_df["labels"] = train_df["labels"].fillna("").astype(str)

n = len(train_df)
y_train = np.zeros((n, len(label_class)), dtype=np.float32)

labels_series = train_df["labels"]

mask_combo = labels_series.str.contains(r"(^| )scab( |$)") & labels_series.str.contains(
    r"(^| )frog_eye_leaf_spot( |$)"
)
y_train[mask_combo.values, class_to_idx["scab frog_eye_leaf_spot"]] = 1.0

labels_clean = labels_series.where(
    ~mask_combo,
    labels_series.str.replace(r"(^| )scab( |$)", " ", regex=True).str.replace(
        r"(^| )frog_eye_leaf_spot( |$)", " ", regex=True
    ),
)


def _set_if_present(tok, idx):
    m = labels_clean.str.contains(rf"(^| ){tok}( |$)", regex=True)
    y_train[m.values, idx] = 1.0


_set_if_present("scab", class_to_idx["scab"])
_set_if_present("healthy", class_to_idx["healthy"])
_set_if_present("frog_eye_leaf_spot", class_to_idx["frog_eye_leaf_spot"])
_set_if_present("cider_apple_rust", class_to_idx["cider_apple_rust"])
_set_if_present("complex", class_to_idx["complex"])
_set_if_present("powdery_mildew", class_to_idx["powdery_mildew"])




## === cell 2
imgfiles = train_df["image"].tolist()
train_paths = [os.path.join(TRAIN_IMG_DIR, f) for f in imgfiles]

BATCH_SIZE = 100
EPOCHS = 20


@tf.function
def _load_and_preprocess(path, y=None):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    if y is None:
        return img
    return img, y


ds_opts = tf.data.Options()
ds_opts.experimental_optimization.map_parallelization = True
ds_opts.experimental_optimization.parallel_batch = True
try:
    ds_opts.experimental_deterministic = False
except Exception:
    pass

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
ds_train = ds_train.with_options(ds_opts)

ds_train = ds_train.shuffle(
    buffer_size=min(len(train_paths), 8192), seed=42, reshuffle_each_iteration=True
)
ds_train = ds_train.map(
    lambda p, y: _load_and_preprocess(p, y), num_parallel_calls=AUTOTUNE
)
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=False)
ds_train = ds_train.prefetch(AUTOTUNE)




## === cell 3
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model

base = ResNet50(
    include_top=False,
    weights=None,
    input_tensor=None,
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling=None,
)

x = GlobalAveragePooling2D()(base.output)
out = Dense(len(label_class), activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=SGD(),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    steps_per_execution=10,
)

model.fit(ds_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)




## === cell 4
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_files = sample_sub["image"].tolist()
test_paths = [os.path.join(TEST_IMG_DIR, f) for f in test_files]

ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.with_options(ds_opts)
ds_test = ds_test.map(
    lambda p: _load_and_preprocess(p, None), num_parallel_calls=AUTOTUNE
)
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTOTUNE)

pred = model.predict(ds_test, verbose=1)

THRESH = 0.5
pred_ge = pred >= THRESH
any_pos = pred_ge.any(axis=1)
argmax_idx = pred.argmax(axis=1)

label_class_arr = np.array(label_class, dtype=object)

pred_labels = []
pred_labels_extend = pred_labels.append
for i in range(pred.shape[0]):
    if any_pos[i]:
        idxs = np.flatnonzero(pred_ge[i])
    else:
        idxs = np.array([argmax_idx[i]], dtype=np.int64)
    pred_labels_extend(" ".join(label_class_arr[idxs].tolist()))

sub = pd.DataFrame({"image": test_files, "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
