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

0.3184672206832874

# 6. Current score

0.35903

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.35903) has done: 'The timeout is dominated by training a full ResNet50 from scratch for 20 epochs on ~15k images at batch 100; that compute load is far beyond 600s even with caching/JIT. To preserve the exact model and training loop semantics while cutting wall time, the main fix is to (1) enable GPU-appropriate input pipeline (remove disk cache bottleneck and use `tf.io.decode_jpeg(..., dct_method="INTEGER_FAST")`), (2) compile the input pipeline once and keep it fully deterministic, and (3) reduce overhead in `tf.data` by moving cache to RAM and setting dataset options that avoid extra bookkeeping. These changes keep architecture/loss/optimizer/epochs unchanged and only reduce I/O + per-step overhead.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import numpy as np
import pandas as pd
import tensorflow as tf

os.environ["PYTHONHASHSEED"] = "42"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ.pop("TF_XLA_FLAGS", None)  # do not force XLA

np.random.seed(42)
tf.random.set_seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        from tensorflow.keras import mixed_precision

        mixed_precision.set_global_policy("mixed_float16")
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
AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.optimizer.set_jit(True)  # XLA JIT where beneficial
except Exception:
    pass
try:
    tf.config.optimizer.set_experimental_options(
        {
            "layout_optimizer": True,
            "constant_folding": True,
            "shape_optimization": True,
            "remapping": True,
            "arithmetic_optimization": True,
        }
    )
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass

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

_pat_scab = r"(^| )scab( |$)"
_pat_fels = r"(^| )frog_eye_leaf_spot( |$)"
mask_scab = labels_series.str.contains(_pat_scab, regex=True)
mask_fels = labels_series.str.contains(_pat_fels, regex=True)
mask_combo = mask_scab & mask_fels
y_train[mask_combo.values, class_to_idx["scab frog_eye_leaf_spot"]] = 1.0

labels_clean = labels_series.where(
    ~mask_combo,
    labels_series.str.replace(_pat_scab, " ", regex=True).str.replace(
        _pat_fels, " ", regex=True
    ),
)

y_train[
    labels_clean.str.contains(_pat_scab, regex=True).values, class_to_idx["scab"]
] = 1.0
y_train[
    labels_clean.str.contains(r"(^| )healthy( |$)", regex=True).values,
    class_to_idx["healthy"],
] = 1.0
y_train[
    labels_clean.str.contains(_pat_fels, regex=True).values,
    class_to_idx["frog_eye_leaf_spot"],
] = 1.0
y_train[
    labels_clean.str.contains(r"(^| )cider_apple_rust( |$)", regex=True).values,
    class_to_idx["cider_apple_rust"],
] = 1.0
y_train[
    labels_clean.str.contains(r"(^| )complex( |$)", regex=True).values,
    class_to_idx["complex"],
] = 1.0
y_train[
    labels_clean.str.contains(r"(^| )powdery_mildew( |$)", regex=True).values,
    class_to_idx["powdery_mildew"],
] = 1.0




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BATCH_SIZE = 100
EPOCHS = 20

train_files = train_df["image"].astype(str).to_numpy()
train_paths = np.array([os.path.join(TRAIN_IMG_DIR, f) for f in train_files], dtype=str)


@tf.function
def _load_and_preprocess_from_path(path, y=None):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST"
    )  # faster decode
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    if y is None:
        return img
    return img, y


ds_opts = tf.data.Options()
ds_opts.experimental_optimization.map_parallelization = True
ds_opts.experimental_optimization.parallel_batch = True
ds_opts.experimental_optimization.map_and_batch_fusion = True
ds_opts.experimental_optimization.autotune_buffers = True
try:
    ds_opts.deterministic = True
except Exception:
    pass

ds_train = tf.data.Dataset.from_tensor_slices((train_paths, y_train))
ds_train = ds_train.with_options(ds_opts)
ds_train = ds_train.shuffle(
    buffer_size=min(n, 8192), seed=42, reshuffle_each_iteration=True
)

ds_train = ds_train.map(_load_and_preprocess_from_path, num_parallel_calls=AUTOTUNE)
ds_train = ds_train.cache()
ds_train = ds_train.batch(BATCH_SIZE, drop_remainder=True)
ds_train = ds_train.prefetch(AUTOTUNE)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3333239554.py in <cell line: 0>()
     28 ds_opts.experimental_optimization.parallel_batch = True
     29 ds_opts.experimental_optimization.map_and_batch_fusion = True
---> 30 ds_opts.experimental_optimization.autotune_buffers = True
     31 try:
     32     ds_opts.deterministic = True

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 2
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
out = Dense(len(label_class), activation="sigmoid", dtype="float32")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=SGD(),
    loss="binary_crossentropy",
    metrics=["binary_accuracy"],
    steps_per_execution=50,
)

model.fit(ds_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2169155084.py in <cell line: 0>()
     23 )
     24 
---> 25 model.fit(ds_train, batch_size=BATCH_SIZE, epochs=EPOCHS, verbose=1)
     26 
     27 

NameError: name 'ds_train' is not defined

## === cell 3
sample_sub = pd.read_csv(SAMPLE_SUB_CSV)
test_files = sample_sub["image"].astype(str).to_numpy()
test_paths = np.array([os.path.join(TEST_IMG_DIR, f) for f in test_files], dtype=str)

ds_test = tf.data.Dataset.from_tensor_slices(test_paths)
ds_test = ds_test.with_options(ds_opts)
ds_test = ds_test.map(
    lambda p: _load_and_preprocess_from_path(p, None), num_parallel_calls=AUTOTUNE
)
ds_test = ds_test.batch(BATCH_SIZE, drop_remainder=False)
ds_test = ds_test.prefetch(AUTOTUNE)

pred = model.predict(ds_test, verbose=1)

THRESH = 0.5
pred_ge = pred >= THRESH
any_pos = pred_ge.any(axis=1)
argmax_idx = pred.argmax(axis=1)

label_class_arr = np.array(label_class, dtype=object)

idx_rows, idx_cols = np.where(pred_ge)
counts = np.bincount(idx_rows, minlength=pred.shape[0])
starts = np.zeros_like(counts)
np.cumsum(counts[:-1], out=starts[1:])

picked = [None] * pred.shape[0]
pos_rows = np.flatnonzero(counts)
for r in pos_rows.tolist():
    s = int(starts[r])
    e = s + int(counts[r])
    picked[r] = idx_cols[s:e].tolist()

no_pos = np.flatnonzero(~any_pos)
for i in no_pos.tolist():
    picked[int(i)] = [int(argmax_idx[int(i)])]

pred_labels = [" ".join(label_class_arr[idxs].tolist()) for idxs in picked]

sub = pd.DataFrame({"image": test_files.tolist(), "labels": pred_labels})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("submission.csv exists:", os.path.exists("submission.csv"))
