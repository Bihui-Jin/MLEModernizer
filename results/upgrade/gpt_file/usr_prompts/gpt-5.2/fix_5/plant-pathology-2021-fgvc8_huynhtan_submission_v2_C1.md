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

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

0.5952908587257614

# 6. Current score

0.38173

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'Main runtime cost here is JPEG decode/resize done every epoch plus input pipeline overhead. To keep the same model and training loop while speeding up, I (1) make the `tf.data` pipeline cache decoded/resized images so subsequent epochs don’t redo image decoding, (2) enable deterministic pipeline behavior explicitly while still using parallel mapping, and (3) remove a small extra `map(lambda...)` indirection for the test set. These changes are provably equivalent in semantics (same images/labels, same augmentation: none), and they primarily reduce repeated CPU work so training completes within the 600s budget.'
- What this solution (achieved 0.21672) has done: 'I fix the immediate runtime crash caused by an incompatible protobuf/tensorflow import path by forcing the pure-Python protobuf implementation before importing TensorFlow (a common Kaggle TF2.18 + protobuf 6 issue). Then I keep your model and training loop identical, but adjust only the prediction-to-label decoding to better match the Mean F1 metric by emitting the top-1 label when nothing passes the threshold (instead of always “healthy”), which should improve score without changing the model. I also make the CSV header/columns exactly match the required submission format and keep all paths unchanged.'
- What this solution (achieved 0.38173) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible pure-Python protobuf runtime and downgrading protobuf to a TensorFlow-2.18-compatible version at runtime (this is the root cause of the `MessageFactory.GetPrototype` error). Then I keep your data pipeline, model, and training loop the same, but change only the prediction-to-label decoding to use per-class thresholds tuned on the validation split (mean-F1 aligned) instead of a single global 0.5 threshold, which should move the score upward toward your target. Finally, I ensure the submission file is written with the exact required columns and a `.csv` suffix.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import subprocess


def _ensure_protobuf_compatible():
    try:
        import google.protobuf  # noqa: F401
        from importlib.metadata import version

        v = version("protobuf")
    except Exception:
        v = None

    if v is None or v.startswith("6.") or v.startswith("5."):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        for m in list(sys.modules.keys()):
            if m.startswith("google.protobuf"):
                del sys.modules[m]


_ensure_protobuf_compatible()

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
BASE = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = f"{BASE}/train.csv"
SAMPLE_SUB = f"{BASE}/sample_submission.csv"
TRAIN_IMG_DIR = f"{BASE}/train_images"
TEST_IMG_DIR = f"{BASE}/test_images"

train_df = pd.read_csv(TRAIN_CSV)
sample_df = pd.read_csv(SAMPLE_SUB)

class_name = [
    "complex",
    "frog_eye_leaf_spot",
    "healthy",
    "powdery_mildew",
    "rust",
    "scab",
]
num_classes = len(class_name)
image_size = 224

label2idx = {c: i for i, c in enumerate(class_name)}
idx2label = {i: c for c, i in label2idx.items()}


def encode_labels(label_str: str) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    for tok in str(label_str).split():
        if tok in label2idx:
            y[label2idx[tok]] = 1.0
    return y


train_df["filepath"] = train_df["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x))
train_df["target"] = train_df["labels"].apply(encode_labels)

sample_df["filepath"] = sample_df["image"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x)
)

print("Train rows:", len(train_df), "Test rows:", len(sample_df))
print("Example labels:", train_df.loc[0, "labels"], train_df.loc[0, "target"])




## === cell 2
@tf.function
def parse_function(filename, label=None):
    image_string = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image_string, channels=3)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32) / 255.0
    if label is None:
        return image
    return image, label


val_frac = 0.1
perm = np.random.permutation(len(train_df))
val_size = int(len(train_df) * val_frac)
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

trn_paths = train_df.iloc[trn_idx]["filepath"].values
trn_y = np.stack(train_df.iloc[trn_idx]["target"].values)

val_paths = train_df.iloc[val_idx]["filepath"].values
val_y = np.stack(train_df.iloc[val_idx]["target"].values)

BATCH = 16

options = tf.data.Options()
options.experimental_deterministic = True

train_ds = tf.data.Dataset.from_tensor_slices((trn_paths, trn_y)).with_options(options)
train_ds = train_ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
train_ds = (
    train_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

val_ds = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(options)
val_ds = (
    val_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

test_paths = sample_df["filepath"].values
test_ds = tf.data.Dataset.from_tensor_slices(test_paths).with_options(options)
test_ds = (
    test_ds.map(parse_function, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH)
    .prefetch(tf.data.AUTOTUNE)
)

print("Split sizes:", len(trn_paths), len(val_paths), len(test_paths))



## === cell 3
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    input_shape=(image_size, image_size, 3),
    weights="imagenet",
    pooling="avg",
)
base.trainable = False

inp = tf.keras.Input(shape=(image_size, image_size, 3))
x = base(inp, training=False)
x = tf.keras.layers.Dropout(0.2)(x)
out = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = tf.keras.Model(inp, out)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

EPOCHS = 3
history = model.fit(train_ds, validation_data=val_ds, epochs=EPOCHS, verbose=1)



## === cell 4
val_pred = model.predict(val_ds, verbose=1)
print("Val pred shape:", val_pred.shape, "Val y shape:", val_y.shape)


def f1_score_micro(y_true, y_pred_bin, eps=1e-9):
    tp = np.sum((y_true == 1) & (y_pred_bin == 1))
    fp = np.sum((y_true == 0) & (y_pred_bin == 1))
    fn = np.sum((y_true == 1) & (y_pred_bin == 0))
    return (2 * tp) / (2 * tp + fp + fn + eps)


def f1_score_samples_mean(y_true, y_pred_bin, eps=1e-9):
    tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=1)
    fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=1)
    fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=1)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


grid = np.array([0.20, 0.25, 0.30, 0.35, 0.40, 0.45, 0.50, 0.55], dtype=np.float32)
best_thr = np.zeros(num_classes, dtype=np.float32)

for j in range(num_classes):
    best_t, best_f1 = 0.5, -1.0
    for t in grid:
        pred_bin = (val_pred[:, j] > t).astype(np.int32)
        tp = np.sum((val_y[:, j] == 1) & (pred_bin == 1))
        fp = np.sum((val_y[:, j] == 0) & (pred_bin == 1))
        fn = np.sum((val_y[:, j] == 1) & (pred_bin == 0))
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)
    best_thr[j] = best_t

val_pred_bin_joint = (val_pred > best_thr[None, :]).astype(np.int32)
empty = np.where(val_pred_bin_joint.sum(axis=1) == 0)[0]
if len(empty) > 0:
    val_pred_bin_joint[empty, np.argmax(val_pred[empty], axis=1)] = 1

print(
    "Chosen thresholds:",
    {class_name[i]: float(best_thr[i]) for i in range(num_classes)},
)
print(
    "Val mean-sample-F1 (proxy):",
    f1_score_samples_mean(val_y.astype(np.int32), val_pred_bin_joint.astype(np.int32)),
)
print(
    "Val micro-F1 (proxy):",
    f1_score_micro(val_y.astype(np.int32), val_pred_bin_joint.astype(np.int32)),
)



## === cell 5
prediction = model.predict(test_ds, verbose=1)
print("Pred shape:", prediction.shape)

labels = []
pred_bin = (prediction > best_thr[None, :]).astype(np.int32)
for i in range(pred_bin.shape[0]):
    row = pred_bin[i]
    if row.sum() == 0:
        row[np.argmax(prediction[i])] = 1
    picked = [class_name[j] for j in range(num_classes) if row[j] == 1]
    labels.append(" ".join(picked))

assert len(labels) == len(sample_df), (len(labels), len(sample_df))

submission = pd.DataFrame({"image": sample_df["image"].values, "labels": labels})
submission = submission[["image", "labels"]]
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote:", os.path.abspath("submission.csv"), "rows:", len(submission))
