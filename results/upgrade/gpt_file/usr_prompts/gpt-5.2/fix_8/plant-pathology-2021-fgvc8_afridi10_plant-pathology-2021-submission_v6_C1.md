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

0.3943859649122804

# 6. Current score

0.73825

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.31188) has done: 'I fix the environment-breaking TensorFlow import error by switching to the Kaggle-stable `tf.keras` stack and forcing the pure-Python protobuf implementation before importing TensorFlow. Since the referenced pretrained `.h5` file doesn’t exist in your provided `/kaggle/input/...` paths, I keep the same “MobileNet-style classifier” core idea but build a MobileNetV2-based multi-label model and train it quickly on `train.csv` + `train_images`, then run inference on `test_images`. I also fix label post-processing to output a space-delimited list, ensure at least one label is predicted (fallback to best class), and match the exact `sample_submission.csv` image order so the submission is valid. Finally, the script always write `/kaggle/working/submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.73825) has done: 'The timeout is dominated by slow input pipeline and per-step overhead (JPEG decode/resize done without caching, small batch, and no dataset options), plus extra Python work in threshold search and label joining. I keep the same model, loss, epochs, and evaluation semantics, but optimize the `tf.data` pipeline with caching (in-memory for 128×128 float images), deterministic options, and fused map/batch/prefetch behavior to reduce CPU bottlenecks. I also vectorize the threshold sweep (exactly equivalent results) and speed up submission label construction without changing the thresholding logic. These changes reduce wall-clock time substantially while preserving accuracy and core logic.'
- What this solution (achieved 0.73825) has done: 'The crash happens before any training because TensorFlow’s protobuf stack is incompatible with the default C++ protobuf in this Kaggle image, producing `MessageFactory.GetPrototype` errors. I fix this deterministically by forcing the pure-Python protobuf implementation *before* importing TensorFlow, which is the standard workaround in Kaggle notebooks. No changes be made to the model, training loop, thresholding, or submission formatting (so score behavior should remain essentially unchanged, aside from negligible nondeterminism). The rest of the pipeline run end-to-end and write `/kaggle/working/submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.73825) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime and (crucially) importing `google.protobuf` first so TensorFlow can’t initialize against the incompatible C++ backend. To keep your solution’s core logic identical, I won’t touch the model, training loop, threshold search, or submission formatting. Because your current score (0.73825) is already far above the target (0.3944) and higher-is-better, I not make any performance-improving changes; the goal here is correctness/stability and producing a valid `submission.csv`. The rest of the pipeline remains the same and write `/kaggle/working/submission.csv`.'
- What this solution (achieved 0.73825) has done: 'I fix the TensorFlow/protobuf crash that happens before training by pinning protobuf to the pure-Python implementation and additionally forcing the “python” protocol buffer runtime via `google.protobuf.internal.api_implementation` before importing TensorFlow. This is a stability-only fix and does not change the model, training loop, threshold search, or submission formatting, so it should keep performance in the same ballpark (your current score is already above the target, so we avoid score-improving changes). I also add a clear import order and a safe fallback if determinism enabling isn’t supported. The rest of the pipeline remains identical and write `/kaggle/working/submission.csv` with the required `image,labels` columns.'
- What this solution (achieved 0.73825) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by reliably forcing the pure-Python protobuf backend *before* anything from protobuf/TensorFlow initializes, and by setting the additional env var that disables the C++ fast implementation. This is a stability-only change that preserves your model/training/inference logic and should keep the score in the same ballpark (we won’t try to improve it since you’re already well above the target). I also keep the same paths and ensure the script always reaches the CSV write step. No changes are made to architecture, epochs, thresholding, or submission formatting.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CPP"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd

import google.protobuf  # noqa: F401

import tensorflow as tf

print("TensorFlow:", tf.__version__)

tf.keras.utils.set_random_seed(42)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE = "/kaggle/input/plant-pathology-2021-fgvc8"
TRAIN_CSV = f"{BASE}/train.csv"
SAMPLE_SUB = f"{BASE}/sample_submission.csv"
TRAIN_DIR = f"{BASE}/train_images"
TEST_DIR = f"{BASE}/test_images"

assert os.path.exists(TRAIN_CSV), f"Missing {TRAIN_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TRAIN_DIR), f"Missing {TRAIN_DIR}"
assert os.path.exists(TEST_DIR), f"Missing {TEST_DIR}"

train_df = pd.read_csv(TRAIN_CSV)
sub_df = pd.read_csv(SAMPLE_SUB)

train_df.head(), sub_df.head(), train_df.shape, sub_df.shape



## === cell 2
label_names = [
    "complex",
    "frog_eye_leaf_spot",
    "frog_eye_leaf_spot complex",
    "healthy",
    "powdery_mildew",
    "powdery_mildew complex",
    "rust",
    "rust complex",
    "rust frog_eye_leaf_spot",
    "scab",
    "scab frog_eye_leaf_spot",
    "scab frog_eye_leaf_spot complex",
]
label_to_idx = {l: i for i, l in enumerate(label_names)}
num_classes = len(label_names)


def encode_labels(label_str: str) -> np.ndarray:
    y = np.zeros(num_classes, dtype=np.float32)
    s = str(label_str)
    for i, name in enumerate(label_names):
        if name in s:
            y[i] = 1.0
    return y


y_all = np.stack([encode_labels(s) for s in train_df["labels"].values])
assert y_all.shape == (len(train_df), num_classes)

empty = (y_all.sum(axis=1) == 0).sum()
print("Rows with 0 parsed labels:", empty)



## === cell 3
IMSIZE = 128
BATCH = 32


def read_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(img, (IMSIZE, IMSIZE), method="bilinear")
    img = tf.cast(img, tf.float32) / 255.0
    return img


idx = np.arange(len(train_df))
rng = np.random.default_rng(42)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(len(idx) * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_paths = (TRAIN_DIR + "/" + train_df.iloc[trn_idx]["image"].values).tolist()
val_paths = (TRAIN_DIR + "/" + train_df.iloc[val_idx]["image"].values).tolist()
trn_y = y_all[trn_idx]
val_y = y_all[val_idx]


def make_ds(paths, labels=None, training=False, cache=True):
    options = tf.data.Options()
    options.experimental_deterministic = True
    options.experimental_optimization.apply_default_optimizations = True

    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.with_options(options)

    ds = ds.map(read_image, num_parallel_calls=tf.data.AUTOTUNE, deterministic=True)
    if cache:
        ds = ds.cache()

    if labels is not None:
        ds_y = tf.data.Dataset.from_tensor_slices(labels).with_options(options)
        ds = tf.data.Dataset.zip((ds, ds_y))

    if training:
        ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)

    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


trn_ds = make_ds(trn_paths, trn_y, training=True, cache=True)
val_ds = make_ds(val_paths, val_y, training=False, cache=True)



## === cell 4
from tensorflow.keras import backend as K


def f1(y_true, y_pred):  # kept same semantics as original metric
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    recall = true_positives / (possible_positives + K.epsilon())
    f1_val = 2 * (precision * recall) / (precision + recall + K.epsilon())
    return f1_val


base = tf.keras.applications.MobileNetV2(
    input_shape=(IMSIZE, IMSIZE, 3),
    include_top=False,
    weights="imagenet",
)
base.trainable = False

inputs = tf.keras.Input(shape=(IMSIZE, IMSIZE, 3))
x = inputs
x = tf.keras.applications.mobilenet_v2.preprocess_input(x * 255.0)
x = base(x, training=False)
x = tf.keras.layers.GlobalAveragePooling2D()(x)
x = tf.keras.layers.Dropout(0.2)(x)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)

model = tf.keras.Model(inputs, outputs)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    metrics=[f1],
)
model.summary()



## === cell 5
EPOCHS = 3
history = model.fit(
    trn_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 6
val_pred = model.predict(val_ds, verbose=0)
val_true = val_y


def mean_f1_over_thresholds(y_true, y_prob, thresholds):
    y_true_b = (y_true > 0.5)[None, :, :]  # (1,N,C)
    thr = thresholds[:, None, None]  # (T,1,1)
    y_hat = y_prob[None, :, :] >= thr  # (T,N,C) bool

    empty = y_hat.sum(axis=2) == 0  # (T,N)
    if empty.any():
        best = y_prob.argmax(axis=1)  # (N,)
        t_idx, n_idx = np.where(empty)
        y_hat[t_idx, n_idx, best[n_idx]] = True

    tp = (y_hat & y_true_b).sum(axis=2).astype(np.float32)  # (T,N)
    fp = (y_hat & (~y_true_b)).sum(axis=2).astype(np.float32)
    fn = ((~y_hat) & y_true_b).sum(axis=2).astype(np.float32)
    f1s = (2.0 * tp) / (2.0 * tp + fp + fn + 1e-9)
    return f1s.mean(axis=1)  # (T,)


thresholds = np.linspace(0.1, 0.9, 17, dtype=np.float32)
scores = mean_f1_over_thresholds(
    val_true.astype(np.float32), val_pred.astype(np.float32), thresholds
)
best_i = int(np.argmax(scores))
best_thr = float(thresholds[best_i])
print("Best threshold:", best_thr, "val mean F1:", float(scores[best_i]))



## === cell 7
test_images = sub_df["image"].astype(str).values.tolist()
test_paths = [f"{TEST_DIR}/{n}" for n in test_images]

test_ds = make_ds(test_paths, labels=None, training=False, cache=True)
test_prob = model.predict(test_ds, verbose=1)



## === cell 8
y_hat = (test_prob >= best_thr).astype(np.int32)

empty = y_hat.sum(axis=1) == 0
if empty.any():
    best = test_prob.argmax(axis=1)
    y_hat[empty, best[empty]] = 1

label_names_arr = np.array(label_names, dtype=object)
rows = [
    label_names_arr[np.flatnonzero(y_hat[i])].tolist() for i in range(y_hat.shape[0])
]
pred_labels = [" ".join(r) for r in rows]

submission = pd.DataFrame({"image": test_images, "labels": pred_labels})
submission.head()



## === cell 9
out_path = "/kaggle/working/submission.csv"
submission.to_csv(out_path, index=False)
print("Wrote:", out_path, "shape:", submission.shape)



## === cell 10
with open("/kaggle/working/submission.csv", "r") as f:
    for _ in range(10):
        print(f.readline().rstrip("\n"))
