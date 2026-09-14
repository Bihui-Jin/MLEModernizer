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

0.8214773776546646

# 6. Current score

0.30808

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.30585) has done: 'I fix the TensorFlow import crash by avoiding the incompatible `sklearn` dependency (it triggers a protobuf `MessageFactory` error in this environment) and replace it with a tiny local label parser that preserves the same semantics. Then I fix the `decode_image` pipeline: the current `decode_and_crop_jpeg` call uses Python `max()` on a Tensor, which breaks graph tracing; I switch to a standard `decode_jpeg` + `resize` path that is stable in `tf.data`. Finally, I make sure `test_dataset` is successfully created before inference, and that we always write a valid `submission.csv` with the required `image,labels` columns and space-delimited labels.'
- What this solution (achieved 0.30748) has done: 'We fix the crash happening at import time by avoiding the TensorFlow/protobuf incompatibility that gets triggered by the current TensorFlow import in this environment, while keeping the rest of your pipeline (InceptionV3 model + sigmoid outputs + thresholding + submission formatting) the same. Concretely, we make TensorFlow import lazy and more robust by setting safe environment flags before importing it, and we add a defensive fallback so the notebook can still run end-to-end and write `submission.csv` even if TF cannot be imported (it then output a simple baseline prediction to ensure a valid file). This unblocks execution and allows the intended model inference path to run when TF loads successfully. No changes are made to your model architecture or post-processing semantics when TF is available.'
- What this solution (achieved 0.30567) has done: 'I fix the TensorFlow/protobuf import crash that prevents the model path from running, by ensuring the environment variables are set before *any* TensorFlow-related imports and by avoiding optional imports that can trigger the protobuf `MessageFactory` issue. I also make the class-to-index mapping consistent with the actual training labels (currently it’s hard-coded in an order that can be wrong), which should significantly improve F1 without changing your model architecture or thresholding logic. Finally, I keep the same inference and submission formatting, but add small defensive checks so `submission.csv` is always produced correctly even if TF is unavailable.'
- What this solution (achieved 0.30517) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation before any TF import and by removing the `tf.function` decorations that can trigger graph/protobuf interactions in this environment. Then I keep your exact model/inference logic but make the label mapping robust (still derived from `train.csv`) and ensure the Dense head always matches the number of discovered classes (and remains 6 as expected). Finally, I keep the same thresholding/submission formatting, while guaranteeing `submission.csv` is always written with the required `image,labels` columns even if TF still fails to import.'
- What this solution (achieved 0.31091) has done: 'I fix the TensorFlow/protobuf crash that prevents the model from running by forcing a compatible protobuf runtime before any TF import and by explicitly avoiding the C++ protobuf backend (which triggers the `MessageFactory.GetPrototype` error here). Then I keep your exact InceptionV3+sigmoid+thresholding submission logic unchanged, but make the TF import path more robust so we don’t silently fall back to all-healthy predictions (which is what’s currently driving the very low score). Finally, I add a small safety check to ensure the correct test image paths are used and that `submission.csv` is always written with the required `image,labels` columns.'
- What this solution (achieved 0.30473) has done: 'We need to fix the TensorFlow/protobuf crash so the intended InceptionV3 inference path runs; right now TF import fails and you fall back to “healthy”, which explains the very low score. I make TF import robust by forcing the pure-Python protobuf backend *before any protobuf-related import* and by ensuring we don’t pre-import `google.protobuf` (which can lock in the incompatible backend). I also add a safe fallback to automatically switch to `tf.keras.applications.InceptionV3(weights="imagenet")` if TF loads but the custom weight file is missing, keeping your model architecture/head and thresholding semantics unchanged. Finally, I keep submission formatting identical but make paths robust and ensure `submission.csv` is always produced.'
- What this solution (achieved 0.31609) has done: 'I fix the TensorFlow/protobuf import crash that currently forces the code into the “all healthy” fallback (which is what’s driving the very low score). The safest minimal change is to force the pure-Python protobuf backend **and** proactively remove any already-imported `google.protobuf` modules before importing TensorFlow, which prevents the incompatible C++ backend from being locked in. I also make the TF import retry once after that cleanup, keeping your model architecture, weights-loading logic, inference, thresholding, and submission formatting unchanged. The script still always write a valid `submission.csv` even if TensorFlow truly cannot be loaded.'
- What this solution (achieved 0.30563) has done: 'I fix the TensorFlow/protobuf crash that currently happens before any training/inference by forcing the pure-Python protobuf implementation earlier and (critically) preventing `tensorflow` from importing the incompatible `protobuf` package version in this environment. To keep your core model/inference logic intact, I add a minimal, safe fallback: if TensorFlow still cannot be imported, the script run end-to-end using a lightweight image+logistic baseline built only from `numpy/pandas` and the raw JPEG bytes, and still write a valid `submission.csv`. This baseline is legitimate (no leakage) and should score substantially higher than the current all-healthy fallback, moving you toward the target. Submission formatting (`image,labels` with space-delimited labels) and paths remain unchanged.'
- What this solution (achieved 0.31197) has done: 'The crash happens before any training/inference because TensorFlow is importing an incompatible protobuf runtime (`MessageFactory.GetPrototype` missing). I fix this by avoiding the unsafe “fake protobuf module” injection and by importing TensorFlow only after forcing the pure-Python protobuf backend and cleaning any already-imported `google.protobuf` modules. This should allow the intended InceptionV3 inference path to run (instead of the weak fallback), which is the smallest legitimate change likely to move your score upward toward the target. I also make the image directory resolution robust (use the real `../input/...` path if present, otherwise fallback), while keeping the model architecture, thresholding, and submission formatting unchanged.'
- What this solution (achieved 0.30808) has done: 'I fix the TensorFlow import crash that currently stops execution by forcing the pure-Python protobuf backend *and* clearing any already-loaded protobuf modules before the first TensorFlow import (the current cleanup misses `google.protobuf.*` submodules and can leave the incompatible backend loaded). I also make the import path more defensive by setting an additional TF/protobuf env flag and performing a single clean retry, so the intended InceptionV3 inference path runs instead of falling back (which is what’s keeping your score very low). I keep your model architecture, thresholding, and submission formatting unchanged, and ensure the notebook always writes a valid `submission.csv`. No training-loop/architecture changes are introduced—this is primarily to unblock the higher-scoring TF path.'

# 9. Code solution

## === cell 0
import os
import sys
import gc
import warnings
import hashlib

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
np.random.seed(0)

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("TF_ENABLE_ONEDNN_OPTS", "0")  # stability; score-neutral

BATCH_SIZE = 64

IMAGE_PATH = "../input/plant-pathology-2021-fgvc8/train_images/"
TRAIN_PATH = "../input/plant-pathology-2021-fgvc8/train.csv"
SUB_PATH = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
TEST_IMAGE_DIR = "../input/plant-pathology-2021-fgvc8/test_images/"

if not os.path.isdir(IMAGE_PATH):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/train_images/"
    if os.path.isdir(alt):
        IMAGE_PATH = alt
if not os.path.isdir(TEST_IMAGE_DIR):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
    if os.path.isdir(alt):
        TEST_IMAGE_DIR = alt
if not os.path.exists(TRAIN_PATH):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
    if os.path.exists(alt):
        TRAIN_PATH = alt
if not os.path.exists(SUB_PATH):
    alt = "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv"
    if os.path.exists(alt):
        SUB_PATH = alt

sub = pd.read_csv(SUB_PATH)
test_data = sub.copy()
train_data = pd.read_csv(TRAIN_PATH)

train_data["labels"] = (
    train_data["labels"].astype(str).apply(lambda s: [x for x in s.split(" ") if x])
)
all_classes = sorted(
    {lab for labs in train_data["labels"].tolist() for lab in labs if lab}
)

print("Loaded train rows:", len(train_data), "test rows:", len(test_data))
print("Classes from train:", all_classes)
if len(all_classes) != 6:
    print("WARNING: Expected 6 classes, got:", len(all_classes), all_classes)

idx2label = {i: c for i, c in enumerate(all_classes)}
label_arr = np.array([idx2label[i] for i in range(len(all_classes))], dtype=object)
print("idx2label mapping:", idx2label)

TF_AVAILABLE = True
TF_IMPORT_ERROR = None


def _purge_protobuf_modules():
    for k in list(sys.modules.keys()):
        if k == "google" or k.startswith("google."):
            if k == "google" or k.startswith("google.protobuf"):
                del sys.modules[k]
        if k == "protobuf" or k.startswith("protobuf"):
            del sys.modules[k]


def _try_import_tf():
    _purge_protobuf_modules()
    import tensorflow as tf  # noqa: F401
    from tensorflow import keras  # noqa: F401
    import tensorflow.keras.layers as L  # noqa: F401

    return tf, keras, L


try:
    tf, keras, L = _try_import_tf()
except Exception as e1:
    try:
        _purge_protobuf_modules()
        tf, keras, L = _try_import_tf()
    except Exception as e2:
        TF_AVAILABLE = False
        TF_IMPORT_ERROR = f"First try: {repr(e1)} | Second try: {repr(e2)}"
        print(
            "WARNING: TensorFlow failed to import; will run a non-TF fallback and still write submission.csv."
        )
        print("TF import error:", TF_IMPORT_ERROR)

if TF_AVAILABLE:
    tf.random.set_seed(0)
    print("TF version:", tf.__version__)
    try:
        tf.config.threading.set_intra_op_parallelism_threads(0)
        tf.config.threading.set_inter_op_parallelism_threads(0)
    except Exception:
        pass
    try:
        tf.config.optimizer.set_jit(True)
    except Exception:
        pass
    AUTO = tf.data.experimental.AUTOTUNE

gc.collect()



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IMG_SIZE = (299, 299)

if TF_AVAILABLE:

    def decode_image(filename, label=None, image_size=IMG_SIZE):
        bits = tf.io.read_file(filename)
        image = tf.image.decode_jpeg(bits, channels=3)
        image = tf.image.convert_image_dtype(image, tf.float32)
        image = tf.image.resize(image, image_size, antialias=True)
        if label is None:
            return image
        return image, label

    test_paths = (
        TEST_IMAGE_DIR.rstrip("/") + "/" + test_data["image"].astype(str)
    ).to_numpy()

    options = tf.data.Options()
    options.experimental_deterministic = True

    test_dataset = (
        tf.data.Dataset.from_tensor_slices(test_paths)
        .with_options(options)
        .map(decode_image, num_parallel_calls=AUTO, deterministic=True)
        .batch(BATCH_SIZE, drop_remainder=False)
        .prefetch(AUTO)
    )
    print("Test dataset ready. Num test images:", len(test_data))



## === cell 2
if TF_AVAILABLE:
    inputs = tf.keras.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3))

    x = tf.keras.applications.InceptionV3(include_top=False, weights=None)(inputs)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)

    num_classes = len(all_classes)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.models.Model(inputs, outputs)

    weights_path = "../input/trail-1-dataset/Inceptionv3.h5"
    if os.path.exists(weights_path):
        model.load_weights(weights_path)
        print("Loaded custom weights:", weights_path)
    else:
        print("Custom weights not found. Falling back to ImageNet backbone weights.")
        backbone = tf.keras.applications.InceptionV3(
            include_top=False, weights="imagenet", input_tensor=inputs
        )
        x = tf.keras.layers.GlobalAveragePooling2D()(backbone.output)
        outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
        model = tf.keras.models.Model(inputs, outputs)

    model.summary()




## === cell 3
def _read_bytes(path):
    with open(path, "rb") as f:
        return f.read()


def _sigmoid(x):
    x = np.clip(x, -50, 50)
    return 1.0 / (1.0 + np.exp(-x))


def _fallback_predict_labels(train_df, test_df, train_dir, test_dir, classes):
    """
    Non-TF fallback that is fast and legitimate:
    - Features: simple hashed byte-statistics from JPEG bytes (no decoding dependency).
    - Model: one-vs-rest logistic regression trained with batch GD in numpy.
    """
    rng = np.random.RandomState(0)
    class2idx = {c: i for i, c in enumerate(classes)}
    n_classes = len(classes)

    y = np.zeros((len(train_df), n_classes), dtype=np.float32)
    for i, labs in enumerate(train_df["labels"].tolist()):
        for lab in labs:
            if lab in class2idx:
                y[i, class2idx[lab]] = 1.0

    def featurize_bytes(b):
        arr = np.frombuffer(b, dtype=np.uint8)
        if arr.size == 0:
            arr = np.zeros((1,), dtype=np.uint8)
        mean = arr.mean()
        std = arr.std()
        size_kb = len(b) / 1024.0
        hist, _ = np.histogram(arr, bins=16, range=(0, 256), density=False)
        p = hist.astype(np.float32)
        p = p / (p.sum() + 1e-6)
        ent = float(-(p * np.log(p + 1e-6)).sum())
        h = hashlib.md5(b[:4096]).digest()
        buckets = np.frombuffer(h, dtype=np.uint8).astype(np.float32) / 255.0
        feats = np.concatenate(
            [[mean / 255.0, std / 255.0, size_kb / 500.0, ent / 10.0], buckets], axis=0
        )
        return feats.astype(np.float32)

    X = np.zeros((len(train_df), 20), dtype=np.float32)
    for i, img in enumerate(train_df["image"].astype(str).tolist()):
        path = os.path.join(train_dir, img)
        try:
            b = _read_bytes(path)
        except Exception:
            b = b""
        X[i] = featurize_bytes(b)

    Xt = np.zeros((len(test_df), 20), dtype=np.float32)
    for i, img in enumerate(test_df["image"].astype(str).tolist()):
        path = os.path.join(test_dir, img)
        try:
            b = _read_bytes(path)
        except Exception:
            b = b""
        Xt[i] = featurize_bytes(b)

    mu = X.mean(axis=0, keepdims=True)
    sd = X.std(axis=0, keepdims=True) + 1e-6
    Xn = (X - mu) / sd
    Xtn = (Xt - mu) / sd

    W = rng.normal(scale=0.01, size=(Xn.shape[1], n_classes)).astype(np.float32)
    b0 = np.zeros((n_classes,), dtype=np.float32)

    lr = 0.15
    l2 = 1e-4
    batch = 512
    epochs = 25

    n = Xn.shape[0]
    for _ in range(epochs):
        idx = rng.permutation(n)
        Xs = Xn[idx]
        ys = y[idx]
        for start in range(0, n, batch):
            xb = Xs[start : start + batch]
            yb = ys[start : start + batch]
            logits = xb @ W + b0
            p = _sigmoid(logits)
            dlog = (p - yb) / max(1, xb.shape[0])
            gW = xb.T @ dlog + l2 * W
            gb = dlog.sum(axis=0)
            W -= lr * gW
            b0 -= lr * gb

    prob = _sigmoid(Xtn @ W + b0)

    threshold = 0.25
    mask = prob >= threshold
    has_any = mask.any(axis=1)
    argmax_idx = prob.argmax(axis=1)
    mask2 = mask.copy()
    mask2[~has_any, :] = False
    mask2[~has_any, argmax_idx[~has_any]] = True

    out_labels = [" ".join(np.array(classes, dtype=object)[m].tolist()) for m in mask2]
    return out_labels


if TF_AVAILABLE:
    preds = model.predict(test_dataset, verbose=0)

    threshold = 0.25
    preds_np = np.asarray(preds)

    if preds_np.ndim != 2 or preds_np.shape[1] != len(all_classes):
        raise ValueError(
            f"Unexpected prediction shape: {preds_np.shape} for num_classes={len(all_classes)}"
        )

    mask = preds_np >= threshold
    has_any = mask.any(axis=1)
    argmax_idx = preds_np.argmax(axis=1)

    mask2 = mask.copy()
    mask2[~has_any, :] = False
    mask2[~has_any, argmax_idx[~has_any]] = True

    testlabels = [" ".join(label_arr[m].tolist()) for m in mask2]

    out = sub.copy()
    out["labels"] = testlabels
else:
    testlabels = _fallback_predict_labels(
        train_df=train_data,
        test_df=test_data,
        train_dir=IMAGE_PATH,
        test_dir=TEST_IMAGE_DIR,
        classes=all_classes,
    )
    out = sub.copy()
    out["labels"] = testlabels

out["image"] = out["image"].astype(str)
out["labels"] = out["labels"].astype(str).str.strip().replace({"": "healthy"})

out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", out.shape)
print(out.head())
