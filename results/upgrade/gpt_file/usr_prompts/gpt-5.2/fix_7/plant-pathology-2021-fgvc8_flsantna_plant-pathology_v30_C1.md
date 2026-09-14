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

0.7200738688827324

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'I fix the two blockers preventing an end-to-end run: (1) the protobuf-related crash triggered by importing standalone `keras` (use `tf.keras` everywhere to match TensorFlow 2.18), and (2) the test image path pointing at a nested `test_images/test_images` directory causing `ReadFile` to receive a directory. I make path resolution robust by detecting the actual directory that contains `.jpg` files, and I ensure the submission is aligned to `sample_submission.csv` ordering and has the required `image,labels` columns. These changes are execution/stability fixes and should not degrade score; they also allow the provided model (if present) to be loaded reliably for better-than-all-healthy predictions.'
- What this solution (achieved 0.24507) has done: 'I fix the protobuf-related crash by avoiding any `google.protobuf` import/usage (it can trigger the `MessageFactory.GetPrototype` issue in this environment) and by forcing `tf.keras`/TF-native loading only. Then I make prediction post-processing metric-consistent by outputting space-delimited multi-label strings from sigmoid scores, and calibrate the decision threshold using a small validation split from `train.csv` to move the mean F1-score toward your target (this is a minimal change that doesn’t alter the model architecture/training). I also harden the inference output handling (dict/shape) and ensure the submission is exactly aligned to `sample_submission.csv` with the required `image,labels` columns and a `.csv` suffix.'
- What this solution (achieved 0.24507) has done: 'I fix the runtime crash happening at import time by forcing TensorFlow to use the pure-Python protobuf implementation before importing `tensorflow` (this avoids the `MessageFactory.GetPrototype` AttributeError in this environment). I keep your model loading, preprocessing, threshold calibration, and submission-writing logic the same, only adding small robustness guards (e.g., consistent TF seed, safer image decoding dtype/shape) that are execution-stability focused. This should let the notebook run end-to-end and generate a valid `submission.csv`, and it should also allow the model to actually load and be used (which is the main driver to move the score upward toward the target). Paths and submission format remain unchanged and aligned to `sample_submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import pandas as pd
import numpy as np
import tensorflow as tf

keras = tf.keras

np.random.seed(1337)
tf.random.set_seed(1337)

print("TensorFlow:", tf.__version__)

output_dir = "./"

test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-onlye1/epoch-1"

image_dims = (300, 300, 3)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def _is_savedmodel_dir(path: str) -> bool:
    return (
        os.path.isdir(path)
        and (
            os.path.exists(os.path.join(path, "saved_model.pb"))
            or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
        )
        and os.path.isdir(os.path.join(path, "variables"))
    )


def _find_savedmodel_dir(root: str) -> str:
    root = os.path.abspath(root)
    if _is_savedmodel_dir(root):
        return root

    if not os.path.exists(root):
        raise FileNotFoundError(f"model_dir does not exist: {root}")

    for dirpath, dirnames, filenames in os.walk(root):
        if ("saved_model.pb" in filenames) or ("saved_model.pbtxt" in filenames):
            if os.path.isdir(os.path.join(dirpath, "variables")):
                return dirpath

    raise FileNotFoundError(
        f"Could not find a TensorFlow SavedModel under: {root}. "
        f"Expected a directory containing saved_model.pb (or pbtxt) and variables/."
    )


def _find_model_file_by_ext(root: str, exts=(".keras", ".h5", ".hdf5")) -> str:
    root = os.path.abspath(root)
    if os.path.isfile(root) and root.lower().endswith(exts):
        return root
    if not os.path.exists(root):
        raise FileNotFoundError(f"model_dir does not exist: {root}")
    for dirpath, dirnames, filenames in os.walk(root):
        for fn in filenames:
            lfn = fn.lower()
            if lfn.endswith(exts):
                return os.path.join(dirpath, fn)
    raise FileNotFoundError(
        f"Could not find any model file with extensions {exts} under: {root}"
    )


def load_inference_model(model_root: str):
    """
    Load a model for inference without importing standalone keras.
    Preference order:
    1) SavedModel via TFSMLayer (TF 2.18 compatible)
    2) Keras .keras
    3) H5
    """
    errors = []

    try:
        sm_dir = _find_savedmodel_dir(model_root)
        layer = keras.layers.TFSMLayer(sm_dir, call_endpoint="serving_default")
        inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
        out = layer(inp)
        if isinstance(out, dict):
            key = sorted(list(out.keys()))[0]
            out = out[key]
        m = keras.Model(inputs=inp, outputs=out)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model as SavedModel from:", sm_dir)
        return m
    except Exception as e:
        errors.append(("SavedModel/TFSMLayer", repr(e)))

    try:
        keras_path = _find_model_file_by_ext(model_root, exts=(".keras",))
        m = keras.models.load_model(keras_path, compile=False)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model from .keras file:", keras_path)
        return m
    except Exception as e:
        errors.append((".keras load_model", repr(e)))

    try:
        h5_path = _find_model_file_by_ext(model_root, exts=(".h5", ".hdf5"))
        m = keras.models.load_model(h5_path, compile=False)
        _ = m(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Loaded model from H5 file:", h5_path)
        return m
    except Exception as e:
        errors.append((".h5 load_model", repr(e)))

    msg = "Failed to load model from provided model_dir. Attempts:\n" + "\n".join(
        [f"- {k}: {v}" for k, v in errors]
    )
    raise FileNotFoundError(msg)


model = None
try:
    model = load_inference_model(model_dir)
    model.trainable = False
    print("Model ready for inference.")
except Exception as e:
    print(
        "WARNING: Could not load model. Will fall back to predicting 'healthy' for all test images."
    )
    print("Reason:", repr(e))
    model = None




## === cell 2
def _resolve_image_dir(path: str) -> str:
    path = os.path.abspath(path)
    if not os.path.exists(path):
        raise FileNotFoundError(f"test_dir does not exist: {path}")

    def has_image_files(p: str) -> bool:
        if not os.path.isdir(p):
            return False
        for fn in os.listdir(p):
            lfn = fn.lower()
            if lfn.endswith(".jpg") or lfn.endswith(".jpeg") or lfn.endswith(".png"):
                return True
        return False

    if os.path.isdir(path) and has_image_files(path):
        return path

    if os.path.isdir(path):
        for sub in sorted(os.listdir(path)):
            subp = os.path.join(path, sub)
            if os.path.isdir(subp) and has_image_files(subp):
                return subp

    for dirpath, dirnames, filenames in os.walk(path):
        for fn in filenames:
            lfn = fn.lower()
            if lfn.endswith(".jpg") or lfn.endswith(".jpeg") or lfn.endswith(".png"):
                return dirpath

    raise FileNotFoundError(f"Could not find any image files under: {path}")


test_dir_resolved = _resolve_image_dir(test_dir)
print("Resolved test image dir:", test_dir_resolved)

images_path_list = sorted(
    [
        fn
        for fn in os.listdir(test_dir_resolved)
        if fn.lower().endswith(".jpg")
        or fn.lower().endswith(".jpeg")
        or fn.lower().endswith(".png")
    ]
)
print("Found test images:", len(images_path_list))


def load_and_preprocess_image(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_image(img_bytes, channels=3, expand_animations=False)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.ensure_shape(img, image_dims)
    img = img * 255.0
    return img


def _to_numpy_logits(pred):
    if isinstance(pred, dict):
        key = sorted(list(pred.keys()))[0]
        pred = pred[key]
    pred = tf.convert_to_tensor(pred)
    pred_np = pred.numpy()
    if pred_np.ndim == 1:
        pred_np = pred_np.reshape(1, -1)
    return pred_np


def predict_single(image_tensor, threshold=0.7):
    if model is None:
        return "healthy"

    batch = tf.expand_dims(image_tensor, axis=0)
    pred = model(batch, training=False)
    pred_np = _to_numpy_logits(pred)

    scores = pred_np[0]
    scores = 1.0 / (1.0 + np.exp(-scores))

    idx = np.where(scores > threshold)[0].tolist()
    if len(idx) == 0:
        idx = [int(np.argmax(scores))]

    max_i = len(dataset_labels) - 1
    idx = [i for i in idx if 0 <= i <= max_i]
    if len(idx) == 0:
        return "healthy"

    return " ".join([dataset_labels[i] for i in idx])




## === cell 3
def mean_f1_multilabel(
    y_true_bin: np.ndarray, y_pred_bin: np.ndarray, eps: float = 1e-9
) -> float:
    tp = (y_true_bin * y_pred_bin).sum(axis=1)
    fp = ((1 - y_true_bin) * y_pred_bin).sum(axis=1)
    fn = (y_true_bin * (1 - y_pred_bin)).sum(axis=1)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def calibrate_threshold_on_train(max_samples: int = 512, seed: int = 1337):
    if model is None:
        return 0.7

    train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
    train_img_dir = os.path.abspath(train_img_dir)
    if not os.path.isdir(train_img_dir):
        print("WARNING: train_images directory not found; using default threshold=0.7")
        return 0.7

    df = data_set[["image", "labels"]].copy()
    rs = np.random.RandomState(seed)
    n = min(max_samples, len(df))
    idx = rs.choice(len(df), size=n, replace=False)
    df = df.iloc[idx].reset_index(drop=True)

    y_true = (
        df["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=dataset_labels, fill_value=0)
        .values.astype(np.int32)
    )

    probs = np.zeros((n, len(dataset_labels)), dtype=np.float32)
    for i, fname in enumerate(df["image"].tolist()):
        p = os.path.join(train_img_dir, fname)
        img = load_and_preprocess_image(p)
        batch = tf.expand_dims(img, axis=0)
        pred = model(batch, training=False)
        pred_np = _to_numpy_logits(pred)
        s = pred_np[0]
        s = 1.0 / (1.0 + np.exp(-s))
        probs[i] = s.astype(np.float32)
        if (i + 1) % 128 == 0 or (i + 1) == n:
            print(f"Calibration processed {i+1}/{n}")

    candidate_thresholds = np.round(np.arange(0.2, 0.81, 0.05), 2)
    best_t = 0.7
    best_f1 = -1.0
    for t in candidate_thresholds:
        y_pred = (probs > t).astype(np.int32)
        empty = y_pred.sum(axis=1) == 0
        if np.any(empty):
            top1 = np.argmax(probs[empty], axis=1)
            y_pred[empty, :] = 0
            y_pred[np.where(empty)[0], top1] = 1
        f1 = mean_f1_multilabel(y_true, y_pred)
        if f1 > best_f1:
            best_f1 = f1
            best_t = float(t)

    print(
        f"Chosen threshold={best_t} (val mean F1={best_f1:.5f}) from grid {candidate_thresholds.tolist()}"
    )
    return best_t


threshold = calibrate_threshold_on_train(max_samples=512, seed=1337)



## === cell 4
values = []

for i, fname in enumerate(images_path_list):
    img_path = os.path.join(test_dir_resolved, fname)
    img = load_and_preprocess_image(img_path)
    labels_str = predict_single(img, threshold=threshold)

    values.append([fname, labels_str])

    if (i + 1) % 500 == 0 or (i + 1) == len(images_path_list):
        print(f"Processed {i+1}/{len(images_path_list)}")

pred_df = pd.DataFrame(values, columns=["image", "labels"])

sample_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
sample_df = pd.read_csv(sample_path)

sub_df = sample_df[["image"]].merge(pred_df, on="image", how="left")
sub_df["labels"] = sub_df["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
sub_df.to_csv(csv_path, index=False)
print("Wrote submission to:", csv_path)
print(sub_df.head())
print("Submission rows:", len(sub_df))
print("Healthy count:", int(sub_df["labels"].eq("healthy").sum()))
