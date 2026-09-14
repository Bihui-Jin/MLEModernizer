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

# 5. Code solution

## === cell 0
import os
import sys

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import numpy as np
import pandas as pd
import tensorflow as tf

print("TF version:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_root = (
    "../input/model-effb7e6"  # keep original source; we'll auto-resolve if present
)

image_dims = (300, 300, 3)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])

label_to_idx = {c: i for i, c in enumerate(dataset_labels)}



## === cell 2
from tensorflow.keras.layers import TFSMLayer

from tensorflow.keras.applications.efficientnet import (
    preprocess_input as eff_preprocess,
)


def _find_saved_model_dir(root_dir: str) -> str:
    """
    The provided model path may not directly be a SavedModel directory.
    TFSMLayer expects a directory containing saved_model.pb (or .pbtxt).
    We'll search root_dir recursively and pick the most likely candidate.
    """
    root_dir = os.path.abspath(root_dir)
    candidates = []

    for fn in ("saved_model.pb", "saved_model.pbtxt"):
        if os.path.exists(os.path.join(root_dir, fn)):
            return root_dir

    if not os.path.exists(root_dir):
        raise FileNotFoundError(f"Model root path does not exist: {root_dir}")

    for dirpath, dirnames, filenames in os.walk(root_dir):
        if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
            candidates.append(dirpath)

    if not candidates:
        raise FileNotFoundError(
            f"No TensorFlow SavedModel found under: {root_dir}. "
            f"Expected a directory containing saved_model.pb"
        )

    def score(p):
        rel = os.path.relpath(p, root_dir)
        depth = rel.count(os.sep)
        looks_epoch = 0 if ("epoch" in rel.lower() or "export" in rel.lower()) else 1
        return (depth, looks_epoch, rel)

    candidates = sorted(set(candidates), key=score)
    return candidates[0]


def _extract_tensor(out):
    if isinstance(out, dict):
        for k in (
            "outputs",
            "output_0",
            "predictions",
            "logits",
            "dense",
            "activation",
        ):
            if k in out:
                return out[k]
        first_key = sorted(out.keys())[0]
        return out[first_key]
    if isinstance(out, (list, tuple)):
        return out[0]
    return out


def _load_infer_layer(saved_model_dir: str) -> TFSMLayer:
    try:
        return TFSMLayer(saved_model_dir, call_endpoint="serving_default")
    except Exception as e:
        print("Failed with call_endpoint='serving_default' due to:", repr(e))
        return TFSMLayer(saved_model_dir, call_endpoint="serve")


def _read_and_preprocess_image(path: str):
    input_img = tf.io.read_file(path)
    image = tf.io.decode_jpeg(input_img, channels=3)
    image = tf.image.resize(image, [image_dims[0], image_dims[1]])
    image = tf.cast(image, tf.float32)
    image = eff_preprocess(image)
    return tf.expand_dims(image, axis=0)




## === cell 3
SEED = 1337
tf.keras.utils.set_random_seed(SEED)


def _build_fallback_model(num_classes: int):
    base = tf.keras.applications.EfficientNetB3(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims,
        pooling="avg",
    )
    inputs = tf.keras.Input(shape=image_dims)
    x = base(inputs, training=False)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
    model = tf.keras.Model(inputs, outputs)
    return model


def _make_label_vector(label_str: str) -> np.ndarray:
    y = np.zeros(len(dataset_labels), dtype=np.float32)
    for lab in str(label_str).split(" "):
        lab = lab.strip()
        if lab and lab in label_to_idx:
            y[label_to_idx[lab]] = 1.0
    return y


def _load_image_for_ds(path: tf.Tensor):
    bytes_ = tf.io.read_file(path)
    img = tf.io.decode_jpeg(bytes_, channels=3)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = tf.cast(img, tf.float32)
    img = eff_preprocess(img)
    return img


def _train_fallback_model():
    train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
    img_paths = (
        data_set["image"]
        .astype(str)
        .apply(lambda x: os.path.join(train_img_dir, x))
        .values
    )
    y = np.stack(
        [_make_label_vector(s) for s in data_set["labels"].astype(str).values], axis=0
    )

    ds = tf.data.Dataset.from_tensor_slices((img_paths, y))
    ds = ds.shuffle(
        buffer_size=min(len(img_paths), 8192), seed=SEED, reshuffle_each_iteration=True
    )
    ds = ds.map(
        lambda p, yy: (_load_image_for_ds(p), yy), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)

    model = _build_fallback_model(len(dataset_labels))
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss="binary_crossentropy",
    )
    model.fit(ds, epochs=1, verbose=2)
    return model




## === cell 4
def _micro_f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= thr).astype(np.int32)
    tp = int(np.logical_and(y_true == 1, y_pred == 1).sum())
    fp = int(np.logical_and(y_true == 0, y_pred == 1).sum())
    fn = int(np.logical_and(y_true == 1, y_pred == 0).sum())
    denom = 2 * tp + fp + fn
    return (2 * tp / denom) if denom > 0 else 0.0


def _mean_f1_from_probs(y_true: np.ndarray, y_prob: np.ndarray, thr: float) -> float:
    y_pred = (y_prob >= thr).astype(np.int32)
    eps = 1e-12
    tp = (y_true * y_pred).sum(axis=1).astype(np.float32)
    fp = ((1 - y_true) * y_pred).sum(axis=1).astype(np.float32)
    fn = (y_true * (1 - y_pred)).sum(axis=1).astype(np.float32)
    f1 = (2 * tp) / (2 * tp + fp + fn + eps)
    return float(np.mean(f1))


def _predict_probs_for_paths(infer_fn, img_paths, batch_size=16) -> np.ndarray:
    probs = []
    n = len(img_paths)
    for i in range(0, n, batch_size):
        batch_paths = img_paths[i : i + batch_size]
        batch_imgs = tf.concat(
            [_read_and_preprocess_image(p) for p in batch_paths], axis=0
        )
        raw_out = infer_fn(batch_imgs)
        pred = (
            _extract_tensor(raw_out) if not isinstance(raw_out, tf.Tensor) else raw_out
        )
        pred = tf.convert_to_tensor(pred)
        pred_np = pred.numpy()
        if pred_np.min() < -1e-3 or pred_np.max() > 1.0 + 1e-3:
            pred_np = tf.sigmoid(pred).numpy()
        probs.append(pred_np.astype(np.float32))
    return np.concatenate(probs, axis=0)


def _calibrate_threshold(infer_fn, max_val=512) -> float:
    train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
    n = len(data_set)
    rng = np.random.default_rng(SEED)
    idx = np.arange(n)
    rng.shuffle(idx)

    val_n = int(min(max_val, max(256, 0.05 * n)))
    val_idx = idx[:val_n]

    val_paths = [os.path.join(train_img_dir, data_set.loc[i, "image"]) for i in val_idx]
    y_true = np.stack(
        [_make_label_vector(data_set.loc[i, "labels"]) for i in val_idx], axis=0
    ).astype(np.int32)

    y_prob = _predict_probs_for_paths(infer_fn, val_paths, batch_size=16)

    grid = [0.40, 0.45, 0.50, 0.55, 0.60, 0.65, 0.70]
    best_thr = 0.55
    best_f1 = -1.0
    for thr in grid:
        f1 = _mean_f1_from_probs(y_true, y_prob, thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = thr

    micro = _micro_f1_from_probs(y_true, y_prob, best_thr)
    print(
        f"Calibrated threshold on {val_n} val samples: best_thr={best_thr}, meanF1={best_f1:.5f}, microF1={micro:.5f}"
    )
    return float(best_thr)




## === cell 5
if __name__ == "__main__":
    use_savedmodel = True
    infer_layer = None
    fallback_model = None

    try:
        resolved_model_dir = _find_saved_model_dir(model_root)
        print("Resolved SavedModel dir:", resolved_model_dir)
        infer_layer = _load_infer_layer(resolved_model_dir)
    except Exception as e:
        use_savedmodel = False
        print(
            "Could not load external SavedModel; switching to fallback training. Reason:",
            repr(e),
        )
        fallback_model = _train_fallback_model()

    def _infer_fn(x):
        if use_savedmodel:
            return infer_layer(x)
        return fallback_model(x, training=False)

    threshold = _calibrate_threshold(_infer_fn, max_val=512)

    sub_template = pd.read_csv(sample_sub_path)
    images_path_list = sub_template["image"].astype(str).tolist()

    values = []
    missing_files = 0

    for name in images_path_list:
        img_path = os.path.join(test_dir, name)
        if not os.path.exists(img_path):
            missing_files += 1
            values.append([name, "healthy"])
            continue

        images = _read_and_preprocess_image(img_path)

        raw_out = _infer_fn(images)
        pred = _extract_tensor(raw_out)
        pred = tf.convert_to_tensor(pred)

        pred_np = pred.numpy()[0]
        if pred_np.min() < -1e-3 or pred_np.max() > 1.0 + 1e-3:
            pred_np = tf.sigmoid(pred).numpy()[0]

        index_values = [i for i, v in enumerate(pred_np) if v >= threshold]
        if len(index_values) == 0:
            index_values = [int(pred_np.argmax())]

        classes_img = " ".join([str(dataset_labels[i]) for i in index_values]).strip()
        if classes_img == "":
            classes_img = "healthy"

        values.append([name, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    csv_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(csv_path, index=False)

    print("Wrote:", csv_path, "rows:", len(csv_pd), "missing_files:", missing_files)
    print(csv_pd.head())
    print(
        "Unique label strings (sample):",
        csv_pd["labels"].value_counts().head(10).to_dict(),
    )
