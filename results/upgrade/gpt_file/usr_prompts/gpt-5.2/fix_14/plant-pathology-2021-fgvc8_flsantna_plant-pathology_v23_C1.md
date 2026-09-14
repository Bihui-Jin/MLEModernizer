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
import subprocess

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import pandas as pd
import numpy as np

import tensorflow as tf
from tensorflow import keras

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

tf.keras.utils.set_random_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"
train_images_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
model_dir = "../input/model-effb7-01/epoch-7/epoch-7"

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])




## === cell 2
if __name__ == "__main__":

    def resolve_existing_dir(candidates):
        for p in candidates:
            if p and os.path.isdir(p):
                return p
        return None

    def resolve_existing_file(candidates):
        for p in candidates:
            if p and os.path.isfile(p):
                return p
        return None

    resolved_test_dir = resolve_existing_dir(
        [
            test_dir,
            "/kaggle/input/plant-pathology-2021-fgvc8/test_images",
            "../input/test_images",
            "/kaggle/input/test_images",
            "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/test_images",
        ]
    )
    if resolved_test_dir is None:
        raise FileNotFoundError(
            "Could not locate test_images directory in expected locations."
        )
    test_dir = resolved_test_dir
    print("Using test_dir:", test_dir)

    resolved_train_images_dir = resolve_existing_dir(
        [
            train_images_dir,
            "/kaggle/input/plant-pathology-2021-fgvc8/train_images",
            "/kaggle/input/plant-pathology-2021-fgvc8/plant-pathology-2021-fgvc8/train_images",
        ]
    )
    if resolved_train_images_dir is None:
        raise FileNotFoundError(
            "Could not locate train_images directory in expected locations."
        )
    train_images_dir = resolved_train_images_dir
    print("Using train_images_dir:", train_images_dir)

    resolved_train_csv = resolve_existing_file(
        [
            train_csv_path,
            "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
            "/kaggle/input/train.csv",
            "/kaggle/data/input/plant-pathology-2021-fgvcvc8/train.csv",
            "/kaggle/data/input/plant-pathology-2021-fgvc8/train.csv",
        ]
    )
    if resolved_train_csv is None:
        raise FileNotFoundError("Could not locate train.csv in expected locations.")
    train_csv_path = resolved_train_csv
    print("Using train_csv_path:", train_csv_path)

    resolved_sample_sub = resolve_existing_file(
        [
            sample_sub_path,
            "/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv",
            "/kaggle/input/sample_submission.csv",
            "/kaggle/data/input/plant-pathology-2021-fgvc8/sample_submission.csv",
        ]
    )
    if resolved_sample_sub is None:
        raise FileNotFoundError(
            "Could not locate sample_submission.csv in expected locations."
        )
    sample_sub_path = resolved_sample_sub
    print("Using sample_sub_path:", sample_sub_path)

    data_set = pd.read_csv(train_csv_path)
    one_hot = data_set["labels"].str.get_dummies(sep=" ")
    dataset_labels = one_hot.columns.to_list()
    num_classes = len(dataset_labels)

    def find_model_artifact(path: str):
        """
        Returns a tuple (kind, resolved_path) where kind is one of:
        - "savedmodel": directory containing saved_model.pb/pbtxt
        - "keras": path to a .keras or .h5 file
        """
        path = os.path.expanduser(path)

        if os.path.isfile(path):
            lower = path.lower()
            if (
                lower.endswith(".keras")
                or lower.endswith(".h5")
                or lower.endswith(".hdf5")
            ):
                return "keras", path

        if os.path.isdir(path):
            if os.path.exists(os.path.join(path, "saved_model.pb")) or os.path.exists(
                os.path.join(path, "saved_model.pbtxt")
            ):
                return "savedmodel", path

            for root, _, files in os.walk(path):
                for fn in files:
                    lower = fn.lower()
                    if (
                        lower.endswith(".keras")
                        or lower.endswith(".h5")
                        or lower.endswith(".hdf5")
                    ):
                        return "keras", os.path.join(root, fn)
                if "saved_model.pb" in files or "saved_model.pbtxt" in files:
                    return "savedmodel", root

        raise FileNotFoundError(
            f"Could not find a TensorFlow SavedModel (saved_model.pb/pbtxt) or Keras model (.keras/.h5) under: {path}"
        )

    def try_resolve_model_dir(primary_path: str):
        if primary_path and os.path.exists(primary_path):
            return primary_path

        candidate_bases = ["../input", "/kaggle/input"]
        for base in candidate_bases:
            if not os.path.isdir(base):
                continue

            candidate_root = os.path.join(base, "model-effb7-01")
            if os.path.exists(candidate_root):
                return candidate_root

            try:
                for name in os.listdir(base):
                    p = os.path.join(base, name)
                    if os.path.isdir(p):
                        if os.path.exists(
                            os.path.join(p, "saved_model.pb")
                        ) or os.path.exists(os.path.join(p, "saved_model.pbtxt")):
                            return p
                        for root, _, files in os.walk(p):
                            if (
                                "saved_model.pb" in files
                                or "saved_model.pbtxt" in files
                            ):
                                return root
                            for fn in files:
                                l = fn.lower()
                                if (
                                    l.endswith(".keras")
                                    or l.endswith(".h5")
                                    or l.endswith(".hdf5")
                                ):
                                    return os.path.join(root, fn)
                            break
            except Exception:
                pass

        return primary_path

    resolved_model_root = try_resolve_model_dir(model_dir)

    model_loaded = False
    resolved_model_path = None
    kind = None

    try:
        kind, resolved_model_path = find_model_artifact(resolved_model_root)

        if kind == "savedmodel":
            loaded_sm = tf.saved_model.load(resolved_model_path)
            available_endpoints = list(getattr(loaded_sm, "signatures", {}).keys())
            call_endpoint = (
                "serving_default"
                if "serving_default" in available_endpoints
                else (
                    available_endpoints[0] if available_endpoints else "serving_default"
                )
            )
            predictor = keras.layers.TFSMLayer(
                resolved_model_path, call_endpoint=call_endpoint
            )
            print("Loaded SavedModel endpoint:", call_endpoint)
        else:
            model = keras.models.load_model(resolved_model_path, compile=False)

            def predictor(x, training=False):
                return model(x, training=training)

            print("Loaded Keras model:", resolved_model_path)

        model_loaded = True

    except FileNotFoundError as e:
        print("WARNING:", str(e))
        print(
            "Falling back to a minimal model and TRAINING it to produce a meaningful submission.csv."
        )

        inputs = keras.Input(shape=image_dims, name="image")
        x = keras.layers.Rescaling(1.0 / 255.0)(inputs)
        x = keras.layers.Conv2D(16, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        outputs = keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
        fallback_model = keras.Model(inputs=inputs, outputs=outputs)

        df = data_set.copy()
        df["path"] = df["image"].apply(lambda x: os.path.join(train_images_dir, x))
        y = (
            df["labels"]
            .str.get_dummies(sep=" ")
            .reindex(columns=dataset_labels, fill_value=0)
            .astype("float32")
            .values
        )
        x_paths = df["path"].values

        rng = np.random.RandomState(42)
        idx_all = np.arange(len(df))
        rng.shuffle(idx_all)
        train_n = int(0.90 * len(idx_all))
        train_idx = idx_all[:train_n]
        fit_val_idx = idx_all[train_n:]

        x_paths_train, y_train = x_paths[train_idx], y[train_idx]
        x_paths_fit_val, y_fit_val = x_paths[fit_val_idx], y[fit_val_idx]

        aug = keras.Sequential(
            [
                keras.layers.RandomFlip("horizontal"),
                keras.layers.RandomRotation(0.05),
                keras.layers.RandomZoom(0.1),
                keras.layers.RandomContrast(0.1),
            ],
            name="aug",
        )

        def _load_img_train(path, label, augment=False):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32)  # [0,255]
            if augment:
                img = aug(img, training=True)
            return img, label

        pos = np.clip(np.sum(y_train, axis=0), 1.0, None)
        neg = np.clip(y_train.shape[0] - pos, 1.0, None)
        pos_weight = (neg / pos).astype("float32")
        pos_weight = np.clip(pos_weight, 1.0, 10.0)  # cap to keep training stable

        bce = keras.losses.BinaryCrossentropy(reduction="none")

        def weighted_bce(y_true, y_pred):
            _ = bce(y_true, y_pred)  # kept for identical logic (though unused)
            y_pred = tf.clip_by_value(y_pred, 1e-7, 1.0 - 1e-7)
            per_class = -(
                y_true * tf.math.log(y_pred) * pos_weight
                + (1.0 - y_true) * tf.math.log(1.0 - y_pred)
            )
            per_sample = tf.reduce_mean(per_class, axis=-1)
            return per_sample

        fallback_model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss=weighted_bce,
        )

        ds_train = tf.data.Dataset.from_tensor_slices((x_paths_train, y_train))
        ds_train = ds_train.shuffle(
            buffer_size=min(len(x_paths_train), 8192),
            seed=42,
            reshuffle_each_iteration=True,
        )
        ds_train = ds_train.map(
            lambda p, l: _load_img_train(p, l, True),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_train = ds_train.batch(16).prefetch(tf.data.AUTOTUNE)

        ds_fit_val = tf.data.Dataset.from_tensor_slices((x_paths_fit_val, y_fit_val))
        ds_fit_val = ds_fit_val.map(
            lambda p, l: _load_img_train(p, l, False),
            num_parallel_calls=tf.data.AUTOTUNE,
        )
        ds_fit_val = ds_fit_val.batch(16).prefetch(tf.data.AUTOTUNE)

        fallback_model.fit(ds_train, validation_data=ds_fit_val, epochs=6, verbose=1)

        def predictor(x, training=False):
            return fallback_model(x, training=training)

        model_loaded = True
        kind = "trained_fallback_keras"
        resolved_model_path = "(in-notebook trained fallback model)"

    def _mean_sample_f1_np(y_true, y_pred_bin, eps=1e-9):
        y_true = y_true.astype(np.int32)
        y_pred_bin = y_pred_bin.astype(np.int32)
        tp = np.sum((y_true == 1) & (y_pred_bin == 1), axis=1).astype(np.float64)
        fp = np.sum((y_true == 0) & (y_pred_bin == 1), axis=1).astype(np.float64)
        fn = np.sum((y_true == 1) & (y_pred_bin == 0), axis=1).astype(np.float64)
        f1 = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)
        return float(np.mean(f1))

    @tf.function(reduce_retracing=True)
    def _predict_batch(xb):
        pred = predictor(xb, training=False)
        if isinstance(pred, dict):
            pred = pred[next(iter(pred.keys()))]
        elif isinstance(pred, (list, tuple)):
            pred = pred[0]
        return tf.convert_to_tensor(pred)

    def _predict_scores_on_paths(paths, batch_size=16):
        def _load_img_infer(path):
            img_bytes = tf.io.read_file(path)
            img = tf.io.decode_jpeg(img_bytes, channels=3)
            img = tf.image.resize(img, [image_dims[0], image_dims[1]])
            img = tf.cast(img, tf.float32)  # [0,255] to match predictor expectation
            return img

        dsx = tf.data.Dataset.from_tensor_slices(paths)
        dsx = dsx.map(_load_img_infer, num_parallel_calls=tf.data.AUTOTUNE)
        dsx = dsx.batch(batch_size).prefetch(tf.data.AUTOTUNE)

        all_scores = []
        for xb in dsx:
            pred = _predict_batch(xb)
            all_scores.append(pred.numpy())
        return np.vstack(all_scores)

    df_all = data_set.copy()
    df_all["path"] = df_all["image"].apply(lambda x: os.path.join(train_images_dir, x))
    y_all = (
        df_all["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=dataset_labels, fill_value=0)
        .astype("int32")
        .values
    )
    paths_all = df_all["path"].values

    rng = np.random.RandomState(42)
    idx = np.arange(len(df_all))
    rng.shuffle(idx)
    val_n = int(0.15 * len(idx))
    val_idx = idx[:val_n]

    val_paths = paths_all[val_idx]
    val_y = y_all[val_idx]

    val_scores = _predict_scores_on_paths(val_paths, batch_size=16)

    candidate_thresholds = np.arange(0.02, 0.981, 0.02).astype(np.float32)

    T = candidate_thresholds.shape[0]
    N = val_scores.shape[0]
    C = val_scores.shape[1]
    eps = 1e-9

    pred_all = (val_scores[None, :, :] > candidate_thresholds[:, None, None]).astype(
        np.int8
    )

    yb = val_y[None, :, :].astype(np.int8)

    tp = np.sum((pred_all == 1) & (yb == 1), axis=2).astype(np.float64)  # (T, N)
    fp = np.sum((pred_all == 1) & (yb == 0), axis=2).astype(np.float64)
    fn = np.sum((pred_all == 0) & (yb == 1), axis=2).astype(np.float64)
    f1_global_t = (2.0 * tp + eps) / (2.0 * tp + fp + fn + eps)  # (T, N)
    mean_f1_global = f1_global_t.mean(axis=1)  # (T,)
    best_t = float(candidate_thresholds[int(np.argmax(mean_f1_global))])
    best_f1 = float(mean_f1_global.max())

    per_class_thresholds = np.full((C,), 0.6, dtype=np.float32)
    base_pred = (val_scores > per_class_thresholds.reshape(1, -1)).astype(
        np.int8
    )  # (N,C)
    base_tp = np.sum((base_pred == 1) & (val_y == 1), axis=1).astype(np.int32)
    base_fp = np.sum((base_pred == 1) & (val_y == 0), axis=1).astype(np.int32)
    base_fn = np.sum((base_pred == 0) & (val_y == 1), axis=1).astype(np.int32)

    for c in range(C):
        b_pred_c = base_pred[:, c]
        y_c = val_y[:, c].astype(np.int8)

        tp0 = base_tp - ((b_pred_c == 1) & (y_c == 1)).astype(np.int32)
        fp0 = base_fp - ((b_pred_c == 1) & (y_c == 0)).astype(np.int32)
        fn0 = base_fn - ((b_pred_c == 0) & (y_c == 1)).astype(np.int32)

        pred_c_all = pred_all[:, :, c].astype(np.int8)  # (T,N)

        tp_c = (
            ((pred_c_all == 1) & (y_c[None, :] == 1)).sum(axis=1).astype(np.int32)
        )  # (T,)
        fp_c = ((pred_c_all == 1) & (y_c[None, :] == 0)).sum(axis=1).astype(np.int32)
        fn_c = ((pred_c_all == 0) & (y_c[None, :] == 1)).sum(axis=1).astype(np.int32)

        tp_tot = tp0[None, :] + (
            ((pred_c_all == 1) & (y_c[None, :] == 1)).astype(np.int32)
        )
        fp_tot = fp0[None, :] + (
            ((pred_c_all == 1) & (y_c[None, :] == 0)).astype(np.int32)
        )
        fn_tot = fn0[None, :] + (
            ((pred_c_all == 0) & (y_c[None, :] == 1)).astype(np.int32)
        )

        f1_tn = (2.0 * tp_tot + eps) / (2.0 * tp_tot + fp_tot + fn_tot + eps)  # (T,N)
        mean_f1_c = f1_tn.mean(axis=1)  # (T,)
        best_idx = int(np.argmax(mean_f1_c))
        per_class_thresholds[c] = float(candidate_thresholds[best_idx])

    threshold = best_t
    print(
        f"Calibrated global threshold on held-out split: {threshold:.3f} (mean-sample-F1={best_f1:.4f})"
    )
    print(
        "Per-class thresholds: "
        f"mean={float(np.mean(per_class_thresholds)):.3f}, "
        f"min={float(np.min(per_class_thresholds)):.3f}, "
        f"max={float(np.max(per_class_thresholds)):.3f}"
    )

    def _apply_thresholds(scores_2d, thresholds_1d):
        return (scores_2d > thresholds_1d.reshape(1, -1)).astype(np.int32)

    val_pred_bin_pc = _apply_thresholds(val_scores, per_class_thresholds)

    val_pred_always_healthy = val_pred_bin_pc.copy()
    val_pred_top1 = val_pred_bin_pc.copy()
    healthy_idx = (
        dataset_labels.index("healthy") if "healthy" in dataset_labels else None
    )

    empty_mask = val_pred_bin_pc.sum(axis=1) == 0
    if np.any(empty_mask):
        if healthy_idx is not None:
            val_pred_always_healthy[empty_mask, healthy_idx] = 1
        top1_idx = np.argmax(val_scores[empty_mask], axis=1)
        val_pred_top1[empty_mask, top1_idx] = 1

    f1_empty_healthy = _mean_sample_f1_np(val_y, val_pred_always_healthy)
    f1_empty_top1 = _mean_sample_f1_np(val_y, val_pred_top1)

    use_top1_on_empty = bool(f1_empty_top1 >= f1_empty_healthy)
    print(
        f"Empty-pred handling on val (mean-sample-F1): always_healthy={f1_empty_healthy:.4f}, "
        f"top1={f1_empty_top1:.4f} -> using {'top1' if use_top1_on_empty else 'always_healthy'}"
    )

    sample_sub = pd.read_csv(sample_sub_path)
    images_path_list = sample_sub["image"].tolist()
    test_paths = np.array(
        [os.path.join(test_dir, n) for n in images_path_list], dtype=object
    )

    test_scores = _predict_scores_on_paths(test_paths, batch_size=32)

    pred_bin_test = (test_scores > per_class_thresholds.reshape(1, -1)).astype(np.int32)
    empty_mask_t = pred_bin_test.sum(axis=1) == 0
    if np.any(empty_mask_t):
        if use_top1_on_empty:
            top1 = np.argmax(test_scores[empty_mask_t], axis=1)
            pred_bin_test[empty_mask_t, top1] = 1
        else:
            if healthy_idx is not None:
                pred_bin_test[empty_mask_t, healthy_idx] = 1
            else:
                top1 = np.argmax(test_scores[empty_mask_t], axis=1)
                pred_bin_test[empty_mask_t, top1] = 1

    values = []
    for i, name in enumerate(images_path_list):
        idxs = np.where(pred_bin_test[i] == 1)[0].tolist()
        labels_out = " ".join([dataset_labels[j] for j in sorted(idxs)]).strip()
        if labels_out == "":
            labels_out = "healthy"
        values.append([os.path.basename(name), labels_out])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"], index=None)
    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print(f"Wrote submission to: {out_path} (rows={len(csv_pd)})")
    print(f"Model loaded: {model_loaded}; source: {resolved_model_path} (kind={kind})")
