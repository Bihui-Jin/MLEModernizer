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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys
import pandas as pd
import tensorflow as tf
import numpy as np

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
model_dir = "../input/model-effb7-01/epoch-5/"  # original path (may not exist)

image_dims = (300, 300, 3)

data_set = pd.read_csv("../input/plant-pathology-2021-fgvc8/train.csv")
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

sample_sub = pd.read_csv("../input/plant-pathology-2021-fgvc8/sample_submission.csv")
test_images_order = sample_sub["image"].tolist()

print("Num classes:", len(dataset_labels))
print("Train rows:", len(data_set))
print("Test images in submission template:", len(test_images_order))



## === cell 2
if __name__ == "__main__":

    def _is_savedmodel_dir(p: str) -> bool:
        return os.path.isdir(p) and (
            os.path.isfile(os.path.join(p, "saved_model.pb"))
            or os.path.isfile(os.path.join(p, "saved_model.pbtxt"))
        )

    def find_model_path(preferred_path: str, root: str = "../input") -> str:
        if preferred_path and (
            _is_savedmodel_dir(preferred_path) or os.path.isfile(preferred_path)
        ):
            return preferred_path

        candidates = [
            "../input/model-effb7-01/epoch-5/",
            "../input/model-effb7-01/",
            "../input/model/",
            "../input/models/",
        ]
        for base in candidates:
            if _is_savedmodel_dir(base):
                return base
            for ep in (
                "epoch-5",
                "epoch-4",
                "epoch-3",
                "epoch-2",
                "epoch-1",
                "epoch-0",
            ):
                p = os.path.join(base, ep)
                if _is_savedmodel_dir(p):
                    return p

        max_depth = 1
        savedmodel_candidates = []
        keras_file_candidates = []
        for dirpath, dirnames, filenames in os.walk(root):
            rel_depth = os.path.relpath(dirpath, root).count(os.sep)
            if rel_depth > max_depth:
                dirnames[:] = []
                continue

            if "saved_model.pb" in filenames or "saved_model.pbtxt" in filenames:
                savedmodel_candidates.append(dirpath)
                dirnames[:] = []
                continue

            for fn in filenames:
                if fn.endswith(".keras") or fn.endswith(".h5"):
                    keras_file_candidates.append(os.path.join(dirpath, fn))

        if savedmodel_candidates:
            savedmodel_candidates_sorted = sorted(
                savedmodel_candidates,
                key=lambda p: (("epoch" not in p.lower()), len(p)),
            )
            return savedmodel_candidates_sorted[0]

        if keras_file_candidates:
            return sorted(keras_file_candidates, key=len)[0]

        raise FileNotFoundError(
            f"Could not find a SavedModel directory or .keras/.h5 model under {root}. "
            f"Also checked preferred_path={preferred_path!r}."
        )

    @tf.function
    def _load_image_tf(img_path, training=False):
        raw = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(raw, channels=3, dct_method="INTEGER_FAST")
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]])
        img.set_shape(image_dims)
        if training:
            img = tf.image.random_flip_left_right(img)
            img = tf.image.random_flip_up_down(img)
        img = img * 255.0  # preserve original scaling logic used in inference
        return img

    def load_image(img_path, training=False):
        return _load_image_tf(img_path, training=training)

    def build_fallback_model(num_classes: int):
        inputs = tf.keras.Input(shape=image_dims, dtype=tf.float32)
        backbone = tf.keras.applications.EfficientNetB0(
            include_top=False,
            weights="imagenet",
            input_tensor=inputs,
            pooling="avg",
        )
        backbone.trainable = False
        x = backbone.output
        outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid")(x)
        model = tf.keras.Model(inputs, outputs)
        model.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3),
            loss="binary_crossentropy",
        )
        return model

    def multilabel_f1(y_true, y_pred, thr):
        y_true = y_true.astype(np.int32)
        y_pred = (y_pred >= thr).astype(np.int32)
        tp = (y_true & y_pred).sum(axis=0)
        fp = ((1 - y_true) & y_pred).sum(axis=0)
        fn = (y_true & (1 - y_pred)).sum(axis=0)
        f1 = (2 * tp) / (2 * tp + fp + fn + 1e-9)
        return float(np.mean(f1))

    model = None
    resolved_model_path = None
    use_fallback_train = False

    try:
        resolved_model_path = find_model_path(model_dir, root="../input")
        print("Resolved model path:", resolved_model_path)

        if _is_savedmodel_dir(resolved_model_path):
            try:
                import keras  # keras 3

                TFSMLayer = keras.layers.TFSMLayer
            except Exception:
                from tensorflow import keras

                TFSMLayer = keras.layers.TFSMLayer

            tfsm_layer = None
            last_err = None
            for endpoint in ["serving_default", "serve"]:
                try:
                    tfsm_layer = TFSMLayer(resolved_model_path, call_endpoint=endpoint)
                    print("Loaded SavedModel via TFSMLayer with endpoint:", endpoint)
                    break
                except Exception as e:
                    last_err = e

            if tfsm_layer is None:
                raise RuntimeError(
                    f"Failed to load SavedModel via TFSMLayer: {last_err!r}"
                )

            inp = tf.keras.Input(shape=image_dims, dtype=tf.float32, name="image")
            out = tfsm_layer(inp)
            model = tf.keras.Model(inp, out, name="wrapped_savedmodel")

        else:
            model = tf.keras.models.load_model(resolved_model_path, compile=False)
            print("Loaded Keras model file:", resolved_model_path)

    except FileNotFoundError as e:
        print(
            "Pretrained model not found; switching to fallback training. Details:",
            str(e),
        )
        use_fallback_train = True

    idx = np.arange(len(data_set))
    rng = np.random.RandomState(42)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]

    train_df = data_set.iloc[tr_idx].reset_index(drop=True)
    val_df = data_set.iloc[va_idx].reset_index(drop=True)

    y_train = (
        train_df["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=dataset_labels, fill_value=0)
        .values.astype(np.float32)
    )
    y_val = (
        val_df["labels"]
        .str.get_dummies(sep=" ")
        .reindex(columns=dataset_labels, fill_value=0)
        .values.astype(np.float32)
    )

    train_paths = [os.path.join(train_dir, fn) for fn in train_df["image"].tolist()]
    val_paths = [os.path.join(train_dir, fn) for fn in val_df["image"].tolist()]

    def make_ds(paths, y=None, training=False, batch_size=16, cache=False):
        options = tf.data.Options()
        options.experimental_deterministic = True

        if y is None:
            ds = tf.data.Dataset.from_tensor_slices(paths)
            ds = ds.with_options(options)
            ds = ds.map(
                lambda p: load_image(p, training=False),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.apply(tf.data.experimental.ignore_errors())
            if cache:
                ds = ds.cache()
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds
        else:
            ds = tf.data.Dataset.from_tensor_slices((paths, y))
            ds = ds.with_options(options)
            if training:
                ds = ds.shuffle(2048, seed=42, reshuffle_each_iteration=True)
            ds = ds.map(
                lambda p, t: (load_image(p, training=training), t),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.apply(tf.data.experimental.ignore_errors())
            if cache and (not training):
                ds = ds.cache()
            ds = ds.batch(batch_size, drop_remainder=False)
            ds = ds.prefetch(tf.data.AUTOTUNE)
            return ds

    if use_fallback_train:
        model = build_fallback_model(num_classes=len(dataset_labels))
        train_ds = make_ds(
            train_paths, y_train, training=True, batch_size=16, cache=False
        )
        val_ds = make_ds(val_paths, y_val, training=False, batch_size=16, cache=True)
        model.fit(train_ds, validation_data=val_ds, epochs=1, verbose=2)

    val_ds_for_pred = make_ds(
        val_paths, y=None, training=False, batch_size=32, cache=True
    )
    val_pred = model.predict(val_ds_for_pred, verbose=0)
    if isinstance(val_pred, dict):
        val_pred = next(iter(val_pred.values()))
    val_pred = np.asarray(val_pred)

    best_thr = 0.8
    best_f1 = -1.0
    for thr in np.linspace(0.2, 0.9, 15):
        f1 = multilabel_f1(y_val.astype(np.int32), val_pred, thr=thr)
        if f1 > best_f1:
            best_f1 = f1
            best_thr = float(thr)
    print(f"Calibrated threshold: {best_thr:.3f} (val mean F1 ~ {best_f1:.4f})")

    test_paths = [os.path.join(test_dir, fn) for fn in test_images_order]

    test_ds = make_ds(test_paths, y=None, training=False, batch_size=32, cache=False)
    test_pred = model.predict(test_ds, verbose=0)
    if isinstance(test_pred, dict):
        test_pred = next(iter(test_pred.values()))
    test_pred = np.asarray(test_pred)

    n_expected = len(test_images_order)
    if test_pred.shape[0] != n_expected:
        if test_pred.shape[0] > n_expected:
            test_pred = test_pred[:n_expected]
        else:
            pad = np.zeros(
                (n_expected - test_pred.shape[0], test_pred.shape[1]),
                dtype=test_pred.dtype,
            )
            test_pred = np.concatenate([test_pred, pad], axis=0)
        print(
            "Warning: test prediction rows != submission rows; padded/truncated to match:",
            test_pred.shape[0],
            "expected:",
            n_expected,
        )

    above = test_pred > best_thr
    has_any = above.any(axis=1)
    argmax_idx = np.argmax(test_pred, axis=1)
    above[np.arange(len(above)), argmax_idx] |= ~has_any

    class_names = np.asarray(dataset_labels, dtype=object)

    true_pos = np.argwhere(above)
    row_ids = true_pos[:, 0]
    col_ids = true_pos[:, 1]
    row_counts = np.bincount(row_ids, minlength=above.shape[0])
    splits = np.cumsum(row_counts[:-1])
    per_row_cols = (
        np.split(col_ids, splits)
        if len(col_ids)
        else [np.array([], dtype=int)] * above.shape[0]
    )

    existing_labels = [
        " ".join(class_names[cols].tolist()) if len(cols) else "healthy"
        for cols in per_row_cols
    ]

    csv_pd = pd.DataFrame({"image": test_images_order, "labels": existing_labels})
    csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
    csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

    out_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(out_path, index=False)

    print(csv_pd.head())
    print("Wrote:", out_path, "rows:", len(csv_pd))
    assert out_path.endswith(".csv")
    assert list(csv_pd.columns) == ["image", "labels"]
    assert len(csv_pd) == len(sample_sub)
