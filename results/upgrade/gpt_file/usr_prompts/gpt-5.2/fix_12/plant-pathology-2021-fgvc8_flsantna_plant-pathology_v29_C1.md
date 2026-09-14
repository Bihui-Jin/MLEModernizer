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
import random
import numpy as np
import pandas as pd

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import tensorflow as tf

keras = tf.keras

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)

os.environ["PYTHONHASHSEED"] = "0"
random.seed(0)
np.random.seed(0)
tf.random.set_seed(0)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")




## === cell 1
def _first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


output_dir = "./"

train_csv_path = _first_existing_path(
    [
        "../input/plant-pathology-2021-fgvc8/train.csv",
        "/kaggle/input/plant-pathology-2021-fgvc8/train.csv",
        "../input/train.csv",
        "/kaggle/input/train.csv",
    ]
)
test_dir = _first_existing_path(
    [
        "../input/plant-pathology-2021-fgvc8/test_images/",
        "/kaggle/input/plant-pathology-2021-fgvc8/test_images/",
        "../input/test_images/",
        "/kaggle/input/test_images/",
    ]
)
train_dir = _first_existing_path(
    [
        "../input/plant-pathology-2021-fgvc8/train_images/",
        "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
        "../input/train_images/",
        "/kaggle/input/train_images/",
    ]
)

model_dir = "../input/dense-e1/epoch-1"

if train_csv_path is None:
    raise FileNotFoundError(
        "Could not locate train.csv in expected Kaggle input paths."
    )
if test_dir is None or not os.path.isdir(test_dir):
    raise FileNotFoundError(
        "Could not locate test_images/ directory in expected Kaggle input paths."
    )
if train_dir is None or not os.path.isdir(train_dir):
    raise FileNotFoundError(
        "Could not locate train_images/ directory in expected Kaggle input paths."
    )

image_dims = (300, 300, 3)
data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

print("Train CSV:", train_csv_path)
print("Train dir:", train_dir)
print("Test dir :", test_dir)
print("Num classes:", num_classes)
print("First classes:", dataset_labels[:10])




## === cell 2
if __name__ == "__main__":

    @tf.function(reduce_retracing=True)
    def _decode_and_resize_jpeg(img_path: tf.Tensor) -> tf.Tensor:
        bytes_ = tf.io.read_file(img_path)
        img = tf.io.decode_jpeg(bytes_, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, [image_dims[0], image_dims[1]], method="bilinear")
        img = img * 255.0  # keep original inference scaling
        return img

    def _build_infer_model_from_savedmodel(saved_model_dir: str) -> keras.Model:
        endpoints_to_try = ["serving_default", "serve", "call"]
        last_err = None
        for endpoint in endpoints_to_try:
            try:
                layer = keras.layers.TFSMLayer(saved_model_dir, call_endpoint=endpoint)
                inp = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
                out = layer(inp)
                if isinstance(out, dict):
                    for k in ["outputs", "predictions", "logits", "output_0"]:
                        if k in out:
                            out = out[k]
                            break
                    if isinstance(out, dict):
                        out = list(out.values())[0]
                return keras.Model(inp, out)
            except Exception as e:
                last_err = e
        raise RuntimeError(
            f"Could not load SavedModel from {saved_model_dir}. Last error: {last_err}"
        )

    def _savedmodel_exists(d: str) -> bool:
        return os.path.exists(os.path.join(d, "saved_model.pb")) or os.path.exists(
            os.path.join(d, "saved_model.pbtxt")
        )

    candidate_model_dirs = [
        model_dir,
        "/kaggle/input/dense-e1/epoch-1",
        "../input/dense-e1/epoch-1",
        "/kaggle/input/dense-e1",
        "../input/dense-e1",
        "/kaggle/input/dense-e1/epoch-1/epoch-1",
        "../input/dense-e1/epoch-1/epoch-1",
    ]
    model_dir_resolved = None
    for d in candidate_model_dirs:
        if d and os.path.isdir(d) and _savedmodel_exists(d):
            model_dir_resolved = d
            break

    def _build_fallback_train_model() -> keras.Model:
        inputs = keras.Input(shape=image_dims, dtype=tf.float32, name="image")
        x = inputs / 255.0
        x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
        x = keras.layers.MaxPooling2D()(x)
        x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
        x = keras.layers.GlobalAveragePooling2D()(x)
        x = keras.layers.Dropout(0.2)(x)
        outputs = keras.layers.Dense(num_classes, activation="sigmoid")(x)
        model = keras.Model(inputs, outputs)
        model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=1e-3),
            loss="binary_crossentropy",
        )
        return model

    cache_dir = os.path.join(output_dir, "tfdata_cache")
    os.makedirs(cache_dir, exist_ok=True)

    if model_dir_resolved is not None:
        print(f"Loading SavedModel from: {model_dir_resolved}")
        model = _build_infer_model_from_savedmodel(model_dir_resolved)
        needs_training = False
    else:
        print(
            f"SavedModel not found at {model_dir}. Training a fallback model from train_images/..."
        )
        needs_training = True
        model = _build_fallback_train_model()

        label_matrix = one_hot.values.astype("float32", copy=False)
        img_paths = (train_dir + data_set["image"].values).astype(object)

        rng = np.random.RandomState(0)
        idx = np.arange(len(img_paths))
        rng.shuffle(idx)
        split = int(0.95 * len(idx))
        tr_idx, va_idx = idx[:split], idx[split:]

        tr_paths, tr_y = img_paths[tr_idx], label_matrix[tr_idx]
        va_paths, va_y = img_paths[va_idx], label_matrix[va_idx]

        base_opts = tf.data.Options()
        base_opts.experimental_deterministic = True

        def _make_ds(paths, y, training: bool):
            ds = tf.data.Dataset.from_tensor_slices((paths, y)).with_options(base_opts)
            ds = ds.map(
                lambda p, lab: (_decode_and_resize_jpeg(p), lab),
                num_parallel_calls=tf.data.AUTOTUNE,
                deterministic=True,
            )
            ds = ds.cache()  # same outputs; avoids re-decode on epoch 2
            if training:
                ds = ds.shuffle(
                    buffer_size=min(8192, int(paths.shape[0])),
                    seed=0,
                    reshuffle_each_iteration=True,
                )
            ds = ds.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
            return ds

        train_ds = _make_ds(tr_paths, tr_y, training=True)
        valid_ds = _make_ds(va_paths, va_y, training=False)

        model.fit(train_ds, validation_data=valid_ds, epochs=2, verbose=2)

    images_path_list = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
    if len(images_path_list) == 0:
        raise RuntimeError(f"No .jpg files found in test_dir={test_dir}")

    classes = np.array(dataset_labels, dtype=object)
    threshold = 0.7

    test_paths = np.array(
        [os.path.join(test_dir, f) for f in images_path_list], dtype=object
    )
    test_names = np.array(images_path_list, dtype=object)

    base_opts = tf.data.Options()
    base_opts.experimental_deterministic = True

    def _make_test_ds(paths):
        ds = tf.data.Dataset.from_tensor_slices(paths).with_options(base_opts)
        ds = ds.map(
            _decode_and_resize_jpeg,
            num_parallel_calls=tf.data.AUTOTUNE,
            deterministic=True,
        )
        ds = ds.batch(64, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
        return ds

    test_ds = _make_test_ds(test_paths)

    preds = model.predict(test_ds, verbose=0)
    preds = np.asarray(preds)
    if preds.ndim == 1:
        preds = preds[:, None]

    mask = preds > threshold

    idx_lists = [np.flatnonzero(row) for row in mask]

    values = []
    classes_list = classes.tolist()
    for fname, idxs in zip(test_names.tolist(), idx_lists):
        if idxs.size:
            classes_img = " ".join(classes_list[i] for i in idxs.tolist()).strip()
        else:
            classes_img = "healthy"
        values.append([fname, classes_img])

    csv_pd = pd.DataFrame(values, columns=["image", "labels"])
    sub_path = os.path.join(output_dir, "submission.csv")
    csv_pd.to_csv(sub_path, index=False)
    print("Wrote submission.csv with shape:", csv_pd.shape, "to", sub_path)
    print(csv_pd.head())
