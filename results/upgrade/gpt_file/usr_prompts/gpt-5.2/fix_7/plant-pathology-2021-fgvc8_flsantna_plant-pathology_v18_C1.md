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

import numpy as np
import pandas as pd
import tensorflow as tf

keras = tf.keras

print("Python:", sys.version)
print("TF:", tf.__version__)
print("tf.keras:", tf.keras.__version__)
try:
    from google.protobuf import __version__ as pb_ver
except Exception:
    pb_ver = None
print("protobuf:", pb_ver)

tf.random.set_seed(42)
np.random.seed(42)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 1
output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
model_dir = "../input/model-25/epoch-25"

image_dims = (300, 300, 3)

train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
train_img_dir = "../input/plant-pathology-2021-fgvc8/train_images/"

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"]
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("First classes:", dataset_labels[:10])

Y_ALL = one_hot.reindex(columns=dataset_labels, fill_value=0).to_numpy(
    dtype=np.float32, copy=False
)
PATHS_ALL = (train_img_dir + data_set["image"]).to_numpy()




## === cell 2
def _is_savedmodel_dir(path: str) -> bool:
    return os.path.isdir(path) and (
        os.path.exists(os.path.join(path, "saved_model.pb"))
        or os.path.exists(os.path.join(path, "saved_model.pbtxt"))
    )


def _find_model_path(base_path: str) -> str:
    """
    Bug fix: the provided model_dir may not exist in this notebook environment.
    Search within base_path for:
      1) a SavedModel directory
      2) a .keras or .h5 model file
    Return the best candidate path, or None if nothing usable found.
    """
    if base_path is None:
        return None
    if os.path.isfile(base_path) and base_path.endswith((".keras", ".h5")):
        return base_path
    if _is_savedmodel_dir(base_path):
        return base_path
    if not os.path.isdir(base_path):
        return None

    candidates = []
    for root, _, files in os.walk(base_path):
        if _is_savedmodel_dir(root):
            candidates.append(("savedmodel", root))
        for fn in files:
            if fn.endswith(".keras") or fn.endswith(".h5"):
                candidates.append(("file", os.path.join(root, fn)))

    savedmodels = [p for t, p in candidates if t == "savedmodel"]
    if savedmodels:
        savedmodels = sorted(savedmodels, key=lambda p: (p.count(os.sep), len(p)))
        return savedmodels[0]

    files = [p for t, p in candidates if t == "file"]
    if files:
        files = sorted(files, key=lambda p: (p.count(os.sep), len(p)))
        return files[0]

    return None


def _load_inference_model(path: str):
    """
    Loads either a Keras model file (.keras/.h5) or a TF SavedModel via TFSMLayer.
    Returns a callable inference_fn(x)->tensor and a string description.
    """
    if path is None:
        return None, "no_model_found"

    if os.path.isfile(path) and path.endswith((".keras", ".h5")):
        model = tf.keras.models.load_model(path, compile=False)

        @tf.function(input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)])
        def _call(x):
            return model(x, training=False)

        return _call, f"keras_model_file:{path}"

    if _is_savedmodel_dir(path):
        layer = tf.keras.layers.TFSMLayer(path, call_endpoint="serving_default")

        @tf.function(input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)])
        def _call(x):
            return layer(x)

        return _call, f"savedmodel_dir:{path}"

    return None, f"unusable_path:{path}"


resolved_model_path = _find_model_path(model_dir)
print("Resolved model path:", resolved_model_path)

inference_fn, model_kind = _load_inference_model(resolved_model_path)
print("Model kind:", model_kind)



## === cell 3
num_classes = len(dataset_labels)


def _build_model():
    inputs = tf.keras.Input(shape=image_dims, name="image")
    x = tf.keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.MaxPooling2D()(x)
    x = tf.keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dense(256, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
    model = tf.keras.Model(inputs, outputs)
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


@tf.function
def _decode_and_resize_from_bytes(img_bytes):
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # identical decode semantics
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, [image_dims[0], image_dims[1]], antialias=False)
    return img


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


def _make_train_dataset_from_arrays(
    paths, y, batch_size=16, shuffle=True, cache_in_memory=True
):
    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    if shuffle:
        ds = ds.shuffle(
            buffer_size=min(int(len(paths)), 2048),
            seed=42,
            reshuffle_each_iteration=True,
        )

    ds = ds.map(
        lambda p, label: (_read_bytes(p), label), num_parallel_calls=tf.data.AUTOTUNE
    )

    if cache_in_memory:
        ds = ds.cache()

    ds = ds.map(
        lambda img_bytes, label: (_decode_and_resize_from_bytes(img_bytes), label),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size).prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    return ds


if inference_fn is None:
    df = data_set.copy()
    df = df.sample(frac=1.0, random_state=42).reset_index(drop=True)

    split = int(0.95 * len(df))
    idx = df.index.to_numpy()
    train_idx = idx[:split]
    val_idx = idx[split:]

    train_paths = PATHS_ALL[train_idx]
    val_paths = PATHS_ALL[val_idx]
    train_y = Y_ALL[train_idx]
    val_y = Y_ALL[val_idx]

    train_ds = _make_train_dataset_from_arrays(
        train_paths, train_y, batch_size=16, shuffle=True, cache_in_memory=True
    )
    val_ds = _make_train_dataset_from_arrays(
        val_paths, val_y, batch_size=16, shuffle=False, cache_in_memory=True
    )

    model = _build_model()
    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    @tf.function(input_signature=[tf.TensorSpec([None, *image_dims], tf.float32)])
    def inference_fn(x):
        return model(x, training=False)

    model_kind = "trained_fallback_cnn"
    print("Using fallback trained model.")
else:
    print("Using provided pretrained model.")



## === cell 4
images_path_list = sorted(list(os.listdir(test_dir)))
print("Num test images:", len(images_path_list))

test_paths = [os.path.join(test_dir, fn) for fn in images_path_list]


def _make_test_dataset(paths, batch_size=32):
    ds = tf.data.Dataset.from_tensor_slices(paths)
    ds = ds.map(_read_bytes, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.map(_decode_and_resize_from_bytes, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    return ds


test_ds = _make_test_dataset(test_paths, batch_size=32)

values = []
classes = dataset_labels
classes_arr = np.asarray(classes, dtype=object)
threshold = 0.7

offset = 0
for batch_imgs in test_ds:
    out = inference_fn(batch_imgs)
    if isinstance(out, dict):
        out = out[next(iter(out.keys()))]

    out_np = out.numpy()  # one host transfer per batch
    mask = out_np > threshold
    bs = mask.shape[0]

    idx_lists = [np.flatnonzero(mask[i]) for i in range(bs)]
    for i, idxs in enumerate(idx_lists):
        name = images_path_list[offset + i]
        if idxs.size == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join(classes_arr[idxs[::-1]])
        values.append([name, classes_img])

    offset += bs

csv_pd = pd.DataFrame(values, columns=["image", "labels"])
out_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(out_path, index=False)

print("Model kind used:", model_kind)
print("Wrote:", out_path)
print(csv_pd.head())
print("Submission shape:", csv_pd.shape)
