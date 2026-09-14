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

0.6584487534626026

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
    if base_path is None:
        return None
    if os.path.isfile(base_path) and base_path.endswith((".keras", ".h5")):
        return base_path
    if _is_savedmodel_dir(base_path):
        return base_path
    if not os.path.isdir(base_path):
        return None

    try:
        entries = [os.path.join(base_path, n) for n in os.listdir(base_path)]
    except Exception:
        entries = []

    for p in entries:
        if os.path.isfile(p) and p.endswith((".keras", ".h5")):
            return p
        if _is_savedmodel_dir(p):
            return p

    base_depth = base_path.rstrip(os.sep).count(os.sep)
    best_savedmodel = None
    best_file = None

    for root, dirs, files in os.walk(base_path):
        depth = root.rstrip(os.sep).count(os.sep) - base_depth
        if depth > 3:
            dirs[:] = []
            continue

        if _is_savedmodel_dir(root):
            best_savedmodel = root
            break  # SavedModel directory is directly usable.

        for fn in files:
            if fn.endswith(".keras") or fn.endswith(".h5"):
                best_file = os.path.join(root, fn)
                break

    return best_savedmodel or best_file


def _load_inference_model(path: str):
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
    raise FileNotFoundError(
        "Pretrained model not found/resolvable under model_dir. "
        "Fallback training is disabled to meet the 600s timeout. "
        f"model_dir={model_dir}"
    )
else:
    print("Using provided pretrained model.")




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_56/837322363.py in <cell line: 0>()
     69 # when the provided model artifact is present.
     70 if inference_fn is None:
---> 71     raise FileNotFoundError(
     72         "Pretrained model not found/resolvable under model_dir. "
     73         "Fallback training is disabled to meet the 600s timeout. "

FileNotFoundError: Pretrained model not found/resolvable under model_dir. Fallback training is disabled to meet the 600s timeout. model_dir=../input/model-25/epoch-25

## === cell 4
pattern = os.path.join(test_dir, "*.jpg")
test_files_ds = tf.data.Dataset.list_files(pattern, shuffle=False)

print("Test glob pattern:", pattern)


@tf.function
def _path_to_name(path):
    parts = tf.strings.split(path, os.sep)
    return parts[-1]


def _make_test_dataset(paths_ds, batch_size=32):
    ds = paths_ds.map(
        lambda p: (p, _read_bytes(p)), num_parallel_calls=tf.data.AUTOTUNE
    )
    ds = ds.map(
        lambda p, b: (p, _decode_and_resize_from_bytes(b)),
        num_parallel_calls=tf.data.AUTOTUNE,
    )
    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    opts = tf.data.Options()
    opts.experimental_deterministic = True
    ds = ds.with_options(opts)
    return ds


test_ds = _make_test_dataset(test_files_ds, batch_size=32)

values = []
classes = dataset_labels
classes_arr = np.asarray(classes, dtype=object)
threshold = 0.7

for batch_paths, batch_imgs in test_ds:
    out = inference_fn(batch_imgs)
    if isinstance(out, dict):
        out = out[next(iter(out.keys()))]

    out_np = out.numpy()  # one host transfer per batch
    mask = out_np > threshold

    names = batch_paths.numpy()
    names = [os.path.basename(n.decode("utf-8")) for n in names]

    for name, row in zip(names, mask):
        idxs = np.flatnonzero(row)
        if idxs.size == 0:
            classes_img = "healthy"
        else:
            classes_img = " ".join(classes_arr[idxs[::-1]])
        values.append([name, classes_img])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])
out_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(out_path, index=False)

print("Model kind used:", model_kind)
print("Wrote:", out_path)
print(csv_pd.head())
print("Submission shape:", csv_pd.shape)

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_56/2983442930.py in <cell line: 0>()
     40 # Speed: vectorize per-batch label string assembly; avoids per-image Python loops and repeated joins.
     41 for batch_paths, batch_imgs in test_ds:
---> 42     out = inference_fn(batch_imgs)
     43     if isinstance(out, dict):
     44         out = out[next(iter(out.keys()))]

TypeError: 'NoneType' object is not callable
