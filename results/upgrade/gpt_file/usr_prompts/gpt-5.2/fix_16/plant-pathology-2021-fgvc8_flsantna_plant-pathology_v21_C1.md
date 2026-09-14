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

0.7885318559556815

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
import random
import numpy as np
import pandas as pd

try:
    import google.protobuf  # noqa: F401

    try:
        import subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf<5"]
        )
    except Exception as _e:
        pass
except Exception:
    pass

import tensorflow as tf

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
tf.keras.backend.clear_session()

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv_path = (
    "../input/plant-pathology-2021-fgvcvc8/train.csv"
    if False
    else "../input/plant-pathology-2021-fgvc8/train.csv"
)
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

model_dir = "../input/model-effb7-01/epoch-5"

image_dims = (300, 300, 3)
IMG_SIZE = (image_dims[0], image_dims[1])

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## === cell 1
def _find_savedmodel_dir_fast(path: str) -> str:
    path = os.path.expanduser(path)

    def is_savedmodel_dir(d: str) -> bool:
        return os.path.isdir(d) and (
            os.path.exists(os.path.join(d, "saved_model.pb"))
            or os.path.exists(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path
    if not os.path.exists(path):
        raise OSError(f"Path does not exist: {path}")

    for sub in ("saved_model", "SavedModel", "export", "model", "1"):
        cand = os.path.join(path, sub)
        if is_savedmodel_dir(cand):
            return cand

    raise OSError(
        f"SavedModel file does not exist under: {path}. "
        f"Expected a directory containing saved_model.pb (or pbtxt)."
    )


def _load_and_preprocess(
    path: tf.Tensor, image_size=(300, 300), scale255=True
) -> tf.Tensor:
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
    img = tf.image.resize(img, list(image_size))
    if scale255:
        img = img * 255.0
    return img


def _labels_series_to_multi_hot(labels_series: pd.Series, classes: list) -> np.ndarray:
    oh = labels_series.astype(str).str.get_dummies(sep=" ")
    oh = oh.reindex(columns=classes, fill_value=0)
    return oh.to_numpy(dtype=np.float32, copy=False)


def _build_predict_fn_from_savedmodel(resolved_model_dir: str):
    sm_layer = tf.keras.layers.TFSMLayer(
        resolved_model_dir, call_endpoint="serving_default"
    )

    dummy = tf.zeros((1, image_dims[0], image_dims[1], image_dims[2]), dtype=tf.float32)
    dummy_out = sm_layer(dummy)
    if isinstance(dummy_out, dict):
        out_key = sorted(dummy_out.keys())[0]
        print("SavedModel output is dict. Using key:", out_key)

        @tf.function(reduce_retracing=True)
        def predict_fn(x):
            return sm_layer(x)[out_key]

    else:
        print("SavedModel output is tensor.")

        @tf.function(reduce_retracing=True)
        def predict_fn(x):
            return sm_layer(x)

    return predict_fn


def _build_and_train_fallback_model(train_df: pd.DataFrame, classes: list):
    base = tf.keras.applications.MobileNetV2(
        include_top=False,
        weights="imagenet",
        input_shape=(224, 224, 3),
        pooling="avg",
    )
    base.trainable = False

    inp = tf.keras.Input(shape=(224, 224, 3), dtype=tf.float32)
    x = base(inp, training=False)
    out = tf.keras.layers.Dense(len(classes), activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(1e-3),
        loss="binary_crossentropy",
    )

    img_names = train_df["image"].astype(str).to_numpy()
    base_path = train_dir.rstrip("/") + "/"
    paths = np.char.add(base_path, img_names)

    y = _labels_series_to_multi_hot(train_df["labels"], classes)

    ds = tf.data.Dataset.from_tensor_slices((paths, y))
    options = tf.data.Options()
    options.experimental_deterministic = True
    ds = ds.with_options(options)

    def _map_fn(p, yv):
        img = _load_and_preprocess(p, image_size=(224, 224), scale255=False)
        return img, yv

    ds = ds.shuffle(4096, seed=SEED, reshuffle_each_iteration=False)
    ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

    model.fit(ds, epochs=2, verbose=2)
    return model


def _resolve_model_dir_or_search(model_dir: str) -> str:
    if os.path.exists(model_dir):
        return model_dir

    candidates = []
    model_dir_norm = os.path.normpath(model_dir)
    parts = [p for p in model_dir_norm.split(os.sep) if p and p not in (".", "..")]

    base = os.path.abspath("../input")

    if len(parts) >= 1:
        tail = os.path.join(*parts[-min(6, len(parts)) :])
        try:
            for entry in os.listdir(base):
                candidates.append(os.path.join(base, entry, tail))
        except Exception:
            pass

    for k in (2, 3, 4, 5):
        if len(parts) >= k:
            tail = os.path.join(*parts[-k:])
            candidates.append(os.path.join(base, tail))

    if len(parts) >= 1 and "input" in parts:
        i = parts.index("input")
        candidates.append(os.path.join(base, *parts[i + 1 :]))

    if len(parts) >= 1:
        epoch_name = parts[-1]
        try:
            for entry in os.listdir(base):
                candidates.append(os.path.join(base, entry, epoch_name))
        except Exception:
            pass

    for cand in candidates:
        if os.path.exists(cand):
            return cand

    return model_dir


use_saved_model = False
predict_fn = None
fallback_model = None

try:
    candidate_model_dir = _resolve_model_dir_or_search(model_dir)
    if os.path.exists(candidate_model_dir):
        resolved_model_dir = _find_savedmodel_dir_fast(candidate_model_dir)
        print("Resolved model dir:", resolved_model_dir)
        predict_fn = _build_predict_fn_from_savedmodel(resolved_model_dir)
        use_saved_model = True
    else:
        print("External model_dir not found, will use fallback training:", model_dir)
except Exception as e:
    print(
        "Could not load external SavedModel; will use fallback training. Error:",
        repr(e),
    )

if not use_saved_model:
    fallback_model = _build_and_train_fallback_model(data_set, dataset_labels)

    @tf.function(reduce_retracing=True)
    def predict_fn(x):
        return fallback_model(x)


images_path_list = sorted(
    [f for f in tf.io.gfile.listdir(test_dir) if f.lower().endswith(".jpg")]
)
print("Num test images found:", len(images_path_list))

threshold = 0.7  # preserve core semantics



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3027319756.py in <cell line: 0>()
    173 
    174 if not use_saved_model:
--> 175     fallback_model = _build_and_train_fallback_model(data_set, dataset_labels)
    176 
    177     @tf.function(reduce_retracing=True)

/tmp/ipykernel_55/3027319756.py in _build_and_train_fallback_model(train_df, classes)
     90     img_names = train_df["image"].astype(str).to_numpy()
     91     base_path = train_dir.rstrip("/") + "/"
---> 92     paths = np.char.add(base_path, img_names)
     93 
     94     y = _labels_series_to_multi_hot(train_df["labels"], classes)

/usr/local/lib/python3.11/dist-packages/numpy/core/defchararray.py in add(x1, x2)
    330         # object dtype itemsize as num chars (worked on short strings).
    331         # bytes + void worked but promoting void->bytes is dubious also.
--> 332         raise TypeError(
    333             "np.char.add() requires both arrays of the same dtype kind, but "
    334             f"got dtypes: '{arr1.dtype}' and '{arr2.dtype}' (the few cases "

TypeError: np.char.add() requires both arrays of the same dtype kind, but got dtypes: '<U49' and 'object' (the few cases where this used to work often lead to incorrect results).

## === cell 2
BATCH_SIZE = 64

options = tf.data.Options()
options.experimental_deterministic = True

test_files = np.array([os.path.join(test_dir, f) for f in images_path_list], dtype=str)
test_names = np.array(images_path_list, dtype=str)


def _load_and_preprocess_for_infer(full_path, fname):
    if use_saved_model:
        img = _load_and_preprocess(
            full_path, image_size=(image_dims[0], image_dims[1]), scale255=True
        )
    else:
        img = _load_and_preprocess(full_path, image_size=(224, 224), scale255=False)
    return fname, img


test_ds = tf.data.Dataset.from_tensor_slices((test_files, test_names))
test_ds = test_ds.with_options(options)
test_ds = test_ds.map(
    _load_and_preprocess_for_infer, num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

values = []
n_classes = len(dataset_labels)
dataset_labels_arr = np.asarray(dataset_labels, dtype=object)

for batch_names, batch_imgs in test_ds:
    preds = predict_fn(batch_imgs)
    preds = tf.convert_to_tensor(preds).numpy()
    if preds.ndim == 1:
        preds = preds[:, None]
    m = min(preds.shape[-1], n_classes)
    preds = preds[:, :m]

    above = preds > threshold  # boolean [B, m]
    batch_names_np = batch_names.numpy().astype(str)

    label_strs = []
    for row_mask in above:
        cols = np.flatnonzero(row_mask)
        if cols.size:
            label_strs.append(" ".join(dataset_labels_arr[cols]).strip())
        else:
            label_strs.append("healthy")

    values.extend(zip(batch_names_np.tolist(), label_strs))

csv_pd = pd.DataFrame(values, columns=["image", "labels"])

sample_sub = pd.read_csv(sample_sub_path)
csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(csv_path, index=False)
print("Wrote:", csv_path, "rows:", len(csv_pd))
print(csv_pd.head())

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2293996051.py in <cell line: 0>()
      4 options.experimental_deterministic = True
      5 
----> 6 test_files = np.array([os.path.join(test_dir, f) for f in images_path_list], dtype=str)
      7 test_names = np.array(images_path_list, dtype=str)
      8 

NameError: name 'images_path_list' is not defined
