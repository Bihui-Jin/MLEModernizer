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
import glob
import random
import pandas as pd
import numpy as np
import tensorflow as tf

print("Python:", sys.version.split()[0])
print("TensorFlow:", tf.__version__)




## === cell 1
output_dir = "./"

CANDIDATE_ROOTS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input",
    "/kaggle/input",
    "../data/plant-pathology-2021-fgvc8",
    "/kaggle/data/plant-pathology-2021-fgvc8",
]
comp_root = None
for r in CANDIDATE_ROOTS:
    if os.path.exists(os.path.join(r, "train.csv")) and os.path.isdir(
        os.path.join(r, "train_images")
    ):
        comp_root = r
        break
if comp_root is None:
    for r in ["../input", "/kaggle/input", "../data", "/kaggle/data"]:
        hits = glob.glob(
            os.path.join(r, "**", "plant-pathology-2021-fgvc8", "train.csv"),
            recursive=True,
        )
        if hits:
            comp_root = os.path.dirname(hits[0])
            break

if comp_root is None:
    raise FileNotFoundError(
        "Could not locate competition directory containing train.csv and train_images."
    )

train_csv_path = os.path.join(comp_root, "train.csv")
sample_sub_path = os.path.join(comp_root, "sample_submission.csv")
train_dir = os.path.join(comp_root, "train_images")
test_dir = os.path.join(comp_root, "test_images")

print("Using comp_root:", comp_root)
print("Train CSV:", train_csv_path)
print("Sample submission:", sample_sub_path)
print("Train dir:", train_dir, "exists:", os.path.isdir(train_dir))
print("Test dir:", test_dir, "exists:", os.path.isdir(test_dir))

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)

one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = list(one_hot.columns.to_list())

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])

class_prevalence = one_hot.mean(axis=0).values.astype(np.float32)

one_hot_full = one_hot.reindex(columns=dataset_labels, fill_value=0).values.astype(
    np.float32
)




## === cell 2
def _find_model_path():
    """
    Try to find a model artifact. Original code expected:
      ../input/model-to-load/model_complete
    which is not present in this environment.
    We search common places; if none found, return None.
    """
    candidates = []

    candidates.append("../input/model-to-load/model_complete")

    candidates += glob.glob("../input/**/model_complete", recursive=True)
    candidates += glob.glob("/kaggle/input/**/model_complete", recursive=True)

    candidates += glob.glob("../input/**/saved_model", recursive=True)
    candidates += glob.glob("/kaggle/input/**/saved_model", recursive=True)

    candidates += glob.glob("../input/**/*.keras", recursive=True)
    candidates += glob.glob("/kaggle/input/**/*.keras", recursive=True)
    candidates += glob.glob("../input/**/*.h5", recursive=True)
    candidates += glob.glob("/kaggle/input/**/*.h5", recursive=True)

    def is_savedmodel_dir(p):
        return os.path.isdir(p) and (
            os.path.exists(os.path.join(p, "saved_model.pb"))
            or os.path.exists(os.path.join(p, "saved_model.pbtxt"))
        )

    for p in candidates:
        if p.endswith((".keras", ".h5")) and os.path.isfile(p):
            return p
        if is_savedmodel_dir(p):
            return p
    return None


def _load_model(model_path: str):
    """
    Load either a Keras model (.keras/.h5) or a SavedModel directory via TFSMLayer.
    Returns a tf.keras.Model that takes float32 images of shape image_dims and outputs logits/probabilities.
    """
    if model_path.endswith((".keras", ".h5")):
        m = tf.keras.models.load_model(model_path, compile=False)
        return m

    try:
        layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint="serving_default")
    except Exception:
        loaded = tf.saved_model.load(model_path)
        sig_keys = list(getattr(loaded, "signatures", {}).keys())
        if not sig_keys:
            raise RuntimeError(
                f"SavedModel at {model_path} has no signatures; cannot infer call endpoint."
            )
        layer = tf.keras.layers.TFSMLayer(model_path, call_endpoint=sig_keys[0])

    inp = tf.keras.Input(shape=image_dims, dtype=tf.float32)
    out = layer(inp)

    if isinstance(out, dict):
        out = out[sorted(out.keys())[0]]
    return tf.keras.Model(inputs=inp, outputs=out)


model_path = _find_model_path()
print("Detected model_path:", model_path)

model = None
if model_path is not None:
    try:
        model = _load_model(model_path)
        _ = model(tf.zeros((1,) + image_dims, dtype=tf.float32), training=False)
        print("Model loaded successfully.")
    except Exception as e:
        print(
            "Model load failed; will use trainable fallback baseline. Error:", repr(e)
        )
        model = None
else:
    print("No model artifact found; will use trainable fallback baseline.")




## === cell 3
SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.data.experimental.enable_debug_mode(False)
except Exception:
    pass

options = tf.data.Options()
options.experimental_deterministic = True
try:
    options.experimental_optimization.parallel_batch = True
except Exception:
    pass

if os.path.isdir(test_dir):
    disk_images = sorted(
        [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
    )
else:
    disk_images = []

if os.path.exists(sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    images_path_list = sample_sub["image"].astype(str).tolist()
else:
    images_path_list = disk_images

if not images_path_list:
    raise FileNotFoundError(
        "No test images found and sample_submission.csv is missing/empty."
    )

print("Num test images (from sample_submission if present):", len(images_path_list))
print("Example test image ids:", images_path_list[:5])

FALLBACK_BACKBONE = "efficientnetb0"  # minimal, fast
NUM_CLASSES = len(dataset_labels)


def _preprocess_for_backbone(x):
    x = tf.clip_by_value(x, 0.0, 1.0)
    x = x * 255.0
    x = tf.keras.applications.efficientnet.preprocess_input(x)
    return x


def _decode_resize_preproc_jpeg(path, for_backbone: bool):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1], float32
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    if for_backbone:
        img = _preprocess_for_backbone(img)
    return img


def load_and_preprocess_image(image_name: str, for_backbone: bool = False):
    img_path = os.path.join(test_dir, image_name)
    image = _decode_resize_preproc_jpeg(img_path, for_backbone)
    return tf.expand_dims(image, axis=0)


def _make_train_ds(df, batch_size=32, shuffle=True, cache_path=None):
    image_names = df["image"].astype(str).to_numpy()
    image_paths = np.array([os.path.join(train_dir, n) for n in image_names], dtype=str)

    y = one_hot_full[df.index.to_numpy()]

    path_ds = tf.data.Dataset.from_tensor_slices(image_paths)
    y_ds = tf.data.Dataset.from_tensor_slices(y)
    ds = tf.data.Dataset.zip((path_ds, y_ds))

    def _load(path, yvec):
        img = _decode_resize_preproc_jpeg(path, True)
        return img, yvec

    if shuffle:
        ds = ds.shuffle(min(len(df), 8192), seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_load, num_parallel_calls=tf.data.AUTOTUNE)

    if cache_path is not None:
        ds = ds.cache(cache_path)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    ds = ds.with_options(options)
    return ds


if model is None:
    idx = np.arange(len(data_set))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(0.9 * len(idx))
    tr_idx, va_idx = idx[:split], idx[split:]
    df_tr = data_set.iloc[tr_idx]
    df_va = data_set.iloc[va_idx]

    train_cache = os.path.join(output_dir, "train_ds_cache.tf-data")
    val_cache = os.path.join(output_dir, "val_ds_cache.tf-data")

    train_ds = _make_train_ds(
        df_tr, batch_size=32, shuffle=True, cache_path=train_cache
    )
    val_ds = _make_train_ds(df_va, batch_size=32, shuffle=False, cache_path=val_cache)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=image_dims,
        pooling="avg",
    )
    backbone.trainable = False

    inp = tf.keras.Input(shape=image_dims, dtype=tf.float32)
    x = backbone(inp, training=False)
    x = tf.keras.layers.Dropout(0.2)(x)
    out = tf.keras.layers.Dense(NUM_CLASSES, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss=tf.keras.losses.BinaryCrossentropy(),
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    model._uses_backbone_preproc = True
else:
    model._uses_backbone_preproc = False




## === cell 4
values = []
threshold = 0.5  # preserve original decision threshold

fallback_indices = np.where(class_prevalence > threshold)[0].tolist()

if model is not None:
    use_backbone_preproc = bool(getattr(model, "_uses_backbone_preproc", False))

    def _test_load(image_name):
        path = tf.strings.join([test_dir, "/", image_name])
        img = _decode_resize_preproc_jpeg(path, use_backbone_preproc)
        return img, image_name

    test_names = tf.convert_to_tensor(images_path_list, dtype=tf.string)
    test_ds = tf.data.Dataset.from_tensor_slices(test_names)
    test_ds = test_ds.map(_test_load, num_parallel_calls=tf.data.AUTOTUNE)
    test_ds = test_ds.batch(64, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    test_ds = test_ds.with_options(options)

    pred_mat = model.predict(test_ds.map(lambda x, n: x), verbose=0)
    if isinstance(pred_mat, dict):
        pred_mat = pred_mat[sorted(pred_mat.keys())[0]]
    pred_mat = np.asarray(pred_mat, dtype=np.float32)

    mask = pred_mat > threshold
    for name, row_mask in zip(images_path_list, mask):
        idxs = np.flatnonzero(row_mask)
        if idxs.size == 0:
            classes_img = "healthy"
        else:
            idxs = idxs[idxs < len(dataset_labels)]
            classes_img = (
                "healthy"
                if idxs.size == 0
                else " ".join(dataset_labels[i] for i in idxs.tolist())
            )
        values.append([name, classes_img])
else:
    for name in images_path_list:
        index_values = fallback_indices
        if len(index_values) == 0:
            classes_img = "healthy"
        else:
            index_values = [i for i in index_values if i < len(dataset_labels)]
            classes_img = (
                "healthy"
                if len(index_values) == 0
                else " ".join([dataset_labels[i] for i in index_values])
            )
        values.append([name, classes_img])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])

csv_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(csv_path, index=False)

print("Wrote:", csv_path)
print(csv_pd.head())
print("Submission shape:", csv_pd.shape)
