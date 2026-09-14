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
import tensorflow as tf

print("Python:", sys.version)
print("TensorFlow:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 1337
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

output_dir = "./"
test_dir = "../input/plant-pathology-2021-fgvc8/test_images/"
train_dir = "../input/plant-pathology-2021-fgvc8/train_images/"
train_csv_path = "../input/plant-pathology-2021-fgvc8/train.csv"
sample_sub_path = "../input/plant-pathology-2021-fgvc8/sample_submission.csv"

model_dir = "../input/model-effb7-01/epoch-5"

local_savedmodel_dir = os.path.join(output_dir, "saved_model_pp2021")
local_keras_path = os.path.join(output_dir, "model_pp2021.keras")

image_dims = (300, 300, 3)
IMG_SIZE = (image_dims[0], image_dims[1])

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].astype(str)
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()

print("Num classes:", len(dataset_labels))
print("Example classes:", dataset_labels[:10])




## === cell 1
def _find_savedmodel_dir(path: str) -> str:
    """
    Searches for a valid SavedModel directory under `path`.
    """
    path = os.path.expanduser(path)

    def is_savedmodel_dir(d: str) -> bool:
        return os.path.isdir(d) and (
            os.path.isfile(os.path.join(d, "saved_model.pb"))
            or os.path.isfile(os.path.join(d, "saved_model.pbtxt"))
        )

    if is_savedmodel_dir(path):
        return path

    if not os.path.exists(path):
        raise OSError(f"Path does not exist: {path}")

    for sub in ["saved_model", "SavedModel", "export", "model", "1"]:
        cand = os.path.join(path, sub)
        if is_savedmodel_dir(cand):
            return cand

    for root, dirs, files in os.walk(path):
        if "saved_model.pb" in files or "saved_model.pbtxt" in files:
            return root
        rel_depth = os.path.relpath(root, path).count(os.sep)
        if (
            rel_depth >= 4
        ):  # was 5; still safely bounded, and avoids deep scans on large inputs
            dirs[:] = []

    raise OSError(
        f"SavedModel file does not exist under: {path}. "
        f"Expected a directory containing saved_model.pb (or pbtxt)."
    )


def _resolve_or_train_model() -> str:
    """
    Bug fix + robustness:
    - If the provided model_dir isn't available, fall back to a local SavedModel if present.
    - Otherwise train a small tf.keras model from train_images/ and save as SavedModel for inference.
    Returns: a directory containing a TensorFlow SavedModel.
    """
    try:
        return _find_savedmodel_dir(model_dir)
    except Exception as e:
        print("Could not use external model_dir:", repr(e))

    try:
        return _find_savedmodel_dir(local_savedmodel_dir)
    except Exception as e:
        print("No local SavedModel found:", repr(e))

    print("Training fallback model because no SavedModel was found...")

    y = data_set["labels"].astype(str).str.get_dummies(sep=" ").astype("float32").values
    filepaths = (
        data_set["image"].astype(str).apply(lambda x: os.path.join(train_dir, x)).values
    )

    idx = np.arange(len(filepaths))
    rng = np.random.default_rng(SEED)
    rng.shuffle(idx)
    split = int(len(idx) * 0.9)
    tr_idx, va_idx = idx[:split], idx[split:]

    tr_files, va_files = filepaths[tr_idx], filepaths[va_idx]
    tr_y, va_y = y[tr_idx], y[va_idx]

    def _load_image(path):
        img_bytes = tf.io.read_file(path)
        img = tf.io.decode_jpeg(img_bytes, channels=3)
        img = tf.image.convert_image_dtype(img, tf.float32)  # [0,1]
        img = tf.image.resize(img, IMG_SIZE)
        img = img * 255.0
        return img

    def _make_ds(files, labels, training):
        ds = tf.data.Dataset.from_tensor_slices((files, labels))
        if training:
            ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

        def _map_fn(p, lab):
            img = _load_image(p)
            if training:
                img = tf.image.random_flip_left_right(img, seed=SEED)
                img = tf.image.random_flip_up_down(img, seed=SEED)
            return img, lab

        ds = ds.map(_map_fn, num_parallel_calls=tf.data.AUTOTUNE)
        ds = ds.batch(16).prefetch(tf.data.AUTOTUNE)
        return ds

    train_ds = _make_ds(tr_files, tr_y, training=True)
    val_ds = _make_ds(va_files, va_y, training=False)

    inputs = tf.keras.Input(shape=image_dims, name="image")
    base = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_tensor=inputs,
        pooling="avg",
    )
    x = base.output
    x = tf.keras.layers.Dropout(0.3)(x)
    outputs = tf.keras.layers.Dense(
        len(dataset_labels), activation="sigmoid", name="pred"
    )(x)
    model = tf.keras.Model(inputs=inputs, outputs=outputs)

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
        loss=tf.keras.losses.BinaryCrossentropy(from_logits=False),
    )

    model.fit(train_ds, validation_data=val_ds, epochs=3, verbose=2)

    model.save(local_keras_path)
    tf.saved_model.save(model, local_savedmodel_dir)
    print("Saved fallback SavedModel to:", local_savedmodel_dir)
    return local_savedmodel_dir




## === cell 2
resolved_model_dir = _resolve_or_train_model()
print("Resolved model dir:", resolved_model_dir)

sm_layer = tf.keras.layers.TFSMLayer(
    resolved_model_dir, call_endpoint="serving_default"
)

dummy = tf.zeros((1, image_dims[0], image_dims[1], image_dims[2]), dtype=tf.float32)
dummy_out = sm_layer(dummy)
if isinstance(dummy_out, dict):
    out_key = sorted(dummy_out.keys())[0]
    print("SavedModel output is dict. Using key:", out_key)

    def predict_fn(x):
        return sm_layer(x)[out_key]

else:
    print("SavedModel output is tensor.")

    def predict_fn(x):
        return sm_layer(x)


images_path_list = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)
threshold = 0.7  # preserve core semantics


def _load_and_preprocess_for_infer(fname):
    full_path = tf.strings.join([tf.constant(test_dir), fname])
    img_bytes = tf.io.read_file(full_path)
    img = tf.io.decode_image(
        img_bytes, channels=3, dtype=tf.dtypes.float32, expand_animations=False
    )
    img.set_shape([None, None, 3])
    img = tf.image.resize(img, [image_dims[0], image_dims[1]])
    img = img * 255.0
    return fname, img


BATCH_SIZE = 64

test_ds = tf.data.Dataset.from_tensor_slices(tf.constant(images_path_list))
test_ds = test_ds.map(
    _load_and_preprocess_for_infer, num_parallel_calls=tf.data.AUTOTUNE
)
test_ds = test_ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

values = []
n_classes = len(dataset_labels)

for batch_names, batch_imgs in test_ds:
    preds = predict_fn(batch_imgs)
    preds = tf.convert_to_tensor(preds).numpy()  # [B, C] (or compatible)
    m = min(preds.shape[-1], n_classes)
    preds = preds[:, :m]

    above = preds > threshold  # bool [B, m]

    batch_names = batch_names.numpy().astype(str).tolist()
    for i, name in enumerate(batch_names):
        idxs = np.flatnonzero(above[i]).tolist()
        classes_img_list = [dataset_labels[j] for j in idxs]
        if len(classes_img_list) == 0:
            classes_img_list = ["healthy"]
        values.append([name, " ".join(classes_img_list).strip()])

csv_pd = pd.DataFrame(values, columns=["image", "labels"])

sample_sub = pd.read_csv(sample_sub_path)
csv_pd = sample_sub[["image"]].merge(csv_pd, on="image", how="left")
csv_pd["labels"] = csv_pd["labels"].fillna("healthy")

csv_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(csv_path, index=False)
print("Wrote:", csv_path, "rows:", len(csv_pd))
print(csv_pd.head())
