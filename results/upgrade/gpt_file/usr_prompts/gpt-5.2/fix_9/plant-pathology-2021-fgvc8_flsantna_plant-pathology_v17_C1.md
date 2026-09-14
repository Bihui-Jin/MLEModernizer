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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TF:", tf.__version__)
print("Eager:", tf.executing_eagerly())

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF pick optimal
except Exception:
    pass




## === cell 1
output_dir = "./"

CANDIDATE_ROOTS = [
    "../input/plant-pathology-2021-fgvc8",
    "/kaggle/input/plant-pathology-2021-fgvc8",
    "../input",
    "/kaggle/input",
]


def _resolve_paths():
    for root in CANDIDATE_ROOTS:
        if os.path.isdir(root):
            direct_train = os.path.join(root, "train.csv")
            direct_test_dir = os.path.join(root, "test_images")
            if os.path.isfile(direct_train) and os.path.isdir(direct_test_dir):
                return direct_train, direct_test_dir, os.path.join(root, "train_images")

            nested = os.path.join(root, "plant-pathology-2021-fgvc8")
            nested_train = os.path.join(nested, "train.csv")
            nested_test_dir = os.path.join(nested, "test_images")
            if os.path.isfile(nested_train) and os.path.isdir(nested_test_dir):
                return (
                    nested_train,
                    nested_test_dir,
                    os.path.join(nested, "train_images"),
                )

    raise FileNotFoundError(
        "Could not resolve dataset paths. Checked: " + ", ".join(CANDIDATE_ROOTS)
    )


train_csv_path, test_dir, train_img_dir = _resolve_paths()

image_dims = (300, 300, 3)

data_set = pd.read_csv(train_csv_path)
df_labels = data_set["labels"].fillna("")
one_hot = df_labels.str.get_dummies(sep=" ")
dataset_labels = one_hot.columns.to_list()
num_classes = len(dataset_labels)

print("Resolved train_csv_path:", train_csv_path)
print("Resolved train_img_dir:", train_img_dir)
print("Resolved test_dir:", test_dir)
print("Num classes:", num_classes)
print("Example classes:", dataset_labels[:10])




## === cell 2
base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=image_dims,
    pooling="avg",
)
base.trainable = False

inputs = tf.keras.Input(shape=image_dims, name="image")
x = tf.keras.applications.efficientnet.preprocess_input(inputs * 255.0)
x = base(x, training=False)
outputs = tf.keras.layers.Dense(num_classes, activation="sigmoid", name="pred")(x)
model = tf.keras.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)
model.summary()




## === cell 3
y = one_hot.reindex(columns=dataset_labels, fill_value=0).astype("float32").values
x_paths = data_set["image"].apply(lambda x: os.path.join(train_img_dir, x)).values

idx = np.arange(len(x_paths))
np.random.shuffle(idx)
val_size = int(0.1 * len(idx))
val_idx = idx[:val_size]
tr_idx = idx[val_size:]

x_tr, y_tr = x_paths[tr_idx], y[tr_idx]
x_va, y_va = x_paths[val_idx], y[val_idx]


@tf.function
def _decode_resize_float32_jpeg(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3)  # fixed JPEG fast-path
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(
        img,
        [image_dims[0], image_dims[1]],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=False,
    )
    img.set_shape([image_dims[0], image_dims[1], 3])
    return img


@tf.function
def load_and_preprocess(path, label):
    return _decode_resize_float32_jpeg(path), label


@tf.function
def preprocess_image_for_infer(path):
    return _decode_resize_float32_jpeg(path)


batch_size = 32

cache_dir = "/kaggle/working/tfdata_cache"
try:
    os.makedirs(cache_dir, exist_ok=True)
except Exception:
    pass
tr_cache = os.path.join(cache_dir, "train.cache")
va_cache = os.path.join(cache_dir, "val.cache")
te_cache = os.path.join(cache_dir, "test.cache")

options = tf.data.Options()
options.experimental_deterministic = True
options.experimental_optimization.apply_default_optimizations = True
options.experimental_optimization.map_parallelization = True

ds_tr = tf.data.Dataset.from_tensor_slices((x_tr, y_tr))
ds_tr = ds_tr.with_options(options)
ds_tr = ds_tr.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
ds_tr = ds_tr.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_tr = ds_tr.cache(tr_cache)
ds_tr = ds_tr.batch(batch_size, drop_remainder=False)
ds_tr = ds_tr.prefetch(tf.data.AUTOTUNE)

ds_va = tf.data.Dataset.from_tensor_slices((x_va, y_va))
ds_va = ds_va.with_options(options)
ds_va = ds_va.map(load_and_preprocess, num_parallel_calls=tf.data.AUTOTUNE)
ds_va = ds_va.cache(va_cache)
ds_va = ds_va.batch(batch_size, drop_remainder=False)
ds_va = ds_va.prefetch(tf.data.AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(ds_tr))
print("Val batches:", tf.data.experimental.cardinality(ds_va))




## === cell 4
epochs = 2
history = model.fit(ds_tr, validation_data=ds_va, epochs=epochs, verbose=2)

images_path_list = sorted(
    [f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")]
)
test_paths = [os.path.join(test_dir, f) for f in images_path_list]

te_options = tf.data.Options()
te_options.experimental_deterministic = True
te_options.experimental_optimization.apply_default_optimizations = True
te_options.experimental_optimization.map_parallelization = True

ds_te = tf.data.Dataset.from_tensor_slices(test_paths)
ds_te = ds_te.with_options(te_options)
ds_te = ds_te.map(preprocess_image_for_infer, num_parallel_calls=tf.data.AUTOTUNE)
ds_te = ds_te.cache(te_cache)
ds_te = ds_te.batch(32, drop_remainder=False).prefetch(tf.data.AUTOTUNE)

pred = model.predict(ds_te, verbose=0)  # shape: (N, num_classes)

thr = 0.6  # keep core behavior (thresholded multi-label, fallback to healthy)
mask = pred > thr  # (N, C) boolean
dataset_labels_arr = np.asarray(dataset_labels, dtype=object)

any_pos = mask.any(axis=1)
labels_out = np.full((mask.shape[0],), "healthy", dtype=object)
pos_rows = np.flatnonzero(any_pos)
for i in pos_rows:
    labels_out[i] = " ".join(dataset_labels_arr[mask[i]].tolist()).strip()

csv_pd = pd.DataFrame({"image": images_path_list, "labels": labels_out.tolist()})
sub_path = os.path.join(output_dir, "submission.csv")
csv_pd.to_csv(sub_path, index=False)

print("Wrote:", sub_path)
print("Rows:", len(csv_pd), "Cols:", list(csv_pd.columns))
print(csv_pd.head())
