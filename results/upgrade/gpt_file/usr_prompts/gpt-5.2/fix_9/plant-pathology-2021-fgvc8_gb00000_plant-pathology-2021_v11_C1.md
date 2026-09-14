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

No external packages required in the script and installed.

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
import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("PYTHONHASHSEED", "42")
np.random.seed(42)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import backend as K

tf.random.set_seed(42)

print("TensorFlow:", tf.__version__)
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## === cell 2
from sklearn.model_selection import train_test_split



## === cell 3
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")
sam_sub.head()



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
train.head()



## === cell 6
test_ids = sorted([f for f in os.listdir(test_dir) if f.lower().endswith(".jpg")])
test_df = pd.DataFrame(test_ids, columns=["image"])
test_df.head()



## === cell 7
all_labels = sorted(
    {lab for s in train["labels"].fillna("").values for lab in s.split(" ") if lab}
)
label2idx = {lab: i for i, lab in enumerate(all_labels)}
idx2label = {i: lab for lab, i in label2idx.items()}
num_classes = len(all_labels)

rows = []
cols = []
for i, s in enumerate(train["labels"].fillna("").values):
    if s:
        for lab in s.split(" "):
            j = label2idx.get(lab)
            if j is not None:
                rows.append(i)
                cols.append(j)

y = np.zeros((len(train), num_classes), dtype=np.float32)
if rows:
    y[np.array(rows, dtype=np.int32), np.array(cols, dtype=np.int32)] = 1.0

y_cols = [f"y_{lab}" for lab in all_labels]

train_ml = train[["image"]].copy()
for j, col in enumerate(y_cols):
    train_ml[col] = y[:, j]

train_ml.head()



## === cell 8
AUTOTUNE = tf.data.AUTOTUNE
IMG_SIZE = (224, 336)
BATCH_SIZE = 16

train_df, val_df = train_test_split(
    train_ml, test_size=0.15, random_state=42, shuffle=True
)

options = tf.data.Options()
options.deterministic = True
options.experimental_slack = True


@tf.function
def _read_bytes(path):
    return tf.io.read_file(path)


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3)
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=True
    )
    return img


@tf.function
def _decode_resize_with_label_from_bytes(img_bytes, label):
    img = _decode_resize_from_bytes(img_bytes)
    return img, label


@tf.function
def _decode_resize_from_bytes_only(img_bytes):
    return _decode_resize_from_bytes(img_bytes)


all_train_paths = tf.constant(
    [os.path.join(train_dir, f) for f in train_ml["image"].values], dtype=tf.string
)

all_train_bytes_ds = (
    tf.data.Dataset.from_tensor_slices(all_train_paths)
    .with_options(options)
    .map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=True)
    .cache()  # caches compressed bytes (much smaller than resized float images)
)

all_train_bytes_tensor = tf.data.experimental.get_single_element(
    all_train_bytes_ds.batch(len(train_ml), drop_remainder=False)
)


def _bytes_ds_for_df(df):
    idx = tf.constant(df.index.values.astype(np.int32))
    gathered = tf.gather(all_train_bytes_tensor, idx)
    return tf.data.Dataset.from_tensor_slices(gathered)


def _make_train_ds(df):
    labels = tf.constant(df[y_cols].values.astype(np.float32))
    bytes_ds = _bytes_ds_for_df(df)
    ds = tf.data.Dataset.zip((bytes_ds, tf.data.Dataset.from_tensor_slices(labels)))
    ds = ds.with_options(options)
    ds = ds.map(
        _decode_resize_with_label_from_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.shuffle(buffer_size=len(df), seed=42, reshuffle_each_iteration=True)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def _make_val_ds(df):
    labels = tf.constant(df[y_cols].values.astype(np.float32))
    bytes_ds = _bytes_ds_for_df(df)
    ds = tf.data.Dataset.zip((bytes_ds, tf.data.Dataset.from_tensor_slices(labels)))
    ds = ds.with_options(options)
    ds = ds.map(
        _decode_resize_with_label_from_bytes,
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_generator = _make_train_ds(train_df)
val_generator = _make_val_ds(val_df)




## === cell 9
def _make_test_ds(df):
    paths = tf.constant(
        [os.path.join(test_dir, f) for f in df["image"].values], dtype=tf.string
    )
    ds = tf.data.Dataset.from_tensor_slices(paths).with_options(options)
    ds = ds.map(_read_bytes, num_parallel_calls=AUTOTUNE, deterministic=True).cache()
    ds = ds.map(
        _decode_resize_from_bytes_only, num_parallel_calls=AUTOTUNE, deterministic=True
    )
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


test_generator = _make_test_ds(test_df)



## === cell 10
data_aug = keras.Sequential(
    [
        keras.layers.RandomFlip("horizontal", seed=42),
        keras.layers.RandomRotation(factor=20.0 / 360.0, fill_mode="reflect", seed=42),
        keras.layers.RandomTranslation(
            height_factor=0.1, width_factor=0.1, fill_mode="reflect", seed=42
        ),
    ],
    name="data_augmentation",
)

base = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 336, 3),
    pooling="avg",
)

inp = keras.Input(shape=(224, 336, 3), name="image")
x = data_aug(inp)
x = keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
trained_model_sub = keras.Model(inp, out)

base.trainable = False
for layer in trained_model_sub.layers:
    if isinstance(layer, keras.layers.Dense):
        layer.trainable = True

trained_model_sub.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

trained_model_sub.summary()



## === cell 11
EPOCHS = 3
_ = trained_model_sub.fit(
    train_generator,
    validation_data=val_generator,
    epochs=EPOCHS,
    verbose=1,
)



## === cell 12
y_pred = trained_model_sub.predict(test_generator, verbose=1)
y_pred = np.asarray(y_pred)
print("y_pred shape:", y_pred.shape)

threshold = 0.5

mask = y_pred >= threshold
any_pos = mask.any(axis=1)
argmax_idx = y_pred.argmax(axis=1)

pred_label_strs = []
for i in range(y_pred.shape[0]):
    inds = np.flatnonzero(mask[i]).tolist()
    if not any_pos[i]:
        inds = [int(argmax_idx[i])]
    pred_label_strs.append(" ".join(idx2label[j] for j in inds))

gen_images = test_df["image"].tolist()

sub = pd.DataFrame({"image": gen_images, "labels": pred_label_strs})
sub.head()



## === cell 13
assert list(sub.columns) == ["image", "labels"]
assert sub["image"].nunique() == len(sub)
assert sub["labels"].isna().sum() == 0

sub = sub.set_index("image").reindex(sam_sub["image"]).reset_index()

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
