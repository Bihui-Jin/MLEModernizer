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
import random
import numpy as np
import pandas as pd

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

tf.config.run_functions_eagerly(False)
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("tf.keras:", keras.__version__)



## === cell 2
sam_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")



## === cell 3
pass



## === cell 4
train_dir = "/kaggle/input/plant-pathology-2021-fgvc8/train_images"
test_dir = "/kaggle/input/plant-pathology-2021-fgvc8/test_images"



## === cell 5
train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")



## === cell 6
test_df = sam_sub[["image"]].copy()



## === cell 7
all_labels = sorted(
    {lab for s in train["labels"].astype(str).values for lab in s.split(" ") if lab}
)
n_classes = len(all_labels)
print("Num classes:", n_classes)
print("Classes:", all_labels)

label2idx = {l: i for i, l in enumerate(all_labels)}

labels_list = train["labels"].astype(str).str.split(" ")
row_idx = np.repeat(
    np.arange(len(train), dtype=np.int32), labels_list.map(len).to_numpy()
)
labs_flat = np.concatenate(labels_list.to_numpy(), axis=0)
valid_mask = labs_flat != ""
row_idx = row_idx[valid_mask]
labs_flat = labs_flat[valid_mask]

col_idx = np.fromiter(
    (label2idx.get(l, -1) for l in labs_flat), dtype=np.int32, count=len(labs_flat)
)
ok = col_idx >= 0
row_idx = row_idx[ok]
col_idx = col_idx[ok]

Y = np.zeros((len(train), n_classes), dtype=np.float32)
Y[row_idx, col_idx] = 1.0

y_cols = [f"y_{l}" for l in all_labels]
train_y = pd.DataFrame(Y, columns=y_cols)
train_ml = pd.concat([train[["image", "labels"]].copy(), train_y], axis=1)



## === cell 8
IMG_SIZE = (432, 648)  # (height, width) for tf.image.resize
BATCH_SIZE = 16
TEST_BATCH_SIZE = 32
VAL_SPLIT = 0.1

n_total = len(train_ml)
n_val = int(np.floor(n_total * VAL_SPLIT))
n_train = n_total - n_val
train_ml = train_ml.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
train_ml_train = train_ml.iloc[:n_train].reset_index(drop=True)
train_ml_val = train_ml.iloc[n_train:].reset_index(drop=True)

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_from_bytes(img_bytes):
    img = tf.image.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize(
        img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


data_augmentation = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(20 / 360.0, fill_mode="reflect", seed=SEED),
        layers.RandomTranslation(0.1, 0.1, fill_mode="reflect", seed=SEED),
    ],
    name="data_augmentation",
)


@tf.function
def _path_to_train_pair(path, y):
    img_bytes = tf.io.read_file(path)
    x = _decode_resize_from_bytes(img_bytes)
    x = data_augmentation(x, training=True)
    return x, y


@tf.function
def _path_to_valid_pair(path, y):
    img_bytes = tf.io.read_file(path)
    x = _decode_resize_from_bytes(img_bytes)
    return x, y


@tf.function
def _path_to_image(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_from_bytes(img_bytes)


def _ds_options():
    opts = tf.data.Options()
    opts.experimental_deterministic = True
    try:
        opts.experimental_optimization.apply_default_optimizations = True
        opts.experimental_optimization.map_parallelization = True
        opts.experimental_optimization.parallel_batch = True
        opts.experimental_optimization.autotune_buffers = True
    except Exception:
        pass
    return opts


def make_train_ds(df):
    paths_np = (train_dir + "/" + df["image"].to_numpy()).astype("U")
    labels_np = df[y_cols].to_numpy(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(
        _ds_options()
    )

    buf = int(min(len(df), 4096))
    ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(_path_to_train_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths_np = (train_dir + "/" + df["image"].to_numpy()).astype("U")
    labels_np = df[y_cols].to_numpy(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(
        _ds_options()
    )

    ds = ds.map(_path_to_valid_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths_np = (test_dir + "/" + df["image"].to_numpy()).astype("U")
    ds = tf.data.Dataset.from_tensor_slices(paths_np).with_options(_ds_options())

    ds = ds.map(_path_to_image, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(TEST_BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_train_ds(train_ml_train)
valid_ds = make_valid_ds(train_ml_val)
test_ds = make_test_ds(test_df)

ordered_test_images = test_df["image"].tolist()

FIT_KW = {}
PRED_KW = {}



## === cell 9
MODEL_DIR = "/kaggle/input/effnet5/pp21_effnet_sub5"


def build_fallback_model(input_shape=(432, 648, 3), n_out=12):
    inputs = keras.Input(shape=input_shape, name="image")
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPool2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(n_out, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
    )
    return model


trained_model_sub = None
used_external_model = False

if os.path.exists(MODEL_DIR):
    try:
        trained_model_sub = keras.models.load_model(MODEL_DIR)
        used_external_model = True
        print("Loaded external model via keras.models.load_model from:", MODEL_DIR)
    except Exception as e1:
        try:
            if os.path.exists(
                os.path.join(MODEL_DIR, "saved_model.pb")
            ) or os.path.exists(os.path.join(MODEL_DIR, "saved_model.pbtxt")):
                tfsml = layers.TFSMLayer(MODEL_DIR, call_endpoint="serving_default")
                inp = keras.Input(shape=(432, 648, 3), name="image")
                out = tfsml(inp)
                if isinstance(out, dict):
                    out = out[list(out.keys())[0]]
                trained_model_sub = keras.Model(inputs=inp, outputs=out)
                used_external_model = True
                print("Loaded external SavedModel via TFSMLayer from:", MODEL_DIR)
        except Exception as e2:
            print(
                "External model load failed, falling back. Errors:", repr(e1), repr(e2)
            )

if trained_model_sub is None:
    trained_model_sub = build_fallback_model((432, 648, 3), n_classes)
    print("Training fallback model...")
    EPOCHS = 2
    trained_model_sub.fit(
        train_ds,
        validation_data=valid_ds,
        epochs=EPOCHS,
        verbose=1,
        **FIT_KW,
    )



## === cell 10
y_pred = trained_model_sub.predict(test_ds, verbose=1, **PRED_KW)

y_pred = np.asarray(y_pred)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)
print("y_pred shape:", y_pred.shape)



## === cell 11
THRESH = 0.5

mask = y_pred >= THRESH
all_labels_arr = np.array(all_labels, dtype=object)

rows = [np.flatnonzero(r) for r in mask]
pred_labels = [
    "healthy" if idxs.size == 0 else " ".join(all_labels_arr[idxs]) for idxs in rows
]



## === cell 12
sub_pred = pd.DataFrame({"image": ordered_test_images, "labels": pred_labels})

sub = sam_sub[["image"]].merge(sub_pred, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")
sub = sub[["image", "labels"]]



## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## === cell 14
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(sam_sub)
print("Submission sanity check passed.")
