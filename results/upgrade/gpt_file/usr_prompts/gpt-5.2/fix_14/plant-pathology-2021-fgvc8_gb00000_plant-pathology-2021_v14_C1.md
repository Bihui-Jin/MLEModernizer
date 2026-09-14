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

# 5. Target score

0.7458541089566028

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

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



## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
lens = labels_list.map(len).to_numpy(np.int32, copy=False)
row_idx = np.repeat(np.arange(len(train), dtype=np.int32), lens)
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
def _path_to_image_decoded(path):
    img_bytes = tf.io.read_file(path)
    return _decode_resize_from_bytes(img_bytes)


@tf.function
def _augment_pair(x, y):
    x = data_augmentation(x, training=True)
    return x, y


@tf.function
def _path_to_valid_pair(path, y):
    x = _path_to_image_decoded(path)
    return x, y


@tf.function
def _path_to_image(path):
    return _path_to_image_decoded(path)


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
    paths_np = train_dir + "/" + df["image"].to_numpy(dtype=str, copy=False)
    labels_np = df[y_cols].to_numpy(np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(
        _ds_options()
    )

    buf = int(min(len(df), 4096))
    ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    ds = ds.map(
        lambda p, y: (_path_to_image_decoded(p), y), num_parallel_calls=AUTOTUNE
    )

    cache_path = "/kaggle/working/train_decode_cache"
    ds = ds.cache(cache_path)

    ds = ds.map(_augment_pair, num_parallel_calls=AUTOTUNE)
    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_valid_ds(df):
    paths_np = train_dir + "/" + df["image"].to_numpy(dtype=str, copy=False)
    labels_np = df[y_cols].to_numpy(np.float32, copy=False)

    ds = tf.data.Dataset.from_tensor_slices((paths_np, labels_np)).with_options(
        _ds_options()
    )
    ds = ds.map(_path_to_valid_pair, num_parallel_calls=AUTOTUNE)

    ds = ds.cache("/kaggle/working/valid_decode_cache")

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def make_test_ds(df):
    paths_np = test_dir + "/" + df["image"].to_numpy(dtype=str, copy=False)
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



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
UFuncTypeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2791668433.py in <cell line: 0>()
    124 
    125 
--> 126 train_ds = make_train_ds(train_ml_train)
    127 valid_ds = make_valid_ds(train_ml_val)
    128 test_ds = make_test_ds(test_df)

/tmp/ipykernel_11/2791668433.py in make_train_ds(df)
     74 def make_train_ds(df):
     75     # CHANGE (timeout): avoid extra dtype conversions; build paths once; cache decoded tensors to disk then augment.
---> 76     paths_np = train_dir + "/" + df["image"].to_numpy(dtype=str, copy=False)
     77     labels_np = df[y_cols].to_numpy(np.float32, copy=False)
     78 

UFuncTypeError: ufunc 'add' did not contain a loop with signature matching types (dtype('<U54'), dtype('<U20')) -> None

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


def _looks_like_saved_model(d: str) -> bool:
    return os.path.exists(os.path.join(d, "saved_model.pb")) or os.path.exists(
        os.path.join(d, "saved_model.pbtxt")
    )


def _iter_likely_model_dirs(base_dir: str):
    yield base_dir
    try:
        with os.scandir(base_dir) as it:
            for ent in it:
                if ent.is_dir(follow_symlinks=False):
                    yield ent.path
    except Exception:
        return


def _try_load_any_model(model_dir: str):
    last_err = None

    for d in _iter_likely_model_dirs(model_dir):
        if not _looks_like_saved_model(d):
            continue
        try:
            m = keras.models.load_model(d)
            return m, True, f"Loaded via keras.models.load_model from: {d}"
        except Exception as e:
            last_err = e

    for d in _iter_likely_model_dirs(model_dir):
        if not _looks_like_saved_model(d):
            continue
        try:
            tfsml = layers.TFSMLayer(d, call_endpoint="serving_default")
            inp = keras.Input(shape=(432, 648, 3), name="image")
            out = tfsml(inp)
            if isinstance(out, dict):
                out = out[list(out.keys())[0]]
            m = keras.Model(inputs=inp, outputs=out)
            return m, True, f"Loaded external SavedModel via TFSMLayer from: {d}"
        except Exception as e:
            last_err = e

    raise RuntimeError(
        f"Could not load model from {model_dir}. Last error: {last_err!r}"
    )


trained_model_sub = None
used_external_model = False

if os.path.exists(MODEL_DIR):
    try:
        trained_model_sub, used_external_model, msg = _try_load_any_model(MODEL_DIR)
        print(msg)
    except Exception as e:
        print("External model load failed, falling back. Error:", repr(e))

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



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1615102028.py in <cell line: 0>()
     89     EPOCHS = 2
     90     trained_model_sub.fit(
---> 91         train_ds,
     92         validation_data=valid_ds,
     93         epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 10
y_pred = trained_model_sub.predict(test_ds, verbose=1, **PRED_KW)

y_pred = np.asarray(y_pred)
if y_pred.ndim == 1:
    y_pred = y_pred.reshape(-1, 1)
print("y_pred shape:", y_pred.shape)



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/972968956.py in <cell line: 0>()
----> 1 y_pred = trained_model_sub.predict(test_ds, verbose=1, **PRED_KW)
      2 
      3 y_pred = np.asarray(y_pred)
      4 if y_pred.ndim == 1:
      5     y_pred = y_pred.reshape(-1, 1)

NameError: name 'test_ds' is not defined

## === cell 11
THRESH = 0.5

mask = y_pred >= THRESH
all_labels_arr = np.array(all_labels, dtype=object)

rows = [np.flatnonzero(r) for r in mask]
pred_labels = [
    "healthy" if idxs.size == 0 else " ".join(all_labels_arr[idxs]) for idxs in rows
]



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/484471939.py in <cell line: 0>()
      1 THRESH = 0.5
      2 
----> 3 mask = y_pred >= THRESH
      4 all_labels_arr = np.array(all_labels, dtype=object)
      5 

NameError: name 'y_pred' is not defined

## === cell 12
sub_pred = pd.DataFrame({"image": ordered_test_images, "labels": pred_labels})

sub = sam_sub[["image"]].merge(sub_pred, on="image", how="left")
sub["labels"] = sub["labels"].fillna("healthy")
sub = sub[["image", "labels"]]



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1600929051.py in <cell line: 0>()
----> 1 sub_pred = pd.DataFrame({"image": ordered_test_images, "labels": pred_labels})
      2 
      3 sub = sam_sub[["image"]].merge(sub_pred, on="image", how="left")
      4 sub["labels"] = sub["labels"].fillna("healthy")
      5 sub = sub[["image", "labels"]]

NameError: name 'ordered_test_images' is not defined

## === cell 13
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3239440014.py in <cell line: 0>()
----> 1 sub.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", sub.shape)
      3 print(sub.head())
      4 

NameError: name 'sub' is not defined

## === cell 14
assert os.path.exists("submission.csv")
chk = pd.read_csv("submission.csv")
assert list(chk.columns) == ["image", "labels"]
assert len(chk) == len(sam_sub)
print("Submission sanity check passed.")

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
/tmp/ipykernel_11/2555127064.py in <cell line: 0>()
----> 1 assert os.path.exists("submission.csv")
      2 chk = pd.read_csv("submission.csv")
      3 assert list(chk.columns) == ["image", "labels"]
      4 assert len(chk) == len(sam_sub)
      5 print("Submission sanity check passed.")

AssertionError:
