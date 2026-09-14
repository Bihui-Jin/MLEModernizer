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

0.8017174515235485

# 6. Current score

0.24507

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.24507) has done: 'The timeout is dominated by the input pipeline repeatedly decoding/resizing JPEGs each epoch and by an unnecessary duplicate dataset build inside `make_dataset` (it constructs `ds` and then rebuilds `ds0`, doing extra work). I remove the duplicate dataset construction, keep a single cached decode/resize stage, and ensure caching happens before augmentation so images are decoded once and augmentations still vary per epoch—preserving training semantics. I also avoid materializing long Python lists of filepaths by using vectorized path joins and `tf.convert_to_tensor`, and I add `steps_per_execution` to reduce `tf.function` overhead during training without changing optimization logic. These changes keep the model, loss, epochs, augmentations, and thresholding logic the same while cutting input/CPU overhead significantly.'
- What this solution (achieved 0.24507) has done: 'Main bottlenecks are (1) decoding/resizing images every epoch and (2) the explicit Python loop building the multi-hot label matrix. To get under 600s without changing model/training logic, I (a) vectorize multi-label encoding with `MultiLabelBinarizer`-equivalent logic via pandas/numpy operations, (b) switch dataset caching to on-disk cache files (so it actually helps across epochs without risking RAM blowups/timeouts), and (c) keep deterministic tf.data ordering while reducing overhead by fusing maps where safe and ensuring prefetching is effective. Model architecture, epochs, loss, thresholds, and prediction semantics remain unchanged.'

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras as keras

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

AUTOTUNE = tf.data.AUTOTUNE

print("TF:", tf.__version__)
print("Keras:", keras.__version__)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(BASE_PATH, "train.csv")
SAMPLE_SUB = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train_images")
TEST_DIR = os.path.join(BASE_PATH, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()




## === cell 2
label_split = train["labels"].astype(str).str.split()

classes = sorted(pd.unique(label_split.explode()).tolist())
class_to_idx = {c: i for i, c in enumerate(classes)}

tmp = pd.DataFrame(
    {"row": np.arange(len(train)), "lab": label_split.explode()}
).dropna()
ct = pd.crosstab(tmp["row"], tmp["lab"]).reindex(columns=classes, fill_value=0)
labels_df = ct.astype(np.int8)

print("Classes:", classes)
labels_df.head()




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/590303937.py in <cell line: 0>()
      7 
      8 # Create a boolean indicator frame via explode + crosstab, then align to all classes.
----> 9 tmp = pd.DataFrame(
     10     {"row": np.arange(len(train)), "lab": label_split.explode()}
     11 ).dropna()

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    688                     f"length {len(index)}"
    689                 )
--> 690                 raise ValueError(msg)
    691         else:
    692             index = default_index(lengths[0])

ValueError: array length 14905 does not match index length 16153

## === cell 3
train_df = train.copy()
for c in classes:
    train_df[c] = labels_df[c].values

for c in classes:
    vc = train_df[c].value_counts(normalize=True)
    print(c, dict(vc))




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/375228460.py in <cell line: 0>()
      1 train_df = train.copy()
      2 for c in classes:
----> 3     train_df[c] = labels_df[c].values
      4 
      5 for c in classes:

NameError: name 'labels_df' is not defined

## === cell 4
h_target = 256
w_target = 256
batch_size = 32

val_frac = 0.1
train_df = train_df.sample(frac=1.0, random_state=SEED).reset_index(drop=True)
n_val = int(len(train_df) * val_frac)
val_df = train_df.iloc[:n_val].reset_index(drop=True)
trn_df = train_df.iloc[n_val:].reset_index(drop=True)

print("Train/Val:", trn_df.shape, val_df.shape)




## === cell 5
augment_layer = keras.Sequential(
    [
        keras.layers.RandomRotation(factor=10.0 / 360.0, seed=SEED),
        keras.layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, seed=SEED
        ),
        keras.layers.RandomZoom(
            height_factor=(-0.1, 0.1), width_factor=(-0.1, 0.1), seed=SEED
        ),
        keras.layers.RandomFlip(mode="horizontal", seed=SEED),
    ],
    name="augment",
)


def _decode_resize_rescale(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, [h_target, w_target], method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.cast(img, tf.float32) / 255.0
    return img


def make_dataset(df, directory, training, with_labels=True, cache_id=""):
    img_names = tf.convert_to_tensor(df["image"].astype(str).to_numpy())
    base = tf.constant(directory + os.sep)
    paths = tf.strings.join([base, img_names])

    options = tf.data.Options()
    options.deterministic = True  # preserve determinism requirement

    if with_labels:
        labels = tf.convert_to_tensor(df[classes].to_numpy(dtype=np.float32))
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    ds = ds.with_options(options)

    if training:
        buf = int(min(len(df), 4096))
        ds = ds.shuffle(buffer_size=buf, seed=SEED, reshuffle_each_iteration=True)

    cache_path = None
    if cache_id:
        cache_path = os.path.join("/kaggle/working", f"tfdata_cache_{cache_id}")

    if with_labels:
        if training:

            def _decode_map(p, y_):
                return _decode_resize_rescale(p), y_

            def _aug_map(x, y_):
                x = augment_layer(x, training=True)
                return x, y_

            ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
            ds = ds.cache(cache_path)
            ds = ds.map(_aug_map, num_parallel_calls=AUTOTUNE, deterministic=True)
        else:

            def _decode_map(p, y_):
                return _decode_resize_rescale(p), y_

            ds = ds.map(_decode_map, num_parallel_calls=AUTOTUNE, deterministic=True)
            ds = ds.cache(cache_path)
    else:

        def _decode_map_x(p):
            return _decode_resize_rescale(p)

        ds = ds.map(_decode_map_x, num_parallel_calls=AUTOTUNE, deterministic=True)
        ds = ds.cache(cache_path)

    ds = ds.apply(tf.data.experimental.ignore_errors())
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_dataset(
    trn_df, TRAIN_DIR, training=True, with_labels=True, cache_id="train_decode"
)
val_ds = make_dataset(
    val_df, TRAIN_DIR, training=False, with_labels=True, cache_id="val_decode"
)
test_ds = make_dataset(
    submissions, TEST_DIR, training=False, with_labels=False, cache_id="test_decode"
)

print("Datasets ready:", train_ds, val_ds, test_ds)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2010544358.py in <cell line: 0>()
     84 
     85 
---> 86 train_ds = make_dataset(
     87     trn_df, TRAIN_DIR, training=True, with_labels=True, cache_id="train_decode"
     88 )

/tmp/ipykernel_11/2010544358.py in make_dataset(df, directory, training, with_labels, cache_id)
     33 
     34     if with_labels:
---> 35         labels = tf.convert_to_tensor(df[classes].to_numpy(dtype=np.float32))
     36         ds = tf.data.Dataset.from_tensor_slices((paths, labels))
     37     else:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6247         if nmissing:
   6248             if nmissing == len(indexer):
-> 6249                 raise KeyError(f"None of [{key}] are in the [{axis_name}]")
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())

KeyError: "None of [Index(['complex', 'frog_eye_leaf_spot', 'healthy', 'powdery_mildew', 'rust',\n       'scab'],\n      dtype='object')] are in the [columns]"

## === cell 6
inputs = keras.Input(shape=(h_target, w_target, 3))

x = keras.layers.Conv2D(32, 3, padding="same", activation="relu")(inputs)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(64, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.Conv2D(128, 3, padding="same", activation="relu")(x)
x = keras.layers.MaxPooling2D()(x)

x = keras.layers.GlobalAveragePooling2D()(x)
x = keras.layers.Dropout(0.2)(x)

outputs = keras.layers.Dense(len(classes), activation="sigmoid")(x)

model = keras.Model(inputs, outputs)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
    steps_per_execution=32,
)
model.summary()




## === cell 7
epochs = 3  # unchanged
history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3641035886.py in <cell line: 0>()
      1 epochs = 3  # unchanged
----> 2 history = model.fit(train_ds, validation_data=val_ds, epochs=epochs, verbose=1)
      3 
      4 

NameError: name 'train_ds' is not defined

## === cell 8
preds = model.predict(test_ds, verbose=1)
print("preds shape:", preds.shape)
assert preds.shape[1] == len(classes), "Prediction dimension mismatch with classes"




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1743371582.py in <cell line: 0>()
----> 1 preds = model.predict(test_ds, verbose=1)
      2 print("preds shape:", preds.shape)
      3 assert preds.shape[1] == len(classes), "Prediction dimension mismatch with classes"
      4 
      5 

NameError: name 'test_ds' is not defined

## === cell 9
thresh = {
    "complex": 0.1321,
    "frog_eye_leaf_spot": 0.2455,
    "healthy": 0.2595,
    "powdery_mildew": 0.0867,
    "rust": 0.1282,
    "scab": 0.3155,
}

thr_vec = np.array([thresh.get(c, 0.5) for c in classes], dtype=np.float32)
healthy_idx = classes.index("healthy") if "healthy" in classes else None




## === cell 10
preds_np = np.asarray(preds, dtype=np.float32)
n, k = preds_np.shape

argmax_idx = preds_np.argmax(axis=1)

is_healthy_max = np.zeros(n, dtype=bool)
if healthy_idx is not None:
    is_healthy_max = argmax_idx == healthy_idx

mask = preds_np > thr_vec[None, :]

pred_labels = []
healthy_str = "healthy"
for i in range(n):
    if is_healthy_max[i]:
        pred_labels.append(healthy_str)
        continue

    chosen_idx = np.flatnonzero(mask[i])
    if chosen_idx.size == 0:
        chosen_idx = np.array([argmax_idx[i]], dtype=np.int64)

    chosen = [classes[j] for j in chosen_idx.tolist()]

    if healthy_str in chosen and len(chosen) > 1:
        chosen = [c for c in chosen if c != healthy_str]
        if len(chosen) == 0:
            chosen = [healthy_str]

    pred_labels.append(" ".join(chosen))

submissions["labels"] = pred_labels
submissions.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2663794727.py in <cell line: 0>()
----> 1 preds_np = np.asarray(preds, dtype=np.float32)
      2 n, k = preds_np.shape
      3 
      4 argmax_idx = preds_np.argmax(axis=1)
      5 

NameError: name 'preds' is not defined

## === cell 11
out_path = "submission.csv"
submissions[["image", "labels"]].to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(submissions))
print(submissions.tail())
