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

import tensorflow as tf
import tensorflow.keras as keras

from sklearn.preprocessing import MultiLabelBinarizer

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

gpus = tf.config.list_physical_devices("GPU")
for _g in gpus:
    try:
        tf.config.experimental.set_memory_growth(_g, True)
    except Exception:
        pass

print("TF:", tf.__version__)
print("Num GPUs:", len(gpus))




## === cell 1
DATA_DIR = "../input/plant-pathology-2021-fgvc8"
TRAIN_CSV = os.path.join(DATA_DIR, "train.csv")
SAMPLE_SUB = os.path.join(DATA_DIR, "sample_submission.csv")
TRAIN_IMG_DIR = os.path.join(DATA_DIR, "train_images")
TEST_IMG_DIR = os.path.join(DATA_DIR, "test_images")

train = pd.read_csv(TRAIN_CSV)
submissions = pd.read_csv(SAMPLE_SUB)

print(train.shape, submissions.shape)
train.head()




## === cell 2
label_split = train.labels.apply(lambda x: x.split())
trans_label = MultiLabelBinarizer().fit(label_split)
y = trans_label.transform(label_split).astype(np.float32)
class_names = list(trans_label.classes_)

labels = pd.DataFrame(y, columns=class_names)
print("Classes:", class_names)
labels.head()




## === cell 3
h_target = 224
w_target = 224
batch_size = 32

AUTOTUNE = tf.data.AUTOTUNE

train_paths = train["image"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x)).values
test_paths = submissions["image"].apply(lambda x: os.path.join(TEST_IMG_DIR, x)).values


@tf.function
def decode_resize(path, label=None):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3, dct_method="INTEGER_FAST")
    img = tf.image.resize_with_pad(
        img, h_target, w_target, method="bilinear", antialias=False
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([h_target, w_target, 3])
    if label is None:
        return img
    return img, label


n = len(train_paths)
idx = np.arange(n)
rng = np.random.default_rng(SEED)
rng.shuffle(idx)

val_frac = 0.1
val_n = int(n * val_frac)
val_idx = idx[:val_n]
trn_idx = idx[val_n:]

trn_paths, trn_y = train_paths[trn_idx], y[trn_idx]
val_paths, val_y = train_paths[val_idx], y[val_idx]

opts = tf.data.Options()
opts.experimental_deterministic = True
try:
    opts.experimental_slack = True
except Exception:
    pass
try:
    opts.experimental_optimization.apply_default_optimizations = True
    opts.experimental_optimization.map_parallelization = True
    opts.experimental_optimization.parallel_batch = True
except Exception:
    pass

CACHE_DIR = "/kaggle/working/tfdata_cache_pp2021"
os.makedirs(CACHE_DIR, exist_ok=True)
train_cache_path = os.path.join(CACHE_DIR, "train.cache")
val_cache_path = os.path.join(CACHE_DIR, "val.cache")

ds_train = tf.data.Dataset.from_tensor_slices((trn_paths, trn_y)).with_options(opts)
ds_train = ds_train.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
ds_train = ds_train.map(decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
ds_train = ds_train.apply(tf.data.experimental.ignore_errors())
ds_train = ds_train.cache(train_cache_path)
ds_train = ds_train.batch(batch_size, drop_remainder=False)
ds_train = ds_train.prefetch(AUTOTUNE)

ds_val = tf.data.Dataset.from_tensor_slices((val_paths, val_y)).with_options(opts)
ds_val = ds_val.map(decode_resize, num_parallel_calls=AUTOTUNE, deterministic=True)
ds_val = ds_val.apply(tf.data.experimental.ignore_errors())
ds_val = ds_val.cache(val_cache_path)
ds_val = ds_val.batch(batch_size, drop_remainder=False)
ds_val = ds_val.prefetch(AUTOTUNE)

ds_test = tf.data.Dataset.from_tensor_slices(test_paths).with_options(opts)
ds_test = ds_test.map(
    lambda p: decode_resize(p, None), num_parallel_calls=AUTOTUNE, deterministic=True
)
ds_test = ds_test.apply(tf.data.experimental.ignore_errors())
ds_test = ds_test.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)

print("Train batches:", tf.data.experimental.cardinality(ds_train).numpy())
print("Val batches:", tf.data.experimental.cardinality(ds_val).numpy())
print("Test batches:", tf.data.experimental.cardinality(ds_test).numpy())




## === cell 4
num_classes = len(class_names)

base = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(h_target, w_target, 3),
    pooling="avg",
)
base.trainable = False  # keep training stable/fast under 600s

inp = keras.Input(shape=(h_target, w_target, 3))
x = inp
x = x * 255.0
x = tf.keras.applications.efficientnet.preprocess_input(x)
x = base(x, training=False)
x = keras.layers.Dropout(0.2)(x)
out = keras.layers.Dense(num_classes, activation="sigmoid")(x)
model = keras.Model(inp, out)

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="binary_crossentropy",
)

model.summary()




## === cell 5
EPOCHS = 3
history = model.fit(ds_train, validation_data=ds_val, epochs=EPOCHS, verbose=1)




## === cell 6
preds = model.predict(ds_test, verbose=1)
print("preds shape:", preds.shape)
print("preds min/max:", preds.min(), preds.max())




## === cell 7
thresh = {
    "complex": 0.37,
    "frog_eye_leaf_spot": 0.41,
    "healthy": 0.25,
    "powdery_mildew": 0.17,
    "rust": 0.36,
    "scab": 0.57,
}

thr_arr = np.array([thresh[c] for c in class_names], dtype=np.float32)
healthy_idx = class_names.index("healthy")




## === cell 8
p = preds
argmax_idx = np.argmax(p, axis=1)
is_healthy_argmax = argmax_idx == healthy_idx

mask = p > thr_arr[None, :]
mask_has_any = mask.any(axis=1)
mask_has_healthy = mask[:, healthy_idx]
fallback = (~mask_has_any) | (mask_has_healthy)

pred_labels = np.empty(p.shape[0], dtype=object)
pred_labels[is_healthy_argmax] = "healthy"

rest_rows = np.where(~is_healthy_argmax)[0]
fallback_rest = rest_rows[fallback[rest_rows]]
nonfallback_rest = rest_rows[~fallback[rest_rows]]

pred_labels[fallback_rest] = [class_names[i] for i in argmax_idx[fallback_rest]]

if nonfallback_rest.size:
    rr, cc = np.nonzero(mask[nonfallback_rest])
    split_points = np.flatnonzero(np.diff(rr)) + 1
    groups = np.split(cc, split_points)
    joined = [" ".join(class_names[j] for j in g).strip() for g in groups]
    pred_labels[nonfallback_rest] = joined

submissions = submissions.copy()
submissions["labels"] = pred_labels.tolist()

assert submissions.shape[0] == preds.shape[0]
assert list(submissions.columns) == ["image", "labels"]
print(submissions.head())




## === cell 9
sub_path = "submission.csv"
submissions.to_csv(sub_path, index=False)
print("Wrote", sub_path, "with shape", submissions.shape)
print("Unique label strings (sample):", submissions["labels"].head(10).tolist())
