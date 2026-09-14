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
import os, random, re, math

import numpy as np
import pandas as pd

import tensorflow as tf
import tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers

print("tf:", tf.__version__)
print("tf.keras:", tf.keras.__version__)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF choose
    tf.config.threading.set_inter_op_parallelism_threads(0)  # let TF choose
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    gpus = tf.config.list_physical_devices("GPU")
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
except Exception:
    pass




## === cell 1
path = "../input/plant-pathology-2021-fgvc8/"

train = pd.read_csv(os.path.join(path, "train.csv"))
sub = pd.read_csv(os.path.join(path, "sample_submission.csv"))
test = sub.copy()

train_images_dir = os.path.join(path, "train_images")
test_images_dir = os.path.join(path, "test_images")

print(train.shape, test.shape)
print(train.head(2))




## === cell 2
AUTO = tf.data.experimental.AUTOTUNE




## === cell 3
import pathlib




## === cell 4
def _resolve_paths_from_csv(images_dir, image_names):
    images_dir = str(images_dir)
    resolved = []
    missing = []
    for name in image_names:
        name = str(name)
        p = os.path.join(images_dir, name)
        if tf.io.gfile.exists(p):
            resolved.append(p)
        else:
            resolved.append(None)
            missing.append(name)
    return resolved, missing


train_image_names = train["image"].astype(str).tolist()
train_paths_resolved, train_missing = _resolve_paths_from_csv(
    train_images_dir, train_image_names
)
if train_missing:
    print(
        f"Warning: missing {len(train_missing)} train images. First few: {train_missing[:5]}"
    )

test_paths = []

print("n_train_images_resolved:", sum(p is not None for p in train_paths_resolved))
print("n_test_images (skipped early scan):", len(test_paths))
print("example test path:", test_paths[0] if test_paths else None)




## === cell 5
kind = np.unique(train["labels"])
kind




## === cell 6
all_classes = sorted(
    {c for s in train["labels"].astype(str).tolist() for c in s.split()}
)
print("classes:", all_classes)

mlb = pd.Series(train["labels"].astype(str)).str.get_dummies(sep=" ")
labels_onehot_features = mlb.reindex(columns=all_classes, fill_value=0).astype(
    np.float32
)

new_train = pd.concat([train[["image"]], labels_onehot_features], axis=1).iloc[:]
new_train.head()




## === cell 7
new_train




## === cell 8
def decode_image(filename, label=None, image_size=(512, 512)):
    filename = tf.cast(filename, tf.string)
    bits = tf.io.read_file(filename)

    image = tf.io.decode_jpeg(
        bits, channels=3, dct_method="INTEGER_FAST", try_recover_truncated=True
    )
    image = tf.image.convert_image_dtype(image, tf.float32)  # [0,1]
    image = tf.image.resize(
        image, image_size, method=tf.image.ResizeMethod.BILINEAR, antialias=False
    )
    image.set_shape([image_size[0], image_size[1], 3])

    if label is None:
        return image
    else:
        return image, label




## === cell 9
test_paths[:5]




## === cell 10
BATCH_SIZE = 64




## === cell 11
import tensorflow as tf
from tensorflow import keras




## === cell 12
IMG_SIZE = (512, 512)
N_CLASSES = len(all_classes)

base = tf.keras.applications.ResNet50(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE[0], IMG_SIZE[1], 3),
    pooling="avg",
)
x = base.output
out = Dense(N_CLASSES, activation="sigmoid")(x)
model = Model(inputs=base.input, outputs=out)

model.compile(
    optimizer=optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    run_eagerly=False,
)

print(model.output_shape)




## === cell 13
df = new_train.copy()
df["path"] = train_paths_resolved
df = df.dropna(subset=["path"]).reset_index(drop=True)

idx = np.arange(len(df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)
split = int(0.9 * len(df))
train_idx, val_idx = idx[:split], idx[split:]

df_train = df.iloc[train_idx].reset_index(drop=True)
df_val = df.iloc[val_idx].reset_index(drop=True)

y_cols = all_classes


@tf.function
def _decode_and_preprocess_with_label(p, y):
    img, y = decode_image(p, y, image_size=IMG_SIZE)
    img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
    return img, y


def make_dataset(frame, training=True):
    paths = frame["path"].astype(str).values
    labels = frame[y_cols].values.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    if training:
        ds = ds.shuffle(2048, seed=SEED, reshuffle_each_iteration=True)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_and_preprocess_with_label, num_parallel_calls=AUTO)
    ds = ds.cache()  # cache both train and val to eliminate repeated preprocessing
    ds = ds.batch(BATCH_SIZE, drop_remainder=training)
    ds = ds.prefetch(AUTO)
    return ds


train_ds = make_dataset(df_train, training=True)
val_ds = make_dataset(df_val, training=False)

train_steps = max(1, int(math.ceil(len(df_train) / BATCH_SIZE)))
val_steps = max(1, int(math.ceil(len(df_val) / BATCH_SIZE)))

print("train_steps:", train_steps, "val_steps:", val_steps)




## === cell 14
EPOCHS = 2

history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=EPOCHS,
    steps_per_epoch=train_steps,
    validation_steps=val_steps,
    verbose=1,
)




## === cell 15
@tf.function
def _decode_and_preprocess_no_label(p):
    img = decode_image(p, label=None, image_size=IMG_SIZE)
    img = tf.keras.applications.resnet.preprocess_input(img * 255.0)
    return img


def make_test_dataset(paths):
    paths = np.asarray([str(p) for p in paths], dtype=object)
    ds = tf.data.Dataset.from_tensor_slices(paths)

    options = tf.data.Options()
    options.deterministic = True
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.map_and_batch_fusion = True
    ds = ds.with_options(options)

    ds = ds.map(_decode_and_preprocess_no_label, num_parallel_calls=AUTO).cache()
    ds = ds.batch(BATCH_SIZE, drop_remainder=False).prefetch(AUTO)
    return ds


expected_names = test["image"].astype(str).tolist()

present_paths, missing = _resolve_paths_from_csv(test_images_dir, expected_names)
present_names = [n for n, p in zip(expected_names, present_paths) if p is not None]
present_paths = [p for p in present_paths if p is not None]

if len(present_paths) == 0:
    test["labels"] = "healthy"
    submission = test[["image", "labels"]].copy()
    submission.to_csv("submission.csv", index=False)
    print(
        "No test images found; wrote default all-healthy submission.csv:",
        submission.shape,
    )
else:
    print("Resolved test images directly in:", test_images_dir)
    print("Will predict for present test images:", len(present_paths))
    if missing:
        print("Warning: missing test images:", len(missing), "first few:", missing[:5])

    ordered_test_ds = make_test_dataset(present_paths)
    probs = model.predict(ordered_test_ds, verbose=1)
    print("probs shape:", probs.shape)

    class_to_idx = {c: i for i, c in enumerate(all_classes)}
    thresholds_by_class = {
        "scab": 0.27,
        "frog_eye_leaf_spot": 0.50,
        "complex": 0.30,
        "rust": 0.50,
        "powdery_mildew": 0.50,
        "healthy": 0.50,  # used only as fallback label
    }
    check_order = ["scab", "frog_eye_leaf_spot", "complex", "rust", "powdery_mildew"]
    check_idx_thr = [
        (class_to_idx[c], float(thresholds_by_class[c]), c)
        for c in check_order
        if c in class_to_idx
    ]

    probs_np = np.asarray(probs)
    idxs = np.array([j for j, _, _ in check_idx_thr], dtype=np.int64)
    thrs = np.array([thr for _, thr, _ in check_idx_thr], dtype=np.float32)
    names = [cls_name for _, _, cls_name in check_idx_thr]

    if len(idxs) == 0:
        pred_string = ["healthy"] * probs_np.shape[0]
    else:
        sel = probs_np[:, idxs] > thrs[None, :]
        pred_string = []
        for row in sel:
            chosen = [names[k] for k, v in enumerate(row) if v]
            if not chosen:
                chosen = ["healthy"]
            pred_string.append(" ".join(chosen))

    pred_df = pd.DataFrame({"image": present_names, "labels": pred_string})

    submission = test[["image"]].merge(pred_df, on="image", how="left")
    submission["labels"] = submission["labels"].fillna("healthy")
    submission = submission[["image", "labels"]]
    submission.to_csv("submission.csv", index=False)

    print(submission.head())
    print("Wrote submission.csv with shape:", submission.shape)
    print("submission.csv exists:", os.path.exists("submission.csv"))
