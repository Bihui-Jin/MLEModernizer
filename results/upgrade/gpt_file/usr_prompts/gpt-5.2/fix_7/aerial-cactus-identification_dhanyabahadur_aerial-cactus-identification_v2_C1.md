# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import zipfile
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.vgg16 import VGG16

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.optimizer.set_jit(False)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

INPUT_DIR = "../input/aerial-cactus-identification"
print("Input listing:", os.listdir(INPUT_DIR))




## === cell 1
WORK_DIR = "/kaggle/working"

train_dir_input = os.path.join(INPUT_DIR, "train")
test_dir_input = os.path.join(INPUT_DIR, "test")

train_dir_work = os.path.join(WORK_DIR, "train")
test_dir_work = os.path.join(WORK_DIR, "test")


def _count_jpgs(d):
    try:
        return sum(1 for n in os.scandir(d) if n.is_file() and n.name.endswith(".jpg"))
    except FileNotFoundError:
        return None


if os.path.isdir(train_dir_input) and os.path.isdir(test_dir_input):
    train_dir = train_dir_input
    test_dir = test_dir_input
    print("Using extracted train/test directories from INPUT_DIR.")
else:
    train_zip = os.path.join(INPUT_DIR, "train.zip")
    test_zip = os.path.join(INPUT_DIR, "test.zip")

    if (not os.path.isdir(train_dir_work)) and os.path.isfile(train_zip):
        with zipfile.ZipFile(train_zip, "r") as z:
            z.extractall(WORK_DIR)

    if (not os.path.isdir(test_dir_work)) and os.path.isfile(test_zip):
        with zipfile.ZipFile(test_zip, "r") as z:
            z.extractall(WORK_DIR)

    train_dir = train_dir_work
    test_dir = test_dir_work

print(
    "train_dir:",
    train_dir,
    "exists:",
    os.path.isdir(train_dir),
    "num_files:",
    _count_jpgs(train_dir) if os.path.isdir(train_dir) else None,
)
print(
    "test_dir:",
    test_dir,
    "exists:",
    os.path.isdir(test_dir),
    "num_files:",
    _count_jpgs(test_dir) if os.path.isdir(test_dir) else None,
)

assert os.path.isdir(train_dir), f"train_dir not found: {train_dir}"
assert os.path.isdir(test_dir), f"test_dir not found: {test_dir}"




## === cell 2
train = pd.read_csv(os.path.join(INPUT_DIR, "train.csv"))
df_test = pd.read_csv(os.path.join(INPUT_DIR, "sample_submission.csv"))

print(train.head())
print(train.dtypes)
print("dataset has {} rows and {} columns".format(train.shape[0], train.shape[1]))
print(train["has_cactus"].value_counts())
print("There are {} rows in test set".format(_count_jpgs(test_dir)))
print("There are {} rows in train set".format(_count_jpgs(train_dir)))
print("There are {} rows in submission data".format(df_test.shape[0]))




## === cell 3
train = train.copy()
train["has_cactus"] = train["has_cactus"].astype(str)

batch_size = 150  # kept identical (also used for bottleneck extraction caches)

n_total = len(train)
val_size = 2500
split_idx = max(1, n_total - val_size)

train_df = train.iloc[:split_idx].reset_index(drop=True)
val_df = train.iloc[split_idx:].reset_index(drop=True)

if val_df["has_cactus"].nunique() < 2:
    split_idx = max(1, n_total - 4000)
    train_df = train.iloc[:split_idx].reset_index(drop=True)
    val_df = train.iloc[split_idx:].reset_index(drop=True)

print(
    "Train/Val sizes:",
    len(train_df),
    len(val_df),
    "Val classes:",
    val_df["has_cactus"].value_counts().to_dict(),
)

datagen = ImageDataGenerator(rescale=1.0 / 255.0)
train_generator = None
validation_generator = None




## === cell 4
SKIP_UNUSED_CNN_TRAINING = True

if not SKIP_UNUSED_CNN_TRAINING:
    train_generator = datagen.flow_from_dataframe(
        dataframe=train_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        class_mode="binary",
        batch_size=batch_size,
        target_size=(150, 150),
        shuffle=True,
        seed=SEED,
    )

    validation_generator = datagen.flow_from_dataframe(
        dataframe=val_df,
        directory=train_dir,
        x_col="id",
        y_col="has_cactus",
        class_mode="binary",
        batch_size=50,
        target_size=(150, 150),
        shuffle=False,
    )

    model = models.Sequential()
    model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(64, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Conv2D(128, (3, 3), activation="relu", input_shape=(150, 150, 3)))
    model.add(layers.MaxPool2D((2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(512, activation="relu"))
    model.add(layers.Dense(1, activation="sigmoid"))

    model.summary()

    model.compile(
        loss="binary_crossentropy",
        optimizer=keras.optimizers.RMSprop(),
        metrics=["acc"],
    )

    epochs = 10
    steps_per_epoch = min(
        100, int(np.ceil(train_generator.n / train_generator.batch_size))
    )
    validation_steps = min(
        50, int(np.ceil(validation_generator.n / validation_generator.batch_size))
    )

    history = model.fit(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=epochs,
        validation_data=validation_generator,
        validation_steps=validation_steps,
        verbose=2,
    )
else:
    print(
        "Skipping unused CNN training to meet runtime constraints (no effect on final submission)."
    )




## === cell 5
model_vg = VGG16(weights="imagenet", include_top=False, input_shape=(150, 150, 3))
model_vg.summary()




## === cell 6
AUTOTUNE = tf.data.AUTOTUNE


@tf.function(reduce_retracing=True)
def _load_image_tf(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, [150, 150], method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    return img


def _build_paths_and_labels(directory, df, y_col):
    base = directory.rstrip("/") + "/"
    paths = (base + df["id"].astype(str).values).astype(str)
    labels = None
    if y_col is not None:
        labels = df[y_col].astype(np.float32).values
    return paths, labels


def _make_dataset(paths, labels, batch_size, shuffle=False):
    paths_tf = tf.convert_to_tensor(paths, dtype=tf.string)
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices(paths_tf)
    else:
        labels_tf = tf.convert_to_tensor(labels, dtype=tf.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths_tf, labels_tf))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(paths), seed=SEED, reshuffle_each_iteration=False
        )

    if labels is None:
        ds = ds.map(_load_image_tf, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = ds.map(
            lambda p, y: (_load_image_tf(p), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    return ds


def extract_features_fast(directory, df, y_col, batch_size):
    paths, labels = _build_paths_and_labels(directory, df, y_col)
    ds = _make_dataset(paths, labels, batch_size=batch_size, shuffle=False)

    feat_ds = ds
    if y_col is None:
        feat_ds = feat_ds.map(
            lambda x: model_vg(x, training=False),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )
    else:
        feat_ds = feat_ds.map(
            lambda x, y: (model_vg(x, training=False), y),
            num_parallel_calls=AUTOTUNE,
            deterministic=True,
        )

    feats_list = []
    labels_list = [] if y_col is not None else None
    for batch in feat_ds:
        if y_col is None:
            f = batch
        else:
            f, y = batch
            labels_list.append(y.numpy())
        feats_list.append(f.numpy())

    features = np.concatenate(feats_list, axis=0).astype(np.float32, copy=False)
    out_labels = None
    if y_col is not None:
        out_labels = np.concatenate(labels_list, axis=0).astype(np.float32, copy=False)
    return features, out_labels


CACHE_DIR = WORK_DIR
train_cache_x = os.path.join(
    CACHE_DIR, f"vgg16_bottleneck_train_x_150x150_bs{batch_size}.npy"
)
train_cache_y = os.path.join(
    CACHE_DIR, f"vgg16_bottleneck_train_y_150x150_bs{batch_size}.npy"
)
test_cache_x = os.path.join(
    CACHE_DIR, f"vgg16_bottleneck_test_x_150x150_bs{batch_size}.npy"
)

train_fe = train.copy()
train_fe["has_cactus"] = train_fe["has_cactus"].astype(int)

if os.path.isfile(train_cache_x) and os.path.isfile(train_cache_y):
    features = np.load(train_cache_x, mmap_mode="r")
    labels = np.load(train_cache_y, mmap_mode="r")
else:
    features, labels = extract_features_fast(
        train_dir, train_fe, y_col="has_cactus", batch_size=batch_size
    )
    np.save(train_cache_x, features)
    np.save(train_cache_y, labels)

train_features = features[:split_idx]
train_labels = labels[:split_idx]

validation_features = features[split_idx:]
validation_labels = labels[split_idx:]

print("Feature tensor shapes:", train_features.shape, validation_features.shape)




## === cell 7
feat_dim = int(np.prod(model_vg.output_shape[1:]))

train_features = np.asarray(train_features, dtype=np.float32).reshape((-1, feat_dim))
validation_features = np.asarray(validation_features, dtype=np.float32).reshape(
    (-1, feat_dim)
)
train_labels = np.asarray(train_labels, dtype=np.float32)
validation_labels = np.asarray(validation_labels, dtype=np.float32)

df_test_fe = df_test[["id"]].copy()

if os.path.isfile(test_cache_x):
    test_features = np.load(test_cache_x, mmap_mode="r")
else:
    test_features, _ = extract_features_fast(
        test_dir, df_test_fe, y_col=None, batch_size=batch_size
    )
    np.save(test_cache_x, test_features)

test_features = np.asarray(test_features, dtype=np.float32).reshape(
    (df_test_fe.shape[0], feat_dim)
)

print("Shapes:", train_features.shape, validation_features.shape, test_features.shape)




## === cell 8
model = models.Sequential()
model.add(
    layers.Dense(
        212,
        activation="relu",
        kernel_regularizer=regularizers.l1_l2(0.001),
        input_dim=feat_dim,
    )
)
model.add(layers.Dropout(0.2))
model.add(layers.Dense(1, activation="sigmoid"))

model.compile(
    loss="binary_crossentropy",
    optimizer=keras.optimizers.RMSprop(),
    metrics=["acc"],
)

history = model.fit(
    train_features,
    train_labels,
    epochs=30,
    batch_size=15,
    validation_data=(validation_features, validation_labels),
    verbose=2,
)

y_pre = model.predict(test_features, verbose=0).reshape(-1).astype(np.float64)

sub = pd.DataFrame({"id": df_test["id"].values, "has_cactus": y_pre})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print("File exists:", os.path.isfile("submission.csv"))
print("Columns:", list(sub.columns))
assert list(sub.columns) == ["id", "has_cactus"]
assert sub.shape[0] == df_test.shape[0]
