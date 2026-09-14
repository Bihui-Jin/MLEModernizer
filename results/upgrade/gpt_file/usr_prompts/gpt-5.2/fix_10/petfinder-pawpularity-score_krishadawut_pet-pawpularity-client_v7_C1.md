# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Predict engagement with a pet's profile based on the photograph for that profile.

## Metric
Root mean squared error.

## Submission Format
For each `Id` in the test set, you must predict a probability for the target variable, `Pawpularity`. The file should contain a header and have the following format:

```
Id, Pawpularity
0008dbfb52aa1dc6ee51ee02adf13537, 99.24
0014a7b528f1682f0cf3b73a991c17a0, 61.71
0019c1388dfcd30ac8b112fb4250c251, 6.23
00307b779c82716b240a24f028b0031b, 9.43
00320c6dd5b4223c62a9670110d47911, 70.89
etc.
```

## Dataset
- **train/** - Folder containing training set photos of the form **{id}.jpg**, where **{id}** is a unique Pet Profile ID.
- **train.csv** - Metadata (described below) for each photo in the training set as well as the target, the photo's Pawpularity score. The Id column gives the photo's unique Pet Profile ID corresponding the photo's file name.

The train.csv and test.csv files contain metadata for photos in the training set and test set, respectively. Each pet photo is labeled with the value of 1 (Yes) or 0 (No) for each of the following features:

- **Focus** - Pet stands out against uncluttered background, not too close / far.
- **Eyes** - Both eyes are facing front or near-front, with at least 1 eye / pupil decently clear.
- **Face** - Decently clear face, facing front or near-front.
- **Near** - Single pet taking up significant portion of photo (roughly over 50% of photo width or height).
- **Action** - Pet in the middle of an action (e.g., jumping).
- **Accessory** - Accompanying physical or digital accessory / prop (i.e. toy, digital sticker), excluding collar and leash.
- **Group** - More than 1 pet in the photo.
- **Collage** - Digitally-retouched photo (i.e. with digital photo frame, combination of multiple photos).
- **Human** - Human in the photo.
- **Occlusion** - Specific undesirable objects blocking part of the pet (i.e. human, cage or fence). Note that not all blocking objects are considered occlusion.
- **Info** - Custom-added text or labels (i.e. pet name, description).
- **Blur** - Noticeably out of focus or noisy, especially for the pet's eyes and face. For Blur entries, "Eyes" column is always set to 0.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
        input/
            description.md (132 lines)
            sample_submission.csv (993 lines)
            sample_submission.csv.zip (22.7 kB)
            test.csv (993 lines)
            test.csv.zip (22.5 kB)
            test.zip (102.2 MB)
            train.csv (8921 lines)
            train.csv.zip (213.0 kB)
            train.zip (926.9 MB)
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
            test/
                a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                ... and 990 other files
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
            train/
                e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                ... and 8918 other files
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
        working/
            petfinder-pawpularity-score/
                description.md (132 lines)
                sample_submission.csv (993 lines)
                ... and 7 other files
                petfinder-pawpularity-score/
                test/
                    a5c4de4c29e2097889f0d4cbd566625c.jpg (57.3 kB)
                    2e5cda2c4cf0530423e8181b7f8c67bc.jpg (94.9 kB)
                    ... and 990 other files
                    test/
                train/
                    e449bbacd2930d6f6ab4589182a09185.jpg (95.7 kB)
                    cfe66786a9c53db0e6936209291cc67d.jpg (247.5 kB)
                    ... and 8918 other files
                    train/
```

-> data/petfinder-pawpularity-score/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/petfinder-pawpularity-score/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/petfinder-pawpularity-score/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> data/sample_submission.csv has 992 rows and 2 columns.
The columns are: Id, Pawpularity

-> data/test.csv has 992 rows and 13 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur

-> data/train.csv has 8920 rows and 14 columns.
The columns are: Id, Subject Focus, Eyes, Face, Near, Action, Accessory, Group, Collage, Human, Occlusion, Info, Blur, Pawpularity

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass

print("TF:", tf.__version__)

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TEST_CSV_PATH = f"{DATA_DIR}/test.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

IMG_SIZE = (224, 224)

train_csv = pd.read_csv(TRAIN_CSV_PATH)
test_csv = pd.read_csv(TEST_CSV_PATH)

feature_cols = [c for c in train_csv.columns if c not in ["Id", "Pawpularity"]]
train_data = train_csv[feature_cols].astype("float32").to_numpy()
train_label = train_csv["Pawpularity"].astype("float32").to_numpy()

print("Train rows:", len(train_csv), "Features:", len(feature_cols))

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def decode_and_resize(path):
    img_bytes = tf.io.read_file(path)
    img = tf.io.decode_jpeg(img_bytes, channels=3, dct_method="INTEGER_FAST")  # uint8
    img = tf.image.resize(img, IMG_SIZE, method=tf.image.ResizeMethod.BILINEAR)
    img = tf.cast(img, tf.float32)  # 0..255 float32 like keras.img_to_array
    return img


train_image_paths = tf.constant(
    [f"{TRAIN_IMG_DIR}/{img_id}.jpg" for img_id in train_csv["Id"].values],
    dtype=tf.string,
)

print(
    "Prepared train paths:",
    train_image_paths.shape,
    "meta:",
    train_data.shape,
    "labels:",
    train_label.shape,
)


## === cell 1
n = len(train_label)
split = int(n * 0.9)

x_img_train = train_image_paths[:split]
x_meta_train = train_data[:split]
y_train = train_label[:split]

x_img_val = train_image_paths[split:]
x_meta_val = train_data[split:]
y_val = train_label[split:]

BATCH_SIZE = 32

CACHE_DIR = "./tfdata_cache_petfinder"
os.makedirs(CACHE_DIR, exist_ok=True)


@tf.function
def _map_fn(p, m, yy):
    img = decode_and_resize(p)
    return (img, m), yy


@tf.function
def _map_fn_nolabel(p, m):
    img = decode_and_resize(p)
    return (img, m)


def make_ds(
    img_paths,
    meta,
    y=None,
    training=False,
    cache_path=None,
    shuffle_buffer=None,
    repeat=False,
):
    img_paths_tf = (
        img_paths
        if isinstance(img_paths, tf.Tensor)
        else tf.convert_to_tensor(img_paths, dtype=tf.string)
    )
    meta_tf = (
        meta
        if isinstance(meta, tf.Tensor)
        else tf.convert_to_tensor(meta, dtype=tf.float32)
    )

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices((img_paths_tf, meta_tf))
        ds = ds.map(_map_fn_nolabel, num_parallel_calls=AUTOTUNE, deterministic=False)
    else:
        y_tf = (
            y if isinstance(y, tf.Tensor) else tf.convert_to_tensor(y, dtype=tf.float32)
        )
        ds = tf.data.Dataset.from_tensor_slices((img_paths_tf, meta_tf, y_tf))
        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=False)

    if cache_path is not None:
        ds = ds.cache(cache_path)
    else:
        ds = ds.cache()

    if training:
        if shuffle_buffer is None:
            shuffle_buffer = int(tf.shape(img_paths_tf)[0])
        shuffle_buffer = int(min(int(shuffle_buffer), 2048))
        ds = ds.shuffle(
            buffer_size=shuffle_buffer, seed=SEED, reshuffle_each_iteration=True
        )

    if repeat:
        ds = ds.repeat()

    ds = ds.batch(BATCH_SIZE, drop_remainder=False)

    options = tf.data.Options()
    options.experimental_deterministic = False
    ds = ds.with_options(options)

    ds = ds.prefetch(AUTOTUNE)
    return ds


train_cache_path = os.path.join(CACHE_DIR, "train_img224.cache")
val_cache_path = os.path.join(CACHE_DIR, "val_img224.cache")

train_ds = make_ds(
    x_img_train,
    x_meta_train,
    y_train,
    training=True,
    cache_path=train_cache_path,
    shuffle_buffer=split,
    repeat=True,
)

val_ds = make_ds(
    x_img_val,
    x_meta_val,
    y_val,
    training=False,
    cache_path=val_cache_path,
    repeat=True,
)

train_steps = int(np.ceil(len(x_meta_train) / BATCH_SIZE))
val_steps = int(np.ceil(len(x_meta_val) / BATCH_SIZE))

print("Train steps/epoch:", train_steps, "Val steps:", val_steps)


## === cell 2
eff = keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(224, 224, 3),
    pooling="avg",
)
eff.trainable = False

inputA = keras.Input(shape=(224, 224, 3), name="image")
x = keras.applications.efficientnet.preprocess_input(inputA)
x = eff(x)
x = layers.BatchNormalization()(x)

inputB = keras.Input(shape=(len(feature_cols),), name="meta")

combined = layers.Concatenate()([x, inputB])
combined = layers.Dense(1)(combined)

model = keras.Model(inputs=[inputA, inputB], outputs=combined)

model.compile(
    optimizer="rmsprop",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
    steps_per_execution=16,
)

model.summary()

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=train_steps,
    validation_data=val_ds,
    validation_steps=val_steps,
    verbose=2,
)

results = model.evaluate(val_ds, steps=val_steps, verbose=0)
print("Validation:", dict(zip(model.metrics_names, results)))


## === cell 3
test_feature_data = test_csv[[c for c in feature_cols]].astype("float32").to_numpy()

test_image_paths = tf.constant(
    [f"{TEST_IMG_DIR}/{img_id}.jpg" for img_id in test_csv["Id"].values],
    dtype=tf.string,
)

test_cache_path = os.path.join(CACHE_DIR, "test_img224.cache")

test_ds = make_ds(
    test_image_paths,
    test_feature_data,
    y=None,
    training=False,
    cache_path=test_cache_path,
    repeat=False,
)

prediction = model.predict(test_ds, verbose=0).reshape(-1)

submission = pd.DataFrame(
    {"Id": test_csv["Id"].values, "Pawpularity": prediction.astype(np.float32)}
)
submission["Pawpularity"] = submission["Pawpularity"].clip(0, 100)

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
