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
    tf.config.experimental.enable_op_determinism()
except Exception as e:
    print("Warning: could not enable_op_determinism(); continuing. Error:", repr(e))

DATA_DIR = "../input/petfinder-pawpularity-score"
TRAIN_CSV_PATH = f"{DATA_DIR}/train.csv"
TEST_CSV_PATH = f"{DATA_DIR}/test.csv"
TRAIN_IMG_DIR = f"{DATA_DIR}/train"
TEST_IMG_DIR = f"{DATA_DIR}/test"

IMG_SIZE = 224
BATCH_SIZE = 32

train_csv = pd.read_csv(TRAIN_CSV_PATH)

train_data = train_csv.drop(columns=["Id", "Pawpularity"]).to_numpy(dtype=np.float32)
train_label = train_csv["Pawpularity"].to_numpy(dtype=np.float32)
train_img_paths = (TRAIN_IMG_DIR + "/" + train_csv["Id"].values + ".jpg").astype(str)

n = len(train_csv)
split = int(n * 0.95)

AUTOTUNE = tf.data.AUTOTUNE


@tf.function
def _decode_resize_preprocess(path):
    img_bytes = tf.io.read_file(path)
    img = tf.image.decode_jpeg(
        img_bytes, channels=3, dct_method="INTEGER_FAST", fancy_upscaling=False
    )
    img = tf.image.resize(img, [IMG_SIZE, IMG_SIZE], method=tf.image.ResizeMethod.AREA)
    img = tf.cast(img, tf.float32)
    img = tf.keras.applications.efficientnet.preprocess_input(img)
    img.set_shape([IMG_SIZE, IMG_SIZE, 3])
    return img


def make_dataset(
    paths,
    meta,
    labels=None,
    training=False,
    batch_size=BATCH_SIZE,
    cache_in_memory=False,
    cache_name="ds_cache",
):
    if labels is None:
        ds = tf.data.Dataset.from_tensor_slices((paths, meta))

        def _map_fn(p, m):
            return (_decode_resize_preprocess(p), m)

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)
    else:
        ds = tf.data.Dataset.from_tensor_slices((paths, meta, labels))

        def _map_fn(p, m, y):
            return ((_decode_resize_preprocess(p), m), y)

        ds = ds.map(_map_fn, num_parallel_calls=AUTOTUNE, deterministic=True)

    if cache_in_memory:
        ds = ds.cache()
    else:
        cache_path = os.path.join("/kaggle/working", f"{cache_name}.cache")
        ds = ds.cache(cache_path)

    if training:
        ds = ds.shuffle(
            buffer_size=min(int(len(paths)), 2048),
            seed=SEED,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)

    options = tf.data.Options()
    options.deterministic = True
    try:
        options.experimental_optimization.map_parallelization = True
        options.experimental_optimization.parallel_batch = True
        options.experimental_optimization.autotune_buffers = True
        options.experimental_optimization.autotune_cpu_budget = True
    except Exception as e:
        print(
            "Warning: could not set some dataset optimization options; continuing. Error:",
            repr(e),
        )
    ds = ds.with_options(options)
    return ds


train_ds = make_dataset(
    train_img_paths[:split],
    train_data[:split],
    train_label[:split],
    training=True,
    cache_in_memory=True,
    cache_name="train",
)
val_ds = make_dataset(
    train_img_paths[split:],
    train_data[split:],
    train_label[split:],
    training=False,
    cache_in_memory=True,
    cache_name="val",
)

train_steps = int(np.ceil(split / BATCH_SIZE))
val_steps = int(np.ceil((n - split) / BATCH_SIZE))

print("train_data:", train_data.shape, train_data.dtype)
print("train_label:", train_label.shape, train_label.dtype)
print("n, split:", n, split)
print("train_steps, val_steps:", train_steps, val_steps)


def _tiny_warmup(ds, steps=2):
    try:
        it = iter(ds)
        for _ in range(steps):
            next(it)
    except Exception as e:
        print("Warning: tiny warm-up skipped due to error:", repr(e))


_tiny_warmup(val_ds, 2)
_tiny_warmup(train_ds, 2)



## === cell 1
eff = tf.keras.applications.EfficientNetB0(
    include_top=False,
    weights="imagenet",
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    pooling="avg",
)
eff.trainable = False

inputA = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image")
x = eff(inputA)
x = layers.BatchNormalization()(x)

inputB = keras.Input(shape=(12,), name="meta")

combined = layers.Concatenate()([x, inputB])
combined = layers.Dense(1)(combined)

model = keras.Model(inputs=[inputA, inputB], outputs=combined)

model.summary()

model.compile(
    optimizer="rmsprop",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError()],
)

history = model.fit(
    train_ds,
    epochs=20,
    steps_per_epoch=train_steps,
    verbose=2,
)

results = model.evaluate(
    val_ds,
    steps=val_steps,
    verbose=0,
)
print("val loss, val rmse:", results)



## === cell 2
test_csv = pd.read_csv(TEST_CSV_PATH)
test_data = test_csv.drop(columns=["Id"]).to_numpy(dtype=np.float32)
test_img_paths = (TEST_IMG_DIR + "/" + test_csv["Id"].values + ".jpg").astype(str)

test_ds = make_dataset(
    test_img_paths,
    test_data,
    labels=None,
    training=False,
    cache_in_memory=True,
    cache_name="test",
)

_tiny_warmup(test_ds, 2)

prediction = model.predict(test_ds, verbose=0).reshape(-1)
prediction = np.clip(prediction, 0.0, 100.0)

submission = pd.DataFrame({"Id": test_csv["Id"].values, "Pawpularity": prediction})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv columns:", list(submission.columns))
