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

# 5. Target score

18.37965012277836

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import random
import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception as e:
    print("Warning: could not enable op determinism:", repr(e))

try:
    tf.config.optimizer.set_jit(False)
except Exception as e:
    print("Warning: could not set JIT flag:", repr(e))

AUTOTUNE = tf.data.AUTOTUNE

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception as e:
    print("Warning: could not set TF threading:", repr(e))

print("TF:", tf.__version__)
gpus = tf.config.list_physical_devices("GPU")
print("Num GPUs:", len(gpus))




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
main_folder = "/kaggle/input/petfinder-pawpularity-score/"

train_image_folder = os.path.join(main_folder, "train")
test_image_folder = os.path.join(main_folder, "test")

train_meta = pd.read_csv(os.path.join(main_folder, "train.csv"))
test_meta = pd.read_csv(os.path.join(main_folder, "test.csv"))

train_meta["img_fnm"] = train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
test_meta["img_fnm"] = test_image_folder + "/" + test_meta["Id"].astype(str) + ".jpg"

sample_n = 64
assert train_meta["img_fnm"].head(sample_n).apply(os.path.exists).all()
assert test_meta["img_fnm"].head(sample_n).apply(os.path.exists).all()

print(train_meta.shape, test_meta.shape)
train_meta.head()




## === cell 2
train_df, valid_df = train_test_split(
    train_meta[["Id", "img_fnm", "Pawpularity"]].copy(),
    test_size=0.15,
    random_state=SEED,
)

train_df["Pawpularity"] = train_df["Pawpularity"].astype(np.float32)
valid_df["Pawpularity"] = valid_df["Pawpularity"].astype(np.float32)

print(train_df.shape, valid_df.shape)




## === cell 3
target_size = 299
BATCH = 32

SHUFFLE_CAP = 2048  # keep shuffle behavior while bounding buffer size

CACHE_DIR = "/kaggle/working/tfdata_cache"
os.makedirs(CACHE_DIR, exist_ok=True)
TRAIN_CACHE = os.path.join(CACHE_DIR, "train_decode.cache")
VALID_CACHE = os.path.join(CACHE_DIR, "valid_decode.cache")
TEST_CACHE = os.path.join(CACHE_DIR, "test_decode.cache")


@tf.function
def _decode_resize(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(
        img, (target_size, target_size), method=tf.image.ResizeMethod.BILINEAR
    )
    img = tf.image.convert_image_dtype(img, tf.float32)  # equivalent to /255.0
    img.set_shape((target_size, target_size, 3))
    return img


augmenter = tf.keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(factor=10.0 / 180.0, fill_mode="reflect", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.05, width_factor=0.05, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.1, 0.1),
            width_factor=(-0.1, 0.1),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="augmenter",
)


@tf.function
def _augment(img):
    return augmenter(img, training=True)


@tf.function
def _load_example(path, t):
    img = _decode_resize(path)
    t = tf.reshape(tf.cast(t, tf.float32), (1,))
    return img, t


@tf.function
def _augment_pair(img, t):
    return _augment(img), t


def _base_options():
    options = tf.data.Options()
    options.experimental_deterministic = False
    options.experimental_optimization.apply_default_optimizations = True
    options.experimental_optimization.map_parallelization = True
    options.experimental_optimization.parallel_batch = True
    options.experimental_optimization.autotune_buffers = True
    options.experimental_optimization.autotune_cpu_budget = 0
    options.experimental_optimization.autotune_ram_budget = 0
    return options


def _maybe_prefetch_to_device(ds):
    if len(gpus) > 0:
        try:
            ds = ds.apply(
                tf.data.experimental.prefetch_to_device("/GPU:0", buffer_size=1)
            )
        except Exception:
            pass
    return ds


def make_train_ds(df):
    paths = df["img_fnm"].astype(str).values
    targets = df["Pawpularity"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, targets)).with_options(
        _base_options()
    )

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache(TRAIN_CACHE)

    ds = ds.shuffle(
        buffer_size=min(len(df), SHUFFLE_CAP),
        seed=SEED,
        reshuffle_each_iteration=True,
    )
    ds = ds.map(_augment_pair, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.batch(BATCH, drop_remainder=True)

    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_valid_ds(df):
    paths = df["img_fnm"].astype(str).values
    targets = df["Pawpularity"].astype(np.float32).values

    ds = tf.data.Dataset.from_tensor_slices((paths, targets)).with_options(
        _base_options()
    )

    ds = ds.map(_load_example, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache(VALID_CACHE)

    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


def make_test_ds(df):
    paths = df["img_fnm"].astype(str).values
    ds = tf.data.Dataset.from_tensor_slices(tf.constant(paths)).with_options(
        _base_options()
    )

    ds = ds.map(_decode_resize, num_parallel_calls=AUTOTUNE, deterministic=False)
    ds = ds.cache(TEST_CACHE)

    ds = ds.batch(BATCH, drop_remainder=False)
    ds = ds.prefetch(AUTOTUNE)
    ds = _maybe_prefetch_to_device(ds)
    return ds


train_ds = make_train_ds(train_df)
valid_ds = make_valid_ds(valid_df)
test_ds = make_test_ds(test_meta[["Id", "img_fnm"]].copy())




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_10/3038974412.py in <cell line: 0>()
    146 
    147 
--> 148 train_ds = make_train_ds(train_df)
    149 valid_ds = make_valid_ds(valid_df)
    150 test_ds = make_test_ds(test_meta[["Id", "img_fnm"]].copy())

/tmp/ipykernel_10/3038974412.py in make_train_ds(df)
     94 
     95     ds = tf.data.Dataset.from_tensor_slices((paths, targets)).with_options(
---> 96         _base_options()
     97     )
     98 

/tmp/ipykernel_10/3038974412.py in _base_options()
     71     options.experimental_optimization.map_parallelization = True
     72     options.experimental_optimization.parallel_batch = True
---> 73     options.experimental_optimization.autotune_buffers = True
     74     options.experimental_optimization.autotune_cpu_budget = 0
     75     options.experimental_optimization.autotune_ram_budget = 0

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/util/options.py in __setattr__(self, name, value)
     59       object.__setattr__(self, name, value)
     60     else:
---> 61       raise AttributeError("Cannot set the property {} on {}.".format(
     62           name,
     63           type(self).__name__))

AttributeError: Cannot set the property autotune_buffers on OptimizationOptions.

## === cell 4
base = tf.keras.applications.InceptionV3(
    include_top=False, weights="imagenet", input_shape=(target_size, target_size, 3)
)
base.trainable = False  # keep fast and stable under Kaggle time limits

inp = layers.Input(shape=(target_size, target_size, 3))
x = base(inp, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.2)(x)
out = layers.Dense(1, activation="linear")(x)

model = models.Model(inp, out)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

model.summary()




## === cell 5
EPOCHS = 3  # keep identical training schedule

history = model.fit(
    train_ds,
    validation_data=valid_ds,
    epochs=EPOCHS,
    verbose=1,
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/2405288561.py in <cell line: 0>()
      2 
      3 history = model.fit(
----> 4     train_ds,
      5     validation_data=valid_ds,
      6     epochs=EPOCHS,

NameError: name 'train_ds' is not defined

## === cell 6
pred = model.predict(test_ds, verbose=1)

pred = pred.reshape(-1)[: len(test_meta)].astype(np.float32)
pred = np.clip(pred, 0.0, 100.0)

print(pred.shape, pred.min(), pred.max())

submission_df = pd.DataFrame({"Id": test_meta["Id"].values, "Pawpularity": pred})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
assert submission_df.shape[0] == test_meta.shape[0]
assert list(submission_df.columns) == ["Id", "Pawpularity"]
assert os.path.exists("submission.csv")

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_10/3020111431.py in <cell line: 0>()
----> 1 pred = model.predict(test_ds, verbose=1)
      2 
      3 pred = pred.reshape(-1)[: len(test_meta)].astype(np.float32)
      4 pred = np.clip(pred, 0.0, 100.0)
      5 

NameError: name 'test_ds' is not defined
