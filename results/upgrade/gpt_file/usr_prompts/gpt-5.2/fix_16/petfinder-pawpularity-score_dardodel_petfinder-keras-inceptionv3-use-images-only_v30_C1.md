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
import os, random
import numpy as np

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

train_mode = False  # is it training mode or submission mode

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.applications.inception_v3 import InceptionV3

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

_CPU = os.cpu_count() or 2
_INTRA = min(4, _CPU)
_INTER = 1
try:
    tf.config.threading.set_intra_op_parallelism_threads(_INTRA)
    tf.config.threading.set_inter_op_parallelism_threads(_INTER)
except Exception:
    pass

print("TensorFlow:", tf.__version__)
print("Keras:", keras.__version__)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 1
main_folder = "/kaggle/input/petfinder-pawpularity-score/"

test_image_folder = os.path.join(main_folder, "test")
test_meta = pd.read_csv(os.path.join(main_folder, "test.csv"))

test_meta["img_fnm"] = test_image_folder + "/" + test_meta["Id"].astype(str) + ".jpg"

if train_mode:
    train_image_folder = os.path.join(main_folder, "train")
    train_meta = pd.read_csv(os.path.join(main_folder, "train.csv"))
    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )




## === cell 2
if not train_mode:
    train_image_folder = os.path.join(main_folder, "train")
    train_meta = pd.read_csv(os.path.join(main_folder, "train.csv"))

    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )

    trn_df, val_df = train_test_split(
        train_meta, test_size=0.15, random_state=SEED, shuffle=True
    )
    trn_df = trn_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)




## === cell 3
target_size = 299

inceptionv3_pretrained = InceptionV3(
    input_shape=(target_size, target_size, 3), include_top=False, weights="imagenet"
)

inceptionv3_pretrained.trainable = False

model = models.Sequential()
model.add(inceptionv3_pretrained)
model.add(layers.GlobalAveragePooling2D())
model.add(Dropout(0.25))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

model.compile(
    loss="mse",
    optimizer=Adam(learning_rate=2e-5),
    metrics=["mse"],
    run_eagerly=False,
)

_ = model.summary()




## === cell 4
EPOCH = 2
BATCH = 32

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=1.0, patience=50, restore_best_weights=True
)

_TFDATA_CACHE_ROOT = "/kaggle/working/tfdata_cache"
os.makedirs(_TFDATA_CACHE_ROOT, exist_ok=True)

_TFDS_OPTIONS = tf.data.Options()
_TFDS_OPTIONS.experimental_deterministic = True
_TFDS_OPTIONS.experimental_optimization.map_parallelization = True
_TFDS_OPTIONS.experimental_optimization.parallel_batch = True
try:
    _TFDS_OPTIONS.experimental_optimization.map_fusion = True
except Exception:
    pass
try:
    _TFDS_OPTIONS.experimental_optimization.slack = True
except Exception:
    pass


@tf.function(reduce_retracing=True)
def _decode_and_resize(path):
    img = tf.io.read_file(path)
    img = tf.io.decode_jpeg(
        img, channels=3
    )  # faster and fixed-rank vs decode_image for JPEGs
    img = tf.image.resize(
        img,
        [target_size, target_size],
        method=tf.image.ResizeMethod.BILINEAR,
        antialias=True,
    )
    img = tf.cast(img, tf.float32) * (1.0 / 255.0)
    img.set_shape([target_size, target_size, 3])
    return img


data_augment = keras.Sequential(
    [
        layers.RandomFlip("horizontal", seed=SEED),
        layers.RandomRotation(factor=30.0 / 360.0, fill_mode="reflect", seed=SEED),
        layers.RandomTranslation(
            height_factor=0.2, width_factor=0.2, fill_mode="reflect", seed=SEED
        ),
        layers.RandomZoom(
            height_factor=(-0.3, 0.3),
            width_factor=(-0.3, 0.3),
            fill_mode="reflect",
            seed=SEED,
        ),
    ],
    name="data_augment",
)


def _cache_name(prefix: str, df_len: int) -> str:
    return f"{prefix}_seed{SEED}_len{df_len}_ts{target_size}_b{BATCH}.cache"


def make_ds(df, training: bool, cache_name: str):
    paths = df["img_fnm"].astype(str).to_numpy()
    labels = df["Pawpularity"].astype(np.float32).to_numpy()

    ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    ds = ds.with_options(_TFDS_OPTIONS)

    def _load_only(path, y):
        return _decode_and_resize(path), y

    ds = ds.map(_load_only, num_parallel_calls=AUTOTUNE)
    cache_path = os.path.join(_TFDATA_CACHE_ROOT, cache_name)
    ds = ds.cache(cache_path)

    if training:
        shuffle_buf = min(len(df), 2048)
        ds = ds.shuffle(
            buffer_size=shuffle_buf, seed=SEED, reshuffle_each_iteration=True
        )

        def _augment(x, y):
            x = data_augment(x, training=True)
            return x, y

        ds = ds.map(_augment, num_parallel_calls=AUTOTUNE)

    ds = ds.batch(BATCH, drop_remainder=bool(training))
    ds = ds.prefetch(AUTOTUNE)
    return ds


train_ds = make_ds(trn_df, training=True, cache_name=_cache_name("train", len(trn_df)))
val_ds = make_ds(val_df, training=False, cache_name=_cache_name("val", len(val_df)))

log = model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=EPOCH,
    callbacks=[early_stopping],
    verbose=1,
)




## === cell 5
model.optimizer.learning_rate = 0.5e-6

log2 = model.fit(
    x=train_ds,
    validation_data=val_ds,
    epochs=2,
    callbacks=[early_stopping],
    verbose=1,
)




## === cell 6
local_model_path = "InceptionV3_2_64FC_191121.h5"
model.save(local_model_path)
print("Saved model to:", local_model_path)




## === cell 7
if not train_mode:
    pass




## === cell 8
if not train_mode:
    print("Test rows:", len(test_meta))
    print(
        "Example test image path exists:", os.path.exists(test_meta.loc[0, "img_fnm"])
    )




## === cell 9
if not train_mode:
    test_paths = test_meta["img_fnm"].astype(str).to_numpy()
    test_ds = tf.data.Dataset.from_tensor_slices(test_paths)
    test_ds = test_ds.with_options(_TFDS_OPTIONS)

    def _map_test(path):
        return _decode_and_resize(path)

    test_ds = test_ds.map(_map_test, num_parallel_calls=AUTOTUNE)

    test_ds = test_ds.cache(
        os.path.join(_TFDATA_CACHE_ROOT, _cache_name("test", len(test_meta)))
    )

    test_ds = test_ds.batch(BATCH, drop_remainder=False)
    test_ds = test_ds.prefetch(AUTOTUNE)

    pred = model.predict(
        test_ds,
        verbose=1,
    )

    pred = np.asarray(pred).reshape(-1)

    if len(pred) != len(test_meta):
        pred = pred[: len(test_meta)]

    pred = np.clip(pred, 0.0, 100.0)

    test_meta["Pawpularity"] = pred
    submission_df = test_meta[["Id", "Pawpularity"]].copy()
    submission_df.to_csv("submission.csv", index=False)

    print(submission_df.head())
    print("Wrote submission.csv with shape:", submission_df.shape)
    print("submission.csv exists:", os.path.exists("submission.csv"))
