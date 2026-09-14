# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

18.870197063329275

# 6. Current score

30.10784

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 42.24638) has done: 'I fix the runtime crash by avoiding TPU auto-connect (it triggers a protobuf incompatibility in this environment) and using the safe default GPU/CPU strategy, while keeping the model and preprocessing unchanged. Then I fix the submission validity error by clipping predictions to the required inclusive range [1, 100] (your current clip allowed 0). I also renumber the notebook cells to start from 1 so it runs cleanly in the provided “cells” format and always writes a `submission.csv`. These changes are execution/format fixes and should only nudge score minimally (mostly unchanged), while producing a valid Kaggle submission.'
- What this solution (achieved 29.29661) has done: 'I fix the runtime crash caused by an incompatible protobuf/tensorflow-protobuf interaction (`MessageFactory.GetPrototype`) by forcing TensorFlow to use the pure-Python protobuf implementation before importing TensorFlow. Then I make the data path resolution robust to both `/kaggle/input/...` and the relative `../input/...` layouts so the script always finds the CSVs/images in this environment. Finally, I keep the model and inference logic unchanged, but add a safe fallback: if the fine-tuned weights file isn’t present, fit only the existing top Dense layer on the provided tabular metadata (same architecture, no new layers) to move RMSE down from the current weak “imagenet-only” baseline toward the target, then generate a valid `submission.csv`.'
- What this solution (achieved 30.10784) has done: 'We fix the immediate TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* its version flag **before any TensorFlow-related import**, plus setting it at the very top of the script to ensure it takes effect in Kaggle. Then we keep your model/training logic intact, but make the fallback training more stable and score-improving (still minimal) by normalizing images the way EfficientNet expects and by adding a tiny validation split to avoid training on all data blindly (no change to architecture/loss). Finally, we ensure the submission is always written with the exact required columns and valid value range [1, 100].'
- What this solution (achieved 30.10784) has done: 'I fix the TensorFlow/protobuf crash by forcing a compatible protobuf package version before importing TensorFlow (the current env var approach isn’t sufficient in this Kaggle image). I keep your model/training logic unchanged, but make execution robust by verifying the resolved data directory and by ensuring image paths exist before building datasets (so inference doesn’t crash mid-epoch). Finally, I keep the same submission format/columns and clipping range, guaranteeing a valid `submission.csv` is written end-to-end.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        import google.protobuf.__version__ as _pb_ver  # type: ignore
    except Exception:
        _pb_ver = None

    needs = True
    try:
        if _pb_ver is not None:
            parts = [int(p) for p in str(_pb_ver).split(".")[:2]]
            needs = not (parts[0] == 3 and parts[1] == 20)
    except Exception:
        needs = True

    if needs:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.3"]
        )


_ensure_protobuf_compat()

import numpy as np
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.applications.efficientnet import (
    preprocess_input as effnet_preprocess,
)

SEED = 42
tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

image_size = 224
batch_size = 128

print("TensorFlow:", tf.__version__)
print("Num GPUs:", len(tf.config.list_physical_devices("GPU")))

try:
    strategy = tf.distribute.get_strategy()
except Exception:
    strategy = tf.distribute.MirroredStrategy()


def _resolve_data_dir():
    candidates = [
        "../input/petfinder-pawpularity-score",
        "/kaggle/input/petfinder-pawpularity-score",
        "/kaggle/data/petfinder-pawpularity-score",
        "/kaggle/input/petfinder-pawpularity-score/petfinder-pawpularity-score",
        "/kaggle/data/petfinder-pawpularity-score/petfinder-pawpularity-score",
    ]
    for c in candidates:
        if os.path.exists(os.path.join(c, "train.csv")) and os.path.exists(
            os.path.join(c, "test.csv")
        ):
            return c
    return "../input/petfinder-pawpularity-score"


DATA_DIR = _resolve_data_dir()
print("Using DATA_DIR:", DATA_DIR)

test = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))
test["file_path"] = test["Id"].apply(
    lambda identifier: os.path.join(DATA_DIR, "test", f"{identifier}.jpg")
)

missing = ~test["file_path"].apply(os.path.exists)
if missing.any():
    print("Warning: missing test images:", int(missing.sum()))
    test = test.loc[~missing].reset_index(drop=True)

print(test.head())



## === cell 1
img_augmentation = Sequential(
    [
        layers.RandomRotation(factor=0.15),
        layers.RandomTranslation(height_factor=0.1, width_factor=0.1),
        layers.RandomFlip(),
        layers.RandomContrast(factor=0.1),
    ],
    name="img_augmentation",
)

tabular_columns = [
    "Subject Focus",
    "Eyes",
    "Face",
    "Near",
    "Action",
    "Accessory",
    "Group",
    "Collage",
    "Human",
    "Occlusion",
    "Info",
    "Blur",
]


def rmse(y_true, y_pred):
    y_true = tf.cast(y_true, tf.float32)
    y_pred = tf.cast(y_pred, tf.float32)
    return tf.sqrt(tf.reduce_mean((y_true - y_pred) ** 2))


def get_tabular_prediciton_model(inputs):
    width = 32
    depth = 3
    activation = "relu"
    kernel_regularizer = keras.regularizers.l2()
    x = keras.layers.Dense(
        width, activation=activation, kernel_regularizer=kernel_regularizer
    )(inputs)
    for i in range(depth):
        if i == 0:
            x = inputs
        x = keras.layers.Dense(
            width, activation=activation, kernel_regularizer=kernel_regularizer
        )(x)
        if (i + 1) % 3 == 0:
            x = keras.layers.Concatenate()([x, inputs])
    return x


def _select_efficientnet_weights():
    """
    Keep architecture identical; just ensure weights reference is valid.
    """
    candidate = "../input/b0-weights/efficientnetb0_notop.h5"
    if os.path.exists(candidate):
        return candidate
    return "imagenet"


def get_model():
    image_inputs = layers.Input(shape=(image_size, image_size, 3))
    image_x = img_augmentation(image_inputs)

    weights_choice = _select_efficientnet_weights()
    base = EfficientNetB0(
        include_top=False,
        input_tensor=image_x,
        weights=weights_choice,
        input_shape=(image_size, image_size, 3),
    )
    base.trainable = False

    image_x = layers.GlobalAveragePooling2D(name="avg_pool")(base.output)
    top_dropout_rate = 0.2
    image_x = layers.Dropout(top_dropout_rate, name="top_dropout")(image_x)

    tabular_inputs = tf.keras.Input((len(tabular_columns),), name="tabular_inputs")
    tabular_x = get_tabular_prediciton_model(tabular_inputs)

    x = tf.keras.layers.Concatenate(axis=1)([image_x, tabular_x])
    outputs = layers.Dense(1)(x)

    optimizer = tf.keras.optimizers.Adam(1e-3)
    model = tf.keras.Model(
        inputs=[image_inputs, tabular_inputs], outputs=[outputs], name="EfficientNet"
    )
    model.compile(optimizer=optimizer, loss=rmse, metrics=["mae", "mape"])
    return model


def preprocess_test_data(image_url, tabular):
    image_string = tf.io.read_file(image_url)
    image = tf.image.decode_jpeg(image_string, channels=3)
    image = tf.image.central_crop(image, 1.0)
    image = tf.image.resize(image, (image_size, image_size))
    image = tf.cast(image, tf.float32)
    image = effnet_preprocess(image)
    tabular = tf.cast(tabular, tf.float32)
    return (image, tabular), tf.constant(0.0, dtype=tf.float32)




## === cell 2
with strategy.scope():
    model_up = get_model()

weights_path = "../input/b0-weights/pet_trained_weights_v6.h5"
has_finetuned = os.path.exists(weights_path)
if has_finetuned:
    model_up.load_weights(weights_path)
    print("Loaded fine-tuned weights:", weights_path)
else:
    print(
        "Fine-tuned weights not found at",
        weights_path,
        "- using base weights only:",
        _select_efficientnet_weights(),
    )

if not has_finetuned:
    train_csv_path = os.path.join(DATA_DIR, "train.csv")
    if os.path.exists(train_csv_path):
        train_df = pd.read_csv(train_csv_path)
        train_df["file_path"] = train_df["Id"].apply(
            lambda identifier: os.path.join(DATA_DIR, "train", f"{identifier}.jpg")
        )

        missing_tr = ~train_df["file_path"].apply(os.path.exists)
        if missing_tr.any():
            print("Warning: missing train images:", int(missing_tr.sum()))
            train_df = train_df.loc[~missing_tr].reset_index(drop=True)

        for layer in model_up.layers:
            layer.trainable = False
        model_up.layers[-1].trainable = True

        model_up.compile(
            optimizer=tf.keras.optimizers.Adam(1e-3), loss=rmse, metrics=["mae", "mape"]
        )

        def preprocess_train_data(image_url, tabular, y):
            image_string = tf.io.read_file(image_url)
            image = tf.image.decode_jpeg(image_string, channels=3)
            image = tf.image.central_crop(image, 1.0)
            image = tf.image.resize(image, (image_size, image_size))
            image = tf.cast(image, tf.float32)
            image = effnet_preprocess(image)
            tabular = tf.cast(tabular, tf.float32)
            y = tf.cast(y, tf.float32)
            return (image, tabular), y

        tabular_all = train_df[tabular_columns].astype(np.float32).values
        y_all = train_df["Pawpularity"].astype(np.float32).values
        paths_all = train_df["file_path"].values

        n = len(train_df)
        idx = np.arange(n)
        rng = np.random.default_rng(SEED)
        rng.shuffle(idx)
        val_size = max(1, int(0.1 * n))
        val_idx = idx[:val_size]
        tr_idx = idx[val_size:]

        paths_train = paths_all[tr_idx]
        tabular_train = tabular_all[tr_idx]
        y_train = y_all[tr_idx]

        paths_val = paths_all[val_idx]
        tabular_val = tabular_all[val_idx]
        y_val = y_all[val_idx]

        ds_train = (
            tf.data.Dataset.from_tensor_slices((paths_train, tabular_train, y_train))
            .shuffle(2048, seed=SEED, reshuffle_each_iteration=True)
            .map(preprocess_train_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )
        ds_val = (
            tf.data.Dataset.from_tensor_slices((paths_val, tabular_val, y_val))
            .map(preprocess_train_data, num_parallel_calls=tf.data.AUTOTUNE)
            .batch(batch_size)
            .prefetch(tf.data.AUTOTUNE)
        )

        model_up.fit(ds_train, validation_data=ds_val, epochs=2, verbose=1)
    else:
        print("train.csv not found at", train_csv_path, "- skipping fallback training")

tabular_np = test[tabular_columns].astype(np.float32).values
paths_np = test["file_path"].values

ds_try = (
    tf.data.Dataset.from_tensor_slices((paths_np, tabular_np))
    .map(preprocess_test_data, num_parallel_calls=tf.data.AUTOTUNE)
    .batch(batch_size)
    .prefetch(tf.data.AUTOTUNE)
)

res_0 = model_up.predict(ds_try, verbose=1).reshape(-1)
print(
    "Pred shape:", res_0.shape, "min/max:", float(np.min(res_0)), float(np.max(res_0))
)



## === cell 3
predictions = res_0.astype(np.float32)
predictions = np.clip(predictions, 1.0, 100.0)

sub = test[["Id"]].copy()
sub["Pawpularity"] = predictions
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
print(
    "Submission min/max:",
    float(sub["Pawpularity"].min()),
    float(sub["Pawpularity"].max()),
)
