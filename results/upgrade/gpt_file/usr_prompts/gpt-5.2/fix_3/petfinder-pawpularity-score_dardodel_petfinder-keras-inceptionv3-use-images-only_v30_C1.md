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

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)



## === cell 1
train_mode = False  # is it training mode or submission mode

import pandas as pd
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import models, layers
from tensorflow.keras.layers import Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.applications.inception_v3 import InceptionV3

try:
    tf.keras.utils.set_random_seed(SEED)
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

print("TensorFlow:", tf.__version__)



## === cell 2
main_folder = "/kaggle/input/petfinder-pawpularity-score/"

test_image_folder = main_folder + "test"
test_meta = pd.read_csv(main_folder + "test.csv")
test_meta["img_fnm"] = test_image_folder + "/" + test_meta["Id"].astype(str) + ".jpg"

if train_mode:
    train_image_folder = main_folder + "train"
    train_meta = pd.read_csv(main_folder + "train.csv")
    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )



## === cell 3
if not train_mode:
    train_image_folder = main_folder + "train"
    train_meta = pd.read_csv(main_folder + "train.csv")
    train_meta["img_fnm"] = (
        train_image_folder + "/" + train_meta["Id"].astype(str) + ".jpg"
    )

    trn_df, val_df = train_test_split(
        train_meta, test_size=0.15, random_state=SEED, shuffle=True
    )
    trn_df = trn_df.reset_index(drop=True)
    val_df = val_df.reset_index(drop=True)



## === cell 4
target_size = 299

inceptionv3_pretrained = InceptionV3(
    input_shape=(target_size, target_size, 3), include_top=False, weights="imagenet"
)

inceptionv3_pretrained.trainable = False

model = models.Sequential()
model.add(inceptionv3_pretrained)
model.add(layers.Flatten())
model.add(Dropout(0.25))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(64, activation="relu"))
model.add(Dropout(0.2))
model.add(layers.Dense(1))

model.compile(loss="mse", optimizer=Adam(learning_rate=2e-5), metrics=["mse"])

_ = model.summary()




## === cell 5
def _build_id_target_lookup(df_with_target):
    ids = df_with_target["Id"].astype(str).values
    y = df_with_target["Pawpularity"].values.astype(np.float32, copy=False)
    order = np.argsort(ids)
    return ids[order], y[order]


def _targets_from_ids(sorted_ids, sorted_y, batch_ids):
    batch_ids = np.asarray(batch_ids).astype(str, copy=False)
    idx = np.searchsorted(sorted_ids, batch_ids)
    return sorted_y[idx]


trn_ids_sorted, trn_y_sorted = _build_id_target_lookup(trn_df)
val_ids_sorted, val_y_sorted = _build_id_target_lookup(val_df)


def gen_flow_for_two_inputs(datagen, batch, x_df, ids_sorted, y_sorted, shuffle=True):
    """
    Generates (images, pawpularity) batches.
    Preserves original semantics: y is retrieved from x_df itself using Ids emitted by the generator.
    """
    gen = datagen.flow_from_dataframe(
        x_df,
        batch_size=batch,
        shuffle=shuffle,
        x_col="img_fnm",
        y_col="Id",
        class_mode="raw",
        target_size=(target_size, target_size),
    )
    while True:
        batch_image, batch_index = next(gen)
        y = _targets_from_ids(ids_sorted, y_sorted, batch_index)
        yield batch_image, y


def make_tf_dataset_from_pygen(pygen_fn, output_signature, prefetch=True):
    ds = tf.data.Dataset.from_generator(pygen_fn, output_signature=output_signature)
    if prefetch:
        ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_datagen = ImageDataGenerator(
    rescale=1.0 / 255.0,
    rotation_range=30,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.1,
    zoom_range=0.3,
    horizontal_flip=True,
)

valid_datagen = ImageDataGenerator(rescale=1.0 / 255.0)

EPOCH = 2
BATCH = 32

early_stopping = EarlyStopping(
    monitor="val_loss", min_delta=1.0, patience=50, restore_best_weights=True
)

output_sig = (
    tf.TensorSpec(shape=(None, target_size, target_size, 3), dtype=tf.float32),
    tf.TensorSpec(shape=(None,), dtype=tf.float32),
)

steps_per_epoch = int(np.ceil(trn_df.shape[0] / BATCH))
validation_steps = int(np.ceil(val_df.shape[0] / BATCH))

train_ds = make_tf_dataset_from_pygen(
    lambda: gen_flow_for_two_inputs(
        train_datagen, BATCH, trn_df, trn_ids_sorted, trn_y_sorted, shuffle=True
    ),
    output_signature=output_sig,
    prefetch=True,
)
val_ds = make_tf_dataset_from_pygen(
    lambda: gen_flow_for_two_inputs(
        valid_datagen, BATCH, val_df, val_ids_sorted, val_y_sorted, shuffle=False
    ),
    output_signature=output_sig,
    prefetch=True,
)

if train_mode or (not train_mode):
    log = model.fit(
        x=train_ds,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_ds,
        validation_steps=validation_steps,
        epochs=EPOCH,
        callbacks=[early_stopping],
        verbose=1,
    )



## === cell 6
model.optimizer.learning_rate = 0.5e-6

log2 = model.fit(
    x=train_ds,
    steps_per_epoch=steps_per_epoch,
    validation_data=val_ds,
    validation_steps=validation_steps,
    epochs=2,
    callbacks=[early_stopping],
    verbose=1,
)



## === cell 7
local_model_path = "InceptionV3_2_64FC_191121.h5"
model.save(local_model_path)
print("Saved model to:", local_model_path)



## === cell 8
if not train_mode:
    model = load_model(local_model_path)



## === cell 9
if not train_mode:
    print("Test rows:", len(test_meta))
    print(
        "Example test image path exists:", os.path.exists(test_meta.loc[0, "img_fnm"])
    )



## === cell 10
if not train_mode:
    test_datagen = ImageDataGenerator(rescale=1.0 / 255.0)
    BATCH = 32

    def gen_flow_for_two_inputs_test(datagen, batch, x_df, shuffle=False):
        gen = datagen.flow_from_dataframe(
            x_df,
            batch_size=batch,
            shuffle=shuffle,
            x_col="img_fnm",
            y_col=None,
            class_mode=None,
            target_size=(target_size, target_size),
        )
        while True:
            batch_image = next(gen)
            yield batch_image

    steps = int(np.ceil(test_meta.shape[0] / BATCH))

    test_output_sig = tf.TensorSpec(
        shape=(None, target_size, target_size, 3), dtype=tf.float32
    )
    test_ds = tf.data.Dataset.from_generator(
        lambda: gen_flow_for_two_inputs_test(
            test_datagen, BATCH, test_meta, shuffle=False
        ),
        output_signature=test_output_sig,
    ).prefetch(tf.data.AUTOTUNE)

    pred = model.predict(
        test_ds,
        steps=steps,
        verbose=1,
    )



## === cell 11
if not train_mode:
    pred = np.asarray(pred).reshape(-1)
    pred = np.clip(pred, 0.0, 100.0)

    test_meta["Pawpularity"] = pred
    submission_df = test_meta[["Id", "Pawpularity"]].copy()
    submission_df.to_csv("submission.csv", index=False)

    print(submission_df.head())
    print("Wrote submission.csv with shape:", submission_df.shape)
