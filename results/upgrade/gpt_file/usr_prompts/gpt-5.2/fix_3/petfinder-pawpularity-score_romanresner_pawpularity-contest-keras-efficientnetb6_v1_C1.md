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
from tensorflow.keras.callbacks import ReduceLROnPlateau, ModelCheckpoint

from sklearn.model_selection import train_test_split

print("TensorFlow:", tf.__version__)



## === cell 1
train_df_path = "../input/petfinder-pawpularity-score/train.csv"
test_df_path = "../input/petfinder-pawpularity-score/test.csv"
train_data_path = "../input/petfinder-pawpularity-score/train"
test_data_path = "../input/petfinder-pawpularity-score/test"

checkpoint_filepath = "save_models2.weights.h5"
checkpoint_filepath2 = "save_models.weights.h5"

AUTOTUNE = tf.data.experimental.AUTOTUNE
IMG_SIZE = 456
TARGET = "Pawpularity"
SEED = 88
BATCH_SIZE = 64
DROPOUT_RATE = 0.2
TEST_SIZE = 0.15
EPOCHS = 15
EPOCHS_2 = 10
DATA_SHAPE = 12

LEARNING_RATE = 1e-3
DECAY_STEPS = 100
DECAY_RATE = 0.96

tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
tf.config.threading.set_inter_op_parallelism_threads(2)

tf.random.set_seed(SEED)
np.random.seed(SEED)
random.seed(SEED)



## === cell 2
df_train = pd.read_csv(train_df_path)
df_test = pd.read_csv(test_df_path)

print("The shape of train dataset: ", df_train.shape)
print("The shape of test dataset: ", df_test.shape)
print(df_train.head(2))




## === cell 3
def join_path_train(Id):
    """Add path and .jpg to photo Id"""
    return os.path.join(train_data_path, f"{Id}.jpg")


def join_path_test(Id):
    """Add path and .jpg to photo Id"""
    return os.path.join(test_data_path, f"{Id}.jpg")


def join_jpg(Id):
    """Add .jpg to photo Id."""
    return f"{Id}.jpg"


df_train["Path"] = train_data_path + "/" + df_train["Id"].astype(str) + ".jpg"
df_train["Filename"] = df_train["Id"].astype(str) + ".jpg"

df_test["Path"] = test_data_path + "/" + df_test["Id"].astype(str) + ".jpg"
df_test["Filename"] = df_test["Id"].astype(str) + ".jpg"

print(df_train.dtypes)



## === cell 4
_RESIZE_METHOD = tf.image.ResizeMethod.BILINEAR


def get_image(path):
    image = tf.io.read_file(path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.image.resize(image, [IMG_SIZE, IMG_SIZE], method=_RESIZE_METHOD)
    image = tf.cast(image, dtype=tf.float32)
    return tf.keras.applications.efficientnet.preprocess_input(image)


def creat_dataset_metadata(
    df,
    drop_colums,
    label=None,
    training=False,
    cache_path=None,
):
    if label is not None:
        meta = (
            df.drop(["Id", "Pawpularity", "Path", "Filename"], axis=1)
            .astype(np.float32)
            .values
        )
        input_dataset_data = tf.data.Dataset.from_tensor_slices(meta)

        paths = df["Path"].values
        input_dataset_img = tf.data.Dataset.from_tensor_slices(paths).map(
            get_image, num_parallel_calls=AUTOTUNE, deterministic=True
        )

        if cache_path is not None:
            input_dataset_img = input_dataset_img.cache(cache_path)

        output_dataset = tf.data.Dataset.from_tensor_slices(
            df[label].astype(np.float32).values
        )

        dataset = tf.data.Dataset.zip(
            ((input_dataset_img, input_dataset_data), output_dataset)
        )

        if training:
            dataset = dataset.shuffle(
                buffer_size=min(len(df), 2048), seed=SEED, reshuffle_each_iteration=True
            )

        dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
            buffer_size=AUTOTUNE
        )
        return dataset
    else:
        meta = df.drop(["Id", "Path", "Filename"], axis=1).astype(np.float32).values
        input_dataset_data = tf.data.Dataset.from_tensor_slices(meta)

        paths = df["Path"].values
        input_dataset_img = tf.data.Dataset.from_tensor_slices(paths).map(
            get_image, num_parallel_calls=AUTOTUNE, deterministic=True
        )

        if cache_path is not None:
            input_dataset_img = input_dataset_img.cache(cache_path)

        dataset = tf.data.Dataset.zip((input_dataset_img, input_dataset_data))
        dataset = dataset.batch(BATCH_SIZE, drop_remainder=False).prefetch(
            buffer_size=AUTOTUNE
        )
        return dataset




## === cell 5
train_df, valid_df = train_test_split(
    df_train, test_size=TEST_SIZE, shuffle=True, random_state=SEED
)

train_cache = os.path.join(os.getcwd(), f"cache_train_img_{IMG_SIZE}_{SEED}.cache")
valid_cache = os.path.join(os.getcwd(), f"cache_valid_img_{IMG_SIZE}_{SEED}.cache")

train_set = creat_dataset_metadata(
    train_df,
    drop_colums=["Id", "Pawpularity", "Path", "Filename"],
    label="Pawpularity",
    training=True,
    cache_path=train_cache,
)
valid_set = creat_dataset_metadata(
    valid_df,
    drop_colums=["Id", "Pawpularity", "Path", "Filename"],
    label="Pawpularity",
    training=False,
    cache_path=valid_cache,
)

print("Train batches:", tf.data.experimental.cardinality(train_set).numpy())
print("Valid batches:", tf.data.experimental.cardinality(valid_set).numpy())



## === cell 6
augementation = tf.keras.Sequential(
    [
        tf.keras.layers.RandomFlip(mode="horizontal"),
        tf.keras.layers.RandomWidth(factor=(0.2, 0.3)),
        tf.keras.layers.RandomRotation(factor=(-0.2, 0.3)),
        tf.keras.layers.RandomZoom(0.3),
        tf.keras.layers.RandomHeight(0.2),
    ],
    name="augmentation",
)


def get_model():
    img_input = tf.keras.layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image input")
    x = augementation(img_input)

    backbone = tf.keras.applications.EfficientNetB0(
        include_top=False,
        weights="imagenet",
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        pooling="avg",
    )
    backbone.trainable = False  # preserve the intended "pretrained frozen" behavior

    x = backbone(x)
    x = tf.keras.layers.BatchNormalization()(x)
    x = tf.keras.layers.Dropout(DROPOUT_RATE)(x)
    img_output = tf.keras.layers.Dense(32, activation="relu")(x)
    img_model = tf.keras.Model(img_input, img_output, name="img_model")

    data_input = tf.keras.layers.Input(shape=(DATA_SHAPE,), name="data input")
    y = tf.keras.layers.Dense(64, activation="relu")(data_input)
    data_output = tf.keras.layers.Dense(32, activation="relu")(y)
    data_model = tf.keras.Model(data_input, data_output, name="meta_model")

    concat_layer = tf.keras.layers.Concatenate(name="concat_layer")(
        [img_model.output, data_model.output]
    )
    combined_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(concat_layer)
    combined_dense = tf.keras.layers.Dense(32, activation="relu")(combined_dropout)
    combined_batch = tf.keras.layers.BatchNormalization()(combined_dense)
    final_dropout = tf.keras.layers.Dropout(DROPOUT_RATE)(combined_batch)

    output_layer = tf.keras.layers.Dense(1, activation="relu")(final_dropout)

    model = tf.keras.Model(
        inputs=[img_model.input, data_model.input],
        outputs=output_layer,
        name="pawpularity_model",
    )
    return model


model = get_model()
model.summary()



## === cell 7
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=LEARNING_RATE,
    decay_steps=DECAY_STEPS,
    decay_rate=DECAY_RATE,
    staircase=True,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=lr_schedule),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

model_checkpoint = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor="val_rmse",
    mode="min",
    verbose=1,
    save_best_only=True,
)

reduce_lr = ReduceLROnPlateau(
    monitor="val_rmse", factor=0.5, patience=2, verbose=1, mode="min", min_lr=1e-6
)



## === cell 8
history = model.fit(
    train_set,
    validation_data=valid_set,
    epochs=EPOCHS,
    callbacks=[model_checkpoint, reduce_lr],
    verbose=2,
)

if os.path.exists(checkpoint_filepath):
    model.load_weights(checkpoint_filepath)



## === cell 9
test_cache = os.path.join(os.getcwd(), f"cache_test_img_{IMG_SIZE}_{SEED}.cache")

test_data = creat_dataset_metadata(
    df_test,
    drop_colums=["Id", "Path", "Filename"],
    label=None,
    training=False,
    cache_path=test_cache,
)

prediction = model.predict(test_data, verbose=1)
prediction = np.asarray(prediction).reshape(-1)

prediction = np.clip(prediction, 0.0, 100.0)

sub = pd.DataFrame({"Id": df_test["Id"].values, "Pawpularity": prediction})
sub.to_csv("submission.csv", index=False)

print(sub.head())
print("Wrote submission.csv with shape:", sub.shape)
