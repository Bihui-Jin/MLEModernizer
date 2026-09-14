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

3.12

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1

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

20.58935

# 6. Current score

23.9319

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 23.9319) has done: 'I fix the two blocking runtime issues: the TensorFlow/protobuf import crash by pinning the pure-Python protobuf implementation early, and the invalid Keras Input shape for the metadata branch. Then I make the data pipeline consistent and deterministic (correct train/valid split, correct generator used for validation, correct dtypes/shapes) so `model.fit()` and `model.predict()` run end-to-end. Finally, I ensure the submission matches the required `Id,Pawpularity` format and that prediction order aligns with `test.csv` (not arbitrary `os.listdir()` order), producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import re
import numpy as np
import pandas as pd
import tensorflow as tf

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
TRAIN_IMG_DIR = "/kaggle/input/petfinder-pawpularity-score/train/"
TEST_IMG_DIR = "/kaggle/input/petfinder-pawpularity-score/test/"
TRAIN_CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/train.csv"
TEST_CSV_PATH = "/kaggle/input/petfinder-pawpularity-score/test.csv"
SAMPLE_SUB_PATH = "/kaggle/input/petfinder-pawpularity-score/sample_submission.csv"

relevant_columns = [
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
target_col = "Pawpularity"

train_df = pd.read_csv(TRAIN_CSV_PATH)
test_df = pd.read_csv(TEST_CSV_PATH)

train_feat_map = {
    row["Id"]: row[relevant_columns].to_numpy(dtype=np.float32)
    for _, row in train_df.iterrows()
}
train_y_map = {row["Id"]: np.float32(row[target_col]) for _, row in train_df.iterrows()}

test_feat_map = {
    row["Id"]: row[relevant_columns].to_numpy(dtype=np.float32)
    for _, row in test_df.iterrows()
}

all_train_ids = train_df["Id"].tolist()
rng = np.random.RandomState(SEED)
rng.shuffle(all_train_ids)
split_idx = int(len(all_train_ids) * 0.95)
train_ids = all_train_ids[:split_idx]
valid_ids = all_train_ids[split_idx:]

len(train_ids), len(valid_ids)



## === cell 2
IMG_SIZE = (256, 256)


def _load_image_np(img_path: str) -> np.ndarray:
    img = tf.keras.utils.load_img(img_path, color_mode="rgb", target_size=IMG_SIZE)
    img = np.array(
        img, dtype=np.float32
    )  # [0..255], no normalization to preserve core semantics
    return img


def pythonic_loader_train():
    for pid in train_ids:
        img = _load_image_np(os.path.join(TRAIN_IMG_DIR, f"{pid}.jpg"))
        features = train_feat_map[pid]
        y = train_y_map[pid]
        yield ({"Image": img, "Feature": features}, y)


def pythonic_loader_valid():
    for pid in valid_ids:
        img = _load_image_np(os.path.join(TRAIN_IMG_DIR, f"{pid}.jpg"))
        features = train_feat_map[pid]
        y = train_y_map[pid]
        yield ({"Image": img, "Feature": features}, y)




## === cell 3
train_loader = tf.data.Dataset.from_generator(
    pythonic_loader_train,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)

valid_loader = tf.data.Dataset.from_generator(
    pythonic_loader_valid,
    output_signature=(
        {
            "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
            "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
        },
        tf.TensorSpec(shape=(), dtype=tf.float32),
    ),
)



## === cell 4
input_image = tf.keras.Input(shape=(256, 256, 3), name="Image")
input_meta = tf.keras.Input(shape=(12,), name="Feature")  # FIX: shape must be a tuple

l2 = tf.keras.layers.MaxPool2D((2, 2))(input_image)
l3 = tf.keras.layers.Conv2D(8, (3, 3), activation="relu")(l2)
l4 = tf.keras.layers.MaxPool2D((2, 2))(l3)
l5 = tf.keras.layers.Conv2D(16, (3, 3), activation="relu")(l4)
l6 = tf.keras.layers.MaxPool2D((2, 2))(l5)
l7 = tf.keras.layers.Conv2D(32, (3, 3), activation="relu")(l6)
l_mid1 = tf.keras.layers.MaxPool2D((2, 2))(l7)
l_mid2 = tf.keras.layers.Conv2D(64, (3, 3), activation="relu")(l_mid1)
l8 = tf.keras.layers.Flatten()(l_mid2)
l10 = tf.keras.layers.Dense(1024, activation="gelu")(l8)

combined = tf.keras.layers.concatenate([l10, input_meta])
l12 = tf.keras.layers.Dense(512, activation="gelu")(combined)
bn1 = tf.keras.layers.BatchNormalization()(l12)
le1 = tf.keras.layers.Dense(512, activation="gelu")(bn1)
bn2 = tf.keras.layers.BatchNormalization()(le1)
le2 = tf.keras.layers.Dense(512, activation="gelu")(bn2)
bn3 = tf.keras.layers.BatchNormalization()(le2)
le3 = tf.keras.layers.Dense(512, activation="gelu")(bn3)
bn4 = tf.keras.layers.BatchNormalization()(le3)
le4 = tf.keras.layers.Dense(256, activation="gelu")(bn4)
bn_5 = tf.keras.layers.BatchNormalization()(le4)
le5 = tf.keras.layers.Dense(256, activation="gelu")(bn_5)
bn_6 = tf.keras.layers.BatchNormalization()(le5)

le6 = tf.keras.layers.Dense(128, activation="gelu")(bn_6)
bn5 = tf.keras.layers.BatchNormalization()(le6)
le9 = tf.keras.layers.Dense(64, activation="gelu")(bn5)
bn6 = tf.keras.layers.BatchNormalization()(le9)
l13 = tf.keras.layers.Dense(16, activation="gelu")(bn6)
bn7 = tf.keras.layers.BatchNormalization()(l13)
l14 = tf.keras.layers.Dense(1, activation="sigmoid")(bn7)
output = l14 * tf.constant([100.0], dtype=tf.float32)

model = tf.keras.Model(
    inputs={"Image": input_image, "Feature": input_meta}, outputs={"Label": output}
)



## === cell 5
model.summary()



## === cell 6
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)



## === cell 7
BATCH_SIZE = 32

gen_train = train_loader.batch(BATCH_SIZE).repeat().prefetch(tf.data.AUTOTUNE)
gen_valid = valid_loader.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

steps_per_epoch = len(train_ids) // BATCH_SIZE
validation_steps = max(1, len(valid_ids) // BATCH_SIZE)

steps_per_epoch, validation_steps



## === cell 8
history = model.fit(
    gen_train,
    steps_per_epoch=steps_per_epoch,
    epochs=2,
    validation_data=gen_valid,
    validation_steps=validation_steps,
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1975889142.py in <cell line: 0>()
----> 1 history = model.fit(
      2     gen_train,
      3     steps_per_epoch=steps_per_epoch,
      4     epochs=2,
      5     validation_data=gen_valid,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/trainers/compile_utils.py in <listcomp>(.0)
    243                     MetricsList(
    244                         [
--> 245                             get_metric(m, y_true[0], y_pred[0])
    246                             for m in metrics
    247                             if m is not None

IndexError: list index out of range

## === cell 9
test_ids = test_df["Id"].tolist()


def pythonic_loader_kaggle_test():
    for pid in test_ids:
        img = _load_image_np(os.path.join(TEST_IMG_DIR, f"{pid}.jpg"))
        features = test_feat_map[pid]
        yield {"Image": img, "Feature": features}


test_loader = tf.data.Dataset.from_generator(
    pythonic_loader_kaggle_test,
    output_signature={
        "Image": tf.TensorSpec(shape=(256, 256, 3), dtype=tf.float32),
        "Feature": tf.TensorSpec(shape=(12,), dtype=tf.float32),
    },
)



## === cell 10
preds = model.predict(test_loader.batch(1), verbose=1)

pawpularity_pred = preds["Label"].reshape(-1).astype(np.float32)

pawpularity_pred = np.clip(pawpularity_pred, 0.0, 100.0)

len(pawpularity_pred), len(test_ids)



## === cell 11
submission = pd.DataFrame({"Id": test_ids, "Pawpularity": pawpularity_pred})
submission.to_csv("submission.csv", index=False)

sample = pd.read_csv(SAMPLE_SUB_PATH)
assert list(submission.columns) == list(sample.columns)
assert submission.shape[0] == sample.shape[0]
submission.head()
