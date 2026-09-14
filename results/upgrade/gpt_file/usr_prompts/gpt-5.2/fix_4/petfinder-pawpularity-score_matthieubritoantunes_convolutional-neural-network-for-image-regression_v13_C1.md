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
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

20.55488

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import importlib.metadata as imd

        tf_ver = imd.version("tensorflow")
        pb_ver = imd.version("protobuf")
        print("Installed tensorflow:", tf_ver)
        print("Installed protobuf:", pb_ver)
        print("Keeping installed protobuf to avoid TF/protobuf incompatibilities.")
    except Exception as e:
        print("Warning: protobuf version check skipped:", repr(e))


_ensure_protobuf_compat()

import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt
import PIL  # noqa: F401

pd.options.mode.chained_assignment = None

np.random.seed(42)
tf.random.set_seed(42)

print("TensorFlow version (runtime):", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
device_name = tf.test.gpu_device_name()
if device_name != "/device:GPU:0":
    print("GPU device not found - On for CPU time!")
else:
    print(f"Found GPU at {device_name}")



## === cell 2
import os

_CANDIDATE_BASES = [
    "/kaggle/input/petfinder-pawpularity-score",
    "/kaggle/data/petfinder-pawpularity-score",
    "../input/petfinder-pawpularity-score",
]
DATA_PATH = None
for p in _CANDIDATE_BASES:
    if os.path.exists(os.path.join(p, "train.csv")) and os.path.exists(
        os.path.join(p, "test.csv")
    ):
        DATA_PATH = p
        break
if DATA_PATH is None:
    raise FileNotFoundError(
        "Could not locate petfinder-pawpularity-score dataset. Tried: "
        + ", ".join(_CANDIDATE_BASES)
    )
DATA_PATH = DATA_PATH if DATA_PATH.endswith("/") else (DATA_PATH + "/")
print("Using DATA_PATH:", DATA_PATH)

TRAIN_IMG_DIR = os.path.join(DATA_PATH, "train")
TEST_IMG_DIR = os.path.join(DATA_PATH, "test")

training_img = [f for f in os.listdir(TRAIN_IMG_DIR) if f.lower().endswith(".jpg")]
print(f"There are {len(training_img)} images in the training directory")

IMG_HEIGHT = 128
IMG_WIDTH = 128
IMG_CHANNELS = 3
print(f"Using fixed resize: {IMG_WIDTH}x{IMG_HEIGHT} (channels={IMG_CHANNELS})")




## === cell 3
def read_and_decode(filename, reshape_dims):
    image = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(image, channels=IMG_CHANNELS)
    image = tf.image.convert_image_dtype(image, tf.float32)
    return tf.image.resize(image, reshape_dims)


def show_image(filename):
    image = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
    plt.imshow(image.numpy())
    plt.axis("off")


def decode_csv(csv_row):
    record_defaults = [
        tf.constant("", dtype=tf.string),
        tf.constant(0.0, dtype=tf.float32),
    ]
    filename, pawpularity = tf.io.decode_csv(csv_row, record_defaults=record_defaults)
    pawpularity = tf.cast(pawpularity, tf.float32)
    image = read_and_decode(filename, [IMG_HEIGHT, IMG_WIDTH])
    return image, pawpularity




## === cell 4
rand_idx = np.random.randint(0, len(training_img))
rand_img = training_img[rand_idx]
show_image(os.path.join(TRAIN_IMG_DIR, rand_img))
plt.show()



## === cell 5
from sklearn.model_selection import StratifiedShuffleSplit

data = pd.read_csv(os.path.join(DATA_PATH, "train.csv"))

n_bins = 20
data["_pawp_bin"] = pd.qcut(
    data["Pawpularity"].astype(float),
    q=n_bins,
    labels=False,
    duplicates="drop",
)

sssplit = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
for train_index, test_index in sssplit.split(data, data["_pawp_bin"]):
    training_set = data.iloc[train_index].copy()
    eval_set = data.iloc[test_index].copy()

training_set["Pawpularity"].hist(label="Training set")
eval_set["Pawpularity"].hist(label="Eval set")
plt.title("Pawpularity score distribution in training and eval set")
plt.xlabel("Pawpularity score")
plt.ylabel("Count")
plt.legend(loc="upper right")
plt.show()

training_set["Id"] = training_set["Id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, x + ".jpg")
)
training_set[["Id", "Pawpularity"]].to_csv(
    "/kaggle/working/training_set.csv", header=False, index=False
)

eval_set["Id"] = eval_set["Id"].apply(lambda x: os.path.join(TRAIN_IMG_DIR, x + ".jpg"))
eval_set[["Id", "Pawpularity"]].to_csv(
    "/kaggle/working/eval_set.csv", header=False, index=False
)



## === cell 6
BATCH_SIZE = 128

train_dataset = (
    tf.data.TextLineDataset("/kaggle/working/training_set.csv")
    .map(decode_csv, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .shuffle(2048, seed=42, reshuffle_each_iteration=True)
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

eval_dataset = (
    tf.data.TextLineDataset("/kaggle/working/eval_set.csv")
    .map(decode_csv, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)



## === cell 7
conv_filters_1 = 64
kernel_size_1 = 3
pooling_size_1 = 2
conv_filters_2 = 32
kernel_size_2 = 3
pooling_size_2 = 2
conv_filters_3 = 32
kernel_size_3 = 3

model = tf.keras.Sequential(
    [
        tf.keras.layers.Conv2D(
            filters=conv_filters_1,
            kernel_size=kernel_size_1,
            activation="relu",
            input_shape=(IMG_HEIGHT, IMG_WIDTH, IMG_CHANNELS),
        ),
        tf.keras.layers.MaxPooling2D(pool_size=pooling_size_1),
        tf.keras.layers.Conv2D(
            filters=conv_filters_2, kernel_size=kernel_size_2, activation="relu"
        ),
        tf.keras.layers.MaxPool2D(pool_size=pooling_size_2),
        tf.keras.layers.Conv2D(
            filters=conv_filters_3, kernel_size=kernel_size_3, activation="relu"
        ),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=32, activation="relu"),
        tf.keras.layers.Dense(units=1, activation=None),
    ]
)



## === cell 8
model.summary()



## === cell 9
try:
    tf.keras.utils.plot_model(model, show_shapes=True, show_layer_names=False)
except Exception as e:
    print("plot_model skipped:", repr(e))



## === cell 10
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.MeanSquaredError(),
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)



## === cell 11
import time

t0 = time.time()
history = model.fit(
    train_dataset,
    validation_data=eval_dataset,
    epochs=5,
    batch_size=BATCH_SIZE,
    verbose=2,
)
print(f"Training time: {time.time() - t0:.1f}s")




## === cell 12
def training_plot(metrics, history):
    f, ax = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 5))
    if len(metrics) == 1:
        ax = [ax]
    for idx, metric in enumerate(metrics):
        ax[idx].plot(history.history[metric], ls="dashed")
        ax[idx].set_xlabel("Epochs")
        ax[idx].set_ylabel(metric)
        ax[idx].plot(history.history["val_" + metric])
        ax[idx].legend(["train_" + metric, "val_" + metric])


training_plot(["loss", "root_mean_squared_error"], history)
plt.show()



## === cell 13
sample_submission = pd.read_csv(os.path.join(DATA_PATH, "sample_submission.csv"))
test_paths = sample_submission["Id"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, x + ".jpg")
)

tmp_test_csv = pd.DataFrame({"Id": test_paths, "Pawpularity": 0.0})
tmp_test_csv.to_csv("/kaggle/working/test_for_pred.csv", index=False, header=False)

test_dataset = (
    tf.data.TextLineDataset("/kaggle/working/test_for_pred.csv")
    .map(decode_csv, num_parallel_calls=tf.data.AUTOTUNE)
    .cache()
    .batch(BATCH_SIZE)
    .prefetch(tf.data.AUTOTUNE)
)

sample_prediction = model.predict(test_dataset, verbose=0).reshape(-1)

submission_output = pd.DataFrame(
    {"Id": sample_submission["Id"].values, "Pawpularity": sample_prediction}
)

submission_output["Pawpularity"] = submission_output["Pawpularity"].clip(0, 100)

submission_path = "/kaggle/working/submission.csv"
submission_output.to_csv(submission_path, index=False)

print("Wrote submission to:", submission_path)
print(submission_output.head())
print("Submission shape:", submission_output.shape)
print("Any NaNs in Pawpularity?:", submission_output["Pawpularity"].isna().any())
print(
    "Pawpularity min/max:",
    submission_output["Pawpularity"].min(),
    submission_output["Pawpularity"].max(),
)

## --- ERROR in outputing the csv:
Invalid submission: Missing Id column in submission
