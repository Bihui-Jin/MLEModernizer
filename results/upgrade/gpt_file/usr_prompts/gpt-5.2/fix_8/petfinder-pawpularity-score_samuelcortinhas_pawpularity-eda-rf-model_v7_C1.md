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
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
tf_keras==2.18.0

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

23.9889

# 6. Current score

20.08345

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 29.91224) has done: 'I first fix the import/runtime crash caused by an incompatibility between the bundled `protobuf==6.x` and TensorFlow by forcing the pure-Python protobuf implementation before importing TensorFlow. Then I fix the `preprocessing` NameError by using Keras augmentation layers directly from `tf.keras.layers`, keeping the same model architecture and training loop. Finally, I ensure the pipeline completes end-to-end and always writes a valid `submission.csv` with the correct `Id`/`Pawpularity` columns, and I make the validation/index alignment robust when computing RMSE.'
- What this solution (achieved 29.76409) has done: 'I fix the TensorFlow import crash by enforcing the pure-Python protobuf implementation *and* disabling the C++ protobuf code path, which resolves the `MessageFactory.GetPrototype` error in environments with `protobuf==6.x`. I also remove the unused `keras.preprocessing.image.ImageDataGenerator` import (it can trigger legacy Keras/TensorFlow import interactions) while keeping your model, training loop, and preprocessing identical. To nudge RMSE toward the target (lower is better) without changing the core approach, I make the bin-to-score mapping computed from the training data (mean Pawpularity per bin) instead of hardcoded constants, which is a calibration fix consistent with your bin-classification setup. Finally, I make validation alignment robust by carrying indices through the split and ensuring the submission CSV has exactly the required `Id,Pawpularity` columns.'
- What this solution (achieved 20.08392) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* disabling the C++ implementation before importing TensorFlow, plus clearing any pre-imported protobuf modules that can keep the bad state. I also correct the train/valid split so `idx_train/idx_valid` align with `X` and `y` (right now they’re mismatched due to incorrect unpacking), which is a real logic bug that hurts validation and model fitting. To move RMSE toward your target (lower is better) without changing the core “bin classification + mean-per-bin mapping” approach, I switch inference from `argmax` to the expected-value over bins using the learned softmax probabilities and the same per-bin mean mapping (a calibration/post-processing fix aligned with RMSE). The script still write a valid `submission.csv` with exactly `Id,Pawpularity`.'
- What this solution (achieved 20.08376) has done: 'I fix the TensorFlow import crash caused by the `protobuf==6.x` incompatibility by switching to the supported workaround for TF 2.18: forcing the pure-Python protobuf runtime and preventing any preloaded protobuf modules from persisting, plus setting `TF_USE_CXX11_ABI=0` to avoid the failing C++ proto path in some Kaggle images. I also make the submission generation robust by ensuring the output file is written as `submission.csv` with exactly the required `Id` and `Pawpularity` columns even if earlier visualization cells are skipped. Since your current score (20.08392 RMSE, lower is better) is already better than the target band around 23.9889, I avoid any score-improving modeling changes and keep training/inference logic identical. The remaining edits are strictly runtime/stability fixes and be score-neutral.'
- What this solution (achieved 20.08367) has done: 'I fix the immediate runtime crash by applying the TensorFlow 2.18 + protobuf 6.x compatibility workaround more robustly (force pure-Python protobuf and also disable the C++ implementation), and ensure this happens before importing TensorFlow. I keep the model/training/inference logic unchanged so your score should remain essentially the same (and not move further away from the target band, since you’re already better than the target). I also make the train/test path selection resilient to either `../input/...` or `/kaggle/input/...` layouts without changing what data is used. Finally, I keep the submission writing step deterministic and guaranteed to output `submission.csv` with exactly `Id,Pawpularity`.'
- What this solution (achieved 20.08345) has done: 'I fix the immediate TensorFlow/protobuf crash by using the TF-recommended workaround for protobuf 6.x in Kaggle: force the pure-Python protobuf runtime, disable the C++ implementation, and also set `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` *before Python imports protobuf via TensorFlow*. I keep your model/training/inference logic unchanged (since your current RMSE is already better than the target band and we don’t want to drift further away), and only add stability guards so the notebook runs end-to-end. Finally, I ensure the submission is always written as `submission.csv` with exactly the required `Id,Pawpularity` columns.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

for _m in list(sys.modules.keys()):
    if _m.startswith("google.protobuf"):
        del sys.modules[_m]

import pandas as pd
from glob import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

sns.set_style("darkgrid")
from pathlib import Path

import time
import math

import cv2

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

print("TensorFlow:", tf.__version__)
print("Eager:", tf.executing_eagerly())



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path_candidates = [
    "../input/petfinder-pawpularity-score/",
    "/kaggle/input/petfinder-pawpularity-score/",
]
path = None
for p in path_candidates:
    if os.path.exists(os.path.join(p, "train.csv")):
        path = p
        break
if path is None:
    raise FileNotFoundError(
        "Could not find petfinder-pawpularity-score dataset in expected locations."
    )

train_df = pd.read_csv(path + "train.csv")
test_df = pd.read_csv(path + "test.csv")

train_jpg = glob(path + "train/*.jpg")
test_jpg = glob(path + "test/*.jpg")

train_jpg[:5]



## === cell 2
print("train_df dimensions: ", train_df.shape)
print("test_df dimensions: ", test_df.shape)
train_df.head()



## === cell 3
plt.figure(figsize=(12, 4))

sns.histplot(data=train_df, x="Pawpularity", bins=100)

plt.axvline(
    train_df["Pawpularity"].mean(), c="red", ls="-", lw=3, label="Mean Pawpularity"
)
plt.axvline(
    train_df["Pawpularity"].median(), c="blue", ls="-", lw=3, label="Median Pawpularity"
)
plt.title("Distribution of Pawpularity Scores", fontsize=20)
plt.legend()
plt.xlabel("Pawpularity", fontsize=15)
plt.ylabel("Count", fontsize=15)



## === cell 4
feature_variables = train_df.columns.values.tolist()

for i in feature_variables[1:-1]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(ax=ax[0], data=train_df, x=i, y="Pawpularity")
    sns.histplot(ax=ax[1], data=train_df, x="Pawpularity", hue=i, kde=True)
    plt.suptitle(i, fontsize=20)
    fig.show()



## === cell 5
for i in range(3):

    image_path = train_jpg[i]

    id_stem = Path(image_path).stem

    id_stem_series = train_df.loc[train_df["Id"] == id_stem, "Pawpularity"]
    pawpularity_by_id = id_stem_series.iloc[0]

    image_array = plt.imread(image_path)

    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)

    title = id_stem + ", Pawpularity score:" + str(pawpularity_by_id)
    plt.title(title)

    plt.axis("off")

    plt.show()




## === cell 6
def pawpularity_pics(
    df=pd.DataFrame, num_images=int, desired_pawpularity=int, random_state=int
):
    """Displays random images near a desired Pawpularity value."""
    random_sample = (
        df.loc[
            (df["Pawpularity"] <= (desired_pawpularity + 1))
            & (df["Pawpularity"] >= (desired_pawpularity - 1))
        ]
        .sample(num_images, random_state=random_state)
        .reset_index(drop=True)
    )

    plt.subplots(1, num_images, figsize=(14, 14))

    for i in range(num_images):

        image_path_stem = random_sample.iloc[i]["Id"]
        root = path + "train/"
        extension = ".jpg"
        image_path = root + str(image_path_stem) + extension

        pawpularity_by_id = random_sample.iloc[i]["Pawpularity"]

        image_array = plt.imread(image_path)

        plt.subplot(1, num_images, i + 1)

        plt.title(pawpularity_by_id)

        plt.axis("off")

        plt.imshow(image_array)

    plt.show()
    plt.close()




## === cell 7
pawpularity_pics(train_df, 4, 10, 0)



## === cell 8
pawpularity_pics(train_df, 4, 50, 0)



## === cell 9
pawpularity_pics(train_df, 4, 100, 0)



## === cell 10
del train_jpg, test_jpg



## === cell 11
y = train_df["Pawpularity"]
X = train_df.drop(["Id", "Pawpularity"], axis=1)



## === cell 12
X_train, X_valid, y_train, y_valid = train_test_split(
    X, y, train_size=0.8, test_size=0.2, random_state=0
)
print(
    "Dimensions: \n X_train:{} \n X_valid{} \n y_train{} \n y_valid{}".format(
        X_train.shape, X_valid.shape, y_train.shape, y_valid.shape
    )
)



## === cell 13
RF = RandomForestRegressor(n_estimators=100, max_depth=4)

start = time.time()
RF.fit(X_train, y_train)
stop = time.time()

RF_pred = RF.predict(X_valid)

print(f"Training time: {round((stop - start),3)} seconds")
RF_RMSE = math.sqrt(mean_squared_error(y_valid, RF_pred))
print(f"RF_RMSE: {round(RF_RMSE,3)}")




## === cell 14
def ActualvPredictionsGraph(y_test, y_pred, title):
    if max(y_test) >= max(y_pred):
        my_range = int(max(y_test))
    else:
        my_range = int(max(y_pred))
    plt.figure(figsize=(12, 3))
    plt.scatter(range(len(y_test)), y_test, color="blue")
    plt.scatter(range(len(y_pred)), y_pred, color="red")
    plt.xlabel("Index ")
    plt.ylabel("Pawpularity ")
    plt.title(title, fontdict={"fontsize": 15})
    plt.legend(
        handles=[
            mpatches.Patch(color="red", label="prediction"),
            mpatches.Patch(color="blue", label="actual"),
        ]
    )
    plt.show()
    return


ActualvPredictionsGraph(y_valid[0:50], RF_pred[0:50], "First 50 Actual v. Predicted")
ActualvPredictionsGraph(y_valid, RF_pred, "All Actual v. Predicted")

plt.figure(figsize=(12, 4))
sns.histplot(RF_pred, color="r", alpha=0.3, stat="probability", kde=True)
sns.histplot(y_valid, color="b", alpha=0.3, stat="probability", kde=True)
plt.legend(labels=["prediction", "actual"])
plt.title("Actual v Predict Distribution")
plt.ylim([0.0, 0.2])
plt.show()



## === cell 15
"""
# Test set
X_test = test_df.drop(['Id'], axis=1)

# Make predictions
test_df['Pawpularity'] = RF.predict(X_test) 

# Save to csv
submission_df = test_df[['Id','Pawpularity']]
submission_df.to_csv("submission.csv", index=False)
submission_df.head()
"""



## === cell 16
del RF




## === cell 17
def train_id_to_path(x):
    return path + "train/" + x + ".jpg"


def test_id_to_path(x):
    return path + "test/" + x + ".jpg"


train_df = train_df.drop(
    [
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
    ],
    axis=1,
)
test_df = test_df.drop(
    [
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
    ],
    axis=1,
)

train_df["img_path"] = train_df["Id"].apply(train_id_to_path)
test_df["img_path"] = test_df["Id"].apply(test_id_to_path)

train_df.head()



## === cell 18
train_df["two_bin_pawp"] = pd.qcut(train_df["Pawpularity"], q=2, labels=False)
train_df = train_df.astype({"two_bin_pawp": str})

train_df["five_bin_pawp"] = pd.qcut(train_df["Pawpularity"], q=5, labels=False)
train_df = train_df.astype({"five_bin_pawp": str})

train_df["ten_bin_pawp"] = pd.qcut(train_df["Pawpularity"], q=10, labels=False)
train_df = train_df.astype({"ten_bin_pawp": str})



## === cell 19
num_bins = 5

del y
y = train_df["five_bin_pawp"]
y = pd.get_dummies(y)
y.head()



## === cell 20
train_df.groupby("five_bin_pawp").describe()



## === cell 21
image_height = 128
image_width = 128


def path_to_eagertensor(image_path):
    raw = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(raw, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, (image_height, image_width))
    return image




## === cell 22
og_example_image = plt.imread(train_df["img_path"][0])
print(og_example_image.shape)

plt.imshow(og_example_image)
plt.title("First Training Image")
plt.axis("off")
plt.show()



## === cell 23
example_image = path_to_eagertensor(train_df["img_path"][0])



## === cell 24
print("type: ", type(example_image), "\n shape: ", example_image.shape)

plt.imshow(example_image)
plt.title("First Training Image - with preprocessing")
plt.axis("off")
plt.show()



## === cell 25
del X, X_train, X_valid, y_train, y_valid



## === cell 26
X = []
for i in train_df["img_path"]:
    X.append(path_to_eagertensor(i))
X = np.array(X)
print(type(X), X.shape)



## === cell 27
X_submission = []
for i in test_df["img_path"]:
    X_submission.append(path_to_eagertensor(i))
X_submission = np.array(X_submission)
print(type(X_submission), X_submission.shape)



## === cell 28
idx_all = np.arange(len(train_df))
idx_train, idx_valid = train_test_split(
    idx_all, train_size=0.9, test_size=0.1, random_state=0
)
X_train, X_valid = X[idx_train], X[idx_valid]
y_train, y_valid = y.iloc[idx_train], y.iloc[idx_valid]

print("idx_train:", idx_train.shape, "idx_valid:", idx_valid.shape)
print("X_train:", X_train.shape, "X_valid:", X_valid.shape)
print("y_train:", y_train.shape, "y_valid:", y_valid.shape)



## === cell 29
model = keras.Sequential(
    [
        layers.RandomRotation(factor=0.05, fill_mode="constant"),
        layers.RandomZoom(
            height_factor=(-0.05, 0.05),
            width_factor=(-0.05, 0.05),
            fill_mode="constant",
        ),
        layers.RandomFlip(mode="horizontal"),
        layers.Conv2D(
            filters=32,
            kernel_size=7,
            strides=1,
            padding="same",
            input_shape=[image_height, image_width, 3],
            activation="relu",
        ),
        layers.MaxPool2D(pool_size=2, padding="same"),
        layers.Dropout(rate=0.4),
        layers.Conv2D(
            filters=64, kernel_size=5, strides=1, padding="same", activation="relu"
        ),
        layers.MaxPool2D(pool_size=2, padding="same"),
        layers.Dropout(rate=0.4),
        layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", activation="relu"
        ),
        layers.MaxPool2D(pool_size=2, padding="same"),
        layers.Dropout(rate=0.4),
        layers.Conv2D(
            filters=128, kernel_size=3, strides=1, padding="same", activation="relu"
        ),
        layers.MaxPool2D(pool_size=2, padding="same"),
        layers.Dropout(rate=0.4),
        layers.Flatten(),
        layers.Dense(units=256, activation="relu"),
        layers.Dropout(rate=0.4),
        layers.Dense(units=num_bins, activation="softmax"),
    ]
)

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["categorical_accuracy"],
)

early_stopping = keras.callbacks.EarlyStopping(
    patience=10,
    min_delta=0.0001,
    restore_best_weights=True,
)



## === cell 30
history = model.fit(
    X_train,
    y_train,
    validation_data=(X_valid, y_valid),
    epochs=100,
    batch_size=500,
    callbacks=[early_stopping],
    verbose=True,
)



## === cell 31
history_df = pd.DataFrame(history.history)
history_df.loc[1:, ["loss", "val_loss"]].plot(title="Categorical cross-entropy")



## === cell 32
bin_means_series = train_df.groupby("five_bin_pawp")["Pawpularity"].mean().sort_index()
bin_means = bin_means_series.to_numpy(dtype=float)


def proba_to_pawp(p_row: np.ndarray) -> float:
    p_row = np.asarray(p_row, dtype=float)
    return float(np.dot(p_row, bin_means))


valid_proba = model.predict(X_valid, verbose=0)
valid_pawp = train_df.loc[idx_valid, "Pawpularity"].to_numpy(dtype=float)
valid_preds = np.apply_along_axis(proba_to_pawp, 1, valid_proba)

cnn_rmse = math.sqrt(mean_squared_error(valid_pawp, valid_preds))
print("Final RMSE on validation set:", cnn_rmse)



## === cell 33
np.argmax(model.predict(X_valid, verbose=0), axis=1)[:100]



## === cell 34
model.summary()



## === cell 35
test_proba = model.predict(X_submission, verbose=0)
final_preds = np.apply_along_axis(proba_to_pawp, 1, test_proba)

sub_df = pd.DataFrame()
sub_df["Id"] = test_df["Id"].values
sub_df["Pawpularity"] = final_preds.astype(float)

sub_df["Pawpularity"] = sub_df["Pawpularity"].clip(0, 100)

sub_df = sub_df[["Id", "Pawpularity"]]
sub_df.to_csv("submission.csv", index=False)

assert list(sub_df.columns) == ["Id", "Pawpularity"]
assert len(sub_df) == len(test_df)

sub_df.head()
