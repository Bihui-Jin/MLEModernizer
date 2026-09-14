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
tqdm==4.67.1

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

20.46629

# 6. Current score

37.80578

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 37.80595) has done: 'We fix the immediate runtime crash caused by an incompatible protobuf version with TensorFlow by forcing the pure-Python protobuf implementation before importing TensorFlow. We also make GPU configuration safe when no GPU is available (avoids index errors). To improve RMSE toward the target with minimal semantic changes, we correct the stratified split to use binned Pawpularity (continuous targets should not be used directly for stratification), which typically yields a more representative validation split and better generalization. Finally, we fix a functional bug in the concatenation head where layers were unintentionally fed the wrong tensor (reusing `concatenated` instead of the evolving `concatModel`), which currently cripples the model’s fusion logic and hurts score; this is a true logic error rather than a modeling change.'
- What this solution (achieved 37.80578) has done: 'We fix the TensorFlow/protobuf crash by upgrading the environment variable to force the pure-Python protobuf implementation *and* disabling the C++ implementation before importing TensorFlow (this is the root cause of the `MessageFactory.GetPrototype` error in recent protobuf). We keep the current model/training logic intact, but make the image-loading pipeline robust to occasional `cv2.imread` failures (which otherwise can crash mid-run) by inserting a safe fallback image. Finally, we keep the submission formatting identical but ensure the script always reaches the CSV write step reliably.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import numpy as np
import pandas as pd
from tqdm import tqdm

from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Dense,
    Flatten,
    Input,
    Concatenate,
    Dropout,
    Conv2D,
    MaxPooling2D,
    GlobalAveragePooling2D,
)

gpus = tf.config.experimental.list_physical_devices("GPU")
if len(gpus) > 0:
    try:
        tf.config.experimental.set_memory_growth(gpus[0], True)
    except Exception:
        pass

import cv2


def getImagePath(imgId, dataType):
    if dataType == "train":
        return os.path.join(
            "../input/petfinder-pawpularity-score/train", imgId + ".jpg"
        )
    if dataType == "test":
        return os.path.join("../input/petfinder-pawpularity-score/test", imgId + ".jpg")
    raise ValueError("dataType must be 'train' or 'test'")


def read_image_rgb_normalized(path, img_size):
    """
    Bugfix/stability: cv2.imread can return None (corrupt/missing file).
    Use a safe fallback image instead of crashing.
    """
    img = cv2.imread(path)
    if img is None:
        img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    else:
        img = cv2.resize(img, (img_size, img_size))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
trainCSV = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
testCSV = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")

n_bins = 20
y_bins = pd.cut(trainCSV["Pawpularity"], bins=n_bins, labels=False, include_lowest=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_index, val_index in split.split(trainCSV, y_bins):
    trainDF = trainCSV.iloc[train_index].reset_index(drop=True)
    valDF = trainCSV.iloc[val_index].reset_index(drop=True)

del trainCSV, split, y_bins




## === cell 2
trainX = trainDF.drop("Pawpularity", axis=1)
trainY = trainDF.Pawpularity.astype(np.float32)
valX = valDF.drop("Pawpularity", axis=1)
valY = valDF.Pawpularity.astype(np.float32)

imgTrainXList = []
imgValXList = []
imgTestXList = []

IMG_SIZE = 64

for imgId in tqdm(trainX.Id, desc="Loading train images"):
    img = read_image_rgb_normalized(getImagePath(imgId, "train"), IMG_SIZE)
    imgTrainXList.append(img)

imgTrainX = np.array(imgTrainXList, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
featuresTrainX = trainX.drop("Id", axis=1).astype(np.float32)
del imgTrainXList, trainX, trainDF

for imgId in tqdm(valX.Id, desc="Loading val images"):
    img = read_image_rgb_normalized(getImagePath(imgId, "train"), IMG_SIZE)
    imgValXList.append(img)

imgValX = np.array(imgValXList, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
featuresValX = valX.drop("Id", axis=1).astype(np.float32)
del imgValXList, valX, valDF

for imgId in tqdm(testCSV.Id, desc="Loading test images"):
    img = read_image_rgb_normalized(getImagePath(imgId, "test"), IMG_SIZE)
    imgTestXList.append(img)

imgTestX = np.array(imgTestXList, dtype=np.float32).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
del imgTestXList

featuresTestX = testCSV.drop("Id", axis=1).astype(np.float32)




## === cell 3
img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3))
imgModel = Conv2D(32, kernel_size=(3, 3), activation="relu")(img_input)
imgModel = Conv2D(64, (3, 3), activation="relu")(imgModel)
imgModel = MaxPooling2D(pool_size=(2, 2))(imgModel)
imgModel = Dropout(0.25)(imgModel)
imgModel = GlobalAveragePooling2D()(imgModel)
imgModel = Dense(128, activation="relu")(imgModel)
imgModel = Dropout(0.5)(imgModel)
imgModelOutput = Dense(1, activation="linear")(imgModel)

tag_input = Input(shape=featuresTrainX.shape[1:])
tagModel = Dense(12, input_dim=12, activation="relu")(tag_input)
tagModel = Dense(32, activation="relu")(tagModel)
tagModel = Dropout(0.5)(tagModel)
tagModel = Dense(32, activation="relu")(tagModel)
tagModelOutput = Dense(1, activation="linear")(tagModel)

concatenated = Concatenate(axis=-1)([imgModelOutput, tagModelOutput])

concatModel = Flatten()(concatenated)
concatModel = Dense(64, activation="relu")(concatModel)
concatModel = Dense(64, activation="relu")(concatModel)
concatModel = Dense(
    2, activation="relu", kernel_initializer=tf.keras.initializers.GlorotNormal()
)(concatModel)
output = Dense(1, activation="linear")(concatModel)

model = Model([img_input, tag_input], output)

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    min_delta=0,
    patience=5,
    verbose=0,
    mode="auto",
    baseline=None,
    restore_best_weights=True,
)

model.compile(
    optimizer="adam", loss="mse", metrics=[tf.keras.metrics.RootMeanSquaredError()]
)
model.summary()




## === cell 4
model.fit(
    [imgTrainX, featuresTrainX],
    trainY,
    validation_data=([imgValX, featuresValX], valY),
    epochs=30,
    batch_size=32,
    callbacks=[earlyStopping],
    verbose=1,
)




## === cell 5
testPred = model.predict([imgTestX, featuresTestX], batch_size=32, verbose=1)
testPred = np.clip(testPred.reshape(-1), 0.0, 100.0)




## === cell 6
submissionDF = pd.DataFrame(
    {"Id": testCSV.Id.values, "Pawpularity": testPred.astype(np.float32)}
)
submissionDF.to_csv("submission.csv", index=False)
print(submissionDF.head())
print("Wrote submission.csv with shape:", submissionDF.shape)
