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
ImageHash==4.3.1
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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

20.48251

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
from tqdm import tqdm

import pandas as pd
import numpy as np
import imagehash
from PIL import Image

from sklearn.model_selection import StratifiedShuffleSplit

import tensorflow as tf
from tensorflow.keras import Model
from tensorflow.keras.layers import (
    Input,
    Conv2D,
    MaxPooling2D,
    Dropout,
    GlobalAveragePooling2D,
    Dense,
    Concatenate,
    Flatten,
)


IMG_SIZE = 64  # image size used throughout the notebook




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def getImagePath(imgId, dataType):
    if dataType == "train":
        return os.path.join(
            "../input/petfinder-pawpularity-score/train", imgId + ".jpg"
        )
    if dataType == "test":
        return os.path.join("../input/petfinder-pawpularity-score/test", imgId + ".jpg")
    raise ValueError("dataType must be 'train' or 'test'")




## === cell 2
trainCSV = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
testCSV = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")

trainImgDir = "../input/petfinder-pawpularity-score/train"
simImgDict = {}

for imgName in tqdm(os.listdir(trainImgDir)):
    if not imgName.lower().endswith(".jpg"):
        continue
    img_path = os.path.join(trainImgDir, imgName)
    try:
        hash_val = imagehash.phash(Image.open(img_path))
    except Exception:
        continue
    if hash_val not in simImgDict:
        simImgDict[hash_val] = [imgName, []]
    else:
        simImgDict[hash_val][1].append(imgName)

print("Len of train images before deduplication:", len(trainCSV))
for _, v in simImgDict.items():
    if v[1]:
        for dup_name in v[1]:
            dup_id = dup_name[:-4]  # strip .jpg
            trainCSV.drop(trainCSV.index[trainCSV["Id"] == dup_id], inplace=True)

del simImgDict
print("Len of train images after deduplication:", len(trainCSV))
trainCSV.reset_index(drop=True, inplace=True)



## === cell 3
split = StratifiedShuffleSplit(n_splits=1, test_size=0.25, random_state=42)
for train_idx, val_idx in split.split(trainCSV, trainCSV["Pawpularity"]):
    trainDF = trainCSV.loc[train_idx].reset_index(drop=True)
    valDF = trainCSV.loc[val_idx].reset_index(drop=True)

del trainCSV, split




## === cell 4
trainY = trainDF["Pawpularity"]
valY = valDF["Pawpularity"]

trainX = trainDF.drop("Pawpularity", axis=1)
valX = valDF.drop("Pawpularity", axis=1)

imgTrainXList = []
for imgId in tqdm(trainX.Id):
    img = (
        cv2.cvtColor(
            cv2.resize(cv2.imread(getImagePath(imgId, "train")), (IMG_SIZE, IMG_SIZE)),
            cv2.COLOR_BGR2RGB,
        )
        / 255.0
    )
    imgTrainXList.append(img)
imgTrainX = np.array(imgTrainXList).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
del imgTrainXList

imgValXList = []
for imgId in tqdm(valX.Id):
    img = (
        cv2.cvtColor(
            cv2.resize(cv2.imread(getImagePath(imgId, "train")), (IMG_SIZE, IMG_SIZE)),
            cv2.COLOR_BGR2RGB,
        )
        / 255.0
    )
    imgValXList.append(img)
imgValX = np.array(imgValXList).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
del imgValXList

imgTestXList = []
for imgId in tqdm(testCSV.Id):
    img = (
        cv2.cvtColor(
            cv2.resize(cv2.imread(getImagePath(imgId, "test")), (IMG_SIZE, IMG_SIZE)),
            cv2.COLOR_BGR2RGB,
        )
        / 255.0
    )
    imgTestXList.append(img)
imgTestX = np.array(imgTestXList).reshape(-1, IMG_SIZE, IMG_SIZE, 3)
del imgTestXList

featuresTrainX = trainX.drop("Id", axis=1)
featuresValX = valX.drop("Id", axis=1)
featuresTestX = testCSV.drop("Id", axis=1)

del trainX, valX, trainDF, valDF




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1977607487.py in <cell line: 0>()
     10 for imgId in tqdm(trainX.Id):
     11     img = (
---> 12         cv2.cvtColor(
     13             cv2.resize(cv2.imread(getImagePath(imgId, "train")), (IMG_SIZE, IMG_SIZE)),
     14             cv2.COLOR_BGR2RGB,

NameError: name 'cv2' is not defined

## === cell 5
img_input = Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="img_input")
x = Conv2D(32, (3, 3), activation="relu")(img_input)
x = Conv2D(96, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.5)(x)

x = Conv2D(192, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.1)(x)

x = Conv2D(224, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.1)(x)

x = Conv2D(32, (3, 3), activation="relu")(x)
x = MaxPooling2D((2, 2))(x)
x = Dropout(0.0)(x)

x = GlobalAveragePooling2D()(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
img_output = Dense(1, activation="linear")(x)

tag_input = Input(shape=featuresTrainX.shape[1:], name="tag_input")
y = Dense(128, activation="relu")(tag_input)
y = Dense(192, activation="relu")(y)
y = Dense(32, activation="relu")(y)
y = Dense(192, activation="relu")(y)
tag_output = Dense(1, activation="linear")(y)

combined = Concatenate(axis=-1)([img_output, tag_output])
z = Dense(48, activation="relu")(combined)
z = Dense(48, activation="relu")(z)
z = Dense(48, activation="relu")(z)
z = Dense(48, activation="relu")(z)
final_output = Dense(1, activation="linear")(z)

model = Model(inputs=[img_input, tag_input], outputs=final_output)
model.summary()

earlyStopping = tf.keras.callbacks.EarlyStopping(
    monitor="val_root_mean_squared_error",
    patience=15,
    restore_best_weights=True,
    verbose=0,
)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError()],
)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3400751238.py in <cell line: 0>()
     23 img_output = Dense(1, activation="linear")(x)
     24 
---> 25 tag_input = Input(shape=featuresTrainX.shape[1:], name="tag_input")
     26 y = Dense(128, activation="relu")(tag_input)
     27 y = Dense(192, activation="relu")(y)

NameError: name 'featuresTrainX' is not defined

## === cell 6
model.fit(
    x=[imgTrainX, featuresTrainX],
    y=trainY,
    epochs=40,
    batch_size=64,
    validation_data=([imgValX, featuresValX], valY),
    callbacks=[earlyStopping],
    verbose=1,
)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2845692307.py in <cell line: 0>()
      1 # Train the model
----> 2 model.fit(
      3     x=[imgTrainX, featuresTrainX],
      4     y=trainY,
      5     epochs=40,

NameError: name 'model' is not defined

## === cell 7
testPred = model.predict([imgTestX, featuresTestX])
submissionData = [[pid, pred[0]] for pid, pred in zip(testCSV.Id, testPred)]
submissionDF = pd.DataFrame(submissionData, columns=["Id", "Pawpularity"])
submissionDF.to_csv("submission.csv", index=False)
print("Submission saved to submission.csv")

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3468727291.py in <cell line: 0>()
      1 # Predict on test set and write submission file
----> 2 testPred = model.predict([imgTestX, featuresTestX])
      3 submissionData = [[pid, pred[0]] for pid, pred in zip(testCSV.Id, testPred)]
      4 submissionDF = pd.DataFrame(submissionData, columns=["Id", "Pawpularity"])
      5 submissionDF.to_csv("submission.csv", index=False)

NameError: name 'model' is not defined
