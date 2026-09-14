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

20.1639

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 20.1639) has done: 'I fixed the import error (removed the Jupyter magic and guarded TensorFlow imports), kept the original RandomForest workflow, and added a new cell that creates predictions for the test set and writes a proper `submission.csv` with the required `Id, Pawpularity` columns. All later CNN‑related cells are omitted to prevent further crashes, so the script now runs end‑to‑end and produces a valid submission file.'

# 9. Code solution

## === cell 0
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import time
import math
import cv2
from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    from tensorflow.keras.layers.experimental import preprocessing
except Exception:
    tf = None
    keras = None
    layers = None
    preprocessing = None



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
path = "../input/petfinder-pawpularity-score/"

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
sns.histplot(data=train_df, x="Pawpularity", bins=100, kde=True)
plt.axvline(train_df["Pawpularity"].mean(), c="red", ls="-", lw=3, label="Mean")
plt.axvline(train_df["Pawpularity"].median(), c="blue", ls="-", lw=3, label="Median")
plt.title("Distribution of Pawpularity Scores")
plt.legend()
plt.show()



## === cell 4
feature_variables = train_df.columns.values.tolist()
for i in feature_variables[1:-1]:
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    sns.boxplot(ax=ax[0], data=train_df, x=i, y="Pawpularity")
    sns.histplot(ax=ax[1], data=train_df, x="Pawpularity", hue=i, kde=True)
    plt.suptitle(i, fontsize=20)
    plt.show()



## === cell 5
for i in range(3):
    image_path = train_jpg[i]
    id_stem = Path(image_path).stem
    pawpularity_by_id = train_df.loc[train_df["Id"] == id_stem, "Pawpularity"].iloc[0]
    image_array = plt.imread(image_path)
    plt.figure(figsize=(8, 8))
    plt.imshow(image_array)
    plt.title(f"{id_stem}, Pawpularity score: {pawpularity_by_id}")
    plt.axis("off")
    plt.show()




## === cell 6
def pawpularity_pics(
    df=pd.DataFrame, num_images=int, desired_pawpularity=int, random_state=int
):
    """Display a few images whose pawpularity is close to a target value."""
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
        img_id = random_sample.iloc[i]["Id"]
        img_path = f"../input/petfinder-pawpularity-score/train/{img_id}.jpg"
        paw = random_sample.iloc[i]["Pawpularity"]
        img = plt.imread(img_path)
        plt.subplot(1, num_images, i + 1)
        plt.title(paw)
        plt.axis("off")
        plt.imshow(img)
    plt.show()




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
    f"Dimensions:\n X_train:{X_train.shape}\n X_valid:{X_valid.shape}"
    f"\n y_train:{y_train.shape}\n y_valid:{y_valid.shape}"
)



## === cell 13
RF = RandomForestRegressor(n_estimators=200, max_depth=None, random_state=0)

start = time.time()
RF.fit(X_train, y_train)
stop = time.time()

RF_pred = RF.predict(X_valid)

print(f"Training time: {round(stop - start,3)} seconds")
RF_RMSE = math.sqrt(mean_squared_error(y_valid, RF_pred))
print(f"RF_RMSE: {round(RF_RMSE,3)}")



## === cell 14
X_test = test_df.drop(["Id"], axis=1)
test_pred = RF.predict(X_test)

submission = pd.DataFrame({"Id": test_df["Id"], "Pawpularity": test_pred})

submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
submission.head()



## === cell 15
pass
