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

geopandas==0.14.4
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

20.5146

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import tensorflow as tf
from tensorflow.keras.layers import (
    Dense,
    Activation,
    Flatten,
    Dropout,
    BatchNormalization,
    Conv2D,
    MaxPooling2D,
)
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras import regularizers, optimizers
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import concatenate
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt
from matplotlib import image
import seaborn as sns
import cv2


## === cell 1
train_csv_path = "../input/petfinder-pawpularity-score/train.csv"
test_csv_path = "../input/petfinder-pawpularity-score/test.csv"

train_image_path = "../input/petfinder-pawpularity-score/train/"
test_image_path = "../input/petfinder-pawpularity-score/test/"


## === cell 2
train_df= pd.read_csv(train_csv_path)
test_df= pd.read_csv(test_csv_path)

train_df["Id"]=train_df["Id"].apply(lambda Id: Id+".jpg")
test_df["Id"]=test_df["Id"].apply(lambda Id: Id+".jpg")


train_df.head()


## === cell 3
train_df.isna().sum()


## === cell 4
train_df.describe()


## === cell 5
train_df.info()


## === cell 6
fig, axs = plt.subplots(3, 3, figsize= (20,20))

for i in range(9):
    img_path = train_image_path + train_df.iloc[i]["Id"]
    img = image.imread(img_path)
    row, col = i // 3, i % 3
    axs[row][col].imshow(img)
    axs[row][col].set_title("Pawpularity :" + str(train_df.iloc[i]["Pawpularity"]))
    axs[row][col].axis("off")


## === cell 7
fig, aix = plt.subplots(3,4)
fig.set_figheight(15)
fig.set_figwidth(15)

df_columns = train_df.drop(["Id","Pawpularity"], axis="columns").columns
for index, df_col in enumerate(df_columns):
    row = index // 4
    col = index % 4
    aix[row][col].hist(train_df[df_col])
    aix[row][col].set_title(df_col)


## === cell 8
plt.hist(train_df["Pawpularity"])
plt.title("Pawpularity")
plt.grid(True)
plt.show()


## === cell 9
train_df[["Pawpularity"]] = train_df[["Pawpularity"]] / 100 


## === cell 10
train_X, validation_X, train_y, Validation_Y = train_test_split(train_df.drop(columns=["Pawpularity"] , axis="columns"), train_df[["Pawpularity"]], test_size=0.2, shuffle=5)


## === cell 11
train_X.shape, validation_X.shape, test_df.shape


## === cell 12
train_X = train_X.reset_index(drop =True)
train_X_mlp= train_X.drop(columns=["Id"], axis="columns")
validation_X_mlp= validation_X.drop(columns=["Id"], axis="columns")


## === cell 13
def get_image_array(train_df, validation_df, test_df):
    train_images = []
    validation_images = []
    test_images = []
    train_image_path = '../input/petfinder-pawpularity-score/train/' 
    test_image_path  =  '../input/petfinder-pawpularity-score/test/'
   
    for img_name in train_df['Id']:
        img_path = f"{train_image_path}{img_name}"
        image = cv2.imread(img_path)
        image = cv2.resize(image, (64,64))
        train_images.append(image)
   
    for validation_img_name in validation_df['Id']:
        val_img_path = f"{train_image_path}{validation_img_name}"
        val_image = cv2.imread(val_img_path)
        val_image = cv2.resize(val_image, (64,64))
        validation_images.append(val_image)
    
    for test_img_name in test_df['Id']:
        test_img_path = f"{test_image_path}{test_img_name}"
        test_img = cv2.imread(test_img_path)
        test_img = cv2.resize(test_img, (64,64))
        test_images.append(test_img)
       
    return np.array(train_images), np.array(validation_images), np.array(test_images)

train_images, validation_images, test_images = get_image_array(train_X, validation_X, test_df)
train_images = train_images / 255.0
validation_images = validation_images / 255.0
test_images = test_images / 255.0


## === cell 14
test_df= test_df.drop(columns=["Id"], axis="columns")


## === cell 15
def create_mlp(dims):
    model = Sequential([
        Dense(512, input_dim=dims, activation="relu"),
        Dropout(0.2),
        Dense(256, activation="relu"),
        Dropout(0.2),
        Dense(4, activation="relu")
    ])
    return model


## === cell 16
def create_cnn(dims):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(filters=128,  kernel_size=(3,3), input_shape=dims, activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Conv2D(filters=40, kernel_size=(3,3), activation='relu'),
        tf.keras.layers.MaxPool2D(),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(40, activation='relu'),
        tf.keras.layers.Dense(4)
    ])
    return model


## === cell 17
mlp = create_mlp(12)
cnn = create_cnn((64, 64, 3))
combinedInput = concatenate([mlp.output, cnn.output])


## === cell 18
x = Dense(4, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)


## === cell 19
model = Model(inputs=[mlp.input, cnn.input], outputs=x)


## === cell 20
model.summary()


## === cell 21
opt = tf.keras.optimizers.Adam(learning_rate=1e-3, decay=1e-3/200)
model.compile(loss='mse', optimizer=opt, metrics = tf.keras.metrics.RootMeanSquaredError())


## === cell 22
model.fit(x=[train_X_mlp.values, train_images], y=train_y,validation_data=([validation_X_mlp.values, validation_images], Validation_Y), epochs=100, batch_size=100)


## === cell 23
predictions = model.predict([test_df.values, test_images])
predictions = predictions * 100
predictions


## === cell 24
submission_output = pd.read_csv('../input/petfinder-pawpularity-score/sample_submission.csv')

predictions_round = np.round(predictions, 2)
submission_output['Pawpularity'] = predictions_round

submission_output.to_csv('submission.csv', index=False)
