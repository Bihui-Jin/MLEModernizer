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
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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

21.88742

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import tensorflow as tf
import math
import random
import shutil

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns

from PIL import Image
from tensorflow import keras
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers
from tensorflow.keras import activations
from tensorflow.keras.utils import Sequence
from tensorflow.keras.optimizers import Adam

from sklearn.model_selection import train_test_split
from IPython.display import display, Markdown, Latex



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
plt.rc("font", size=15)
plt.rc("axes", titlesize=18)
plt.rc("xtick", labelsize=10)
plt.rc("ytick", labelsize=10)

sns.set(font_scale=1.2)
sns.set_style("whitegrid")

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4157817424.py in <cell line: 0>()
      7 sns.set_style("whitegrid")
      8 
----> 9 os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
     10 
     11 

NameError: name 'os' is not defined

## === cell 2
class Cfg:
    RANDOM_STATE = 2021
    TRAIN_DATA = "../input/petfinder-pawpularity-score/train.csv"
    TEST_DATA = "../input/petfinder-pawpularity-score/test.csv"
    SUBMISSION = "../input/petfinder-pawpularity-score/sample_submission.csv"
    IMG_FOLDER = "../input/petfinder-pawpularity-score/train"
    IMG_TEST_FOLDER = "../input/petfinder-pawpularity-score/test"
    IMG_RESIZE_FOLDER = "./resized"
    SUBMISSION_FILE = "./submission.csv"

    SAMPLE_FRAC = 1
    NUM_EPOCHS = 10
    LEARNING_RATE = 0.0001
    TEST_SIZE = 0.3
    BATCH_SIZE = 64
    IMG_SIZE = 128

    INDEX = "Id"
    TARGET = "Pawpularity"
    FEATURES = [
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




## === cell 3
if not os.path.isdir(Cfg.IMG_RESIZE_FOLDER):
    os.makedirs(Cfg.IMG_RESIZE_FOLDER)




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2999718078.py in <cell line: 0>()
----> 1 if not os.path.isdir(Cfg.IMG_RESIZE_FOLDER):
      2     os.makedirs(Cfg.IMG_RESIZE_FOLDER)
      3 
      4 

NameError: name 'os' is not defined

## === cell 4
def read_data(
    train_file: str = Cfg.TRAIN_DATA, test_file: str = Cfg.TEST_DATA
) -> (pd.DataFrame, pd.DataFrame):
    """Reads the csv files `train.csv` and `test.csv` and returns
    them as pandas data frames.
    """
    train_df = pd.read_csv(Cfg.TRAIN_DATA, index_col=Cfg.INDEX)
    test_df = pd.read_csv(Cfg.TEST_DATA, index_col=Cfg.INDEX)

    return train_df, test_df


train_df, test_df = read_data()



## === cell 5
train_df



## === cell 6
test_df



## === cell 7
train_df.describe().drop("count")



## === cell 8
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 5))

sns.histplot(data=train_df, x=Cfg.TARGET, bins=30, legend=True, kde=True, ax=ax[0])

ax[0].set_title("Target Distribution")
ax[0].set_xlabel("Pawpularity")
ax[0].set_ylabel("Count")

sns.boxplot(data=train_df, y=Cfg.TARGET, ax=ax[1])

plt.show()



## === cell 9
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

for f, ax in zip(Cfg.FEATURES, axes.flatten()):
    sns.histplot(
        data=train_df,
        x=Cfg.TARGET,
        bins=30,
        hue=f,
        legend=True,
        kde=True,
        ax=ax,
        alpha=0.3,
    )

fig.tight_layout()
plt.show()



## === cell 10
fig, axes = plt.subplots(nrows=3, ncols=4, figsize=(20, 15))

for f, ax in zip(Cfg.FEATURES, axes.flatten()):
    sns.countplot(data=train_df, x=f, alpha=0.8, ax=ax)

fig.tight_layout()
plt.show()



## === cell 11
corr_df = train_df.corr()

fig, ax = plt.subplots(figsize=(15, 15))

mask = np.triu(np.ones_like(corr_df, dtype=bool))
cmap = sns.diverging_palette(230, 20, as_cmap=True)

sns.heatmap(
    corr_df,
    mask=mask,
    cmap=cmap,
    vmin=-0.75,
    vmax=0.75,
    center=0,
    square=True,
    annot=True,
    fmt="0.0",
    linewidths=0.5,
)

fig.tight_layout()
plt.show()




## === cell 12
def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
    resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, f"{image_id}.jpg")
    if os.path.isfile(resized_path):
        img = Image.open(resized_path)
        return img

    img_path = os.path.join(image_folger, f"{image_id}.jpg")
    img = Image.open(img_path)
    img = img.resize((Cfg.IMG_SIZE, Cfg.IMG_SIZE))
    img.save(resized_path)
    return img




## === cell 13
def plot_images(data, nrows=5, ncols=5, figsize=(15, 15)):
    """ """
    indices = data.sample(nrows * ncols).index

    fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
    for index, ax in zip(indices, axes.flatten()):
        img = get_image(index)
        ax.imshow(img)

        ax.set_title(train_df.loc[index][Cfg.TARGET], fontsize=18)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    fig.tight_layout()
    plt.show()




## === cell 14
pawpularity_range = [0, 20, 40, 60, 80, 100]
query = (
    lambda i: f"{pawpularity_range[i]} < Pawpularity and Pawpularity <= {pawpularity_range[i+1]}"
)

for i in range(0, 5):
    display(
        Markdown(
            f"### Pawpularity `{pawpularity_range[i]}` - `{pawpularity_range[i+1]}`"
        )
    )
    df = train_df.query(query(i))
    plot_images(df, nrows=1, ncols=5, figsize=(15, 5))




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1037844153.py in <cell line: 0>()
     11     )
     12     df = train_df.query(query(i))
---> 13     plot_images(df, nrows=1, ncols=5, figsize=(15, 5))
     14 
     15 

/tmp/ipykernel_55/2519191972.py in plot_images(data, nrows, ncols, figsize)
      5     fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=figsize)
      6     for index, ax in zip(indices, axes.flatten()):
----> 7         img = get_image(index)
      8         ax.imshow(img)
      9 

/tmp/ipykernel_55/3320905384.py in get_image(image_id, image_folger, data, resize)
      1 def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
----> 2     resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, f"{image_id}.jpg")
      3     if os.path.isfile(resized_path):
      4         img = Image.open(resized_path)
      5         return img

NameError: name 'os' is not defined

## === cell 15
class DataGenerator(Sequence):
    """ """

    def __init__(
        self, data, target=None, img_folder=Cfg.IMG_FOLDER, batch_size=Cfg.BATCH_SIZE
    ):
        self.data = data
        self.target = target
        self.batch_size = batch_size
        self.img_folder = img_folder

    def __len__(self):
        return math.ceil(len(self.data) / self.batch_size)

    def __getitem__(self, idx):
        start_idx = idx * self.batch_size
        end_idx = (idx + 1) * self.batch_size
        ids = self.data[start_idx:end_idx].index.values

        images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
        meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)

        if self.target is None:
            return (images, meta)

        target = np.array(self.target[start_idx:end_idx]).astype(np.float32)
        return ((images, meta), target)




## === cell 16
def get_image_model(img_size=Cfg.IMG_SIZE, n_channel=3):
    """ """
    inputs = layers.Input((img_size, img_size, n_channel))
    x = inputs

    x = layers.Conv2D(filters=32, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(filters=64, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(filters=128, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Conv2D(filters=256, kernel_size=(3, 3), activation="relu")(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)

    x = layers.Flatten()(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.Dropout(0.5)(x)

    outputs = x
    model = keras.Model(inputs=inputs, outputs=outputs, name="image_cnn_model")

    return model




## === cell 17
image_model = get_image_model()
image_model.summary()




## === cell 18
def get_meta_model(n_meta_features=12):
    """ """
    inputs = layers.Input(shape=((n_meta_features,)))
    x = inputs

    x = layers.Dense(12, activation="relu")(x)
    x = layers.Dense(24, activation="relu")(x)
    x = layers.Dense(12, activation="relu")(x)

    outputs = x
    model = keras.Model(inputs=inputs, outputs=outputs, name="meta_nn_model")

    return model




## === cell 19
meta_model = get_meta_model()
meta_model.summary()




## === cell 20
def get_model(image_model, meta_model):
    """ """
    x = layers.Concatenate(axis=1)([image_model.output, meta_model.output])
    x = layers.Dense(1, activation="linear")(x)
    output = x

    model = keras.Model(inputs=[image_model.input, meta_model.input], outputs=output)
    return model




## === cell 21
model = get_model(image_model, meta_model)
model.summary()



## === cell 22
data = train_df.sample(frac=Cfg.SAMPLE_FRAC, random_state=Cfg.RANDOM_STATE)

X_train, X_val, y_train, y_val = train_test_split(
    data[Cfg.FEATURES],
    data[Cfg.TARGET],
    test_size=Cfg.TEST_SIZE,
    random_state=Cfg.RANDOM_STATE,
)

train_generator = DataGenerator(X_train, y_train)
val_generator = DataGenerator(X_val, y_val)



## === cell 23
model.compile(
    optimizer=keras.optimizers.Adam(
        learning_rate=Cfg.LEARNING_RATE,
    ),
    loss=keras.losses.MeanSquaredError(),
    metrics=[keras.metrics.RootMeanSquaredError(name="rmse")],
)

callbacks = [keras.callbacks.EarlyStopping(monitor="loss", patience=3)]



## === cell 24
result = model.fit(
    train_generator,
    epochs=Cfg.NUM_EPOCHS,
    validation_data=val_generator,
    callbacks=callbacks,
    verbose=2,
)



## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2717092727.py in <cell line: 0>()
----> 1 result = model.fit(
      2     train_generator,
      3     epochs=Cfg.NUM_EPOCHS,
      4     validation_data=val_generator,
      5     callbacks=callbacks,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/1404647577.py in __getitem__(self, idx)
     18         ids = self.data[start_idx:end_idx].index.values
     19 
---> 20         images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
     21         meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)
     22 

/tmp/ipykernel_55/1404647577.py in <listcomp>(.0)
     18         ids = self.data[start_idx:end_idx].index.values
     19 
---> 20         images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
     21         meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)
     22 

/tmp/ipykernel_55/3320905384.py in get_image(image_id, image_folger, data, resize)
      1 def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
----> 2     resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, f"{image_id}.jpg")
      3     if os.path.isfile(resized_path):
      4         img = Image.open(resized_path)
      5         return img

NameError: name 'os' is not defined

## === cell 25
y_pred = model.predict(val_generator, batch_size=Cfg.BATCH_SIZE).reshape(-1)

df = pd.DataFrame({"y_pred": y_pred, "y_val": y_val})
df["error"] = np.abs(y_pred - y_val)



## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2852424335.py in <cell line: 0>()
----> 1 y_pred = model.predict(val_generator, batch_size=Cfg.BATCH_SIZE).reshape(-1)
      2 
      3 df = pd.DataFrame({"y_pred": y_pred, "y_val": y_val})
      4 df["error"] = np.abs(y_pred - y_val)
      5 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_55/1404647577.py in __getitem__(self, idx)
     18         ids = self.data[start_idx:end_idx].index.values
     19 
---> 20         images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
     21         meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)
     22 

/tmp/ipykernel_55/1404647577.py in <listcomp>(.0)
     18         ids = self.data[start_idx:end_idx].index.values
     19 
---> 20         images = np.array([np.array(get_image(id, self.img_folder)) for id in ids])
     21         meta = np.array(self.data[start_idx:end_idx][Cfg.FEATURES]).astype(np.float32)
     22 

/tmp/ipykernel_55/3320905384.py in get_image(image_id, image_folger, data, resize)
      1 def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
----> 2     resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, f"{image_id}.jpg")
      3     if os.path.isfile(resized_path):
      4         img = Image.open(resized_path)
      5         return img

NameError: name 'os' is not defined

## === cell 26
fig, ax = plt.subplots(1, 4, figsize=(23, 5))

ax[0].plot(result.history["rmse"])
ax[0].plot(result.history["val_rmse"])
ax[0].set_title("Model RMSE")
ax[0].set_ylabel("RMSE")
ax[0].set_xlabel("Epoch")
ax[0].legend(["train", "val"], loc="upper right")

sns.regplot(data=df, x="y_val", y="y_pred", ax=ax[1], x_estimator=np.mean, x_bins=30)
ax[1].set_title("Regression True vs. Pred")
ax[1].set_ylabel("Pred (val)")
ax[1].set_xlabel("True (val)")

sns.histplot(data=df, x="y_pred", bins=30, legend=True, kde=True, ax=ax[2])
ax[2].set_title("Pred Target Distribution")
ax[2].set_xlabel("Pred Pawpularity")
ax[2].set_ylabel("Count")

sns.scatterplot(data=df, x="y_pred", y="error", ax=ax[3])
ax[3].set_title("Residuals")
ax[3].set_xlabel("Prediction")
ax[3].set_ylabel("Absolute error")

plt.tight_layout()
plt.show()



## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3577164469.py in <cell line: 0>()
      1 fig, ax = plt.subplots(1, 4, figsize=(23, 5))
      2 
----> 3 ax[0].plot(result.history["rmse"])
      4 ax[0].plot(result.history["val_rmse"])
      5 ax[0].set_title("Model RMSE")

NameError: name 'result' is not defined

## === cell 27
test_ids = test_df.index.values
test_images = np.stack([np.array(get_image(i, Cfg.IMG_TEST_FOLDER)) for i in test_ids])
test_meta = test_df[Cfg.FEATURES].astype(np.float32).values

y_pred_submission = model.predict(
    [test_images, test_meta], batch_size=Cfg.BATCH_SIZE
).reshape(-1)

y_pred_submission = np.clip(y_pred_submission, 1, 100)



## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/54390228.py in <cell line: 0>()
      1 test_ids = test_df.index.values
----> 2 test_images = np.stack([np.array(get_image(i, Cfg.IMG_TEST_FOLDER)) for i in test_ids])
      3 test_meta = test_df[Cfg.FEATURES].astype(np.float32).values
      4 
      5 y_pred_submission = model.predict(

/tmp/ipykernel_55/54390228.py in <listcomp>(.0)
      1 test_ids = test_df.index.values
----> 2 test_images = np.stack([np.array(get_image(i, Cfg.IMG_TEST_FOLDER)) for i in test_ids])
      3 test_meta = test_df[Cfg.FEATURES].astype(np.float32).values
      4 
      5 y_pred_submission = model.predict(

/tmp/ipykernel_55/3320905384.py in get_image(image_id, image_folger, data, resize)
      1 def get_image(image_id, image_folger=Cfg.IMG_FOLDER, data=train_df, resize=True):
----> 2     resized_path = os.path.join(Cfg.IMG_RESIZE_FOLDER, f"{image_id}.jpg")
      3     if os.path.isfile(resized_path):
      4         img = Image.open(resized_path)
      5         return img

NameError: name 'os' is not defined

## === cell 28
submission_df = pd.DataFrame(
    {
        Cfg.INDEX: test_ids,
        Cfg.TARGET: y_pred_submission,
    }
).set_index(Cfg.INDEX)

submission_df.head()



## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/244653406.py in <cell line: 0>()
      2     {
      3         Cfg.INDEX: test_ids,
----> 4         Cfg.TARGET: y_pred_submission,
      5     }
      6 ).set_index(Cfg.INDEX)

NameError: name 'y_pred_submission' is not defined

## === cell 29
submission_df.to_csv(Cfg.SUBMISSION_FILE, index=True)



## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1779648856.py in <cell line: 0>()
----> 1 submission_df.to_csv(Cfg.SUBMISSION_FILE, index=True)
      2 

NameError: name 'submission_df' is not defined

## === cell 30
shutil.rmtree(Cfg.IMG_RESIZE_FOLDER)

## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/4266062315.py in <cell line: 0>()
----> 1 shutil.rmtree(Cfg.IMG_RESIZE_FOLDER)

/usr/lib/python3.11/shutil.py in rmtree(path, ignore_errors, onerror, dir_fd)
    740             orig_st = os.lstat(path, dir_fd=dir_fd)
    741         except Exception:
--> 742             onerror(os.lstat, path, sys.exc_info())
    743             return
    744         try:

/usr/lib/python3.11/shutil.py in rmtree(path, ignore_errors, onerror, dir_fd)
    738         # lstat()/open()/fstat() trick.
    739         try:
--> 740             orig_st = os.lstat(path, dir_fd=dir_fd)
    741         except Exception:
    742             onerror(os.lstat, path, sys.exc_info())

FileNotFoundError: [Errno 2] No such file or directory: './resized'
