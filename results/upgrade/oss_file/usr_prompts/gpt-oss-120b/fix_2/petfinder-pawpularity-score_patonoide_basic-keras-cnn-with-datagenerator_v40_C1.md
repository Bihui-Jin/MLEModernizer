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

30.2035

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import numpy as np
import pandas as pd
import cv2 as cv
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    MaxPooling2D,
    Activation,
    Flatten,
    Conv2D,
    Dropout,
)
from tensorflow.keras.optimizers import SGD
from tensorflow import keras

BASE_INPUT = os.path.abspath("./input")
TRAIN_CSV = os.path.join(BASE_INPUT, "petfinder-pawpularity-score", "train.csv")
TEST_CSV = os.path.join(BASE_INPUT, "petfinder-pawpularity-score", "test.csv")
TRAIN_PATH = os.path.join(BASE_INPUT, "petfinder-pawpularity-score", "train")
TEST_PATH = os.path.join(BASE_INPUT, "petfinder-pawpularity-score", "test")

train_data = pd.read_csv(TRAIN_CSV).to_numpy()
test_data = pd.read_csv(TEST_CSV).to_numpy()

dimensions = (128, 128, 1)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class DataGenerator(keras.utils.Sequence):
    """Generates batches of images (and labels when available)."""

    def __init__(
        self,
        directory,
        list_IDs,
        batch_size=8,
        dim=dimensions,
        n_channels=3,
        shuffle=True,
    ):
        self.directory = directory
        self.list_IDs = list_IDs
        self.batch_size = batch_size
        self.dim = dim
        self.n_channels = n_channels
        self.shuffle = shuffle
        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        X = np.empty((self.batch_size, *self.dim), dtype=np.float32)
        y = []

        for batch_number, entry in enumerate(list_IDs_temp):
            img_path = os.path.join(self.directory, f"{entry[0]}.jpg")
            img = cv.imread(img_path, cv.IMREAD_GRAYSCALE)
            if img is None:
                img = np.zeros((self.dim[0], self.dim[1]), dtype=np.uint8)
            img = cv.resize(img, (self.dim[0], self.dim[1]))
            img = img.reshape(self.dim)
            X[batch_number] = img.astype(np.float32) / 255.0

            if len(entry) >= 14:
                y.append(entry[13])

        y = np.array(y, dtype=np.float32)
        return X, y

    def __getitem__(self, index):
        batch_idxs = self.indexes[
            index * self.batch_size : (index + 1) * self.batch_size
        ]
        batch_entries = [self.list_IDs[k] for k in batch_idxs]

        X, y = self.__data_generation(batch_entries)

        if y.size == 0:
            return X
        return X, y




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1598210275.py in <cell line: 0>()
----> 1 class DataGenerator(keras.utils.Sequence):
      2     """Generates batches of images (and labels when available)."""
      3 
      4     def __init__(
      5         self,

/tmp/ipykernel_55/1598210275.py in DataGenerator()
      7         list_IDs,
      8         batch_size=8,
----> 9         dim=dimensions,
     10         n_channels=3,
     11         shuffle=True,

NameError: name 'dimensions' is not defined

## === cell 2
model = Sequential()
model.add(keras.Input(shape=dimensions))

model.add(Conv2D(filters=128, kernel_size=(15, 15), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(filters=128, kernel_size=(11, 11), padding="same"))
model.add(Activation("relu"))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Flatten())
model.add(Dense(16, activation="relu"))
model.add(Dropout(0.45))

model.add(Dense(16, activation="relu"))

model.add(Dense(1))

lr_schedule = keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=1e-4, decay_steps=100000, decay_rate=1.2, staircase=True
)

opt = SGD(learning_rate=lr_schedule)
loss_fn = keras.losses.MeanAbsoluteError()
model.compile(optimizer=opt, loss=loss_fn, metrics=["mae", "mse"])



## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1402842848.py in <cell line: 0>()
      1 # Build the CNN model (identical to the original architecture)
      2 model = Sequential()
----> 3 model.add(keras.Input(shape=dimensions))
      4 
      5 model.add(Conv2D(filters=128, kernel_size=(15, 15), padding="same"))

NameError: name 'dimensions' is not defined

## === cell 3
train_gen = DataGenerator(TRAIN_PATH, train_data, batch_size=64)
model.fit(train_gen, epochs=10, verbose=2)

test_gen = DataGenerator(TEST_PATH, test_data, batch_size=1, shuffle=False)
results = model.predict(test_gen, verbose=0)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1138804713.py in <cell line: 0>()
      1 # Generators
----> 2 train_gen = DataGenerator(TRAIN_PATH, train_data, batch_size=64)
      3 # No validation generator because test labels are unknown
      4 model.fit(train_gen, epochs=10, verbose=2)
      5 

NameError: name 'DataGenerator' is not defined

## === cell 4
submission = []
for idx, pred in enumerate(results):
    submission.append([test_data[idx][0], float(pred[0])])

df = pd.DataFrame(submission, columns=["Id", "Pawpularity"])
df = df.sort_values("Id")
df.to_csv("submission.csv", index=False)
print(df.head())

## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3058175959.py in <cell line: 0>()
      1 # Prepare submission
      2 submission = []
----> 3 for idx, pred in enumerate(results):
      4     submission.append([test_data[idx][0], float(pred[0])])
      5 

NameError: name 'results' is not defined
