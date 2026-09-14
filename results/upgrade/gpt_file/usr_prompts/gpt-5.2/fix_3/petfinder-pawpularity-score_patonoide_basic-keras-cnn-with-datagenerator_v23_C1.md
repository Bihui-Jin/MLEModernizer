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

22.29563

# 6. Current score

35.58657

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 35.56506) has done: 'I fix the environment crash caused by the TensorFlow/protobuf incompatibility by pinning `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` before importing TensorFlow/Keras. Then I fix the generator/fit error by ensuring the validation generator always returns `(X, y)` during training (using training data for validation), and I keep the test generator as input-only for `predict()`. Finally, I ensure predictions align 1:1 with `test.csv` (no dropped remainder) and write a valid `submission.csv` with the exact required columns, clipping predictions to the valid [0, 100] range for stability under RMSE.'
- What this solution (achieved 35.58657) has done: 'I fix the TensorFlow/protobuf crash by switching the protobuf implementation to `upb` (the default for protobuf 4+ and TF 2.18) and, if needed, fall back to `python` before importing TensorFlow, ensuring the import succeeds in this Kaggle runtime. I also ensure the image resize uses (width, height) ordering (OpenCV expects `(w, h)`), which is a small but legitimate correctness fix that typically improves model signal and should move RMSE down toward your target without changing the model/training approach. Finally, I keep the generator semantics intact (train/val return `(X,y)`, test returns `X`) and ensure the submission is written as `submission.csv` with correct alignment and clipping.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "upb")

import numpy as np
import pandas as pd
import cv2 as cv

try:
    import tensorflow as tf
    from tensorflow import keras
except Exception as e:
    os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
    import tensorflow as tf
    from tensorflow import keras

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    MaxPooling2D,
    Activation,
    Flatten,
    Conv2D,
    Dropout,
)
from tensorflow.keras.optimizers import Adam

np.random.seed(42)
tf.random.set_seed(42)

INPUT_DIR = "../input/petfinder-pawpularity-score"

test_df = pd.read_csv(f"{INPUT_DIR}/test.csv")
train_df = pd.read_csv(f"{INPUT_DIR}/train.csv")

test_data = test_df.to_numpy()
train_data = train_df.to_numpy()

TEST_PATH = f"{INPUT_DIR}/test"
TRAIN_PATH = f"{INPUT_DIR}/train"

print("train_data shape:", train_data.shape, "test_data shape:", test_data.shape)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
class DataGenerator(keras.utils.Sequence):
    """
    Minimal, stable generator:
    - `return_y` controls whether generator returns labels.
    - Uses ceil in __len__ so predictions cover ALL samples (no dropped remainder).
    - Always returns (X, y) when return_y=True.
    """

    def __init__(
        self,
        directory,
        list_IDs,
        n_channels=3,
        batch_size=8,
        dim=(256, 256, 3),
        shuffle=True,
        return_y=True,
    ):
        self.dim = dim
        self.directory = directory
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.return_y = return_y
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(len(self.list_IDs) / self.batch_size))

    def on_epoch_end(self):
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        cur_bs = len(list_IDs_temp)
        X = np.empty((cur_bs, *self.dim), dtype=np.float32)

        y = None
        if self.return_y:
            y = np.empty((cur_bs,), dtype=np.float32)

        for batch_number, id_temp_entry in enumerate(list_IDs_temp):
            path = self.directory + "/" + str(id_temp_entry[0]) + ".jpg"
            image = cv.imread(path)

            if image is None:
                image = np.zeros((self.dim[0], self.dim[1], 3), dtype=np.uint8)
            else:
                image = cv.resize(image, (self.dim[1], self.dim[0]))

            X[batch_number] = image.astype(np.float32) / 255.0

            if self.return_y:
                y[batch_number] = float(id_temp_entry[13])

        if self.return_y:
            return X, y
        return X

    def __getitem__(self, index):
        indexes = self.indexes[index * self.batch_size : (index + 1) * self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indexes]
        return self.__data_generation(list_IDs_temp)




## === cell 2
model = Sequential()
model.add(keras.Input(shape=(256, 256, 3)))

model.add(Conv2D(filters=64, padding="same", kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(filters=64, padding="same", kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))

model.add(Conv2D(filters=64, padding="same", kernel_size=(3, 3)))
model.add(Activation("relu"))
model.add(MaxPooling2D(2, 2))

model.add(Dropout(0.25))

model.add(Flatten())
model.add(Dense(256))
model.add(Activation("relu"))

model.add(Dense(1))

opt = Adam(learning_rate=0.000001)
loss_fn = keras.losses.MeanSquaredError()
model.compile(loss=loss_fn, optimizer=opt, metrics=["mse"])

model.summary()



## === cell 3
val_size = int(0.1 * len(train_data))
val_data = train_data[:val_size]
tr_data = train_data[val_size:]

training_generator = DataGenerator(
    TRAIN_PATH, tr_data, batch_size=32, shuffle=True, return_y=True
)
validation_generator = DataGenerator(
    TRAIN_PATH, val_data, batch_size=32, shuffle=False, return_y=True
)

history = model.fit(
    training_generator, validation_data=validation_generator, epochs=1, verbose=1
)



## === cell 4
test_generator = DataGenerator(
    TEST_PATH, test_data, batch_size=32, shuffle=False, return_y=False
)
preds = model.predict(test_generator, verbose=1).reshape(-1)

preds = preds[: len(test_df)]

preds = np.clip(preds, 0.0, 100.0)

submission_df = pd.DataFrame(
    {"Id": test_df["Id"].values, "Pawpularity": preds.astype(float)}
)

submission_df.to_csv("submission.csv", index=False)
print(submission_df.head())
print("Wrote submission.csv with shape:", submission_df.shape)
