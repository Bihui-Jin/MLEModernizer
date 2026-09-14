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

43.37331

# 6. Current score

22.95529

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

- What this solution (achieved 22.95529) has done: 'I fix TensorFlow import/runtime failures by pinning the protobuf Python implementation before importing TF (this resolves the `MessageFactory` error in this environment). I update deprecated `fit_generator` calls to `fit` while keeping the same generator-based training loop and epochs/steps so the core training behavior remains the same. I make image loading robust (correct path joining + handle unreadable files) to stop OpenCV resize crashes, and I ensure the test Id ordering matches `test.csv`. Finally, I clip predictions to the required [1, 100] range so the submission passes validation and is written as a proper `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
import cv2

import random

random.seed(42)
np.random.seed(42)



## === cell 1
dat = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")



## === cell 2
y = dat["Pawpularity"].to_numpy()
y = y / 100.0




## === cell 3
def read_image_rgb_128(path):
    img = cv2.imread(path)
    if img is None:
        return None
    img = cv2.resize(img, (128, 128), interpolation=cv2.INTER_AREA)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img.astype(np.float32) / 255.0




## === cell 4
class data:
    def __init__(self, path, x=128, y=128, labels=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.image_list = [t.split(".")[0] for t in os.listdir(path)]
        self.path = path
        self.batch = 0

    def load_batch(self, batch_size=1, shuffle=False):
        if shuffle:
            b = self.batch
            batch_list = self.image_list[b * batch_size : (b + 1) * batch_size]
            self.batch = b + 1
            if self.batch > len(self.image_list) // batch_size:
                self.batch = 0
        else:
            batch_list = sample(self.image_list, batch_size)

        images = []
        good_ids = []
        for image_id in batch_list:
            img_path = os.path.join(self.path, image_id + ".jpg")
            img = read_image_rgb_128(img_path)
            if img is None:
                continue
            images.append(img)
            good_ids.append(image_id)

        if len(images) == 0:
            tries = 0
            while len(images) == 0 and tries < 10:
                image_id = sample(self.image_list, 1)[0]
                img_path = os.path.join(self.path, image_id + ".jpg")
                img = read_image_rgb_128(img_path)
                if img is not None:
                    images.append(img)
                    good_ids.append(image_id)
                tries += 1

        images = np.asarray(images, dtype=np.float32)
        labels = self.labels.loc[good_ids].to_numpy().astype(np.float32) / 100.0
        labels = labels.reshape(-1, 1)
        return images, labels


path = "../input/petfinder-pawpularity-score/train/"
labels = dat.set_index("Id")["Pawpularity"]
c = data(path, labels=labels)



## === cell 5
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 6
model = Sequential()
model.add(Input(shape=(128, 128, 3)))
model.add(Conv2D(10, 5))
model.add(MaxPooling2D())
model.add(Conv2D(20, 3))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="relu"))



## === cell 7
model.summary()



## === cell 8
model.compile(optimizer=Adam(1e-3), loss="mae")



## === cell 9
ids = dat["Id"].to_list()
train_ids = ids[: int(len(ids) * 0.8)]
val_ids = ids[int(len(ids) * 0.8) : int(len(ids) * 0.9)]
test_ids = ids[int(len(ids) * 0.9) :]




## === cell 10
class data:
    def __init__(self, path, ids, x=128, y=128, labels=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.image_list = ids
        self.path = path
        self.batch = 0

    def load_batch(self, batch_size=1, shuffle=False):
        if shuffle:
            b = self.batch
            batch_list = self.image_list[b * batch_size : (b + 1) * batch_size]
            self.batch = b + 1
            if self.batch > len(self.image_list) // batch_size:
                self.batch = 0
        else:
            batch_list = sample(self.image_list, batch_size)

        images = []
        good_ids = []
        for image_id in batch_list:
            img_path = os.path.join(self.path, image_id + ".jpg")
            img = read_image_rgb_128(img_path)
            if img is None:
                continue
            images.append(img)
            good_ids.append(image_id)

        if len(images) == 0:
            tries = 0
            while len(images) == 0 and tries < 10:
                image_id = sample(self.image_list, 1)[0]
                img_path = os.path.join(self.path, image_id + ".jpg")
                img = read_image_rgb_128(img_path)
                if img is not None:
                    images.append(img)
                    good_ids.append(image_id)
                tries += 1

        images = np.asarray(images, dtype=np.float32)
        labels = self.labels.loc[good_ids].to_numpy().astype(np.float32) / 100.0
        labels = labels.reshape(-1, 1)
        return images, labels

    def loader(self, batch_size=1, shuffle=False):
        while True:
            x, y = self.load_batch(batch_size, shuffle)
            yield x, y


path = "../input/petfinder-pawpularity-score/train/"
labels = dat.set_index("Id")["Pawpularity"]
c = data(path, labels=labels, ids=ids)
c_train = data(path, labels=labels, ids=train_ids)
c_val = data(path, labels=labels, ids=val_ids)
c_test = data(path, labels=labels, ids=test_ids)



## === cell 11
model.fit(c.loader(batch_size=128), steps_per_epoch=100, epochs=2, verbose=1)
model.fit(c.loader(batch_size=32), steps_per_epoch=100, epochs=1, verbose=1)



## === cell 12
testpath = "../input/petfinder-pawpularity-score/test/"
test_df = pd.read_csv("../input/petfinder-pawpularity-score/test.csv")
ids = test_df["Id"].tolist()



## === cell 13
batch_size = 200
y_pred = np.zeros((len(ids), 1), dtype=np.float32)

for start in range(0, len(ids), batch_size):
    end = min(start + batch_size, len(ids))
    batch_ids = ids[start:end]

    images = []
    valid_positions = []
    for j, image_id in enumerate(batch_ids):
        img_path = os.path.join(testpath, image_id + ".jpg")
        img = read_image_rgb_128(img_path)
        if img is None:
            continue
        images.append(img)
        valid_positions.append(j)

    if len(images) == 0:
        continue

    images = np.asarray(images, dtype=np.float32)
    preds = model.predict(images, verbose=0).astype(np.float32) * 100.0
    for k, j in enumerate(valid_positions):
        y_pred[start + j, 0] = preds[k, 0]



## === cell 14
y_pred = np.clip(y_pred, 1.0, 100.0)

round2 = lambda x, y=None: round(float(x) + 1e-15, y)
y_pred_list = [round2(t[0], 2) for t in y_pred]



## === cell 15
df = pd.DataFrame()
df["Id"] = ids
df["Pawpularity"] = y_pred_list



## === cell 16
df.to_csv("submission.csv", index=False)



## === cell 17
df.head()
