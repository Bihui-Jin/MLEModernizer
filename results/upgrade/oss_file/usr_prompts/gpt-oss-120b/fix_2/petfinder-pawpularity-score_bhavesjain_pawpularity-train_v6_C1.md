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

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

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



## === cell 1
dat = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")



## === cell 2
y = dat["Pawpularity"].to_numpy()
y = y / 100.0




## === cell 3
class data:
    def __init__(self, path, ids, labels=None, x=128, y=128):
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
        for image_id in batch_list:
            img_path = os.path.join(self.path, image_id + ".jpg")
            img = cv2.imread(img_path)
            if img is None:
                img = np.zeros((self.x, self.y, 3), dtype=np.uint8)
            img_resized = cv2.resize(img, (self.x, self.y))
            img_rgb = cv2.cvtColor(img_resized, cv2.COLOR_BGR2RGB) / 255.0
            images.append(img_rgb)
        images = np.array(images)

        labels = self.labels.loc[batch_list].to_numpy() / 100.0
        return images, labels

    def loader(self, batch_size=1, shuffle=False):
        while True:
            x_batch, y_batch = self.load_batch(batch_size, shuffle)
            yield x_batch, y_batch




## === cell 4
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
model = Sequential()
model.add(Input(shape=(128, 128, 3)))
model.add(Conv2D(10, 5, activation="relu"))
model.add(MaxPooling2D())
model.add(Conv2D(20, 3, activation="relu"))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(10, activation="relu"))
model.add(Dense(1, activation="relu"))



## === cell 6
model.summary()



## === cell 7
model.compile(optimizer=Adam(1e-3), loss="mae")



## === cell 8
ids = dat["Id"].to_list()
train_ids = ids[: int(len(ids) * 0.8)]
val_ids = ids[int(len(ids) * 0.8) : int(len(ids) * 0.9)]
test_ids = ids[int(len(ids) * 0.9) :]



## === cell 9
labels = dat.set_index("Id")["Pawpularity"]
train_path = "../input/petfinder-pawpularity-score/train/"

c_train = data(train_path, ids=train_ids, labels=labels)
c_val = data(train_path, ids=val_ids, labels=labels)



## === cell 10
model.fit(
    c_train.loader(batch_size=64, shuffle=True),
    steps_per_epoch=len(train_ids) // 64,
    validation_data=c_val.loader(batch_size=64),
    validation_steps=len(val_ids) // 64,
    epochs=8,
    verbose=2,
)



## === cell 11
test_path = "../input/petfinder-pawpularity-score/test/"
test_ids = [
    t.split(".")[0] for t in os.listdir(test_path) if t.lower().endswith(".jpg")
]



## === cell 12
batch_size = 200
y_pred = np.zeros((len(test_ids), 1), dtype=np.float32)

num_batches = (len(test_ids) + batch_size - 1) // batch_size
for i in range(num_batches):
    batch_ids = test_ids[i * batch_size : (i + 1) * batch_size]
    images = []
    for image_id in batch_ids:
        img_path = os.path.join(test_path, image_id + ".jpg")
        img = cv2.imread(img_path)
        if img is None:
            img = np.zeros((128, 128, 3), dtype=np.uint8)
        img = cv2.cvtColor(cv2.resize(img, (128, 128)), cv2.COLOR_BGR2RGB) / 255.0
        images.append(img)
    images = np.array(images)
    preds = model.predict(images, verbose=0) * 100.0
    preds = np.clip(preds, 1.0, 100.0)
    y_pred[i * batch_size : i * batch_size + len(batch_ids), 0] = preds[:, 0]



## === cell 13
round2 = lambda x, y=None: round(x + 1e-15, y) if y is not None else round(x + 1e-15)
y_pred = [round2(val, 2) for val in y_pred]



## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3081381204.py in <cell line: 0>()
      1 round2 = lambda x, y=None: round(x + 1e-15, y) if y is not None else round(x + 1e-15)
----> 2 y_pred = [round2(val, 2) for val in y_pred]
      3 

/tmp/ipykernel_55/3081381204.py in <listcomp>(.0)
      1 round2 = lambda x, y=None: round(x + 1e-15, y) if y is not None else round(x + 1e-15)
----> 2 y_pred = [round2(val, 2) for val in y_pred]
      3 

/tmp/ipykernel_55/3081381204.py in <lambda>(x, y)
----> 1 round2 = lambda x, y=None: round(x + 1e-15, y) if y is not None else round(x + 1e-15)
      2 y_pred = [round2(val, 2) for val in y_pred]
      3 

TypeError: type numpy.ndarray doesn't define __round__ method

## === cell 14
df = pd.DataFrame({"Id": test_ids, "Pawpularity": y_pred})



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/162435652.py in <cell line: 0>()
----> 1 df = pd.DataFrame({"Id": test_ids, "Pawpularity": y_pred})
      2 

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __init__(self, data, index, columns, dtype, copy)
    776         elif isinstance(data, dict):
    777             # GH#38939 de facto copy defaults to False only in non-dict cases
--> 778             mgr = dict_to_mgr(data, index, columns, dtype=dtype, copy=copy, typ=manager)
    779         elif isinstance(data, ma.MaskedArray):
    780             from numpy.ma import mrecords

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in dict_to_mgr(data, index, columns, dtype, typ, copy)
    501             arrays = [x.copy() if hasattr(x, "dtype") else x for x in arrays]
    502 
--> 503     return arrays_to_mgr(arrays, columns, index, dtype=dtype, typ=typ, consolidate=copy)
    504 
    505 

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in arrays_to_mgr(arrays, columns, index, dtype, verify_integrity, typ, consolidate)
    112         # figure out the index, if necessary
    113         if index is None:
--> 114             index = _extract_index(arrays)
    115         else:
    116             index = ensure_index(index)

/usr/local/lib/python3.11/dist-packages/pandas/core/internals/construction.py in _extract_index(data)
    662             raw_lengths.append(len(val))
    663         elif isinstance(val, np.ndarray) and val.ndim > 1:
--> 664             raise ValueError("Per-column arrays must each be 1-dimensional")
    665 
    666     if not indexes and not raw_lengths:

ValueError: Per-column arrays must each be 1-dimensional

## === cell 15
df.to_csv("submission.csv", index=False)



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2377727887.py in <cell line: 0>()
----> 1 df.to_csv("submission.csv", index=False)
      2 

NameError: name 'df' is not defined

## === cell 16
df.head()

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/964094849.py in <cell line: 0>()
----> 1 df.head()

NameError: name 'df' is not defined
