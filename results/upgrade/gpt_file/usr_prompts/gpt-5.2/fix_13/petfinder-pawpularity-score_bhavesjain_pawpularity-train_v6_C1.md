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

21.18934

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 22.95529) has done: 'I fix TensorFlow import/runtime failures by pinning the protobuf Python implementation before importing TF (this resolves the `MessageFactory` error in this environment). I update deprecated `fit_generator` calls to `fit` while keeping the same generator-based training loop and epochs/steps so the core training behavior remains the same. I make image loading robust (correct path joining + handle unreadable files) to stop OpenCV resize crashes, and I ensure the test Id ordering matches `test.csv`. Finally, I clip predictions to the required [1, 100] range so the submission passes validation and is written as a proper `submission.csv`.'
- What this solution (achieved 22.95702) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime **and** disabling the C++ implementation before importing TensorFlow, which resolves the `MessageFactory.GetPrototype` error in TF 2.18 on Kaggle. I also set TensorFlow random seeds for deterministic behavior (score-neutral) and keep your model architecture, generator-based training loop, steps, and epochs unchanged to preserve core logic. Finally, I make submission generation safer by ensuring every test row gets a numeric prediction (fallback to train mean for any unreadable images) and still clip to `[1, 100]`, producing a valid `submission.csv`.'
- What this solution (achieved 21.78626) has done: 'I fix the TensorFlow/protobuf crash causing `MessageFactory.GetPrototype` by enforcing the pure-Python protobuf runtime *and* disabling the C++ implementation before any TensorFlow import, which is required in this Kaggle environment with TF 2.18. I also correct one logic bug in your data loader: the `shuffle` flag is currently inverted (it takes sequential slices when `shuffle=True`), so I make it actually shuffle batches; this is a small, legitimate training fix likely to move score toward your target (worse/larger RMSE) without changing the model architecture or training schedule. Finally, I keep submission generation identical but add a safety check that the output has the required columns and row count, ensuring a valid `submission.csv` is always written.'
- What this solution (achieved 22.95575) has done: 'I fix the TensorFlow/protobuf runtime crash by enforcing the pure-Python protobuf implementation *before* any TensorFlow import and by importing `google.protobuf` once to lock the runtime, which prevents the `MessageFactory.GetPrototype` error in this environment. I also keep your model, training schedule, and data pipeline semantics the same, but remove the duplicated `data` class definition that can shadow earlier fixes and cause inconsistent behavior. Finally, I add a small safety fallback for empty batches during training to prevent sporadic shape errors, and keep the submission generation identical while ensuring the output file is written as `submission.csv` with correct columns and row count.'
- What this solution (achieved 21.51227) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by forcing the pure-Python protobuf runtime early and pinning protobuf to the compatible major version *in-process* before importing TensorFlow (this is the minimal change that unblocks execution in this TF 2.18 Kaggle environment). I keep your model, data loader, training steps/epochs, and prediction logic unchanged to avoid moving the score further away from the target (your current RMSE is already much better than the target and we don’t want to improve it further). I also add a tiny safety check to ensure the submission is written with the right filename/columns/row count even if something unexpected happens during prediction.'
- What this solution (achieved 20.6672) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *and* importing protobuf’s message module before TensorFlow, which is the reliable workaround for the `MessageFactory.GetPrototype` error in TF 2.18 on Kaggle. I keep your model, training schedule, data pipeline, and prediction logic the same to avoid moving the score further away from your target (your current RMSE is already better than the target, and lower is better). I also add a small defensive check to ensure the submission is always written as a valid `submission.csv` with the correct columns and row count.'
- What this solution (achieved 20.34029) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no `GetPrototype`) by ensuring the pure-Python protobuf runtime is forced early and consistently, and by importing TensorFlow only after that lock-in. I also keep your model and training loop intact, but add a tiny safety fallback in the batch loader when `labels` is `None` (so the same class can be safely reused for test/inference if needed). Finally, I ensure the submission is always written as `submission.csv` with the required columns and the same test-row ordering, without changing prediction logic (so your score should stay close to the current level rather than improving further away from the target).'
- What this solution (achieved 20.3401) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf runtime *before* any TensorFlow import and by ensuring no other imports trigger protobuf C++ bindings; this is the root cause of the `MessageFactory.GetPrototype` error in TF 2.18 here. I keep your model, training loop, steps/epochs, and preprocessing exactly the same so evaluation semantics don’t change. I also keep the submission generation identical but add a small safety assertion that the output file is actually written with the `.csv` suffix. This should run end-to-end and produce a valid `submission.csv` while keeping the score behavior essentially unchanged (your current score is already better than the target).'
- What this solution (achieved 21.66163) has done: 'I fix the TensorFlow import crash by applying the protobuf runtime workaround earlier and more reliably, before any TensorFlow-related modules are loaded, which resolves the `MessageFactory.GetPrototype` error in this TF 2.18 environment. I keep your model, preprocessing, training loop, and prediction logic unchanged so the score behavior stays essentially the same (your current RMSE is already better than the target, and lower is better). I also keep the submission writing identical but add a small path auto-detection guard so the code works whether the dataset is mounted under `../input/...` or `/kaggle/input/...`, without changing any modeling semantics.'
- What this solution (achieved 22.95675) has done: 'I fix the TensorFlow import crash (`MessageFactory` has no attribute `GetPrototype`) by ensuring the protobuf runtime is forced to the pure-Python implementation *and* that TensorFlow is imported only after that lock-in, using an additional environment variable that TF respects in this Kaggle setup. This is a correctness/stability fix only and not change your model, training schedule, preprocessing, or prediction logic (so it should keep your score in the same ballpark, avoiding further improvement away from the target). I also keep your existing dataset path auto-detection and submission checks intact so the notebook reliably produces `submission.csv` with the required columns/row count.'
- What this solution (achieved 22.95541) has done: 'I fix the TensorFlow import crash (`MessageFactory` missing `GetPrototype`) by forcing the pure-Python protobuf runtime *and* ensuring TensorFlow uses the Python protobuf implementation before any TF import happens. This is a stability/runtime fix only and does not change your model, preprocessing, training schedule, or submission logic, so it should keep the score behavior essentially the same (already better than target; lower is better). I also keep the existing dataset base-path auto-detection and make sure the pipeline still writes a valid `submission.csv` with the required columns and correct row count.'
- What this solution (achieved 21.18934) has done: 'I fix the TensorFlow/protobuf crash by enforcing the pure-Python protobuf runtime even earlier and preventing any accidental TensorFlow import before that lock-in (this is the root cause of the `MessageFactory.GetPrototype` error). I keep your model, preprocessing, training loop, and submission logic unchanged to avoid moving the score further away from your target (your current RMSE is already better than the target and lower is better). I also add a small guard to ensure the dataset base path resolves correctly and that the train/test image directories exist before training/inference, so the pipeline runs end-to-end reliably. The script still write a valid `submission.csv` with the required columns and row count.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_DISABLE_CXX"] = "1"
os.environ["TF_PROTOBUF_VERSION"] = "python"

import google.protobuf  # noqa: F401
from google.protobuf import message as _pb_message  # noqa: F401

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from random import sample
import cv2
import random

random.seed(42)
np.random.seed(42)

CANDIDATES = [
    "../input/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score",
    "/kaggle/data/petfinder-pawpularity-score",
    "/kaggle/data/petfinder-pawpularity-score/petfinder-pawpularity-score",
    "/kaggle/input/petfinder-pawpularity-score/petfinder-pawpularity-score",
]
BASE = None
for c in CANDIDATES:
    if os.path.exists(os.path.join(c, "train.csv")):
        BASE = c
        break
if BASE is None:
    raise FileNotFoundError(
        "Could not locate train.csv under expected Kaggle dataset paths."
    )

TRAIN_IMG_DIR = os.path.join(BASE, "train")
TEST_IMG_DIR = os.path.join(BASE, "test")
if not os.path.isdir(TRAIN_IMG_DIR):
    raise FileNotFoundError(f"Train image dir not found: {TRAIN_IMG_DIR}")
if not os.path.isdir(TEST_IMG_DIR):
    raise FileNotFoundError(f"Test image dir not found: {TEST_IMG_DIR}")



## === cell 1
dat = pd.read_csv(os.path.join(BASE, "train.csv"))



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
    def __init__(self, path, ids, x=128, y=128, labels=None):
        self.x = x
        self.y = y
        self.labels = labels
        self.image_list = list(ids)
        self.path = path
        self.batch = 0

    def load_batch(self, batch_size=1, shuffle=False):
        if shuffle:
            batch_list = sample(self.image_list, batch_size)
        else:
            b = self.batch
            batch_list = self.image_list[b * batch_size : (b + 1) * batch_size]
            self.batch = b + 1
            if self.batch > len(self.image_list) // batch_size:
                self.batch = 0

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

        if images.shape[0] == 0:
            images = np.zeros((1, 128, 128, 3), dtype=np.float32)
            if self.labels is None:
                labels = np.zeros((1, 1), dtype=np.float32)
            else:
                mean_label = float(self.labels.mean()) / 100.0
                labels = np.array([[mean_label]], dtype=np.float32)
            return images, labels

        if self.labels is None:
            labels = np.zeros((images.shape[0], 1), dtype=np.float32)
            return images, labels

        labels = self.labels.loc[good_ids].to_numpy().astype(np.float32) / 100.0
        labels = labels.reshape(-1, 1)
        return images, labels

    def loader(self, batch_size=1, shuffle=False):
        while True:
            x, y = self.load_batch(batch_size, shuffle)
            yield x, y


path = os.path.join(BASE, "train")
labels = dat.set_index("Id")["Pawpularity"]
ids = dat["Id"].to_list()

c = data(path, labels=labels, ids=ids)



## === cell 5
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense
from tensorflow.keras.optimizers import Adam

tf.keras.utils.set_random_seed(42)



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

path = os.path.join(BASE, "train")
labels = dat.set_index("Id")["Pawpularity"]
c = data(path, labels=labels, ids=ids)
c_train = data(path, labels=labels, ids=train_ids)
c_val = data(path, labels=labels, ids=val_ids)
c_test = data(path, labels=labels, ids=test_ids)



## === cell 10
model.fit(
    c.loader(batch_size=128, shuffle=True), steps_per_epoch=100, epochs=2, verbose=1
)
model.fit(
    c.loader(batch_size=32, shuffle=True), steps_per_epoch=100, epochs=1, verbose=1
)



## === cell 11
testpath = os.path.join(BASE, "test")
test_df = pd.read_csv(os.path.join(BASE, "test.csv"))
ids = test_df["Id"].tolist()



## === cell 12
batch_size = 200
y_pred = np.full((len(ids), 1), np.nan, dtype=np.float32)

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



## === cell 13
fallback = float(dat["Pawpularity"].mean())
y_pred = np.where(np.isfinite(y_pred), y_pred, fallback).astype(np.float32)

y_pred = np.clip(y_pred, 1.0, 100.0)

round2 = lambda x, y=None: round(float(x) + 1e-15, y)
y_pred_list = [round2(t[0], 2) for t in y_pred]



## === cell 14
df = pd.DataFrame()
df["Id"] = ids
df["Pawpularity"] = y_pred_list

assert list(df.columns) == ["Id", "Pawpularity"]
assert len(df) == len(test_df)
assert df["Id"].isna().sum() == 0
assert df["Pawpularity"].isna().sum() == 0



## === cell 15
df.to_csv("submission.csv", index=False)
assert os.path.exists("submission.csv") and os.path.getsize("submission.csv") > 0



## === cell 16
df.head()
