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
scikit-image==0.25.2
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

20.94962

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import pandas as pd

DATA_DIR = "../input/petfinder-pawpularity-score"
train_data_csv = pd.read_csv(os.path.join(DATA_DIR, "train.csv"))
test_data_csv = pd.read_csv(os.path.join(DATA_DIR, "test.csv"))

print(train_data_csv.shape)
print(test_data_csv.shape)

train_img_path = os.path.join(DATA_DIR, "train")
test_img_path = os.path.join(DATA_DIR, "test")

assert os.path.isdir(
    train_img_path
), f"Train image directory not found: {train_img_path}"
assert os.path.isdir(test_img_path), f"Test image directory not found: {test_img_path}"
print("Train images dir:", train_img_path)
print("Test images dir:", test_img_path)



## === cell 1
from tqdm import trange
import numpy as np
import cv2

try:
    from skimage.feature import graycomatrix as greycomatrix, graycoprops as greycoprops
except Exception:
    from skimage.feature import greycomatrix, greycoprops

properties = ["contrast", "dissimilarity", "energy", "homogeneity", "correlation"]


def calc_glcm_all_agls(
    img,
    props,
    dists=[5],
    agls=[0, np.pi / 4, np.pi / 2, 3 * np.pi / 4],
    lvl=256,
    sym=True,
    norm=True,
):
    glcm = greycomatrix(
        img,
        distances=dists,
        angles=agls,
        levels=lvl,
        symmetric=sym,
        normed=norm,
    )
    feature = []
    glcm_props = [propery for name in props for propery in greycoprops(glcm, name)[0]]
    for item in glcm_props:
        feature.append(item)
    return feature


train_features = []
train_pawpul = train_data_csv["Pawpularity"].astype(np.float32).to_numpy()
train_data_array = np.array(train_data_csv)

for i in trange(train_data_csv.shape[0]):
    path = os.path.join(train_img_path, train_data_csv["Id"].iloc[i] + ".jpg")
    img_org = cv2.imread(path)
    if img_org is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)

    feats = calc_glcm_all_agls(img_resize, props=properties)
    for j in range(12):
        feats.append(np.float32(train_data_array[i][j + 1]))
    train_features.append(feats)

print(len(train_features))
print(len(train_pawpul))
print(train_features[0])
print(train_pawpul[0])



## === cell 2
test_features = []
test_data_array = np.array(test_data_csv)

for i in trange(test_data_csv.shape[0]):
    path = os.path.join(test_img_path, test_data_csv["Id"].iloc[i] + ".jpg")
    img_org = cv2.imread(path)
    if img_org is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_gray = cv2.cvtColor(img_org, cv2.COLOR_BGR2GRAY)
    img_resize = cv2.resize(img_gray, (256, 256), interpolation=cv2.INTER_AREA)

    feats = calc_glcm_all_agls(img_resize, props=properties)
    for j in range(12):
        feats.append(np.float32(test_data_array[i][j + 1]))
    test_features.append(feats)

print(len(test_features))
print(test_features[0])



## === cell 3
total_cnt = len(train_features)
print(total_cnt)
split_rate = 0.8

train_feat = np.array(train_features[: int(total_cnt * split_rate)], dtype=np.float32)
val_feat = np.array(train_features[int(total_cnt * split_rate) :], dtype=np.float32)
train_pawp = np.array(train_pawpul[: int(total_cnt * split_rate)], dtype=np.float32)
val_pawp = np.array(train_pawpul[int(total_cnt * split_rate) :], dtype=np.float32)

print(train_feat.shape, val_feat.shape)
print(train_pawp.shape, val_pawp.shape)



## === cell 4
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

import tensorflow as tf
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import losses, metrics, schedules

print(tf.__version__)

inp = Input(shape=(train_feat.shape[1],))
model_feat = Dense(32, activation="relu")(inp)
model_feat = Dense(16, activation="relu")(model_feat)
out = Dense(1)(model_feat)
model = Model(inputs=inp, outputs=out)

lr_schedule = schedules.ExponentialDecay(
    initial_learning_rate=1e-3,
    decay_steps=100,
    decay_rate=0.96,
    staircase=True,
)

model.compile(
    optimizer=Adam(learning_rate=lr_schedule),
    loss=losses.MeanSquaredError(),
    metrics=[metrics.RootMeanSquaredError()],
)

model.summary()



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
from tensorflow.keras.utils import Sequence
from sklearn.utils import shuffle


class DataGenerator(Sequence):
    def __init__(self, feat_data, pawp_data, batch_size=64, shuffle_data=True):
        self.feat_data = np.asarray(feat_data, dtype=np.float32)
        self.pawp_data = np.asarray(pawp_data, dtype=np.float32).reshape(-1)
        self.batch_size = batch_size
        self.shuffle_data = shuffle_data
        self.on_epoch_end()

    def __len__(self):
        return int(np.ceil(self.pawp_data.shape[0] / self.batch_size))

    def __getitem__(self, index):
        start = index * self.batch_size
        end = (index + 1) * self.batch_size
        feat_batch = self.feat_data[start:end]
        pawp_batch = self.pawp_data[start:end]
        return feat_batch, pawp_batch

    def on_epoch_end(self):
        if self.shuffle_data:
            self.feat_data, self.pawp_data = shuffle(
                self.feat_data, self.pawp_data, random_state=42
            )




## === cell 6
from tensorflow.keras.callbacks import EarlyStopping

early_stop = EarlyStopping(monitor="val_loss", patience=5, restore_best_weights=True)



## === cell 7
train_gen = DataGenerator(train_feat, train_pawp, shuffle_data=False)
val_gen = DataGenerator(val_feat, val_pawp, shuffle_data=False)

history = model.fit(
    train_gen,
    epochs=100,
    validation_data=val_gen,
    callbacks=[early_stop],
    verbose=1,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3996945673.py in <cell line: 0>()
      2 val_gen = DataGenerator(val_feat, val_pawp, shuffle_data=False)
      3 
----> 4 history = model.fit(
      5     train_gen,
      6     epochs=100,

NameError: name 'model' is not defined

## === cell 8
import matplotlib.pyplot as plt

rmse = history.history.get("root_mean_squared_error", [])
val_rmse = history.history.get("val_root_mean_squared_error", [])
loss = history.history.get("loss", [])
val_loss = history.history.get("val_loss", [])

epochs = range(1, len(loss) + 1)

if len(rmse) and len(val_rmse):
    plt.plot(epochs, rmse, "bo", label="Training rmse")
    plt.plot(epochs, val_rmse, "b", label="Validation rmse")
    plt.title("Training and validation rmse")
    plt.legend()
    plt.show()

plt.figure()
plt.plot(epochs, loss, "bo", label="Training loss")
plt.plot(epochs, val_loss, "b", label="Validation loss")
plt.title("Training and validation loss")
plt.legend()
plt.show()



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/792209422.py in <cell line: 0>()
      1 import matplotlib.pyplot as plt
      2 
----> 3 rmse = history.history.get("root_mean_squared_error", [])
      4 val_rmse = history.history.get("val_root_mean_squared_error", [])
      5 loss = history.history.get("loss", [])

NameError: name 'history' is not defined

## === cell 9
Id = np.array(test_data_csv["Id"])

test_feat_arr = np.array(test_features, dtype=np.float32)
predictions = model.predict(test_feat_arr, verbose=1)

predictions = predictions.reshape(-1).astype(np.float32)
predictions = np.clip(predictions, 0.0, 100.0)

submission_df = pd.DataFrame({"Id": Id, "Pawpularity": predictions})
submission_df.to_csv("submission.csv", index=False)

print(submission_df.head(10))
print("Wrote submission.csv with shape:", submission_df.shape)
print("Finished")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2724353132.py in <cell line: 0>()
      2 
      3 test_feat_arr = np.array(test_features, dtype=np.float32)
----> 4 predictions = model.predict(test_feat_arr, verbose=1)
      5 
      6 predictions = predictions.reshape(-1).astype(np.float32)

NameError: name 'model' is not defined
