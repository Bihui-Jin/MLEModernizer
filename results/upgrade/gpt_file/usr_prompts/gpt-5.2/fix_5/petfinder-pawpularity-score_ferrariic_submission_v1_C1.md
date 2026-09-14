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

No external packages required in the script and installed.

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

29.06754310772829

# 6. Current score

35.53384

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 35.53384) has done: 'I fix the immediate runtime crash caused by importing unnecessary heavy libraries (notably `dask`) that trigger the protobuf `MessageFactory.GetPrototype` error in this environment. Then I remove the dependency on a non-existent external dataset (`../input/paw-model/...`) by training the same CNN architecture locally on the provided `train/` images and `train.csv`, so `pre_trained_model` is always defined. Finally, I keep your existing image preprocessing and the `model_adjusted()` post-calibration logic, but compute `y_pred` from the model’s training predictions (instead of loading a missing pickle) so the output scaling is consistent and a valid `submission.csv` is always produced.'
- What this solution (achieved 35.53384) has done: 'I fix the TensorFlow/protobuf crash by removing the forced `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION=python` setting (it triggers the `MessageFactory.GetPrototype` issue in this environment) and keeping TF logging suppression only. Then I fix the submission validity error by ensuring final predictions are clipped to the required `[1, 100]` range (the competition disallows 0), and by calibrating with the *training* prediction distribution (the prior code mistakenly re-fit min/max using test predictions inside `model_adjusted`, which is inconsistent and can push values to 0). These changes keep your CNN, preprocessing, training loop, and calibration logic intact while making the notebook run end-to-end and produce a valid `submission.csv`.'
- What this solution (achieved 35.53384) has done: 'I fix the TensorFlow/protobuf crash happening at import time by forcing TensorFlow to use the pure-Python protobuf backend before importing `tensorflow` (this avoids the `MessageFactory.GetPrototype` issue in this Kaggle environment). Then I keep your exact CNN, preprocessing, training loop, and calibration logic, but add deterministic settings and make sure the submission columns and value range are always valid. Finally, I ensure paths resolve correctly and the script always writes `submission.csv` with the required header and `[1, 100]` clipping.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import random as rand

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense, Input, Dropout, Conv2D, MaxPooling2D, Flatten

from sklearn.model_selection import train_test_split

SEED = 42
rand.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
INPUT_ROOT = "/kaggle/input/petfinder-pawpularity-score"
if not os.path.exists(INPUT_ROOT):
    INPUT_ROOT = "../input/petfinder-pawpularity-score"

TRAIN_IMG_DIR = os.path.join(INPUT_ROOT, "train")
TEST_IMG_DIR = os.path.join(INPUT_ROOT, "test")
TRAIN_CSV = os.path.join(INPUT_ROOT, "train.csv")
TEST_CSV = os.path.join(INPUT_ROOT, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_ROOT, "sample_submission.csv")

assert os.path.exists(TRAIN_CSV), f"Missing train.csv at {TRAIN_CSV}"
assert os.path.exists(TEST_CSV), f"Missing test.csv at {TEST_CSV}"
assert os.path.isdir(TRAIN_IMG_DIR), f"Missing train/ at {TRAIN_IMG_DIR}"
assert os.path.isdir(TEST_IMG_DIR), f"Missing test/ at {TEST_IMG_DIR}"
assert os.path.exists(SAMPLE_SUB), f"Missing sample_submission.csv at {SAMPLE_SUB}"



## === cell 2
train_path = os.path.join(TRAIN_IMG_DIR, "*.jpg")
test_path = os.path.join(TEST_IMG_DIR, "*.jpg")

train_images = glob.glob(train_path)
test_images = glob.glob(test_path)

df_train = pd.read_csv(TRAIN_CSV)
df_test = pd.read_csv(TEST_CSV)

df_test = df_test.copy()
df_test["img_path"] = df_test["Id"].apply(
    lambda x: os.path.join(TEST_IMG_DIR, f"{x}.jpg")
)

df_train.head()



## === cell 3
sq_shape = 128


def process_image(image):
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)  # reads image as greyscale
    if im is None:
        raise FileNotFoundError(f"Could not read image: {image}")
    im = cv2.resize(im, (sq_shape, sq_shape))  # resizes for processing speed
    im = im / 255.0  # normalizes
    return im




## === cell 4
df_train = df_train.copy()
df_train["img_path"] = df_train["Id"].apply(
    lambda x: os.path.join(TRAIN_IMG_DIR, f"{x}.jpg")
)

missing = df_train.loc[~df_train["img_path"].apply(os.path.exists)]
if len(missing) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing)} training images. Example: {missing.iloc[0]['img_path']}"
    )

X = np.zeros((len(df_train), sq_shape, sq_shape, 1), dtype=np.float32)
y = df_train["Pawpularity"].astype(np.float32).values

for i, p in enumerate(df_train["img_path"].values):
    im = process_image(p)
    X[i, :, :, 0] = im

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.15, random_state=SEED
)



## === cell 5
pre_trained_model = Sequential(
    [
        Input(shape=(sq_shape, sq_shape, 1)),
        Conv2D(32, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation="relu", padding="same"),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dropout(0.25),
        Dense(128, activation="relu"),
        Dropout(0.25),
        Dense(1, activation="linear"),
    ]
)

pre_trained_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss="mse",
    metrics=[tf.keras.metrics.RootMeanSquaredError(name="rmse")],
)

pre_trained_model.summary()



## === cell 6
history = pre_trained_model.fit(
    X_train, y_train, validation_data=(X_val, y_val), epochs=5, batch_size=64, verbose=2
)



## === cell 7
y_pred_train_ref = pre_trained_model.predict(
    X_train, batch_size=128, verbose=0
).reshape(-1)




## === cell 8
def model_adjusted(prediction, y_pred):
    ref = np.asarray(y_pred).reshape(-1)
    min_val = float(ref.min())
    denom = float(ref.max() - ref.min())
    if denom <= 1e-12:
        return np.array([50.0], dtype=np.float32)
    output = ((float(prediction) - min_val) / denom) * 100.0
    return np.array([output], dtype=np.float32)




## === cell 9
X_test_images = np.zeros((len(df_test), sq_shape, sq_shape, 1), dtype=np.float32)
Name = df_test["Id"].tolist()

missing_test = df_test.loc[~df_test["img_path"].apply(os.path.exists)]
if len(missing_test) > 0:
    raise FileNotFoundError(
        f"Missing {len(missing_test)} test images. Example: {missing_test.iloc[0]['img_path']}"
    )

for i, p in enumerate(df_test["img_path"].values):
    im = process_image(p)
    X_test_images[i, :, :, 0] = im

predictions = pre_trained_model.predict(
    X_test_images, batch_size=128, verbose=0
).reshape(-1)

outputs = []
for pred in predictions:
    output = model_adjusted(pred, y_pred_train_ref)[0]
    outputs.append(float(output))

outputs = np.clip(np.asarray(outputs, dtype=np.float32), 1.0, 100.0)

df_final = pd.DataFrame({"Id": Name, "Pawpularity": outputs})
df_final.to_csv("./submission.csv", index=False)

df_final.head()
