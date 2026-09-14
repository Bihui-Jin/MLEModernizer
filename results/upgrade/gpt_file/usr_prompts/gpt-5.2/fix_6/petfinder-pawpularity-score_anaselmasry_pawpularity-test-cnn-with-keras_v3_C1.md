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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
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

43.0839

# 6. Current score

23.0355

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 23.00919) has done: 'I fix the TensorFlow import crash by pinning protobuf to the Kaggle-compatible pure-Python implementation via an environment flag set before importing TensorFlow. Then I fix the label/target bug: the model must predict only the `Pawpularity` regression target (scaled 0–1), not all 13 metadata columns, and the output layer must match that shape. Finally, I fix submission post-processing by converting predictions back to the required 1–100 range and clipping, ensuring the output CSV is valid and produced end-to-end.'
- What this solution (achieved 23.18739) has done: 'I fix the TensorFlow/protobuf crash by forcing the pure-Python protobuf implementation *and* ensuring TensorFlow sees it early enough (including setting the version env var), which removes the `MessageFactory.GetPrototype` import error in this environment. I also correct a small but important `load_img` argument bug (`target_size` must be 2-tuple, not 3D) that can break image loading depending on Keras version. Since your current score (23.00919 RMSE) is already much better than the target (43.0839) for a lower-is-better metric, I avoid any score-changing “degradation” and keep the modeling/training logic intact—only stability/runtime fixes. The script still train end-to-end and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 23.74467) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *and* disabling the C++ protobuf backend before TensorFlow is imported, which directly addresses the `MessageFactory.GetPrototype` error in this environment. I also make the input data path resolution robust by selecting the first existing competition directory among the common Kaggle locations, without changing any modeling/training logic. Finally, I keep your model architecture, loss, training loop, and prediction post-processing identical, ensuring the notebook runs end-to-end and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 22.63587) has done: 'The crash happens before any modeling because TensorFlow 2.18 is importing a C++ protobuf runtime that’s incompatible in this environment, so we force TensorFlow to use the pure-Python protobuf implementation and ensure that no stale protobuf/tensorflow modules are already loaded. I also harden the TF import by clearing any pre-imported `google.protobuf` modules (common in notebook-style runs) before importing TensorFlow. These changes are runtime-only and do not alter your model, training loop, or prediction post-processing, so the score should remain in the same ballpark (and since your current score is already better than the target for a lower-is-better metric, we avoid any score-changing edits). The rest of the pipeline remains identical and still writes a valid `submission.csv`.'
- What this solution (achieved 23.0355) has done: 'The crash is happening at TensorFlow import due to an incompatibility between TF 2.18 and the protobuf runtime in this environment (`MessageFactory.GetPrototype` missing). I fix this by force-reinstalling a protobuf version that provides that API (3.20.x) and then importing TensorFlow cleanly; this is a runtime-only change and should not affect the model/training logic or score. I also keep your existing environment flags and module cleanup, but ensure they’re applied before the TF import. Everything else (data loading, model architecture, training loop, prediction scaling/clipping, and submission writing) is preserved so you still get a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import subprocess
import importlib

subprocess.run(
    [sys.executable, "-m", "pip", "install", "-q", "--no-deps", "protobuf==3.20.3"],
    check=True,
)

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"
os.environ["PROTOCOL_BUFFERS_PYTHON_DISABLE_CPP_IMPLEMENTATION"] = "1"

for m in list(sys.modules.keys()):
    if m.startswith("google.protobuf"):
        del sys.modules[m]
if "tensorflow" in sys.modules:
    del sys.modules["tensorflow"]

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing import image

np.random.seed(42)
tf.random.set_seed(42)

import google.protobuf

print("TF version:", tf.__version__)
print("protobuf version:", google.protobuf.__version__)



## === cell 1
from pathlib import Path

CANDIDATE_ROOTS = [
    Path("../input/petfinder-pawpularity-score"),
    Path("/kaggle/input/petfinder-pawpularity-score"),
    Path("/kaggle/data/petfinder-pawpularity-score"),
    Path("../input/petfinder-pawpularity-score/petfinder-pawpularity-score"),
    Path("/kaggle/input/petfinder-pawpularity-score/petfinder-pawpularity-score"),
    Path("/kaggle/data/petfinder-pawpularity-score/petfinder-pawpularity-score"),
]
path = next((p for p in CANDIDATE_ROOTS if (p / "train.csv").exists()), None)
if path is None:
    raise FileNotFoundError(
        "Could not locate competition data folder containing train.csv in known locations."
    )

print("Using data path:", path)



## === cell 2
train = pd.read_csv(path / "train.csv")
train.head()



## === cell 3
train["Pawpularity"] = train["Pawpularity"] / 100.0
train.head()



## === cell 4
train.columns



## === cell 5
img_size = 140
train_image = []

train_dir = path / "train"
for i in tqdm(range(train.shape[0])):
    img = image.load_img(
        str(train_dir / f"{train['Id'][i]}.jpg"),
        target_size=(img_size, img_size),
    )
    img = image.img_to_array(img)
    img = img / 255.0
    train_image.append(img)

X = np.array(train_image)



## === cell 6
X.shape



## === cell 7
plt.imshow(X[5])
plt.axis("off")



## === cell 8
train["Pawpularity"][2]



## === cell 9
y = train["Pawpularity"].values.astype(np.float32)
y.shape



## === cell 10
X_train, X_test, y_train, y_test = train_test_split(
    X, y, random_state=42, test_size=0.2
)



## === cell 11
y_train[0:4]



## === cell 12
from tensorflow.keras.metrics import RootMeanSquaredError
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import (
    Conv2D,
    MaxPool2D,
    Dense,
    Dropout,
    GlobalAveragePooling2D,
)



## === cell 13
inputs = keras.Input(shape=(img_size, img_size, 3))

Z = Conv2D(filters=64, kernel_size=3, strides=2, padding="same", activation="relu")(
    inputs
)
Z = MaxPool2D(pool_size=(2, 2))(Z)

Z = Conv2D(filters=128, kernel_size=3, strides=2, padding="same", activation="relu")(Z)
Z = MaxPool2D(pool_size=(2, 2))(Z)

Z = Conv2D(filters=256, kernel_size=3, strides=2, padding="same", activation="relu")(Z)

Z = GlobalAveragePooling2D()(Z)

Z = Dense(512, activation="relu")(Z)
Z = Dropout(0.3)(Z)

outputs = Dense(1)(Z)

model = keras.Model(inputs=inputs, outputs=outputs)



## === cell 14
model.summary()



## === cell 15
LEARNING_RATE = 0.001
model.compile(
    optimizer=Adam(learning_rate=LEARNING_RATE),
    loss="msle",
    metrics=[RootMeanSquaredError()],
)



## === cell 16
model.fit(X_train, y_train, epochs=64, validation_data=(X_test, y_test), batch_size=64)



## === cell 17
from tensorflow.keras.preprocessing.image import ImageDataGenerator



## === cell 18
df_test = pd.read_csv(path / "test.csv", index_col="Id")

test_path = path / "test"
df_test.loc[:, "filename"] = df_test.index + ".jpg"

RESCALE = 1.0 / 255.0
datagen = ImageDataGenerator(rescale=RESCALE)

test_generator = datagen.flow_from_dataframe(
    dataframe=df_test,
    directory=str(test_path),
    x_col="filename",
    y_col=None,
    batch_size=32,
    seed=42,
    shuffle=False,
    class_mode=None,
    target_size=(img_size, img_size),
)



## === cell 19
yhat_test = model.predict(test_generator, verbose=1)



## === cell 20
yhat_test



## === cell 21
yhat_test_1 = yhat_test.reshape(-1) * 100.0
yhat_test_1 = np.clip(yhat_test_1, 1.0, 100.0)
print(yhat_test_1)



## === cell 22
frames = [
    pd.Series(df_test.index.values, name="Id"),
    pd.Series(yhat_test_1.ravel(), name="Pawpularity"),
]
submission = pd.concat(frames, axis=1)
submission.head()



## === cell 23
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.describe(include="all"))
