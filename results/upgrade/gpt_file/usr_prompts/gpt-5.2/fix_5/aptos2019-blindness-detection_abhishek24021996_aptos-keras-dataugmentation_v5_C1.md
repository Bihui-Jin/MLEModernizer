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
Create a classifier to predict the severity of diabetic retinopathy.

## Metric
Quadratic weighted kappa, which measures the agreement between two ratings. This metric typically varies from 0 (random agreement between raters) to 1 (complete agreement between raters). In the event that there is less agreement between the raters than expected by chance, this metric may go below 0. The quadratic weighted kappa is calculated between the scores assigned by the human rater and the predicted scores.

Images have five possible ratings, 0,1,2,3,4.  Each image is characterized by a tuple *(e*,*e)*, which corresponds to its scores by *Rater A* (human) and *Rater B* (predicted).  The quadratic weighted kappa is calculated as follows. First, an N x N histogram matrix *O* is constructed, such that *O* corresponds to the number of images that received a rating *i* by *A* and a rating *j* by *B*. An *N-by-N* matrix of weights, *w*, is calculated based on the difference between raters' scores:

An *N-by-N* histogram matrix of expected ratings, *E*, is calculated, assuming that there is no correlation between rating scores.  This is calculated as the outer product between each rater's histogram vector of ratings, normalized such that *E* and *O* have the same sum.

## Submission Format
```
id_code,diagnosis
0005cfc8afb6,0
003f0afdcd15,0
etc.
```

## Dataset
You are provided with a large set of retina images taken using [fundus photography](https://en.wikipedia.org/wiki/Fundus_photography) under a variety of imaging conditions.

Labels are on a scale of 0 to 4:

> 0 - No DR
> 1 - Mild
> 2 - Moderate
> 3 - Severe
> 4 - Proliferative DR

Images may contain artifacts, be out of focus, underexposed, or overexposed. The images were gathered from multiple clinics using a variety of cameras over an extended period of time, which will introduce further variation.

- **train.csv** - the training labels
- **test.csv** - the test set (you must predict the `diagnosis` value for these variables)
- **sample_submission.csv** - a sample submission file in the correct format
- **train.zip** - the training set images
- **test.zip** - the public test set images

# 2. Python version

3.7

# 3. Installed packages



# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
        input/
            description.md (118 lines)
            sample_submission.csv (368 lines)
            sample_submission.csv.zip (3.2 kB)
            test.csv (368 lines)
            test.csv.zip (2.9 kB)
            test.zip (160 Bytes)
            test_images.zip (902.9 MB)
            train.csv (3296 lines)
            train.csv.zip (27.5 kB)
            train.zip (162 Bytes)
            train_images.zip (7.7 GB)
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
            test_images/
                218c822a3dd9.png (5.7 MB)
                0e82bcacc475.png (5.2 MB)
                ... and 365 other files
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
            train_images/
                184a185e7447.png (337.5 kB)
                c4aef0d88d1b.png (876.6 kB)
                ... and 3293 other files
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
        working/
            aptos2019-blindness-detection/
                description.md (118 lines)
                sample_submission.csv (368 lines)
                ... and 9 other files
                aptos2019-blindness-detection/
                test_images/
                    218c822a3dd9.png (5.7 MB)
                    0e82bcacc475.png (5.2 MB)
                    ... and 365 other files
                    test_images/
                train_images/
                    184a185e7447.png (337.5 kB)
                    c4aef0d88d1b.png (876.6 kB)
                    ... and 3293 other files
                    train_images/
```

-> data/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/aptos2019-blindness-detection/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/aptos2019-blindness-detection/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> data/test.csv has 367 rows and 1 columns.
The columns are: id_code

-> data/train.csv has 3295 rows and 2 columns.
The columns are: id_code, diagnosis

-> input/aptos2019-blindness-detection/sample_submission.csv has 367 rows and 2 columns.
The columns are: id_code, diagnosis

-> (stopped after 10 files for performance)

# 5. Target score

0.470048

# 6. Current score

0.70433

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.70433) has done: 'I fix the runtime-breaking API mismatches by switching the imports from legacy `keras` to `tensorflow.keras`, which avoids the protobuf `MessageFactory` error and restores the expected training/prediction methods. I minimally update deprecated/removed calls (`fit_generator`, `predict_classes`) to their current equivalents while keeping the same CNN architecture and training loop semantics. I also fix a few critical data bugs (a stray indent, missing `/255` scaling for test images, and incorrect input paths) so train/test tensors have consistent shapes and ranges. Finally, I ensure a valid `submission.csv` with the exact required columns is always written.'
- What this solution (achieved 0.70433) has done: 'The runtime error is happening before your notebook even reaches the training code, due to a TensorFlow/Protobuf incompatibility that can surface as `MessageFactory` missing `GetPrototype`. The safest minimal fix in Kaggle is to force the pure-Python protobuf implementation *before* importing TensorFlow, which avoids that failing code path. Since your current score (0.70433) is already above the target (0.470048) and higher-is-better, I not change any modeling/training logic that would intentionally move score downward; I focus strictly on restoring end-to-end execution and producing a valid `submission.csv`. I keep the same architecture, data pipeline, training loop, and submission formatting.'
- What this solution (achieved 0.70433) has done: 'I fix the TensorFlow import crash (`MessageFactory` / protobuf incompatibility) by ensuring the pure-Python protobuf implementation is forced *before* any TensorFlow/protobuf-dependent import, and by also disabling the C++ implementation explicitly. This is execution-critical and score-neutral because it doesn’t change your model, data, or training logic. I also add a small safety fallback to use the alternative dataset root (`/kaggle/data/...`) if `/kaggle/input/...` isn’t present, preventing path-related failures while keeping the same files. Finally, I keep the same submission writing logic but ensure the output is always a proper `submission.csv` with required columns and integer diagnoses.'
- What this solution (achieved 0.70433) has done: 'I fix the TensorFlow/protobuf import crash by forcing the pure-Python protobuf implementation *before* any TensorFlow-related import and by also setting the `PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION` override (not just `setdefault`), which is the direct cause of the `MessageFactory.GetPrototype` error. I keep the exact same model, data loading, preprocessing, training loop, and prediction logic so the score behavior remains consistent with your previous 0.70433 submission (we won’t intentionally move it toward a lower target since higher-is-better). I also add a small, score-neutral safety check to ensure the submission is aligned to `test.csv` order and always writes `submission.csv` with the required columns. No architectural/training changes are introduced.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION"] = "2"

import sys
import random
import numpy as np
import pandas as pd
import cv2

from tqdm import tqdm
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "/kaggle/input/aptos2019-blindness-detection"
if not os.path.exists(BASE_PATH):
    alt = "/kaggle/data/aptos2019-blindness-detection"
    if os.path.exists(alt):
        BASE_PATH = alt

train_img_path = os.path.join(BASE_PATH, "train_images")
test_img_path = os.path.join(BASE_PATH, "test_images")
train_label_path = os.path.join(BASE_PATH, "train.csv")
test_label_path = os.path.join(BASE_PATH, "test.csv")

df_train = pd.read_csv(train_label_path)
df_test = pd.read_csv(test_label_path)

print("BASE_PATH:", BASE_PATH)
print("num of train images ", len(os.listdir(train_img_path)))
print("num of test images ", len(os.listdir(test_img_path)))
print(df_train.head())
print(df_test.head())



## === cell 2
pass



## === cell 3
import matplotlib.pyplot as plt

df_train["diagnosis"].value_counts().plot(kind="bar")
plt.title("Level of diagnosis")
plt.show()



## === cell 4
samp = random.sample(df_train["id_code"].tolist(), 3)
for i, sid in enumerate(samp, 1):
    plt.figure(figsize=(6, 6))
    file_path = os.path.join(train_img_path, f"{sid}.png")
    img = cv2.imread(file_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    plt.imshow(img)
    plt.axis("off")
    plt.title(sid)
    plt.show()



## === cell 5
pass



## === cell 6
df_train_img = []
train_list = df_train["id_code"].tolist()

for item in tqdm(train_list, desc="Loading train images"):
    file_path = os.path.join(train_img_path, f"{item}.png")
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {file_path}")
    img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
    df_train_img.append(img)

df_train_img = np.array(df_train_img, np.float32) / 255.0
print(
    "Train tensor:",
    df_train_img.shape,
    df_train_img.dtype,
    df_train_img.min(),
    df_train_img.max(),
)



## === cell 7
df_test_img = []
for item in tqdm(df_test["id_code"].tolist(), desc="Loading test images"):
    file_path = os.path.join(test_img_path, f"{item}.png")
    img = cv2.imread(file_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {file_path}")
    img = cv2.resize(img, (150, 150), interpolation=cv2.INTER_AREA)
    df_test_img.append(img)

df_test_img = np.array(df_test_img, np.float32) / 255.0
print(
    "Test tensor:",
    df_test_img.shape,
    df_test_img.dtype,
    df_test_img.min(),
    df_test_img.max(),
)



## === cell 8
y_train = (df_train.iloc[:, 1].values).astype("int32")
print("y_train shape:", y_train.shape, "classes:", np.unique(y_train))



## === cell 9
y_train



## === cell 10
X = df_train_img
Y = y_train
x_train, x_val, y_train, y_val = train_test_split(
    X, Y, test_size=0.15, random_state=42, stratify=Y
)
print("Split:", x_train.shape, x_val.shape, y_train.shape, y_val.shape)



## === cell 11
pass



## === cell 12
pass



## === cell 13
gen = ImageDataGenerator()



## === cell 14
batches = gen.flow(x_train, y_train, batch_size=64, shuffle=True)
val_batches = gen.flow(x_val, y_val, batch_size=64, shuffle=False)



## === cell 15
pass



## === cell 16
pass



## === cell 17
classifier = Sequential()
classifier.add(Conv2D(32, (3, 3), input_shape=(150, 150, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Conv2D(32, (3, 3), activation="relu"))
classifier.add(MaxPooling2D(pool_size=(2, 2)))
classifier.add(Flatten())
classifier.add(Dense(units=75, activation="relu"))
classifier.add(Dense(units=5, activation="softmax"))

classifier.compile(
    optimizer="nadam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)
classifier.summary()



## === cell 18
hist = classifier.fit(
    batches,
    steps_per_epoch=len(batches),
    epochs=3,
    validation_data=val_batches,
    validation_steps=len(val_batches),
    verbose=1,
)



## === cell 19
pass



## === cell 20
probs = classifier.predict(df_test_img, batch_size=64, verbose=1)
predictions = np.argmax(probs, axis=1).astype(int)

if len(predictions) != len(df_test):
    raise ValueError(f"Pred length {len(predictions)} != test length {len(df_test)}")

submissions = pd.DataFrame(
    {"id_code": df_test["id_code"].values, "diagnosis": predictions}
)
submissions["diagnosis"] = submissions["diagnosis"].astype(int)

submissions.to_csv("submission.csv", index=False, header=True)

print(submissions.head())
print("Wrote submission.csv with shape:", submissions.shape)
print("Submission columns:", submissions.columns.tolist())
