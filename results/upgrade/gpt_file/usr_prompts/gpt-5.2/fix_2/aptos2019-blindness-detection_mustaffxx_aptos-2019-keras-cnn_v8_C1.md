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

geopandas==0.14.4
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
tf_keras==2.18.0

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

-0.000652

# 6. Current score

0.63546

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

- What this solution (achieved 0.63546) has done: 'I first fix the environment crash caused by an incompatible `protobuf` version with TensorFlow by pinning protobuf to a TF-compatible version at runtime before importing TensorFlow. Next, I update Keras imports that broke under Keras 3 (`to_categorical`, `Conv2D/MaxPooling2D`) while keeping the same model and training loop. Finally, I fix image loading paths to match Kaggle’s dataset directory and correct the inference loop that was indexing predictions incorrectly, ensuring the output CSV contains exactly `id_code` and `diagnosis` with the right length.'

# 9. Code solution

## === cell 0
import os, sys, subprocess

try:
    import google.protobuf  # noqa: F401
    import google.protobuf.__version__ as pb_ver  # type: ignore
except Exception:
    pb_ver = None


def _ver_tuple(v):
    try:
        return tuple(int(x) for x in str(v).split(".")[:3])
    except Exception:
        return (999, 999, 999)


if pb_ver is None or _ver_tuple(pb_ver) >= (5, 0, 0):
    subprocess.check_call(
        [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
    )

import tensorflow as tf
from tensorflow import keras
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

print("TF version:", tf.__version__)
print("Keras version:", keras.__version__)



## === cell 1
import os

print(os.listdir("../input"))



## === cell 2
PATH = "../input/aptos2019-blindness-detection/"
train = pd.read_csv(os.path.join(PATH, "train.csv"))
train.head()



## === cell 3
train["diagnosis"].value_counts().plot(kind="bar")



## === cell 4
train["diagnosis"].value_counts().sort_index()[0]



## === cell 5
total = train["diagnosis"].value_counts().sum()
w0 = train["diagnosis"].value_counts().sort_index()[0] / total
w1 = train["diagnosis"].value_counts().sort_index()[1] / total
w2 = train["diagnosis"].value_counts().sort_index()[2] / total
w3 = train["diagnosis"].value_counts().sort_index()[3] / total
w4 = train["diagnosis"].value_counts().sort_index()[4] / total
class_wg = {0: w3, 1: w1, 2: w4, 3: w0, 4: w2}

print("class weights:")
print("0: ", w0)
print("1: ", w1)
print("2: ", w2)
print("3: ", w3)
print("4: ", w4)



## === cell 6
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.imagenet_utils import preprocess_input

img = image.load_img(
    os.path.join(PATH, "train_images", train["id_code"][0] + ".png"),
    target_size=(100, 100, 3),
)
plt.imshow(img)




## === cell 7
def load_images(df, dfPath):
    xdata = np.zeros((df.shape[0], 100, 100, 3), dtype=np.float32)
    index = 0
    for id_code in df["id_code"].values:
        img = image.load_img(
            os.path.join(PATH, dfPath, id_code + ".png"),
            target_size=(100, 100, 3),
        )
        x = image.img_to_array(img)
        x = preprocess_input(x)
        xdata[index] = x
        index += 1
    xdata = xdata / 255.0
    return xdata




## === cell 8
x_train = train.drop(["diagnosis"], axis=1)
y_train = train["diagnosis"]



## === cell 9
trainpath = "train_images"
x_train = load_images(train, trainpath)



## === cell 10
plt.imshow(x_train[0][:, :, :])



## === cell 11
plt.imshow(x_train[0][:, :, 0])



## === cell 12
plt.imshow(x_train[0][:, :, 1])



## === cell 13
plt.imshow(x_train[0][:, :, 2])



## === cell 14
y_train.head()



## === cell 15
print(y_train.shape)



## === cell 16
del train



## === cell 17
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.utils import to_categorical

lb = LabelEncoder()
y_train = lb.fit_transform(y_train)
y_train = to_categorical(y_train, num_classes=5)



## === cell 18
y_train



## === cell 19
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D

model = Sequential()

model.add(
    Conv2D(filters=30, kernel_size=(5, 5), input_shape=(100, 100, 3), activation="relu")
)
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Conv2D(filters=15, kernel_size=(3, 3), activation="relu"))
model.add(Conv2D(filters=15, kernel_size=(3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dense(64, activation="relu"))
model.add(Dense(32, activation="relu"))

model.add(Dense(5, activation="softmax"))

model.compile(
    loss="categorical_crossentropy", optimizer="adam", metrics=["categorical_accuracy"]
)
model.summary()



## === cell 20
model.fit(x_train, y_train, epochs=50, batch_size=200, class_weight=class_wg)



## === cell 21
del x_train, y_train



## === cell 22
testPath = "test_images"
test = pd.read_csv(os.path.join(PATH, "test.csv"))
test.head()



## === cell 23
x_test = load_images(test, testPath)



## === cell 24
predictions = model.predict(x_test, verbose=1)



## === cell 25
test["diagnosis"] = np.argmax(predictions, axis=1).astype(int)



## === cell 26
test.head()



## === cell 27
submission = test[["id_code", "diagnosis"]].copy()
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
