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
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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

28.67934

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D
from tensorflow.keras.callbacks import EarlyStopping

from tensorflow.keras.utils import to_categorical




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
trainimg0 = []
trainlabel0 = []
for im in tqdm(os.listdir(train_dir)):
    image_path = os.path.join(train_dir, im)
    image = cv2.imread(image_path)
    if image is None:
        continue
    image2 = cv2.resize(image, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
    trainimg0.append(image2)
    trainlabel0.append(train[train["Id"] == im[0:-4]]["Pawpularity"].tolist()[0])




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/105354496.py in <cell line: 0>()
      1 trainimg0 = []
      2 trainlabel0 = []
----> 3 for im in tqdm(os.listdir(train_dir)):
      4     image_path = os.path.join(train_dir, im)
      5     image = cv2.imread(image_path)

NameError: name 'tqdm' is not defined

## === cell 2
TESTX = []
testim = []
for im in tqdm(os.listdir(test_dir)):
    image_path = os.path.join(test_dir, im)
    image = cv2.imread(image_path)
    if image is None:
        continue
    image2 = cv2.resize(image, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
    TESTX.append(image2)
    testim.append(im[0:-4])




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2646647056.py in <cell line: 0>()
      1 TESTX = []
      2 testim = []
----> 3 for im in tqdm(os.listdir(test_dir)):
      4     image_path = os.path.join(test_dir, im)
      5     image = cv2.imread(image_path)

NameError: name 'tqdm' is not defined

## === cell 3
import numpy as np
import pandas as pd
import os
import cv2
import random
from tqdm import tqdm
import seaborn as sns
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import matplotlib.pyplot as plt




## === cell 4
train_dir = "../input/petfinder-pawpularity-score/train"
test_dir = "../input/petfinder-pawpularity-score/test"




## === cell 5
path0 = (
    "../input/petfinder-pawpularity-score/train/0007de18844b0dbbb5e1f607da0606e0.jpg"
)
image = cv2.imread(path0)
print(image.shape)
plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))




## === cell 6
image2 = cv2.resize(image, dsize=(60, 60), interpolation=cv2.INTER_CUBIC)
print(image2.shape)
plt.imshow(cv2.cvtColor(image2, cv2.COLOR_BGR2RGB))




## === cell 7
train = pd.read_csv("../input/petfinder-pawpularity-score/train.csv")
train




## === cell 8
train["Pawpularity"].unique()




## === cell 9
N0 = list(range(100))
N1 = list(range(1, 101))
normal_mapping = dict(zip(N1, N0))
reverse_mapping = dict(zip(N0, N1))




## === cell 10
train[train["Id"] == "0007de18844b0dbbb5e1f607da0606e0"]["Pawpularity"].tolist()[0]




## === cell 11
trainlabel1 = pd.Series(trainlabel0).map(normal_mapping)




## === cell 12
trainimg = np.array(trainimg0)
trainlabel = np.array(trainlabel1)




## === cell 13
m = len(trainimg)
M = list(range(m))
random.seed(2021)
random.shuffle(M)




## === cell 14
trainX = trainimg[M[0 : (m // 4) * 3]]
trainY0 = trainlabel[M[0 : (m // 4) * 3]]

testX = trainimg[M[(m // 4) * 3 :]]
testY0 = trainlabel[M[(m // 4) * 3 :]]




## === cell 15
labels1 = to_categorical(trainY0)
trainY = np.array(labels1)




## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/1338698178.py in <cell line: 0>()
----> 1 labels1 = to_categorical(trainY0)
      2 trainY = np.array(labels1)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/numerical_utils.py in to_categorical(x, num_classes)
     94     x = x.reshape(-1)
     95     if not num_classes:
---> 96         num_classes = np.max(x) + 1
     97     batch_size = x.shape[0]
     98     categorical = np.zeros((batch_size, num_classes))

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in max(a, axis, out, keepdims, initial, where)
   2808     5
   2809     """
-> 2810     return _wrapreduction(a, np.maximum, 'max', axis, None, out,
   2811                           keepdims=keepdims, initial=initial, where=where)
   2812 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
     86                 return reduction(axis=axis, out=out, **passkwargs)
     87 
---> 88     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
     89 
     90 

ValueError: zero-size array to reduction operation maximum which has no identity

## === cell 16
trainx, testx, trainy, testy = train_test_split(
    trainX, trainY, test_size=0.2, random_state=44
)




## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1730578828.py in <cell line: 0>()
      1 trainx, testx, trainy, testy = train_test_split(
----> 2     trainX, trainY, test_size=0.2, random_state=44
      3 )
      4 
      5 

NameError: name 'trainY' is not defined

## === cell 17
print(trainx.shape)
print(testx.shape)
print(trainy.shape)
print(testy.shape)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/206581060.py in <cell line: 0>()
----> 1 print(trainx.shape)
      2 print(testx.shape)
      3 print(trainy.shape)
      4 print(testy.shape)
      5 

NameError: name 'trainx' is not defined

## === cell 18
model = Sequential()
model.add(Conv2D(32, (4, 4), input_shape=(60, 60, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.2))
model.add(Flatten())
model.add(Dense(400, activation="relu"))
model.add(Dense(100, activation="softmax"))




## === cell 19
model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"])




## === cell 20
model.summary()




## === cell 21
his = model.fit(
    trainx, trainy, validation_split=0.2, epochs=30, batch_size=92, verbose=2
)




## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1834593052.py in <cell line: 0>()
      1 his = model.fit(
----> 2     trainx, trainy, validation_split=0.2, epochs=30, batch_size=92, verbose=2
      3 )
      4 
      5 

NameError: name 'trainx' is not defined

## === cell 22
y_pred = model.predict(testx)
pred = np.argmax(y_pred, axis=1)
ground = np.argmax(testy, axis=1)
print(classification_report(ground, pred))




## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/769250184.py in <cell line: 0>()
----> 1 y_pred = model.predict(testx)
      2 pred = np.argmax(y_pred, axis=1)
      3 ground = np.argmax(testy, axis=1)
      4 print(classification_report(ground, pred))
      5 

NameError: name 'testx' is not defined

## === cell 23
get_acc = his.history["accuracy"]
value_acc = his.history["val_accuracy"]
get_loss = his.history["loss"]
validation_loss = his.history["val_loss"]

epochs = range(len(get_acc))
plt.plot(epochs, get_acc, "r", label="Accuracy of Training data")
plt.plot(epochs, value_acc, "b", label="Accuracy of Validation data")
plt.title("Training vs validation accuracy")
plt.legend(loc=0)
plt.figure()
plt.show()




## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3063648374.py in <cell line: 0>()
----> 1 get_acc = his.history["accuracy"]
      2 value_acc = his.history["val_accuracy"]
      3 get_loss = his.history["loss"]
      4 validation_loss = his.history["val_loss"]
      5 

NameError: name 'his' is not defined

## === cell 24
epochs = range(len(get_loss))
plt.plot(epochs, get_loss, "r", label="Loss of Training data")
plt.plot(epochs, validation_loss, "b", label="Loss of Validation data")
plt.title("Training vs validation loss")
plt.legend(loc=0)
plt.figure()
plt.show()




## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2860667940.py in <cell line: 0>()
----> 1 epochs = range(len(get_loss))
      2 plt.plot(epochs, get_loss, "r", label="Loss of Training data")
      3 plt.plot(epochs, validation_loss, "b", label="Loss of Validation data")
      4 plt.title("Training vs validation loss")
      5 plt.legend(loc=0)

NameError: name 'get_loss' is not defined

## === cell 25
pred2 = model.predict(testX)
print(pred2.shape)

PRED = []
for item in pred2:
    value2 = np.argmax(item)
    PRED += [value2]
print(pd.Series(PRED).value_counts())




## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/311518695.py in <cell line: 0>()
----> 1 pred2 = model.predict(testX)
      2 print(pred2.shape)
      3 
      4 PRED = []
      5 for item in pred2:

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 26
ANS = testY0
print(pd.Series(ANS).value_counts())
accuracy = accuracy_score(ANS, PRED)
print(accuracy)




## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1531957908.py in <cell line: 0>()
      1 ANS = testY0
      2 print(pd.Series(ANS).value_counts())
----> 3 accuracy = accuracy_score(ANS, PRED)
      4 print(accuracy)
      5 

NameError: name 'PRED' is not defined

## === cell 27
import seaborn as sns

fig, ax = plt.subplots(figsize=(14, 5))
sns.histplot(ANS, label="ANS", ax=ax, color="black", bins=100)
sns.histplot(PRED, label="PRED", ax=ax, color="C1", bins=100)
ax.legend()
ax.grid()
plt.show()




## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/4204716671.py in <cell line: 0>()
      3 fig, ax = plt.subplots(figsize=(14, 5))
      4 sns.histplot(ANS, label="ANS", ax=ax, color="black", bins=100)
----> 5 sns.histplot(PRED, label="PRED", ax=ax, color="C1", bins=100)
      6 ax.legend()
      7 ax.grid()

NameError: name 'PRED' is not defined

## === cell 28
fig, axs = plt.subplots(3, 3, figsize=(12, 12))
for i in range(9):
    r = i // 3
    c = i % 3
    img1 = testX[i]
    ax = axs[r][c].axis("off")
    actual = reverse_mapping[testY0[i]]
    predict = reverse_mapping[PRED[i]]
    ax = axs[r][c].set_title(str(actual) + "==" + str(predict))
    ax = axs[r][c].imshow(img1)
plt.show()




## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_55/3389521722.py in <cell line: 0>()
      3     r = i // 3
      4     c = i % 3
----> 5     img1 = testX[i]
      6     ax = axs[r][c].axis("off")
      7     actual = reverse_mapping[testY0[i]]

IndexError: index 0 is out of bounds for axis 0 with size 0

## === cell 29
TESTX = np.array(TESTX)
print(TESTX.shape)




## === cell 30
test_pred2 = model.predict(TESTX)

TESTPRED = []
for item in test_pred2:
    value = np.argmax(item)
    value2 = reverse_mapping[value]
    TESTPRED += [float(value2)]
print(pd.Series(TESTPRED).value_counts())




## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_55/3782595101.py in <cell line: 0>()
----> 1 test_pred2 = model.predict(TESTX)
      2 
      3 TESTPRED = []
      4 for item in test_pred2:
      5     value = np.argmax(item)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/progbar.py in update(self, current, values, finalize)
    117 
    118             if self.target is not None:
--> 119                 numdigits = int(math.log10(self.target)) + 1
    120                 bar = ("%" + str(numdigits) + "d/%d") % (current, self.target)
    121                 bar = f"\x1b[1m{bar}\x1b[0m "

ValueError: math domain error

## === cell 31
sample = pd.read_csv("../input/petfinder-pawpularity-score/sample_submission.csv")
sample




## === cell 32
result = pd.DataFrame(testim)
result[1] = TESTPRED
result.columns = ["Id", "Pawpularity"]
result2 = result.sort_values("Id")
result2




## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3709399590.py in <cell line: 0>()
      1 result = pd.DataFrame(testim)
----> 2 result[1] = TESTPRED
      3 result.columns = ["Id", "Pawpularity"]
      4 result2 = result.sort_values("Id")
      5 result2

NameError: name 'TESTPRED' is not defined

## === cell 33
result2.to_csv("submission.csv", index=False)

## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1096448722.py in <cell line: 0>()
----> 1 result2.to_csv("submission.csv", index=False)

NameError: name 'result2' is not defined
