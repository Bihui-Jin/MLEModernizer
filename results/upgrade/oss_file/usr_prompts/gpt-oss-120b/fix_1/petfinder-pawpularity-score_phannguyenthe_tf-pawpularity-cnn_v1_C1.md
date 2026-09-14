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

20.73367

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import matplotlib
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import cv2

import tensorflow as tf
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Input, Dense, Flatten, Dropout, Activation, BatchNormalization, concatenate
from tensorflow.keras.models import Model

import os
path = '../input/petfinder-pawpularity-score/'
print(os.listdir(path))
    


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_data = pd.read_csv(path + 'train.csv')
train_data.head(5)


## === cell 2
train_data.info()


## === cell 3
train_data['Pawpularity'].hist(bins=100,figsize=(15, 6))
plt.title("Pawpularity distribution",fontsize=20)


## === cell 4
train_data.nunique()


## === cell 5
X_train_data = train_data.drop(['Id','Pawpularity'], axis=1)
y = train_data['Pawpularity']
print(X_train_data.shape)
y.shape


## === cell 6
X_train, X_test, y_train, y_test = train_test_split(X_train_data, y, test_size=0.2, random_state=42)
print ('X_train: ', X_train.shape)
print ('X_test: ', X_test.shape)
print ('y_train: ', y_train.shape)
print ('y_test: ', y_test.shape)


## === cell 19
imgSize = 128


## === cell 20
X_img = []
for i, row in train_data.iterrows():
    rawImg = cv2.imread(path+'train/'+row['Id']+'.jpg')
    image = cv2.resize(rawImg, (imgSize,imgSize), interpolation = cv2.INTER_AREA)
    img = np.array(image)
    img = img.astype('float32')
    img /= 255 
    
    X_img.append(img)

X_img = np.array(X_img)
X_img.shape


## === cell 21
img = cv2.resize(X_img[0], (imgSize,imgSize), interpolation = cv2.INTER_AREA)
print('Image shape:', img.shape)

fig, axs = plt.subplots(1, 1, figsize=(7, 7))
axs.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
axs.set_xticklabels([])
axs.set_yticklabels([])
plt.show()


## === cell 22

X_trainNum, X_testNum, y_trainNum, y_testNum = train_test_split(X_train_data, y, test_size=0.2, random_state=42)
print ('X_trainNum: ', X_trainNum.shape)
print ('X_testNum: ', X_testNum.shape)
print ('y_train: ', y_train.shape)
print ('y_test: ', y_test.shape)

X_trainImg, X_testImg, y_trainImg, y_testImg = train_test_split(X_img, y, test_size=0.2, random_state=42)
print ('X_trainImg: ', X_trainImg.shape)
print ('X_testImg: ', X_testImg.shape)


## === cell 23
def create_mlp(dim,regress=False):
    model = keras.Sequential()
    model.add(Dense(64, input_dim=dim, activation="relu"))
    model.add(Dropout(0.5))
    if regress:
        model.add(Dense(1, activation="linear"))
    return model

def create_cnn(width, height, depth, filters=(16, 32, 64), regress=False):
    inputShape = (128,128,3)

    inputs = Input(shape=inputShape)

    for (i, f) in enumerate(filters):

        if i == 0:
            x = inputs

        x = Conv2D(f, (3, 3), padding="same")(x)
        x = Activation("relu")(x)
        x = BatchNormalization(axis=-1)(x)
        x = MaxPooling2D(pool_size=(2, 2))(x)
    
    x = Flatten()(x)
    x = Dense(64)(x)
    x = Activation("relu")(x)
    x = BatchNormalization(axis=-1)(x)
    x = Dropout(0.5)(x)

    x = Dense(16)(x)
    x = Activation("relu")(x)

    if regress:
        x = Dense(1, activation="linear")(x)
        
    model = Model(inputs, x)

    return model


## === cell 24
mlp = create_mlp(X_trainNum.shape[1], regress=False)
cnn = create_cnn(imgSize, imgSize, 3, regress=False)

combinedInput = concatenate([mlp.output, cnn.output])

x = Dense(64, activation="relu")(combinedInput)
x = Dense(1, activation="linear")(x)

model = Model(inputs=[mlp.input, cnn.input], outputs=x)

model.compile('Adam', 'mse', metrics=[tf.keras.metrics.RootMeanSquaredError()])
model.summary()


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_12/1474710370.py in <cell line: 0>()
      2 cnn = create_cnn(imgSize, imgSize, 3, regress=False)
      3 
----> 4 combinedInput = concatenate([mlp.output, cnn.output])
      5 
      6 x = Dense(64, activation="relu")(combinedInput)

/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py in output(self)
    264             Output tensor or list of output tensors.
    265         """
--> 266         return self._get_node_attribute_at_index(0, "output_tensors", "output")
    267 
    268     def _get_node_attribute_at_index(self, node_index, attr, attr_name):

/usr/local/lib/python3.11/dist-packages/keras/src/ops/operation.py in _get_node_attribute_at_index(self, node_index, attr, attr_name)
    283         """
    284         if not self._inbound_nodes:
--> 285             raise AttributeError(
    286                 f"The layer {self.name} has never been called "
    287                 f"and thus has no defined {attr_name}."

AttributeError: The layer sequential has never been called and thus has no defined output.

## === cell 25
history = model.fit([X_trainNum, X_trainImg], y_train, 
                    validation_split = 0.2,
                    batch_size = 4,
                    epochs = 20)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/854991327.py in <cell line: 0>()
----> 1 history = model.fit([X_trainNum, X_trainImg], y_train, 
      2                     validation_split = 0.2,
      3                     batch_size = 4,
      4                     epochs = 20)

NameError: name 'model' is not defined

## === cell 26
model.save('v2210_model.h5')


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/289190277.py in <cell line: 0>()
----> 1 model.save('v2210_model.h5')

NameError: name 'model' is not defined

## === cell 27
acc = history.history['root_mean_squared_error']
val_acc = history.history['val_root_mean_squared_error']
loss = history.history['loss']
val_loss = history.history['val_loss']
epochs = range(1, len(acc)+1)

plt.plot(epochs, acc, 'o', label='Training rmse',markerfacecolor='blue')
plt.plot(epochs, val_acc, marker='', label='Validation rmse',)
plt.title('Training and validation root_mean_squared_error')
plt.legend()
plt.figure()

plt.plot(epochs, loss, 'o', label='Training loss',markerfacecolor='blue')
plt.plot(epochs, val_loss, marker='', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/357182340.py in <cell line: 0>()
----> 1 acc = history.history['root_mean_squared_error']
      2 val_acc = history.history['val_root_mean_squared_error']
      3 loss = history.history['loss']
      4 val_loss = history.history['val_loss']
      5 epochs = range(1, len(acc)+1)

NameError: name 'history' is not defined

## === cell 28
test_data = pd.read_csv(path+'test.csv')
print(test_data.shape)
test_data.head(5)


## === cell 29
samp_sub = pd.read_csv(path+'sample_submission.csv')
samp_sub.head(5)


## === cell 30
X_test_data = test_data.drop(['Id'], axis=1)
X_test_data.head(5)


## === cell 31
X_test_img = []
for i, row in test_data.iterrows():
    rawImg = cv2.imread(path+'test/'+row['Id']+'.jpg')
    image = cv2.resize(rawImg, (imgSize,imgSize), interpolation = cv2.INTER_AREA)
    img = np.array(image)
    img = img.astype('float32')
    img /= 255 
    
    X_test_img.append(img)

X_test_img = np.array(X_test_img)
X_test_img.shape


## === cell 32
preds = model.predict([X_test_data, X_test_img])
preds


## --- ERROR in cell 32, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/156666706.py in <cell line: 0>()
----> 1 preds = model.predict([X_test_data, X_test_img])
      2 preds

NameError: name 'model' is not defined

## === cell 33
preds = preds.flatten()
preds


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/1769088118.py in <cell line: 0>()
----> 1 preds = preds.flatten()
      2 preds

NameError: name 'preds' is not defined

## === cell 34
mySubmit = pd.DataFrame(test_data.Id)
mySubmit['Pawpularity'] = preds
mySubmit.head(5)


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_12/199571363.py in <cell line: 0>()
      1 mySubmit = pd.DataFrame(test_data.Id)
----> 2 mySubmit['Pawpularity'] = preds
      3 mySubmit.head(5)

NameError: name 'preds' is not defined

## === cell 35
mySubmit.to_csv('submission.csv', index=False)


## --- ERROR in outputing the csv:
Invalid submission: Missing Pawpularity column in submission
