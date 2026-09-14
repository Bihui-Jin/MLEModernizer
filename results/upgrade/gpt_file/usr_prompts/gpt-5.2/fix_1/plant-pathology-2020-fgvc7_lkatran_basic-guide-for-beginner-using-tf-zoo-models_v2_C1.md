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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.8

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.7953659417082056

# 6. Current score

0.35014

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 1
from tqdm import tqdm
import cv2
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from datetime import datetime as dt

## === cell 2
submission=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
train=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test=pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')

## === cell 3
train.head()

## === cell 4
train.loc[:,'healthy':'scab'].sum(axis=0)/(train.shape[0]/100)

## === cell 5
test.head()

## === cell 6
submission.head()

## === cell 7
train_img=[]
train_label=[]
path='/kaggle/input/plant-pathology-2020-fgvc7/images'
for im in tqdm(train['image_id']):
    im=im+".jpg"
    final_path=os.path.join(path,im)
    img=cv2.imread(final_path)
    img=cv2.resize(img,(224,224))
    img=img.astype('float32')
    train_img.append(img)

## === cell 8
test_img=[]
path='/kaggle/input/plant-pathology-2020-fgvc7/images'
for im in tqdm(test['image_id']):
    im=im+".jpg"
    final_path=os.path.join(path,im)
    img=cv2.imread(final_path)
    img=cv2.resize(img,(224,224))
    img=img.astype(('float32'))
    test_img.append(img)

## === cell 9
train_label=train.loc[:,'healthy':'scab']

## === cell 10
train_img=np.array(train_img)/255.0
test_img=np.array(test_img)/255.0
train_label=np.array(train_label)

## === cell 11
print(train_img.shape)
print(test_img.shape)
print(train_label.shape)

## === cell 12
from tensorflow.keras.preprocessing.image import ImageDataGenerator
datagen = ImageDataGenerator(
        featurewise_center=False,  # set input mean to 0 over the dataset
        samplewise_center=False,  # set each sample mean to 0
        featurewise_std_normalization=False,  # divide inputs by std of the dataset
        samplewise_std_normalization=False,  # divide each input by its std
        zca_whitening=False,  # apply ZCA whitening
        rotation_range=15,  # randomly rotate images in the range (degrees, 0 to 180)
        zoom_range = 0.25, # Randomly zoom image 
        width_shift_range=0.2,  # randomly shift images horizontally (fraction of total width)
        height_shift_range=0.2,  # randomly shift images vertically (fraction of total height)
        horizontal_flip=True,  # randomly flip images
        vertical_flip=False)  # randomly flip images


datagen.fit(train_img)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 14
from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.nasnet import NASNetLarge

from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

## === cell 15
base_model=DenseNet201(include_top=False, weights='imagenet',input_shape=(224,224,3), pooling='avg')

model=Sequential()
model.add(base_model)

model.add(BatchNormalization())
model.add(Dropout(0.8))
model.add(Dense(128,activation='relu'))
model.add(Dense(4,activation='softmax'))

froze = True # 13
if froze is True:
    base_model.trainable=False
else:
    for layer in base_model.layers[:-int(froze)]:
        layer.trainable = False

reduce_learning_rate = ReduceLROnPlateau(monitor='categorical_accuracy',
                                         factor=0.1,
                                         patience=2,
                                         cooldown=2,
                                         min_lr=0.0000001,
                                         verbose=1)
early_stopping = EarlyStopping(monitor='categorical_accuracy', patience=5)

check_point = ModelCheckpoint(filepath='resnet_50.h5', monitor='categorical_accuracy', save_best_only=True)

callbacks = [reduce_learning_rate, early_stopping, check_point]
    
model.compile( optimizer='adam',loss='categorical_crossentropy',metrics=['categorical_accuracy'])


## === cell 16
model.summary()

## === cell 17
start = dt.now()
history = model.fit_generator(datagen.flow(train_img, train_label, batch_size=32),
                    epochs=200,callbacks=callbacks)
print("Время работы модели: {}. Количество эпох: {}.".format(dt.now() - start, len(history.epoch)))

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2471624962.py in <cell line: 0>()
      1 # запускаем обучение
      2 start = dt.now()
----> 3 history = model.fit_generator(datagen.flow(train_img, train_label, batch_size=32),
      4                     epochs=200,callbacks=callbacks)
      5 print("Время работы модели: {}. Количество эпох: {}.".format(dt.now() - start, len(history.epoch)))

AttributeError: 'Sequential' object has no attribute 'fit_generator'

## === cell 18
import gc
del train_img, train_label

## === cell 19
%matplotlib inline
def plot_loss(his, title):
    epoch = len(his.epoch)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['loss'], label='train_loss')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Loss')
    plt.legend(loc='upper right')
    plt.show()

def plot_acc(his, title):
    epoch = len(his.epoch)
    plt.style.use('ggplot')
    plt.figure()
    plt.plot(np.arange(0, epoch), his.history['accuracy'], label='accuracy')

    plt.title(title)
    plt.xlabel('Epoch #')
    plt.ylabel('Accuracy')
    plt.legend(loc='upper right')
    plt.show()

## === cell 20
plot_loss(history,'Training Dataset')
plot_acc(history, 'Training Dataset')

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1335521900.py in <cell line: 0>()
      1 # строим графики
----> 2 plot_loss(history,'Training Dataset')
      3 plot_acc(history, 'Training Dataset')

NameError: name 'history' is not defined

## === cell 21
y_pred=model.predict(test_img)
print(y_pred)

## === cell 22
submission.loc[:,'healthy':'scab']=y_pred

## === cell 23
submission.to_csv('submission.csv',index=False)
