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
Given a dataset of images of dogs and cats, predict if an image is a dog or a cat.

## Metric
Log loss.

## Submission Format
For each image in the test set, you must submit a probability that image is a dog. The file should have a header and be in the following format:

```
id,label
1,0.5
2,0.5
3,0.5
...
```

## Dataset
The train folder contains 25,000 images of dogs and cats. Each image in this folder has the label as part of the filename. The test folder contains 12,500 images, named according to a numeric id.

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
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
        input/
            cat.1714.jpg (7.8 kB)
            cat.10025.jpg (18.4 kB)
            ... and 24998 other files
            description.md (50 lines)
            sample_submission.csv (2501 lines)
            sample_submission.csv.zip (6.0 kB)
            test.zip (56.6 MB)
            train.zip (513.0 MB)
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
            test/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                unknown/
                    900.jpg (42.3 kB)
                    572.jpg (30.6 kB)
                    ... and 2498 other files
            train/
                cat/
                    cat.4838.jpg (20.2 kB)
                    cat.1314.jpg (21.7 kB)
                    ... and 11240 other files
                dog/
                    dog.6712.jpg (35.3 kB)
                    dog.7152.jpg (36.1 kB)
                    ... and 11256 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
        working/
            dogs-vs-cats-redux-kernels-edition/
                cat.1714.jpg (7.8 kB)
                cat.10025.jpg (18.4 kB)
                ... and 24998 other files
                description.md (50 lines)
                sample_submission.csv (2501 lines)
                sample_submission.csv.zip (6.0 kB)
                test.zip (56.6 MB)
                train.zip (513.0 MB)
                dogs-vs-cats-redux-kernels-edition/
                test/
                    test/
                    unknown/
                        900.jpg (42.3 kB)
                        572.jpg (30.6 kB)
                        ... and 2498 other files
                train/
                    cat/
                        cat.4838.jpg (20.2 kB)
                        cat.1314.jpg (21.7 kB)
                        ... and 11240 other files
                    dog/
                        dog.6712.jpg (35.3 kB)
                        dog.7152.jpg (36.1 kB)
                        ... and 11256 other files
                    train/
```

-> data/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> input/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

-> working/dogs-vs-cats-redux-kernels-edition/sample_submission.csv has 2500 rows and 2 columns.
The columns are: id, label

# 5. Target score

16.01799

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
import glob
import matplotlib.pyplot as plt
import shutil
from tqdm import tqdm
import cv2

import os
import gc
import random
import re
print(os.listdir(".."))

from keras import backend
from keras.applications.inception_v3 import InceptionV3,preprocess_input
from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import SGD
from keras.models import Model, load_model
from keras.layers import Dense, GlobalAveragePooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.utils import to_categorical


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
train_dir = '../input/train'
test_dir = '../input/test'

test_imgs = ['../input/test/{}'.format(i) for i in os.listdir(test_dir)]

train_dogs = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'dog' in i]
train_cats = ['../input/train/{}'.format(i) for i in os.listdir(train_dir) if 'cat' in i]

train_imgs = train_dogs[:500]+train_cats[:500]
random.shuffle(train_imgs)

del train_dogs
del train_cats

gc.collect()


## === cell 2
Image_width,Image_height = 299,299
Number_FC_Neurons=1024
labels=['dog','cat']
num_classes = len(labels)


## === cell 3
def readAndProcessImg(image_list):
    X=[]
    y=[]
    
    for img in tqdm(image_list):
        X.append(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(Image_width,Image_height)))
        if 'dog' in img:
            y.append(1)
        elif 'cat' in img:
            y.append(0)
            
    return X,y


## === cell 4
X,y= readAndProcessImg(train_imgs)

del train_imgs
gc.collect()

X=np.array(X)
y=np.array(y)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/2950387840.py in <cell line: 0>()
----> 1 X,y= readAndProcessImg(train_imgs)
      2 
      3 del train_imgs
      4 gc.collect()
      5 

/tmp/ipykernel_11/2243465792.py in readAndProcessImg(image_list)
      6 
      7     for img in tqdm(image_list):
----> 8         X.append(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(Image_width,Image_height)))
      9         if 'dog' in img:
     10             y.append(1)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 5
print('Shape of train images: ',X.shape)
print('Shape of train label: ',y.shape)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3941526832.py in <cell line: 0>()
----> 1 print('Shape of train images: ',X.shape)
      2 print('Shape of train label: ',y.shape)

NameError: name 'X' is not defined

## === cell 6
def create_img_gen():
    return ImageDataGenerator(
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3
    )


## === cell 7
from sklearn.model_selection import train_test_split

X_train,X_val,y_train,y_val = train_test_split(X,y, test_size = 0.2, shuffle=True, stratify=y) 
y_train = to_categorical(y_train,num_classes=num_classes)
y_val = to_categorical(y_val,num_classes=num_classes)


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/659300422.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
      2 
----> 3 X_train,X_val,y_train,y_val = train_test_split(X,y, test_size = 0.2, shuffle=True, stratify=y)
      4 #Convert class vectors to binary class matrices using One-hot encoding
      5 y_train = to_categorical(y_train,num_classes=num_classes)

NameError: name 'X' is not defined

## === cell 8
print('Shape of train images: ',X_train.shape)
print('Shape of train label: ',y_train.shape)
print('Shape of validation images: ',X_val.shape)
print('Shape of validation label: ',y_val.shape)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/883352121.py in <cell line: 0>()
----> 1 print('Shape of train images: ',X_train.shape)
      2 print('Shape of train label: ',y_train.shape)
      3 print('Shape of validation images: ',X_val.shape)
      4 print('Shape of validation label: ',y_val.shape)

NameError: name 'X_train' is not defined

## === cell 9
n_train=len(X_train)
n_val=len(X_val)
print(n_train,n_val)
num_epoch = 2
batch_size = 50


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3497809594.py in <cell line: 0>()
----> 1 n_train=len(X_train)
      2 n_val=len(X_val)
      3 print(n_train,n_val)
      4 num_epoch = 2
      5 batch_size = 50

NameError: name 'X_train' is not defined

## === cell 10
train_image_gen = ImageDataGenerator(rescale=1/255,
        preprocessing_function=preprocess_input,
        rotation_range=30,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        validation_split=0.3
    )

val_image_gen = ImageDataGenerator(rescale=1/255)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2644512729.py in <cell line: 0>()
      1 # Define data pre-processing
      2 #   Define image generators for training and testing
----> 3 train_image_gen = ImageDataGenerator(rescale=1/255,
      4         preprocessing_function=preprocess_input,
      5         rotation_range=30,

NameError: name 'ImageDataGenerator' is not defined

## === cell 11
train_generator = train_image_gen.flow(X_train,y_train,batch_size=batch_size,seed=42,shuffle=True)
val_generator = val_image_gen.flow(X_val,y_val,batch_size=batch_size,seed=42,shuffle=True)


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1809710992.py in <cell line: 0>()
----> 1 train_generator = train_image_gen.flow(X_train,y_train,batch_size=batch_size,seed=42,shuffle=True)
      2 val_generator = val_image_gen.flow(X_val,y_val,batch_size=batch_size,seed=42,shuffle=True)

NameError: name 'train_image_gen' is not defined

## === cell 12

InceptionV3_base_model = InceptionV3(weights='imagenet', include_top=False)    #To exclude final conv layer 
print('Inception v3 base model without last FC loaded')


## === cell 13
x = InceptionV3_base_model.output
x_pool = GlobalAveragePooling2D()(x)
x_dense = Dense(Number_FC_Neurons,activation='relu')(x_pool)
final_pred = Dense(num_classes,activation='softmax')(x_dense)
model = Model(inputs=InceptionV3_base_model.input,outputs=final_pred)

model.summary()


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2911976257.py in <cell line: 0>()
      2 #Using Functional APIs
      3 x = InceptionV3_base_model.output
----> 4 x_pool = GlobalAveragePooling2D()(x)
      5 x_dense = Dense(Number_FC_Neurons,activation='relu')(x_pool)
      6 final_pred = Dense(num_classes,activation='softmax')(x_dense)

NameError: name 'GlobalAveragePooling2D' is not defined

## === cell 14
from keras.callbacks import EarlyStopping
my_callback=[EarlyStopping(monitor='val_loss',patience=5,mode=min,restore_best_weights=True)]


## === cell 15
print('Performing basic learning')

for layer in InceptionV3_base_model.layers:
    layer.trainable=False
    
model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3920648160.py in <cell line: 0>()
      7 
      8 #Define model compile for basic transfer learning
----> 9 model.compile(optimizer='adam',loss='binary_crossentropy',metrics=['accuracy'])

NameError: name 'model' is not defined

## === cell 16

history_transfer_learning = model.fit_generator(train_generator,epochs=12,
                                                steps_per_epoch=n_train//batch_size,
                                                validation_data=val_generator,
                                                validation_steps=n_val//batch_size,
                                                verbose=1,
                                                callbacks=my_callback,
                                                class_weight='auto')
model.save('model.hd5')


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2861691239.py in <cell line: 0>()
      4 # Please not the difference between fit() and fit_generator()
      5 
----> 6 history_transfer_learning = model.fit_generator(train_generator,epochs=12,
      7                                                 steps_per_epoch=n_train//batch_size,
      8                                                 validation_data=val_generator,

NameError: name 'model' is not defined

## === cell 18
gc.collect()


## === cell 19


score = model.evaluate_generator(val_generator,verbose=1)
print('Test loss: ', score[0])
print('Test accuracy', score[1])


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/252588800.py in <cell line: 0>()
      6 #y_pred[y_pred < 0.5]=0
      7 
----> 8 score = model.evaluate_generator(val_generator,verbose=1)
      9 print('Test loss: ', score[0])
     10 print('Test accuracy', score[1])

NameError: name 'model' is not defined

## === cell 20
epoch_list = list(range(1,len(history_transfer_learning.history['acc'])+1))  #Values for x axis[1,2,3,4...# of epochs]
plt.plot(epoch_list, history_transfer_learning.history['acc'],epoch_list,history_transfer_learning.history['val_acc'])
plt.legend(('Training accuracy','Validation Accuracy'))
plt.show()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/849194018.py in <cell line: 0>()
----> 1 epoch_list = list(range(1,len(history_transfer_learning.history['acc'])+1))  #Values for x axis[1,2,3,4...# of epochs]
      2 plt.plot(epoch_list, history_transfer_learning.history['acc'],epoch_list,history_transfer_learning.history['val_acc'])
      3 plt.legend(('Training accuracy','Validation Accuracy'))
      4 plt.show()

NameError: name 'history_transfer_learning' is not defined

## === cell 21
epoch_list = list(range(1,len(history_transfer_learning.history['loss'])+1))  #Values for x axis[1,2,3,4...# of epochs]
plt.plot(epoch_list, history_transfer_learning.history['loss'],epoch_list,history_transfer_learning.history['val_loss'])
plt.legend(('Training loss','Validation loss'))
plt.show()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3024211204.py in <cell line: 0>()
----> 1 epoch_list = list(range(1,len(history_transfer_learning.history['loss'])+1))  #Values for x axis[1,2,3,4...# of epochs]
      2 plt.plot(epoch_list, history_transfer_learning.history['loss'],epoch_list,history_transfer_learning.history['val_loss'])
      3 plt.legend(('Training loss','Validation loss'))
      4 plt.show()

NameError: name 'history_transfer_learning' is not defined

## === cell 22
X_test , y_test = readAndProcessImg(test_imgs[:10])
x=np.array(X_test)
test_datagen=ImageDataGenerator(rescale=1/255)  #rescale to reduce the dimension as 255 feature will become to heavy for the CPU to handle


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/2181527093.py in <cell line: 0>()
      1 #Lets predict and look at the test data
----> 2 X_test , y_test = readAndProcessImg(test_imgs[:10])
      3 x=np.array(X_test)
      4 test_datagen=ImageDataGenerator(rescale=1/255)  #rescale to reduce the dimension as 255 feature will become to heavy for the CPU to handle

/tmp/ipykernel_11/2243465792.py in readAndProcessImg(image_list)
      6 
      7     for img in tqdm(image_list):
----> 8         X.append(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(Image_width,Image_height)))
      9         if 'dog' in img:
     10             y.append(1)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 23
i=0
test_label=[]
columns=5
plt.figure(figsize=(30,20))
for img in test_datagen.flow(x,batch_size=1):
    pred=model.predict(img)
    label_pred = np.argmax(pred,axis=1)
    plt.subplot(5/columns+1,columns,i+1)
    if(label_pred > 0.5):
        test_label.append('dog')
    elif(label_pred < 0.5):
        test_label.append('cat')
    plt.title('This is a '+test_label[i])
    imgplot = plt.imshow(img[0])
    i+=1
    if i%10 == 0:
        break
    
plt.show()


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4004587503.py in <cell line: 0>()
      3 columns=5
      4 plt.figure(figsize=(30,20))
----> 5 for img in test_datagen.flow(x,batch_size=1):
      6     pred=model.predict(img)
      7     label_pred = np.argmax(pred,axis=1)

NameError: name 'test_datagen' is not defined

## === cell 24
X_test , y_test = readAndProcessImg(test_imgs)
x=np.array(X_test)
n_pred = len(X_test)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
error                                     Traceback (most recent call last)
/tmp/ipykernel_11/375883238.py in <cell line: 0>()
      1 #Lets predict the test data for submission
----> 2 X_test , y_test = readAndProcessImg(test_imgs)
      3 x=np.array(X_test)
      4 n_pred = len(X_test)

/tmp/ipykernel_11/2243465792.py in readAndProcessImg(image_list)
      6 
      7     for img in tqdm(image_list):
----> 8         X.append(cv2.resize(cv2.imread(img,cv2.IMREAD_COLOR),(Image_width,Image_height)))
      9         if 'dog' in img:
     10             y.append(1)

error: OpenCV(4.12.0) /io/opencv/modules/imgproc/src/resize.cpp:4208: error: (-215:Assertion failed) !ssize.empty() in function 'resize'


## === cell 25
y_pred = model.predict(x,batch_size=50,verbose=1)


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2357336256.py in <cell line: 0>()
----> 1 y_pred = model.predict(x,batch_size=50,verbose=1)

NameError: name 'model' is not defined

## === cell 26
final_pred_label = np.argmax(y_pred,axis=1)
submission = pd.DataFrame({'id':test_imgs[:], 'label':final_pred_label})


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1055967088.py in <cell line: 0>()
----> 1 final_pred_label = np.argmax(y_pred,axis=1)
      2 submission = pd.DataFrame({'id':test_imgs[:], 'label':final_pred_label})

NameError: name 'y_pred' is not defined

## === cell 27
submission['id']=[ re.findall('\d+',x)[0] for x in submission['id']]
submission.sort_values(ascending=True,by='id',inplace=True)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/183097700.py in <cell line: 0>()
----> 1 submission['id']=[ re.findall('\d+',x)[0] for x in submission['id']]
      2 submission.sort_values(ascending=True,by='id',inplace=True)

NameError: name 'submission' is not defined

## === cell 28
submission.head(10)


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3300208274.py in <cell line: 0>()
----> 1 submission.head(10)

NameError: name 'submission' is not defined

## === cell 29
submission.shape


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1635457632.py in <cell line: 0>()
----> 1 submission.shape

NameError: name 'submission' is not defined

## === cell 30
submission.to_csv('DogVsCats_submission.csv',index=False)


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4071899168.py in <cell line: 0>()
----> 1 submission.to_csv('DogVsCats_submission.csv',index=False)

NameError: name 'submission' is not defined
