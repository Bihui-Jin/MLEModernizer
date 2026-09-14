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
Given a dataset of images of dogs, predict the breed of each image.

## Metric
Multi Class Log Loss.

## Submission Format
For each image in the test set, you must predict a probability for each of the different breeds. The file should contain a header and have the following format:
```
id,affenpinscher,afghan_hound,..,yorkshire_terrier
000621fb3cbb32d8935728e48679680e,0.0083,0.0,...,0.0083
etc.
```

## Dataset Description
- `train.zip` - the training set, you are provided the breed for these dogs
- `test.zip` - the test set, you must predict the probability of each breed for each image
- `sample_submission.csv` - a sample submission file in the correct format
- `labels.csv` - the breeds for the images in the train set

# 2. Python version

3.7

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
        input/
            description.md (169 lines)
            labels.csv (9200 lines)
            labels.csv.zip (201.6 kB)
            sample_submission.csv (1024 lines)
            sample_submission.csv.zip (32.1 kB)
            test.zip (36.4 MB)
            train.zip (324.7 MB)
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
            test/
                bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                ... and 1021 other files
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
            train/
                868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                ... and 9197 other files
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
        working/
            dog-breed-identification/
                description.md (169 lines)
                labels.csv (9200 lines)
                ... and 5 other files
                dog-breed-identification/
                test/
                    bca88d42e4fc84b3169b13a615f5fdbf.jpg (38.6 kB)
                    53cb3ed2547cdaf15ec7983d8325f007.jpg (32.9 kB)
                    ... and 1021 other files
                    test/
                train/
                    868decd906bb483bac17a005a3f06bc3.jpg (22.6 kB)
                    7b44341b91b48e2eafe679c00ba1a0a6.jpg (48.4 kB)
                    ... and 9197 other files
                    train/
```

-> data/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> data/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> data/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> input/dog-breed-identification/labels.csv has 9199 rows and 2 columns.
The columns are: id, breed

-> input/dog-breed-identification/sample_submission.csv has 1023 rows and 121 columns.
The columns are: id, affenpinscher, afghan_hound, african_hunting_dog, airedale, american_staffordshire_terrier, appenzeller, australian_terrier, basenji, basset, beagle, bedlington_terrier, bernese_mountain_dog, black-and-tan_coonhound, blenheim_spaniel... and 106 more columns

-> (stopped after 10 files for performance)

# 5. Target score

4.88533

# 6. Current score

Not yielded

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import os
import argparse

import time
import datetime

import random
import numpy as np 
import pandas as pd

import shutil
from glob import glob
from tqdm import tqdm
import gc

import re

from sklearn import preprocessing
from sklearn.utils import class_weight as cw
from sklearn.utils import shuffle

from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score

from sklearn import svm
from sklearn.svm import LinearSVC, SVC

from sklearn import metrics 
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score, confusion_matrix, classification_report, roc_auc_score


from keras.preprocessing.image import ImageDataGenerator

import keras
from keras import models
from keras.models import Model
from keras.models import Sequential

from keras.layers import Convolution1D, concatenate, SpatialDropout1D, GlobalMaxPool1D, GlobalAvgPool1D, Embedding, \
    Conv2D, SeparableConv1D, Add, BatchNormalization, Activation, GlobalAveragePooling2D, LeakyReLU, Flatten
from keras.layers import Dense, Input, Dropout, MaxPool2D, MaxPooling2D, Concatenate, GlobalMaxPooling2D, GlobalAveragePooling2D, \
    Lambda, Multiply, LSTM, Bidirectional, PReLU, MaxPooling1D
from keras.layers.pooling import _GlobalPooling1D

from keras.regularizers import l2

from keras.applications.xception import Xception
from keras.applications.resnet50 import ResNet50
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.densenet import DenseNet201
from keras.applications.nasnet import NASNetMobile, NASNetLarge

from keras.applications.nasnet import preprocess_input

from keras import optimizers
from keras.optimizers import Adam, SGD , RMSprop
from keras.losses import mae, sparse_categorical_crossentropy, binary_crossentropy

from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard, ReduceLROnPlateau, LearningRateScheduler
from keras import backend as K

import tensorflow as tf

import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import matplotlib.image as mpimg
import seaborn as sns
from mlxtend.plotting import plot_confusion_matrix

import cv2
from PIL import Image
from IPython.display import display


%matplotlib inline

print(os.listdir("../input/"))


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
def date_time(x):
    if x==1:
        return 'Timestamp: {:%Y-%m-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==2:    
        return 'Timestamp: {:%Y-%b-%d %H:%M:%S}'.format(datetime.datetime.now())
    if x==3:  
        return 'Date now: %s' % datetime.datetime.now()
    if x==4:  
        return 'Date today: %s' % datetime.date.today()  


## === cell 2
input_directory = r"../input/dog-breed-identification/"
output_directory = r"../output/"

training_dir = input_directory + "train"
testing_dir = input_directory + "test"

if not os.path.exists(output_directory):
    os.mkdir(output_directory)
    
figure_directory = "../output/figures"
if not os.path.exists(figure_directory):
    os.mkdir(figure_directory)    
    
file_name_pred_batch = figure_directory+r"/result"
file_name_pred_sample = figure_directory+r"/sample"


## === cell 3
train_df = pd.read_csv(input_directory + "/labels.csv")
train_df.rename(columns={"breed": "label"}, inplace=True)
train_df["id"] = train_df["id"].apply(lambda x: x+"."+"jpg")
train_df.head()


## === cell 4
classes = list(train_df["label"].unique())
classes.sort()


## === cell 5
test_files = os.listdir(testing_dir)
    
test_df = pd.DataFrame({"id": test_files, "label": "boston_bull"})

test_df.head()


## === cell 6
len(train_df), len(test_df)


## === cell 7
def plot_image(file, directory=None, sub=False, aspect=None, title=False):
    path = directory + "/" + file
    
    img = plt.imread(path)
    
    plt.imshow(img, aspect=aspect)
    if title:
        plt.title(file)
    plt.xticks([])
    plt.yticks([])
    
    if sub:
        plt.show()
        
def plot_img_dir(directory=training_dir, count=5):
    selected_files = random.sample(os.listdir(directory), count)
    
    ncols = 5
    nrows = count//ncols if count%ncols==0 else count//ncols+1
    
    figsize=(20, ncols*nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    
    for file in selected_files:        
        plt.subplot(nrows, ncols, i+1)
        path = directory + file
        plot_image(file, directory, aspect=None)

        i=i+1
    
    plt.tight_layout()
    plt.show()
    
def plot_img_df(directory=None, df=None, filename="id", label = "label", count=5, num_cat=-1):
    label_map = {}
    
    classes = list(set(df[label]))
    
    for l in classes:
        label_map[l] = df[df[label]==l][filename]
        label_map[l] = label_map[l].sample(count, replace=True)
        
    
    
    ncols = 5
    nrows = count//ncols if count%ncols==0 else count//ncols+1
    
    figsize=(20, ncols*nrows)

    ticksize = 14
    titlesize = ticksize + 8
    labelsize = ticksize + 5


    params = {'figure.figsize' : figsize,
              'axes.labelsize' : labelsize,
              'axes.titlesize' : titlesize,
              'xtick.labelsize': ticksize,
              'ytick.labelsize': ticksize}

    plt.rcParams.update(params)
    
    i=0
    if num_cat==-1:
        print("Showing {} classes...".format(len(label_map)))
    else:
        print("Showing {} classes...".format(num_cat))
        
    for label in label_map:
        if num_cat==i:
            break
        label2 = re.sub("_", " ", label)
        label2 = label2.title()
        print(str(i+1) + ". " + label2)
        
        j=0
        for id, file in label_map[label].iteritems():
            plt.subplot(nrows, 5, j+1)
            plot_image(file, directory, aspect='auto')
            j=j+1
            
        plt.tight_layout()
        plt.show()
        
        
        i+=1
        
def plot_img_dir_main(directory=training_dir, count=5):
    labels = os.listdir(directory)
    
    for label in labels:
        label2 = re.sub("_", " ", label)
        label2 = label2.title()
        print(label2)
        plot_img_dir(directory=directory+"/"+label, count=count)
        num_cat-=1
        


## === cell 8
plot_img_df(directory=training_dir, df=train_df, filename="id", label = "label", count=5, num_cat=5)


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2139958037.py in <cell line: 0>()
----> 1 plot_img_df(directory=training_dir, df=train_df, filename="id", label = "label", count=5, num_cat=5)

/tmp/ipykernel_11/1050109912.py in plot_img_df(directory, df, filename, label, count, num_cat)
     75               'ytick.labelsize': ticksize}
     76 
---> 77     plt.rcParams.update(params)
     78 
     79     i=0

NameError: name 'plt' is not defined

## === cell 9
train_df2 = train_df.copy()
train_df2["label"] = train_df["label"].apply(lambda x: re.sub("_", " ", x))
train_df2["label"] = train_df2["label"].apply(lambda x: x.title())

classes2 = train_df2["label"].unique()
classes2.sort()

rows = (len(classes) - 1)/4


## === cell 10
plt.figure(figsize=(18, rows))
ax = sns.countplot(y="label", data=train_df2, order=train_df2["label"].value_counts().index)
plt.title("Countplot sorted by Value count of Categories")
plt.tight_layout()
plt.show() 


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/582579586.py in <cell line: 0>()
----> 1 plt.figure(figsize=(18, rows))
      2 ax = sns.countplot(y="label", data=train_df2, order=train_df2["label"].value_counts().index)
      3 plt.title("Countplot sorted by Value count of Categories")
      4 plt.tight_layout()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 11
plt.figure(figsize=(18, rows))
ax = sns.countplot(y="label", data=train_df2, order=classes2)
plt.title("Countplot Sorted by Alphebetical Order of Category Names")
plt.tight_layout()
plt.show() 


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2422742234.py in <cell line: 0>()
----> 1 plt.figure(figsize=(18, rows))
      2 ax = sns.countplot(y="label", data=train_df2, order=classes2)
      3 plt.title("Countplot Sorted by Alphebetical Order of Category Names")
      4 plt.tight_layout()
      5 plt.show()

NameError: name 'plt' is not defined

## === cell 12
def get_data(batch_size=32, target_size=(299, 299), class_mode="categorical", training_dir=training_dir, testing_dir=testing_dir, x_col="id", y_col="label"):
    print("Preprocessing and Generating Data Batches.......\n")
    
    rescale = 1.0/255

    train_batch_size = batch_size
    validation_batch_size = batch_size*5
    test_batch_size = batch_size*5
    
    train_shuffle = True
    val_shuffle = True
    test_shuffle = False
    
    train_datagen = ImageDataGenerator(
        horizontal_flip=True,
        rotation_range=45,
        shear_range=15,
        rescale=rescale,
        validation_split=0.25)

    train_generator = train_datagen.flow_from_dataframe(
        train_df, 
        training_dir,
        x_col=x_col,
        y_col=y_col,  
        target_size=target_size, 
        class_mode=class_mode, 
        batch_size=batch_size, 
        shuffle=True, 
        seed=42,
        subset='training')
    
    
    validation_generator = train_datagen.flow_from_dataframe(
        train_df, 
        training_dir,
        x_col=x_col,
        y_col=y_col,  
        target_size=target_size, 
        class_mode=class_mode, 
        batch_size=validation_batch_size, 
        shuffle=True, 
        seed=42,
        subset='validation')
    
    test_datagen = ImageDataGenerator(rescale=rescale)
    
    test_generator = test_datagen.flow_from_dataframe(
        test_df, 
        testing_dir,
        x_col=x_col,
        y_col=y_col,  
        target_size=target_size, 
        class_mode=class_mode, 
        batch_size=test_batch_size, 
        shuffle=False, 
        seed=42)
    
    class_weights = get_weight(train_generator.classes, binary=False)
    
    steps_per_epoch = len(train_generator)
    validation_steps = len(validation_generator)
    
    print("\nPreprocessing and Data Batch Generation Completed.\n")
    
    
    return train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps

def get_weight(y, binary=True, n_samples=-1):
    if binary==False:
        class_weights =  cw.compute_class_weight('balanced', np.unique(y), y)
    else:
        d = {x:y.count(x) for x in set(y)}
        num_class_temp = 2
        class_weights = {}

        for cls in d:
            count = d.get(cls)
            class_weight = n_samples / (num_class_temp * count)
            class_weights[cls] = class_weight
    return class_weights


## === cell 13
def get_model(model_name, input_shape=(96, 96, 3), num_class=2, weights='imagenet', dense_units=1024):
    inputs = Input(input_shape)
    
    if model_name == "Xception":
        base_model = Xception(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet50":
        base_model = ResNet50(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet101":
        base_model = keras.applications.resnet.ResNet101(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet152":
        base_model = keras.applications.resnet.ResNet152(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet50V2":
        base_model = resnet_v2.ResNet50V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet101V2":
        base_model = resnet_v2.ResNet101V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNet152V2":
        base_model = resnet_v2.ResNet152V2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNeXt50":
        base_model = resnext.ResNeXt50(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "ResNeXt101":
        base_model = resnext.ResNeXt101(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "InceptionV3":
        base_model = InceptionV3(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "InceptionResNetV2":
        base_model = InceptionResNetV2(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "DenseNet201":
        base_model = DenseNet201(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "NASNetMobile":
        base_model = NASNetMobile(include_top=False, weights=weights, input_shape=input_shape)
    elif model_name == "NASNetLarge":
        base_model = NASNetLarge(include_top=False, weights=weights, input_shape=input_shape)
        

    x = base_model(inputs)
    x = Dropout(0.8)(x) 
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dropout(0.8)(x)     

    
    if num_class>1:
        outputs = Dense(num_class, activation="softmax")(x)
    else:
        outputs = Dense(1, activation="sigmoid")(x)

    model = Model(inputs=inputs, outputs=outputs)

    model.summary()
    
    
    return model


## === cell 14
def get_conv_model(num_class=2, input_shape=(150,150, 3)):
    model = Sequential()

    model.add(Conv2D(32, (3, 3), input_shape=(32, 32, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(32, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(64, (3, 3), padding='same', use_bias=False, kernel_regularizer=l2(1e-4)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPool2D())
    model.add(Dropout(0.2))

    model.add(GlobalAveragePooling2D())
    
    if num_class>1:
        model.add(Dense(num_class, activation='softmax'))
    else:
        model.add(Dense(num_class, activation='sigmoid'))
    

    return model


## === cell 15
main_model_dir = output_directory + r"models/"
main_log_dir = output_directory + r"logs/"

try:
    os.mkdir(main_model_dir)
except:
    print("Could not create main model directory")
    
try:
    os.mkdir(main_log_dir)
except:
    print("Could not create main log directory")



model_dir = main_model_dir + time.strftime('%Y-%m-%d %H-%M-%S') + "/"
log_dir = main_log_dir + time.strftime('%Y-%m-%d %H-%M-%S')


try:
    os.mkdir(model_dir)
except:
    print("Could not create model directory")
    
try:
    os.mkdir(log_dir)
except:
    print("Could not create log directory")
    
model_file = model_dir + "{epoch:02d}-val_acc-{val_acc:.2f}-val_loss-{val_loss:.2f}.hdf5"


## === cell 16
print("Settting Callbacks")

def step_decay(epoch, lr):
    lrate = lr
    if epoch==2:
        lrate = 0.0001  
    return lrate


checkpoint = ModelCheckpoint(
    model_file, 
    monitor='val_acc', 
    save_best_only=True)


early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=3,
    verbose=1,
    restore_best_weights=True)


reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=1,
    min_lr=0.0000001,
    verbose=1)


learning_rate_scheduler = LearningRateScheduler(step_decay, verbose=1)



callbacks = [reduce_lr, early_stopping]


print("Set Callbacks at ", date_time(1))


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2854019017.py in <cell line: 0>()
     10 
     11 
---> 12 checkpoint = ModelCheckpoint(
     13     model_file,
     14     monitor='val_acc',

NameError: name 'ModelCheckpoint' is not defined

## === cell 17
dim = 299

input_shape = (dim, dim, 3)

num_class = len(classes)

weights = 'imagenet'
dense_units = 256


## === cell 18
print("Getting Base Model", date_time(1))

model = get_model(model_name="InceptionV3", input_shape=input_shape, num_class=num_class, weights=weights, dense_units=dense_units)
print("Loaded Base Model", date_time(1))


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3783401498.py in <cell line: 0>()
      1 print("Getting Base Model", date_time(1))
      2 
----> 3 model = get_model(model_name="InceptionV3", input_shape=input_shape, num_class=num_class, weights=weights, dense_units=dense_units)
      4 # model = get_conv_model(num_class=num_class, input_shape=input_shape)
      5 print("Loaded Base Model", date_time(1))

/tmp/ipykernel_11/2181131655.py in get_model(model_name, input_shape, num_class, weights, dense_units)
      1 def get_model(model_name, input_shape=(96, 96, 3), num_class=2, weights='imagenet', dense_units=1024):
----> 2     inputs = Input(input_shape)
      3 
      4     if model_name == "Xception":
      5         base_model = Xception(include_top=False, weights=weights, input_shape=input_shape)

NameError: name 'Input' is not defined

## === cell 19
batch_size = 64

class_mode = "categorical"

target_size = (dim, dim)
y_col = "label"


## === cell 20
train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size,
                                                                                                                   target_size=target_size,
                                                                                                                   class_mode=class_mode,
                                                                                                                   y_col=y_col)    


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2292636779.py in <cell line: 0>()
----> 1 train_generator, validation_generator, test_generator, class_weights, steps_per_epoch, validation_steps = get_data(batch_size=batch_size,
      2                                                                                                                    target_size=target_size,
      3                                                                                                                    class_mode=class_mode,
      4                                                                                                                    y_col=y_col)    

/tmp/ipykernel_11/3920610513.py in get_data(batch_size, target_size, class_mode, training_dir, testing_dir, x_col, y_col)
     12     test_shuffle = False
     13 
---> 14     train_datagen = ImageDataGenerator(
     15         horizontal_flip=True,
     16 #         vertical_flip=True,

NameError: name 'ImageDataGenerator' is not defined

## === cell 21
print("Compliling Model ...")


learning_rate = 0.0001
optimizer = Adam(learning_rate)


loss = 'categorical_crossentropy'
metrics = ['accuracy']


model.compile(optimizer=optimizer, loss=loss, metrics=metrics)

print("Completed Model Compilation.\n")


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1427827282.py in <cell line: 0>()
      3 
      4 learning_rate = 0.0001
----> 5 optimizer = Adam(learning_rate)
      6 # optimizer = Adam()
      7 

NameError: name 'Adam' is not defined

## === cell 22
steps_per_epoch = len(train_generator)
validation_steps = len(validation_generator)

verbose = 1
epochs = 1


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1687695208.py in <cell line: 0>()
----> 1 steps_per_epoch = len(train_generator)
      2 validation_steps = len(validation_generator)
      3 
      4 verbose = 1
      5 epochs = 1

NameError: name 'train_generator' is not defined

## === cell 25
def plot_performance(history=None, figure_directory=None):
    xlabel = 'Epoch'
    legends = ['Training', 'Validation']

    ylim_pad = [0, 0]


    plt.figure(figsize=(20, 5))


    y1 = history.history['acc']
    y2 = history.history['val_acc']

    min_y = min(min(y1), min(y2))-ylim_pad[0]
    max_y = max(max(y1), max(y2))+ylim_pad[0]
    


    plt.subplot(121)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Accuracy\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Accuracy', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()



    y1 = history.history['loss']
    y2 = history.history['val_loss']

    min_y = min(min(y1), min(y2))-ylim_pad[1]
    max_y = max(max(y1), max(y2))+ylim_pad[1]


    plt.subplot(122)

    plt.plot(y1)
    plt.plot(y2)

    plt.title('Model Loss\n'+date_time(1), fontsize=17)
    plt.xlabel(xlabel, fontsize=15)
    plt.ylabel('Loss', fontsize=15)
    plt.ylim(min_y, max_y)
    plt.legend(legends, loc='upper left')
    plt.grid()
    if figure_directory:
        plt.savefig(figure_directory+"/history")

    plt.show()


## === cell 28
label_map = (train_generator.class_indices)
label_map_inv = {v: k for k, v in label_map.items()}


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2203689430.py in <cell line: 0>()
----> 1 label_map = (train_generator.class_indices)
      2 label_map_inv = {v: k for k, v in label_map.items()}

NameError: name 'train_generator' is not defined

## === cell 29
ypreds = model.predict_generator(generator=test_generator, steps = len(test_generator),  verbose=1)


## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/520296762.py in <cell line: 0>()
----> 1 ypreds = model.predict_generator(generator=test_generator, steps = len(test_generator),  verbose=1)

NameError: name 'model' is not defined

## === cell 30
ypreds


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/73110516.py in <cell line: 0>()
----> 1 ypreds

NameError: name 'ypreds' is not defined

## === cell 31
ypred = ypreds.argmax(axis=-1)
ypred


## --- ERROR in cell 31, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2982563054.py in <cell line: 0>()
----> 1 ypred = ypreds.argmax(axis=-1)
      2 ypred

NameError: name 'ypreds' is not defined

## === cell 32
def get_rand_test_img(test_generator=None, labels_test=None, count=5):

    filepaths = test_generator.filepaths
    file_names = test_generator.filenames

    selected_filepaths = []
    selected_file_names = []
    selected_labels = []

    mem = set()

    i = count

    num_test_sample = test_generator.n

    for file in test_generator.labels:
        if i<=0:
            break

        rnd = random.randint(0, test_generator.n)
        while rnd in mem:
            rnd = random.randint(0, test_generator.n)

        selected_filepaths.append(filepaths[rnd])
        selected_file_names.append(file_names[rnd])
        
        lbl = label_map_inv[labels_test[rnd]]
        lbl = re.sub("_", " ", lbl)
        lbl = lbl.title()

        selected_labels.append(lbl)
        
        i-=1
    
    return selected_file_names, selected_file_names, selected_labels


## === cell 33
test_img_count = 10
labels_test = ypred
selected_file_names, selected_file_names, selected_labels = get_rand_test_img(test_generator=test_generator, labels_test=labels_test, count=test_img_count)


## --- ERROR in cell 33, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1911339386.py in <cell line: 0>()
      1 test_img_count = 10
----> 2 labels_test = ypred
      3 selected_file_names, selected_file_names, selected_labels = get_rand_test_img(test_generator=test_generator, labels_test=labels_test, count=test_img_count)

NameError: name 'ypred' is not defined

## === cell 34
count = test_img_count

ncols = 5
nrows = count//ncols if count%ncols==0 else count//ncols+1

figsize=(20, ncols*nrows)

ticksize = 14
titlesize = ticksize + 8
labelsize = ticksize + 5


params = {'figure.figsize' : figsize,
          'axes.labelsize' : labelsize,
          'axes.titlesize' : titlesize,
          'xtick.labelsize': ticksize,
          'ytick.labelsize': ticksize}

plt.rcParams.update(params)


i = 0

for i in range(0, count):
    if i>test_img_count:
        break    
    plt.subplot(nrows, 5, i+1)
    plot_image(selected_file_names[i], testing_dir, aspect='auto')
    plt.title(selected_labels[i])

plt.tight_layout()
plt.show()     


## --- ERROR in cell 34, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3910682853.py in <cell line: 0>()
     17           'ytick.labelsize': ticksize}
     18 
---> 19 plt.rcParams.update(params)
     20 
     21 

NameError: name 'plt' is not defined

## === cell 36
sample_submission = pd.read_csv(input_directory+"sample_submission.csv")
sample_submission.head()


## === cell 37
test_dir_files = os.listdir(testing_dir)
test_gen_files = test_generator.filenames
sample_submission_files = sample_submission["id"]
len(test_dir_files), len(test_gen_files), len(sample_submission_files)


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2136419228.py in <cell line: 0>()
      1 test_dir_files = os.listdir(testing_dir)
----> 2 test_gen_files = test_generator.filenames
      3 sample_submission_files = sample_submission["id"]
      4 len(test_dir_files), len(test_gen_files), len(sample_submission_files)

NameError: name 'test_generator' is not defined

## === cell 38
m = {}
l = len(test_gen_files)
for i in range(l):
    m[test_gen_files[i]] = ypreds[i] 


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1713776496.py in <cell line: 0>()
      1 m = {}
----> 2 l = len(test_gen_files)
      3 for i in range(l):
      4     m[test_gen_files[i]] = ypreds[i]

NameError: name 'test_gen_files' is not defined

## === cell 39
labels = (train_generator.class_indices)
labels = list(labels.keys())


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1653532271.py in <cell line: 0>()
----> 1 labels = (train_generator.class_indices)
      2 labels = list(labels.keys())

NameError: name 'train_generator' is not defined

## === cell 40
ypreds_sync = []
for f in sample_submission_files:
    ypreds_sync.append(m[sample_submission_files[i]+".jpg"]) 


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4184806839.py in <cell line: 0>()
      1 ypreds_sync = []
----> 2 for f in sample_submission_files:
      3     ypreds_sync.append(m[sample_submission_files[i]+".jpg"])

NameError: name 'sample_submission_files' is not defined

## === cell 41
test_df = pd.DataFrame(data=ypreds_sync, columns=labels)
test_df.head()


## --- ERROR in cell 41, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2120027895.py in <cell line: 0>()
----> 1 test_df = pd.DataFrame(data=ypreds_sync, columns=labels)
      2 test_df.head()

NameError: name 'labels' is not defined

## === cell 42
test_df["id"]  = sample_submission_files

cols = test_df.columns.tolist()
cols = cols[-1:] + cols[:-1]
test_df = test_df[cols]

test_df.to_csv('submission.csv', index=False)
test_df.head()


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1653131549.py in <cell line: 0>()
----> 1 test_df["id"]  = sample_submission_files
      2 
      3 cols = test_df.columns.tolist()
      4 cols = cols[-1:] + cols[:-1]
      5 test_df = test_df[cols]

NameError: name 'sample_submission_files' is not defined
