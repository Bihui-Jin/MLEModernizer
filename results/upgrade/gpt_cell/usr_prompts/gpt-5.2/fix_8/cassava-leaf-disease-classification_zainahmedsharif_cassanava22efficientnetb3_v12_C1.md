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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
kt-legacy==1.0.5
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.6128

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    import google.protobuf  # noqa: F401
    from packaging.version import Version
    import google.protobuf as _pb

    if Version(_pb.__version__) >= Version("6.0.0"):
        import sys, subprocess

        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==5.28.3"]
        )
        import importlib

        importlib.invalidate_caches()
        importlib.reload(_pb)
except Exception:
    pass

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
import cv2
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, Dense, Dropout, MaxPool2D, Flatten

kt = None
RandomSearch = None
HyperModel = None
HyperParameters = None

from tensorflow.keras.callbacks import TensorBoard
import time

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
import matplotlib.pyplot as plt
for f in os.listdir('../input/cassava-leaf-disease-classification/train_images')[1000:1001]:
    img=cv2.imread(os.path.join('../input/cassava-leaf-disease-classification/train_images',f))
    plt.imshow(img)
    print(img.shape)
    plt.show        
    
epochs=10


## === cell 2
def create_df(training_images=100,image_directory='../input/cassava-leaf-disease-classification/train_images',test_dir='../input/cassava-leaf-disease-classification/test_images'):
    

    '''
    Returns dataframe given the training images and image directory  
     
    '''

    train=pd.read_csv('../input/cassava-leaf-disease-classification/train.csv') 
    label=list(train.iloc[:,1])
    label=[[x] for x in label]

    im_di=[]
    
    test_di=[]

    for f in os.listdir(image_directory)[:training_images]:
        im_di.append(os.path.join(image_directory,f))
        

    train_lis=list(zip(im_di,label))
    train_df=pd.DataFrame(train_lis)
    train_df=train_df.rename(columns={0:'x_col',1:'y_col'})
    
    for fi in os.listdir(test_dir):
        test_di.append(os.path.join(test_dir,fi))
        
        
    test_df=pd.DataFrame(test_di,columns=['x'])
    test_df['y']=None 
    
    
    return train_df,test_df


def hyper_model():  
    
    '''
    Hypermodel for keras tuner 
    
    '''


    model = tf.keras.applications.ResNet50V2(include_top=False,classes=5,input_shape=(128,128,3))

    for layers in model.layers[:]:                      ####NON_TRAIN_LAY
        layers.trainable=False

    flat=tf.keras.layers.Flatten()(model.output)
    
    
    
    out2=Dense(5,activation='softmax')(flat)

    f_mod=tf.keras.Model(inputs=model.input,outputs=out2)     
    
    f_mod.compile(loss='categorical_crossentropy',optimizer='adam',metrics=['accuracy'])
    
    return f_mod



def model1():
    
    model=Sequential()
    
    model.add(Conv2D(64,kernel_size=3,activation='relu',input_shape=(128,128,3),padding='same'))
        
    model.add(Conv2D(128,kernel_size=3,activation='relu',padding='same'))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(256,kernel_size=3,activation='relu',padding='same'))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(2,2)))
    
    model.add(Conv2D(512,kernel_size=3,activation='relu',padding='same'))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(3,3)))
            
    model.add(Conv2D(1024,kernel_size=3,activation='relu',padding='same'))
    
    model.add(MaxPool2D(pool_size=(2,2),strides=(3,3)))

    model.add(Flatten())
    
    model.add(Dense(units=512,activation='relu'))
    
    model.add(Dense(units=128,activation='relu'))

    model.add(Dense(units=5,activation='softmax'))
    
    model.compile(optimizer="Adam", loss="categorical_crossentropy", metrics=["accuracy"])

    return model 


def test_pre_process(dataframe,test_df,image_size=128,batch_size=64,valid_split=0.15):

    '''
    
    Data preprocessing based on image size, batch size and validation split using image data generator
    
    return train and validation image data generators 
    
    '''
    
    im_dg=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255,validation_split=valid_split)
    
    te_dg=tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    
    
    train=im_dg.flow_from_dataframe(dataframe,x_col="x_col",y_col="y_col",class_mode='categorical',batch_size=batch_size,target_size=(image_size,image_size),subset='training')
    
    valid=im_dg.flow_from_dataframe(dataframe,x_col="x_col",y_col="y_col",class_mode='categorical',batch_size=batch_size,target_size=(image_size,image_size),subset='validation')
    
    test =te_dg.flow_from_dataframe(test_df,x_col='x',y_col=None,batch_size=batch_size,target_size=(image_size,image_size),class_mode=None)

    return train,valid,test


## === cell 3
model=model1()
model.summary()


## === cell 4
train_df,test_df= create_df(training_images=90)

len(train_df)
test_df


## === cell 5
train,valid,test = test_pre_process(train_df,test_df=test_df)


## === cell 7
ear_sto=tf.keras.callbacks.EarlyStopping(monitor="val_loss",patience=3)
model=model
history=model.fit(x=train,epochs=epochs,verbose=2,callbacks=[ear_sto],validation_data=valid,shuffle=True)


## === cell 8
plt.plot(history.history['val_accuracy'])
plt.plot(history.history['accuracy'])
plt.title('model accuracy')
plt.xlabel('epochs')
plt.ylabel('accuracy')

plt.legend(['val_accuracy','train_accuracy'],loc='upper left')
plt.show()


## === cell 9
                  




pred=model.predict(x=test)
label=np.argmax(pred,axis=1)

image_id=test_df.iloc[:,0]
image_ids=[]
for path in image_id:
    path=path.split('/')[-1]
    image_ids.append(path)


## === cell 10
n_pred = len(label)
final_df = pd.DataFrame({"image_id": image_ids[:n_pred], "label": label})

final_df.to_csv("submission.csv", index=False)
