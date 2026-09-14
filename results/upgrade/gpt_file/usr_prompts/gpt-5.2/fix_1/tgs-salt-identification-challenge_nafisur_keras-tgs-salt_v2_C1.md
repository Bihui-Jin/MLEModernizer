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
Segment regions of salt in seismic images.

## Metric
Mean average precision at different intersection over union (IoU) thresholds. The IoU of a proposed set of object pixels and a set of true object pixels is calculated as:

$$\text{IoU}(A, B)=\frac{A \cap B}{A \cup B}$$

The metric sweeps over a range of IoU thresholds, at each point calculating an average precision value. The threshold values range from 0.5 to 0.95 with a step size of 0.05: `(0.5, 0.55, 0.6, 0.65, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95)`. In other words, at a threshold of 0.5, a predicted object is considered a "hit" if its intersection over union with a ground truth object is greater than 0.5.

At each threshold value 𝑡t, a precision value is calculated based on the number of true positives (TP), false negatives (FN), and false positives (FP) resulting from comparing the predicted object to all ground truth objects:

$$\frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

A true positive is counted when a single predicted object matches a ground truth object with an IoU above the threshold. A false positive indicates a predicted object had no associated ground truth object. A false negative indicates a ground truth object had no associated predicted object. The average precision of a single image is then calculated as the mean of the above precision values at each IoU threshold:

$$\frac{1}{\mid \text { thresholds } \mid} \sum_t \frac{T P(t)}{T P(t)+F P(t)+F N(t)}$$

## Submission Format
Use run-length encoding on the pixel values. Instead of submitting an exhaustive list of indices for your segmentation, you will submit pairs of values that contain a start position and a run length. E.g. '1 3' implies starting at pixel 1 and running a total of 3 pixels (1,2,3).

The competition format requires a space delimited list of pairs. For example, '1 3 10 5' implies pixels 1,2,3,10,11,12,13,14 are to be included in the mask. The pixels are one-indexed\
and numbered from top to bottom, then left to right: 1 is pixel (1,1), 2 is pixel (2,1), etc.

The metric checks that the pairs are sorted, positive, and the decoded pixel values are not duplicated. It also checks that no two predicted masks for the same image are overlapping.

The file should contain a header and have the following format. Each row in your submission represents a single predicted salt segmentation for the given image.

```
id,rle_mask
3e06571ef3,1 1
a51b08d882,1 1
c32590b06f,1 1
etc.
```

## Dataset
The data is a set of images chosen at various locations chosen at random in the subsurface. The images are 101 x 101 pixels and each pixel is classified as either salt or sediment. In addition to the seismic images, the depth of the imaged location is provided for each image.

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
pillow==11.3.0
protobuf==6.33.0
scikit-image==0.25.2
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
tf_keras==2.18.0
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
        input/
            depths.csv (3001 lines)
            depths.csv.zip (25.0 kB)
            description.md (83 lines)
            sample_submission.csv (1001 lines)
            sample_submission.csv.zip (6.8 kB)
            test.zip (9.8 MB)
            train.csv (3001 lines)
            train.csv.zip (293.9 kB)
            train.zip (30.9 MB)
            test/
                images/
                    a05ae39815.png (10.5 kB)
                    9bf8a38b92.png (10.7 kB)
                    ... and 998 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
            train/
                images/
                    66cf41c563.png (7.7 kB)
                    429bf7c665.png (8.2 kB)
                    ... and 2998 other files
                masks/
                    6eeeda7f4a.png (230 Bytes)
                    5d600057f5.png (230 Bytes)
                    ... and 2998 other files
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
        working/
            tgs-salt-identification-challenge/
                depths.csv (3001 lines)
                depths.csv.zip (25.0 kB)
                ... and 7 other files
                test/
                    images/
                        a05ae39815.png (10.5 kB)
                        9bf8a38b92.png (10.7 kB)
                        ... and 998 other files
                    test/
                tgs-salt-identification-challenge/
                train/
                    images/
                        66cf41c563.png (7.7 kB)
                        429bf7c665.png (8.2 kB)
                        ... and 2998 other files
                    masks/
                        6eeeda7f4a.png (230 Bytes)
                        5d600057f5.png (230 Bytes)
                        ... and 2998 other files
                    train/
```

-> data/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> data/tgs-salt-identification-challenge/sample_submission.csv has 1000 rows and 2 columns.
The columns are: id, rle_mask

-> data/tgs-salt-identification-challenge/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> data/train.csv has 3000 rows and 2 columns.
The columns are: id, rle_mask

-> input/depths.csv has 3000 rows and 2 columns.
The columns are: id, z

-> (stopped after 10 files for performance)

# 5. Target score

0.42183

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix,f1_score
from skimage.transform import resize
import tqdm
from tqdm import tqdm_notebook,tnrange
from keras.layers import Input,BatchNormalization,Dense,Dropout,Flatten,MaxPooling2D,GlobalAveragePooling2D,Conv2D,UpSampling2D,Conv2DTranspose,MaxPool2D
from keras.models import Model,load_model,Sequential
from keras.losses import binary_crossentropy,categorical_crossentropy
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,ModelCheckpoint,ReduceLROnPlateau
from keras.utils import to_categorical
from keras.applications.densenet import DenseNet121,preprocess_input as densenet_preprocessing
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input as inception_preprocessing
from keras.applications.resnet50 import ResNet50,preprocess_input as resnet_preprocessing
from keras.applications.xception import Xception,preprocess_input as xception_preprocessing
from keras.applications.inception_resnet_v2 import InceptionResNetV2,preprocess_input as inceptionresnet_prepross
from keras.preprocessing.image import ImageDataGenerator,array_to_img,load_img,img_to_array
from keras.layers.merge import concatenate
from keras.layers.core import Lambda
from keras import backend as K
import tensorflow as tf
import os
import sys
import seaborn as sn
from PIL import Image
pd.set_option('display.max_colwidth',100)
%matplotlib inline
import gc
gc.collect()


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
Train_Image_folder='../input/train/images/'
Train_Mask_folder='../input/train/masks/'
Test_Image_folder='../input/test/images/'
Train_Image_name=os.listdir(path=Train_Image_folder)
Test_Image_name=os.listdir(path=Test_Image_folder)
Train_Image_path=[]
Train_Mask_path=[]
Train_id=[]
for i in Train_Image_name:
    path1=Train_Image_folder+i
    path2=Train_Mask_folder+i
    id1=i.split(sep='.')[0]
    Train_Image_path.append(path1)
    Train_Mask_path.append(path2)
    Train_id.append(id1)
  

Test_Image_path=[]
Test_id=[]
for i in Test_Image_name:
    path=Test_Image_folder+i
    id2=i.split(sep='.')[0]
    Test_Image_path.append(path)
    Test_id.append(id2)
    
df_Train_path=pd.DataFrame({'id':Train_id,'Train_Image_path':Train_Image_path,'Train_Mask_path':Train_Mask_path})
df_Test_path=pd.DataFrame({'id':Test_id,'Test_Image_path':Test_Image_path})

df_depths=pd.read_csv('../input/depths.csv')
df_sub=pd.read_csv('../input/sample_submission.csv')
df_Train_path=df_Train_path.merge(df_depths,on='id',how='left')
df_Test_path=df_Test_path.merge(df_depths,on='id',how='left')
df_Test_path=df_sub.merge(df_Test_path,on='id',how='left')
print(df_Train_path.shape,df_Test_path.shape)
df_Train_path.head()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1297717060.py in <cell line: 0>()
      2 Train_Mask_folder='../input/train/masks/'
      3 Test_Image_folder='../input/test/images/'
----> 4 Train_Image_name=os.listdir(path=Train_Image_folder)
      5 Test_Image_name=os.listdir(path=Test_Image_folder)
      6 Train_Image_path=[]

NameError: name 'os' is not defined

## === cell 2
df_Test_path.head()


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/485998125.py in <cell line: 0>()
----> 1 df_Test_path.head()

NameError: name 'df_Test_path' is not defined

## === cell 3
def read_image(path,img_height,img_width,img_chan):
    pixel=np.zeros((len(path), img_height, img_width, img_chan),dtype=np.float32)
    for n, p in tqdm_notebook(enumerate(path), total=len(path)):
        img = load_img(p)
        x = img_to_array(img)[:,:,1]
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        x=x/255
        pixel[n]=x
    return pixel

img_height=128
img_width=128
img_chan=1
Train_Image_pixel=read_image(df_Train_path.Train_Image_path,img_height,img_width,img_chan)
Train_Mask_pixel=read_image(df_Train_path.Train_Mask_path,img_height,img_width,img_chan)

print('Train Image shape: ',Train_Image_pixel.shape)
print('Train Mask shape: ',Train_Mask_pixel.shape)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2877750287.py in <cell line: 0>()
     12 img_width=128
     13 img_chan=1
---> 14 Train_Image_pixel=read_image(df_Train_path.Train_Image_path,img_height,img_width,img_chan)
     15 Train_Mask_pixel=read_image(df_Train_path.Train_Mask_path,img_height,img_width,img_chan)
     16 

NameError: name 'df_Train_path' is not defined

## === cell 4
def read_image2(path,img_height,img_width,img_chan):
    pixel=np.zeros((len(path), img_height, img_width, img_chan),dtype=np.float32)
    sizes_test = []
    for n, p in tqdm_notebook(enumerate(path), total=len(path)):
        img = load_img(p)
        x = img_to_array(img)[:,:,1]
        sizes_test.append([x.shape[0], x.shape[1]])
        x = resize(x, (128, 128, 1), mode='constant', preserve_range=True)
        x=x/255
        pixel[n]=x
    return pixel,sizes_test

Test_Image_pixel,sizes_test=read_image2(df_Test_path.Test_Image_path,img_height,img_width,img_chan)
print('Test Image shape: ',Test_Image_pixel.shape)


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1687294488.py in <cell line: 0>()
     12     return pixel,sizes_test
     13 
---> 14 Test_Image_pixel,sizes_test=read_image2(df_Test_path.Test_Image_path,img_height,img_width,img_chan)
     15 print('Test Image shape: ',Test_Image_pixel.shape)

NameError: name 'df_Test_path' is not defined

## === cell 5
array_to_img(Train_Image_pixel[0])


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2953950913.py in <cell line: 0>()
----> 1 array_to_img(Train_Image_pixel[0])

NameError: name 'array_to_img' is not defined

## === cell 6
array_to_img(Train_Mask_pixel[0])


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/814505643.py in <cell line: 0>()
----> 1 array_to_img(Train_Mask_pixel[0])

NameError: name 'array_to_img' is not defined

## === cell 7
array_to_img(Test_Image_pixel[0])


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/820379223.py in <cell line: 0>()
----> 1 array_to_img(Test_Image_pixel[0])

NameError: name 'array_to_img' is not defined

## === cell 8
def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


## === cell 9
def dice_coef(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + K.epsilon()) / (K.sum(y_true_f) + K.sum(y_pred_f) + K.epsilon())


## === cell 10
X=Train_Image_pixel
Y=Train_Mask_pixel
test=Test_Image_pixel
X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.20,random_state=42)
print(X_train.shape,y_train.shape)
print(X_val.shape,y_val.shape)
print(test.shape)


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/197091394.py in <cell line: 0>()
----> 1 X=Train_Image_pixel
      2 Y=Train_Mask_pixel
      3 test=Test_Image_pixel
      4 X_train,X_val,y_train,y_val=train_test_split(X,Y,test_size=0.20,random_state=42)
      5 print(X_train.shape,y_train.shape)

NameError: name 'Train_Image_pixel' is not defined

## === cell 11
inputs = Input((img_height, img_width, img_chan))

c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (inputs)
c1 = Conv2D(8, (3, 3), activation='relu', padding='same') (c1)
p1 = MaxPooling2D((2, 2)) (c1)

c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (p1)
c2 = Conv2D(16, (3, 3), activation='relu', padding='same') (c2)
p2 = MaxPooling2D((2, 2)) (c2)

c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (p2)
c3 = Conv2D(32, (3, 3), activation='relu', padding='same') (c3)
p3 = MaxPooling2D((2, 2)) (c3)

c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (p3)
c4 = Conv2D(64, (3, 3), activation='relu', padding='same') (c4)
p4 = MaxPooling2D(pool_size=(2, 2)) (c4)

c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (p4)
c5 = Conv2D(128, (3, 3), activation='relu', padding='same') (c5)

u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
u6 = concatenate([u6, c4])
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

u7 = Conv2DTranspose(32, (2, 2), strides=(2, 2), padding='same') (c6)
u7 = concatenate([u7, c3])
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (u7)
c7 = Conv2D(32, (3, 3), activation='relu', padding='same') (c7)

u8 = Conv2DTranspose(16, (2, 2), strides=(2, 2), padding='same') (c7)
u8 = concatenate([u8, c2])
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (u8)
c8 = Conv2D(16, (3, 3), activation='relu', padding='same') (c8)

u9 = Conv2DTranspose(8, (2, 2), strides=(2, 2), padding='same') (c8)
u9 = concatenate([u9, c1], axis=3)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (u9)
c9 = Conv2D(8, (3, 3), activation='relu', padding='same') (c9)

outputs = Conv2D(1, (1, 1), activation='sigmoid') (c9)

model = Model(inputs=[inputs], outputs=[outputs])
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[mean_iou])


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4067941581.py in <cell line: 0>()
     22 
     23 u6 = Conv2DTranspose(64, (2, 2), strides=(2, 2), padding='same') (c5)
---> 24 u6 = concatenate([u6, c4])
     25 c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (u6)
     26 c6 = Conv2D(64, (3, 3), activation='relu', padding='same') (c6)

NameError: name 'concatenate' is not defined

## === cell 12
earlystopper = EarlyStopping(patience=2, verbose=1)
checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
results = model.fit(X, Y,validation_split=0.1, batch_size=8, epochs=2,callbacks=[earlystopper, checkpointer])


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3749698703.py in <cell line: 0>()
      1 earlystopper = EarlyStopping(patience=2, verbose=1)
      2 checkpointer = ModelCheckpoint('model-tgs-salt-1.h5', verbose=1, save_best_only=True)
----> 3 results = model.fit(X, Y,validation_split=0.1, batch_size=8, epochs=2,callbacks=[earlystopper, checkpointer])

NameError: name 'model' is not defined

## === cell 13
model = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})
pred_train = model.predict(X_train, verbose=1)
pred_val = model.predict(X_val, verbose=1)
preds_test = model.predict(test, verbose=1)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4066897443.py in <cell line: 0>()
      1 # Predict on train, val and test
----> 2 model = load_model('model-tgs-salt-1.h5', custom_objects={'mean_iou': mean_iou})
      3 pred_train = model.predict(X_train, verbose=1)
      4 pred_val = model.predict(X_val, verbose=1)
      5 preds_test = model.predict(test, verbose=1)

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'model-tgs-salt-1.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 14
preds_test_upsampled = []
for i in tnrange(len(preds_test)):
    preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
                                       (sizes_test[i][0], sizes_test[i][1]), 
                                       mode='constant', preserve_range=True))


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3247420400.py in <cell line: 0>()
      1 # Create list of upsampled test masks
      2 preds_test_upsampled = []
----> 3 for i in tnrange(len(preds_test)):
      4     preds_test_upsampled.append(resize(np.squeeze(preds_test[i]), 
      5                                        (sizes_test[i][0], sizes_test[i][1]),

NameError: name 'preds_test' is not defined

## === cell 15
def RLenc(img, order='F', format=True):
    """
    img is binary mask image, shape (r,c)
    order is down-then-right, i.e. Fortran
    format determines if the order needs to be preformatted (according to submission rules) or not

    returns run length as an array or string (if format is True)
    """
    bytes = img.reshape(img.shape[0] * img.shape[1], order=order)
    runs = []  ## list of run lengths
    r = 0  ## the current run length
    pos = 1  ## count starts from 1 per WK
    for c in bytes:
        if (c == 0):
            if r != 0:
                runs.append((pos, r))
                pos += r
                r = 0
            pos += 1
        else:
            r += 1

    if r != 0:
        runs.append((pos, r))
        pos += r
        r = 0

    if format:
        z = ''

        for rr in runs:
            z += '{} {} '.format(rr[0], rr[1])
        return z[:-1]
    else:
        return runs

pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm_notebook(enumerate(df_Test_path.Test_Image_path))}


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1893228935.py in <cell line: 0>()
     36         return runs
     37 
---> 38 pred_dict = {fn[:-4]:RLenc(np.round(preds_test_upsampled[i])) for i,fn in tqdm_notebook(enumerate(df_Test_path.Test_Image_path))}

NameError: name 'df_Test_path' is not defined

## === cell 16
sub = pd.DataFrame.from_dict(pred_dict,orient='index')
rle=sub[0].values


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1012465976.py in <cell line: 0>()
----> 1 sub = pd.DataFrame.from_dict(pred_dict,orient='index')
      2 rle=sub[0].values

NameError: name 'pred_dict' is not defined

## === cell 17
df_sub.rle_mask=rle
df_sub.to_csv('sub1.csv',index=False)
df_sub.head()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1931986623.py in <cell line: 0>()
----> 1 df_sub.rle_mask=rle
      2 df_sub.to_csv('sub1.csv',index=False)
      3 df_sub.head()

NameError: name 'rle' is not defined
