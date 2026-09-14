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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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

0.68244

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import zipfile
from PIL import Image
import matplotlib.image as mpimg
import os
import random as rn
import tensorflow as tf
from keras import backend as K

os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
rn.seed(1234)
num_train_images= 4000

session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
tf.set_random_seed(1234)

sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
K.set_session(sess)

print(os.listdir("../input"))

df_train = pd.read_csv("../input/train.csv")
print("\n Train files shape is ", df_train.shape)

df_depths = pd.read_csv("../input/depths.csv")

df_depths['z'] = df_depths['z'] / np.max(df_depths['z'])
print("\nDepths files shape is ", df_depths.shape)

df_train = df_train.join(df_depths, lsuffix='idl', rsuffix='idr', how ='inner')
df_train.pop('ididr')
df_train.columns=['id', 'rle_mask','depth']
                        


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_train['images'] = [np.array(Image.open("../input/train/images/{}.png".format(idx))) for 
                               idx in df_train['id']]
print("Sample Image Shape is ", df_train['images'][0].shape)

df_train['images'] = pd.Series(map(lambda x: np.delete(x,np.s_[1:],2), df_train['images']))
print("After optimization,  Image Shape is ", df_train['images'][0].shape)
print("No. of train images are ", len(df_train))

df_train['images']=df_train['images']/255
print("After normalization,  pixel value is ", df_train['images'][0][1,0])

df_train['masks'] = [np.array(Image.open("../input/train/masks/{}.png".format(idx))) for 
                               idx in df_train['id']]
print("\nSample Mask Shape is ", df_train['masks'][0].shape)

print("No. of mask images are ", len(df_train))
print("Before normalization,  pixel value is ", df_train['masks'][15][10,0])

print("train df columns are ", df_train.columns)


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4126030441.py in <cell line: 0>()
      1 #2. Read train & mask images and add them up df_train
      2 df_train['images'] = [np.array(Image.open("../input/train/images/{}.png".format(idx))) for 
----> 3                                idx in df_train['id']]
      4 print("Sample Image Shape is ", df_train['images'][0].shape)
      5 

NameError: name 'df_train' is not defined

## === cell 2
train_x = np.array(df_train['images'])
train_x = np.concatenate(train_x,axis=0)
train_x = np.reshape(train_x,(4000,101,101,1))
print("Train Shape = ",train_x.shape)
print("Sample Pixel VAlue", train_x[0,1,0])

train_y = np.array(df_train['masks'])
train_y = np.concatenate(train_y,axis=0)
train_y = np.reshape(train_y,(4000,101,101,1))
train_y = train_y / train_y.max()
print("Mask Shape = ",train_y.shape)
train_y = np.round(train_y)
print("Sample Pixel VAlue", train_y[15,10,0])


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/157889684.py in <cell line: 0>()
----> 1 train_x = np.array(df_train['images'])
      2 train_x = np.concatenate(train_x,axis=0)
      3 train_x = np.reshape(train_x,(4000,101,101,1))
      4 print("Train Shape = ",train_x.shape)
      5 print("Sample Pixel VAlue", train_x[0,1,0])

NameError: name 'df_train' is not defined

## === cell 3
print(train_x.shape)
print(train_y.shape)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/376544986.py in <cell line: 0>()
      1 #train_x = np.append(train_x, [np.fliplr(x) for x in train_x], axis=0)
      2 #train_y = np.append(train_y, [np.fliplr(x) for x in train_y], axis=0)
----> 3 print(train_x.shape)
      4 #print(train_x_hflip.shape)
      5 print(train_y.shape)

NameError: name 'train_x' is not defined

## === cell 4
import os
import scipy.signal as sg
import tensorflow as tf
from keras.models import Model
from keras import layers, regularizers
from keras import backend as K
import numpy as np
from keras import layers
Height = 101
Width = 101


## === cell 5
import random as rn
os.environ['PYTHONHASHSEED'] = '0'
np.random.seed(1234)
rn.seed(1234)
num_train_images= 4000

df_train['depth_image'] = pd.Series(map(lambda x:np.full(shape=(50,50,1), 
                                                         fill_value=x), df_train['depth']))

depth_np = np.array(df_train['depth_image'])
depth_np = np.concatenate(depth_np,axis=0)
depth_np = np.reshape(depth_np,(num_train_images,50,50,1))
depth_np_new = np.array(df_train['depth'])
print("depth", depth_np_new[0:5])
print(depth_np_new.shape)
depth_np_new = depth_np_new.reshape((num_train_images,1,1,1))
print("depth shape ", depth_np_new.shape)
depth_np= depth_np/depth_np.max()
print(df_train.columns)
print(depth_np.shape)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1217474005.py in <cell line: 0>()
     10 
     11 df_train['depth_image'] = pd.Series(map(lambda x:np.full(shape=(50,50,1), 
---> 12                                                          fill_value=x), df_train['depth']))
     13 
     14 depth_np = np.array(df_train['depth_image'])

NameError: name 'df_train' is not defined

## === cell 6
Activation1 = 'elu'
Activation2 = 'tanh'
def BatchActivate(x):
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    return x

def convolution_block(x, filters, size, strides=(1,1), padding='same', activation=True):
    x = layers.Conv2D(filters, size, strides=strides, padding=padding)(x)
    if activation == True:
        x = layers.BatchNormalization()(x)
        x = layers.Activation(Activation1)(x)
    return x

def residual_block(blockInput, num_filters=16, activate = True):
    x = convolution_block(blockInput, num_filters, (3,3) )
    x = convolution_block(x, num_filters, (3,3), activation=False)
    x = layers.Add()([x, blockInput])
    x = layers.BatchNormalization()(x)
    if activate:
        x = layers.Activation(Activation2)(x)
    return x


## === cell 7
reg  = regularizers.l2(0.01)
img_input = layers.Input(shape=(Height, Width,1), name = 'img_input')
depth_input = layers.Input(shape=(50,50,1), name='depth_input')
depth_input_new = layers.Input(shape=(1,1,1), name='depth_input_new')

print(img_input)
print(depth_input)

print("101->100")
x = layers.Conv2D(filters = 16, kernel_size= (2,2), padding='valid', activation='elu')(img_input)
print(x)

x_depth = layers.Conv2D(filters = 16, kernel_size= (2,2), padding='valid', activation='elu')(depth_input)
print(x_depth)

print("100 -> 50") #100-> 50
x_16_pool = layers.MaxPooling2D(pool_size=(2,2))(x)
x_16_pool = layers.BatchNormalization()(x_16_pool)
print(x)

x_16_pool_depth = layers.AveragePooling2D(pool_size=(2,2))(x_depth)
x_16_pool_depth = layers.BatchNormalization()(x_16_pool_depth)
x_16_pool_depth = layers.Dropout(0.2)(x_16_pool_depth)
print(x_16_pool_depth)

print("50-> 48") #50 > 48
x = layers.Conv2D(32, 3, padding='valid',activation='elu')(x_16_pool)
x2 = layers.Conv2D(32, 3, padding='same',activation='tanh', kernel_regularizer=reg)(x) 
x = layers.add([x, x2])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(32, 3, padding='valid',activation='elu')(x_16_pool_depth)
x2_depth = layers.Conv2D(32, 3, padding='same',activation='tanh', 
                         kernel_regularizer=reg)(x_depth)
x_depth = layers.add([x_depth, x2_depth])#
x_depth = layers.BatchNormalization()(x_depth)#

print(x_depth)
print("48-> 24")#48->24
x_32_pool = layers.MaxPooling2D(2)(x)
print(x_32_pool)

x_32_pool_depth = layers.AveragePooling2D(2)(x_depth)
print(x_32_pool_depth)

x = layers.Dropout(0.25)(x_32_pool)
x_depth = layers.Dropout(0.2)(x_32_pool_depth)

x = layers.Conv2D(64, 3, padding = 'same', activation='elu')(x) 
x3 = layers.Conv2D(64, 3, padding = 'same', activation='elu', kernel_regularizer=reg)(x)#removed+2closed
x = layers.add([x, x3])
x = layers.BatchNormalization()(x)
print(x)

x_depth = layers.Conv2D(64, 3, padding='same',activation='elu')(x_32_pool_depth)
x2_depth = layers.Conv2D(64, 3, padding='same',activation='tanh', kernel_regularizer=reg)(x_depth)
x_depth = layers.add([x_depth, x2_depth])
x_depth = layers.BatchNormalization()(x_depth)

print(x_depth)
print("24->12") #24->12
x_64_pool = layers.MaxPooling2D(2)(x)
x_64_pool_depth = layers.AveragePooling2D(2)(x_depth)

print(x)
x_64_pool = layers.Dropout(0.25)(x_64_pool)
x_64_pool_depth = layers.Dropout(0.2)(x_64_pool_depth)


x = layers.Conv2D(128, 3, padding = 'same', activation='elu')(x_64_pool)
x2 = layers.Conv2D(128, 3, padding = 'same', activation='elu', 
                   kernel_regularizer=reg)(x)#removed+1closed
x = layers.add([x,x2])
print("12->6") #12->6
x_128_pool = layers.MaxPooling2D(2)(x)
print(x_128_pool)

x_128_pool = layers.Dropout(0.25)(x_128_pool)

x = layers.Conv2D(192, 3, padding = 'same', activation='elu', 
                  kernel_regularizer=reg)(x_128_pool)
print(x)
x_192_pool = layers.MaxPooling2D(2)(x)
print(x_192_pool)

x_192_pool = layers.Dropout(0.25)(x_192_pool)

x = layers.Conv2D(192, 2, padding = 'valid', activation='elu', 
                  kernel_regularizer=regularizers.l2(0.01))(x_192_pool)
print(x)
x = layers.MaxPooling2D(2)(x)
print(x)

x = layers.Dropout(0.25)(x)



## === cell 8
inverse = layers.Conv2DTranspose(filters = 192, kernel_size=(3,3), strides=(2, 2), padding='valid', 
                                 activation = 'elu', kernel_regularizer=reg)(x)
print(inverse)

inverse = layers.Concatenate()([inverse,x_192_pool])

inverse = layers.Conv2D(filters=192, kernel_size=(2,2), padding='same', activation='elu',
                       kernel_regularizer=reg)(inverse)
print(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)



print(inverse)

inverse = layers.Conv2DTranspose(filters = 128, kernel_size=(2,2), strides=(2, 2), padding='valid',
                                 kernel_regularizer=reg, activation = 'elu')(inverse)
print("ok ",inverse)

inverse = layers.Concatenate()([inverse,x_128_pool])
print("concatination ",inverse)
inverse = layers.Conv2D(filters=128, kernel_size=(3,3), padding='same', activation='elu')(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)

inverse = layers.Conv2DTranspose(filters = 64, kernel_size=(2,2), strides=(2, 2), 
                                 padding='valid', activation = 'elu')(inverse)
print(inverse)

inverse = layers.Concatenate()([inverse,x_64_pool])

inverse = layers.Conv2D(filters=64, kernel_size=(3,3), padding='same', activation='elu',
                       kernel_regularizer=reg)(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)




inverse = layers.Conv2DTranspose(filters = 32, kernel_size=(2,2), strides=(2, 2), 
                                 padding='valid', activation = 'elu')(inverse)
print(inverse)

inverse = layers.Conv2D(filters=32, kernel_size=(3,3), padding='same', activation='elu',
                       kernel_regularizer=reg)(inverse)
inverse = layers.Concatenate()([inverse,x_32_pool])
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)



inverse = layers.Conv2DTranspose(filters = 16, kernel_size=(4,4), strides=(2, 2), 
                                 padding='valid', activation = 'elu')(inverse)
print(inverse)

inverse = layers.Concatenate()([inverse,x_16_pool])
inverse = layers.Concatenate()([inverse,depth_input])

inverse = layers.Conv2D(filters=16, kernel_size=(3,3), padding='same', activation='elu')(inverse)
inverse = layers.BatchNormalization()(inverse)

inverse = layers.Dropout(0.25)(inverse)

print(inverse)





inverse2 = layers.Conv2DTranspose(filters = 1, kernel_size=(3,3), strides=(2, 2), 
                                  padding='valid',activation = 'sigmoid')(inverse)
print(inverse2)





model = Model([img_input,depth_input], inverse2)
model.summary()


## === cell 9
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

Batch_size = 96
gen = ImageDataGenerator(horizontal_flip = True,
                         vertical_flip = True)

def gen_flow_for_two_inputs(X1, X2, y):
    genX1 = gen.flow(X1,y,  batch_size=Batch_size,seed = 1234, shuffle=True)
    genX2 = gen.flow(X1,X2, batch_size=Batch_size,seed = 1234, shuffle=True)
    while True:
            X1i = genX1.next()
            X2i = genX2.next()
            yield [X1i[0], X2i[1]], X1i[1]
            
def get_iou(y_true, y_pred):
    y_true = y_true.flatten()
    y_pred = y_pred.flatten()
    y_pred = np.round(y_pred)
    intersect = np.sum(np.logical_not(np.logical_xor(y_true, y_pred)))
    union = y_true.shape[0] #np.sum(y_true) + np.sum(y_pred)
    return (intersect + 1e-9) / (union +1e-9)

def my_iou_metric(label, pred):
    return tf.py_func(get_iou, [label, pred>0.5], tf.float64)


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/235112939.py in <cell line: 0>()
----> 1 from keras.preprocessing.image import ImageDataGenerator
      2 from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
      3 
      4 Batch_size = 96
      5 gen = ImageDataGenerator(horizontal_flip = True,

ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image' (/usr/local/lib/python3.11/dist-packages/keras/api/preprocessing/image/__init__.py)

## === cell 10
import keras.losses as losses
def dice_coeff(y_true, y_pred):
    smooth = 0.00001
    y_true_f = tf.reshape(y_true, [-1])
    y_pred_f = tf.reshape(y_pred, [-1])
    intersection = tf.reduce_sum(y_true_f * y_pred_f)
    score = (2. * intersection + smooth) / (tf.reduce_sum(y_true_f) + tf.reduce_sum(y_pred_f) + smooth)
    return score

def dice_loss(y_true, y_pred):
    loss = 1 - dice_coeff(y_true, y_pred)
    return loss
def bce_dice_loss(y_true, y_pred):
    loss = 0.2 * losses.binary_crossentropy(y_true, y_pred) + 0.8 * dice_loss(y_true, y_pred)
    return loss


## === cell 11
from sklearn.model_selection import train_test_split

train_x1,val_x,depth_np1, val_depth, train_y1, val_y = train_test_split(
    train_x, depth_np, train_y, test_size=0.25, random_state=1234)
epochs = 60
from  keras.optimizers import Adam

model.compile(optimizer='adam', loss=bce_dice_loss, metrics=[dice_coeff, 'acc'])#added
from keras.callbacks import EarlyStopping


early_stopping = EarlyStopping(monitor='dice_coeff', patience=3, mode='max')
history = model.fit([train_x,depth_np],train_y, #added
                   batch_size = Batch_size,
                   epochs=epochs
                   ,callbacks=[early_stopping]
                   )




model.save('model_7.h5')


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2672558095.py in <cell line: 0>()
      2 
      3 train_x1,val_x,depth_np1, val_depth, train_y1, val_y = train_test_split(
----> 4     train_x, depth_np, train_y, test_size=0.25, random_state=1234)
      5 epochs = 60
      6 from  keras.optimizers import Adam

NameError: name 'train_x' is not defined

## === cell 12
print("Predicing the validation set....")
val_x_pred =model.predict([val_x, val_depth])
print("Validation set prediction completed.")


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/775685885.py in <cell line: 0>()
      1 print("Predicing the validation set....")
----> 2 val_x_pred =model.predict([val_x, val_depth])
      3 print("Validation set prediction completed.")

NameError: name 'val_x' is not defined

## === cell 13
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(16, 8))
columns = 3
rows = 4
%matplotlib inline
fig, axs = plt.subplots(rows, columns)
index = np.random.randint(0,1000,rows)
val_x_pred_new = np.round(val_x_pred-0.25)#-0.05)
print(index)
for i in range(4):
        
    axs[i, 0].imshow(val_x[index[i],:,:,0])#, cmap=cmap)
    axs[i,1].imshow(val_y[index[i],:,:,0])#, cmap=cmap)
    axs[i,2].imshow(val_x_pred_new[index[i],:,:,0])#, cmap=cmap)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1951876662.py in <cell line: 0>()
      9 #print(y_pred[0,:,:,0].shape)
     10 #index = [206, 831, 524, 676]
---> 11 val_x_pred_new = np.round(val_x_pred-0.25)#-0.05)
     12 print(index)
     13 #plt.imshow(y_pred[100,:,:,0])

NameError: name 'val_x_pred' is not defined

## === cell 14
df_test = df_depths[~df_depths['id'].isin(df_train['id'])]

print("Length of df_test is ", len(df_test))
df_test = df_test.reset_index()
df_test.pop('index')
print(df_test.head())



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3293207553.py in <cell line: 0>()
      1 #lmodel = load_model('model_7.h5')
----> 2 df_test = df_depths[~df_depths['id'].isin(df_train['id'])]
      3 
      4 print("Length of df_test is ", len(df_test))
      5 df_test = df_test.reset_index()

NameError: name 'df_depths' is not defined

## === cell 15
df_test['images'] = [np.array(Image.open("../input/test/images/{}.png".format(idx))) for 
                               idx in df_test['id']]


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/455189203.py in <cell line: 0>()
      1 df_test['images'] = [np.array(Image.open("../input/test/images/{}.png".format(idx))) for 
----> 2                                idx in df_test['id']]

NameError: name 'df_test' is not defined

## === cell 16
print("Sample Image Shape is ", df_test['images'][10].shape)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1518613967.py in <cell line: 0>()
      1 #print(df_test.head())
----> 2 print("Sample Image Shape is ", df_test['images'][10].shape)
      3 #print("Sample Image Shape is ", df_test.loc['images'][0].shape)

NameError: name 'df_test' is not defined

## === cell 17
df_test['images'] = pd.Series(map(lambda x: np.delete(x,np.s_[1:],2), df_test['images']))
print("After optimization,  Image Shape is ", df_test['images'][0].shape)
print("No. of test images are ", len(df_test))
print("Before normalization,  pixel value is ", df_test['images'][10][2,0])
df_test['images']=df_test['images'] / 255
print("After normalization,  pixel value is ", df_test['images'][10][2,0])

print(df_test.columns)

test_x = np.array(df_test['images'])
test_x = np.concatenate(test_x,axis=0)
test_x = np.reshape(test_x,(18000,101,101,1))
print(test_x.max())
print(test_x.min())
print(train_x.max())
print(train_x.min())
print("Train Shape = ",test_x.shape)
print("Sample Pixel VAlue", test_x[10,2,0])
print(df_test.columns)


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3803956017.py in <cell line: 0>()
----> 1 df_test['images'] = pd.Series(map(lambda x: np.delete(x,np.s_[1:],2), df_test['images']))
      2 print("After optimization,  Image Shape is ", df_test['images'][0].shape)
      3 print("No. of test images are ", len(df_test))
      4 print("Before normalization,  pixel value is ", df_test['images'][10][2,0])
      5 df_test['images']=df_test['images'] / 255

NameError: name 'df_test' is not defined

## === cell 18
df_test['depth_image'] = pd.Series(map(lambda x:np.full(shape=(50,50,1), 
                                                         fill_value=x), df_test['z']))

depth_test_np = np.array(df_test['depth_image'])
depth_test_np = np.concatenate(depth_test_np,axis=0)
depth_test_np = np.reshape(depth_test_np,(18000,50,50,1))

print(df_test.columns)
print(depth_test_np.shape)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2711258016.py in <cell line: 0>()
      1 df_test['depth_image'] = pd.Series(map(lambda x:np.full(shape=(50,50,1), 
----> 2                                                          fill_value=x), df_test['z']))
      3 
      4 depth_test_np = np.array(df_test['depth_image'])
      5 depth_test_np = np.concatenate(depth_test_np,axis=0)

NameError: name 'df_test' is not defined

## === cell 19
def post_process(x):
    num_files= x.shape[0]
    main_list = []
    print('x shape ', x.shape)
    pic_size = Height * Width
    for i in range(num_files):
        pic = x[i].T.flatten()
        s = ''
        length = 0
        start = 0
        for j in range(pic_size):
            if (j == pic_size-1):
                if((pic[j]==1) and (length==0)):
                    s="{} {} 1".format(s, str(j))
                if( (pic[j]==1) and (length != 0)):
                    s="{} {} {}".format(s, str(start), str(length+1))

            else: #if(j != pic_size-1):
                if(pic[j]==1):
                    length += 1
                    if(length==1):
                        start = j+1
                
            if ((pic[j]==0) and (length>0)):
                s = "{} {} {}".format(s, str(start), str(length))
                length = 0
        main_list.append(s)
    return main_list


## === cell 20
print("Predicting now.....")
y_pred = model.predict([test_x, depth_test_np])
print("Prediction Complete")


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1484806107.py in <cell line: 0>()
      1 #Make the prediction
      2 print("Predicting now.....")
----> 3 y_pred = model.predict([test_x, depth_test_np])
      4 print("Prediction Complete")

NameError: name 'test_x' is not defined

## === cell 21
import matplotlib.pyplot as plt

fig=plt.figure(figsize=(8, 8))
columns = 1
rows = 4
%matplotlib inline
print(y_pred[0,:,:,0].shape)
y_pred_new = np.round(y_pred-0.25)
for i in range(1, 4):
    fig.add_subplot(rows, columns, i)
    x = np.random.randint(0,4000,1)
    print(x[0])
    plt.imshow(y_pred_new[x[0],:,:,0])
plt.show()


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/781645841.py in <cell line: 0>()
      5 rows = 4
      6 get_ipython().run_line_magic('matplotlib', 'inline')
----> 7 print(y_pred[0,:,:,0].shape)
      8 y_pred_new = np.round(y_pred-0.25)
      9 #plt.imshow(y_pred[100,:,:,0])

NameError: name 'y_pred' is not defined

## === cell 22
for i in range(5,7):
    print(i)
    
    new_y_pred = post_process(np.round(y_pred- (i * 0.05)).astype(int))
    output = pd.DataFrame( data = {'id': df_test['id'], 'rle_mask' : new_y_pred} )
    file_name = "output_"+ str(i) +".csv"
    print(file_name)
    output.to_csv(file_name, index=False)


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3488215933.py in <cell line: 0>()
      4     print(i)
      5 
----> 6     new_y_pred = post_process(np.round(y_pred- (i * 0.05)).astype(int))
      7     output = pd.DataFrame( data = {'id': df_test['id'], 'rle_mask' : new_y_pred} )
      8     file_name = "output_"+ str(i) +".csv"

NameError: name 'y_pred' is not defined
