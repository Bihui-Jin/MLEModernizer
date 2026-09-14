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

-0.087519

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
import os
import glob
import matplotlib.pyplot as plt
import cv2
from tqdm import tqdm_notebook as tqdm
%matplotlib inline

pd.set_option('display.max_rows', 10)


## === cell 1
base_data_folder = "/kaggle/input"
train_data_folder = os.path.join(base_data_folder, "train_images")

print(os.listdir(base_data_folder))


## === cell 2
train_files_names = os.listdir(train_data_folder)
train_files_names[:5]


## === cell 3
train_images = []
for file in tqdm(glob.glob(train_data_folder + '/*.png')):
    image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_bgr, dsize=(0,0), fx = 0.12, fy = 0.12)
    train_images.append(image_resized)


## === cell 4
len(train_images)


## === cell 5
train_labels = pd.read_csv(base_data_folder+"/train.csv", index_col = 0)
train_labels


## === cell 6
image_labels = pd.DataFrame(columns = ['id_code'])

for i in train_files_names:
    splited = i.split('.')[0]
    temp = pd.DataFrame({'id_code':[splited]})
    image_labels = pd.concat([image_labels, temp], ignore_index=True)

image_labels


## === cell 7
labels = pd.merge(image_labels, train_labels, on='id_code')
labels


## === cell 8
y_data = labels['diagnosis']
y_data[:5]


## === cell 9
labels['diagnosis'].hist()
labels['diagnosis'].value_counts()


## === cell 10
def crop_image_from_gray(img,tol=7):
    gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    mask = gray_img > tol
    
    img1=img[:,:,0][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img2=img[:,:,1][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img3=img[:,:,2][np.ix_(mask.any(axis=1),mask.any(axis=0))]
    img = np.stack([img1,img2,img3], axis=-1)
    
    return img


## === cell 11
def circle_crop(img):
    img = crop_image_from_gray(img)
    
    height, width, depth = img.shape
    largest_side = np.max((height, width))
    img = cv2.resize(img, dsize=(largest_side, largest_side),interpolation = cv2.INTER_CUBIC)
    
    height, width, depth = img.shape
    x = int(width/2)
    y = int(height/2)
    r = np.amin((x,y))
    
    
    background = np.zeros(shape=(height, width), dtype=np.uint8)
    circle_mask = cv2.circle(background, (x,y), int(r), 1, thickness=-1)
    
    img = cv2.bitwise_and(img, img, mask = background)
    
    return img


## === cell 12
pic_num = 43

img = train_images[pic_num]
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.imshow(img_rgb);


## === cell 13
img1 = crop_image_from_gray(img_rgb)
plt.imshow(img1);


## === cell 14
img2 = circle_crop(img_rgb)
plt.imshow(img2);


## === cell 15
X_data = []
for image in tqdm(train_images):
    img_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(img_rgb)
    image_resized = cv2.resize(circle_img, dsize=(224, 224), interpolation = cv2.INTER_CUBIC)
    X_data.append(image_resized)


## === cell 16
X_data = np.array(X_data).reshape(-1, 224, 224, 3)
print(X_data.shape)


## === cell 17
def values_in_mask(X):
    background = np.zeros(shape=(224, 224), dtype=np.uint8)
    circle_mask = cv2.circle(background, (112,112), 110, 1, thickness=-1)
    
    dim1 = X[:,:,0]
    dim2 = X[:,:,1]
    dim3 = X[:,:,2]
    
    circle_locations = (circle_mask == 1)
    R = dim1[circle_locations]
    G = dim2[circle_locations]
    B = dim3[circle_locations]
    
    return R, G, B


## === cell 18
def min_max_scaler_rgb(X):
    
    R, G, B = values_in_mask(X)   
    
    dim1 = X[:,:,0].astype('float32')
    dim2 = X[:,:,1].astype('float32')
    dim3 = X[:,:,2].astype('float32')
    
    min_R = np.min(R)
    min_G = np.min(G)
    min_B = np.min(B)
    
    max_R = np.max(R)
    max_G = np.max(G)
    max_B = np.max(B)
    
    img_R = (dim1 - min_R) / (max_R - min_R)
    img_G = (dim2 - min_G) / (max_G - min_G)
    img_B = (dim3 - min_B) / (max_B - min_B)
    
    img_R = np.where(img_R < 0, 0, img_R)
    img_G = np.where(img_G < 0, 0, img_G)
    img_B = np.where(img_B < 0, 0, img_B)
    
    img_R = np.where(img_R > 1, 1, img_R)
    img_G = np.where(img_G > 1, 1, img_G)
    img_B = np.where(img_B > 1, 1, img_B)
    
    img = np.stack([img_R, img_G, img_B], axis=-1)
    return img


## === cell 19
def min_max_scaler_gray(X):
    img = (X-np.min(X)) / (np.max(X)-np.min(X))
    return img


## === cell 20
fig = plt.figure(figsize=(14,8))

for idx, image in enumerate(train_images[:10]):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    fig.add_subplot(2, 5, idx+1)
    plt.imshow(image_rgb)
    plt.title("Label:{0}".format(labels['diagnosis'][idx]))
    plt.xlabel(labels['id_code'][idx])
    plt.tight_layout()


## === cell 21
fig = plt.figure(figsize=(14,8))

for idx, image in enumerate(X_data[:10]):
    fig.add_subplot(2, 5, idx+1)
    plt.imshow(image)
    plt.title("Label:{0}".format(labels['diagnosis'][idx]))
    plt.xlabel(labels['id_code'][idx])
    plt.tight_layout()


## === cell 22
del train_images


## === cell 23
from sklearn.model_selection import train_test_split
X_train, X_valid, y_train, y_valid = train_test_split(X_data, y_data, test_size=0.2,
                                                      stratify = y_data, random_state = 123)

print(X_train.shape, y_train.shape)
print(X_valid.shape, y_valid.shape)


## === cell 24
from tensorflow.keras.utils import to_categorical

y_train_onehot = to_categorical(y_train, num_classes=5)
y_valid_onehot = to_categorical(y_valid, num_classes=5)

print(y_train_onehot.shape)
print(y_valid_onehot.shape)


## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 25
plt.hist(y_train)
plt.hist(y_valid)
plt.title("Train and Validation set Distribution")
plt.legend(['Train', 'Validation'])
plt.show()


## === cell 26
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=123456)
X_resampled, y_resampled = ros.fit_resample(X_train.reshape(-1, 224*224*3), y_train)


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1867660675.py in <cell line: 0>()
----> 1 from imblearn.over_sampling import RandomOverSampler
      2 
      3 ros = RandomOverSampler(random_state=123456)
      4 X_resampled, y_resampled = ros.fit_resample(X_train.reshape(-1, 224*224*3), y_train)

/usr/local/lib/python3.11/dist-packages/imblearn/__init__.py in <module>
     50     # process, as it may not be compiled yet
     51 else:
---> 52     from . import (
     53         combine,
     54         ensemble,

/usr/local/lib/python3.11/dist-packages/imblearn/combine/__init__.py in <module>
      3 """
      4 
----> 5 from ._smote_enn import SMOTEENN
      6 from ._smote_tomek import SMOTETomek
      7 

/usr/local/lib/python3.11/dist-packages/imblearn/combine/_smote_enn.py in <module>
     10 from sklearn.utils import check_X_y
     11 
---> 12 from ..base import BaseSampler
     13 from ..over_sampling import SMOTE
     14 from ..over_sampling.base import BaseOverSampler

/usr/local/lib/python3.11/dist-packages/imblearn/base.py in <module>
     10 from sklearn.base import BaseEstimator, OneToOneFeatureMixin
     11 from sklearn.preprocessing import label_binarize
---> 12 from sklearn.utils._metadata_requests import METHODS
     13 from sklearn.utils.multiclass import check_classification_targets
     14 

ModuleNotFoundError: No module named 'sklearn.utils._metadata_requests'

## === cell 27
X_resampled = X_resampled.reshape(-1, 224, 224, 3)
y_resampled_onehot = to_categorical(y_resampled, num_classes=5)

print(X_resampled.shape, y_resampled_onehot.shape)


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2838997981.py in <cell line: 0>()
----> 1 X_resampled = X_resampled.reshape(-1, 224, 224, 3)
      2 y_resampled_onehot = to_categorical(y_resampled, num_classes=5)
      3 
      4 print(X_resampled.shape, y_resampled_onehot.shape)

NameError: name 'X_resampled' is not defined

## === cell 28
plt.hist(y_resampled)
plt.hist(y_valid)
plt.title("Train and Validation set Distribution (Train oversampled)")
plt.legend(['Train resampled', 'Validation'], loc='right')
plt.show()


## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1993242277.py in <cell line: 0>()
----> 1 plt.hist(y_resampled)
      2 plt.hist(y_valid)
      3 plt.title("Train and Validation set Distribution (Train oversampled)")
      4 plt.legend(['Train resampled', 'Validation'], loc='right')
      5 plt.show()

NameError: name 'y_resampled' is not defined

## === cell 29
del X_data, X_train


## === cell 30
from tensorflow.keras import layers, models

model = models.Sequential()


## === cell 31
model.add(layers.Conv2D(filters = 64, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', input_shape = (224, 224, 3), name = 'Conv1-1'))
model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool1'))


## === cell 32
model.add(layers.Conv2D(filters = 128, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv2-1'))
model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool2'))


## === cell 33
model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv3-1'))
model.add(layers.Conv2D(filters = 256, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv3-2'))
model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool3'))


## === cell 34
model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv4-1'))
model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv4-2'))
model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool4'))


## === cell 35
model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv5-1'))
model.add(layers.Conv2D(filters = 512, kernel_size = (3, 3), padding = 'same',
                        activation = 'relu', name = 'Conv5-2'))
model.add(layers.MaxPool2D(pool_size = (2, 2), strides = (2, 2), name = 'pool5'))


## === cell 36
model.add(layers.Flatten())


## === cell 37
model.add(layers.Dense(256, activation = 'relu', name='Dense1'))
model.add(layers.Dropout(0.3))


## === cell 38
model.add(layers.Dense(256, activation = 'relu', name='Dense2'))
model.add(layers.Dropout(0.3))


## === cell 39
model.add(layers.Dense(5, activation = 'softmax', name='Final')) # output layer


## === cell 40
model.summary()


## === cell 41
model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['acc'])


## === cell 42
import time
from tensorflow.keras.callbacks import ModelCheckpoint

callback_list = [ModelCheckpoint(filepath='cnn_checkpoint.h5',
                                 monitor = 'val_acc',
                                 save_best_only = True)]


## === cell 43
                    batch_size = 200, epochs = 100,
                    validation_data = (X_valid, y_valid_onehot),
                    callbacks = callback_list)


## --- ERROR in cell 43, traceback:
  File "/tmp/ipykernel_11/1490334366.py", line 2
    batch_size = 200, epochs = 100,
    ^
IndentationError: unexpected indent


## === cell 44
epochs = np.arange(1, len(history.history['loss']) + 1)

plt.plot(epochs, history.history['loss'], label = 'Training')
plt.plot(epochs, history.history['val_loss'], label = 'Validation')
plt.title('Loss History Plot')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()

plt.plot(epochs, history.history['acc'], label = 'Training')
plt.plot(epochs, history.history['val_acc'], label = 'Validation')
plt.title('Accuracy History Plot')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()


## --- ERROR in cell 44, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4167677848.py in <cell line: 0>()
----> 1 epochs = np.arange(1, len(history.history['loss']) + 1)
      2 
      3 plt.plot(epochs, history.history['loss'], label = 'Training')
      4 plt.plot(epochs, history.history['val_loss'], label = 'Validation')
      5 plt.title('Loss History Plot')

NameError: name 'history' is not defined

## === cell 45
model.save('cnn_model.h5')


## === cell 46
from tensorflow.keras.models import load_model

restored_model = load_model('cnn_model.h5')
restored_model.load_weights('cnn_checkpoint.h5')


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3550941282.py in <cell line: 0>()
      2 
      3 restored_model = load_model('cnn_model.h5')
----> 4 restored_model.load_weights('cnn_checkpoint.h5')

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = 'cnn_checkpoint.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 47
restored_model.evaluate(X_valid, y_valid_onehot)


## === cell 48
y_pred = np.argmax(restored_model.predict(X_valid), axis = 1)

print('Predict:', y_pred[:10])
print('Validation:', np.array(y_valid[:10]))


## === cell 49
from sklearn.metrics import confusion_matrix

cm = confusion_matrix(y_true = y_valid,
                      y_pred = y_pred)

print("Confusion Matrix")
print(cm)
print()
print("Shape :", cm.shape)
print("Accurcy: {0:.2f}%".format(np.trace(cm) / np.sum(cm)*100))


## === cell 50
from sklearn.metrics import classification_report

print(classification_report(y_valid, y_pred, digits=4, target_names = ['No DR', 'Mild', 'Moderate', 'Severe', 'Proliferative DR']))


## === cell 51
test_data_folder = os.path.join(base_data_folder, "test_images")


## === cell 52
X_test = []
for file in tqdm(glob.glob(test_data_folder + '/*.png')):
    image_bgr = cv2.imread(file, cv2.IMREAD_COLOR)
    image_resized = cv2.resize(image_bgr, dsize=(0,0), fx = 0.12, fy = 0.12)
    
    image_rgb = cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB)
    circle_img = circle_crop(image_rgb)
    image_resized2 = cv2.resize(circle_img, dsize=(224, 224), interpolation = cv2.INTER_CUBIC)
    
    X_test.append(image_resized2)


## === cell 53
X_test = np.array(X_test)
X_test.shape


## === cell 54
glob.glob(test_data_folder + '/*.png')[:5]


## === cell 55
test_files_names = os.listdir(test_data_folder)
test_files_names[:5]


## === cell 56
test_image_labels = pd.DataFrame(columns = ['id_code'])

for i in test_files_names:
    splited = i.split('.')[0]
    temp = pd.DataFrame({'id_code':[splited]})
    test_image_labels = pd.concat([test_image_labels, temp], ignore_index=True)

test_image_labels


## === cell 57
preds = np.argmax(restored_model.predict(X_test), axis = 1)

print('Predicted:', y_pred[:10])


## === cell 58
test_image_labels['diagnosis'] = pd.Series(preds)
test_image_labels


## === cell 59
test_image_labels.sort_values(by=['id_code'], inplace=True)
test_image_labels.reset_index(drop=True, inplace = True)


## === cell 60
test_image_labels


## === cell 61
fig = plt.figure(figsize=(14,8))

for idx, image in enumerate(X_test[:20]):
    fig.add_subplot(4, 5, idx+1)
    plt.imshow(image)
    plt.title('diagnosed:{0}'.format(test_image_labels['diagnosis'][idx]))
    plt.xlabel(test_image_labels['id_code'][idx])
    plt.tight_layout()


## === cell 62
plt.hist(test_image_labels['diagnosis'])
plt.show()


## === cell 63
test_image_labels.to_csv('submission.csv', index=False, header=True)


## === cell 64
!ls -al


## --- ERROR in outputing the csv:
Invalid submission: Submission must have the same id_codes as answers
