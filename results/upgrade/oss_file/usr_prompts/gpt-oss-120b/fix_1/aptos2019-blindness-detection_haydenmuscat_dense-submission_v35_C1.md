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

No external packages required in the script and installed.

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

0.8830832943471097

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy  as np 
import pandas as pd 
import os
import cv2 
import gc
import psutil
import matplotlib.pyplot as plt
import ray
from sklearn.metrics import cohen_kappa_score, confusion_matrix

import keras
from keras.preprocessing import image
from keras.models import Sequential, Model
from keras.applications import DenseNet121
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Input
from keras.layers import Dropout, Flatten, Dense, BatchNormalization
from keras.callbacks import Callback, ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from keras.activations import softmax, relu
from keras.optimizers import Adam

IMG_DIM       = 300
BATCH_SIZE    = 16
CHANNELS      = 3
NUM_CLASSES   = 5

NORMAL_WEIGHTS = "../input/densenetmulti/300_ben_normal_-0.8925.h5"

print(os.listdir("../"))
print(os.listdir("../input/"))
print(os.listdir("../input/aptos2019-blindness-detection"))
print(os.listdir("../input/densenetmulti"))

INPUT_FOLDER = '../input/aptos2019-blindness-detection/'

print(psutil.cpu_count())


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2

def crop(gray, img, percent_smaller):
    
    thresh = 8
    
    top    = 0
    left   = 0
    bottom = gray.shape[0] - 1
    right  = gray.shape[1] - 1
    
    middleCol = gray[:, int(gray.shape[1]/2)] > thresh
    while middleCol[top] == 0:
        top += 1
    while middleCol[bottom] == 0:
        bottom -= 1
        
    middleRow = gray[int(gray.shape[0]/2)] > thresh
    while middleRow[left] == 0:
        left += 1
    while middleRow[right] == 0:
        right -= 1
        
    height = bottom - top
    width  = right - left
    
    bottom -= int(percent_smaller*height)
    top    += int(percent_smaller*height)
    right  -= int(percent_smaller*width)
    left   += int(percent_smaller*width)
        
    if height < 100 or width < 100:
        print("Error: squareUp: bottom:", bottom, "top:", top)
        print("Error: squareUp: right:", right, "left:", left)
        return img
    
    return img[top:bottom, left:right]


def benYCC(bgr, weight=4, gamma=20):
        
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0,0), gamma), -weight, 128)

    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    
    return bens 


def benSimple(img, weight=4, gamma=20):
        
    bens = cv2.addWeighted(img, weight, cv2.GaussianBlur(img, (0,0), gamma), -weight, 128)
    
    return bens 


def reflectAndSquareUp(img):
    
    height = img.shape[0]
    width  = img.shape[1]
    
    if (height > width):
        
        offset = int((height - width)/2)
        return img[offset:offset+width]
    
    else:
        if len(img.shape) == 3:
            new_img = np.zeros((width, width, img.shape[2]), np.uint8)
        else:
            new_img = np.zeros((width, width), np.uint8)


        h1 = int((width - height)/2)
        h2 = h1 + height

        new_img[h1:h2,:] = img

        for i in range(h1):
            new_img[h1-i] = img[i]

        for i in range(width - h2):
            new_img[h2+i] = img[height - i - 1]

        return new_img

def circleMask(img):
    
    if (img.shape[0] != img.shape[1]):
        print("Error: circle mask assumes square image")
        return img
    
    dim = img.shape[0]
    half = int(dim/2)
    
    circle_mask = np.zeros((dim, dim), np.uint8)
    circle_mask = cv2.circle(circle_mask, (half, half), half, 1, thickness=-1)

    return cv2.bitwise_and(img, img, mask=circle_mask)

def clahe_gray(gray, clipLimit=3.5, grid=4):

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid,grid))
    return clahe.apply(gray)

def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")

    return cv2.LUT(image, table)

 
def processBenNormal(bgr):
    
    green   = bgr[:,:,1] # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2*IMG_DIM, 2*IMG_DIM))
    circled = circleMask(resized)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(np.median(circled)))
    
    resized_again = cv2.resize(benYCC(equalised), (IMG_DIM, IMG_DIM))
    
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)

    
    

def processBenWeird(bgr):
    
    green   = bgr[:,:,1] # use green as a greyscale
    cropped = crop(green, bgr, 0.02)
    squared = reflectAndSquareUp(cropped)
    resized = cv2.resize(squared, (2*IMG_DIM, 2*IMG_DIM))
    circled = circleMask(resized)
    equalised = adjust_gamma(circled, 1 + np.log(90) - np.log(np.median(circled)))
    
    resized_again = cv2.resize(benSimple(equalised), (IMG_DIM, IMG_DIM))
    
    return cv2.cvtColor(resized_again, cv2.COLOR_BGR2RGB)

## === cell 3

def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(rescale=1./255,
                                       horizontal_flip=True and (jitter > 0.01), 
                                       vertical_flip=True and (jitter > 0.01),
                                       rotation_range=int(800*jitter),
                                       brightness_range=[1-jitter, 1],
                                       channel_shift_range=int(30*jitter),
                                       zoom_range=[(1-jitter), (1+jitter/2)],
                                       fill_mode="reflect",
                                      )
    return datagen


## === cell 6
def create_model(weights):
    
    model = Sequential()
    model.add(DenseNet121(weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    model.load_weights(weights)
    model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy',  metrics=['accuracy'])
    
    return model


## === cell 8
def make_predictions(d_set, processing_function, model, jitters=5):

    images_dir  = f"{INPUT_FOLDER}{d_set}_images/"
    df          = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code  = df.id_code.apply(lambda x: x + ".png")

    block_size  = 512
    total       = df.index.size
    predictions = np.zeros((df.index.size, NUM_CLASSES))
    
    print(f"Making predictions on the {d_set} dataset. Total: {total}")

    for start in range(0, total, block_size):

        end = start + block_size
        if end > total:
            end = total

        img_block = np.empty((end-start, IMG_DIM, IMG_DIM, CHANNELS))
        for i, filename in enumerate(df[start:end].id_code):
            try:
                bgr = cv2.imread(images_dir + filename)
                img_block[i,:,:,:] = processing_function(bgr)
            except:
                print("Error opening or manipulating image")
                img_block[i,:,:,:] = 128.

        prediction_jitters = np.zeros((len(img_block), jitters, NUM_CLASSES))
        jit = 0
        for i in range(jitters):
            datagen = dataGenerator(jit).flow(img_block, shuffle=False)
            prediction_jitters[:, i] = model.predict_generator(generator=datagen, 
                                                               steps=len(datagen), 
                                                               workers=4, verbose=1)
            gc.collect()
            jit += 0.02
        predictions[start:end] = np.median(prediction_jitters, axis=1)

        print(f"{start} - {end} finished")
        gc.collect()
        
    return predictions

## === cell 12
def label_convert(preds):
    y_val = preds > 0.5
    return y_val.astype(int).sum(axis=1) - 1

test_predictions = make_predictions("test", processBenNormal, create_model(NORMAL_WEIGHTS))


test_classes = label_convert(test_predictions)
print(test_predictions[:10])
print(test_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + 'test.csv')
test_df['diagnosis'] = test_classes
test_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/2503565851.py in <cell line: 0>()
      4 
      5 # make test predictions
----> 6 test_predictions = make_predictions("test", processBenNormal, create_model(NORMAL_WEIGHTS))
      7 # weird_preds  = make_predictions("test", processBenWeird, create_model(WEIRD_WEIGHTS))
      8 

/tmp/ipykernel_11/4000824712.py in create_model(weights)
      7     model.add(Dense(NUM_CLASSES, activation='sigmoid'))
      8 
----> 9     model.load_weights(weights)
     10     model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy',  metrics=['accuracy'])
     11 

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '../input/densenetmulti/300_ben_normal_-0.8925.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)
