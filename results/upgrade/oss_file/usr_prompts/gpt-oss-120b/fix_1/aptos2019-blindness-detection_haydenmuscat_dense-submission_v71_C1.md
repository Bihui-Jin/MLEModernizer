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

0.8924754745779371

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

IMG_DIM       = 256
BATCH_SIZE    = 32
CHANNELS      = 3
NUM_CLASSES   = 5

print(os.listdir("."))
print(os.listdir("../"))
print(os.listdir("../input/"))
print(os.listdir("../input/aptos2019-blindness-detection"))
print(os.listdir("../input/densenetmulti"))

INPUT_FOLDER = '../input/aptos2019-blindness-detection/'


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


def bensYCC(bgr, weight=4, gamma=15):
        
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    y = cv2.addWeighted(y, weight, cv2.GaussianBlur(y, (0,0), gamma), -weight, 128)

    ycc_modified = cv2.merge((y, cr, cb))
    bens = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    
    return bens 


def claheYCC(bgr, clipLimit=5, grid=8):
    
    ycc = cv2.cvtColor(bgr, cv2.COLOR_BGR2YCrCb)
    y, cr, cb = cv2.split(ycc)

    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(grid, grid))
    y = clahe.apply(y)
    y = adjust_gamma(y, 1+np.log(110)-np.log(np.median(y)))

    ycc_modified = cv2.merge((y, cr, cb))
    img = cv2.cvtColor(ycc_modified, cv2.COLOR_YCrCb2BGR)
    
    return img 

def bensSimple(bgr, weight=4, gamma=15):
        
    img = cv2.addWeighted(bgr, weight, cv2.GaussianBlur(bgr, (0,0), gamma), -weight, 128)
    
    return img 


    
    
        
    







    
    
    



def adjust_gamma(image, gamma=1.0):
    invGamma = 1.0 / gamma
    table = np.array([((i / 255.0) ** invGamma) * 255
                     for i in np.arange(0, 256)]).astype("uint8")
    return cv2.LUT(image, table)


def process(bgr, model):
    
    green = bgr[:,:,1] # use green as a greyscale
    
    if bgr.shape != (480, 640, 3):
        
        cropped = crop(green, bgr, 0.02)
        width  = int(cropped.shape[1] * 0.9)
        height = int(width * 480 / 640)
        if height > cropped.shape[0]:
            height = cropped.shape[0] - 2
        h = int((cropped.shape[0] - height) / 2)
        w = int((cropped.shape[1] - width) / 2)

        test_crop = cropped[h:height+h,w:width+w,:]
        
    else:
        test_crop = bgr
    
    if model == "normal":
        colouring_fn = bensYCC
    elif model == "weird":
        colouring_fn = bensSimple
    elif model == "clahe":
        colouring_fn = claheYCC
    else:
        print(f"Error, invalid model type given: {model}")
        
    resized = cv2.resize(test_crop, (IMG_DIM, IMG_DIM), interpolation=cv2.INTER_AREA)
    img     = colouring_fn(resized)
    
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


## === cell 3

def dataGenerator(jitter=0.1):
    datagen = image.ImageDataGenerator(rescale=1./255,
                                       horizontal_flip  = True and (jitter > 0.01), 
                                       vertical_flip    = True and (jitter > 0.01),
                                       zoom_range       = [max(0.8, 1-5*jitter), 1],
                                       rotation_range   = int(600*jitter),
                                       brightness_range = [1-jitter/3, 1+jitter/3],
                                       fill_mode        = "mirror",
                                       channel_shift_range=int(30*jitter),
                                      )
    return datagen


## === cell 4
def test_datagen_plot(processing_function, jitter=0.03):
    
    images_dir  = f"{INPUT_FOLDER}test_images/"
    df          = pd.read_csv(f"{INPUT_FOLDER}test.csv")
    df.id_code  = df.id_code.apply(lambda x: x + ".png")

    img_block = np.empty((100, IMG_DIM, IMG_DIM, CHANNELS))
    j = 1
    for i, filename in enumerate(df[:100].id_code):
        bgr = cv2.imread(images_dir + filename)
        img_block[i,:,:,:] = process(bgr, processing_function)
        if bgr.shape != (480, 640, 3) and j <= 8:
            ax = figure.add_subplot(4,4, j)
            plt.imshow(img_block[i,:,:,:]/255.)
            j += 1
        elif bgr.shape == (480, 640, 3) and j > 8:
            ax = figure.add_subplot(4,4, j)
            plt.imshow(img_block[i,:,:,:]/255.)
            j += 1
            if j > 16: 
                return

    datagen_sample = dataGenerator(jitter).flow(img_block)
    for x in datagen_sample:
        for j in range(16):
            ax = figure.add_subplot(4,4, j+1)
            plt.imshow(x[j])
        break

figure=plt.figure(figsize=(22,20))
test_datagen_plot("clahe")
gc.collect()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2948865691.py in <cell line: 0>()
     30 
     31 figure=plt.figure(figsize=(22,20))
---> 32 test_datagen_plot("clahe")
     33 gc.collect()

/tmp/ipykernel_11/2948865691.py in test_datagen_plot(processing_function, jitter)
      1 def test_datagen_plot(processing_function, jitter=0.03):
      2 
----> 3     images_dir  = f"{INPUT_FOLDER}test_images/"
      4     df          = pd.read_csv(f"{INPUT_FOLDER}test.csv")
      5     df.id_code  = df.id_code.apply(lambda x: x + ".png")

NameError: name 'INPUT_FOLDER' is not defined

## === cell 6
def load_network(network_name):
    
    weights = f"../input/densenetmulti/dense-0.800.h5"
    
    model = Sequential()
    model.add(DenseNet121(weights=None, include_top=False, input_shape=(IMG_DIM, IMG_DIM, CHANNELS)))
    model.add(GlobalAveragePooling2D())
    model.add(Dropout(0.5))
    model.add(Dense(NUM_CLASSES, activation='sigmoid'))
    
    model.load_weights(weights)
    model.compile(optimizer=Adam(lr=0.00005), loss='binary_crossentropy',  metrics=['accuracy'])
    
    return model


## === cell 8
    
    

def prediction_convert_highest(predictions, thresholds):
    thresholded = np.zeros(predictions.shape)
    for i in range(NUM_CLASSES):
        thresholded[:,i] = predictions[:,i] > thresholds[i]
        
    y_val = np.zeros((predictions.shape[0]), dtype=np.int)
    for i in range(predictions.shape[0]):
        for j in range(4, -1, -1):
            if thresholded[i][j]:
                y_val[i] = j
                break
    return y_val

def make_predictions(d_set, models):

    images_dir  = f"{INPUT_FOLDER}{d_set}_images/"
    df          = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
    df.id_code  = df.id_code.apply(lambda x: x + ".png")

    block_size  = 512
    total       = df.index.size
    
    jitter_amounts = [0, 0.01, 0.01, 0.01, 0.02, 0.02, 0.02, 0.05, 0.05, 0.05, 0.2, 0.2, 0.2]
    
    ensemble_predictions = np.zeros((df.index.size, len(jitter_amounts)*len(models), NUM_CLASSES))
    
    for m, model in enumerate(models):
    
        print(f"Making predictions with the {model} model on the {d_set} dataset.")
        neural_net = load_network(model)

        for start in range(0, total, block_size):

            end = start + block_size
            if end > total:
                end = total

            img_block = np.empty((end-start, IMG_DIM, IMG_DIM, CHANNELS))
            for i, filename in enumerate(df[start:end].id_code):
                try:
                    bgr = cv2.imread(images_dir + filename)
                    img_block[i,:,:,:] = process(bgr, model)
                except:
                    print("Error opening or manipulating image")
                    img_block[i,:,:,:] = 128.

            for i, jit in enumerate(jitter_amounts):
                datagen = dataGenerator(jit).flow(img_block, shuffle=False)
                ensemble_predictions[start:end, i + len(models)*m] = neural_net.predict_generator(generator=datagen, 
                                                                      steps=len(datagen), workers=4, verbose=1)
            
            print(f"{start} - {end} finished")
            gc.collect()
        
    return np.median(ensemble_predictions, axis=1)

## === cell 10
def find_best_thresholds(train_predictions):
    
    print("Finding best thresholds...")
    
    prediction_convert = prediction_convert_highest
    
    gc.collect()
    
    train_df = pd.read_csv(f"{INPUT_FOLDER}train.csv")
    y_actual = train_df.diagnosis.astype(int).values
    
    thresholds = [0.5 for i in range(NUM_CLASSES)]
    d_thresh   = 0.25
    
    for sweep in range(5):

        for label in range(5):
            
            currKappa = cohen_kappa_score(y_actual, prediction_convert(train_predictions, thresholds), weights='quadratic')
            
            print(currKappa)
            
            thresholds[label] += d_thresh
            kappaUp = cohen_kappa_score(y_actual, prediction_convert(train_predictions, thresholds), weights='quadratic')

            thresholds[label] -= 2*d_thresh
            kappaDown = cohen_kappa_score(y_actual, prediction_convert(train_predictions, thresholds), weights='quadratic')
            
            thresholds[label] += d_thresh 
            
            if kappaUp > currKappa:
                thresholds[label] += d_thresh
            elif kappaDown > currKappa:
                thresholds[label] -= d_thresh

        d_thresh /= 2
    
    gc.collect()
    return thresholds



## === cell 13
thresholds = [0.5 for i in range(NUM_CLASSES)]

predictions = make_predictions("test", ["normal"])

as_classes = prediction_convert_highest(predictions, thresholds)
print(as_classes[:10])

test_df = pd.read_csv(INPUT_FOLDER + 'test.csv')
test_df['diagnosis'] = as_classes
test_df.to_csv('submission.csv', index=False)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1641531683.py in <cell line: 0>()
      3 
      4 # make test predictions
----> 5 predictions = make_predictions("test", ["normal"])
      6 
      7 as_classes = prediction_convert_highest(predictions, thresholds)

/tmp/ipykernel_11/2153817345.py in make_predictions(d_set, models)
     23 def make_predictions(d_set, models):
     24 
---> 25     images_dir  = f"{INPUT_FOLDER}{d_set}_images/"
     26     df          = pd.read_csv(f"{INPUT_FOLDER}{d_set}.csv")
     27     df.id_code  = df.id_code.apply(lambda x: x + ".png")

NameError: name 'INPUT_FOLDER' is not defined
