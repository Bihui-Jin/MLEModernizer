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

0.8348565227449207

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import math

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


TRAINING = False

## === cell 7
!ls ../input/aptos-2019-densenet121-weights

## === cell 8

train = pd.read_csv('../input/aptos2019-blindness-detection/train.csv')
test = pd.read_csv('../input/aptos2019-blindness-detection/test.csv')

print('Number of train samples: ', train.shape[0])
print('Number of test samples: ', test.shape[0])

## === cell 10
train['id_code'] = train['id_code'].apply(lambda x: str(x) + ".png")
test['id_code'] = test['id_code'].apply(lambda x: str(x) + ".png")
train['diagnosis'] = train['diagnosis'].astype(str)

label_cols = ['lbl_0', 'lbl_1', 'lbl_2', 'lbl_3', 'lbl_4']
label_mat = np.zeros((train.shape[0], len(label_cols)), dtype=np.int32)

for i in range(train.shape[0]):
    for j in range(int(train['diagnosis'][i]) + 1):
        label_mat[i, j] = 1

train = pd.concat([train, pd.DataFrame(label_mat, columns=label_cols)], axis=1)

print(train.head(10))

## === cell 11
import cv2
from tensorflow.keras.preprocessing.image import ImageDataGenerator

IMG_SIZE = 224
NB_CHANNELS = 3
NB_CLASSES = 5     # 0, 1, 2, 3, 4
BATCH_SIZE = 32
TEST_BATCH_SIZE = 1

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 12
"""
crops black parts around the image (intensity is <= tol)
"""
def crop_image(img, tol=10):
    
    def crop_image_1(img):
        mask = img > tol
        return img[np.ix_(mask.any(1), mask.any(0))]
    
    if img.ndim == 2:
        return crop_image_1(img)
    
    elif img.ndim == 3:
        try:
            img_cpy = img.copy()
            h, w, _ = img.shape
            img1 = cv2.resize(crop_image_1(img[:, :, 0]), (w, h))
            img2 = cv2.resize(crop_image_1(img[:, :, 1]), (w, h))
            img3 = cv2.resize(crop_image_1(img[:, :, 2]), (w, h))

            img[:,:,0] = img1
            img[:,:,1] = img2
            img[:,:,2] = img3
            
        except:
            return img_cpy
        
        
        """
        # add edges to img
        
        sobelx = cv2.Sobel(img,cv2.CV_32F,1,0,ksize=5)
        sobely = cv2.Sobel(img,cv2.CV_32F,0,1,ksize=5)
        sobelx = cv2.cvtColor(sobelx, cv2.COLOR_RGB2GRAY)
        sobely = cv2.cvtColor(sobely, cv2.COLOR_RGB2GRAY)
        
        img4 = sobelx ** 2 + sobely ** 2
        
        img4 -= np.min(img4)
        img4 = img4 / np.max(img4)
        
        img4 *= 255.
        img4.astype(np.uint8)
        """
        
        return img


"""
crops black parts and enhances image (Ben Graham's method)
"""
def preprocess_image(img):
    
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = crop_image(img)
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.addWeighted(img, 4, cv2.GaussianBlur(img, (0, 0), IMG_SIZE/10), -4, 128)
    
    return img

## === cell 15
train_datagen = ImageDataGenerator(rescale=1./255,
                                   validation_split=0.2,
                                   horizontal_flip=True,
                                   preprocessing_function=preprocess_image)

train_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory="../input/aptos2019-blindness-detection/train_images/",
    x_col='id_code',
    y_col=label_cols,
    batch_size=BATCH_SIZE,
    class_mode="other",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="training",
    shuffle=True
)

val_gen = train_datagen.flow_from_dataframe(
    dataframe=train,
    directory="../input/aptos2019-blindness-detection/train_images/",
    x_col='id_code',
    y_col=label_cols,
    batch_size=BATCH_SIZE,
    class_mode="other",
    target_size=(IMG_SIZE, IMG_SIZE),
    subset="validation",
    shuffle=True
)


test_datagen = ImageDataGenerator(rescale=1./255,
                                  preprocessing_function=preprocess_image)

test_gen = test_datagen.flow_from_dataframe(  
    dataframe=test,
    directory="../input/aptos2019-blindness-detection/test_images/",
    x_col='id_code',
    batch_size=TEST_BATCH_SIZE,
    class_mode=None,
    target_size=(IMG_SIZE, IMG_SIZE),
    shuffle=False,
)

## === cell 17
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (Input,
                                            GlobalAveragePooling2D,
                                            Dense,
                                            Dropout,
                                            BatchNormalization,
                                            Conv2D,
                                            MaxPooling2D)
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import (CSVLogger,
                                        ModelCheckpoint,
                                        EarlyStopping)
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.densenet import DenseNet121

from sklearn.metrics import cohen_kappa_score


MODEL_NAME = 'densenet121'

NB_WARMUP_EPOCHS = 2
NB_EPOCHS = 30
INITIAL_LR = 1e-3


weights_path_template = os.path.join("../input/aptos-2019-densenet121-weights/", "{}_weights.hdf5")
log_path_template     = os.path.join("logs/", "{}_training_log.csv")

## === cell 18
!mkdir -p weights
!mkdir -p logs

## === cell 21
"""
ResNet50 based model
"""
def get_resnet50(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    base_model = ResNet50(weights='imagenet', include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation='softmax', name='final_output')(x)

    model = Model(inputs, output)

    return model

## === cell 23
def get_densenet121(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    base_model = DenseNet121(weights=None, include_top=False, input_tensor=inputs)

    x = GlobalAveragePooling2D()(base_model.output)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)

    output = Dense(nb_out, activation='sigmoid')(x)

    model = Model(inputs, output)

    return model

## === cell 25
"""
simple CNN (conv1)
"""
def get_conv1(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    x = Conv2D(64, (7, 7), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (7, 7), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(256, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(512, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)
    
    x = Dense(2048, activation='relu')(x)
    x = Dropout(0.5)(x)
    
    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation='softmax', name='final_output')(x)

    model = Model(inputs, output)

    return model

## === cell 27
"""
simple CNN v2 (conv2)
"""
def get_conv2(input_shape, nb_out):
    inputs = Input(shape=input_shape)

    x = Conv2D(32, (7, 7), activation='relu')(inputs)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(64, (5, 5), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = Conv2D(128, (3, 3), activation='relu')(x)
    x = MaxPooling2D((2, 2))(x)
    x = BatchNormalization()(x)
    
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.5)(x)

    x = Dense(1024, activation='relu')(x)
    x = Dropout(0.5)(x)

    output = Dense(nb_out, activation='softmax')(x)

    model = Model(inputs, output)

    return model

## === cell 29
"""
returns model
"""
def get_model(name, input_shape, nb_out):
    
    models = {
        'resnet50':    get_resnet50,
        'conv1':       get_conv1,
        'conv2':       get_conv2,
        'densenet121': get_densenet121
    }
    
    if name not in models:
        print(f"No model named '{name}'")
        return
    
    model = models[name](input_shape, nb_out)
    
    weights_path = weights_path_template.format(name)
    if os.path.isfile(weights_path):
        model.load_weights(weights_path)
        print(f"loaded model weights from {weights_path}")
        
    return model

## === cell 34
"""
trains a ResNet50-based model
"""
def train_resnet50(model, train_generator, val_generator, weights_path, log_path):
    
    
    for i in range(len(model.layers)):
        model.layers[i].trainable = False

    for i in range(-5, 0):
        model.layers[i].trainable = True

    metrics_list = ["accuracy"]
    optimizer = Adam(lr=INITIAL_LR)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)
    
    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1
    )
    
    
    
    train_gen.reset()
    val_gen.reset()
    
    for i in range(len(model.layers)):
        model.layers[i].trainable = True
    
    optimizer = Adam(lr=INITIAL_LR)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)
    
    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_WARMUP_EPOCHS,
        callbacks=[mc, cl],
        verbose=1
    )

## === cell 36
def train_densenet121(model, train_generator, val_generator, weights_path, log_path):
    
    metrics_list = ["accuracy"]
    
    optimizer = Adam(lr=INITIAL_LR)

    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=metrics_list)
    
    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_EPOCHS,
        callbacks=[mc, cl],
        verbose=1
    )

## === cell 38
"""
trains the simple CNN
"""
def train_conv1(model, train_generator, val_generator, weights_path, log_path):
    
    metrics_list = ["accuracy"]
    optimizer = Adam(lr=INITIAL_LR)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)
    
    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_EPOCHS,
        callbacks=[mc, cl],
        verbose=1
    )

## === cell 40
def train_conv2(model, train_generator, val_generator, weights_path, log_path):
        
    metrics_list = ["accuracy"]
    optimizer = Adam(lr=INITIAL_LR)

    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=metrics_list)
    
    mc = ModelCheckpoint(weights_path, monitor='val_loss', save_best_only=True, verbose=1)
    es = EarlyStopping(monitor='val_loss', mode='min', restore_best_weights=True, verbose=1)
    cl = CSVLogger(log_path)

    STEP_SIZE_TRAIN = train_generator.n // train_generator.batch_size
    STEP_SIZE_VAL = val_generator.n // val_generator.batch_size

    model.fit_generator(
        generator=train_generator,
        steps_per_epoch=STEP_SIZE_TRAIN,
        validation_data=val_generator,
        validation_steps=STEP_SIZE_VAL,
        epochs=NB_EPOCHS,
        callbacks=[mc, cl],
        verbose=1
    )

## === cell 42
def train_model(name, input_shape, nb_out, train_generator, val_generator):
    model = get_model(name, input_shape, nb_out)

    trainers = {
        'resnet50':    train_resnet50,
        'conv1':       train_conv1,
        'conv2':       train_conv2,
        'densenet121': train_densenet121
    }

    if name not in trainers:
        print(f"No model named '{name}'")
        return

    trainers[name](model, train_generator, val_generator,
                   weights_path_template.format(name),
                   log_path_template.format(name))

## === cell 43
if TRAINING:
    train_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES, train_gen, val_gen)

## === cell 45
model = get_model(MODEL_NAME, (IMG_SIZE, IMG_SIZE, NB_CHANNELS), NB_CLASSES)
test_gen.reset()

preds = model.predict_generator(test_gen, verbose=1)
preds = preds > 0.5
predictions = preds.astype(int).sum(axis=1) - 1

filenames = test_gen.filenames
results = pd.DataFrame({'id_code': filenames, 'diagnosis': predictions})
results['id_code'] = results['id_code'].map(lambda x: str(x)[:-4])   # remove .png
results.to_csv('submission.csv', index=False)


## --- ERROR in cell 45, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/2884621301.py in <cell line: 0>()
      3 # STEP_SIZE_TEST = math.ceil(test_gen.n / test_gen.batch_size)
      4 
----> 5 preds = model.predict_generator(test_gen, verbose=1)
      6 # predictions = np.argmax(preds, axis=1)
      7 preds = preds > 0.5

AttributeError: 'Functional' object has no attribute 'predict_generator'
