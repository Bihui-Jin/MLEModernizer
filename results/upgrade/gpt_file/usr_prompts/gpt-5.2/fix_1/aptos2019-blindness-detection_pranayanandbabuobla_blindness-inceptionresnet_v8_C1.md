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

3.13

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

0.3338402537284629

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import random
from tqdm import tqdm
from tqdm.keras import TqdmCallback
from sklearn.model_selection import train_test_split

## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2, preprocess_input
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

## === cell 2
import warnings
warnings.filterwarnings("ignore")

## === cell 3
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)
random.seed(SEED)

## === cell 4
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print("GPUs available:")
    for gpu in gpus:
        print(gpu)
else:
    print("No GPU available. Using CPU.")

## === cell 5
TRAIN_IMG_DIR = "../input/aptos2019-blindness-detection/train_images"
TEST_IMG_DIR  = "../input/aptos2019-blindness-detection/test_images"

## === cell 6
train_df = pd.read_csv("../input/aptos2019-blindness-detection/train.csv")
test_df = pd.read_csv("../input/aptos2019-blindness-detection/test.csv")

## === cell 7
def get_image_path(id_code, is_train=True):
    ext = ".png"  # Change extension if needed.
    if is_train:
        return os.path.join(TRAIN_IMG_DIR, id_code + ext)
    else:
        return os.path.join(TEST_IMG_DIR, id_code + ext)

## === cell 8
train_df["filepath"] = train_df["id_code"].apply(lambda x: get_image_path(x, is_train=True))
test_df["filepath"]  = test_df["id_code"].apply(lambda x: get_image_path(x, is_train=False))

## === cell 10
train_transform = ImageDataGenerator(
    horizontal_flip=True,
    brightness_range=(0.8, 1.2),
    rotation_range=180,
    shear_range=20,
    zoom_range=(0.8, 1.2),
    width_shift_range=0.2,
    height_shift_range=0.2,
    fill_mode='reflect',
    rescale=1./255
)

## === cell 12
valid_transform = ImageDataGenerator()

## === cell 13
IMG_SIZE = 299

## === cell 14
def load_and_preprocess_image(path, transform=None):
    image = cv2.imread(path)
    if image is None:
        raise ValueError(f"Image not found at path: {path}")
    
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (IMG_SIZE, IMG_SIZE))
    
    if transform:
        image = transform.random_transform(image)
        image = transform.standardize(image)
    
    image = image.astype(np.float32)
    image = preprocess_input(image)
    return image


## === cell 15
def __getitem__(self, index):
    batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
    batch_df = self.df.iloc[batch_indexes]
    
    images = []
    labels = []
    for _, row in batch_df.iterrows():
        image = load_and_preprocess_image(row["filepath"], transform=self.transform)
        images.append(image)
        if self.is_train:
            label = tf.keras.utils.to_categorical(row["diagnosis"], num_classes=self.num_classes)
            labels.append(label)
    
    images = np.stack(images, axis=0)
    if self.is_train:
        labels = np.stack(labels, axis=0)
        return images, labels
    else:
        return images

## === cell 16
class DataGenerator(tf.keras.utils.Sequence):
    def __init__(self, df, batch_size=32, transform=None, is_train=True, num_classes=5, shuffle=True):
        self.df = df.copy().reset_index(drop=True)
        self.batch_size = batch_size
        self.transform = transform
        self.is_train = is_train
        self.num_classes = num_classes
        self.shuffle = shuffle
        self.indexes = np.arange(len(self.df))
        self.on_epoch_end()
        
    def __len__(self):
        return int(np.ceil(len(self.df) / self.batch_size))
    
    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)
    
    def __getitem__(self, index):
        batch_indexes = self.indexes[index * self.batch_size:(index + 1) * self.batch_size]
        batch_df = self.df.iloc[batch_indexes]
        
        images = []
        labels = []
        for _, row in batch_df.iterrows():
            image = load_and_preprocess_image(row["filepath"], transform=self.transform)
            images.append(image)
            if self.is_train:
                label = tf.keras.utils.to_categorical(row["diagnosis"], num_classes=self.num_classes)
                labels.append(label)
        
        images = np.stack(images, axis=0)
        if self.is_train:
            labels = np.stack(labels, axis=0)
            return images, labels
        else:
            return images

## === cell 17
train_df_split, valid_df_split = train_test_split(train_df, test_size=0.2, random_state=SEED, stratify=train_df["diagnosis"])

## === cell 18
BATCH_SIZE = 32
train_gen = DataGenerator(train_df_split, batch_size=BATCH_SIZE, transform=train_transform, is_train=True)
valid_gen = DataGenerator(valid_df_split, batch_size=BATCH_SIZE, transform=valid_transform, is_train=True)

## === cell 19
weights_path = '/kaggle/input/inceptionresnetv2/keras/default/1/inception_resnet_v2_weights_tf_dim_ordering_tf_kernels_notop.h5'

## === cell 20
base_model = InceptionResNetV2(include_top=False, 
                               weights=weights_path, 
                               input_shape=(IMG_SIZE, IMG_SIZE, 3))

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2861981612.py in <cell line: 0>()
----> 1 base_model = InceptionResNetV2(include_top=False, 
      2                                weights=weights_path,
      3                                input_shape=(IMG_SIZE, IMG_SIZE, 3))

/usr/local/lib/python3.11/dist-packages/keras/src/applications/inception_resnet_v2.py in InceptionResNetV2(include_top, weights, input_tensor, input_shape, pooling, classes, classifier_activation, name)
     96     """
     97     if not (weights in {"imagenet", None} or file_utils.exists(weights)):
---> 98         raise ValueError(
     99             "The `weights` argument should be either "
    100             "`None` (random initialization), `imagenet` "

ValueError: The `weights` argument should be either `None` (random initialization), `imagenet` (pre-training on ImageNet), or the path to the weights file to be loaded.

## === cell 21
for layer in base_model.layers:
    layer.trainable = True

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3397613831.py in <cell line: 0>()
----> 1 for layer in base_model.layers:
      2     layer.trainable = True

NameError: name 'base_model' is not defined

## === cell 22
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(100)(x)
x = Dropout(0.3)(x)
predictions = Dense(5, activation='softmax')(x)
model = Model(inputs=base_model.input, outputs=predictions)

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3753439025.py in <cell line: 0>()
----> 1 x = base_model.output
      2 x = GlobalAveragePooling2D()(x)
      3 x = Dense(100)(x)
      4 x = Dropout(0.3)(x)
      5 predictions = Dense(5, activation='softmax')(x)

NameError: name 'base_model' is not defined

## === cell 23
model.compile(optimizer=Adam(learning_rate=1e-4), 
              loss='categorical_crossentropy', 
              metrics=['accuracy', 'precision', 'recall'])


## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2383680643.py in <cell line: 0>()
----> 1 model.compile(optimizer=Adam(learning_rate=1e-4), 
      2               loss='categorical_crossentropy',
      3               metrics=['accuracy', 'precision', 'recall'])
      4 # model.summary()

NameError: name 'model' is not defined

## === cell 24
checkpoint = ModelCheckpoint("best_model.keras", monitor='val_accuracy', verbose=1, save_best_only=True, mode='max')
earlystop  = EarlyStopping(monitor='val_accuracy', patience=10, verbose=1, mode='max', restore_best_weights=True)
reduce_lr  = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=5, verbose=1, min_lr=1e-7)

## === cell 25
EPOCHS = 1
history = model.fit(
    train_gen,
    epochs=EPOCHS,
    validation_data=valid_gen,
    callbacks=[TqdmCallback(verbose=1), checkpoint, earlystop, reduce_lr],
)

## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/883553960.py in <cell line: 0>()
      1 EPOCHS = 1
----> 2 history = model.fit(
      3     train_gen,
      4     epochs=EPOCHS,
      5     validation_data=valid_gen,

NameError: name 'model' is not defined

## === cell 26
test_gen = DataGenerator(test_df, 
                         batch_size=BATCH_SIZE, 
                         transform=valid_transform, 
                         is_train=False, 
                         shuffle=False)

## === cell 27
preds = model.predict(test_gen, verbose=1)
test_df["diagnosis"] = np.argmax(preds, axis=1)

## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2749509764.py in <cell line: 0>()
----> 1 preds = model.predict(test_gen, verbose=1)
      2 test_df["diagnosis"] = np.argmax(preds, axis=1)

NameError: name 'model' is not defined

## === cell 28
submission_csv = "submission.csv"
test_df[["id_code", "diagnosis"]].to_csv(submission_csv, index=False)
print(f"Submission file saved as {submission_csv}")

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/1515310621.py in <cell line: 0>()
      1 submission_csv = "submission.csv"
----> 2 test_df[["id_code", "diagnosis"]].to_csv(submission_csv, index=False)
      3 print(f"Submission file saved as {submission_csv}")

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4106             if is_iterator(key):
   4107                 key = list(key)
-> 4108             indexer = self.columns._get_indexer_strict(key, "columns")[1]
   4109 
   4110         # take() does not accept boolean indexers

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _get_indexer_strict(self, key, axis_name)
   6198             keyarr, indexer, new_indexer = self._reindex_non_unique(keyarr)
   6199 
-> 6200         self._raise_if_missing(keyarr, indexer, axis_name)
   6201 
   6202         keyarr = self.take(indexer)

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in _raise_if_missing(self, key, indexer, axis_name)
   6250 
   6251             not_found = list(ensure_index(key)[missing_mask.nonzero()[0]].unique())
-> 6252             raise KeyError(f"{not_found} not in index")
   6253 
   6254     @overload

KeyError: "['diagnosis'] not in index"
