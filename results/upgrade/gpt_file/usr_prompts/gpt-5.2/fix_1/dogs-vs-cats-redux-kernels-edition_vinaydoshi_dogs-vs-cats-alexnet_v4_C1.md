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

3.8

# 3. Installed packages

No external packages required in the script and installed.

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

1.1773532322739644

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
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.image import extract_patches_2d
import progressbar
import tqdm
from tqdm import tqdm_notebook
import json
import csv
import cv2
import h5py
import matplotlib.pyplot as plt
import seaborn as sns



import os



## === cell 2
import os

image_types = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")


def list_images(basePath, contains=None):
    return list_files(basePath, validExts=image_types, contains=contains)


def list_files(basePath, validExts=None, contains=None):
    for (rootDir, dirNames, filenames) in os.walk(basePath):
        for filename in filenames:
            if contains is not None and filename.find(contains) == -1:
                continue

            ext = filename[filename.rfind("."):].lower()

            if validExts is None or ext.endswith(validExts):
                imagePath = os.path.join(rootDir, filename)
                yield imagePath

def resize(image, width=None, height=None, inter=cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image

    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation=inter)

    return resized

## === cell 4
train_path = '../input/dogs-vs-cats-redux-kernels-edition/train/train/'
final_test_path = '../input/dogs-vs-cats-redux-kernels-edition/test/test/'
train_img_paths = list(list_images(train_path))
final_test_img_paths = list(list_images(final_test_path))

## === cell 5
len(train_img_paths), len(final_test_img_paths)

## === cell 6
os.listdir(train_path),

## === cell 7
NUM_CLASSES=2
NUM_VAL_IMAGES = 1250*NUM_CLASSES
NUM_TEST_IMAGES = 1250*NUM_CLASSES

train_hdf5 = '/kaggle/working/train.hdf5'
val_hdf5 = '/kaggle/working/val.hdf5'
test_hdf5 = '/kaggle/working/test.hdf5'
MODEL_PATH = '/kaggle/working/alexnet_dogs_vs_cats.model'
dataset_mean = '/kaggle/working/dogs_vs_cats_mean.json'
output_path = '/kaggle/working/'

## === cell 8
import cv2

class SimplePreprocessor:
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.inter = inter
    def preprocess(self, image):
        return cv2.resize(image,(self.width, self.height), interpolation = self.inter)

## === cell 9
class AspectAwarePreprocessor:
    
    def __init__(self, width, height, inter = cv2.INTER_AREA):
        self.width = width 
        self.height= height
        self.inter = inter 
        
    def preprocess(self, image):
        (h,w) = image.shape[:2]
        dH, dW = 0, 0
        
        if w < h:
            image = resize(image, width = self.width, inter = self.inter)
            dH = (image.shape[0] - self.height)//2
            
        else:
            image = resize(image, height = self.height, inter = self.inter)
            dW = (image.shape[1] - self.width)//2      
            
        (h,w) = image.shape[:2]
        image = image[dH:h-dH, dW:w-dW]
        
        return cv2.resize(image, (self.width, self.height), interpolation = self.inter)

## === cell 10
class HDF5DatasetWriter:
    def __init__(self, dims, outputPath, dataKey = 'images', bufSize=1000):
        if os.path.exists(outputPath):
            raise ValueError('The supplied "outputPath" already exists. Manually delete the file before continuing.',outputPath)
            
        self.db = h5py.File(outputPath, mode='w')
        self.data = self.db.create_dataset(dataKey, dims, dtype='float')
        self.labels = self.db.create_dataset('labels', (dims[0],), dtype='int')
        
        self.bufSize = bufSize
        self.buffer = {'data':[], 'labels':[]}
        self.idx = 0
        
    def add(self, rows, labels):
        self.buffer['data'].extend(rows)
        self.buffer['labels'].extend(labels)
        
        if len(self.buffer['data']) >= self.bufSize:
            self.flush()
    
    def flush(self):
        i = self.idx + len(self.buffer['data'])
        self.data[self.idx:i] = self.buffer['data']
        self.labels[self.idx:i] = self.buffer['labels']
        self.idx = i
        
        self.buffer = {'data':[], 'labels':[]}
        
    def storeClassLabels(self, classLabels):
        
        dt = h5py.special_dtype(vlen=str)
        labelSet = self.db.create_dataset('label_name', (len(classLabels),), dtype = dt)
        labelSet[:] = classLabels
        
    def close(self):
        if len(self.buffer['data']) > 0:
            self.flush()
        
        self.db.close()

## === cell 11
trainLabels = []
count_rej = 0
for p in train_img_paths:
    if p.split(os.path.sep)[-1].split('.')[-1] == 'jpg':
        trainLabels.append(p.split(os.path.sep)[-1].split('.')[0])
    else:
        count_rej +=1
print(len(trainLabels),'\n', np.unique(trainLabels))

## === cell 12
le = LabelEncoder()
trainLabels = le.fit_transform(trainLabels)
print(len(trainLabels),'\n', np.unique(trainLabels))

## === cell 13
sns.countplot(trainLabels)

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
IndexError                                Traceback (most recent call last)
/tmp/ipykernel_11/1170725970.py in <cell line: 0>()
----> 1 sns.countplot(trainLabels)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in countplot(data, x, y, hue, order, hue_order, orient, color, palette, saturation, width, dodge, ax, **kwargs)
   2941         raise ValueError("Cannot pass values for both `x` and `y`")
   2942 
-> 2943     plotter = _CountPlotter(
   2944         x, y, hue, data, order, hue_order,
   2945         estimator, errorbar, n_boot, units, seed,

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in __init__(self, x, y, hue, data, order, hue_order, estimator, errorbar, n_boot, units, seed, orient, color, palette, saturation, width, errcolor, errwidth, capsize, dodge)
   1528                  errcolor, errwidth, capsize, dodge):
   1529         """Initialize the plotter."""
-> 1530         self.establish_variables(x, y, hue, data, orient,
   1531                                  order, hue_order, units)
   1532         self.establish_colors(color, palette, saturation)

/usr/local/lib/python3.11/dist-packages/seaborn/categorical.py in establish_variables(self, x, y, hue, data, orient, order, hue_order, units)
    484                 if hasattr(data, "shape"):
    485                     if len(data.shape) == 1:
--> 486                         if np.isscalar(data[0]):
    487                             plot_data = [data]
    488                         else:

IndexError: index 0 is out of bounds for axis 0 with size 0

## === cell 14
train_img_paths, test_img_paths, y_train, y_test = train_test_split(train_img_paths, trainLabels,
                                                                    test_size = NUM_TEST_IMAGES, random_state=42,
                                                                    stratify=trainLabels
                                                                   )

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1451134884.py in <cell line: 0>()
----> 1 train_img_paths, test_img_paths, y_train, y_test = train_test_split(train_img_paths, trainLabels,
      2                                                                     test_size = NUM_TEST_IMAGES, random_state=42,
      3                                                                     stratify=trainLabels
      4                                                                    )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in train_test_split(test_size, train_size, random_state, shuffle, stratify, *arrays)
   2560 
   2561     n_samples = _num_samples(arrays[0])
-> 2562     n_train, n_test = _validate_shuffle_split(
   2563         n_samples, test_size, train_size, default_test_size=0.25
   2564     )

/usr/local/lib/python3.11/dist-packages/sklearn/model_selection/_split.py in _validate_shuffle_split(n_samples, test_size, train_size, default_test_size)
   2179         and (test_size <= 0 or test_size >= 1)
   2180     ):
-> 2181         raise ValueError(
   2182             "test_size={0} should be either positive and smaller"
   2183             " than the number of samples {1} or a float in the "

ValueError: test_size=2500 should be either positive and smaller than the number of samples 0 or a float in the (0, 1) range

## === cell 15
train_img_paths, val_img_paths, y_train, y_val = train_test_split(train_img_paths, y_train,
                                                                    test_size = NUM_VAL_IMAGES, random_state=42,
                                                                    stratify=y_train
                                                                   )

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/269532339.py in <cell line: 0>()
----> 1 train_img_paths, val_img_paths, y_train, y_val = train_test_split(train_img_paths, y_train,
      2                                                                     test_size = NUM_VAL_IMAGES, random_state=42,
      3                                                                     stratify=y_train
      4                                                                    )

NameError: name 'y_train' is not defined

## === cell 16
train_dataset, val_dataset, test_dataset=[], [], []

## === cell 17
datasets = [('train',train_img_paths, y_train, train_dataset)]


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4275291504.py in <cell line: 0>()
----> 1 datasets = [('train',train_img_paths, y_train, train_dataset)]
      2 #             ('val',val_img_paths, y_train, val_hdf5),
      3 #             ('test',test_img_paths, y_train, test_hdf5)
      4 #            ]

NameError: name 'y_train' is not defined

## === cell 18
aap = AspectAwarePreprocessor(227, 227)


## === cell 19
R, G, B = [], [], []

## === cell 20
os.chdir('/kaggle/working/')

## === cell 21
for dtype, paths, labels, outputPath in datasets:
    print(dtype)
    for i,(path,label) in tqdm.tqdm(enumerate(zip(paths, labels))):
        image = cv2.imread(path)
        image = aap.preprocess(image)
        
        if dtype == 'train':
            (b,g,r) = cv2.mean(image)[:3]
            R.append(r)
            G.append(g)
            B.append(b)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3693255698.py in <cell line: 0>()
----> 1 for dtype, paths, labels, outputPath in datasets:
      2     print(dtype)
      3 #     writer = HDF5DatasetWriter((len(paths),256,256,3),outputPath)
      4     for i,(path,label) in tqdm.tqdm(enumerate(zip(paths, labels))):
      5         image = cv2.imread(path)

NameError: name 'datasets' is not defined

## === cell 22
B_mean = np.mean(B)
G_mean = np.mean(G)
R_mean = np.mean(R)
B_mean, G_mean, R_mean

## === cell 23
class MeanPreprocessor:
    def __init__(self, rMean, gMean, bMean):
        self.rMean = rMean
        self.gMean = gMean
        self.bMean = bMean
    
    def preprocess(self, image):
        (B, G, R) = cv2.split(image.astype('float32'))
        R -= self.rMean
        G -= self.gMean
        B -= self.bMean
        return cv2.merge([B,G,R])

## === cell 24
class PatchPreprocessor:
    def __init__(self, width, height):
        self.width = width
        self.height = height
    
    def preprocess(self, image):
        (h,w) = image.shape[:2]
        if h <= self.height:
            image = aap.preprocess(image)
        elif w <= self.width:
            image = aap.preprocess(image)
        else:
            image = image
        return extract_patches_2d(image, (self.height, self.width), max_patches=1)[0]

## === cell 25
class CropPreprocessor:
    def __init__(self, height, width, horiz=True, inter = cv2.INTER_AREA):
        self.width = width
        self.height = height
        self.horiz = horiz
        self.inter = inter
    def preprocess(self,image):
        crops = []
        (h,w) = image.shape[:2]
        coords = [[0,0, self.width, self.height],
                  [w-self.width, 0, w,self.height],
                  [w-self.width, h-self.height, w,h],
                  [0, h-self.height, self.width, h]
                 ]
        dW = int(0.5*(w-self.width))
        dH = int(0.5*(h-self.height))
        coords.append([dW, dH, w-dW, h-dH])
        
        for (startX, startY, endX, endY) in coords:
            crop = image[startY:endY, startX:endX]
            crop = cv2.resize(crop, (self.width, self.height), interpolation = self.inter)
            crops.append(crop)
        if self.horiz:
            mirrors = [cv2.flip(c,1) for c in crops]
            crops.extend(mirrors)
        return np.array(crops)

## === cell 26
from keras.utils import np_utils
import numpy as np
import cv2

class HDF5DatasetGenerator:
    def __init__(self, dbPath, batchSize, preprocessors=None, aug=None, binarize=True, classes=2):
        self.dbPath =dbPath
        self.batchSize =batchSize
        self.preprocessors =preprocessors
        self.aug =aug
        self.binarize =binarize
        self.classes =classes
        
        self.db = h5py.File(dbPath)
        self.numImages = self.db['labels'].shape[0]
        
    def generator(passes = np.inf):
        epochs=0
        if epochs < passes:
            for i in np.arange(0, self.numImages, self.batchSize):
                images = self.db['images'][i:i+self.batchSize]
                labels = se;f.db['labels'][i:i+self.batchSize]
                if self.binarize:
                    labels = np_utils.to_categorical(labels, self.classes)
                
                if self.preprocessors is not None:
                    procImages=[]
                    
                    for image in images:
                        for p in self.preprocessors:
                            image = p.preprocess(image)
                            procImages.append(image)
                            
                    images = np.array(procImages)
                
                if self.aug is not None:
                    (images,labels) = next(self.aug.flow(images, labels, batch_size = self.batchSize))
                    yield (images, labels)
                    
        epochs +=1
    
    def close(self):
        sef.db.close()

## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 27
from keras.models import Sequential
from keras.layers import BatchNormalization, Conv2D, MaxPooling2D, Activation, Flatten, Dropout, Dense
from keras.regularizers import l2
from keras import backend as K

class Alexnet:
    def build (width, height, depth, classes, reg = 0.0002):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1
        
        if K.image_data_format() == 'channels_first':
            inputShape = (depth, height, width)
            chanDim = 1
        
        model.add(Conv2D(96, (11,11), strides=(4,4), input_shape = inputShape, padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(256, (5,5), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(384, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(256, (3,3), strides=(1,1), padding='same', kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization(axis=chanDim))
        
        model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2)))
        model.add(Dropout(0.25))
        
        model.add(Flatten())
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(4096, kernel_regularizer=l2(reg)))
        model.add(Activation('relu'))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))
        
        model.add(Dense(classes, activation='softmax', kernel_regularizer=l2(reg)))
        
        return model

## === cell 28
from keras.preprocessing.image import img_to_array

class imageToArrayPreprocessor:
    def __init__(self, dataFormat=None):
        self.dataFormat = dataFormat
    
    def preprocess(self, image):
        return img_to_array(image, data_format = self.dataFormat)

## === cell 29
from keras.preprocessing.image import ImageDataGenerator
aug = ImageDataGenerator(rotation_range = 20, zoom_range=0.15, shear_range=0.15,
                         width_shift_range = 0.2, height_shift_range=0.2,
                         horizontal_flip=True, fill_mode='nearest'
                        )

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
ImportError                               Traceback (most recent call last)
/tmp/ipykernel_11/995110230.py in <cell line: 0>()
----> 1 from keras.preprocessing.image import ImageDataGenerator
      2 aug = ImageDataGenerator(rotation_range = 20, zoom_range=0.15, shear_range=0.15,
      3                          width_shift_range = 0.2, height_shift_range=0.2,
      4                          horizontal_flip=True, fill_mode='nearest'
      5                         )

ImportError: cannot import name 'ImageDataGenerator' from 'keras.preprocessing.image' (/usr/local/lib/python3.11/dist-packages/keras/api/preprocessing/image/__init__.py)

## === cell 30
sp = SimplePreprocessor(227, 227)
pp = PatchPreprocessor(227, 227)
mp = MeanPreprocessor(R_mean, G_mean, B_mean)
iap = imageToArrayPreprocessor()


## === cell 33
def image_data_generator(directory_list, labels, bs = 128, mode='train', binarize=True,preprocessors=None, aug=None,classes=2):
    while True:
        for i in range(0, labels.shape[0], bs):
            images=[]
            imagePaths = directory_list[i:i+bs]
            label_vals = labels[i:i+bs]
            if binarize:
                label_vals = np_utils.to_categorical(label_vals, classes)
            if preprocessors is not None:
                procImages=[]
                for path in imagePaths:
                    image = cv2.imread(path)
                    for p in preprocessors:
                        image = p.preprocess(image)
                    procImages.append(image)
                images = np.array(procImages)
            if aug is not None:
                (images,label_vals) = next(aug.flow(images, label_vals, batch_size = bs))
           
            yield (images, label_vals)

## === cell 35
y_train[:8].shape[0]

## --- ERROR in cell 35, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1857606409.py in <cell line: 0>()
----> 1 y_train[:8].shape[0]

NameError: name 'y_train' is not defined

## === cell 36
train_img_paths[:4], y_train[:4]

## --- ERROR in cell 36, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3311116305.py in <cell line: 0>()
----> 1 train_img_paths[:4], y_train[:4]

NameError: name 'y_train' is not defined

## === cell 37
yield_chk=image_data_generator(train_img_paths[:8], y_train[:8], preprocessors=[pp,mp,iap],aug=aug)
yield_chk#.Generator()

## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2408607322.py in <cell line: 0>()
      1 # train_image_batch,train_label_batch = image_data_generator(train_img_paths[:8], y_train[:8], preprocessors=[pp,mp,iap],aug=aug)
      2 # train_image_batch.shape, len(train_image_batch), train_label_batch
----> 3 yield_chk=image_data_generator(train_img_paths[:8], y_train[:8], preprocessors=[pp,mp,iap],aug=aug)
      4 yield_chk#.Generator()

NameError: name 'y_train' is not defined

## === cell 39

from keras.models import load_model
model = load_model('/kaggle/input/dogs-vs-cats-alexnet-trained-model/alexnet_dogs_vs_cats_model_same_padd.hdf5')

## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/3814510291.py in <cell line: 0>()
      1 from keras.models import load_model
----> 2 model = load_model('/kaggle/input/dogs-vs-cats-alexnet-trained-model/alexnet_dogs_vs_cats_model_same_padd.hdf5')

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

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/dogs-vs-cats-alexnet-trained-model/alexnet_dogs_vs_cats_model_same_padd.hdf5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 42
train_gen = image_data_generator(train_img_paths, y_train, bs=64,preprocessors=[pp,mp,iap],aug=aug)
val_gen = image_data_generator(val_img_paths, y_val, bs=64,preprocessors=[sp,mp,iap])

## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/327398542.py in <cell line: 0>()
----> 1 train_gen = image_data_generator(train_img_paths, y_train, bs=64,preprocessors=[pp,mp,iap],aug=aug)
      2 val_gen = image_data_generator(val_img_paths, y_val, bs=64,preprocessors=[sp,mp,iap])

NameError: name 'y_train' is not defined

## === cell 43
y_train.shape[0]

## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/755117343.py in <cell line: 0>()
----> 1 y_train.shape[0]

NameError: name 'y_train' is not defined

## === cell 49
len(test_img_paths),y_test.shape[0]

## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4166620704.py in <cell line: 0>()
----> 1 len(test_img_paths),y_test.shape[0]

NameError: name 'test_img_paths' is not defined

## === cell 50
def test_data_generator(directory_list, bs=128, mode='test', binarize=False,preprocessors=None, aug=None,classes=2, passes=np.inf):
    epochs=0
    if epochs<passes:
        for i in range(0, len(directory_list), bs):
            images=[]
            imagePaths = directory_list[i:i+bs]
            if binarize:
                label_vals = np_utils.to_categorical(label_vals, classes)
            if preprocessors is not None:
                procImages=[]
                for path in imagePaths:
                    image = cv2.imread(path)
                    for p in preprocessors:
                        image = p.preprocess(image)
                    procImages.append(image)
                images = np.array(procImages)
            if aug is not None:
                (images,label_vals) = next(aug.flow(images, label_vals, batch_size = bs))
           
            yield images
        epochs+=1

## === cell 51
testgen = test_data_generator(test_img_paths, bs=64,preprocessors=[sp,mp,iap])

## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4245204473.py in <cell line: 0>()
----> 1 testgen = test_data_generator(test_img_paths, bs=64,preprocessors=[sp,mp,iap])

NameError: name 'test_img_paths' is not defined

## === cell 52
testgen, y_test.shape

## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/514761286.py in <cell line: 0>()
----> 1 testgen, y_test.shape

NameError: name 'testgen' is not defined

## === cell 54
predictions = model.predict_generator(testgen, steps = y_test.shape[0]//64, max_queue_size = 64*2, verbose=1)

## --- ERROR in cell 54, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1612639927.py in <cell line: 0>()
----> 1 predictions = model.predict_generator(testgen, steps = y_test.shape[0]//64, max_queue_size = 64*2, verbose=1)

NameError: name 'model' is not defined

## === cell 55
def rank5_accuracy(preds, labels):
    rank1=0
    rank5=0
    
    for pred,label in zip(preds, labels):
        pred  = np.argsort(pred)[::-1]
        
        if label in pred[:5]:
            rank5 += 1
        
        if label == pred[0]:
            rank1 += 1
    
    rank5 /= float(len(labels))
    rank1 /= float(len(labels))
    
    return (rank1, rank5)

## === cell 56
(rank1, _) = rank5_accuracy(predictions, y_test)
rank1

## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3105789584.py in <cell line: 0>()
----> 1 (rank1, _) = rank5_accuracy(predictions, y_test)
      2 rank1

NameError: name 'predictions' is not defined

## === cell 57
import pandas as pd
final_predict=[]
cp = CropPreprocessor(227,227)
aap2 = AspectAwarePreprocessor(256,256)
predictions2=[]

## === cell 58
import pyprind
pbar = pyprind.ProgBar(y_test.shape[0])
for i,images in enumerate(test_data_generator(test_img_paths, bs=128,preprocessors=[mp], passes=1)):
    for image in images:
        (h,w)=image.shape[:2]
        if h <= 227:
            image = aap2.preprocess(image)
        elif w <= 227:
            image = aap2.preprocess(image)
        else:
            image = image
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops])
        pred = model.predict(crops)
        predictions2.append(pred.mean(axis=0))
    pbar.update(i)

## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1873381885.py in <cell line: 0>()
----> 1 import pyprind
      2 pbar = pyprind.ProgBar(y_test.shape[0])
      3 for i,images in enumerate(test_data_generator(test_img_paths, bs=128,preprocessors=[mp], passes=1)):
      4     #print(i)
      5     for image in images:

ModuleNotFoundError: No module named 'pyprind'

## === cell 59
(rank1, _) = rank5_accuracy(predictions2, y_test)
rank1

## --- ERROR in cell 59, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4000941437.py in <cell line: 0>()
----> 1 (rank1, _) = rank5_accuracy(predictions2, y_test)
      2 rank1

NameError: name 'y_test' is not defined

## === cell 60
import pyprind
pbar = pyprind.ProgBar(y_test.shape[0])
for i,images in enumerate(test_data_generator(final_test_img_paths, bs=128,preprocessors=[mp], passes=1)):
    for image in images:
        (h,w)=image.shape[:2]
        if h <= 227:
            image = aap2.preprocess(image)
        elif w <= 227:
            image = aap2.preprocess(image)
        else:
            image = image
        crops = cp.preprocess(image)
        crops = np.array([iap.preprocess(c) for c in crops])
        pred = model.predict(crops)
        final_predict.append(pred.mean(axis=0))
    pbar.update(i)

## --- ERROR in cell 60, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_11/1782069765.py in <cell line: 0>()
----> 1 import pyprind
      2 pbar = pyprind.ProgBar(y_test.shape[0])
      3 for i,images in enumerate(test_data_generator(final_test_img_paths, bs=128,preprocessors=[mp], passes=1)):
      4     #print(i)
      5     for image in images:

ModuleNotFoundError: No module named 'pyprind'

## === cell 61
len(final_predict)

## === cell 62
final_label = np.max(final_predict,axis=1)

## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
AxisError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2015810459.py in <cell line: 0>()
----> 1 final_label = np.max(final_predict,axis=1)

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in max(a, axis, out, keepdims, initial, where)
   2808     5
   2809     """
-> 2810     return _wrapreduction(a, np.maximum, 'max', axis, None, out,
   2811                           keepdims=keepdims, initial=initial, where=where)
   2812 

/usr/local/lib/python3.11/dist-packages/numpy/core/fromnumeric.py in _wrapreduction(obj, ufunc, method, axis, dtype, out, **kwargs)
     86                 return reduction(axis=axis, out=out, **passkwargs)
     87 
---> 88     return ufunc.reduce(obj, axis, dtype, out, **passkwargs)
     89 
     90 

AxisError: axis 1 is out of bounds for array of dimension 1

## === cell 63
final_img_names=[i.split(os.path.sep)[-1].split('.jpg')[0] for i in final_test_img_paths]

## === cell 64
submission = pd.DataFrame({'id':final_img_names, 'label':final_label})


## --- ERROR in cell 64, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4260734110.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({'id':final_img_names, 'label':final_label})

NameError: name 'final_label' is not defined

## === cell 65
submission.head()

## --- ERROR in cell 65, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3365464162.py in <cell line: 0>()
----> 1 submission.head()

NameError: name 'submission' is not defined

## === cell 66
submission.to_csv('submission.csv',index=False)

## --- ERROR in cell 66, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3349756476.py in <cell line: 0>()
----> 1 submission.to_csv('submission.csv',index=False)

NameError: name 'submission' is not defined
