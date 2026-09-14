# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.9

# 2. Installed packages

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
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
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
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

# 3. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 4. Code solution

## === cell 0
import os
import pydicom as dicom
import matplotlib.pylab as plt

search_roots = [
    "/kaggle/data/rsna-miccai-brain-tumor-radiogenomic-classification/train",
    "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train",
    "/kaggle/data/train",
    "/kaggle/input/train",
]

image_path = None
for root in search_roots:
    if os.path.isdir(root):
        for dirpath, _, filenames in os.walk(root):
            for fn in filenames:
                if fn.lower().endswith(".dcm"):
                    image_path = os.path.join(dirpath, fn)
                    break
            if image_path is not None:
                break
    if image_path is not None:
        break

if image_path is None:
    raise FileNotFoundError(
        "No .dcm files found under expected dataset roots: " + ", ".join(search_roots)
    )

ds = dicom.dcmread(image_path)
plt.imshow(ds.pixel_array)


## === cell 1
import numpy as np
import pydicom
def load_dicom(path):
    
    data = pydicom.dcmread(path)
    '''
    Returns the image data as a numpy array.
    '''  
    if np.max(data.pixel_array)==0:
        img = data.pixel_array
    else:
        img = data.pixel_array/np.max(data.pixel_array)
        img = (img * 255).astype(np.uint8)
        
    return img


## === cell 2
import pandas as pd
from pandas import ExcelWriter
from pandas import ExcelFile

df = pd.read_csv('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv')
data = df.set_index("BraTS21ID")
data = data.drop([109, 123, 709], axis=0)
df=data.reset_index()


## === cell 3
import numpy as np
labels=np.array(df['MGMT_value'])


## === cell 4
from glob import glob
imagePatches = glob('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/*/', recursive=True)


## === cell 5
import os
files=[]
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00109/')
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00709/')
files.append ('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/train/00123/')
for file in files:
    imagePatches.remove(file)


## === cell 6
len(imagePatches)


## === cell 7
import re

def atoi(text):
    return int(text) if text.isdigit() else text

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    '''
    return [ atoi(c) for c in re.split(r'(\d+)', text) ]


## === cell 8
imagePatches.sort(key=natural_keys)
flair_patches = []
t1w_patches=[]
t1wce_patches=[]
t2w_patches=[]
for subfolder in imagePatches:
    flair_patches.append(glob(subfolder + 'FLAIR/**/**/*.dcm', recursive=True))
    t1w_patches.append(glob(subfolder + 'T1w/**/**/*.dcm', recursive=True))
    t1wce_patches.append(glob(subfolder + 'T1wCE/**/**/*.dcm', recursive=True))
    t2w_patches.append(glob(subfolder + 'T2w/**/**/*.dcm', recursive=True))


## === cell 9
def all_slice(sequence, list_name):
    for x in range(0,len(sequence)):
        sequence[x].sort(key=natural_keys)
        list_name.append((sequence[x][0:len(sequence[x])]))
    return list_name


## === cell 10
all_flair_patches = []
all_slice(flair_patches,all_flair_patches)
all_t1w_patches =[]
all_t1w_patches = all_slice(t1w_patches,all_t1w_patches)
all_t1wce_patches=[]
all_t1wce_patches = all_slice(t1wce_patches,all_t1wce_patches)
all_t2w_patches=[]
all_t2w_patches = all_slice(t2w_patches,all_t2w_patches)


## === cell 11
import cv2
def create_input(patches):
    inputs=np.zeros((582,256,256,3))
    for i in range(0,len(patches)):
        for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
            img= load_dicom(patches[i][j])
            image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, -1)
        inputs[i]=image_array
    return inputs


## === cell 12
import cv2


def create_input(patches):
    inputs = np.zeros((len(patches), 256, 256, 3), dtype=np.float32)
    for i in range(0, len(patches)):
        n = len(patches[i])
        if n == 0:
            continue

        start = n // 2
        end = min(start + 3, n)  # exclusive end; prevents out-of-range
        image_array = None

        for j in range(start, end):
            img = load_dicom(patches[i][j])
            image_array = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, -1)

        if image_array is None:
            continue

        inputs[i] = image_array
    return inputs


## === cell 13
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

try:
    from google.protobuf import message_factory as _message_factory

    MessageFactory = getattr(_message_factory, "MessageFactory", None)
    if MessageFactory is not None and not hasattr(MessageFactory, "GetPrototype"):
        _get_message_class = getattr(_message_factory, "GetMessageClass", None)
        if callable(_get_message_class):

            def _GetPrototype(self, descriptor):
                return _get_message_class(descriptor)

            try:
                setattr(MessageFactory, "GetPrototype", _GetPrototype)
            except Exception:
                pass
except Exception:
    pass

import numpy as np
import pandas as pd
import random
import cv2
import matplotlib.pyplot as plt

get_ipython().run_line_magic("matplotlib", "inline")
import keras

import keras.backend as K
from keras.models import Model, Sequential
from keras.layers import Input, Dense, Flatten, Dropout, BatchNormalization
from keras.layers import (
    Conv2D,
    SeparableConv2D,
    MaxPool2D,
    LeakyReLU,
    Activation,
    GlobalAveragePooling2D,
)
from keras.optimizers import Adam, Adamax, Adagrad

try:
    from keras.preprocessing.image import ImageDataGenerator
except ImportError:
    from tensorflow.keras.preprocessing.image import ImageDataGenerator

from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.optimizers import SGD

try:
    from keras.utils import to_categorical
except Exception:
    from tensorflow.keras.utils import to_categorical


## === cell 14
flair_inputs = create_input(all_flair_patches)
t1w_inputs = create_input(all_t1w_patches)
t1wce_inputs = create_input(all_t1wce_patches)
t2w_inputs = create_input(all_t2w_patches)

flair_inputs = np.asarray(flair_inputs)
t1w_inputs = np.asarray(t1w_inputs)
t1wce_inputs = np.asarray(t1wce_inputs)
t2w_inputs = np.asarray(t2w_inputs)


## === cell 15
from sklearn.model_selection import train_test_split
from IPython.display import Image
from sklearn.metrics import confusion_matrix
np.random.seed(1)


## === cell 16
from tensorflow.keras import layers
def get_model(optimizer):
    """Build a 3D convolutional neural network model."""

    inputs = keras.Input((256,256,3,1))
    
    

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)


    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu",padding='same')(inputs)
    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.BatchNormalization()(x)

    
    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    
    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    model.compile(loss='binary_crossentropy',
                  optimizer=optimizer, metrics=['accuracy'])
    return model


model = get_model('Adam')


## === cell 17
reduce_lr = ReduceLROnPlateau(
    monitor="val_loss", factor=0.3, patience=3, mode="auto", verbose=1
)
callbacks = [
    ModelCheckpoint(
        filepath="/kaggle/output/models/best_model.h5",
        monitor="val_loss",
        save_best_only=True,
    )
]

n = min(len(flair_inputs), len(labels))
flair_inputs = flair_inputs[:n]
labels = labels[:n]

x_train, x_valid, y_train, y_valid = train_test_split(
    flair_inputs, labels, test_size=0.2, random_state=1
)
optimizers = ["SGD", "RMSprop", "Adagrad", "Adadelta", "Adam", "Adamax", "Nadam"]
early_stop = EarlyStopping(monitor="val_loss", min_delta=0.1, patience=3, mode="min")
x_train = x_train / 255
x_valid = x_valid / 255
model = get_model("Adam")
history = model.fit(
    x_train,
    y_train,
    epochs=50,
    batch_size=8,
    shuffle=True,
    validation_data=(x_valid, y_valid),
    callbacks=[early_stop],
)


## === cell 18
model.save('/kaggle/t1winputs.hdf5')
model_t1w= keras.models.load_model('/kaggle/t1winputs.hdf5')


## === cell 19
model.save('/kaggle/t1wceinputs.hdf5')
model_t1wce= keras.models.load_model('/kaggle/t1wceinputs.hdf5')


## === cell 20
model.save('/kaggle/t2winputs.hdf5')
model_t2w= keras.models.load_model('/kaggle/t2winputs.hdf5')


## === cell 21
model.save('/kaggle/flairinputs.hdf5')
model_flair= keras.models.load_model('/kaggle/flairinputs.hdf5')


## === cell 22
preds_flair=model_flair.predict(x_valid)
preds_t1w=model_t1w.predict(x_valid)
preds_t2w=model_t2w.predict(x_valid)
preds_t1wce=model_t1wce.predict(x_valid)


## === cell 23
mean = [(g + h + a + b) / 4 for g, h, a, b in zip(preds_flair,preds_t1w, preds_t2w,preds_t1wce)]


## === cell 24
from sklearn.metrics import roc_curve,roc_auc_score

fpr , tpr , thresholds = roc_curve ( y_valid , mean)


## === cell 25
def plot_roc_curve(fpr,tpr): 
  plt.plot(fpr,tpr) 
  plt.axis([0,1,0,1]) 
  plt.xlabel('False Positive Rate') 
  plt.ylabel('True Positive Rate') 
  plt.show()    
  
plot_roc_curve (fpr,tpr) 


## === cell 26
auc_score=roc_auc_score(y_valid,mean)
print(auc_score)


## === cell 27
model_t1w= keras.models.load_model('/kaggle/t1winputs.hdf5')
model_t1wce= keras.models.load_model('/kaggle/t1wceinputs.hdf5')
model_t2w= keras.models.load_model('/kaggle/t2winputs.hdf5')
model_flair= keras.models.load_model('/kaggle/flairinputs.hdf5')


## === cell 28
import cv2
def create_input(patches, number_image):
    inputs=np.zeros((number_image,256,256,3))
    for i in range(0,len(patches)):
        for j in range(len(patches[i])//2,(len(patches[i])//2)+3):
            img= load_dicom(patches[i][j])
            image_array = cv2.resize(img, (256,256), interpolation=cv2.INTER_AREA)
            image_array = np.expand_dims(image_array, -1)
        inputs[i]=image_array
    return inputs


## === cell 29
testimages= glob('/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test/*/', recursive=True)
testimages.sort(key=natural_keys)
flair_patches = []
t1w_patches=[]
t1wce_patches=[]
t2w_patches=[]
for subfolder in testimages:
    flair_patches.append(glob(subfolder + 'FLAIR/**/**/*.dcm', recursive=True))
    t1w_patches.append(glob(subfolder + 'T1w/**/**/*.dcm', recursive=True))
    t1wce_patches.append(glob(subfolder + 'T1wCE/**/**/*.dcm', recursive=True))
    t2w_patches.append(glob(subfolder + 'T2w/**/**/*.dcm', recursive=True))
all_flair_patches = []
all_slice(flair_patches,all_flair_patches)
all_t1w_patches =[]
all_t1w_patches = all_slice(t1w_patches,all_t1w_patches)
all_t1wce_patches=[]
all_t1wce_patches = all_slice(t1wce_patches,all_t1wce_patches)
all_t2w_patches=[]
all_t2w_patches = all_slice(t2w_patches,all_t2w_patches)
t2w_inputs = create_input(all_t2w_patches, len(testimages))
flair_inputs = create_input(all_flair_patches,len(testimages))
t1wce_inputs = create_input(all_t1wce_patches,len(testimages))
t1w_inputs = create_input(all_t1w_patches,len(testimages))
testflair_inputs= np.asarray(flair_inputs)/255
testt1w_inputs= np.asarray(t1w_inputs)/255
testt1wce_inputs= np.asarray(t1wce_inputs)/255
testt2w_inputs = np.asarray(t2w_inputs)/255
preds_flair=model_flair.predict(testflair_inputs)
preds_t1w=model_t1w.predict(testt1w_inputs)
preds_t2w=model_t2w.predict(testt2w_inputs)
preds_t1wce=model_t1wce.predict(testt1wce_inputs)
mean = [(g + h + a + b) / 4 for g, h, a, b in zip(preds_flair,preds_t1w, preds_t2w,preds_t1wce)]


## --- ERROR in cell 29, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mIndexError[0m                                Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/592727933.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     18[0m [0mall_t2w_patches[0m[0;34m=[0m[0;34m[[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[1;32m     19[0m [0mall_t2w_patches[0m [0;34m=[0m [0mall_slice[0m[0;34m([0m[0mt2w_patches[0m[0;34m,[0m[0mall_t2w_patches[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 20[0;31m [0mt2w_inputs[0m [0;34m=[0m [0mcreate_input[0m[0;34m([0m[0mall_t2w_patches[0m[0;34m,[0m [0mlen[0m[0;34m([0m[0mtestimages[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     21[0m [0mflair_inputs[0m [0;34m=[0m [0mcreate_input[0m[0;34m([0m[0mall_flair_patches[0m[0;34m,[0m[0mlen[0m[0;34m([0m[0mtestimages[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m     22[0m [0mt1wce_inputs[0m [0;34m=[0m [0mcreate_input[0m[0;34m([0m[0mall_t1wce_patches[0m[0;34m,[0m[0mlen[0m[0;34m([0m[0mtestimages[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/175905925.py[0m in [0;36mcreate_input[0;34m(patches, number_image)[0m
[1;32m      4[0m     [0;32mfor[0m [0mi[0m [0;32min[0m [0mrange[0m[0;34m([0m[0;36m0[0m[0;34m,[0m[0mlen[0m[0;34m([0m[0mpatches[0m[0;34m)[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m      5[0m         [0;32mfor[0m [0mj[0m [0;32min[0m [0mrange[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mpatches[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m[0;34m//[0m[0;36m2[0m[0;34m,[0m[0;34m([0m[0mlen[0m[0;34m([0m[0mpatches[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m)[0m[0;34m//[0m[0;36m2[0m[0;34m)[0m[0;34m+[0m[0;36m3[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m             [0mimg[0m[0;34m=[0m [0mload_dicom[0m[0;34m([0m[0mpatches[0m[0;34m[[0m[0mi[0m[0;34m][0m[0;34m[[0m[0mj[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m      7[0m             [0mimage_array[0m [0;34m=[0m [0mcv2[0m[0;34m.[0m[0mresize[0m[0;34m([0m[0mimg[0m[0;34m,[0m [0;34m([0m[0;36m256[0m[0;34m,[0m[0;36m256[0m[0;34m)[0m[0;34m,[0m [0minterpolation[0m[0;34m=[0m[0mcv2[0m[0;34m.[0m[0mINTER_AREA[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m             [0mimage_array[0m [0;34m=[0m [0mnp[0m[0;34m.[0m[0mexpand_dims[0m[0;34m([0m[0mimage_array[0m[0;34m,[0m [0;34m-[0m[0;36m1[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mIndexError[0m: list index out of range

## === cell 30
sample=pd.read_csv("/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv",index_col="BraTS21ID")
sample["MGMT_value"] = 0
sample["MGMT_value"] = [str(a)[1:-1] for a in mean]
sample["MGMT_value"].to_csv("submission.csv")
