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
Detect apple diseases from images.

## Metric
Mean column-wise ROC AUC.

## Submission Format
For each image_id in the test set, you must predict a probability for each target variable. The file should contain a header and have the following format:

```
image_id,
test_0,0.25,0.25,0.25,0.25
test_1,0.25,0.25,0.25,0.25
test_2,0.25,0.25,0.25,0.25
etc.
```

## Dataset
Given a photo of an apple leaf, can you accurately assess its health? This competition will challenge you to distinguish between leaves which are healthy, those which are infected with apple rust, those that have apple scab, and those with more than one disease.

**train.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

**images**

A folder containing the train and test images, in jpg format.

**test.csv**

- `image_id`: the foreign key

**sample_submission.csv**

- `image_id`: the foreign key
- combinations: one of the target labels
- healthy: one of the target labels
- rust: one of the target labels
- scab: one of the target labels

# 2. Python version

3.9

# 3. Installed packages

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
plotly==5.24.1
plotly-express==0.4.1
protobuf==6.33.0
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
scipy==1.15.3
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
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        input/
            description.md (94 lines)
            images.zip (397.8 MB)
            sample_submission.csv (184 lines)
            sample_submission.csv.zip (682 Bytes)
            test.csv (184 lines)
            test.csv.zip (542 Bytes)
            train.csv (1639 lines)
            train.csv.zip (4.6 kB)
            images/
                Train_370.jpg (133.2 kB)
                Test_59.jpg (220.5 kB)
                ... and 1819 other files
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
        working/
            plant-pathology-2020-fgvc7/
                description.md (94 lines)
                images.zip (397.8 MB)
                ... and 6 other files
                images/
                    Train_370.jpg (133.2 kB)
                    Test_59.jpg (220.5 kB)
                    ... and 1819 other files
                plant-pathology-2020-fgvc7/
```

-> data/plant-pathology-2020-fgvc7/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/plant-pathology-2020-fgvc7/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/plant-pathology-2020-fgvc7/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/sample_submission.csv has 183 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> data/test.csv has 183 rows and 1 columns.
The columns are: image_id

-> data/train.csv has 1638 rows and 5 columns.
The columns are: image_id, healthy, multiple_diseases, rust, scab

-> (stopped after 10 files for performance)

# 5. Target score

0.95978

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)

import os
import gc
import re

import cv2
import math
import scipy as sp

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None

import os
from keras.applications.vgg16 import VGG16
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    Activation,
    MaxPooling2D,
    Flatten,
    Conv2D,
    BatchNormalization,
)
from tensorflow.keras import Sequential
from tensorflow.keras import layers
import tensorflow as tf
from keras import models
import tensorflow as tf
from IPython.display import SVG
from keras.utils import plot_model
import sklearn
from tensorflow import keras
import tensorflow.keras.layers as L
from keras.utils import model_to_dot
import tensorflow.keras.backend as K
from tensorflow.keras.models import Model

try:
    from kaggle_datasets import KaggleDatasets
except Exception:
    KaggleDatasets = None

from tensorflow.keras.applications import DenseNet121

import seaborn as sns

sns.set_style("darkgrid")
from tqdm import tqdm
import matplotlib.cm as cm
from sklearn import metrics
import matplotlib.pyplot as plt

plt.style.use("ggplot")
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split

tqdm.pandas()
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
from plotly.subplots import make_subplots

np.random.seed(0)
tf.random.set_seed(0)

import warnings

warnings.filterwarnings("ignore")


import os

for dirname, _, filenames in os.walk("/kaggle/input"):
    for filename in filenames:
        print(os.path.join(dirname, filename))


## === cell 1
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train.head()


## === cell 2
train.head()


## === cell 3
test.head()


## === cell 4
%%time
SAMPLE_LEN = 100
def load_image(image_id):
    file_path = image_id + ".jpg"
    image = cv2.imread(IMAGE_PATH + file_path)
    return cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

train_images = train["image_id"][:SAMPLE_LEN].progress_apply(load_image)


## === cell 5
fig = px.imshow(cv2.resize(train_images[0], (205, 136)))
fig.show()


## === cell 6
def show_images_in(path, n=16):
    files_to_show = os.listdir(path)[:n]
    assert files_to_show[0].endswith((".jpg",".jpeg",".png"))
    np.random.shuffle(files_to_show)
    img_paths = [os.path.join(path,file) for file in files_to_show]
    plt.figure(figsize=(12,12))
    for i in range(n):
        img = plt.imread(img_paths[i])
        plt.subplot(4,4,i+1)
        plt.imshow(img)
    plt.tight_layout()
show_images_in("../input/plant-pathology-2020-fgvc7/images/",n=16)


## === cell 7
s.head()


## === cell 8
train_df=pd.read_csv('../input/plant-pathology-2020-fgvc7/train.csv')
test_df=pd.read_csv('../input/plant-pathology-2020-fgvc7/test.csv')
samp=pd.read_csv('../input/plant-pathology-2020-fgvc7/sample_submission.csv')


## === cell 9
X_paths = [(os.path.join('../input/plant-pathology-2020-fgvc7/images',i+'.jpg')) for i in train_df.image_id]
train_df=train_df.drop(['image_id'],axis=1)
y_train = train_df.to_numpy().astype('float32')


## === cell 10
IMAGE_SIZE=100
images=[]
import os
for i in range(len(X_paths)):
    img = cv2.imread(os.path.join(X_paths[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    images.append(img)#


## === cell 11
import matplotlib.pyplot as plt
plt.imshow(images[0])
print(y_train[0])


## === cell 12
X_X = np.array(images).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)


## === cell 13
vgg= VGG16(include_top=False,pooling='avg',weights='imagenet',input_shape=(IMAGE_SIZE,IMAGE_SIZE,3))


## === cell 14
data_augmentation = tf.keras.Sequential([
  layers.experimental.preprocessing.RandomFlip(),
  layers.experimental.preprocessing.RandomRotation(0.5),
])


## === cell 15
model=Sequential()
model.add(data_augmentation)

model.add(vgg)

model.add(Dense(4))
model.add(BatchNormalization())
model.add(Activation('softmax'))

for layer in vgg.layers[:-12]:
    layer.trainable = False
    
for layer in vgg.layers:
    print(layer, layer.trainable)
 
 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


## === cell 16
X = np.array(X_X)
Y=np.array(y_train)
modeL=model.fit(X,Y,batch_size=64,epochs=250,validation_split=0.3)


## === cell 17
plt.title('model accuracy')
plt.plot(modeL.history['val_accuracy'])
plt.plot(modeL.history['accuracy'])
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['val', 'train'], loc='upper left')
plt.show()


## === cell 18
X_test_paths = [(os.path.join('../input/plant-pathology-2020-fgvc7/images',i+'.jpg')) for i in test_df.image_id]


## === cell 19
X_test_paths
images_test=[]
import os
for i in range(len(X_test_paths)):
    img = cv2.imread(os.path.join(X_test_paths[i]))
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (IMAGE_SIZE,IMAGE_SIZE))
    images_test.append(img)#


## === cell 20
X_T = np.array(images_test).reshape(-1, IMAGE_SIZE, IMAGE_SIZE, 3)


## === cell 21
y_pred = model.predict(np.array(X_T))


## === cell 22
temp=test_df.image_id


## === cell 23
final_sample=pd.DataFrame((temp))


## === cell 24
final_sample['healthy']=y_pred[:,0:1]
final_sample['multiple_diseases']=y_pred[:,1:2]
final_sample['rust']=y_pred[:,2:3]
final_sample['scab']=y_pred[:,3:4]


## === cell 25
final_sample.head()


## === cell 26
final_sample.to_csv('submission.csv',index=False)


## === cell 29
!pip install -q efficientnet


## === cell 30
import numpy as np
import pandas as pd
import os
import tensorflow as tf
import keras
import efficientnet.tfkeras as efn
from kaggle_datasets import KaggleDatasets
from tensorflow.keras.models import Sequential
import tensorflow.keras.layers as L
from tensorflow.keras.applications import InceptionResNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt


## === cell 31
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()
    
def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

seed = 2048
seed_everything(seed)
print("REPLICAS: ", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path()

EPOCHS = 40
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUTO = tf.data.experimental.AUTOTUNE


## === cell 32
def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'


## === cell 33
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
image_size = 800
train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values
train_labels = train.loc[:, 'healthy':].values
SPLIT_VALIDATION =True
if SPLIT_VALIDATION:
    train_paths, valid_paths, train_labels, valid_labels =train_test_split(train_paths, train_labels, test_size=0.15, random_state=seed)


## === cell 34
def decode_image(filename, label=None, image_size=(image_size, image_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    
    if label is None:
        return image
    else:
        return image, label


## === cell 35
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)
train_dataset_1 = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(64)
    .prefetch(AUTO)
)
valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


## === cell 36
LR_START = 0.0001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.0001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 6
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


## === cell 37
with strategy.scope():
    model = tf.keras.Sequential([InceptionResNetV2(
        input_shape=(image_size, image_size, 3),
        weights='imagenet',
        include_top=False),
        L.GlobalMaxPooling2D(),
        L.Dense(4, activation='softmax')])
        
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics=['categorical_accuracy'])
    model.summary()


## === cell 38
STEPS_PER_EPOCH = train_labels.shape[0] // BATCH_SIZE
history = model.fit(train_dataset, epochs=EPOCHS, callbacks=[lr_callback],steps_per_epoch=STEPS_PER_EPOCH,validation_data=valid_dataset if SPLIT_VALIDATION else None,)


## === cell 39
def display_training_curves(training, validation, title, subplot):
    """
    Source: https://www.kaggle.com/mgornergoogle/getting-started-with-100-flowers-on-tpu
    """
    if subplot%10==1: # set up the subplots on the first call
        plt.subplots(figsize=(10,10), facecolor='#F0F0F0')
        plt.tight_layout()
    ax = plt.subplot(subplot)
    ax.set_facecolor('#F8F8F8')
    ax.plot(training)
    ax.plot(validation)
    ax.set_title('model '+ title)
    ax.set_ylabel(title)
    ax.set_xlabel('epoch')
    ax.legend(['train', 'valid.'])


## === cell 40
display_training_curves(
    history.history['loss'], 
    history.history['val_loss'], 
    'loss', 211)
display_training_curves(
    history.history['categorical_accuracy'], 
    history.history['val_categorical_accuracy'], 
    'accuracy', 212)


## === cell 41
probs1 = model.predict(test_dataset, verbose=1)
sub.loc[:, 'healthy':] = probs1
sub.to_csv('submission.csv', index=False)
sub.head()
