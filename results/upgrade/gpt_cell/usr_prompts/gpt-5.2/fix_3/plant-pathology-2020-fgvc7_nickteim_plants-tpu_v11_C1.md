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

3.8

# 3. Installed packages

geopandas==0.14.4
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
protobuf==6.33.0
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

0.54861

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 49
import os
import sys
import subprocess


def _ensure_protobuf_compat():
    try:
        import google.protobuf  # noqa: F401
        from google.protobuf import __version__ as pb_ver
    except Exception:
        pb_ver = None

    def _major(v):
        try:
            return int(str(v).split(".")[0])
        except Exception:
            return None

    if pb_ver is None or (_major(pb_ver) is not None and _major(pb_ver) >= 5):
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", "-q", "protobuf==4.25.3"]
        )
        os.execv(sys.executable, [sys.executable] + sys.argv)


_ensure_protobuf_compat()

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import numpy as np
import pandas as pd
import random, re, math
import tensorflow as tf, tensorflow.keras.backend as K
from tensorflow.keras.layers import Dense
from tensorflow.keras.models import Model
from tensorflow.keras import optimizers
from kaggle_datasets import KaggleDatasets

print(tf.__version__)
print(tf.keras.__version__)


## === cell 50
!pip install efficientnet
import efficientnet.tfkeras as efn


## === cell 52
AUTO = tf.data.experimental.AUTOTUNE #basic Convert a number or string to an integer, or return 0 if no arguments
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()  # TPU detection. No parameters necessary if TPU_NAME environment variable is set. On Kaggle this is always the case.
    print('Running on TPU ', tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy() # default distribution strategy in Tensorflow. Works on CPU and single GPU.

print("REPLICAS: ", strategy.num_replicas_in_sync)


GCS_DS_PATH = KaggleDatasets().get_gcs_path()


## === cell 53
from matplotlib import pyplot as plt

img = plt.imread('../input/plant-pathology-2020-fgvc7/images/Train_0.jpg')
print(img.shape)
plt.imshow(img)


## === cell 54
path='../input/plant-pathology-2020-fgvc7/'
train = pd.read_csv(path + 'train.csv')
test = pd.read_csv(path + 'test.csv')
sub = pd.read_csv(path + 'sample_submission.csv')

train_paths = train.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values #put out datapath on the TPU
test_paths = test.image_id.apply(lambda x: GCS_DS_PATH + '/images/' + x + '.jpg').values #put out datapath on the TPU

train_labels = train.loc[:, 'healthy':].values #you can also use '1' instead of 'healthy' but all it does is taking all the results the 4 different labels


## === cell 55
train #Seeing the training csv-file


## === cell 56
train_labels #all the rows for the labels we want to classify


## === cell 57
nb_classes = 4 #number of labels, this will be used for our output layer
BATCH_SIZE = 8 * strategy.num_replicas_in_sync # this is 8 on TPU v3-8, it is 1 on CPU and GPU #try change it to 16
img_size = 768 #u decide but bigger images take longer to train but higher kvali and smaller images is faster but lower kvali
EPOCHS = 40 #number of training rounds


## === cell 58
x = tf.constant([1.8, 2.2], dtype=tf.float32)
print(type(x))
tf.dtypes.cast(x, tf.int32)  # [1, 2], dtype=tf.int32
print(type(x))


## === cell 59
type(x)


## === cell 60
def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)#get the filename
    image = tf.image.decode_jpeg(bits, channels=3) #channel representing some aspect of information about the image 
    image = tf.cast(image, tf.float32) / 255.0 #tf.cast means = Casts a tensor to a new type. so the images tensor becomes a type float
    image = tf.image.resize(image, image_size) #resize the image for the size given above
    if label is None:  #if there is no label for an image
        return image #jusst return the image
    else:
        return image, label #else return the image with the label
    
def data_augment(image, label=None, seed=2020): 
    image = tf.image.random_flip_left_right(image, seed=seed) #flip randomly images to left or right
    image = tf.image.random_flip_up_down(image, seed=seed) #flip the images randomly up or down
    image=tf.image.adjust_saturation(image, 10)
    image=tf.image.resize_with_crop_or_pad(img, 800, 900)
    
    if label is None:  #if there is no label for an image
        return image#jusst return the image
    else:
        return image, label #else return the image with the label


## === cell 61
saturated = tf.image.adjust_saturation(img, 10)
plt.imshow(saturated)


## === cell 62
plt.imshow(img) #if we look at the before and after picture it is clear to see that illness of the plant is much clear on the saturated image, so we are gonna use it in our data aurgment function above 


## === cell 63
train_dataset = (
    tf.data.Dataset #explaned above nut just a API blok iniziator
    .from_tensor_slices((train_paths, train_labels)) #Creates a `Dataset` whose elements are slices of the given tensors
    .map(decode_image, num_parallel_calls=AUTO) #Maps `map_func` across the elements of this dataset# note here is 'map_func'='decode_image' which returned the labels with the images
    .map(data_augment, num_parallel_calls=AUTO)#note here we use data_augment wich fliped the images for ech element in the dataset
    .repeat() #Repeats this dataset so each original value is seen `count` times. #see exsampel below #  The default behavior (if

    .shuffle(512) #Randomly shuffles the elements of this dataset.#and it randomize by 512 (u set value) each time
    .batch(BATCH_SIZE) 
    .prefetch(AUTO) #Creates a `Dataset` that prefetches elements from this dataset.
    )


## === cell 65
test_dataset = (
    tf.data.Dataset #explaned above nut just a API blok iniziator
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO) 
    .batch(BATCH_SIZE)
)


## === cell 67
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8
def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS: #if epoch is lower then 15
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START #take this learning rate
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:#if epoch is lower then 15+3=18 
        lr = LR_MAX #then take this learning rate
    else: #else if is above 18 then 
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN#take this learning rate
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True) #create a callback to make the functionality possible when we do the actual training. 

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


## === cell 68
def get_model():
    pretrained_model = efn.EfficientNetB7(weights='imagenet', include_top=False, pooling='avg', input_shape=(img_size, img_size, 3))
    pretrained_model.trainable = True
    
    model = tf.keras.Sequential([
        pretrained_model,
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(4, activation='softmax') #4 since there are 4 labels and therefor 4 diferent predictions 
    ])

    return model


## === cell 69
with strategy.scope():
    model = get_model()
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['accuracy'])


## === cell 71
%%time
history=model.fit(
    train_dataset, 
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    callbacks=[lr_callback],
    epochs=16
)


## === cell 72
%%time
probs = model.predict(test_dataset)


## === cell 73
sub.loc[:, 'healthy':] = probs
sub.to_csv('submission.csv', index=False)
sub.head()
