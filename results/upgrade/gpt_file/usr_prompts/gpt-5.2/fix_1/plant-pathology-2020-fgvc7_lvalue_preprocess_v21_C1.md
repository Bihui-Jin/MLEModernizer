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

0.49073

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import cv2
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
tqdm.pandas()
import os
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import ImageDataGenerator
!pip install efficientnet pandarallel
import efficientnet.tfkeras as efn 
import tensorflow as tf
import tensorflow.keras as keras
from pandarallel import pandarallel
pandarallel.initialize(progress_bar=True)


## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
USE_TPU = 'TPU_NAME' in os.environ
if USE_TPU:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)

    strategy = tf.distribute.experimental.TPUStrategy(tpu)
    tf.compat.v1.enable_eager_execution()
else:
    strategy = tf.distribute.MirroredStrategy()


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2953329165.py in <cell line: 0>()
     10     tf.compat.v1.enable_eager_execution()
     11 else:
---> 12     strategy = tf.distribute.MirroredStrategy()

NameError: name 'tf' is not defined

## === cell 2
IMAGE_PATH = "../input/plant-pathology-2020-fgvc7/images/"
TEST_PATH = "../input/plant-pathology-2020-fgvc7/test.csv"
TRAIN_PATH = "../input/plant-pathology-2020-fgvc7/train.csv"
SUB_PATH = "../input/plant-pathology-2020-fgvc7/sample_submission.csv"

sub = pd.read_csv(SUB_PATH)
test_data = pd.read_csv(TEST_PATH)
train_data = pd.read_csv(TRAIN_PATH)


## === cell 3
def init_grabcut_mask(h, w):
    mask = np.ones((h, w), np.uint8) * cv2.GC_PR_BGD
    mask[h//4:3*h//4, w//4:3*w//4] = cv2.GC_PR_FGD
    mask[2*h//5:3*h//5, 2*w//5:3*w//5] = cv2.GC_FGD
    return mask


def remove_background(image, h=136, w=205):
    orig_image = image
    image = cv2.resize(image, (w, h))
    mask = init_grabcut_mask(h, w)
    bgm = np.zeros((1, 65), np.float64)
    fgm = np.zeros((1, 65), np.float64)
    cv2.grabCut(image, mask, None, bgm, fgm, 1, cv2.GC_INIT_WITH_MASK)
    mask_binary = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
    h, w = orig_image.shape[:2]
    mask_binary = cv2.resize(mask_binary, (w, h))
    result = cv2.bitwise_and(orig_image, orig_image, mask=mask_binary)
    return result


## === cell 4
def rotate(x: tf.Tensor) -> tf.Tensor:
    """Rotation augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    return tf.image.rot90(x, tf.random.uniform(shape=[], minval=0, maxval=4, dtype=tf.int32))


def flip(x: tf.Tensor) -> tf.Tensor:
    """Flip augmentation

    Args:
        x: Image to flip

    Returns:
        Augmented image
    """
    x = tf.image.random_flip_left_right(x)
    x = tf.image.random_flip_up_down(x)

    return x


def color(x: tf.Tensor) -> tf.Tensor:
    """Color augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """
    x = tf.image.random_hue(x, 0.08)
    x = tf.image.random_saturation(x, 0.6, 1.6)
    x = tf.image.random_brightness(x, 0.05)
    x = tf.image.random_contrast(x, 0.7, 1.3)
    return x


def zoom(x: tf.Tensor) -> tf.Tensor:
    """Zoom augmentation

    Args:
        x: Image

    Returns:
        Augmented image
    """

    scales = list(np.arange(0.8, 1.0, 0.01))
    boxes = np.zeros((len(scales), 4))

    for i, scale in enumerate(scales):
        x1 = y1 = 0.5 - (0.5 * scale)
        x2 = y2 = 0.5 + (0.5 * scale)
        boxes[i] = [x1, y1, x2, y2]

    def random_crop(img):
        crops = tf.image.crop_and_resize([img], boxes=boxes, box_indices=np.zeros(len(scales)), crop_size=(32, 32))
        return crops[tf.random.uniform(shape=[], minval=0, maxval=len(scales), dtype=tf.int32)]


    choice = tf.random.uniform(shape=[], minval=0., maxval=1., dtype=tf.float32)

    return tf.cond(choice < 0.5, lambda: x, lambda: random_crop(x))


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1958823466.py in <cell line: 0>()
----> 1 def rotate(x: tf.Tensor) -> tf.Tensor:
      2     """Rotation augmentation
      3 
      4     Args:
      5         x: Image

NameError: name 'tf' is not defined

## === cell 5
def get_data_generators(preprocess=True, augment=True, IMAGE_SIZE=(3*136, 3*205)):
    
    def load_image(image_id):
        file_path = image_id + ".jpg"
        image = cv2.imread(IMAGE_PATH + file_path)
        image = cv2.resize(image, IMAGE_SIZE[::-1])
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if preprocess:
            image = remove_background(image)
        return image

    print("Preprocessing training images...")
    train_images = np.stack(train_data["image_id"].parallel_apply(load_image))
    plt.imshow(train_images[0])
    labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']]
    dataset = tf.data.Dataset.from_tensor_slices((train_images, np.stack(labels.values)))
    
    def map_func(image, label):
        image = tf.cast(image, tf.float32) / 255
        return image, label
    dataset = dataset.map(map_func, 
                          num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                          deterministic=False)

    if augment:
        augmentations = [flip, color, rotate, zoom]
    
        for f in augmentations:
            dataset = dataset.map(lambda x, y: (f(x), y),
                                  num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                                  deterministic=False)

        dataset = dataset.map(lambda x, y: (tf.clip_by_value(x, 0, 1), y), 
                              num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                              deterministic=False)
    
    train_size = int(len(train_data) * 0.85)
    valid_size = len(train_data) - train_size
    
    train = dataset.take(train_size)
    train = train.repeat().batch(32)
    train = train.prefetch(2)
    
    valid = dataset.skip(train_size).take(valid_size)
    valid = valid.repeat().batch(32)
    valid = valid.prefetch(2)
    
    print("Preprocessing test images...")
    test_images = np.stack(test_data["image_id"].parallel_apply(load_image))
    test = tf.data.Dataset.from_tensor_slices((np.stack(test_images,)))
    
    test = test.map(lambda image: tf.cast(image, tf.float32) / 255, 
                    num_parallel_calls=tf.data.experimental.AUTOTUNE, 
                    deterministic=False)
    test = test.batch(32).prefetch(2)
    
    return train, valid, test


## === cell 6
train, val, test = get_data_generators(False, True)


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3661367199.py in <cell line: 0>()
----> 1 train, val, test = get_data_generators(False, True)

/tmp/ipykernel_11/4090462216.py in get_data_generators(preprocess, augment, IMAGE_SIZE)
     11 
     12     print("Preprocessing training images...")
---> 13     train_images = np.stack(train_data["image_id"].parallel_apply(load_image))
     14     plt.imshow(train_images[0])
     15     labels = train_data[['healthy', 'multiple_diseases', 'rust', 'scab']]

/usr/local/lib/python3.11/dist-packages/pandas/core/generic.py in __getattr__(self, name)
   6297         ):
   6298             return self[name]
-> 6299         return object.__getattribute__(self, name)
   6300 
   6301     @final

AttributeError: 'Series' object has no attribute 'parallel_apply'

## === cell 7
def get_model(): 
    model = keras.Sequential()
    model.add(efn.EfficientNetB7(
        include_top=False, weights='imagenet', input_tensor=None, input_shape=None,
        pooling=None, classes=4))
    model.add(keras.layers.GlobalAveragePooling2D())
    model.add(keras.layers.Dense(128, activation="relu"))
    model.add(keras.layers.Dense(64, activation="relu"))
    model.add(keras.layers.Dense(4, activation="softmax"))
    model.summary()
    return model


## === cell 8
with strategy.scope():
    model = get_model()
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss=keras.losses.categorical_crossentropy,
        metrics=[keras.metrics.categorical_accuracy])


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1158637717.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     model = get_model()
      3     model.compile(
      4         optimizer=keras.optimizers.Adam(learning_rate=1e-3),
      5         loss=keras.losses.categorical_crossentropy,

NameError: name 'strategy' is not defined

## === cell 9
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard

history = model.fit(train,                                    
    steps_per_epoch=50, 
    epochs=40,
    validation_data=val,
    validation_steps=50,
    validation_freq=1,
    verbose=1,
    callbacks=[
    ])


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/288679593.py in <cell line: 0>()
      1 from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard
      2 
----> 3 history = model.fit(train,                                    
      4     steps_per_epoch=50,
      5     epochs=40,

NameError: name 'model' is not defined

## === cell 10
test_pr = model.predict(test, verbose=1)
sub.loc[:, 'healthy':] = test_pr
sub.to_csv('submission.csv', index=False)
sub.head()


## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1528956638.py in <cell line: 0>()
      1 # Submission
----> 2 test_pr = model.predict(test, verbose=1)
      3 sub.loc[:, 'healthy':] = test_pr
      4 sub.to_csv('submission.csv', index=False)
      5 sub.head()

NameError: name 'model' is not defined
