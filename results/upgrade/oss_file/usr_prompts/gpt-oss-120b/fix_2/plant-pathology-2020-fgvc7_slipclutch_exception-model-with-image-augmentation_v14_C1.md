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

0.83966

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
import tensorflow as tf
from tensorflow import keras
from keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from keras.models import Sequential, Model
from keras.layers import (
    Activation,
    Dropout,
    Flatten,
    Dense,
    Conv2D,
    MaxPooling2D,
    BatchNormalization,
)
from keras.optimizers import Adam
from tensorflow.keras.applications import Xception

try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.TPUStrategy(tpu)
except Exception:
    strategy = tf.distribute.get_strategy()
print("REPLICAS:", strategy.num_replicas_in_sync)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
kaggle_dir = "/kaggle/input/plant-pathology-2020-fgvc7/"
images_dir = os.path.join(kaggle_dir, "images")
GCS_DS_PATH = kaggle_dir  # use local path for image files
AUTO = tf.data.experimental.AUTOTUNE

sample_submission = pd.read_csv(os.path.join(kaggle_dir, "sample_submission.csv"))
test = pd.read_csv(os.path.join(kaggle_dir, "test.csv"))
train = pd.read_csv(os.path.join(kaggle_dir, "train.csv"))


def format_path(st):
    return os.path.join(images_dir, st + ".jpg")




## === cell 2
IMG_SIZE = 300


def seed_everything(seed=0):
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    os.environ["TF_DETERMINISTIC_OPS"] = "1"


seed = 2048
seed_everything(seed)

train_paths = train["image_id"].apply(format_path).values
test_paths = test["image_id"].apply(format_path).values
train_labels = train.loc[:, "healthy":].values.astype(np.float32)

SPLIT_VALIDATION = True
if SPLIT_VALIDATION:
    train_paths, valid_paths, train_labels, valid_labels = train_test_split(
        train_paths, train_labels, test_size=0.15, random_state=seed
    )


def decode_image(filename, label=None, size=(IMG_SIZE, IMG_SIZE)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, size)
    return (image, label) if label is not None else image


def data_augment(image, label=None):
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    return (image, label) if label is not None else image




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/67041119.py in <cell line: 0>()
     19 SPLIT_VALIDATION = True
     20 if SPLIT_VALIDATION:
---> 21     train_paths, valid_paths, train_labels, valid_labels = train_test_split(
     22         train_paths, train_labels, test_size=0.15, random_state=seed
     23     )

NameError: name 'train_test_split' is not defined

## === cell 3
BATCH_SIZE = 32

train_dataset = (
    tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset.from_tensor_slices((valid_paths, valid_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .cache()
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset.from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)



## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1996614059.py in <cell line: 0>()
      3 train_dataset = (
      4     tf.data.Dataset.from_tensor_slices((train_paths, train_labels))
----> 5     .map(decode_image, num_parallel_calls=AUTO)
      6     .cache()
      7     .map(data_augment, num_parallel_calls=AUTO)

NameError: name 'decode_image' is not defined

## === cell 4
LR_START = 0.0001
LR_MAX = 0.00005 * strategy.num_replicas_in_sync
LR_MIN = 0.0001
LR_RAMPUP_EPOCHS = 4
LR_SUSTAIN_EPOCHS = 6
LR_EXP_DECAY = 0.8


def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY ** (
            epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS
        ) + LR_MIN
    return lr


lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)



## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/959597149.py in <cell line: 0>()
      1 LR_START = 0.0001
----> 2 LR_MAX = 0.00005 * strategy.num_replicas_in_sync
      3 LR_MIN = 0.0001
      4 LR_RAMPUP_EPOCHS = 4
      5 LR_SUSTAIN_EPOCHS = 6

NameError: name 'strategy' is not defined

## === cell 5
print(
    "Label sums:", train.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].sum()
)
train.loc[:, ["healthy", "multiple_diseases", "rust", "scab"]].mean().plot(
    kind="bar", title="Mean label frequency"
)



## === cell 6
with strategy.scope():
    base_model = Xception(
        input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet", include_top=False
    )
    x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)
    outputs = tf.keras.layers.Dense(4, activation="softmax")(x)
    model = keras.Model(inputs=base_model.input, outputs=outputs)
    model.compile(
        loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]
    )



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/449313533.py in <cell line: 0>()
----> 1 with strategy.scope():
      2     base_model = Xception(
      3         input_shape=(IMG_SIZE, IMG_SIZE, 3), weights="imagenet", include_top=False
      4     )
      5     x = tf.keras.layers.GlobalAveragePooling2D()(base_model.output)

NameError: name 'strategy' is not defined

## === cell 7
datagen = ImageDataGenerator(
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
)



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/434783126.py in <cell line: 0>()
----> 1 datagen = ImageDataGenerator(
      2     rotation_range=20,
      3     width_shift_range=0.2,
      4     height_shift_range=0.2,
      5     shear_range=0.2,

NameError: name 'ImageDataGenerator' is not defined

## === cell 8
model.fit(
    train_dataset,
    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
    epochs=50,
    validation_data=valid_dataset if SPLIT_VALIDATION else None,
    callbacks=[lr_callback],
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3900712816.py in <cell line: 0>()
----> 1 model.fit(
      2     train_dataset,
      3     steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
      4     epochs=50,
      5     validation_data=valid_dataset if SPLIT_VALIDATION else None,

NameError: name 'model' is not defined

## === cell 9
predict = model.predict(test_dataset)

prediction_df = pd.DataFrame(
    predict, columns=["healthy", "multiple_diseases", "rust", "scab"]
)
submission = pd.concat([test["image_id"], prediction_df], axis=1)
submission.to_csv("submission.csv", index=False)

from IPython.display import FileLink

FileLink(r"submission.csv")

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3317403615.py in <cell line: 0>()
      1 # Predict probabilities for the test set
----> 2 predict = model.predict(test_dataset)
      3 
      4 # Build submission DataFrame with required column order
      5 prediction_df = pd.DataFrame(

NameError: name 'model' is not defined
