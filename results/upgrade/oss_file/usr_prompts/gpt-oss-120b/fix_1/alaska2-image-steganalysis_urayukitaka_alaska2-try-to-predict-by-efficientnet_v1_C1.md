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
Determine which of the images have hidden messages embedded using one of three steganography algorithms (JMiPOD, JUNIWARD, UERD).

## Metric
Weighted AUC. Each region of the ROC curve is weighted according to these chosen parameters:

```
tpr_thresholds = [0.0, 0.4, 1.0]
weights = [2, 1]
```

In other words, the area between the true positive rate of 0 and 0.4 is weighted 2X, the area between 0.4 and 1 is now weighed (1X). The total area is normalized by the sum of weights such that the final weighted AUC is between 0 and 1.

## Submission Format
For each `Id` (image) in the test set, you must provide a score that indicates how likely this image contains hidden data: the higher the score, the more it is assumed that image contains secret data. The file should contain a header and have the following format:

```
Id,Label
0001.jpg,0.1
0002.jpg,0.99
0003.jpg,1.2
0004.jpg,-2.2
etc.
```
## Dataset
The only available information on the test set is:

1. Each embedding algorithm is used with the same probability.
2. The payload (message length) is adjusted such that the "difficulty" is approximately the same regardless the content of the image. Images with smooth content are used to hide shorter messages while highly textured images will be used to hide more secret bits. The payload is adjusted in the same manner for testing and training sets.
3. The average message length is 0.4 bit per non-zero AC DCT coefficient.
4. The images are all compressed with one of the three following JPEG quality factors: 95, 90 or 75.

### Files
- `Cover/` contains 75k unaltered images meant for use in training.
- `JMiPOD/` contains 75k examples of the JMiPOD algorithm applied to the cover images.
- `JUNIWARD/`contains 75k examples of the JUNIWARD algorithm applied to the cover images.
- `UERD/` contains 75k examples of the UERD algorithm applied to the cover images.
- `Test/` contains 5k test set images. These are the images for which you are predicting.
- `sample_submission.csv` contains an example submission in the correct format.

# 2. Python version

3.8

# 3. Installed packages

geopandas==0.14.4
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

# 4. Data file paths

```
/
    kaggle/
        data/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        input/
            Cover.zip (7.4 GB)
            JMiPOD.zip (7.4 GB)
            JUNIWARD.zip (7.4 GB)
            Test.zip (528.5 MB)
            UERD.zip (7.4 GB)
            description.md (91 lines)
            sample_submission.csv (5001 lines)
            sample_submission.csv.zip (10.7 kB)
            Cover/
                54965.jpg (237.9 kB)
                54517.jpg (126.7 kB)
                ... and 69998 other files
            JMiPOD/
                06809.jpg (36.6 kB)
                42490.jpg (78.8 kB)
                ... and 69998 other files
            JUNIWARD/
                03684.jpg (106.2 kB)
                42131.jpg (144.4 kB)
                ... and 69998 other files
            Test/
                3630.jpg (79.5 kB)
                3197.jpg (208.4 kB)
                ... and 4998 other files
            UERD/
                42300.jpg (47.1 kB)
                59199.jpg (41.9 kB)
                ... and 69998 other files
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
        working/
            alaska2-image-steganalysis/
                Cover.zip (7.4 GB)
                JMiPOD.zip (7.4 GB)
                ... and 6 other files
                Cover/
                    54965.jpg (237.9 kB)
                    54517.jpg (126.7 kB)
                    ... and 69998 other files
                JMiPOD/
                    06809.jpg (36.6 kB)
                    42490.jpg (78.8 kB)
                    ... and 69998 other files
                JUNIWARD/
                    03684.jpg (106.2 kB)
                    42131.jpg (144.4 kB)
                    ... and 69998 other files
                Test/
                    3630.jpg (79.5 kB)
                    3197.jpg (208.4 kB)
                    ... and 4998 other files
                UERD/
                    42300.jpg (47.1 kB)
                    59199.jpg (41.9 kB)
                    ... and 69998 other files
                alaska2-image-steganalysis/
```

-> data/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> data/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> input/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

-> working/alaska2-image-steganalysis/sample_submission.csv has 5000 rows and 2 columns.
The columns are: Id, Label

# 5. Target score

0.574

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
! pip install -q efficientnet


## === cell 1
import numpy as np 
import pandas as pd 
import os

import cv2# Open cv
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import seaborn as sns
plt.style.use('fivethirtyeight')

import tensorflow as tf
import tensorflow.keras.layers as l
import efficientnet.tfkeras as efn

from kaggle_datasets import KaggleDatasets


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)


## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2285912477.py in <cell line: 0>()
      1 # TPU setting
      2 # detect and init the TPU
----> 3 tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      4 tf.config.experimental_connect_to_cluster(tpu)
      5 tf.tpu.experimental.initialize_tpu_system(tpu)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/distribute/cluster_resolver/tpu/tpu_cluster_resolver.py in __init__(self, tpu, zone, project, job_name, coordinator_name, coordinator_address, credentials, service, discovery_url)
    233     if tpu != 'local':
    234       # Default Cloud environment
--> 235       self._cloud_tpu_client = client.Client(
    236           tpu=tpu,
    237           zone=zone,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/tpu/client/client.py in __init__(self, tpu, zone, project, credentials, service, discovery_url)
    157         zone = zone or tpu_node_config.get('zone')
    158       else:
--> 159         raise ValueError('Please provide a TPU Name to connect to.')
    160 
    161     self._tpu = _as_text(tpu)

ValueError: Please provide a TPU Name to connect to.

## === cell 3
print(tpu.master())
print(tpu_strategy.num_replicas_in_sync)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4031315719.py in <cell line: 0>()
----> 1 print(tpu.master())
      2 print(tpu_strategy.num_replicas_in_sync)

NameError: name 'tpu' is not defined

## === cell 4
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

gcs_path = KaggleDatasets().get_gcs_path()


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/493473504.py in <cell line: 0>()
      5 
      6 # Pass
----> 7 gcs_path = KaggleDatasets().get_gcs_path()

/usr/local/lib/python3.11/dist-packages/kaggle_datasets.py in get_gcs_path(self, dataset_dir)
     39             'IntegrationType': integration_type,
     40         }
---> 41         result = self.web_client.make_post_request(data, self.GET_GCS_PATH_ENDPOINT, self.TIMEOUT_SECS)
     42         return result['destinationBucket']

/usr/local/lib/python3.11/dist-packages/kaggle_web_client.py in make_post_request(self, data, endpoint, timeout)
     47                 response_json = json.loads(response.read())
     48                 if not response_json.get('wasSuccessful') or 'result' not in response_json:
---> 49                     raise BackendError(
     50                         f'Unexpected response from the service. Response: {response_json}.')
     51                 return response_json['result']

BackendError: Unexpected response from the service. Response: {'errors': ['Unauthenticated'], 'error': {'code': 16}, 'wasSuccessful': False}.

## === cell 5
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")


## === cell 6
BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync


## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2470032129.py in <cell line: 0>()
      1 # batch size in tpu
----> 2 BATCH_SIZE = 16 * tpu_strategy.num_replicas_in_sync

NameError: name 'tpu_strategy' is not defined

## === cell 7
dir_name = ['Test', 'JUNIWARD', 'JMiPOD', 'Cover', 'UERD']
df = pd.DataFrame({})

lists = []
cate = []

for dir_ in dir_name:
    list_ = os.listdir("/kaggle/input/alaska2-image-steganalysis/"+dir_+"/")
    lists = lists+list_
    cate_ = np.tile(dir_,len(list_))
    cate = np.concatenate([cate,cate_])
    
df["cate"] = cate
df["name"] = lists


## === cell 8
df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2728240691.py in <cell line: 0>()
      1 # path line
----> 2 df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]

/tmp/ipykernel_11/2728240691.py in <listcomp>(.0)
      1 # path line
----> 2 df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]

NameError: name 'gcs_path' is not defined

## === cell 9
def cate_label(x):
    if x["cate"] == "Cover":
        res = 0
    else:
        res = 1
    return res

Test_df = df.query("cate=='Test'").sort_values(by="name")
Train_df = df.query("cate!='Test'")
Train_df["flg"] = df.apply(cate_label, axis=1)


## === cell 10
Train_df["cate"].value_counts()


## === cell 11
Train_df = Train_df.sample(1000)
Train_df["cate"].value_counts()


## === cell 12
X = Train_df["path"]
y = Train_df["flg"]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10)


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2415241779.py in <cell line: 0>()
      1 # Create train data and val data
----> 2 X = Train_df["path"]
      3 y = Train_df["flg"]
      4 
      5 # split train and val data

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'path'

## === cell 13
X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)


## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3481101661.py in <cell line: 0>()
      1 # Change to numpy array
----> 2 X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

NameError: name 'X_train' is not defined

## === cell 14
X_test = np.array(Test_df["path"])


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3804         try:
-> 3805             return self._engine.get_loc(casted_key)
   3806         except KeyError as err:

index.pyx in pandas._libs.index.IndexEngine.get_loc()

index.pyx in pandas._libs.index.IndexEngine.get_loc()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

pandas/_libs/hashtable_class_helper.pxi in pandas._libs.hashtable.PyObjectHashTable.get_item()

KeyError: 'path'

The above exception was the direct cause of the following exception:

KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_11/2558718149.py in <cell line: 0>()
      1 # Create test data
----> 2 X_test = np.array(Test_df["path"])

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __getitem__(self, key)
   4100             if self.columns.nlevels > 1:
   4101                 return self._getitem_multilevel(key)
-> 4102             indexer = self.columns.get_loc(key)
   4103             if is_integer(indexer):
   4104                 indexer = [indexer]

/usr/local/lib/python3.11/dist-packages/pandas/core/indexes/base.py in get_loc(self, key)
   3810             ):
   3811                 raise InvalidIndexError(key)
-> 3812             raise KeyError(key) from err
   3813         except TypeError:
   3814             # If we have a listlike key, _check_indexing_error will raise

KeyError: 'path'

## === cell 15
def decode_image(filename, label=None, image_size=(512,512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)/255.0
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


## === cell 16
train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(decode_image, num_parallel_calls=AUTO)
                    .map(data_augment, num_parallel_calls=AUTO).repeat().shuffle(512).batch(BATCH_SIZE).prefetch(AUTO))

valid_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(decode_image, num_parallel_calls=AUTO)
                    .cache().batch(BATCH_SIZE).prefetch(AUTO))

test_dataset = (tf.data.Dataset.from_tensor_slices((X_test)).map(decode_image, num_parallel_calls=AUTO)
                    .batch(BATCH_SIZE))


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3104542589.py in <cell line: 0>()
      1 # For create dataset for TPU
----> 2 train_dataset = (tf.data.Dataset.from_tensor_slices((X_train, y_train)).map(decode_image, num_parallel_calls=AUTO)
      3                     .map(data_augment, num_parallel_calls=AUTO).repeat().shuffle(512).batch(BATCH_SIZE).prefetch(AUTO))
      4 
      5 valid_dataset = (tf.data.Dataset.from_tensor_slices((X_val, y_val)).map(decode_image, num_parallel_calls=AUTO)

NameError: name 'X_train' is not defined

## === cell 17
with tpu_strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB3(input_shape=(512,512,3),weights='imagenet',include_top=False),
        l.GlobalAveragePooling2D(),
        l.Dense(1, activation="sigmoid")
    ])
    
    opt = tf.keras.optimizers.SGD(learning_rate=0.00001)
    
    model.compile(optimizer=opt, loss="binary_crossentropy", metrics=["accuracy"])
    model.summary()


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3787472066.py in <cell line: 0>()
----> 1 with tpu_strategy.scope():
      2     model = tf.keras.Sequential([
      3         efn.EfficientNetB3(input_shape=(512,512,3),weights='imagenet',include_top=False),
      4         l.GlobalAveragePooling2D(),
      5         l.Dense(1, activation="sigmoid")

NameError: name 'tpu_strategy' is not defined

## === cell 18
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE

EPOCHS = 10
hist = model.fit(train_dataset, epochs=EPOCHS,
                   steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset)


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4077530499.py in <cell line: 0>()
----> 1 STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
      2 
      3 EPOCHS = 10
      4 hist = model.fit(train_dataset, epochs=EPOCHS,
      5                    steps_per_epoch=STEPS_PER_EPOCH, validation_data=valid_dataset)

NameError: name 'X_train' is not defined

## === cell 19
train_loss = hist.history["loss"]
val_loss = hist.history["val_loss"]
train_acc = hist.history["accuracy"]
val_acc = hist.history["val_accuracy"]

fig, ax = plt.subplots(1,2,figsize=(10,6))
ax[0].plot(range(len(train_loss)), train_loss, label="train_loss")
ax[0].plot(range(len(val_loss)), val_loss, label="val_loss")
ax[0].set_xlabel("epochs")
ax[0].set_ylabel("loss")
ax[0].set_title("loss")

ax[1].plot(range(len(train_acc)), train_acc, label="train_accuracy")
ax[1].plot(range(len(val_acc)), val_acc, label="val_accuracy")
ax[1].set_xlabel("epochs")
ax[1].set_ylabel("accuracy")
ax[1].set_title("accurary")


## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1133851847.py in <cell line: 0>()
      1 # training history
----> 2 train_loss = hist.history["loss"]
      3 val_loss = hist.history["val_loss"]
      4 train_acc = hist.history["accuracy"]
      5 val_acc = hist.history["val_accuracy"]

NameError: name 'hist' is not defined

## === cell 20
pred = model.predict(test_dataset, verbose=1)


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2105780358.py in <cell line: 0>()
      1 # Prediction
----> 2 pred = model.predict(test_dataset, verbose=1)

NameError: name 'model' is not defined

## === cell 21
sample["Label"] = pred
sample.to_csv("submission.csv", index=False)


## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2440184883.py in <cell line: 0>()
      1 # Submission file
----> 2 sample["Label"] = pred
      3 sample.to_csv("submission.csv", index=False)

NameError: name 'pred' is not defined

## === cell 22
sample["Label"]
