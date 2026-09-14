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

No external packages required in the script and installed.

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

0.7185074836498475

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install -q efficientnet

## === cell 1
!pip install transformers

## === cell 3
import numpy as np 
import pandas as pd 
import os
from matplotlib import pyplot as plt
import seaborn as sns

import tensorflow as tf
import tensorflow.keras.layers as l
from keras.regularizers import l2
from keras.optimizers import Adam
import efficientnet.tfkeras as efn
from sklearn.model_selection import train_test_split
from kaggle_datasets import KaggleDatasets


from transformers import BertForSequenceClassification, AdamW, BertConfig
from transformers import BertTokenizer
from transformers import get_linear_schedule_with_warmup

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 5
tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
tf.config.experimental_connect_to_cluster(tpu)
tf.tpu.experimental.initialize_tpu_system(tpu)

tpu_strategy = tf.distribute.experimental.TPUStrategy(tpu)

print(tpu.master())
print(tpu_strategy.num_replicas_in_sync)

## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3666696384.py in <cell line: 0>()
----> 1 tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
      2 tf.config.experimental_connect_to_cluster(tpu)
      3 tf.tpu.experimental.initialize_tpu_system(tpu)
      4 
      5 # instantiate a distribution strategy

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

## === cell 7
AUTO = tf.data.experimental.AUTOTUNE
ignore_order = tf.data.Options()
ignore_order.experimental_deterministic = False

gcs_path = KaggleDatasets().get_gcs_path('alaska2-image-steganalysis')

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/1350034804.py in <cell line: 0>()
      5 
      6 # Pass
----> 7 gcs_path = KaggleDatasets().get_gcs_path('alaska2-image-steganalysis')

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

## === cell 9
sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
BATCH_SIZE = 8 * tpu_strategy.num_replicas_in_sync # batch size in tpu
EPOCHS = 1


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

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1979266259.py in <cell line: 0>()
      1 sample = pd.read_csv("/kaggle/input/alaska2-image-steganalysis/sample_submission.csv")
----> 2 BATCH_SIZE = 8 * tpu_strategy.num_replicas_in_sync # batch size in tpu
      3 EPOCHS = 1
      4 
      5 #Variables

NameError: name 'tpu_strategy' is not defined

## === cell 11
df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]

def cate_label(x):
    if x["cate"] == "Cover":
        res = 0
    else:
        res = 1
    return res

Test_df = df.query("cate=='Test'").sort_values(by="name")
Train_df = df.query("cate!='Test'")

Train_df["labled"] = df.apply(cate_label, axis=1)

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3629986997.py in <cell line: 0>()
      1 #add path to df
----> 2 df["path"] = [str(os.path.join(gcs_path,cate,name)) for cate, name in zip(df["cate"], df["name"])]
      3 
      4 #Labeling func
      5 def cate_label(x):

NameError: name 'df' is not defined

## === cell 12
print("Training set: \n",Train_df["cate"].value_counts())

print('\n', Train_df["path"].head())
print('\n',Train_df["labled"].head())
print('\n',Test_df["path"].head())

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2254849770.py in <cell line: 0>()
----> 1 print("Training set: \n",Train_df["cate"].value_counts())
      2 
      3 print('\n', Train_df["path"].head())
      4 print('\n',Train_df["labled"].head())
      5 print('\n',Test_df["path"].head())

NameError: name 'Train_df' is not defined

## === cell 13


X = Train_df["path"]
y = Train_df["labled"]
z = Test_df["path"]

X_train, X_val, y_train, y_val = train_test_split(X,y, test_size=0.2, random_state=10)

X_train, X_val, y_train, y_val = np.array(X_train), np.array(X_val), np.array(y_train), np.array(y_val)

X_test = np.array(Test_df["path"])

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3060572797.py in <cell line: 0>()
      2 
      3 
----> 4 X = Train_df["path"]
      5 y = Train_df["labled"]
      6 z = Test_df["path"]

NameError: name 'Train_df' is not defined

## === cell 14
print(X_test[7])
print(X_train[7])
print(X_val[7])
print(y_train[7])
print(y_val[7])

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3907430055.py in <cell line: 0>()
----> 1 print(X_test[7])
      2 print(X_train[7])
      3 print(X_val[7])
      4 print(y_train[7])
      5 print(y_val[7])

NameError: name 'X_test' is not defined

## === cell 16
def decode_image(filename, label=None, image_size=(512,512)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32)/255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label

## === cell 18
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_train, y_train))
    .map(decode_image, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(1024)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

valid_dataset = (
    tf.data.Dataset
    .from_tensor_slices((X_val, y_val))
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(X_test)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3734624373.py in <cell line: 0>()
      1 train_dataset = (
      2     tf.data.Dataset
----> 3     .from_tensor_slices((X_train, y_train))
      4     .map(decode_image, num_parallel_calls=AUTO)
      5     .repeat()

NameError: name 'X_train' is not defined

## === cell 20
with tpu_strategy.scope():
    model = tf.keras.Sequential([
        efn.EfficientNetB3(
            input_shape=(512, 512, 3),
            weights='imagenet',
            include_top=False
        ),
        l.GlobalAveragePooling2D(),
        l.Dropout(0.1),
        l.Dense(1, activation='sigmoid')
    ])
    opt = Adam(lr=0.002, beta_1=0.9, beta_2=0.999, decay=0.01, amsgrad=False)    
    model.compile(
        optimizer=opt,
        loss = 'binary_crossentropy',
        metrics=['accuracy']
    )

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4219438371.py in <cell line: 0>()
----> 1 with tpu_strategy.scope():
      2     model = tf.keras.Sequential([
      3         efn.EfficientNetB3(
      4             input_shape=(512, 512, 3),
      5             weights='imagenet',

NameError: name 'tpu_strategy' is not defined

## === cell 21
model.summary()

## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3470139634.py in <cell line: 0>()
----> 1 model.summary()

NameError: name 'model' is not defined

## === cell 23
STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE

history = model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=valid_dataset)

## --- ERROR in cell 23, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2736582283.py in <cell line: 0>()
----> 1 STEPS_PER_EPOCH = X_train.shape[0] // BATCH_SIZE
      2 #callbacks = [tf.keras.callbacks.EarlyStopping(patience=3, restore_best_weights=True)]
      3 
      4 history = model.fit(train_dataset, steps_per_epoch=STEPS_PER_EPOCH, epochs=EPOCHS, validation_data=valid_dataset)

NameError: name 'X_train' is not defined

## === cell 24
model.save("Mymodel.h5") 

## --- ERROR in cell 24, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2631773081.py in <cell line: 0>()
----> 1 model.save("Mymodel.h5")

NameError: name 'model' is not defined

## === cell 28
pred = model.predict(test_dataset, verbose=1)

## --- ERROR in cell 28, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3226153887.py in <cell line: 0>()
----> 1 pred = model.predict(test_dataset, verbose=1)

NameError: name 'model' is not defined

## === cell 29
my_sample = sample.copy()
my_sample["Label"] = pred
my_sample.to_csv("my_sample.csv", index=False)
my_sample.head()

## --- ERROR in cell 29, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1254637923.py in <cell line: 0>()
      1 my_sample = sample.copy()
----> 2 my_sample["Label"] = pred
      3 my_sample.to_csv("my_sample.csv", index=False)
      4 my_sample.head()

NameError: name 'pred' is not defined
