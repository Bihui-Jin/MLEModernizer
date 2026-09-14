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
Predict whether a lesion is malignant (0 denotes **benign**, and 1 indicates **malignant**).

## Metric
Area under the ROC curve.

## Submission Format
For each `image_name` in the test set, you must predict the probability (`target`) that the sample is **malignant**. The file should contain a header and have the following format:

```
image_name,target
ISIC_0052060,0.7
ISIC_0052349,0.9
ISIC_0058510,0.8
ISIC_0073313,0.5
ISIC_0073502,0.5
etc.
```

## Dataset 
The images are provided in DICOM format.

Images are also provided in JPEG and TFRecord format (in the `jpeg` and `tfrecords` directories, respectively). Images in TFRecord format have been resized to a uniform 1024x1024.

Metadata is also provided outside of the DICOM format, in CSV files. See the `Columns` section for a description.

### Files
- **train.csv** - the training set
- **test.csv** - the test set
- **sample_submission.csv** - a sample submission file in the correct format

### Columns
- `image_name` - unique identifier, points to filename of related DICOM image
- `patient_id` - unique patient identifier
- `sex` - the sex of the patient (when unknown, will be blank)
- `age_approx` - approximate patient age at time of imaging
- `anatom_site_general_challenge` - location of imaged site
- `diagnosis` - detailed diagnosis information (train only)
- `benign_malignant` - indicator of malignancy of imaged lesion
- `target` - binarized version of the target variable

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
tf_keras==2.18.0

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
        input/
            description.md (176 lines)
            jpeg.zip (24.7 GB)
            sample_submission.csv (4143 lines)
            sample_submission.csv.zip (16.4 kB)
            test.csv (4143 lines)
            test.csv.zip (42.5 kB)
            test.zip (6.4 GB)
            tfrecords.zip (9.3 GB)
            train.csv (28985 lines)
            train.csv.zip (299.7 kB)
            train.zip (46.0 GB)
            jpeg/
                test/
                    ISIC_1440063.jpg (1.1 MB)
                    ISIC_0815802.jpg (853.1 kB)
                    ... and 4140 other files
                train/
                    ISIC_1845271.jpg (1.0 MB)
                    ISIC_1970027.jpg (138.4 kB)
                    ... and 28982 other files
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
            test/
                ISIC_0052212.dcm (1.5 MB)
                ISIC_0076545.dcm (4.0 MB)
                ... and 4140 other files
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
            tfrecords/
                test00-2071.tfrec (579.6 MB)
                test01-2071.tfrec (583.5 MB)
                ... and 14 other files
            train/
                ISIC_0015719.dcm (2.4 MB)
                ISIC_0068279.dcm (1.3 MB)
                ... and 28982 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
        working/
            siim-isic-melanoma-classification/
                description.md (176 lines)
                jpeg.zip (24.7 GB)
                ... and 9 other files
                jpeg/
                    test/
                        ISIC_1440063.jpg (1.1 MB)
                        ISIC_0815802.jpg (853.1 kB)
                        ... and 4140 other files
                    train/
                        ISIC_1845271.jpg (1.0 MB)
                        ISIC_1970027.jpg (138.4 kB)
                        ... and 28982 other files
                siim-isic-melanoma-classification/
                test/
                    ISIC_0052212.dcm (1.5 MB)
                    ISIC_0076545.dcm (4.0 MB)
                    ... and 4140 other files
                    test/
                tfrecords/
                    test00-2071.tfrec (579.6 MB)
                    test01-2071.tfrec (583.5 MB)
                    ... and 14 other files
                train/
                    ISIC_0015719.dcm (2.4 MB)
                    ISIC_0068279.dcm (1.3 MB)
                    ... and 28982 other files
                    train/
```

-> data/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> data/siim-isic-melanoma-classification/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/siim-isic-melanoma-classification/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> data/test.csv has 4142 rows and 5 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge

-> data/train.csv has 28984 rows and 8 columns.
The columns are: image_name, patient_id, sex, age_approx, anatom_site_general_challenge, diagnosis, benign_malignant, target

-> input/sample_submission.csv has 4142 rows and 2 columns.
The columns are: image_name, target

-> (stopped after 10 files for performance)

# 5. Target score

0.8015

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)


import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    for filename in filenames:
        print(os.path.join(dirname, filename))



## === cell 1
import os
import re
import seaborn as sns
import numpy as np
import pandas as pd
import math

from matplotlib import pyplot as plt

from sklearn import metrics
from sklearn.model_selection import train_test_split

import tensorflow as tf
import tensorflow.keras.layers as L



from kaggle_datasets import KaggleDatasets


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
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

print("REPLICAS: ", strategy.num_replicas_in_sync)


## === cell 3
AUTO = tf.data.experimental.AUTOTUNE

GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')

EPOCHS = 10
BATCH_SIZE = 8 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/2745629270.py in <cell line: 0>()
      3 
      4 # Data access
----> 5 GCS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
      6 
      7 # Configuration

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

## === cell 4
def append_path(pre):
    return np.vectorize(lambda file: os.path.join(GCS_DS_PATH, pre, file))


## === cell 5

sub = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/sample_submission.csv')
train = pd.read_csv('/kaggle/input/siim-isic-melanoma-classification/train.csv')


## === cell 6
sns.countplot(train['target'])


## === cell 7
TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')

CLASSES = [0,1]


## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1517658137.py in <cell line: 0>()
      1 # getting the training and testing filenames
----> 2 TRAINING_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/train*.tfrec')
      3 TEST_FILENAMES = tf.io.gfile.glob(GCS_PATH + '/tfrecords/test*.tfrec')
      4 
      5 CLASSES = [0,1]

NameError: name 'GCS_PATH' is not defined

## === cell 8
TRAINING_FILENAMES[:3]


## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2410557074.py in <cell line: 0>()
      1 #  Lets look at how the training file names look like
----> 2 TRAINING_FILENAMES[:3]

NameError: name 'TRAINING_FILENAMES' is not defined

## === cell 9
def decode_image(image_data):
    image =  tf.image.decode_jpeg(image_data, channels = 3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.reshape(image, [*IMAGE_SIZE, 3])
    return image



def read_labeled_tfrecord(example):
    LABELED_TFREC_FORMAT = {
        "image" : tf.io.FixedLenFeature([], tf.string),
        "target" : tf.io.FixedLenFeature([], tf.int64)
    }
    
    example = tf.io.parse_single_example(example, LABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    label = tf.cast(example['target'], tf.int32)
    return image, label


def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT = {
        "image" : tf.io.FixedLenFeature([], tf.string),
        "image_name" : tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    id_num = example['image_name']
    return image, id_num
    
    
def load_dataset(filenames, labeled = True, ordered = False):
    
    ignore_order = tf.data.Options()
    
    if not ordered:
        ignore_order.experimental_deterministic = False
        
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads = AUTO) #automatically reads from different files
    dataset =  dataset.with_options(ignore_order) # reads dataset as it comes in rather than its original order
    
    dataset  = dataset.map(read_labeled_tfrecord if  labeled else read_unlabeled_tfrecord,
                           num_parallel_calls = AUTO)
    
    return dataset


def data_augment(image, label):
    image = tf.image.random_flip_left_right(image)
    return image, label


def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled = True)
    dataset = dataset.map(data_augment, num_parallel_calls = AUTO)
    dataset = dataset.repeat() # the training  dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch the next batch while training (autotune prefetch buffer size)
    return dataset


def get_validation_dataset(ordered = False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled = True, ordered = ordered)
    dataset =  dataset.batch(BATCH_SIZE)
    dataset =  dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch the next batch while training (autotune prefetch buffer size)
    return dataset
    
    
    
def get_test_dataset(ordered = False):
    dataset = load_dataset(TEST_FILENAMES, labeled = False, ordered = ordered)
    dataset =  dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch the next batch while training (autotune prefetch buffer size)
    return dataset
    
    
    
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)
    
    
    
NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
print('Dataset: {} training images, {} unlabeled test images'.format(NUM_TRAINING_IMAGES, NUM_TEST_IMAGES))


## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/189272503.py in <cell line: 0>()
     91 
     92 
---> 93 NUM_TRAINING_IMAGES = count_data_items(TRAINING_FILENAMES)
     94 NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
     95 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

NameError: name 'TRAINING_FILENAMES' is not defined

## === cell 10
def build_lrfn(lr_start=0.00001, lr_max=0.0001, 
               lr_min=0.000001, lr_rampup_epochs=20, 
               lr_sustain_epochs=0, lr_exp_decay=.8):
    lr_max = lr_max * strategy.num_replicas_in_sync

    def lrfn(epoch):
        if epoch < lr_rampup_epochs:
            lr = (lr_max - lr_start) / lr_rampup_epochs * epoch + lr_start
        elif epoch < lr_rampup_epochs + lr_sustain_epochs:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_exp_decay**(epoch - lr_rampup_epochs - lr_sustain_epochs) + lr_min
        return lr
    
    return lrfn


## === cell 11
with strategy.scope():
    model = tf.keras.Sequential([
        tf.keras.applications.ResNet50(
            weights =  'imagenet',
            input_shape  = [*IMAGE_SIZE, 3],
            include_top =  False
        ),
        L.GlobalAveragePooling2D(),
        L.Dense(1024, activation = 'relu'), 
        L.Dropout(0.3), 
        L.Dense(512, activation= 'relu'), 
        L.Dropout(0.2), 
        L.Dense(256, activation='relu'), 
        L.Dropout(0.2), 
        L.Dense(128, activation='relu'), 
        L.Dropout(0.1), 
        L.Dense(1, activation='sigmoid')
        
    ]) 
    
    
model.compile(loss = tf.keras.losses.BinaryCrossentropy(label_smoothing = 0.1),
             metrics = ['accuracy'],
             optimizer =  'adam')

model.summary()
    
    


## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4043723441.py in <cell line: 0>()
      3         tf.keras.applications.ResNet50(
      4             weights =  'imagenet',
----> 5             input_shape  = [*IMAGE_SIZE, 3],
      6             include_top =  False
      7         ),

NameError: name 'IMAGE_SIZE' is not defined

## === cell 12
lrfn = build_lrfn()
lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/138162656.py in <cell line: 0>()
      1 lrfn = build_lrfn()
      2 lr_schedule = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=1)
----> 3 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

NameError: name 'NUM_TRAINING_IMAGES' is not defined

## === cell 13
from keras.optimizers import RMSprop,  Adam, SGD
from keras.callbacks import ModelCheckpoint,  EarlyStopping,  ReduceLROnPlateau

checkpoint  =  ModelCheckpoint('../input/output/model.h5',
                              monitor  =  'val_loss',
                              mode =  'min',
                              save_best_only =  True,
                              verbose = 1)

earlystop = EarlyStopping(monitor  = 'val_loss',
                         min_delta  =  0,
                         patience =  3,
                         verbose = 1,
                         restore_best_weights =  True)

callbacks = [earlystop, checkpoint, lr_schedule]


## === cell 14
STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE

history = model.fit(
            get_training_dataset(),
            epochs = EPOCHS,
            steps_per_epoch = STEPS_PER_EPOCH,
            callbacks = callbacks,
            verbose= 1
)


## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2596747876.py in <cell line: 0>()
----> 1 STEPS_PER_EPOCH = NUM_TRAINING_IMAGES // BATCH_SIZE
      2 
      3 history = model.fit(
      4             get_training_dataset(),
      5             epochs = EPOCHS,

NameError: name 'NUM_TRAINING_IMAGES' is not defined

## === cell 15
model.save('resnet.h5')


## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3527457630.py in <cell line: 0>()
----> 1 model.save('resnet.h5')

NameError: name 'model' is not defined

## === cell 16
test_ds = get_test_dataset(ordered=True)

print('Computing predictions...')
test_images_ds = test_ds.map(lambda image, idnum: image)
probabilities = model.predict(test_images_ds)


## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3329820609.py in <cell line: 0>()
----> 1 test_ds = get_test_dataset(ordered=True)
      2 
      3 print('Computing predictions...')
      4 test_images_ds = test_ds.map(lambda image, idnum: image)
      5 probabilities = model.predict(test_images_ds)

/tmp/ipykernel_11/189272503.py in get_test_dataset(ordered)
     77 
     78 def get_test_dataset(ordered = False):
---> 79     dataset = load_dataset(TEST_FILENAMES, labeled = False, ordered = ordered)
     80     dataset =  dataset.batch(BATCH_SIZE)
     81     dataset = dataset.prefetch(AUTO) # prefetch the next batch while training (autotune prefetch buffer size)

NameError: name 'TEST_FILENAMES' is not defined

## === cell 17
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch


## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3522666981.py in <cell line: 0>()
      1 print('Generating submission.csv file...')
----> 2 test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
      3 test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch

NameError: name 'test_ds' is not defined

## === cell 18
pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
pred_df.head()


## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2484733623.py in <cell line: 0>()
----> 1 pred_df = pd.DataFrame({'image_name': test_ids, 'target': np.concatenate(probabilities)})
      2 pred_df.head()

NameError: name 'test_ids' is not defined

## === cell 19
sub.head()


## === cell 20
del sub['target']
sub = sub.merge(pred_df, on='image_name')
sub.to_csv('submission_b5.csv', index=False)
sub.head()


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/183588590.py in <cell line: 0>()
      1 del sub['target']
----> 2 sub = sub.merge(pred_df, on='image_name')
      3 #sub.to_csv('submission_label_smoothing.csv', index=False)
      4 sub.to_csv('submission_b5.csv', index=False)
      5 sub.head()

NameError: name 'pred_df' is not defined
