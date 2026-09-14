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
Detect the presence and position of catheters and lines on chest x-rays.

## Metric
Area under the ROC curve for each label, with the final score being the average of the individual AUCs of each predicted column.

## Submission Format
For each ID in the test set, you must predict a probability for all target variables. The file should contain a header and have the following format:
```
StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present
1.2.826.0.1.3680043.8.498.62451881164053375557257228990443168843,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.83721761279899623084220697845011427274,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.12732270010839808189235995393981377825,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.11769539755086084996287023095028033598,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.87838627504097587943394933987052577153,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.53211840524738036417560823327351887819,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.93555795394184819372299157360228027866,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.52241894131170494723503100795076463919,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.36500167484503936720548852591033878284,0,0,0,0,0,0,0,0,0,0,0
1.2.826.0.1.3680043.8.498.86199852603457900780565655267977637728,0,0,0,0,0,0,0,0,0,0,0
```

## Dataset
`train.csv` contains image IDs, binary labels, and patient IDs.

TFRecords are available for both train and test.

We've also included `train_annotations.csv`. These are segmentation annotations for training samples that have them. They are included solely as additional information for competitors.

- train.csv - contains image IDs, binary labels, and patient IDs.
- sample_submission.csv - a sample submission file in the correct format
- test - test images
- train - training images

### Columns
- `StudyInstanceUID` - unique ID for each image
- `ETT - Abnormal` - endotracheal tube placement abnormal
- `ETT - Borderline` - endotracheal tube placement borderline abnormal
- `ETT - Normal` - endotracheal tube placement normal
- `NGT - Abnormal` - nasogastric tube placement abnormal
- `NGT - Borderline` - nasogastric tube placement borderline abnormal
- `NGT - Incompletely Imaged` - nasogastric tube placement inconclusive due to imaging
- `NGT - Normal` - nasogastric tube placement borderline normal
- `CVC - Abnormal` - central venous catheter placement abnormal
- `CVC - Borderline` - central venous catheter placement borderline abnormal
- `CVC - Normal` - central venous catheter placement normal
- `Swan Ganz Catheter Present`
- `PatientID` - unique ID for each patient in the dataset

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
        input/
            description.md (172 lines)
            sample_submission.csv (3010 lines)
            sample_submission.csv.zip (64.2 kB)
            test.zip (642.8 MB)
            train.csv (27075 lines)
            train.csv.zip (798.6 kB)
            train.zip (5.8 GB)
            train_annotations.csv (16262 lines)
            train_annotations.csv.zip (1.4 MB)
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
            test/
                1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                ... and 3007 other files
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
            train/
                1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                ... and 27072 other files
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
        working/
            ranzcr-clip-catheter-line-classification/
                description.md (172 lines)
                sample_submission.csv (3010 lines)
                ... and 7 other files
                ranzcr-clip-catheter-line-classification/
                test/
                    1.2.826.0.1.3680043.8.498.57477258980718966370268402373568359767.jpg (201.5 kB)
                    1.2.826.0.1.3680043.8.498.70369997506092680321332747830211112877.jpg (176.2 kB)
                    ... and 3007 other files
                    test/
                train/
                    1.2.826.0.1.3680043.8.498.22695482406757766723043400617436591679.jpg (306.8 kB)
                    1.2.826.0.1.3680043.8.498.67485046061813043086989609654834659261.jpg (315.8 kB)
                    ... and 27072 other files
                    train/
```

-> data/ranzcr-clip-catheter-line-classification/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/ranzcr-clip-catheter-line-classification/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/ranzcr-clip-catheter-line-classification/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> data/sample_submission.csv has 3009 rows and 10 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline

-> data/train.csv has 27074 rows and 13 columns.
The columns are: StudyInstanceUID, ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal, NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal, CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present, PatientID

-> data/train_annotations.csv has 16261 rows and 3 columns.
The columns are: StudyInstanceUID, label, data

-> (stopped after 10 files for performance)

# 5. Target score

0.7393840001637286

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 2
import math, os, re, warnings, random, glob
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow.keras.layers as L
import tensorflow.keras.backend as K
from tensorflow.keras import Sequential
from kaggle_datasets import KaggleDatasets

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print(f'Running on TPU {tpu.master()}')
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = tf.distribute.get_strategy()

AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print(f'REPLICAS: {REPLICAS}')

## === cell 6
BATCH_SIZE = 16 * REPLICAS
HEIGHT = 512
WIDTH = 512 
CHANNELS = 3
N_CLASSES = 5
TTA_STEPS = 3 # Do TTA if > 0 
IMAGE_SIZE = [512, 512] # At this size, a GPU will run out of memory. Use the TPU.
SEED =555    
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
AUG_BATCH = BATCH_SIZE

## === cell 8
def data_augment(image, label):
    image = tf.image.rot90(image,k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image , seed=SEED)
    image=  image = tf.image.random_flip_up_down(image, seed=SEED)
    IMG_SIZE=IMAGE_SIZE[0]
    image = tf.image.resize_with_crop_or_pad(image, IMG_SIZE + 6, IMG_SIZE + 6) 
    image = tf.image.random_crop(image, size=[IMG_SIZE, IMG_SIZE, 3])
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    
    return image, label 

## === cell 10
def to_float32_2(image, label):
    max_val = tf.reduce_max(label, axis=-1,keepdims=True)
    cond = tf.equal(label, max_val)
    label = tf.where(cond, tf.ones_like(label), tf.zeros_like(label))
    return tf.cast(image, tf.float32), tf.cast(label, tf.int32)

def to_float32(image, label):
    return tf.cast(image, tf.float32), label

def decode_image(image_data):
    image = tf.image.decode_jpeg(image_data, channels=3)
    image = tf.cast(image, tf.float32) / 255.0  # convert image to floats in [0, 1] range
    image = tf.reshape(image, [1024,1024, 3]) # explicit size needed for TPU
    return image





def read_labeled_tfrecord(example):
    train_feature_description = {
        "CVC - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "CVC - Normal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "ETT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Abnormal": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Borderline": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Incompletely Imaged": tf.io.FixedLenFeature([], tf.int64),
        "NGT - Normal": tf.io.FixedLenFeature([], tf.int64),
        "StudyInstanceUID" : tf.io.FixedLenFeature([], tf.string),
        "Swan Ganz Catheter Present" : tf.io.FixedLenFeature([], tf.int64),
        "image" : tf.io.FixedLenFeature([], tf.string),
    }
    example = tf.io.parse_single_example(example, train_feature_description)
    image = decode_image(example['image'])    
    uid= example["StudyInstanceUID"]
    cvca = example["CVC - Abnormal"]
    cvcb = example["CVC - Borderline"]
    cvcn = example["CVC - Normal"]
    etta = example["ETT - Abnormal"]
    ettb = example["ETT - Borderline"]
    ettn = example["ETT - Normal"]
    ngta = example["NGT - Abnormal"]
    ngtb = example["NGT - Borderline"]
    ngti = example["NGT - Incompletely Imaged"]
    ngtn = example["NGT - Normal"]
    sgcp = example["Swan Ganz Catheter Present"]

    values  = [  etta, ettb, ettn, ngta, ngtb, ngti, ngtn,cvca, cvcb, cvcn , sgcp]
    label = tf.cast(0, tf.int32)
    for i in range(len(values)):
        if ( values[i]==1):
            label = tf.cast(i, tf.int32)
    return image,label # returns a dataset of (image, label) pairs

def read_unlabeled_tfrecord(example):
    UNLABELED_TFREC_FORMAT  = {
    "StudyInstanceUID" : tf.io.FixedLenFeature([], tf.string),
    "image" : tf.io.FixedLenFeature([], tf.string)
    }
    example = tf.io.parse_single_example(example, UNLABELED_TFREC_FORMAT)
    image = decode_image(example['image'])
    image= tf.image.resize(image, [IMAGE_SIZE[0],IMAGE_SIZE[0]])
    image_name = example['StudyInstanceUID']
    return image, image_name # returns a dataset of image(s)

def load_dataset(filenames, labeled=True, ordered=False):

    ignore_order = tf.data.Options()
    if not ordered:
        ignore_order.experimental_deterministic = False # disable order, increase speed

    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTO) # automatically interleaves reads from multiple files
    dataset = dataset.with_options(ignore_order) # uses data as soon as it streams in, rather than in its original order
    dataset = dataset.map(read_labeled_tfrecord if labeled else read_unlabeled_tfrecord, num_parallel_calls=AUTO)
    return dataset

def data_augment(image, label):
    image = tf.image.rot90(image,k=np.random.randint(4))
    image = tf.image.random_flip_left_right(image , seed=SEED)
    image= tf.image.random_flip_up_down(image, seed=SEED)
    IMG_SIZE=IMAGE_SIZE[0]
    image = tf.image.random_brightness(image, max_delta=0.5) # Random brightness
    image = tf.image.random_saturation(image, 0, 2, seed=SEED)
    image = tf.image.adjust_saturation(image, 3)
    
    return image, label   

def get_training_dataset():
    dataset = load_dataset(TRAINING_FILENAMES, labeled=True)
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_training_dataset(dataset, do_aug=True , do_onehot=False):
    dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.repeat() # the training dataset must repeat for several epochs
    dataset = dataset.batch(AUG_BATCH)
    if do_onehot: dataset = dataset.map(onehot, num_parallel_calls=AUTO) 
    dataset = dataset.unbatch()
    
    dataset = dataset.shuffle(2048)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_validation_dataset(ordered=False):
    dataset = load_dataset(VALIDATION_FILENAMES, labeled=True, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def get_test_dataset(ordered=False , tta= False):
    dataset = load_dataset(TEST_FILENAMES, labeled=False, ordered=ordered)
    dataset = dataset.batch(BATCH_SIZE)
    if tta:
        dataset = dataset.map(data_augment, num_parallel_calls=AUTO)
    dataset = dataset.prefetch(AUTO) # prefetch next batch while training (autotune prefetch buffer size)
    return dataset

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)


## === cell 12
database_base_path = '/kaggle/input/ranzcr-clip-catheter-line-classification/'
submission = pd.read_csv(f'{database_base_path}sample_submission.csv')
display(submission.head())
TEST_FILENAMES = tf.io.gfile.glob(f'{database_base_path}test_tfrecords/*.tfrec') # predic
NUM_TEST_IMAGES = count_data_items(TEST_FILENAMES)
print(f'GCS: test: {NUM_TEST_IMAGES}')

## === cell 14
model_path_list = glob.glob('/kaggle/input/ranzcr-clip/model*.h5')
model_path_list.sort()

print('Models to predict:')
print(*model_path_list, sep='\n')

## === cell 16
from tensorflow import keras
    
models = []    
i = 0
for model_path in model_path_list:
    print(model_path)
    K.clear_session()
    models.append(keras.models.load_model(model_path))

## === cell 18
print(" TTA_STEPS = {} ".format(TTA_STEPS))
if TTA_STEPS > 0:
    for step in range(TTA_STEPS):
        test_ds = get_test_dataset(ordered=True, tta=True)
        print(f'TTA step {step+1}/{TTA_STEPS}')
        test_images_ds = test_ds.map(lambda image, image_name: image)
        probabilities = np.average([models[i].predict(test_images_ds) for i in range(len(models))], axis = 0)
else:
    test_ds = get_test_dataset(ordered=True, tta=True)
    test_images_ds = test_ds.map(lambda image, image_name: image)
    probabilities = np.average([models[i].predict(test_images_ds) for i in range(1)], axis = 0)



## === cell 20
print('Generating submission.csv file...')
test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
np.savetxt('submission.csv', np.rec.fromarrays([test_ids] +  [probabilities[:,i] for i in range(probabilities.shape[1])]), fmt=['%s', '%f','%f' , '%f', '%f','%f' , '%f', '%f','%f' , '%f', '%f','%f'  ], delimiter=',', header='StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present', comments='')


## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/366083465.py in <cell line: 0>()
      1 print('Generating submission.csv file...')
      2 test_ids_ds = test_ds.map(lambda image, idnum: idnum).unbatch()
----> 3 test_ids = next(iter(test_ids_ds.batch(NUM_TEST_IMAGES))).numpy().astype('U') # all in one batch
      4 np.savetxt('submission.csv', np.rec.fromarrays([test_ids] +  [probabilities[:,i] for i in range(probabilities.shape[1])]), fmt=['%s', '%f','%f' , '%f', '%f','%f' , '%f', '%f','%f' , '%f', '%f','%f'  ], delimiter=',', header='StudyInstanceUID,ETT - Abnormal,ETT - Borderline,ETT - Normal,NGT - Abnormal,NGT - Borderline,NGT - Incompletely Imaged,NGT - Normal,CVC - Abnormal,CVC - Borderline,CVC - Normal,Swan Ganz Catheter Present', comments='')

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in batch(self, batch_size, drop_remainder, num_parallel_calls, deterministic, name)
   1913     # pylint: disable=g-import-not-at-top,protected-access,redefined-outer-name
   1914     from tensorflow.python.data.ops import batch_op
-> 1915     return batch_op._batch(self, batch_size, drop_remainder, num_parallel_calls,
   1916                            deterministic, name)
   1917     # pylint: enable=g-import-not-at-top,protected-access,redefined-outer-name

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/batch_op.py in _batch(input_dataset, batch_size, drop_remainder, num_parallel_calls, deterministic, name)
     37       warnings.warn("The `deterministic` argument has no effect unless the "
     38                     "`num_parallel_calls` argument is specified.")
---> 39     return _BatchDataset(input_dataset, batch_size, drop_remainder, name=name)
     40   else:
     41     return _ParallelBatchDataset(

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/batch_op.py in __init__(self, input_dataset, batch_size, drop_remainder, name)
     75 
     76     self._name = name
---> 77     variant_tensor = gen_dataset_ops.batch_dataset_v2(
     78         input_dataset._variant_tensor,
     79         batch_size=self._batch_size,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in batch_dataset_v2(input_dataset, batch_size, drop_remainder, output_types, output_shapes, parallel_copy, metadata, name)
    778       return _result
    779     except _core._NotOkStatusException as e:
--> 780       _ops.raise_from_not_ok_status(e, name)
    781     except _core._FallbackException:
    782       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

InvalidArgumentError: {{function_node __wrapped__BatchDatasetV2_device_/job:localhost/replica:0/task:0/device:CPU:0}} Batch size must be greater than zero. [Op:BatchDatasetV2] name: 

## === cell 21
!head submission.csv
