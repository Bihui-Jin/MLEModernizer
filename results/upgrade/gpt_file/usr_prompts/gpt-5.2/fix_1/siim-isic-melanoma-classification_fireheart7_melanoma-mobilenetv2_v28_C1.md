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
tqdm==4.67.1

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

0.7964

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
for dirname, _, filenames in os.walk('/kaggle/input'):
    print(os.path.join(dirname))


## === cell 1
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly
import plotly.graph_objects as go
import cv2
import tensorflow as tf
from kaggle_datasets import KaggleDatasets
from functools import partial
import sklearn
from tqdm import tqdm_notebook as tqdm
import gc
%matplotlib inline


## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
except:
    strategy = tf.distribute.get_strategy()


## === cell 3
print('Device:', tpu.master())
print('Number of replicas:', strategy.num_replicas_in_sync)
print("Version of Tensorflow used : ", tf.__version__)


## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2145986948.py in <cell line: 0>()
----> 1 print('Device:', tpu.master())
      2 print('Number of replicas:', strategy.num_replicas_in_sync)
      3 print("Version of Tensorflow used : ", tf.__version__)

NameError: name 'tpu' is not defined

## === cell 4
AUTOTUNE = tf.data.experimental.AUTOTUNE
GCS_PATH = KaggleDatasets().get_gcs_path()
BATCH_SIZE = 16 * strategy.num_replicas_in_sync
IMAGE_SIZE = [1024, 1024]
SHAPE = [256, 256]


## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/1997331617.py in <cell line: 0>()
      1 AUTOTUNE = tf.data.experimental.AUTOTUNE
----> 2 GCS_PATH = KaggleDatasets().get_gcs_path()
      3 BATCH_SIZE = 16 * strategy.num_replicas_in_sync
      4 IMAGE_SIZE = [1024, 1024]
      5 SHAPE = [256, 256]

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
print("Batch Size = ", BATCH_SIZE)
print("GCS Path = ", GCS_PATH)


## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/572600619.py in <cell line: 0>()
----> 1 print("Batch Size = ", BATCH_SIZE)
      2 print("GCS Path = ", GCS_PATH)

NameError: name 'BATCH_SIZE' is not defined

## === cell 6
train = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/train.csv"))
train.head()


## === cell 7
test = pd.DataFrame(pd.read_csv("../input/siim-isic-melanoma-classification/test.csv"))
test.head()


## === cell 8
train.info()


## === cell 9
test.info()


## === cell 10
train_dir = "/kaggle/input/siim-isic-melanoma-classification/jpeg/train/"


## === cell 11
image_names = train["image_name"].values + ".jpg"
random_images = [np.random.choice(image_names) for i in range(4)] # Generates a random sample from a given 1-D array
random_images 


## === cell 12
sample_images = []


## === cell 13
plt.figure(figsize = (12, 8))
for i in range(4) : 
    plt.subplot(2, 2, i + 1) 
    image = cv2.imread(os.path.join(train_dir, random_images[i]))
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    sample_images.append(image)
    plt.imshow(image, cmap = "gray")
    plt.grid(True)
plt.tight_layout()


## === cell 14
def non_local_means_denoising(image) : 
    denoised_image = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    return denoised_image


## === cell 15
sample_image = cv2.imread(os.path.join(train_dir, random_images[0]))
sample_image = cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB)
denoised_image = non_local_means_denoising(sample_image)


plt.figure(figsize = (12, 8))
plt.subplot(1,2,1)
plt.imshow(sample_image, cmap = "gray")
plt.grid(False)
plt.title("Normal Image")

plt.subplot(1,2,2)  
plt.imshow(denoised_image, cmap = "gray")
plt.grid(False)
plt.title("Denoised image")    
plt.tight_layout() 


## === cell 16
def histogram_equalization(image) : 
    image_ycrcb = cv2.cvtColor(image, cv2.COLOR_RGB2YCR_CB)
    y_channel = image_ycrcb[:,:,0] # apply local histogram processing on this channel
    cr_channel = image_ycrcb[:,:,1]
    cb_channel = image_ycrcb[:,:,2]
    
    clahe = cv2.createCLAHE(clipLimit = 2.0, tileGridSize=(8,8))
    equalized = clahe.apply(y_channel)
    equalized_image = cv2.merge([equalized, cr_channel, cb_channel])
    equalized_image = cv2.cvtColor(equalized_image, cv2.COLOR_YCR_CB2RGB)
    return equalized_image


## === cell 17
equalized_image = histogram_equalization(denoised_image)


## === cell 18
plt.figure(figsize = (12, 8))
plt.subplot(1,3,1)
plt.imshow(sample_image, cmap = "gray")
plt.grid(False)
plt.title("Normal Image", fontsize = 14)

plt.subplot(1,3,2)  
plt.imshow(denoised_image, cmap = "gray")
plt.grid(False)
plt.title("denoised image after histogram processing", fontsize = 14)

plt.subplot(1,3,3)  
plt.imshow(equalized_image, cmap = "gray")
plt.grid(False)
plt.title("Histogram equalized image", fontsize = 14)
plt.tight_layout()


## === cell 19
def segmentation(image, k, attempts) : 
    vectorized = np.float32(image.reshape((-1, 3)))
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 20, 1.0)
    res , label , center = cv2.kmeans(vectorized, k, None, criteria, attempts, cv2.KMEANS_PP_CENTERS)
    center = np.uint8(center)
    res = center[label.flatten()]
    segmented_image = res.reshape((image.shape))
    return segmented_image


## === cell 20
plt.figure(figsize = (12, 8))
plt.subplot(1,1,1)
plt.imshow(denoised_image, cmap = "gray")
plt.grid(False)
plt.title("de Noised Image")


## === cell 21
plt.figure(figsize = (12, 8))
segmented_image = segmentation(denoised_image, 3, 10) # k = 3, attempt = 10
plt.subplot(1,3,1)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Image with k = 3")

segmented_image = segmentation(denoised_image, 4, 10) # k = 4, attempt = 10
plt.subplot(1,3,2)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Image with k = 4")

segmented_image = segmentation(denoised_image, 5, 10) # k = 5, attempt = 10
plt.subplot(1,3,3)
plt.imshow(segmented_image, cmap = "gray")
plt.grid(False)
plt.title("Segmented Image with k = 5")


## === cell 22
from sklearn.model_selection import train_test_split 
training_files, validation_files = train_test_split(tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec"),
                                                   test_size = 0.1, random_state = 42)

testing_files = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")

print("Number of training files = ", len(training_files))
print("Number of validation files = ", len(validation_files))
print("Number of test files = ", len(testing_files))


## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/665935033.py in <cell line: 0>()
      1 from sklearn.model_selection import train_test_split
----> 2 training_files, validation_files = train_test_split(tf.io.gfile.glob(GCS_PATH + "/tfrecords/train*.tfrec"),
      3                                                    test_size = 0.1, random_state = 42)
      4 
      5 testing_files = tf.io.gfile.glob(GCS_PATH + "/tfrecords/test*.tfrec")

NameError: name 'GCS_PATH' is not defined

## === cell 23
def decode_image(image) : 
    image = tf.image.decode_jpeg(image, channels = 3)
    image = tf.cast(image, tf.float32)
    image = image / 255.0
    image = tf.reshape(image, [IMAGE_SIZE[0], IMAGE_SIZE[1], 3])
    return image


## === cell 24
sample_images[0].shape


## === cell 25
training_files


## --- ERROR in cell 25, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1916863276.py in <cell line: 0>()
----> 1 training_files

NameError: name 'training_files' is not defined

## === cell 26
sample_picked = training_files[0]
sample_picked


## --- ERROR in cell 26, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2979101347.py in <cell line: 0>()
----> 1 sample_picked = training_files[0]
      2 sample_picked

NameError: name 'training_files' is not defined

## === cell 27
file = tf.data.TFRecordDataset(sample_picked)
file


## --- ERROR in cell 27, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/296233215.py in <cell line: 0>()
----> 1 file = tf.data.TFRecordDataset(sample_picked)
      2 file

NameError: name 'sample_picked' is not defined

## === cell 28
feature_description = {"image" : tf.io.FixedLenFeature([], tf.string), 
                      "target" : tf.io.FixedLenFeature([], tf.int64)}


## === cell 29
def parse_function(example) : 
    return tf.io.parse_single_example(example, feature_description)


## === cell 30
parsed_dataset = file.map(parse_function)
parsed_dataset


## --- ERROR in cell 30, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/263189074.py in <cell line: 0>()
----> 1 parsed_dataset = file.map(parse_function)
      2 parsed_dataset

NameError: name 'file' is not defined

## === cell 31
def read_tfrecord(example, labeled) : 
    if labeled == True : 
        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),
                           "target" : tf.io.FixedLenFeature([], tf.int64)}
    else:
        tfrecord_format = {"image" : tf.io.FixedLenFeature([], tf.string),
                          "image_name" : tf.io.FixedLenFeature([], tf.string)}
    
    example = tf.io.parse_single_example(example, tfrecord_format)
    image = decode_image(example["image"])
    if labeled == True : 
        label = tf.cast(example["target"], tf.int32)
        return image, label
    else:
        image_name = example["image_name"]
        return image, image_name     


## === cell 32
def load_dataset(filenames, labeled, ordered):
    ignore_order = tf.data.Options()
    if ordered == False: # dataset is unordered, so we ignore the order to load data quickly.
        ignore_order.experimental_deterministic = False # This disables the order and enhances the speed
    dataset = tf.data.TFRecordDataset(filenames, num_parallel_reads=AUTOTUNE) 
    dataset = dataset.with_options(ignore_order) 
    dataset = dataset.map(partial(read_tfrecord, labeled=labeled), num_parallel_calls=AUTOTUNE)
    return dataset


## === cell 33
def image_augmentation(image, label) :     
    image = tf.image.resize(image, SHAPE)
    image = tf.image.random_flip_left_right(image)
    return image, label


## === cell 34
def get_training_dataset() : 
    dataset = load_dataset(training_files, labeled = True, ordered = False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.repeat()
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


## === cell 35
def get_validation_dataset() : 
    dataset = load_dataset(validation_files, labeled = True, ordered = False)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


## === cell 36
def get_test_dataset() : 
    dataset = load_dataset(testing_files, labeled = False, ordered = True)
    dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
    dataset = dataset.batch(BATCH_SIZE)
    dataset = dataset.cache()
    dataset = dataset.prefetch(AUTOTUNE) 
    return dataset


## === cell 37
training_dataset = get_training_dataset()


## --- ERROR in cell 37, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3862791175.py in <cell line: 0>()
----> 1 training_dataset = get_training_dataset()

/tmp/ipykernel_11/63298333.py in get_training_dataset()
      1 def get_training_dataset() :
----> 2     dataset = load_dataset(training_files, labeled = True, ordered = False)
      3     dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
      4     dataset = dataset.repeat()
      5     dataset = dataset.batch(BATCH_SIZE)

NameError: name 'training_files' is not defined

## === cell 38
validation_dataset = get_validation_dataset()


## --- ERROR in cell 38, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/967353300.py in <cell line: 0>()
----> 1 validation_dataset = get_validation_dataset()

/tmp/ipykernel_11/3077536662.py in get_validation_dataset()
      1 def get_validation_dataset() :
----> 2     dataset = load_dataset(validation_files, labeled = True, ordered = False)
      3     dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
      4     dataset = dataset.batch(BATCH_SIZE)
      5     dataset = dataset.cache()

NameError: name 'validation_files' is not defined

## === cell 39
def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

num_training_images = count_data_items(training_files)
num_validation_images = count_data_items(validation_files)
num_testing_images = count_data_items(testing_files)

STEPS_PER_EPOCH_TRAIN = num_training_images // BATCH_SIZE
STEPS_PER_EPOCH_VAL = num_validation_images // BATCH_SIZE

print("Number of Training Images = ", num_training_images)
print("Number of Validation Images = ", num_validation_images)
print("Number of Testing Images = ", num_testing_images)
print("\n")
print("Numer of steps per epoch in Train = ", STEPS_PER_EPOCH_TRAIN)
print("Numer of steps per epoch in Validation = ", STEPS_PER_EPOCH_VAL)


## --- ERROR in cell 39, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4090651076.py in <cell line: 0>()
      3     return np.sum(n)
      4 
----> 5 num_training_images = count_data_items(training_files)
      6 num_validation_images = count_data_items(validation_files)
      7 num_testing_images = count_data_items(testing_files)

NameError: name 'training_files' is not defined

## === cell 40
image_batch, label_batch = next(iter(training_dataset))


## --- ERROR in cell 40, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/632975622.py in <cell line: 0>()
----> 1 image_batch, label_batch = next(iter(training_dataset))

NameError: name 'training_dataset' is not defined

## === cell 41
def show_batch(image_batch, label_batch) :
    plt.figure(figsize = (20, 20))
    for n in range(8) : 
        ax = plt.subplot(2,4,n+1)
        plt.imshow(image_batch[n])
        if label_batch[n] == 0 : 
            plt.title("BENIGN")
        else:
            plt.title("MALIGNANT")
    plt.grid(False)
    plt.tight_layout()       


## === cell 42
show_batch(image_batch.numpy(), label_batch.numpy())


## --- ERROR in cell 42, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3295174013.py in <cell line: 0>()
----> 1 show_batch(image_batch.numpy(), label_batch.numpy())

NameError: name 'image_batch' is not defined

## === cell 43
del image_batch
del label_batch
gc.collect()


## --- ERROR in cell 43, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3158841352.py in <cell line: 0>()
----> 1 del image_batch
      2 del label_batch
      3 gc.collect()

NameError: name 'image_batch' is not defined

## === cell 44
malignant = len(train[train["target"] == 1])
benign = len(train[train["target"] == 0 ])
total = len(train) 

print("Malignant Cases in Train Data = ", malignant)
print("Benign Cases In Train Dataset = ",benign)
print("Total Cases In Train Dataset = ",total)
print("Ratio of Malignant to Benign = ",malignant/benign)


## === cell 45
weight_malignant = (total/malignant)/2.0
weight_benign = (total/benign)/2.0

class_weight = {0 : weight_benign , 1 : weight_malignant}

print("Weight for benign cases = ", class_weight[0])
print("Weight for malignant cases = ", class_weight[1])


## === cell 46
callback_early_stopping = tf.keras.callbacks.EarlyStopping(patience = 15, verbose = 0, restore_best_weights = True)

callbacks_lr_reduce = tf.keras.callbacks.ReduceLROnPlateau(monitor = "val_auc", factor = 0.1, patience = 10, 
                                                          verbose = 0, min_lr = 1e-6)

callback_checkpoint = tf.keras.callbacks.ModelCheckpoint("melanoma_weights.h5",
                                                         save_weights_only=True, monitor='val_auc',
                                                         mode='max', save_best_only = True)


## --- ERROR in cell 46, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1479813516.py in <cell line: 0>()
      4                                                           verbose = 0, min_lr = 1e-6)
      5 
----> 6 callback_checkpoint = tf.keras.callbacks.ModelCheckpoint("melanoma_weights.h5",
      7                                                          save_weights_only=True, monitor='val_auc',
      8                                                          mode='max', save_best_only = True)

/usr/local/lib/python3.11/dist-packages/keras/src/callbacks/model_checkpoint.py in __init__(self, filepath, monitor, verbose, save_best_only, save_weights_only, mode, save_freq, initial_value_threshold)
    182         if save_weights_only:
    183             if not self.filepath.endswith(".weights.h5"):
--> 184                 raise ValueError(
    185                     "When using `save_weights_only=True` in `ModelCheckpoint`"
    186                     ", the filepath provided must end in `.weights.h5` "

ValueError: When using `save_weights_only=True` in `ModelCheckpoint`, the filepath provided must end in `.weights.h5` (Keras weights format). Received: filepath=melanoma_weights.h5

## === cell 47
with strategy.scope() : 
    bias = np.log(malignant/benign)
    bias = tf.keras.initializers.Constant(bias)
    base_model = tf.keras.applications.MobileNetV2(input_shape = (SHAPE[0], SHAPE[1], 3), include_top = False,
                                               weights = "imagenet")
    base_model.trainable = False
    model = tf.keras.Sequential([base_model,
                                 tf.keras.layers.GlobalAveragePooling2D(),
                                 tf.keras.layers.Dense(20, activation = "relu"),
                                 tf.keras.layers.Dropout(0.4),
                                 tf.keras.layers.Dense(10, activation = "relu"),
                                 tf.keras.layers.Dropout(0.3),
                                 tf.keras.layers.Dense(1, activation = "sigmoid", bias_initializer = bias)                                     
                                ])
    model.compile(optimizer = tf.keras.optimizers.Adam(lr = 1e-2), loss = "binary_crossentropy", metrics = [tf.keras.metrics.AUC(name = 'auc')])
    model.summary()
    
    EPOCHS = 500
    history = model.fit(training_dataset, epochs = EPOCHS, steps_per_epoch = STEPS_PER_EPOCH_TRAIN,
                       validation_data = validation_dataset, validation_steps = STEPS_PER_EPOCH_VAL,
                       callbacks = [callback_early_stopping, callbacks_lr_reduce, callback_checkpoint],
                       class_weight = class_weight)


## --- ERROR in cell 47, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/785082977.py in <cell line: 0>()
      2     bias = np.log(malignant/benign)
      3     bias = tf.keras.initializers.Constant(bias)
----> 4     base_model = tf.keras.applications.MobileNetV2(input_shape = (SHAPE[0], SHAPE[1], 3), include_top = False,
      5                                                weights = "imagenet")
      6     base_model.trainable = False

NameError: name 'SHAPE' is not defined

## === cell 48
n_epochs_it_ran_for = len(history.history['loss'])
n_epochs_it_ran_for


## --- ERROR in cell 48, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3645842311.py in <cell line: 0>()
----> 1 n_epochs_it_ran_for = len(history.history['loss'])
      2 n_epochs_it_ran_for

NameError: name 'history' is not defined

## === cell 49
X = np.arange(0,n_epochs_it_ran_for,1)
plt.figure(1, figsize = (20, 12))
plt.subplot(1,2,1)
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.plot(X, history.history["loss"], label = "Training Loss")
plt.plot(X, history.history["val_loss"], label = "Validation Loss")
plt.grid(True)
plt.legend()

plt.subplot(1,2,2)
plt.xlabel("Epochs")
plt.ylabel("Accuracy")
plt.plot(X, history.history["auc"], label = "Training Accuracy")
plt.plot(X, history.history["val_auc"], label = "Validation Accuracy")
plt.grid(True)
plt.legend()


## --- ERROR in cell 49, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4033153224.py in <cell line: 0>()
----> 1 X = np.arange(0,n_epochs_it_ran_for,1)
      2 plt.figure(1, figsize = (20, 12))
      3 plt.subplot(1,2,1)
      4 plt.xlabel("Epochs")
      5 plt.ylabel("Loss")

NameError: name 'n_epochs_it_ran_for' is not defined

## === cell 50
testing_dataset = get_test_dataset()
testing_dataset_images = testing_dataset.map(lambda image, image_name : image)
testing_image_names = testing_dataset.map(lambda image, image_name : image_name)


## --- ERROR in cell 50, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1607468938.py in <cell line: 0>()
----> 1 testing_dataset = get_test_dataset()
      2 testing_dataset_images = testing_dataset.map(lambda image, image_name : image)
      3 testing_image_names = testing_dataset.map(lambda image, image_name : image_name)

/tmp/ipykernel_11/590774169.py in get_test_dataset()
      1 def get_test_dataset() :
----> 2     dataset = load_dataset(testing_files, labeled = False, ordered = True)
      3     dataset = dataset.map(image_augmentation, num_parallel_calls=AUTOTUNE)
      4     dataset = dataset.batch(BATCH_SIZE)
      5     dataset = dataset.cache()

NameError: name 'testing_files' is not defined

## === cell 51
resulting_probabilities = model.predict(testing_dataset_images, verbose = 1)


## --- ERROR in cell 51, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2029566039.py in <cell line: 0>()
----> 1 resulting_probabilities = model.predict(testing_dataset_images, verbose = 1)

NameError: name 'model' is not defined

## === cell 52
len(resulting_probabilities)


## --- ERROR in cell 52, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1344563961.py in <cell line: 0>()
----> 1 len(resulting_probabilities)

NameError: name 'resulting_probabilities' is not defined

## === cell 53
sample_submission_file = pd.read_csv("../input/siim-isic-melanoma-classification/sample_submission.csv")
sample_submission_file.head()


## === cell 54
del sample_submission_file["target"]
sample_submission_file.head()


## === cell 55
testing_image_names


## --- ERROR in cell 55, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2698235127.py in <cell line: 0>()
----> 1 testing_image_names

NameError: name 'testing_image_names' is not defined

## === cell 56
testing_image_names = np.concatenate([x for x in testing_image_names], axis=0)
testing_image_names = np.array(testing_image_names)


## --- ERROR in cell 56, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2762588620.py in <cell line: 0>()
----> 1 testing_image_names = np.concatenate([x for x in testing_image_names], axis=0)
      2 testing_image_names = np.array(testing_image_names)

NameError: name 'testing_image_names' is not defined

## === cell 57
decoded_test_names = []
for names in testing_image_names : 
    names = names.decode('utf-8')
    decoded_test_names.append(names)
decoded_test_names = np.array(decoded_test_names)
del testing_image_names


## --- ERROR in cell 57, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3300792048.py in <cell line: 0>()
      1 decoded_test_names = []
----> 2 for names in testing_image_names :
      3     names = names.decode('utf-8')
      4     decoded_test_names.append(names)
      5 decoded_test_names = np.array(decoded_test_names)

NameError: name 'testing_image_names' is not defined

## === cell 58
len(decoded_test_names), type(decoded_test_names), decoded_test_names.shape


## --- ERROR in cell 58, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/110328696.py in <cell line: 0>()
----> 1 len(decoded_test_names), type(decoded_test_names), decoded_test_names.shape

AttributeError: 'list' object has no attribute 'shape'

## === cell 59
decoded_test_names


## === cell 60
testing_image_names = pd.DataFrame(decoded_test_names, columns=["image_name"])
testing_image_names.head()


## === cell 61
pred_dataframe = pd.DataFrame({"image_name" : decoded_test_names, 
                               "target" : np.concatenate(resulting_probabilities)})
pred_dataframe


## --- ERROR in cell 61, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/501859269.py in <cell line: 0>()
      1 pred_dataframe = pd.DataFrame({"image_name" : decoded_test_names, 
----> 2                                "target" : np.concatenate(resulting_probabilities)})
      3 pred_dataframe

NameError: name 'resulting_probabilities' is not defined

## === cell 62
sample_submission_file = sample_submission_file.merge(pred_dataframe, on = "image_name")
sample_submission_file.to_csv("submission.csv", index = False)
sample_submission_file.head()


## --- ERROR in cell 62, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2244097225.py in <cell line: 0>()
----> 1 sample_submission_file = sample_submission_file.merge(pred_dataframe, on = "image_name")
      2 sample_submission_file.to_csv("submission.csv", index = False)
      3 sample_submission_file.head()

NameError: name 'pred_dataframe' is not defined
