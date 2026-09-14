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

No external packages required in the script and installed.

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

0.8687963943891686

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
!pip install efficientnet > /dev/null

## === cell 1
import os
import re
import random
import pandas as pd
import numpy as np
import math

import tensorflow as tf
import tensorflow.keras.backend as K
from kaggle_datasets import KaggleDatasets
import efficientnet.tfkeras as efn

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 2
GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
TEST_FILES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')

BATCH_SIZE = 256

## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/3624454637.py in <cell line: 0>()
----> 1 GCS_DS_PATH = KaggleDatasets().get_gcs_path('siim-isic-melanoma-classification')
      2 TEST_FILES = tf.io.gfile.glob(GCS_DS_PATH + '/tfrecords/test*')
      3 
      4 BATCH_SIZE = 256

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

## === cell 3
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

def count_data_items(filenames):
    n = [int(re.compile(r"-([0-9]*)\.").search(filename).group(1)) for filename in filenames]
    return np.sum(n)

## === cell 4
ROT_ = 180.0
SHR_ = 2.0
HZOOM_ = 8.0
WZOOM_ = 8.0
HSHIFT_ = 8.0
WSHIFT_ = 8.0

def get_mat(rotation, shear, height_zoom, width_zoom, height_shift, width_shift):
        
    rotation = math.pi * rotation / 180.
    shear    = math.pi * shear    / 180.

    def get_3x3_mat(lst):
        return tf.reshape(tf.concat([lst],axis=0), [3,3])
    
    c1   = tf.math.cos(rotation)
    s1   = tf.math.sin(rotation)
    one  = tf.constant([1],dtype='float32')
    zero = tf.constant([0],dtype='float32')
    
    rotation_matrix = get_3x3_mat([c1,   s1,   zero, 
                                   -s1,  c1,   zero, 
                                   zero, zero, one])    
    c2 = tf.math.cos(shear)
    s2 = tf.math.sin(shear)    
    
    shear_matrix = get_3x3_mat([one,  s2,   zero, 
                                zero, c2,   zero, 
                                zero, zero, one])        
    zoom_matrix = get_3x3_mat([one/height_zoom, zero,           zero, 
                               zero,            one/width_zoom, zero, 
                               zero,            zero,           one])    
    shift_matrix = get_3x3_mat([one,  zero, height_shift, 
                                zero, one,  width_shift, 
                                zero, zero, one])
    
    return K.dot(K.dot(rotation_matrix, shear_matrix), 
                 K.dot(zoom_matrix,     shift_matrix))


def transform(image, DIM=256):    
    XDIM = DIM%2 #fix for size 331
    
    rot = ROT_ * tf.random.normal([1], dtype='float32')
    shr = SHR_ * tf.random.normal([1], dtype='float32') 
    h_zoom = 1.0 + tf.random.normal([1], dtype='float32') / HZOOM_
    w_zoom = 1.0 + tf.random.normal([1], dtype='float32') / WZOOM_
    h_shift = HSHIFT_ * tf.random.normal([1], dtype='float32') 
    w_shift = WSHIFT_ * tf.random.normal([1], dtype='float32') 

    m = get_mat(rot,shr,h_zoom,w_zoom,h_shift,w_shift) 

    x   = tf.repeat(tf.range(DIM//2, -DIM//2,-1), DIM)
    y   = tf.tile(tf.range(-DIM//2, DIM//2), [DIM])
    z   = tf.ones([DIM*DIM], dtype='int32')
    idx = tf.stack( [x,y,z] )
    
    idx2 = K.dot(m, tf.cast(idx, dtype='float32'))
    idx2 = K.cast(idx2, dtype='int32')
    idx2 = K.clip(idx2, -DIM//2+XDIM+1, DIM//2)
    
    idx3 = tf.stack([DIM//2-idx2[0,], DIM//2-1+idx2[1,]])
    d    = tf.gather_nd(image, tf.transpose(idx3))
        
    return tf.reshape(d,[DIM, DIM,3])

def data_augment(image, label):
    image = transform(image, DIM=128)
    image = tf.image.random_flip_left_right(image)
    image = tf.image.rot90(image)
    
    return image, label  

def process_test_data(data_file):
    LABELED_TFREC_FORMAT = {
        "image": tf.io.FixedLenFeature([], tf.string),
        "image_name": tf.io.FixedLenFeature([], tf.string),
    }
    data = tf.io.parse_single_example(data_file, LABELED_TFREC_FORMAT)
    img = tf.image.decode_jpeg(data['image'], channels=3)
    img = tf.image.resize(img, (128,128))
    img = tf.cast(img, tf.float32) / 255.0
    img = tf.reshape(img, [128,128, 3])

    idnum = data['image_name']

    return img, idnum

## === cell 5
def efficientnetbx():
    return tf.keras.Sequential([
                    efn.EfficientNetB0(
                        input_shape=(128,128, 3),
                        include_top=False
                    ),
                    tf.keras.layers.GlobalAveragePooling2D(),
                    tf.keras.layers.Dense(1, activation='sigmoid')
                ])


## === cell 6
TTA = 11
if strategy != None:
    with strategy.scope():
        model = efficientnetbx()
else:
    model = efficientnetbx()

ignore_order = tf.data.Options()
test_dataset = (
    tf.data.TFRecordDataset(
        TEST_FILES,  
        num_parallel_reads=tf.data.experimental.AUTOTUNE
    ).with_options(
        ignore_order
    ).map(
        process_test_data,
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).map(
        data_augment, 
        num_parallel_calls=tf.data.experimental.AUTOTUNE
    ).repeat(
    ).batch(
        BATCH_SIZE * 4 
    ).prefetch(
        tf.data.experimental.AUTOTUNE
    )
)



model.load_weights("../input/efficientnetb0-128x128/fold-3.h5")

test_imgs = test_dataset.map(lambda images, ids: images)
img_ids_ds = test_dataset.map(lambda images, ids: ids).unbatch()

ct_test = count_data_items(TEST_FILES)
STEPS = TTA * ct_test/(BATCH_SIZE * 4)
pred = model.predict(test_imgs,steps=STEPS,verbose=1)[:TTA*ct_test,] 
predictions = np.mean(pred.reshape((ct_test,TTA),order='F'),axis=1) 

test_ids = next(iter(img_ids_ds.batch(ct_test))).numpy().astype('U') 

pd.DataFrame({
     'image_name'  : test_ids, 
     'target'      : predictions
    }).to_csv('submission.csv', index=False)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
HTTPError                                 Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    310             try:
--> 311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:

/usr/lib/python3.11/urllib/request.py in urlretrieve(url, filename, reporthook, data)
    240 
--> 241     with contextlib.closing(urlopen(url, data)) as fp:
    242         headers = fp.info()

/usr/lib/python3.11/urllib/request.py in urlopen(url, data, timeout, cafile, capath, cadefault, context)
    215         opener = _opener
--> 216     return opener.open(url, data, timeout)
    217 

/usr/lib/python3.11/urllib/request.py in open(self, fullurl, data, timeout)
    524             meth = getattr(processor, meth_name)
--> 525             response = meth(req, response)
    526 

/usr/lib/python3.11/urllib/request.py in http_response(self, request, response)
    633         if not (200 <= code < 300):
--> 634             response = self.parent.error(
    635                 'http', request, response, code, msg, hdrs)

/usr/lib/python3.11/urllib/request.py in error(self, proto, *args)
    562             args = (dict, 'default', 'http_error_default') + orig_args
--> 563             return self._call_chain(*args)
    564 

/usr/lib/python3.11/urllib/request.py in _call_chain(self, chain, kind, meth_name, *args)
    495             func = getattr(handler, meth_name)
--> 496             result = func(*args)
    497             if result is not None:

/usr/lib/python3.11/urllib/request.py in http_error_default(self, req, fp, code, msg, hdrs)
    642     def http_error_default(self, req, fp, code, msg, hdrs):
--> 643         raise HTTPError(req.full_url, code, msg, hdrs, fp)
    644 

HTTPError: HTTP Error 404: Not Found

During handling of the above exception, another exception occurred:

Exception                                 Traceback (most recent call last)
/tmp/ipykernel_11/2014408195.py in <cell line: 0>()
      2 if strategy != None:
      3     with strategy.scope():
----> 4         model = efficientnetbx()
      5 else:
      6     model = efficientnetbx()

/tmp/ipykernel_11/825641283.py in efficientnetbx()
      1 def efficientnetbx():
      2     return tf.keras.Sequential([
----> 3                     efn.EfficientNetB0(
      4                         input_shape=(128,128, 3),
      5                         include_top=False

/usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py in wrapper(*args, **kwargs)
     55         kwargs['models'] = tfkeras.models
     56         kwargs['utils'] = tfkeras.utils
---> 57         return func(*args, **kwargs)
     58 
     59     return wrapper

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNetB0(include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    469         **kwargs
    470 ):
--> 471     return EfficientNet(
    472         1.0, 1.0, 224, 0.2,
    473         model_name='efficientnet-b0',

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNet(width_coefficient, depth_coefficient, default_resolution, dropout_rate, drop_connect_rate, depth_divisor, blocks_args, model_name, include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    430             file_name = model_name + '_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'
    431             file_hash = IMAGENET_WEIGHTS_HASHES[model_name][1]
--> 432         weights_path = keras_utils.get_file(
    433             file_name,
    434             IMAGENET_WEIGHTS_PATH + file_name,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py in get_file(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)
    311                 urlretrieve(origin, download_target, DLProgbar())
    312             except urllib.error.HTTPError as e:
--> 313                 raise Exception(error_msg.format(origin, e.code, e.msg))
    314             except urllib.error.URLError as e:
    315                 raise Exception(error_msg.format(origin, e.errno, e.reason))

Exception: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b0_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: 404 -- Not Found
