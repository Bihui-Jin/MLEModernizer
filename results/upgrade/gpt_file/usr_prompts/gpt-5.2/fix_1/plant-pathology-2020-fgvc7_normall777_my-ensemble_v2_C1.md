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

3.9

# 3. Installed packages

No external packages required in the script and installed.

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

0.9695857290459388

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
import math, re, os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from kaggle_datasets import KaggleDatasets
import tensorflow as tf
import tensorflow.keras.layers as L
from sklearn import metrics
from sklearn.model_selection import train_test_split

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_resnet_v2 import InceptionResNetV2
from tensorflow.keras.applications.densenet import DenseNet121, DenseNet169, DenseNet201 
from tensorflow.keras.applications.xception import Xception
from tensorflow.keras.applications.resnet50 import ResNet50
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.nasnet import NASNetLarge
from efficientnet.tfkeras import EfficientNetB7, EfficientNetL2

from tensorflow.keras.layers import Flatten,Dense,Dropout,BatchNormalization
from tensorflow.keras.models import Model,Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from keras.optimizers import Adam,SGD,Adagrad,Adadelta,RMSprop
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint

import tensorflow as tf
from tensorflow.keras.applications.densenet import DenseNet201
from tensorflow.keras.layers import Dropout, BatchNormalization, GaussianDropout
from efficientnet.tfkeras import EfficientNetB5
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense
from tensorflow.keras.applications.xception import Xception
from kaggle_datasets import KaggleDatasets
import re
import numpy as np
import random
import matplotlib.pyplot as plt
%matplotlib inline 
print("Tensorflow version " + tf.__version__)

## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
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
    strategy = tf.distribute.get_strategy() # если TPU отсутствует, то испльзуем стратегию по умолчанию для TF (CPU or GPU)

print("REPLICAS: ", strategy.num_replicas_in_sync)

GCS_DS_PATH = KaggleDatasets().get_gcs_path("plant-pathology-2020-fgvc7")

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync

## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
BackendError                              Traceback (most recent call last)
/tmp/ipykernel_11/380682719.py in <cell line: 0>()
     18 
     19 # Путь к данным. Если работаете на Google Colaboratory, то замените KaggleDatasets().get_gcs_path() на путь к данным, который будет у вас
---> 20 GCS_DS_PATH = KaggleDatasets().get_gcs_path("plant-pathology-2020-fgvc7")
     21 
     22 # Конфигурация

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
def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'

## === cell 6
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train.loc[:, 'healthy':].values

train_paths, valid_paths, train_labels, valid_labels = train_test_split(
     train_paths, train_labels, test_size=0.15, random_state=2020)

## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4294180073.py in <cell line: 0>()
      3 sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')
      4 
----> 5 train_paths = train.image_id.apply(format_path).values
      6 test_paths = test.image_id.apply(format_path).values
      7 

/usr/local/lib/python3.11/dist-packages/pandas/core/series.py in apply(self, func, convert_dtype, args, by_row, **kwargs)
   4922             args=args,
   4923             kwargs=kwargs,
-> 4924         ).apply()
   4925 
   4926     def _reindex_indexer(

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply(self)
   1425 
   1426         # self.func is Callable
-> 1427         return self.apply_standard()
   1428 
   1429     def agg(self):

/usr/local/lib/python3.11/dist-packages/pandas/core/apply.py in apply_standard(self)
   1505         #  Categorical (GH51645).
   1506         action = "ignore" if isinstance(obj.dtype, CategoricalDtype) else None
-> 1507         mapped = obj._map_values(
   1508             mapper=curried, na_action=action, convert=self.convert_dtype
   1509         )

/usr/local/lib/python3.11/dist-packages/pandas/core/base.py in _map_values(self, mapper, na_action, convert)
    919             return arr.map(mapper, na_action=na_action)
    920 
--> 921         return algorithms.map_array(arr, mapper, na_action=na_action, convert=convert)
    922 
    923     @final

/usr/local/lib/python3.11/dist-packages/pandas/core/algorithms.py in map_array(arr, mapper, na_action, convert)
   1741     values = arr.astype(object, copy=False)
   1742     if na_action is None:
-> 1743         return lib.map_infer(values, mapper, convert=convert)
   1744     else:
   1745         return lib.map_infer_mask(

lib.pyx in pandas._libs.lib.map_infer()

/tmp/ipykernel_11/2653035707.py in format_path(st)
      1 # функция, которая превращает айди картинки в полный путь к ней
      2 def format_path(st):
----> 3     return GCS_DS_PATH + '/images/' + st + '.jpg'

NameError: name 'GCS_DS_PATH' is not defined

## === cell 8
img_size = 768

def decode_image(filename, label=None, image_size=(img_size, img_size)):
    bits = tf.io.read_file(filename)
    image = tf.image.decode_jpeg(bits, channels=3)
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.resize(image, image_size)
    
    if label is None:
        return image
    else:
        return image, label
def data_augment(image, label=None, seed=2020):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    
    if label is None:
        return image
    else:
        return image, label

## === cell 9
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
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
    .batch(BATCH_SIZE)
)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3866109037.py in <cell line: 0>()
      1 train_dataset = (
      2     tf.data.Dataset
----> 3     .from_tensor_slices((train_paths, train_labels))
      4     .map(decode_image, num_parallel_calls=AUTO)
      5     .map(data_augment, num_parallel_calls=AUTO)

NameError: name 'train_paths' is not defined

## === cell 11
def get_model(use_model):
    base_model =  use_model(weights='imagenet',
                                 include_top=False, pooling='avg',
                                 input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions) 
    model.compile(optimizer='nadam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model

with strategy.scope():    
    model1 = get_model(EfficientNetB7) # тут подставить свою модель
model1.load_weights("/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5")


## --- ERROR in cell 11, traceback:
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
/tmp/ipykernel_11/2117685614.py in <cell line: 0>()
     10 
     11 with strategy.scope():
---> 12     model1 = get_model(EfficientNetB7) # тут подставить свою модель
     13 model1.load_weights("/kaggle/input/tf-zoo-models-on-tpu-efficientnetb7/my_ef_net_b7.h5")

/tmp/ipykernel_11/2117685614.py in get_model(use_model)
      1 def get_model(use_model):
----> 2     base_model =  use_model(weights='imagenet',
      3                                  include_top=False, pooling='avg',
      4                                  input_shape=(img_size, img_size, 3))
      5     x = base_model.output

/usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py in wrapper(*args, **kwargs)
     55         kwargs['models'] = tfkeras.models
     56         kwargs['utils'] = tfkeras.utils
---> 57         return func(*args, **kwargs)
     58 
     59     return wrapper

/usr/local/lib/python3.11/dist-packages/efficientnet/model.py in EfficientNetB7(include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)
    598         **kwargs
    599 ):
--> 600     return EfficientNet(
    601         2.0, 3.1, 600, 0.5,
    602         model_name='efficientnet-b7',

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

Exception: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: 404 -- Not Found

## === cell 12
def get_model(use_model):
    base_model =  use_model(weights='imagenet',
                                 include_top=False, pooling='avg',
                                 input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(train_labels.shape[1], activation="softmax")(x)
    model = Model(inputs=base_model.input, outputs=predictions) 
    model.compile(optimizer='nadam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
    return model

with strategy.scope():    
    model2 = get_model(DenseNet201) # тут подставить свою модель
model2.load_weights("/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5")


## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3948948195.py in <cell line: 0>()
     10 
     11 with strategy.scope():
---> 12     model2 = get_model(DenseNet201) # тут подставить свою модель
     13 model2.load_weights("/kaggle/input/tf-zoo-models-on-tpu-densenet201/my_dense_net_201.h5")

/tmp/ipykernel_11/3948948195.py in get_model(use_model)
      4                                  input_shape=(img_size, img_size, 3))
      5     x = base_model.output
----> 6     predictions = Dense(train_labels.shape[1], activation="softmax")(x)
      7     model = Model(inputs=base_model.input, outputs=predictions)
      8     model.compile(optimizer='nadam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])

NameError: name 'train_labels' is not defined

## === cell 14
best_alpha = 0.52
print('Вычисляем предсказания...')
probabilities1 = model1.predict(test_dataset, verbose=1)
probabilities2 = model2.predict(test_dataset, verbose=1)

probabilities = best_alpha * probabilities1 + (1 - best_alpha) * probabilities2 

print(probabilities)

sub.loc[:, 'healthy':] = probabilities
sub.to_csv('submission.csv', index=False)
sub.head()

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1099329161.py in <cell line: 0>()
      1 best_alpha = 0.52
      2 print('Вычисляем предсказания...')
----> 3 probabilities1 = model1.predict(test_dataset, verbose=1)
      4 probabilities2 = model2.predict(test_dataset, verbose=1)
      5 

NameError: name 'model1' is not defined
