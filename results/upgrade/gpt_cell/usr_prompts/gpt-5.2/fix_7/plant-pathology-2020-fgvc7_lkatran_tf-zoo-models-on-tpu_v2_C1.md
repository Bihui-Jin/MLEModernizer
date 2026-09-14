# Goal

You will receive environment details and a partial notebook export.

# Requirements

- Fix the bug that causes the error in cell k.
- Do NOT adjust any other non-buggy cells.
- You may reference cell k+1 only to preserve variable/interface compatibility.
- Do not complete or extend code logic in cell k, k+1, or later cells.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (bug fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Output must follow your strict format: Diagnosis / Patch summary / Updated cells / Compatibility notes for cell k+1 / Assumptions.


# 1. Python version

3.8

# 2. Installed packages

No external packages required in the script and installed.

# 3. Data file paths

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

# 4. Code solution

## === cell 1
!pip install -q efficientnet


## === cell 2
import sys, subprocess

subprocess.check_call(
    [sys.executable, "-m", "pip", "install", "-q", "protobuf==3.20.*"]
)

import math, re, os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

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

from tensorflow.keras.layers import Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization

from tensorflow.keras.optimizers import Adam, SGD, Adagrad, Adadelta, RMSprop

from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


## === cell 3
AUTO = tf.data.experimental.AUTOTUNE
try:
    tpu = tf.distribute.cluster_resolver.TPUClusterResolver()
    print("Running on TPU ", tpu.master())
except ValueError:
    tpu = None

if tpu:
    tf.config.experimental_connect_to_cluster(tpu)
    tf.tpu.experimental.initialize_tpu_system(tpu)
    strategy = tf.distribute.experimental.TPUStrategy(tpu)
else:
    strategy = (
        tf.distribute.get_strategy()
    )  # если TPU отсутствует, то испльзуем стратегию по умолчанию для TF (CPU or GPU)

print("REPLICAS: ", strategy.num_replicas_in_sync)

try:
    GCS_DS_PATH = KaggleDatasets().get_gcs_path()
except Exception:
    GCS_DS_PATH = "/kaggle/input/plant-pathology-2020-fgvc7"

EPOCHS = 40
BATCH_SIZE = 8 * strategy.num_replicas_in_sync


## === cell 4
def format_path(st):
    return GCS_DS_PATH + '/images/' + st + '.jpg'


## === cell 5
train = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/train.csv')
test = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/test.csv')
sub = pd.read_csv('/kaggle/input/plant-pathology-2020-fgvc7/sample_submission.csv')

train_paths = train.image_id.apply(format_path).values
test_paths = test.image_id.apply(format_path).values

train_labels = train.loc[:, 'healthy':].values



## === cell 6
from matplotlib import pyplot as plt

f, ax = plt.subplots(3, 6, figsize=(18, 7))
ax = ax.flatten()
for i in range(18):
    img = plt.imread(f'../input/plant-pathology-2020-fgvc7/images/Train_{i}.jpg')
    ax[i].set_title(train[train['image_id']==f'Train_{i}'].melt()[train[train['image_id']==f'Train_{i}'].melt().value == 1]['variable'].values[0])
    ax[i].imshow(img)
print(img.shape)


## === cell 7
nb_classes = 4
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
def data_augment(image, label=None, seed=42):
    image = tf.image.random_flip_left_right(image, seed=seed)
    image = tf.image.random_flip_up_down(image, seed=seed)
    
    if label is None:
        return image
    else:
        return image, label


## === cell 8
train_dataset = (
    tf.data.Dataset
    .from_tensor_slices((train_paths, train_labels))
    .map(decode_image, num_parallel_calls=AUTO)
    .cache()
    .map(data_augment, num_parallel_calls=AUTO)
    .repeat()
    .shuffle(512)
    .batch(BATCH_SIZE)
    .prefetch(AUTO)
)

test_dataset = (
    tf.data.Dataset
    .from_tensor_slices(test_paths)
    .map(decode_image, num_parallel_calls=AUTO)
    .batch(BATCH_SIZE)
)


## === cell 9
LR_START = 0.00001
LR_MAX = 0.0001 * strategy.num_replicas_in_sync
LR_MIN = 0.00001
LR_RAMPUP_EPOCHS = 15
LR_SUSTAIN_EPOCHS = 3
LR_EXP_DECAY = .8

def lrfn(epoch):
    if epoch < LR_RAMPUP_EPOCHS:
        lr = (LR_MAX - LR_START) / LR_RAMPUP_EPOCHS * epoch + LR_START
    elif epoch < LR_RAMPUP_EPOCHS + LR_SUSTAIN_EPOCHS:
        lr = LR_MAX
    else:
        lr = (LR_MAX - LR_MIN) * LR_EXP_DECAY**(epoch - LR_RAMPUP_EPOCHS - LR_SUSTAIN_EPOCHS) + LR_MIN
    return lr
    
lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=True)

rng = [i for i in range(EPOCHS)]
y = [lrfn(x) for x in rng]
plt.plot(rng, y)
print("Learning rate schedule: {:.3g} to {:.3g} to {:.3g}".format(y[0], max(y), y[-1]))


## === cell 10

def get_model():
    base_model =  EfficientNetB7(weights='imagenet',
                                 include_top=False, pooling='avg',
                                 input_shape=(img_size, img_size, 3))
    x = base_model.output
    predictions = Dense(nb_classes, activation="softmax")(x)
    return Model(inputs=base_model.input, outputs=predictions)

with strategy.scope():
    model = get_model()
    
model.compile(optimizer='adam', loss='categorical_crossentropy',metrics=['categorical_accuracy'])
model.summary()


## --- ERROR in cell 10, traceback:
[0;31m---------------------------------------------------------------------------[0m
[0;31mHTTPError[0m                                 Traceback (most recent call last)
[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py[0m in [0;36mget_file[0;34m(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)[0m
[1;32m    310[0m             [0;32mtry[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 311[0;31m                 [0murlretrieve[0m[0;34m([0m[0morigin[0m[0;34m,[0m [0mdownload_target[0m[0;34m,[0m [0mDLProgbar[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    312[0m             [0;32mexcept[0m [0murllib[0m[0;34m.[0m[0merror[0m[0;34m.[0m[0mHTTPError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36murlretrieve[0;34m(url, filename, reporthook, data)[0m
[1;32m    240[0m [0;34m[0m[0m
[0;32m--> 241[0;31m     [0;32mwith[0m [0mcontextlib[0m[0;34m.[0m[0mclosing[0m[0;34m([0m[0murlopen[0m[0;34m([0m[0murl[0m[0;34m,[0m [0mdata[0m[0;34m)[0m[0;34m)[0m [0;32mas[0m [0mfp[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    242[0m         [0mheaders[0m [0;34m=[0m [0mfp[0m[0;34m.[0m[0minfo[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36murlopen[0;34m(url, data, timeout, cafile, capath, cadefault, context)[0m
[1;32m    215[0m         [0mopener[0m [0;34m=[0m [0m_opener[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 216[0;31m     [0;32mreturn[0m [0mopener[0m[0;34m.[0m[0mopen[0m[0;34m([0m[0murl[0m[0;34m,[0m [0mdata[0m[0;34m,[0m [0mtimeout[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    217[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mopen[0;34m(self, fullurl, data, timeout)[0m
[1;32m    524[0m             [0mmeth[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mprocessor[0m[0;34m,[0m [0mmeth_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 525[0;31m             [0mresponse[0m [0;34m=[0m [0mmeth[0m[0;34m([0m[0mreq[0m[0;34m,[0m [0mresponse[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    526[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mhttp_response[0;34m(self, request, response)[0m
[1;32m    633[0m         [0;32mif[0m [0;32mnot[0m [0;34m([0m[0;36m200[0m [0;34m<=[0m [0mcode[0m [0;34m<[0m [0;36m300[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 634[0;31m             response = self.parent.error(
[0m[1;32m    635[0m                 'http', request, response, code, msg, hdrs)

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36merror[0;34m(self, proto, *args)[0m
[1;32m    562[0m             [0margs[0m [0;34m=[0m [0;34m([0m[0mdict[0m[0;34m,[0m [0;34m'default'[0m[0;34m,[0m [0;34m'http_error_default'[0m[0;34m)[0m [0;34m+[0m [0morig_args[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 563[0;31m             [0;32mreturn[0m [0mself[0m[0;34m.[0m[0m_call_chain[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    564[0m [0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36m_call_chain[0;34m(self, chain, kind, meth_name, *args)[0m
[1;32m    495[0m             [0mfunc[0m [0;34m=[0m [0mgetattr[0m[0;34m([0m[0mhandler[0m[0;34m,[0m [0mmeth_name[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 496[0;31m             [0mresult[0m [0;34m=[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    497[0m             [0;32mif[0m [0mresult[0m [0;32mis[0m [0;32mnot[0m [0;32mNone[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/lib/python3.11/urllib/request.py[0m in [0;36mhttp_error_default[0;34m(self, req, fp, code, msg, hdrs)[0m
[1;32m    642[0m     [0;32mdef[0m [0mhttp_error_default[0m[0;34m([0m[0mself[0m[0;34m,[0m [0mreq[0m[0;34m,[0m [0mfp[0m[0;34m,[0m [0mcode[0m[0;34m,[0m [0mmsg[0m[0;34m,[0m [0mhdrs[0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 643[0;31m         [0;32mraise[0m [0mHTTPError[0m[0;34m([0m[0mreq[0m[0;34m.[0m[0mfull_url[0m[0;34m,[0m [0mcode[0m[0;34m,[0m [0mmsg[0m[0;34m,[0m [0mhdrs[0m[0;34m,[0m [0mfp[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    644[0m [0;34m[0m[0m

[0;31mHTTPError[0m: HTTP Error 404: Not Found

During handling of the above exception, another exception occurred:

[0;31mException[0m                                 Traceback (most recent call last)
[0;32m/tmp/ipykernel_11/774883187.py[0m in [0;36m<cell line: 0>[0;34m()[0m
[1;32m     12[0m [0;34m[0m[0m
[1;32m     13[0m [0;32mwith[0m [0mstrategy[0m[0;34m.[0m[0mscope[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 14[0;31m     [0mmodel[0m [0;34m=[0m [0mget_model[0m[0;34m([0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     15[0m [0;34m[0m[0m
[1;32m     16[0m [0mmodel[0m[0;34m.[0m[0mcompile[0m[0;34m([0m[0moptimizer[0m[0;34m=[0m[0;34m'adam'[0m[0;34m,[0m [0mloss[0m[0;34m=[0m[0;34m'categorical_crossentropy'[0m[0;34m,[0m[0mmetrics[0m[0;34m=[0m[0;34m[[0m[0;34m'categorical_accuracy'[0m[0;34m][0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;32m/tmp/ipykernel_11/774883187.py[0m in [0;36mget_model[0;34m()[0m
[1;32m      4[0m [0;34m[0m[0m
[1;32m      5[0m [0;32mdef[0m [0mget_model[0m[0;34m([0m[0;34m)[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m----> 6[0;31m     base_model =  EfficientNetB7(weights='imagenet',
[0m[1;32m      7[0m                                  [0minclude_top[0m[0;34m=[0m[0;32mFalse[0m[0;34m,[0m [0mpooling[0m[0;34m=[0m[0;34m'avg'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m      8[0m                                  input_shape=(img_size, img_size, 3))

[0;32m/usr/local/lib/python3.11/dist-packages/efficientnet/__init__.py[0m in [0;36mwrapper[0;34m(*args, **kwargs)[0m
[1;32m     55[0m         [0mkwargs[0m[0;34m[[0m[0;34m'models'[0m[0;34m][0m [0;34m=[0m [0mtfkeras[0m[0;34m.[0m[0mmodels[0m[0;34m[0m[0;34m[0m[0m
[1;32m     56[0m         [0mkwargs[0m[0;34m[[0m[0;34m'utils'[0m[0;34m][0m [0;34m=[0m [0mtfkeras[0m[0;34m.[0m[0mutils[0m[0;34m[0m[0;34m[0m[0m
[0;32m---> 57[0;31m         [0;32mreturn[0m [0mfunc[0m[0;34m([0m[0;34m*[0m[0margs[0m[0;34m,[0m [0;34m**[0m[0mkwargs[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m     58[0m [0;34m[0m[0m
[1;32m     59[0m     [0;32mreturn[0m [0mwrapper[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/efficientnet/model.py[0m in [0;36mEfficientNetB7[0;34m(include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)[0m
[1;32m    598[0m         [0;34m**[0m[0mkwargs[0m[0;34m[0m[0;34m[0m[0m
[1;32m    599[0m ):
[0;32m--> 600[0;31m     return EfficientNet(
[0m[1;32m    601[0m         [0;36m2.0[0m[0;34m,[0m [0;36m3.1[0m[0;34m,[0m [0;36m600[0m[0;34m,[0m [0;36m0.5[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    602[0m         [0mmodel_name[0m[0;34m=[0m[0;34m'efficientnet-b7'[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/efficientnet/model.py[0m in [0;36mEfficientNet[0;34m(width_coefficient, depth_coefficient, default_resolution, dropout_rate, drop_connect_rate, depth_divisor, blocks_args, model_name, include_top, weights, input_tensor, input_shape, pooling, classes, **kwargs)[0m
[1;32m    430[0m             [0mfile_name[0m [0;34m=[0m [0mmodel_name[0m [0;34m+[0m [0;34m'_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5'[0m[0;34m[0m[0;34m[0m[0m
[1;32m    431[0m             [0mfile_hash[0m [0;34m=[0m [0mIMAGENET_WEIGHTS_HASHES[0m[0;34m[[0m[0mmodel_name[0m[0;34m][0m[0;34m[[0m[0;36m1[0m[0;34m][0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 432[0;31m         weights_path = keras_utils.get_file(
[0m[1;32m    433[0m             [0mfile_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m
[1;32m    434[0m             [0mIMAGENET_WEIGHTS_PATH[0m [0;34m+[0m [0mfile_name[0m[0;34m,[0m[0;34m[0m[0;34m[0m[0m

[0;32m/usr/local/lib/python3.11/dist-packages/keras/src/utils/file_utils.py[0m in [0;36mget_file[0;34m(fname, origin, untar, md5_hash, file_hash, cache_subdir, hash_algorithm, extract, archive_format, cache_dir, force_download)[0m
[1;32m    311[0m                 [0murlretrieve[0m[0;34m([0m[0morigin[0m[0;34m,[0m [0mdownload_target[0m[0;34m,[0m [0mDLProgbar[0m[0;34m([0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[1;32m    312[0m             [0;32mexcept[0m [0murllib[0m[0;34m.[0m[0merror[0m[0;34m.[0m[0mHTTPError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[0;32m--> 313[0;31m                 [0;32mraise[0m [0mException[0m[0;34m([0m[0merror_msg[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0morigin[0m[0;34m,[0m [0me[0m[0;34m.[0m[0mcode[0m[0;34m,[0m [0me[0m[0;34m.[0m[0mmsg[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m
[0m[1;32m    314[0m             [0;32mexcept[0m [0murllib[0m[0;34m.[0m[0merror[0m[0;34m.[0m[0mURLError[0m [0;32mas[0m [0me[0m[0;34m:[0m[0;34m[0m[0;34m[0m[0m
[1;32m    315[0m                 [0;32mraise[0m [0mException[0m[0;34m([0m[0merror_msg[0m[0;34m.[0m[0mformat[0m[0;34m([0m[0morigin[0m[0;34m,[0m [0me[0m[0;34m.[0m[0merrno[0m[0;34m,[0m [0me[0m[0;34m.[0m[0mreason[0m[0;34m)[0m[0;34m)[0m[0;34m[0m[0;34m[0m[0m

[0;31mException[0m: URL fetch failure on https://github.com/Callidior/keras-applications/releases/download/efficientnet/efficientnet-b7_weights_tf_dim_ordering_tf_kernels_autoaugment_notop.h5: 404 -- Not Found

## === cell 11
%%time
history = model.fit(
                    train_dataset, 
                    steps_per_epoch=train_labels.shape[0] // BATCH_SIZE,
                    callbacks=[lr_callback],
                    epochs=EPOCHS
                    validation_data=valid_dataset
                  )
