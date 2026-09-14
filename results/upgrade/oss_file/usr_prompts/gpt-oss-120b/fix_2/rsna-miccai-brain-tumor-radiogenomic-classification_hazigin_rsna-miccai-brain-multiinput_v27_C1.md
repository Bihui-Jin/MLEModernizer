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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import random
import sys
import os
import datetime
import argparse
import cv2
import numpy as np
import gc
from numpy.core.fromnumeric import _trace_dispatcher
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Activation, PReLU
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import initializers
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import math



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_preds = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
weightdatapath = Path("../input/weight-multi-20210919")
testdatapaht = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



## === cell 2
height = 256
width = 256
channel = 3
batch_size = 32
epochs = 400
seed = 26
epoch = "0923"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)




## === cell 4
class Mish(Activation):
    def __init__(self, activation, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({"Mish": Mish(mish)})




## === cell 5
class Siren(Activation):
    def __init__(self, activation, **kwargs):
        super(Siren, self).__init__(activation, **kwargs)
        self.__name__ = "Siren"


def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


get_custom_objects().update({"Siren": Siren(siren)})




## === cell 6
class ARelu(Activation):
    def __init__(self, activation, **kwargs):
        super(ARelu, self).__init__(activation, **kwargs)
        self.__name__ = "ARelu"


def arelu(x, alpha=0.90, beta=2.0):
    alpha = tf.clip_by_value(alpha, clip_value_min=0.01, clip_value_max=0.99)
    beta = 1 + tf.math.sigmoid(beta)
    return tf.nn.relu(x) * beta - tf.nn.relu(-x) * alpha


get_custom_objects().update({"ARelu": ARelu(arelu)})




## === cell 7
class Celu(Activation):
    def __init__(self, activation, **kwargs):
        super(Celu, self).__init__(activation, **kwargs)
        self.__name__ = "Celu"


def celu(x, alpha=2.0):
    mask_greater = tf.cast(tf.greater_equal(x, 0), tf.float32) * x
    mask_smaller = tf.cast(tf.less(x, 0), tf.float32) * x
    middle = alpha * (tf.exp(tf.divide(mask_smaller, alpha)) - 1)
    return middle + mask_greater


get_custom_objects().update({"Celu": Celu(celu)})




## === cell 8
class TanhExp(Activation):
    def __init__(self, activation, **kwargs):
        super(TanhExp, self).__init__(activation, **kwargs)
        self.__name__ = "TanhExp"


def tanhexp(x):
    return x * tf.math.tanh(tf.math.exp(x))


get_custom_objects().update({"TanhExp": TanhExp(tanhexp)})




## === cell 9
class Silu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Silu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_variable(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs]
        )

    def call(self, input):
        return tf.nn.silu(input)




## === cell 10
def load_imgs(idx, view, ignore_zeros=True):
    imgs = {}
    save_ds = []
    dir_path = os.walk(os.path.join(testdatapaht, idx, view))
    for path, subdirs, files in dir_path:
        for name in files:
            image_path = os.path.join(path, name)
            pyds = pydicom.filereader.dcmread(image_path)
            slope = float(pyds.RescaleSlope)
            intercept = float(pyds.RescaleIntercept)
            img = intercept + pyds.pixel_array * slope
            img = cv2.resize(img, [height, width])
            save_ds.append(np.array(img))
    if len(save_ds) == 0:
        save_ds = np.zeros((1, 256, 256))
    imgs = np.array(save_ds)
    return imgs




## === cell 11
dim = (height, width)
t_imgs = np.empty((channel, *dim))


def data_generation(ID, view, is_Train=True):
    "Generates data containing batch_size samples"  # X : (n_samples, *dim, n_channels)

    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False)
    t_size = imgs.shape
    t_a = math.ceil(t_size[0] / 3)
    t_img = imgs[:t_a]
    t_imgs[0] = t_img.mean(axis=0) * 0.3
    t_img = imgs[t_a : t_size[0] - t_a]
    t_imgs[1] = t_img.mean(axis=0) * 0.4
    t_img = imgs[t_size[0] - t_a :]
    t_imgs[2] = t_img.mean(axis=0) * 0.3
    img_ = t_imgs.transpose(1, 2, 0)
    return img_




## === cell 12
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()  # BytesList won't unpack a string from an EagerTensor.
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 13
def serialize_example_test(feature0):
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 14
for i in range(4):
    with tf.io.TFRecordWriter(
        str("./") + str("brain_test_" + views[i] + ".tfrec")
    ) as writer:
        for x in df_preds["BraTS21ID"]:
            img = data_generation(x, views[i], False)
            example = serialize_example_test(img)
            writer.write(example)




## === cell 15
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )
    image = tf.io.decode_raw(parsed_record["image"], tf.float64)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 16
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 17
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            128,
            5,
            strides=2,
            kernel_initializer=initializers.TruncatedNormal(),
            activation="TanhExp",
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.15)
        )
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128,
            5,
            strides=2,
            kernel_initializer=initializers.TruncatedNormal(),
            activation="TanhExp",
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.15)
        )
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128,
            5,
            strides=2,
            kernel_initializer=initializers.TruncatedNormal(),
            activation="TanhExp",
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.15)
        )
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128,
            5,
            strides=2,
            kernel_initializer=initializers.TruncatedNormal(),
            activation="TanhExp",
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.15)
        )
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        self.conv0 = tf.keras.layers.Conv2D(
            128,
            5,
            strides=2,
            kernel_initializer=initializers.TruncatedNormal(),
            activation="TanhExp",
        )
        self.bn0 = tf.keras.layers.BatchNormalization()
        self.rule0 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.15)
        )
        self.max0 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.4)
        para_relu2 = tf.keras.layers.LeakyReLU(alpha=0.8)
        self.dence256_1 = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="ARelu"
        )
        self.dence256_1_1 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="Mish"
        )
        self.dence128_1 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_1 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="Celu"
        )
        self.dropoup5_1 = tf.keras.layers.Dropout(0.3)
        self.dropoup4_1 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_1 = tf.keras.layers.Dropout(0.2)

        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="ARelu"
        )
        self.dence256_2_1 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="Mish"
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="Celu"
        )
        self.dropoup5_2 = tf.keras.layers.Dropout(0.3)
        self.dropoup4_2 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_2 = tf.keras.layers.Dropout(0.2)

        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="ARelu"
        )
        self.dence256_3_1 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="Mish"
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="Celu"
        )
        self.dropoup5_3 = tf.keras.layers.Dropout(0.3)
        self.dropoup4_3 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_3 = tf.keras.layers.Dropout(0.2)

        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="ARelu"
        )
        self.dence256_4_1 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="Mish"
        )
        self.dence128_4 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_4 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="Celu"
        )
        self.dropoup5_4 = tf.keras.layers.Dropout(0.3)
        self.dropoup4_4 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_4 = tf.keras.layers.Dropout(0.2)

        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="Mish"
        )
        self.dence256_X = tf.keras.layers.Dense(
            256, kernel_initializer=initializers.TruncatedNormal(), activation="gelu"
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="Mish"
        )
        self.dence128_X = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="selu"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="gelu"
        )
        self.dence32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation="Siren"
        )
        self.dropoup5 = tf.keras.layers.Dropout(0.5)
        self.dropoup4 = tf.keras.layers.Dropout(0.4)
        self.dropoup3 = tf.keras.layers.Dropout(0.3)
        self.dropoup2 = tf.keras.layers.Dropout(0.2)
        self.dropoup = tf.keras.layers.Dropout(0.1)
        self.dence1 = tf.keras.layers.Dense(1, activation="sigmoid")
        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor, training=True):
        x1 = input_tensor[0]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dence256_1(x1)
        x1 = self.dropoup5_1(x1)
        x1 = self.dence256_1_1(x1)
        x1 = self.dence128_1(x1)
        x1 = self.dence64_1(x1)
        x1 = self.dropoup3_1(x1)

        x2 = input_tensor[1]
        x2 = self.conv2(x2)
        x2 = self.bn2(x2)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup5_2(x2)
        x2 = self.dence256_2_1(x2)
        x2 = self.dence128_2(x2)
        x2 = self.dence64_2(x2)
        x2 = self.dropoup3_2(x2)

        x3 = input_tensor[2]
        x3 = self.conv3(x3)
        x3 = self.bn3(x3)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup5_3(x3)
        x3 = self.dence256_3_1(x3)
        x3 = self.dence128_3(x3)
        x3 = self.dence64_3(x3)
        x3 = self.dropoup3_3(x3)

        x4 = input_tensor[3]
        x4 = self.conv4(x4)
        x4 = self.bn4(x4)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup5_4(x4)
        x4 = self.dence256_4_1(x4)
        x4 = self.dence128_4(x4)
        x4 = self.dence64_4(x4)
        x4 = self.dropoup3_4(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dence256(x)
        x = self.conv0(x)
        x = self.bn0(x)
        x = self.rule0(x)
        x = self.max0(x)
        x = self.dence128_X(x)
        x = self.flatten(x)
        x = self.dence64(x)
        x = self.dence32(x)
        return self.dence1(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            predictions = self(x, training=True)
            loss = self.compiled_loss(y, predictions, regularization_losses=self.losses)
        gradients = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.trainable_variables))
        self.compiled_metrics.update_state(y, predictions)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}




## === cell 18
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=True)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
f_pre = []



## === cell 19
testset0 = (
    tf.data.TFRecordDataset("./brain_test_FLAIR.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset1 = (
    tf.data.TFRecordDataset("./brain_test_T1w.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset2 = (
    tf.data.TFRecordDataset("./brain_test_T1wCE.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset3 = (
    tf.data.TFRecordDataset("./brain_test_T2w.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)

input_a = tf.keras.Input(shape=(height, width, channel), name="input_a")
input_b = tf.keras.Input(shape=(height, width, channel), name="input_b")
input_c = tf.keras.Input(shape=(height, width, channel), name="input_c")
input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")

model = RegressionModel()
model([input_a, input_b, input_c, input_d])

model.compile(
    optimizer=opt,
    loss=loss_func,
    metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryCrossentropy()],
)

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3)).batch(
    batch_size
)

for f in range(5):  # Fold loop
    t_weight = f"weight-Regression-multi_fold_0{f}-{epoch}.ckpt"
    model.load_weights(Path(weightdatapath, t_weight))
    f_pre.append(model.predict(test_ds, batch_size=batch_size))
    tf.keras.backend.clear_session()



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3874486987.py in <cell line: 0>()
     25 input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")
     26 
---> 27 model = RegressionModel()
     28 model([input_a, input_b, input_c, input_d])
     29 

/tmp/ipykernel_11/2213483019.py in __init__(self, **kwargs)
      2     def __init__(self, **kwargs):
      3         super(RegressionModel, self).__init__(**kwargs)
----> 4         self.conv1 = tf.keras.layers.Conv2D(
      5             128,
      6             5,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/conv2d.py in __init__(self, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, **kwargs)
    107         **kwargs,
    108     ):
--> 109         super().__init__(
    110             rank=2,
    111             filters=filters,

/usr/local/lib/python3.11/dist-packages/keras/src/layers/convolutional/base_conv.py in __init__(self, rank, filters, kernel_size, strides, padding, data_format, dilation_rate, groups, activation, use_bias, kernel_initializer, bias_initializer, kernel_regularizer, bias_regularizer, activity_regularizer, kernel_constraint, bias_constraint, lora_rank, **kwargs)
    116         self.padding = standardize_padding(padding, allow_causal=rank == 1)
    117         self.data_format = standardize_data_format(data_format)
--> 118         self.activation = activations.get(activation)
    119         self.use_bias = use_bias
    120         self.kernel_initializer = initializers.get(kernel_initializer)

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in get(identifier)
    124     if callable(obj):
    125         return obj
--> 126     raise ValueError(
    127         f"Could not interpret activation function identifier: {identifier}"
    128     )

ValueError: Could not interpret activation function identifier: TanhExp

## === cell 20
pre = np.array(f_pre)  # shape (folds, samples, 1)
finpre = pre.mean(axis=0).reshape(-1)  # average over folds, flatten to (samples,)

subfilename = "submission.csv"
df_preds["MGMT_value"] = finpre

df_preds.to_csv(subfilename, index=False)

## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/111393095.py in <cell line: 0>()
      4 subfilename = "submission.csv"
      5 # Align predictions with the original IDs
----> 6 df_preds["MGMT_value"] = finpre
      7 
      8 df_preds.to_csv(subfilename, index=False)

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in __setitem__(self, key, value)
   4309         else:
   4310             # set column
-> 4311             self._set_item(key, value)
   4312 
   4313     def _setitem_slice(self, key: slice, value) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _set_item(self, key, value)
   4522         ensure homogeneity.
   4523         """
-> 4524         value, refs = self._sanitize_column(value)
   4525 
   4526         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/frame.py in _sanitize_column(self, value)
   5264 
   5265         if is_list_like(value):
-> 5266             com.require_length_match(value, self.index)
   5267         arr = sanitize_array(value, self.index, copy=True, allow_2d=True)
   5268         if (

/usr/local/lib/python3.11/dist-packages/pandas/core/common.py in require_length_match(data, index)
    571     """
    572     if len(data) != len(index):
--> 573         raise ValueError(
    574             "Length of values "
    575             f"({len(data)}) "

ValueError: Length of values (1) does not match length of index (59)
