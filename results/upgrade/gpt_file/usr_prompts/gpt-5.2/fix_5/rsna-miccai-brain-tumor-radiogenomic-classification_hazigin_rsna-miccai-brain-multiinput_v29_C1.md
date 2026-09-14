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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the early TensorFlow/protobuf crash by removing the forced pure-Python protobuf setting and making the imports tolerant, so the notebook can start reliably in the Kaggle runtime. Then I fix the custom activation registration so Keras can resolve `"TanhExp"`, `"Mish"`, `"ARelu"`, `"Celu"`, and `"Siren"` when passed by string to layers (the current registration keys don’t match the strings used in the model). Finally, I make inference robust: verify weight files exist before loading, fall back to neutral 0.5 predictions if weights are missing or prediction fails, and always write a valid `submission.csv` with the required columns.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import random
import math
import gc
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

import cv2
import tensorflow as tf
from tensorflow.keras.utils import get_custom_objects
from tensorflow.keras import initializers

print("TF version:", tf.__version__)



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

print("Sample submission shape:", df_preds.shape)
print("Weights dir exists:", weightdatapath.exists())
print("Test dir exists:", Path(testdatapaht).exists())



## === cell 2
height = 256
width = 256
channel = 3
batch_size = 32
epochs = 400
seed = 26
epoch = "0924"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)




## === cell 4
@tf.keras.utils.register_keras_serializable(package="custom")
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="custom")
def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


@tf.keras.utils.register_keras_serializable(package="custom")
def arelu(x, alpha=0.90, beta=2.0):
    alpha = tf.clip_by_value(alpha, clip_value_min=0.01, clip_value_max=0.99)
    beta = 1 + tf.math.sigmoid(beta)
    return tf.nn.relu(x) * beta - tf.nn.relu(-x) * alpha


@tf.keras.utils.register_keras_serializable(package="custom")
def celu(x, alpha=2.0):
    mask_greater = tf.cast(tf.greater_equal(x, 0), tf.float32) * x
    mask_smaller = tf.cast(tf.less(x, 0), tf.float32) * x
    middle = alpha * (tf.exp(tf.divide(mask_smaller, alpha)) - 1)
    return middle + mask_greater


@tf.keras.utils.register_keras_serializable(package="custom")
def tanhexp(x):
    return x * tf.math.tanh(tf.math.exp(x))


get_custom_objects().update(
    {
        "Mish": mish,
        "Siren": siren,
        "ARelu": arelu,
        "Celu": celu,
        "TanhExp": tanhexp,
        "mish": mish,
        "siren": siren,
        "arelu": arelu,
        "celu": celu,
        "tanhexp": tanhexp,
    }
)

_ = tf.keras.activations.get("TanhExp")




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/2120886097.py in <cell line: 0>()
     50 
     51 # Sanity: ensure activation can be resolved
---> 52 _ = tf.keras.activations.get("TanhExp")
     53 
     54 

/usr/local/lib/python3.11/dist-packages/keras/src/activations/__init__.py in get(identifier)
    124     if callable(obj):
    125         return obj
--> 126     raise ValueError(
    127         f"Could not interpret activation function identifier: {identifier}"
    128     )

ValueError: Could not interpret activation function identifier: TanhExp

## === cell 5
class Silu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Silu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            "kernel", shape=[int(input_shape[-1]), self.num_outputs]
        )

    def call(self, input):
        return tf.nn.silu(input)




## === cell 6
def load_imgs(idx, view, ignore_zeros=True):
    import pydicom

    view_dir = os.path.join(testdatapaht, idx, view)
    try:
        files = os.listdir(view_dir)
    except FileNotFoundError:
        files = []

    if not files:
        return np.zeros((1, height, width), dtype=np.float32)

    files.sort()
    n = len(files)
    imgs = np.empty((n, height, width), dtype=np.float32)

    for i, name in enumerate(files):
        image_path = os.path.join(view_dir, name)
        try:
            pyds = pydicom.dcmread(image_path, force=True)
            slope = float(getattr(pyds, "RescaleSlope", 1.0))
            intercept = float(getattr(pyds, "RescaleIntercept", 0.0))
            arr = pyds.pixel_array
            img = intercept + arr.astype(np.float32) * slope
            imgs[i] = cv2.resize(img, (width, height)).astype(np.float32)
        except Exception:
            imgs[i] = np.zeros((height, width), dtype=np.float32)

    return imgs




## === cell 7
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False).astype(np.float32)

    n = imgs.shape[0]
    t_a = int(math.ceil(n / 3))

    t_imgs[0] = imgs[:t_a].mean(axis=0) * 0.25

    mid = imgs[t_a : n - t_a]
    if mid.shape[0] == 0:
        mid = imgs
    t_imgs[1] = mid.mean(axis=0) * 0.4

    t_imgs[2] = imgs[n - t_a :].mean(axis=0) * 0.25

    return t_imgs.transpose(1, 2, 0)




## === cell 8
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 9
def serialize_example_test(feature0):
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 10
ids = df_preds["BraTS21ID"].tolist()
for i in range(4):
    out_path = "./" + "brain_test_" + views[i] + ".tfrec"
    with tf.io.TFRecordWriter(out_path) as writer:
        for x in tqdm(ids, desc=f"Writing TFRecord {views[i]}", leave=False):
            img = data_generation(x, views[i], False).astype(np.float32)
            writer.write(serialize_example_test(img))




## === cell 11
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )
    image = tf.io.decode_raw(parsed_record["image"], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 12
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 13
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
        x1 = self.bn1(x1, training=training)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dence256_1(x1)
        x1 = self.dropoup5_1(x1, training=training)
        x1 = self.dence256_1_1(x1)
        x1 = self.dence128_1(x1)
        x1 = self.dence64_1(x1)
        x1 = self.dropoup3_1(x1, training=training)

        x2 = input_tensor[1]
        x2 = self.conv2(x2)
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup5_2(x2, training=training)
        x2 = self.dence256_2_1(x2)
        x2 = self.dropoup3_2(x2, training=training)
        x2 = self.dence128_2(x2)
        x2 = self.dence64_2(x2)
        x2 = self.dropoup3_2(x2, training=training)

        x3 = input_tensor[2]
        x3 = self.conv3(x3)
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup5_3(x3, training=training)
        x3 = self.dence256_3_1(x3)
        x3 = self.dropoup3_3(x3, training=training)
        x3 = self.dence128_3(x3)
        x3 = self.dence64_3(x3)
        x3 = self.dropoup3_3(x3, training=training)

        x4 = input_tensor[3]
        x4 = self.conv4(x4)
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup5_4(x4, training=training)
        x4 = self.dence256_4_1(x4)
        x4 = self.dropoup3_4(x4, training=training)
        x4 = self.dence128_4(x4)
        x4 = self.dence64_4(x4)
        x4 = self.dropoup3_4(x4, training=training)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dence256(x)
        x = self.conv0(x)
        x = self.bn0(x, training=training)
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




## === cell 14
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
f_pre = []

AUTO = tf.data.AUTOTUNE

testset0 = (
    tf.data.TFRecordDataset("./" + "brain_test_FLAIR.tfrec", num_parallel_reads=AUTO)
    .map(deserialize_example, num_parallel_calls=AUTO)
    .map(argument_image_tw2_val, num_parallel_calls=AUTO)
)
testset1 = (
    tf.data.TFRecordDataset("./" + "brain_test_T1w.tfrec", num_parallel_reads=AUTO)
    .map(deserialize_example, num_parallel_calls=AUTO)
    .map(argument_image_tw2_val, num_parallel_calls=AUTO)
)
testset2 = (
    tf.data.TFRecordDataset("./" + "brain_test_T1wCE.tfrec", num_parallel_reads=AUTO)
    .map(deserialize_example, num_parallel_calls=AUTO)
    .map(argument_image_tw2_val, num_parallel_calls=AUTO)
)
testset3 = (
    tf.data.TFRecordDataset("./" + "brain_test_T2w.tfrec", num_parallel_reads=AUTO)
    .map(deserialize_example, num_parallel_calls=AUTO)
    .map(argument_image_tw2_val, num_parallel_calls=AUTO)
)

test_ds = (
    tf.data.Dataset.zip((testset0, testset1, testset2, testset3))
    .batch(10, drop_remainder=False)
    .prefetch(AUTO)
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
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="bce"),
    ],
)

available = 0
for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    wpath = Path(weightdatapath, t_weight)
    if not wpath.exists():
        print("Missing weight:", str(wpath))
        continue
    try:
        model.load_weights(wpath)
        preds = model.predict(test_ds, batch_size=batch_size, verbose=0)
        f_pre.append(preds.astype(np.float32))
        available += 1
        print("Loaded and predicted with:", str(wpath), "preds shape:", preds.shape)
    except Exception as e:
        print("Failed weight/predict for", str(wpath), "error:", repr(e))
        continue

print("Successful folds:", available)



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/3542440959.py in <cell line: 0>()
     37 input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")
     38 
---> 39 model = RegressionModel()
     40 # Build model weights by calling once
     41 model([input_a, input_b, input_c, input_d])

/tmp/ipykernel_11/2886786964.py in __init__(self, **kwargs)
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

## === cell 15
N = len(df_preds)
if len(f_pre) > 0:
    pre = np.stack([p.reshape(-1) for p in f_pre], axis=0)  # (folds, N)
    finpre = pre.mean(axis=0).reshape(-1)  # (N,)
else:
    finpre = np.full(N, 0.5, dtype=np.float32)

if finpre.shape[0] != N:
    finpre = np.full(N, 0.5, dtype=np.float32)

finpre = np.clip(finpre.astype(np.float32), 0.0, 1.0)

subfilename = "submission.csv"
df_preds["MGMT_value"] = finpre
df_preds.to_csv(subfilename, index=False)

print("Wrote:", subfilename)
print(df_preds.head())
print("Submission shape:", df_preds.shape)
print(
    "MGMT_value min/mean/max:",
    float(df_preds["MGMT_value"].min()),
    float(df_preds["MGMT_value"].mean()),
    float(df_preds["MGMT_value"].max()),
)
