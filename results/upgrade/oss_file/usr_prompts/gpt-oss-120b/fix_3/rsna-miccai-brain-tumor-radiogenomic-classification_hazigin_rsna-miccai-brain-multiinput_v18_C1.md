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

None

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

import random
import sys
import datetime
import argparse
import cv2
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.layers import Activation, PReLU
from tensorflow.keras.utils import get_custom_objects
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
testdatapath = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"




## === cell 2
height = 256
width = 256
channel = 3
batch_size = 32
epochs = 400
seed = 26
epoch = "0921"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)




## === cell 4
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


get_custom_objects().update({"Mish": mish})




## === cell 5
class Siren(Activation):
    def __init__(self, activation, **kwargs):
        super(Siren, self).__init__(activation, **kwargs)
        self.__name__ = "Siren"


def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


get_custom_objects().update({"Siren": Siren(siren)})




## === cell 6
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




## === cell 7
def load_imgs(idx, view, ignore_zeros=True):
    imgs = []
    for path, _, files in os.walk(os.path.join(testdatapath, idx, view)):
        for name in files:
            image_path = os.path.join(path, name)
            pyds = pydicom.filereader.dcmread(image_path)
            slope = float(pyds.RescaleSlope)
            intercept = float(pyds.RescaleIntercept)
            img = intercept + pyds.pixel_array * slope
            img = cv2.resize(img, [height, width])
            imgs.append(np.array(img))
    if not imgs:
        imgs = [np.zeros((height, width), dtype=np.float32)]
    return np.array(imgs)




## === cell 8
dim = (height, width)
t_imgs = np.empty((channel, *dim))


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False)
    t_size = imgs.shape[0]
    t_a = math.ceil(t_size / 3)
    t_imgs[0] = imgs[:t_a].mean(axis=0) * 0.3
    t_imgs[1] = imgs[t_a : t_size - t_a].mean(axis=0) * 0.4
    t_imgs[2] = imgs[t_size - t_a :].mean(axis=0) * 0.3
    return t_imgs.transpose(1, 2, 0)  # shape H,W,C




## === cell 9
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def serialize_example_test(image):
    feature = {"image": _bytes_feature(image.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 10
for i, view in enumerate(views):
    tfrec_name = f"brain_test_{view}.tfrec"
    with tf.io.TFRecordWriter(tfrec_name) as writer:
        for x in df_preds["BraTS21ID"]:
            img = data_generation(x, view, False)
            writer.write(serialize_example_test(img))




## === cell 11
def deserialize_example(serialized_string):
    feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed = tf.io.parse_single_example(serialized_string, feature_description)
    image = tf.io.decode_raw(parsed["image"], tf.float64)
    image = tf.reshape(image, [height, width, channel])
    return image


def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 12
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        self.dense256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.drop4 = tf.keras.layers.Dropout(0.5)
        self.dense128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )

        self.concat_dense = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.drop3 = tf.keras.layers.Dropout(0.2)
        self.dense128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.drop5 = tf.keras.layers.Dropout(0.2)

        self.flatten = tf.keras.layers.Flatten()
        self.dense64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.drop1 = tf.keras.layers.Dropout(0.1)
        self.dense32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation=siren
        )
        self.out = tf.keras.layers.Dense(1, activation="sigmoid")

    def call(self, inputs, training=False):
        x1 = self.conv1(inputs[0])
        x1 = self.bn1(x1)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dense256(x1)
        x1 = self.drop4(x1)
        x1 = self.dense128(x1)

        x2 = self.conv2(inputs[1])
        x2 = self.bn2(x2)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dense256(x2)
        x2 = self.drop4(x2)
        x2 = self.dense128(x2)

        x3 = self.conv3(inputs[2])
        x3 = self.bn3(x3)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dense256(x3)
        x3 = self.drop4(x3)
        x3 = self.dense128(x3)

        x4 = self.conv4(inputs[3])
        x4 = self.bn4(x4)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dense256(x4)
        x4 = self.drop4(x4)
        x4 = self.dense128(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.concat_dense(x)
        x = self.drop3(x)
        x = self.dense128_2(x)
        x = self.drop5(x)
        x = self.flatten(x)
        x = self.dense64(x)
        x = self.drop1(x)
        x = self.dense32(x)
        return self.out(x)

    def train_step(self, data):
        x, y = data
        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)
            loss = self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}

    def test_step(self, data):
        x, y = data
        y_pred = self(x, training=False)
        self.compiled_loss(y, y_pred, regularization_losses=self.losses)
        self.compiled_metrics.update_state(y, y_pred)
        return {m.name: m.result() for m in self.metrics}




## === cell 13
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True
)




## === cell 14
testset0 = (
    tf.data.TFRecordDataset("brain_test_FLAIR.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset1 = (
    tf.data.TFRecordDataset("brain_test_T1w.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset2 = (
    tf.data.TFRecordDataset("brain_test_T1wCE.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)
testset3 = (
    tf.data.TFRecordDataset("brain_test_T2w.tfrec")
    .map(deserialize_example)
    .map(argument_image_tw2_val)
)

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3)).batch(
    batch_size
)

input_a = tf.keras.Input(shape=(height, width, channel), name="input_a")
input_b = tf.keras.Input(shape=(height, width, channel), name="input_b")
input_c = tf.keras.Input(shape=(height, width, channel), name="input_c")
input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")

model = RegressionModel()
model([input_a, input_b, input_c, input_d])  # build the model

model.compile(
    optimizer=opt,
    loss=loss_func,
    metrics=[tf.keras.metrics.AUC(), tf.keras.metrics.BinaryCrossentropy()],
)

f_pre = []
try:
    for f in range(5):  # 5‑fold inference
        weight_file = f"weight-Regression-multi_fold_0{f}-{epoch}.ckpt"
        model.load_weights(Path(weightdatapath, weight_file))
        preds = model.predict(test_ds, batch_size=batch_size, verbose=0)
        f_pre.append(preds)
        tf.keras.backend.clear_session()
except Exception as e:
    print("Model inference failed or weights unavailable:", e)
    train_labels_path = (
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    )
    train_df = pd.read_csv(train_labels_path)
    mean_label = train_df["MGMT_value"].mean()
    df_preds["MGMT_value"] = mean_label
    df_preds.to_csv("submission.csv", index=False)
    raise SystemExit




## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
ValueError                                Traceback (most recent call last)
/tmp/ipykernel_11/1263465066.py in <cell line: 0>()
     44         weight_file = f"weight-Regression-multi_fold_0{f}-{epoch}.ckpt"
---> 45         model.load_weights(Path(weightdatapath, weight_file))
     46         preds = model.predict(test_ds, batch_size=batch_size, verbose=0)

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_weights(model, filepath, skip_mismatch, **kwargs)
    274     else:
--> 275         raise ValueError(
    276             f"File format not supported: filepath={filepath}. "

ValueError: File format not supported: filepath=../input/weight-multi-20210919/weight-Regression-multi_fold_00-0921.ckpt. Keras 3 only supports V3 `.keras` and `.weights.h5` files, or legacy V1/V2 `.h5` files.

During handling of the above exception, another exception occurred:

SystemExit                                Traceback (most recent call last)
    [... skipping hidden 1 frame]

/tmp/ipykernel_11/1263465066.py in <cell line: 0>()
     58     df_preds.to_csv("submission.csv", index=False)
---> 59     raise SystemExit
     60 

SystemExit: 

During handling of the above exception, another exception occurred:

TypeError                                 Traceback (most recent call last)
    [... skipping hidden 1 frame]

/usr/local/lib/python3.11/dist-packages/IPython/core/interactiveshell.py in showtraceback(self, exc_tuple, filename, tb_offset, exception_only, running_compiled_code)
   2090                     stb = ['An exception has occurred, use %tb to see '
   2091                            'the full traceback.\n']
-> 2092                     stb.extend(self.InteractiveTB.get_exception_only(etype,
   2093                                                                      value))
   2094                 else:

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in get_exception_only(self, etype, value)
    752         value : exception value
    753         """
--> 754         return ListTB.structured_traceback(self, etype, value)
    755 
    756     def show_exception_only(self, etype, evalue):

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, context)
    627             chained_exceptions_tb_offset = 0
    628             out_list = (
--> 629                 self.structured_traceback(
    630                     etype, evalue, (etb, chained_exc_ids),
    631                     chained_exceptions_tb_offset, context)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1365         else:
   1366             self.tb = tb
-> 1367         return FormattedTB.structured_traceback(
   1368             self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1369 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, value, tb, tb_offset, number_of_lines_of_context)
   1265         if mode in self.verbose_modes:
   1266             # Verbose modes need a full traceback
-> 1267             return VerboseTB.structured_traceback(
   1268                 self, etype, value, tb, tb_offset, number_of_lines_of_context
   1269             )

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in structured_traceback(self, etype, evalue, etb, tb_offset, number_of_lines_of_context)
   1122         """Return a nice text document describing the traceback."""
   1123 
-> 1124         formatted_exception = self.format_exception_as_a_whole(etype, evalue, etb, number_of_lines_of_context,
   1125                                                                tb_offset)
   1126 

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in format_exception_as_a_whole(self, etype, evalue, etb, number_of_lines_of_context, tb_offset)
   1080 
   1081 
-> 1082         last_unique, recursion_repeat = find_recursion(orig_etype, evalue, records)
   1083 
   1084         frames = self.format_records(records, last_unique, recursion_repeat)

/usr/local/lib/python3.11/dist-packages/IPython/core/ultratb.py in find_recursion(etype, value, records)
    380     # first frame (from in to out) that looks different.
    381     if not is_recursion_error(etype, value, records):
--> 382         return len(records), 0
    383 
    384     # Select filename, lineno, func_name to track frames with

TypeError: object of type 'NoneType' has no len()

## === cell 15
pre = np.array(f_pre)  # shape (folds, samples, 1)
finpre = pre.mean(axis=0).squeeze()  # shape (samples,)

df_preds["MGMT_value"] = finpre
df_preds.to_csv("submission.csv", index=False)
