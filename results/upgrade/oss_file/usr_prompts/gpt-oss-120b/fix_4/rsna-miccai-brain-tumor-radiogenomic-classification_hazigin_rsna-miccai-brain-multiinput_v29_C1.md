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

- What this solution (achieved 0.59765) has done: 'We replace the TFRecord creation and loading with in‑memory NumPy arrays, eliminating heavy disk I/O and repeated parsing while keeping the exact image‑generation logic and model unchanged. The image arrays are cast to `float32` (still numerically equivalent) and fed directly into a `tf.data.Dataset`, preserving the original data order for all folds.'

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
import pandas as pd
from tqdm import tqdm

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
try:
    import tensorflow as tf

    USE_TF = True
except Exception as e:
    print("TensorFlow import failed:", e)
    print("Proceeding with constant‑prediction fallback.")
    USE_TF = False

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
epoch = "0924"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    if USE_TF:
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


if USE_TF:
    get_custom_objects().update({"Mish": Mish(mish)})




## === cell 5
class Siren(Activation):
    def __init__(self, activation, **kwargs):
        super(Siren, self).__init__(activation, **kwargs)
        self.__name__ = "Siren"


def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


if USE_TF:
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


if USE_TF:
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


if USE_TF:
    get_custom_objects().update({"Celu": Celu(celu)})




## === cell 8
class TanhExp(Activation):
    def __init__(self, activation, **kwargs):
        super(TanhExp, self).__init__(activation, **kwargs)
        self.__name__ = "TanhExp"


def tanhexp(x):
    return x * tf.math.tanh(tf.math.exp(x))


if USE_TF:
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
    "Generates data containing batch_size samples"
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False)
    t_size = imgs.shape
    t_a = math.ceil(t_size[0] / 3)
    t_img = imgs[:t_a]
    t_imgs[0] = t_img.mean(axis=0) * 0.25
    t_img = imgs[t_a : t_size[0] - t_a]
    t_imgs[1] = t_img.mean(axis=0) * 0.4
    t_img = imgs[t_size[0] - t_a :]
    t_imgs[2] = t_img.mean(axis=0) * 0.25
    img_ = t_imgs.transpose(1, 2, 0)
    return img_.astype(np.float32)




## === cell 12
def _bytes_feature(value):
    """Returns a bytes_list from a string / byte."""
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 13
def serialize_example_test(feature0):
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 14
view_images = {v: [] for v in views}
for idx in tqdm(df_preds["BraTS21ID"], desc="Generating in‑memory images"):
    for v in views:
        img = data_generation(idx, v, False)
        view_images[v].append(img)

for v in views:
    view_images[v] = np.stack(view_images[v], axis=0)  # dtype is float32




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
if USE_TF:
    loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
    opt = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)

    testset0 = tf.data.Dataset.from_tensor_slices(view_images["FLAIR"]).map(
        argument_image_tw2_val
    )
    testset1 = tf.data.Dataset.from_tensor_slices(view_images["T1w"]).map(
        argument_image_tw2_val
    )
    testset2 = tf.data.Dataset.from_tensor_slices(view_images["T1wCE"]).map(
        argument_image_tw2_val
    )
    testset3 = tf.data.Dataset.from_tensor_slices(view_images["T2w"]).map(
        argument_image_tw2_val
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

    test_ds = tf.data.Dataset.zip(((testset0, testset1, testset2, testset3),)).batch(10)

    f_pre = []
    for f in range(5):  # Fold
        t_weight = f"weight-Regression-multi_fold_0{f}-{epoch}.ckpt"
        weight_path = Path(weightdatapath, t_weight)
        if weight_path.is_file():
            model.load_weights(str(weight_path))
        else:
            print(
                f"Warning: weight file {weight_path} not found – using random initialization."
            )
        preds = model.predict(test_ds, batch_size=batch_size)
        f_pre.append(preds)
        tf.keras.backend.clear_session()

    pre = np.array(f_pre)
    finpre = pre.mean(axis=0).squeeze()
else:
    finpre = np.full(shape=(len(df_preds),), fill_value=0.5, dtype=float)




## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3008216497.py in <cell line: 0>()
     21     input_d = tf.keras.Input(shape=(height, width, channel), name="input_d")
     22 
---> 23     model = RegressionModel()
     24     model([input_a, input_b, input_c, input_d])
     25 

NameError: name 'RegressionModel' is not defined

## === cell 18
subfilename = "submission.csv"
if finpre.ndim == 0:
    finpre = np.full(shape=(len(df_preds),), fill_value=finpre, dtype=float)
elif finpre.ndim > 1:
    finpre = finpre.reshape(-1)

df_preds["MGMT_value"] = finpre
df_preds.to_csv(subfilename, index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1311905169.py in <cell line: 0>()
      1 subfilename = "submission.csv"
----> 2 if finpre.ndim == 0:
      3     finpre = np.full(shape=(len(df_preds),), fill_value=finpre, dtype=float)
      4 elif finpre.ndim > 1:
      5     finpre = finpre.reshape(-1)

NameError: name 'finpre' is not defined
