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

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the two blockers preventing an end-to-end run: (1) the `AttributeError: MessageFactory...GetPrototype` caused by importing `pydicom` with an incompatible protobuf runtime, and (2) Keras 3 refusing to load legacy `.ckpt` weights. To keep the core inference logic intact, I avoid changing the model and dataset pipeline, and instead add a robust fallback path: if weights can’t be loaded, we produce a valid submission using the mean training label as a calibrated constant probability (score-neutral baseline and guaranteed valid CSV). I also fix the dataset loader bug (`int(id_tensor)` inside a graph) that would otherwise break prediction, and ensure `f_pre` has the correct shape so the submission length matches the sample submission.'
- What this solution (achieved 0.5) has done: 'I fix the hard import-time crash by avoiding `pydicom` entirely (the protobuf incompatibility triggers before any model code runs) and instead read DICOM pixels via OpenCV, which is already installed. I also correct a path typo (`testdatapaht`) and make sure the DICOM loader targets the correct root folder and returns valid float32 arrays consistently. The rest of the pipeline (dataset construction, model architecture, fold ensembling, and the constant-probability fallback) stays the same so scoring behavior is preserved, but it now run end-to-end and always write a valid `submission.csv`. Finally, I keep the fold-weight loading logic unchanged; if weights still can’t be loaded under Keras 3, the mean-label fallback still guarantees a valid submission.'
- What this solution (achieved 0.5) has done: 'I fix the two execution blockers that prevent an end-to-end run and thus a valid `submission.csv`: (1) the import-time crash is coming from mixing `tensorflow.compat.v1` (TF1 graph mode) with Keras 3 internals, so I switch to standard `tensorflow` (TF2 eager) without changing the model layers/architecture; and (2) the model build fails due to shape inference issues inside the subclassed model, so I build it with real tensors (or `build()` + dummy call) to stabilize shapes. I also fix the `testdatapaht` typo by standardizing to a single `testdatapath` variable used by the DICOM loader. Finally, I keep the existing constant-probability fallback (mean train label) so a valid submission is always produced even if `.ckpt` weights can’t be loaded in this environment.'
- What this solution (achieved 0.5) has done: 'I remove the remaining import-time protobuf/pydicom blocker by ensuring we never import pydicom transitively (the error is happening before your `pydicom=None` can help), keeping the rest of the pipeline intact. Then I fix a logic issue that currently makes the image loader always read from the test folder (even if later reused), by passing an explicit root path into `load_imgs`/`data_generation`—this is score-positive because it makes the fallback/training-label mean and any potential local validation consistent and prevents silent data-path bugs. Finally, I make the weight-loading more Keras-3 compatible without changing the model by using `expect_partial()` and trying both checkpoint prefix and `.index` forms, so if the provided weights are loadable in this environment we get better-than-constant predictions (moving score upward from 0.5 toward a more typical AUC for this model), while still keeping the constant-probability fallback if not.'
- What this solution (achieved 0.5) has done: 'I fix the import-time crash happening before any cells run by preventing TensorFlow/keras from importing the incompatible `protobuf` implementation (this is what triggers the `MessageFactory.GetPrototype` error). Then I keep your existing model/dataset/inference logic unchanged, only adding a safe environment setup and minor robustness guards so prediction always completes and the submission length/format is correct. This should restore end-to-end execution and, if the provided fold checkpoints can be loaded, move the score upward from the constant-mean fallback (0.5) toward the intended model performance; otherwise it still reliably produce a valid `submission.csv`.'
- What this solution (achieved 0.5) has done: 'I fix the import-time `protobuf`/`MessageFactory.GetPrototype` crash by forcing TensorFlow to use the pure-Python protobuf implementation *before* TensorFlow is imported, and by clearing any already-imported `google.protobuf` modules to ensure the setting takes effect. This is a runtime blocker fix and should be score-neutral (it just allows the intended inference to run). I also add a small, safe fallback if TensorFlow still fails to import (write a valid constant-probability submission based on mean train label), ensuring a `.csv` is always produced. The rest of your model, dataset pipeline, weight-loading, and prediction logic remains unchanged.'

# 9. Code solution

## === cell 0
import os

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import sys

for k in list(sys.modules.keys()):
    if k.startswith("google.protobuf"):
        sys.modules.pop(k, None)

import random
import math
import gc
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

TF_AVAILABLE = True
try:
    import tensorflow as tf
    from tensorflow.keras.layers import Activation  # noqa: E402
    from tensorflow.keras.utils import get_custom_objects  # noqa: E402
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
df_preds = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
weightdatapath = Path("../input/weight-multi-20210919")

train_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train"
test_root = "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test"



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
if not TF_AVAILABLE:
    train_labels_path = (
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    )
    df_train = pd.read_csv(train_labels_path)
    base_p = float(df_train["MGMT_value"].mean())
    base_p = min(max(base_p, 1e-6), 1.0 - 1e-6)
    df_preds["MGMT_value"] = base_p
    subfilename = "submission.csv"
    df_preds.to_csv(subfilename, index=False)
    print("WARNING: TensorFlow import failed; wrote constant-mean submission instead.")
    print("TF import error:", TF_IMPORT_ERROR)
    print("Wrote:", subfilename)
    print(df_preds.head())
    raise SystemExit(0)




## === cell 4
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)

AUTOTUNE = tf.data.AUTOTUNE




## === cell 5
@tf.keras.utils.register_keras_serializable(package="Custom")
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


class Mish(Activation):
    def __init__(self, activation=mish, **kwargs):
        super(Mish, self).__init__(activation, **kwargs)
        self.__name__ = "Mish"


get_custom_objects().update({"Mish": Mish(mish), "mish": mish})




## === cell 6
@tf.keras.utils.register_keras_serializable(package="Custom")
def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


class Siren(Activation):
    def __init__(self, activation=siren, **kwargs):
        super(Siren, self).__init__(activation, **kwargs)
        self.__name__ = "Siren"


get_custom_objects().update({"Siren": Siren(siren), "siren": siren})




## === cell 7
class Silu(tf.keras.layers.Layer):
    def __init__(self, num_outputs):
        super(Silu, self).__init__()
        self.num_outputs = num_outputs

    def build(self, input_shape):
        self.kernel = self.add_weight(
            name="kernel",
            shape=[int(input_shape[-1]), self.num_outputs],
            initializer="glorot_uniform",
            trainable=True,
        )

    def call(self, input):
        return tf.nn.silu(input)




## === cell 8
def _list_dcm_files_sorted(folder: str):
    try:
        files = os.listdir(folder)
    except FileNotFoundError:
        return []
    files = [f for f in files if f.lower().endswith(".dcm")]

    def _key(name):
        base = os.path.splitext(name)[0]
        n = ""
        for ch in reversed(base):
            if ch.isdigit():
                n = ch + n
            else:
                if n:
                    break
        return int(n) if n else base

    files.sort(key=_key)
    return [os.path.join(folder, f) for f in files]


def _read_dcm_opencv(path: str):
    """
    Read DICOM pixels via OpenCV. If not supported, return None.
    """
    try:
        img = cv2.imread(path, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None
        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception:
        return None


def load_imgs(root_path, idx, view, ignore_zeros=True):
    folder = os.path.join(root_path, idx, view)
    paths = _list_dcm_files_sorted(folder)
    if not paths:
        return np.zeros((1, height, width), dtype=np.float32)

    save_ds = []
    for image_path in paths:
        arr = _read_dcm_opencv(image_path)
        if arr is None:
            arr = np.zeros((height, width), dtype=np.uint16)

        img = arr.astype(np.float32)
        img = cv2.resize(img, (width, height), interpolation=cv2.INTER_LINEAR)
        save_ds.append(img.astype(np.float32))

    if len(save_ds) == 0:
        return np.zeros((1, height, width), dtype=np.float32)
    return np.stack(save_ds, axis=0).astype(np.float32)




## === cell 9
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(root_path, ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(root_path, idx, view, ignore_zeros=False)

    t_size = imgs.shape
    t_a = math.ceil(t_size[0] / 3)

    t_img = imgs[:t_a]
    t_imgs[0] = t_img.mean(axis=0) * 0.3

    t_img = imgs[t_a : t_size[0] - t_a]
    if t_img.shape[0] == 0:
        t_img = imgs[:t_a]
    t_imgs[1] = t_img.mean(axis=0) * 0.4

    t_img = imgs[t_size[0] - t_a :]
    t_imgs[2] = t_img.mean(axis=0) * 0.3

    img_ = t_imgs.transpose(1, 2, 0)
    return img_.astype(np.float32)




## === cell 10
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 11
def serialize_example_test(feature0):
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 12
pass




## === cell 13
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )

    image = tf.io.decode_raw(parsed_record["image"], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 14
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 15
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25)
        )
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25)
        )
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25)
        )
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = tf.keras.layers.PReLU(
            alpha_initializer=tf.initializers.constant(0.25)
        )
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.3)

        self.dence256_1 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_1 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation=mish
        )
        self.dence64_1 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="swish"
        )
        self.dropoup4_1 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_1 = tf.keras.layers.Dropout(0.2)

        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation=mish
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="swish"
        )
        self.dropoup4_2 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_2 = tf.keras.layers.Dropout(0.2)

        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation=mish
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="swish"
        )
        self.dropoup4_3 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_3 = tf.keras.layers.Dropout(0.2)

        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_4 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation=mish
        )
        self.dence64_4 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="swish"
        )
        self.dropoup4_4 = tf.keras.layers.Dropout(0.3)
        self.dropoup3_4 = tf.keras.layers.Dropout(0.2)

        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=siren
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation=mish
        )
        self.dence32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation=para_relu
        )
        self.dropoup5 = tf.keras.layers.Dropout(0.5)
        self.dropoup4 = tf.keras.layers.Dropout(0.4)
        self.dropoup3 = tf.keras.layers.Dropout(0.3)
        self.dropoup2 = tf.keras.layers.Dropout(0.2)
        self.dropoup = tf.keras.layers.Dropout(0.1)
        self.dence1 = tf.keras.layers.Dense(1, activation="sigmoid")

        self.flatten = tf.keras.layers.Flatten()
        self.concat = tf.keras.layers.Concatenate()

    def call(self, input_tensor, training=True):
        x1 = input_tensor[0]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1, training=training)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dropoup4_1(x1, training=training)
        x1 = self.dence128_1(x1)
        x1 = self.dence64_1(x1)

        x2 = input_tensor[1]
        x2 = self.conv2(x2)
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dropoup4_2(x2, training=training)
        x2 = self.dence128_2(x2)
        x2 = self.dence64_2(x2)

        x3 = input_tensor[2]
        x3 = self.conv3(x3)
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dropoup4_3(x3, training=training)
        x3 = self.dence128_3(x3)
        x3 = self.dence64_3(x3)

        x4 = input_tensor[3]
        x4 = self.conv4(x4)
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dropoup4_4(x4, training=training)
        x4 = self.dence128_4(x4)
        x4 = self.dence64_4(x4)

        x = self.concat([x1, x2, x3, x4])
        x = self.dence256(x)
        x = self.flatten(x)
        x = self.dence128(x)
        x = self.dropoup2(x, training=training)
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




## === cell 16
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.Adam(learning_rate=1e-5, beta_1=0.9, beta_2=0.999)
f_pre = []



## === cell 17
_ids = df_preds["BraTS21ID"].astype(int).tolist()
_img_cache = {}


def _py_get_image(id_int, view_str):
    key = (int(id_int), view_str)
    arr = _img_cache.get(key)
    if arr is None:
        arr = data_generation(test_root, int(id_int), view_str, False).astype(
            np.float32
        )
        _img_cache[key] = arr
    return arr


def make_view_dataset(view):
    def _loader(id_np):
        id_int = int(np.asarray(id_np).item())
        img = _py_get_image(id_int, view)
        return img.astype(np.float32)

    ds = tf.data.Dataset.from_tensor_slices(np.array(_ids, dtype=np.int32))
    ds = ds.map(
        lambda x: tf.numpy_function(_loader, [x], Tout=tf.float32),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(
        lambda img: tf.ensure_shape(img, (height, width, channel)),
        num_parallel_calls=AUTOTUNE,
        deterministic=True,
    )
    ds = ds.map(argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


testset0 = make_view_dataset("FLAIR")
testset1 = make_view_dataset("T1w")
testset2 = make_view_dataset("T1wCE")
testset3 = make_view_dataset("T2w")

model = RegressionModel()
dummy = [
    tf.zeros((1, height, width, channel), dtype=tf.float32),
    tf.zeros((1, height, width, channel), dtype=tf.float32),
    tf.zeros((1, height, width, channel), dtype=tf.float32),
    tf.zeros((1, height, width, channel), dtype=tf.float32),
]
_ = model(dummy, training=False)

model.compile(
    optimizer=opt,
    loss=loss_func,
    metrics=[
        tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.BinaryCrossentropy(name="bce"),
    ],
)

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3))
test_ds = test_ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)


def _try_load_weights(m, ckpt_path: Path):
    try:
        m.load_weights(str(ckpt_path)).expect_partial()
        return True, None
    except Exception as e1:
        try:
            if str(ckpt_path).endswith(".ckpt"):
                m.load_weights(str(ckpt_path)[:-5]).expect_partial()
                return True, None
        except Exception as e2:
            return False, (e1, e2)
        return False, (e1,)


can_use_weights = True
for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    wpath = Path(weightdatapath, t_weight)
    ok, err = _try_load_weights(model, wpath)
    if not ok:
        print("WARNING: failed to load weights for", str(wpath), "Errors:", repr(err))
        can_use_weights = False
        break
    preds = model.predict(test_ds, verbose=0)
    f_pre.append(preds)

if not can_use_weights or len(f_pre) == 0:
    train_labels_path = (
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    )
    df_train = pd.read_csv(train_labels_path)
    base_p = float(df_train["MGMT_value"].mean())
    base_p = min(max(base_p, 1e-6), 1.0 - 1e-6)
    finpre = np.full((len(df_preds),), base_p, dtype=np.float32)
else:
    pre = np.array(f_pre)  # (5, N, 1)
    finpre = pre.mean(axis=0)  # (N, 1)
    finpre = np.asarray(finpre).reshape(-1)  # (N,)



## === cell 18
if finpre.shape[0] != len(df_preds):
    print(
        "WARNING: prediction length mismatch:",
        finpre.shape[0],
        "!= expected",
        len(df_preds),
        "Falling back to constant mean probability.",
    )
    train_labels_path = (
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
    )
    df_train = pd.read_csv(train_labels_path)
    base_p = float(df_train["MGMT_value"].mean())
    base_p = min(max(base_p, 1e-6), 1.0 - 1e-6)
    finpre = np.full((len(df_preds),), base_p, dtype=np.float32)

subfilename = "submission.csv"
df_preds["MGMT_value"] = finpre.astype(float)
df_preds.to_csv(subfilename, index=False)

print("Wrote:", subfilename)
print(df_preds.head())
print("MGMT_value stats:", df_preds["MGMT_value"].describe())
