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
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import cv2
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import math
from concurrent.futures import ProcessPoolExecutor

try:
    import pydicom  # type: ignore

    _HAS_PYDICOM = True
except Exception as e:
    print(
        "Warning: pydicom failed to import; falling back to OpenCV-based DICOM decode. Error:",
        repr(e),
    )
    pydicom = None
    _HAS_PYDICOM = False



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
epoch = "0919"
views = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 3
def set_seed(seed=200):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)




## === cell 4
@tf.keras.utils.register_keras_serializable(package="custom", name="Mish")
def mish(x):
    return x * tf.math.tanh(tf.math.softplus(x))


@tf.keras.utils.register_keras_serializable(package="custom", name="Siren")
def siren(x):
    return 1.0 / (1.0 + tf.math.exp(-x))


tf.keras.utils.get_custom_objects().update({"Mish": mish, "Siren": siren})




## === cell 5
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




## === cell 6
def _safe_dicom_pixel_array_pydicom(dcm_path):
    """Read a DICOM file and return float32 pixel array, or None on failure."""
    if not _HAS_PYDICOM:
        return None
    try:
        ds = pydicom.dcmread(dcm_path, stop_before_pixels=False, force=True)
        arr = ds.pixel_array.astype(np.float32)
        return arr
    except Exception:
        return None


def _safe_dicom_pixel_array_opencv(dcm_path):
    """
    Fallback DICOM decoder using OpenCV's DICOM support.
    Returns float32 pixel array, or None on failure.
    """
    try:
        with open(dcm_path, "rb") as f:
            data = f.read()
        arr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        if arr is None:
            return None
        if arr.ndim == 3:
            arr = arr[..., 0]
        if arr.ndim != 2:
            return None
        return arr.astype(np.float32)
    except Exception:
        return None


def _safe_dicom_pixel_array(dcm_path):
    arr = _safe_dicom_pixel_array_pydicom(dcm_path)
    if arr is None:
        arr = _safe_dicom_pixel_array_opencv(dcm_path)
    return arr


def _instance_number(path):
    if _HAS_PYDICOM:
        try:
            ds = pydicom.dcmread(path, stop_before_pixels=True, force=True)
            return int(getattr(ds, "InstanceNumber", 0))
        except Exception:
            return 0
    return 0


def load_imgs(idx, view, ignore_zeros=True):
    """
    Load a DICOM series as a (z, height, width) float32 numpy array.
    Uses robust pydicom/OpenCV fallback to avoid protobuf-related crashes.
    """
    series_dir = os.path.join(testdatapath, idx, view)
    if not os.path.isdir(series_dir):
        return np.zeros((1, height, width), dtype=np.float32)

    try:
        files = [
            os.path.join(series_dir, f)
            for f in os.listdir(series_dir)
            if f.lower().endswith(".dcm")
        ]
    except Exception:
        files = []

    if not files:
        return np.zeros((1, height, width), dtype=np.float32)

    try:
        if _HAS_PYDICOM:
            files = sorted(files, key=_instance_number)
        else:
            files = sorted(files)
    except Exception:
        files = sorted(files)

    slices = []
    for fp in files:
        arr = _safe_dicom_pixel_array(fp)
        if arr is None or arr.ndim != 2:
            continue
        img = cv2.resize(arr, (width, height), interpolation=cv2.INTER_LINEAR)
        slices.append(img.astype(np.float32))

    if len(slices) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    vol = np.stack(slices, axis=0)  # (z, y, x)

    if ignore_zeros:
        mask = np.array([np.any(s != 0) for s in vol], dtype=bool)
        if mask.any():
            vol = vol[mask]
        else:
            vol = vol[:1]

    return vol




## === cell 7
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False).astype(np.float32)
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
    return img_




## === cell 8
_ids = df_preds["BraTS21ID"].astype(str).str.zfill(5).tolist()
_ids_int = [int(x) for x in _ids]


def _compute_one_subject(subject_id_int):
    out = []
    for v in views:
        out.append(data_generation(subject_id_int, v, False).astype(np.float32))
    return out  # [FLAIR, T1w, T1wCE, T2w]


max_workers = max(1, min(os.cpu_count() or 1, 8))
all_views = [[] for _ in range(4)]
with ProcessPoolExecutor(max_workers=max_workers) as ex:
    for out in tqdm(
        ex.map(_compute_one_subject, _ids_int, chunksize=1),
        total=len(_ids_int),
        desc="Precompute test features",
    ):
        for i in range(4):
            all_views[i].append(out[i])

X0 = np.stack(all_views[0], axis=0)  # (N,256,256,3)
X1 = np.stack(all_views[1], axis=0)
X2 = np.stack(all_views[2], axis=0)
X3 = np.stack(all_views[3], axis=0)
del all_views
gc.collect()




## === cell 9
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 10
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)

        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = Silu(0)
        self.rule1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = Silu(0)
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = Silu(0)
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation=mish
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = Silu(0)
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.5)
        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation=mish
        )
        self.dence128_4 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_4 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )

        self.dence32 = tf.keras.layers.Dense(
            32, kernel_initializer="he_normal", activation=siren
        )
        self.dence64_5 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation=para_relu
        )
        self.dropoup4 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_2 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_3 = tf.keras.layers.Dropout(0.5)
        self.dropoup4_4 = tf.keras.layers.Dropout(0.5)
        self.dropoup3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup3_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_2 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_3 = tf.keras.layers.Dropout(0.2)
        self.dropoup2_4 = tf.keras.layers.Dropout(0.2)
        self.dropoup = tf.keras.layers.Dropout(0.1)
        self.dence1 = tf.keras.layers.Dense(1, activation="sigmoid")

        self.flatten = tf.keras.layers.Flatten()

    def call(self, input_tensor, training=True):
        x1 = input_tensor[0]
        x1 = self.conv1(x1)
        x1 = self.bn1(x1, training=training)
        x1 = self.rule1(x1)
        x1 = self.max1(x1)
        x1 = self.dence256(x1)
        x1 = self.dropoup4(x1, training=training)
        x1 = self.dence128(x1)
        x1 = self.dropoup3(x1, training=training)
        x1 = self.dence64(x1)

        x2 = self.conv2(input_tensor[1])
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup4_2(x2, training=training)
        x2 = self.dence128_2(x2)
        x2 = self.dropoup3_2(x2, training=training)
        x2 = self.dence64_2(x2)

        x3 = self.conv3(input_tensor[2])
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup4_3(x3, training=training)
        x3 = self.dence128_3(x3)
        x3 = self.dropoup3_3(x3, training=training)
        x3 = self.dence64_3(x3)

        x4 = self.conv4(input_tensor[3])
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup4_4(x4, training=training)
        x4 = self.dence128_4(x4)
        x4 = self.dropoup3_4(x4, training=training)
        x4 = self.dence64_4(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.flatten(x)
        x = self.dence64_5(x)
        x = self.dropoup(x, training=training)
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




## === cell 11
def load_tf_checkpoint_into_keras_model(model, ckpt_prefix_path):
    """
    Assign variables from a TensorFlow checkpoint (prefix path without .index/.data suffix)
    into a built Keras model by matching variable names (best-effort, strict on shape match).
    """
    reader = tf.compat.v1.train.NewCheckpointReader(str(ckpt_prefix_path))
    ckpt_var_to_shape = reader.get_variable_to_shape_map()

    assigned = 0
    skipped = 0

    for var in model.variables:
        name = var.name
        if name.endswith(":0"):
            name = name[:-2]

        candidates = [name]
        if "/" in name:
            candidates.append(name.split("/", 1)[1])

        found = None
        for c in candidates:
            if c in ckpt_var_to_shape:
                found = c
                break

        if found is None:
            skipped += 1
            continue

        tensor = reader.get_tensor(found)
        if tuple(tensor.shape) != tuple(var.shape):
            skipped += 1
            continue

        var.assign(tensor)
        assigned += 1

    if assigned == 0:
        raise ValueError(f"No variables assigned from checkpoint: {ckpt_prefix_path}")
    return assigned, skipped


loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True
)
f_pre = []

ds0 = tf.data.Dataset.from_tensor_slices(X0).map(
    argument_image_tw2_val, num_parallel_calls=tf.data.AUTOTUNE
)
ds1 = tf.data.Dataset.from_tensor_slices(X1).map(
    argument_image_tw2_val, num_parallel_calls=tf.data.AUTOTUNE
)
ds2 = tf.data.Dataset.from_tensor_slices(X2).map(
    argument_image_tw2_val, num_parallel_calls=tf.data.AUTOTUNE
)
ds3 = tf.data.Dataset.from_tensor_slices(X3).map(
    argument_image_tw2_val, num_parallel_calls=tf.data.AUTOTUNE
)

test_ds = (
    tf.data.Dataset.zip((ds0, ds1, ds2, ds3))
    .batch(10, drop_remainder=False)
    .cache()
    .prefetch(tf.data.AUTOTUNE)
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

for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    ckpt_path = Path(weightdatapath, t_weight)
    assigned, skipped = load_tf_checkpoint_into_keras_model(model, ckpt_path)
    f_pre.append(model.predict(test_ds, batch_size=batch_size, verbose=0))

pre = np.stack(f_pre, axis=0)  # (n_folds, n_test, 1)
finpre = pre.mean(axis=0).reshape(-1)  # (n_test,)

df_preds["BraTS21ID"] = df_preds["BraTS21ID"].astype(str).str.zfill(5)
df_preds["MGMT_value"] = finpre.astype(np.float32)

subfilename = "submission.csv"
df_preds.to_csv(subfilename, index=False)

print("Wrote", subfilename, "with shape", df_preds.shape)
print(df_preds.head())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in NewCheckpointReader(filepattern)
     91   try:
---> 92     return CheckpointReader(compat.as_bytes(filepattern))
     93   # TODO(b/143319754): Remove the RuntimeError casting logic once we resolve the

RuntimeError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ../input/weight-multi-20210919/weight-Regression-multi_fold_00-0919.ckpt

During handling of the above exception, another exception occurred:

NotFoundError                             Traceback (most recent call last)
/tmp/ipykernel_11/260742684.py in <cell line: 0>()
     90     t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
     91     ckpt_path = Path(weightdatapath, t_weight)
---> 92     assigned, skipped = load_tf_checkpoint_into_keras_model(model, ckpt_path)
     93     # print(f"Loaded {ckpt_path}: assigned={assigned}, skipped={skipped}")
     94     f_pre.append(model.predict(test_ds, batch_size=batch_size, verbose=0))

/tmp/ipykernel_11/260742684.py in load_tf_checkpoint_into_keras_model(model, ckpt_prefix_path)
      6     into a built Keras model by matching variable names (best-effort, strict on shape match).
      7     """
----> 8     reader = tf.compat.v1.train.NewCheckpointReader(str(ckpt_prefix_path))
      9     ckpt_var_to_shape = reader.get_variable_to_shape_map()
     10 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in NewCheckpointReader(filepattern)
     94   # issue with throwing python exceptions from C++.
     95   except RuntimeError as e:
---> 96     error_translator(e)

/usr/local/lib/python3.11/dist-packages/tensorflow/python/training/py_checkpoint_reader.py in error_translator(e)
     29       'Failed to find any '
     30       'matching files for') in error_message:
---> 31     raise errors_impl.NotFoundError(None, None, error_message)
     32   elif 'Sliced checkpoints are not supported' in error_message or (
     33       'Data type '

NotFoundError: Unsuccessful TensorSliceReader constructor: Failed to find any matching files for ../input/weight-multi-20210919/weight-Regression-multi_fold_00-0919.ckpt
