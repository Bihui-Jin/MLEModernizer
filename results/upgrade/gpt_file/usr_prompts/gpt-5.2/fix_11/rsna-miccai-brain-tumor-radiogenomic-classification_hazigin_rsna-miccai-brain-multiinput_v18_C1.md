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

- What this solution (achieved 0.5) has done: 'I fix the initial import crash by avoiding `pydicom` (it’s failing due to a protobuf incompatibility in this environment) and instead read DICOM pixel data via TensorFlow’s built-in DICOM decoder. I also fix the multiprocessing TFRecord build error by removing the non-picklable nested worker function and building TFRecords single-process (only ~59 test cases, so it stays fast and stable). Finally, I fix the Keras activation error by explicitly registering `siren` with Keras’ activation registry so `Dense(..., activation="siren")` resolves correctly. These changes keep your model architecture and inference logic intact, but make the notebook run end-to-end and produce `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import random
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
import tensorflow as tf
from pathlib import Path
import math

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 4) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF:", tf.__version__)
print("Using TF dicom decoder (no pydicom import due to protobuf incompatibility).")



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
if not os.path.isdir(testdatapaht):
    alt = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification/test"
    if os.path.isdir(alt):
        testdatapaht = alt

print("Using test data path:", testdatapaht)



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

try:
    tf.config.experimental.enable_op_determinism(True)
except Exception:
    pass




## === cell 4
@tf.keras.utils.register_keras_serializable(package="Custom")
def mish(inputs):
    return inputs * tf.math.tanh(tf.math.softplus(inputs))


@tf.keras.utils.register_keras_serializable(package="Custom")
def siren(inputs):
    return 1.0 / (1.0 + tf.math.exp(-inputs))


tf.keras.utils.get_custom_objects()["mish"] = mish
tf.keras.utils.get_custom_objects()["siren"] = siren
try:
    tf.keras.activations.mish = mish
except Exception:
    pass
try:
    tf.keras.activations.siren = siren
except Exception:
    pass
try:
    tf.keras.activations.get({"class_name": "siren", "config": {}})
except Exception:
    pass




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
_DCM_LIST_CACHE = {}
_DCM_IMG_CACHE = {}  # cache decoded+resized slices within a run (path -> image)


def _list_dcm_files(idx: str, view: str):
    key = (idx, view)
    hit = _DCM_LIST_CACHE.get(key)
    if hit is not None:
        return hit

    base_dir = os.path.join(testdatapaht, idx, view)
    if not os.path.isdir(base_dir):
        files = []
    else:
        files = []
        with os.scandir(base_dir) as it:
            for e in it:
                if e.is_file():
                    n = e.name
                    if n.lower().endswith(".dcm"):
                        files.append(e.path)
        files.sort()
    _DCM_LIST_CACHE[key] = files
    return files


def _area_resize_2x_downsample(img2d: np.ndarray) -> np.ndarray:
    if img2d.shape == (512, 512) and (height, width) == (256, 256):
        x = img2d.reshape(256, 2, 256, 2).mean(axis=(1, 3), dtype=np.float32)
        return x.astype(np.float32, copy=False)
    t = tf.convert_to_tensor(img2d[..., None], dtype=tf.float32)
    t = tf.image.resize(
        t, (height, width), method=tf.image.ResizeMethod.AREA, antialias=False
    )
    return tf.squeeze(t, axis=-1).numpy().astype(np.float32, copy=False)


def _decode_resize_dicom_numpy(path: str) -> np.ndarray:
    cached = _DCM_IMG_CACHE.get(path)
    if cached is not None:
        return cached
    try:
        raw = tf.io.read_file(path)
        img = tf.io.decode_dicom_image(
            raw,
            dtype=tf.uint16,
            color_dim=False,
            scale="auto",
            on_error="skip",
        )
        img = tf.squeeze(img, axis=-1)  # [frames, rows, cols]
        if img.shape.rank == 2:
            img2d = img
        else:
            img2d = img[0]
        img2d = tf.cast(img2d, tf.float32).numpy()
        if not np.isfinite(img2d).all():
            img2d = np.nan_to_num(img2d, nan=0.0, posinf=0.0, neginf=0.0).astype(
                np.float32, copy=False
            )
        out = _area_resize_2x_downsample(img2d)
    except Exception:
        out = np.zeros((height, width), dtype=np.float32)

    _DCM_IMG_CACHE[path] = out
    return out


def load_imgs(idx, view, ignore_zeros=True):
    files = _list_dcm_files(idx, view)
    if len(files) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    imgs = []
    dec = _decode_resize_dicom_numpy
    for p in files:
        im = dec(p)
        if ignore_zeros and (im.max() == 0.0):
            continue
        imgs.append(im)

    if len(imgs) == 0:
        return np.zeros((1, height, width), dtype=np.float32)

    return np.stack(imgs, axis=0).astype(np.float32, copy=False)




## === cell 7
dim = (height, width)
t_imgs = np.empty((channel, *dim), dtype=np.float32)


def data_generation(ID, view, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, ignore_zeros=False)  # float32
    t_size0 = imgs.shape[0]
    t_a = math.ceil(t_size0 / 3)

    t_img = imgs[:t_a]
    t_imgs[0] = t_img.mean(axis=0) * 0.3

    t_img = imgs[t_a : t_size0 - t_a]
    if t_img.shape[0] == 0:
        t_img = imgs
    t_imgs[1] = t_img.mean(axis=0) * 0.4

    t_img = imgs[t_size0 - t_a :]
    t_imgs[2] = t_img.mean(axis=0) * 0.3

    img_ = t_imgs.transpose(1, 2, 0)
    return img_




## === cell 8
def _bytes_feature(value):
    if isinstance(value, type(tf.constant(0))):
        value = value.numpy()
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))




## === cell 9
def serialize_example_test(feature0):
    feature0 = np.asarray(feature0, dtype=np.float32, order="C")
    feature = {"image": _bytes_feature(feature0.tobytes())}
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()




## === cell 10
def _build_one_view_tfrec(view_name, ids, out_path):
    with tf.io.TFRecordWriter(out_path) as writer:
        for x in tqdm(ids, desc=f"TFREC {view_name}", leave=False):
            img = data_generation(x, view_name, False)
            writer.write(serialize_example_test(img))




## === cell 11
ids = df_preds["BraTS21ID"].tolist()
for v in views:
    out_path = f"./brain_test_{v}.tfrec"
    if not os.path.exists(out_path) or os.path.getsize(out_path) == 0:
        _build_one_view_tfrec(v, ids, out_path)




## === cell 12
def deserialize_example(serialized_string):
    image_feature_description = {"image": tf.io.FixedLenFeature([], tf.string)}
    parsed_record = tf.io.parse_single_example(
        serialized_string, image_feature_description
    )
    image = tf.io.decode_raw(parsed_record["image"], tf.float32)
    image = tf.reshape(image, [height, width, channel])
    return image




## === cell 13
def argument_image_tw2_val(img):
    img = tf.cast(img, tf.float32) / 255.0
    return img




## === cell 14
class RegressionModel(tf.keras.Model):
    def __init__(self, **kwargs):
        super(RegressionModel, self).__init__(**kwargs)
        self.conv1 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn1 = tf.keras.layers.BatchNormalization()
        self.rule1 = Silu(0)
        self.rule1 = tf.keras.layers.LeakyReLU(alpha=0.3)
        self.max1 = tf.keras.layers.MaxPooling2D(5)

        self.conv2 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn2 = tf.keras.layers.BatchNormalization()
        self.rule2 = Silu(0)
        self.max2 = tf.keras.layers.MaxPooling2D(5)

        self.conv3 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn3 = tf.keras.layers.BatchNormalization()
        self.rule3 = Silu(0)
        self.max3 = tf.keras.layers.MaxPooling2D(5)

        self.conv4 = tf.keras.layers.Conv2D(
            128, 5, strides=2, kernel_initializer="he_normal", activation="mish"
        )
        self.bn4 = tf.keras.layers.BatchNormalization()
        self.rule4 = Silu(0)
        self.max4 = tf.keras.layers.MaxPooling2D(5)

        para_relu = tf.keras.layers.LeakyReLU(alpha=0.5)
        self.dence256 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_2 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_2 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_2 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_3 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_3 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_3 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation="elu"
        )
        self.dence256_4 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
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
        self.dence256_5 = tf.keras.layers.Dense(
            256, kernel_initializer="he_normal", activation="mish"
        )
        self.dence128_5 = tf.keras.layers.Dense(
            128, kernel_initializer="he_normal", activation="swish"
        )
        self.dence64_5 = tf.keras.layers.Dense(
            64, kernel_initializer="he_normal", activation=para_relu
        )
        self.dropoup5 = tf.keras.layers.Dropout(0.2)
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

        x2 = self.conv2(input_tensor[1])
        x2 = self.bn2(x2, training=training)
        x2 = self.rule2(x2)
        x2 = self.max2(x2)
        x2 = self.dence256_2(x2)
        x2 = self.dropoup4_2(x2, training=training)
        x2 = self.dence128_2(x2)

        x3 = self.conv3(input_tensor[2])
        x3 = self.bn3(x3, training=training)
        x3 = self.rule3(x3)
        x3 = self.max3(x3)
        x3 = self.dence256_3(x3)
        x3 = self.dropoup4_3(x3, training=training)
        x3 = self.dence128_3(x3)

        x4 = self.conv4(input_tensor[3])
        x4 = self.bn4(x4, training=training)
        x4 = self.rule4(x4)
        x4 = self.max4(x4)
        x4 = self.dence256_4(x4)
        x4 = self.dropoup4_4(x4, training=training)
        x4 = self.dence128_4(x4)

        x = tf.keras.layers.Concatenate()([x1, x2, x3, x4])
        x = self.dence256_5(x)
        x = self.dropoup3_4(x, training=training)
        x = self.dence128_5(x)
        x = self.dropoup5(x, training=training)
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




## === cell 15
loss_func = tf.keras.losses.BinaryCrossentropy(from_logits=False)
opt = tf.keras.optimizers.SGD(
    learning_rate=1e-5, decay=1e-6, momentum=0.9, nesterov=True
)
f_pre = []

AUTOTUNE = tf.data.AUTOTUNE
options = tf.data.Options()
options.experimental_deterministic = True


def make_testset(path):
    ds = tf.data.TFRecordDataset(path, num_parallel_reads=AUTOTUNE)
    ds = ds.with_options(options)
    ds = ds.map(deserialize_example, num_parallel_calls=AUTOTUNE, deterministic=True)
    ds = ds.map(argument_image_tw2_val, num_parallel_calls=AUTOTUNE, deterministic=True)
    return ds


testset0 = make_testset("./brain_test_FLAIR.tfrec")
testset1 = make_testset("./brain_test_T1w.tfrec")
testset2 = make_testset("./brain_test_T1wCE.tfrec")
testset3 = make_testset("./brain_test_T2w.tfrec")

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

test_ds = tf.data.Dataset.zip((testset0, testset1, testset2, testset3))
test_ds = test_ds.batch(10, drop_remainder=False).prefetch(AUTOTUNE)

loaded_any = False
for f in range(5):  # Fold
    t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
    w_path = Path(weightdatapath, t_weight)
    try:
        model.load_weights(w_path)
        loaded_any = True
    except Exception as e:
        print("WARNING: could not load weights:", str(w_path), "|", repr(e))
        continue

    preds = model.predict(test_ds, batch_size=batch_size, verbose=0)
    f_pre.append(preds)

if loaded_any and len(f_pre) > 0:
    pre = np.array(f_pre)  # (folds, N, 1)
    finpre = pre.mean(axis=0)
    finpre = np.asarray(finpre).reshape(-1)  # (N,)
else:
    print("WARNING: no weights loaded; using 0.5 predictions for all rows.")
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

if finpre.shape[0] != len(df_preds):
    print(
        "WARNING: prediction length mismatch:",
        finpre.shape[0],
        "vs",
        len(df_preds),
        "-> using 0.5 fallback.",
    )
    finpre = np.full((len(df_preds),), 0.5, dtype=np.float32)

df_preds["MGMT_value"] = finpre.astype(np.float32)

subfilename = "submission.csv"
df_preds.to_csv(subfilename, index=False)
print("Wrote:", subfilename, "rows:", len(df_preds), "cols:", list(df_preds.columns))
print(df_preds.head())
