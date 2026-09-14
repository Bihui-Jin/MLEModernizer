# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'I fix the two blocking runtime issues: (1) prevent the pydicom/protobuf import crash by defaulting to the OpenCV DICOM decode path (so the notebook can run reliably in the Kaggle environment), and (2) handle missing external checkpoint files by falling back to a deterministic, label-based baseline prediction that still produces a valid `submission.csv`. This keeps the model architecture and inference pipeline intact when weights exist, but guarantees an end-to-end run even when they don’t. I also make the test path selection robust (try the competition folder first, then the flat `/input/test`) and ensure the submission columns/types match the required format.'
- What this solution (achieved 0.5) has done: 'I fix the blocking import/runtime issue by avoiding the `protobuf`/`pydicom` code path entirely (the current crash happens even though pydicom is “disabled”), and make the DICOM loading deterministic and safe by using OpenCV-only decoding with filename sorting. I also remove the multiprocessing precompute (it can fail/pickle in Kaggle and isn’t needed for correctness) and replace it with a single-process loop that reliably finishes and produces the same feature tensors. Finally, I keep the existing checkpoint-loading/inference logic intact, ensure the test path is correct, and always write a valid `submission.csv` with the exact required columns and ID formatting.'
- What this solution (achieved 0.5) has done: 'I fix the blocking `MessageFactory.GetPrototype` crash by avoiding any import path that triggers the incompatible protobuf/pydicom machinery, and by making the OpenCV-based DICOM decode self-contained and safe. I also correct a subtle but important data bug: `load_imgs()` currently always reads from `testdatapath`, which breaks training/inference consistency and can silently produce all-zero features if reused; I make it accept an explicit `datapath` and use the correct one for test. Finally, because your checkpoints path is missing in this environment, I keep the existing “fallback prior” behavior but ensure it deterministically uses the training-label prior and always writes a valid `submission.csv` with correct dtypes/ID formatting (score remain around the 0.5 baseline unless real weights are present).'
- What this solution (achieved 0.5) has done: 'I fix the blocking runtime crash by ensuring TensorFlow is imported in a way that avoids the known protobuf `MessageFactory.GetPrototype` incompatibility in this Kaggle image (the current environment variable setting forces the pure-Python protobuf path, which triggers the crash). I also remove the now-unneeded “pydicom disabled” scaffolding that still doesn’t prevent the protobuf crash, while keeping your OpenCV-only DICOM loading logic intact. The rest of the pipeline (feature generation, model, checkpoint loading/fallback prior, and submission writing) remain the same so behavior/score stays consistent with your current ~0.5 baseline when checkpoints are missing. Finally, I keep the robust path selection and ensure the submission is always written to `submission.csv` with the required columns.'
- What this solution (achieved 0.5) has done: 'I fix the immediate TensorFlow/protobuf crash by setting the protobuf environment variable *before* importing TensorFlow, forcing the C++ implementation that’s compatible with Kaggle’s TF build. I keep your OpenCV-only DICOM loading and the model/inference logic unchanged, only making the import sequence and environment configuration robust so the notebook runs end-to-end. I also add a small safety fallback so that if TensorFlow still cannot import for any reason, the script still writes a valid `submission.csv` using the training-label prior (same semantics as your existing fallback). These changes are score-neutral when checkpoints are missing, but they unblock execution so you can actually generate a submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "cpp")

import random
import cv2
import numpy as np
import gc
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import math

TF_AVAILABLE = True
try:
    import tensorflow as tf
except Exception as e:
    TF_AVAILABLE = False
    TF_IMPORT_ERROR = repr(e)

print("Info: Using OpenCV-based DICOM decode only.")
if not TF_AVAILABLE:
    print(
        "Warning: TensorFlow failed to import; will use prior-only submission fallback."
    )
    print("TF import error:", TF_IMPORT_ERROR)



## === cell 1
df_preds = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
weightdatapath = Path("../input/weight-multi-20210919")

_candidate_test_paths = [
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/test",
    "../input/test",
]
testdatapath = None
for p in _candidate_test_paths:
    if os.path.isdir(p):
        testdatapath = p
        break
if testdatapath is None:
    testdatapath = _candidate_test_paths[0]

print("Using testdatapath:", testdatapath)



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
    if TF_AVAILABLE:
        tf.random.set_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


set_seed(seed)



## === cell 4
if TF_AVAILABLE:

    @tf.keras.utils.register_keras_serializable(package="custom", name="Mish")
    def mish(x):
        return x * tf.math.tanh(tf.math.softplus(x))

    @tf.keras.utils.register_keras_serializable(package="custom", name="Siren")
    def siren(x):
        return 1.0 / (1.0 + tf.math.exp(-x))

    tf.keras.utils.get_custom_objects().update({"Mish": mish, "Siren": siren})



## === cell 5
if TF_AVAILABLE:

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
def _safe_dicom_pixel_array_opencv(dcm_path):
    """
    DICOM decoder using OpenCV's DICOM support.
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
    return _safe_dicom_pixel_array_opencv(dcm_path)


def load_imgs(idx, view, datapath, ignore_zeros=True):
    """
    Load a DICOM series as a (z, height, width) float32 numpy array.
    Deterministic OpenCV-based decode; sort by filename for stability.
    """
    series_dir = os.path.join(datapath, idx, view)
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


def data_generation(ID, view, datapath, is_Train=True):
    idx = str(ID).zfill(5)
    imgs = load_imgs(idx, view, datapath=datapath, ignore_zeros=False).astype(
        np.float32
    )
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

X0_list, X1_list, X2_list, X3_list = [], [], [], []
for sid in tqdm(_ids_int, total=len(_ids_int), desc="Precompute test features"):
    X0_list.append(
        data_generation(sid, views[0], datapath=testdatapath, is_Train=False).astype(
            np.float32
        )
    )
    X1_list.append(
        data_generation(sid, views[1], datapath=testdatapath, is_Train=False).astype(
            np.float32
        )
    )
    X2_list.append(
        data_generation(sid, views[2], datapath=testdatapath, is_Train=False).astype(
            np.float32
        )
    )
    X3_list.append(
        data_generation(sid, views[3], datapath=testdatapath, is_Train=False).astype(
            np.float32
        )
    )

X0 = np.stack(X0_list, axis=0)
X1 = np.stack(X1_list, axis=0)
X2 = np.stack(X2_list, axis=0)
X3 = np.stack(X3_list, axis=0)
del X0_list, X1_list, X2_list, X3_list
gc.collect()

print("Precomputed X shapes:", X0.shape, X1.shape, X2.shape, X3.shape)



## === cell 9
if TF_AVAILABLE:

    def argument_image_tw2_val(img):
        img = tf.cast(img, tf.float32) / 255.0
        return img




## === cell 10
if TF_AVAILABLE:

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
                loss = self.compiled_loss(
                    y, predictions, regularization_losses=self.losses
                )
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
def _load_prior_from_train_labels():
    train_labels_path_candidates = [
        "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv",
        "../input/train_labels.csv",
    ]
    train_labels_path = None
    for p in train_labels_path_candidates:
        if os.path.isfile(p):
            train_labels_path = p
            break
    if train_labels_path is None:
        print("Warning: train_labels.csv not found; using prior=0.5")
        return 0.5
    ytrain = pd.read_csv(train_labels_path)["MGMT_value"].astype(float).values
    prior = float(np.clip(ytrain.mean(), 1e-4, 1 - 1e-4))
    print("Using baseline prior from train labels:", prior)
    return prior


if not TF_AVAILABLE:
    prior = _load_prior_from_train_labels()
    finpre = np.full((len(df_preds),), prior, dtype=np.float32)
else:

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
            raise ValueError(
                f"No variables assigned from checkpoint: {ckpt_prefix_path}"
            )
        return assigned, skipped

    def _ckpt_exists(prefix_path: Path) -> bool:
        return (
            Path(str(prefix_path) + ".index").exists()
            and len(list(Path(prefix_path).parent.glob(prefix_path.name + ".data*")))
            > 0
        )

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

    ckpt_missing = False
    for f in range(5):  # Fold
        t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
        ckpt_path = Path(weightdatapath, t_weight)
        if not _ckpt_exists(ckpt_path):
            print("Warning: checkpoint not found:", ckpt_path)
            ckpt_missing = True
            break

    if not ckpt_missing:
        for f in range(5):  # Fold
            t_weight = "weight-Regression-multi_fold_0" + str(f) + "-" + epoch + ".ckpt"
            ckpt_path = Path(weightdatapath, t_weight)
            assigned, skipped = load_tf_checkpoint_into_keras_model(model, ckpt_path)
            f_pre.append(model.predict(test_ds, batch_size=batch_size, verbose=0))

        pre = np.stack(f_pre, axis=0)  # (n_folds, n_test, 1)
        finpre = pre.mean(axis=0).reshape(-1)  # (n_test,)
    else:
        prior = _load_prior_from_train_labels()
        finpre = np.full((len(df_preds),), prior, dtype=np.float32)

df_preds["BraTS21ID"] = df_preds["BraTS21ID"].astype(str).str.zfill(5)
df_preds["MGMT_value"] = finpre.astype(np.float32)

subfilename = "submission.csv"
df_preds.to_csv(subfilename, index=False)

print("Wrote", subfilename, "with shape", df_preds.shape)
print(df_preds.head())
