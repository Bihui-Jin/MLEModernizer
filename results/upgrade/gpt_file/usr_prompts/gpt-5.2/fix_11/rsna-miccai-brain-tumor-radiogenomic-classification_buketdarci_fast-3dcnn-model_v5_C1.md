# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

geopandas==0.14.4
google-api-python-client==2.177.0
ipython==7.34.0
ipython-genutils==0.2.0
ipython_pygments_lexers==1.1.1
ipython-sql==0.5.0
keras==3.8.0
keras-core==0.1.7
keras-cv==0.9.0
keras-hub==0.18.1
keras-nlp==0.18.1
keras-tuner==1.4.7
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
protobuf==6.33.0
pydicom==3.0.1
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
tensorflow==2.18.0
tensorflow-cloud==0.1.5
tensorflow-datasets==4.9.9
tensorflow_decision_forests==1.11.0
tensorflow-hub==0.16.1
tensorflow-io==0.37.1
tensorflow-io-gcs-filesystem==0.37.1
tensorflow-metadata==1.17.2
tensorflow-probability==0.25.0
tensorflow-text==2.18.1
tf_keras==2.18.0

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

# 5. Code solution

## === cell 0
import os
from glob import glob
import numpy as np
import pandas as pd
import re
import cv2
import pydicom
import matplotlib.pyplot as plt

np.random.seed(1)

DATA_DIR = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")
LABELS_CSV = os.path.join(DATA_DIR, "train_labels.csv")
SAMPLE_SUB_CSV = os.path.join(DATA_DIR, "sample_submission.csv")

print("Train dir exists:", os.path.isdir(TRAIN_DIR))
print("Test dir exists:", os.path.isdir(TEST_DIR))
print("Labels exists:", os.path.isfile(LABELS_CSV))
print("Sample submission exists:", os.path.isfile(SAMPLE_SUB_CSV))



## === cell 1
dicom_files = glob(os.path.join(TRAIN_DIR, "*", "T2w", "*.dcm"))
if len(dicom_files) == 0:
    raise RuntimeError("No DICOM files found under train/*/T2w/*.dcm (unexpected).")
print("Found example DICOM:", sorted(dicom_files)[0])




## === cell 2
def load_dicom(path: str) -> np.ndarray:
    """Read DICOM and return uint8 image scaled to [0,255]."""
    data = pydicom.dcmread(path, force=True, defer_size=64)
    img = data.pixel_array.astype(np.float32)
    mx = float(np.max(img))
    if mx > 0:
        img = img / mx
    img = (img * 255.0).astype(np.uint8)
    return img


def atoi(text):
    return int(text) if text.isdigit() else text


def natural_keys(text):
    return [atoi(c) for c in re.split(r"(\d+)", str(text))]


def all_slice(sequence, list_name):
    """Sort each subject's list of dicoms and append."""
    for x in range(len(sequence)):
        sequence[x].sort(key=natural_keys)
        list_name.append(sequence[x][0 : len(sequence[x])])
    return list_name




## === cell 3
df = pd.read_csv(LABELS_CSV)
df = df.set_index("BraTS21ID")
df = df.drop([109, 123, 709], axis=0, errors="ignore")
df = df.reset_index()

labels = np.array(df["MGMT_value"]).astype(np.float32)
print("Train labels shape:", labels.shape, "positives:", labels.mean())



## === cell 4
exclude_ids = {"00109", "00123", "00709"}

imagePatches = []
with os.scandir(TRAIN_DIR) as it:
    for e in it:
        if not e.is_dir():
            continue
        name = e.name
        if not name.isdigit():
            continue
        if name in exclude_ids:
            continue
        imagePatches.append(e.path)

imagePatches.sort(key=natural_keys)

print("Num train subjects (folders):", len(imagePatches))
print("Num labels after drop:", len(df))

folder_ids = [int(os.path.basename(os.path.normpath(p))) for p in imagePatches]
label_ids = df["BraTS21ID"].astype(int).tolist()
id_to_folder = {i: p for i, p in zip(folder_ids, imagePatches)}

missing_folders = [i for i in label_ids if i not in id_to_folder]
extra_folders = [i for i in folder_ids if i not in set(label_ids)]
print(
    "Missing folders for some labels:",
    missing_folders[:5],
    "count:",
    len(missing_folders),
)
print("Extra folders without labels:", extra_folders[:5], "count:", len(extra_folders))

common_ids = [i for i in label_ids if i in id_to_folder]
df = df[df["BraTS21ID"].astype(int).isin(common_ids)].copy()
df = df.sort_values("BraTS21ID").reset_index(drop=True)
labels = df["MGMT_value"].astype(np.float32).to_numpy()

imagePatches = [id_to_folder[int(i)] for i in df["BraTS21ID"].astype(int).tolist()]
print("Aligned train subjects:", len(imagePatches), "labels:", len(labels))



## === cell 5
MODALITIES = ("FLAIR", "T1w", "T1wCE", "T2w")

_image_num_re = re.compile(r"(\d+)")


def _dicom_num_key(path: str) -> int:
    base = os.path.basename(path)
    m = _image_num_re.search(base)
    return int(m.group(1)) if m else 0


def list_subject_modality_dicoms(subject_dir: str, modality: str):
    files = glob(os.path.join(subject_dir, modality, "*.dcm"))
    files.sort(key=_dicom_num_key)
    return files


def central_three(files):
    if len(files) == 0:
        return []
    mid = len(files) // 2
    idxs = [
        max(0, min(len(files) - 1, mid - 1)),
        max(0, min(len(files) - 1, mid)),
        max(0, min(len(files) - 1, mid + 1)),
    ]
    return [files[i] for i in idxs]




## === cell 6
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ReduceLROnPlateau

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

print("TF:", tf.__version__, "Keras:", keras.__version__)
tf.keras.utils.set_random_seed(1)

CACHE_DIR = "/kaggle/working/dicom_cache_256_central3"
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_path(subject_dir: str, modality: str) -> str:
    sid = os.path.basename(os.path.normpath(subject_dir))
    return os.path.join(CACHE_DIR, f"{sid}_{modality}.npy")


def build_central3_index(subject_dirs, modalities=MODALITIES):
    index = {}
    for subject_dir in subject_dirs:
        sd = os.path.normpath(subject_dir)
        per_mod = {}
        for modality in modalities:
            files = list_subject_modality_dicoms(sd, modality)
            per_mod[modality] = central_three(files)
        index[sd] = per_mod
    return index


def _build_one_cache_item(args):
    subject_dir, modality, files3 = args
    cp = _cache_path(subject_dir, modality)
    if os.path.isfile(cp):
        return 0
    x = np.zeros((256, 256, 3, 1), dtype=np.float32)
    for d, f in enumerate(files3):
        img = load_dicom(f)  # uint8 [0,255]
        img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
        x[:, :, d, 0] = img.astype(np.float32) / 255.0
    tmp = cp + ".tmp.npy"
    np.save(tmp, x)
    os.replace(tmp, cp)
    return 1


def prebuild_npy_cache(subject_dirs, modalities=MODALITIES, index=None, workers=None):
    from concurrent.futures import ProcessPoolExecutor, as_completed

    subject_dirs = [os.path.normpath(p) for p in subject_dirs]
    tasks = []
    for sd in subject_dirs:
        for modality in modalities:
            cp = _cache_path(sd, modality)
            if os.path.isfile(cp):
                continue
            files3 = (
                index[sd][modality]
                if index is not None
                else central_three(list_subject_modality_dicoms(sd, modality))
            )
            tasks.append((sd, modality, files3))

    if not tasks:
        return

    if workers is None:
        workers = max(1, min(8, (os.cpu_count() or 2) - 1))

    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = [ex.submit(_build_one_cache_item, t) for t in tasks]
        for _ in as_completed(futs):
            pass


class Dicom3DSequence(keras.utils.Sequence):
    def __init__(
        self, subject_dirs, y, modality, batch_size=8, shuffle=True, index=None
    ):
        self.subject_dirs = [os.path.normpath(p) for p in subject_dirs]
        self.y = None if y is None else np.asarray(y, dtype=np.float32)
        self.modality = modality
        self.batch_size = int(batch_size)
        self.shuffle = bool(shuffle)
        self.indexes = np.arange(len(self.subject_dirs), dtype=np.int32)
        self._cache_mem = {}
        self._index = index  # optional precomputed central3 file list
        self.on_epoch_end()

    def __len__(self):
        return (len(self.subject_dirs) + self.batch_size - 1) // self.batch_size

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indexes)

    def _get_item_x(self, subj_idx: int):
        if subj_idx in self._cache_mem:
            return self._cache_mem[subj_idx]

        subject_dir = self.subject_dirs[subj_idx]
        cp = _cache_path(subject_dir, self.modality)

        if os.path.isfile(cp):
            x = np.load(cp)
            if x.dtype != np.float32:
                x = x.astype(np.float32, copy=False)
            self._cache_mem[subj_idx] = x
            return x

        if self._index is not None:
            files3 = self._index[subject_dir][self.modality]
        else:
            files3 = central_three(
                list_subject_modality_dicoms(subject_dir, self.modality)
            )

        x = np.zeros((256, 256, 3, 1), dtype=np.float32)
        for d, f in enumerate(files3):
            img = load_dicom(f)  # uint8 [0,255]
            img = cv2.resize(img, (256, 256), interpolation=cv2.INTER_AREA)
            x[:, :, d, 0] = img.astype(np.float32) / 255.0

        np.save(cp, x)
        self._cache_mem[subj_idx] = x
        return x

    def __getitem__(self, batch_index):
        sl = slice(batch_index * self.batch_size, (batch_index + 1) * self.batch_size)
        batch_ids = self.indexes[sl]
        xb = np.empty((len(batch_ids), 256, 256, 3, 1), dtype=np.float32)
        for i, subj_idx in enumerate(batch_ids):
            xb[i] = self._get_item_x(int(subj_idx))
        if self.y is None:
            return xb
        yb = self.y[batch_ids]
        return xb, yb


def make_dataset(subject_dirs, y, modality, batch_size, shuffle, seed=1):
    subject_dirs = [os.path.normpath(p) for p in subject_dirs]
    paths = np.array([_cache_path(sd, modality) for sd in subject_dirs], dtype=object)

    if y is None:
        ds = tf.data.Dataset.from_tensor_slices(paths)
    else:
        y = np.asarray(y, dtype=np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, y))

    if shuffle:
        ds = ds.shuffle(
            buffer_size=len(subject_dirs), seed=seed, reshuffle_each_iteration=True
        )

    def _load_x(p):
        x = np.load(p.decode("utf-8")).astype(np.float32, copy=False)
        return x

    if y is None:

        def _map_x(p):
            x = tf.numpy_function(_load_x, [p], Tout=tf.float32)
            x.set_shape((256, 256, 3, 1))
            return x

        ds = ds.map(_map_x, num_parallel_calls=tf.data.AUTOTUNE)
    else:

        def _map_xy(p, yy):
            x = tf.numpy_function(_load_x, [p], Tout=tf.float32)
            x.set_shape((256, 256, 3, 1))
            yy = tf.cast(yy, tf.float32)
            return x, yy

        ds = ds.map(_map_xy, num_parallel_calls=tf.data.AUTOTUNE)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 7
def get_model(optimizer):
    inputs = keras.Input((256, 256, 3, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu", padding="same")(
        inputs
    )
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu", padding="same")(x)
    x = layers.MaxPool3D(pool_size=(2, 2, 1))(x)
    x = layers.BatchNormalization()(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=512, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    x = layers.Dense(units=256, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3dcnn")
    model.compile(loss="binary_crossentropy", optimizer=optimizer, metrics=["accuracy"])
    return model


model = get_model(keras.optimizers.Adam())
model.summary()



## === cell 8
idx = np.arange(len(imagePatches))
idx_train, idx_valid, y_train, y_valid = train_test_split(
    idx, labels, test_size=0.2, random_state=1, stratify=labels
)
train_dirs = [imagePatches[i] for i in idx_train]
valid_dirs = [imagePatches[i] for i in idx_valid]

print(
    "Train/valid sizes:",
    len(train_dirs),
    len(valid_dirs),
    "valid positives:",
    float(y_valid.mean()),
)

all_train_valid_dirs = [os.path.normpath(p) for p in (train_dirs + valid_dirs)]
central3_index = build_central3_index(all_train_valid_dirs, modalities=MODALITIES)

prebuild_npy_cache(all_train_valid_dirs, modalities=MODALITIES, index=central3_index)
print("Prebuilt cache files:", len(glob(os.path.join(CACHE_DIR, "*.npy"))))




## === cell 9
def train_one_modality_fixedsplit(train_dirs, y_train, valid_dirs, y_valid, modality):
    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss", factor=0.3, patience=3, mode="auto", verbose=1
    )

    train_ds = make_dataset(
        train_dirs, y_train, modality=modality, batch_size=8, shuffle=True, seed=1
    )
    valid_ds = make_dataset(
        valid_dirs, y_valid, modality=modality, batch_size=8, shuffle=False, seed=1
    )

    m = get_model(keras.optimizers.Adam())

    h = m.fit(
        train_ds,
        epochs=50,
        validation_data=valid_ds,
        callbacks=[reduce_lr],
        verbose=2,
    )
    return m, h


model_flair, hist_flair = train_one_modality_fixedsplit(
    train_dirs, y_train, valid_dirs, y_valid, "FLAIR"
)
model_t1w, hist_t1w = train_one_modality_fixedsplit(
    train_dirs, y_train, valid_dirs, y_valid, "T1w"
)
model_t1wce, hist_t1wce = train_one_modality_fixedsplit(
    train_dirs, y_train, valid_dirs, y_valid, "T1wCE"
)
model_t2w, hist_t2w = train_one_modality_fixedsplit(
    train_dirs, y_train, valid_dirs, y_valid, "T2w"
)



## === cell 10
os.makedirs("/kaggle/working/models", exist_ok=True)
model_flair.save("/kaggle/working/models/flairinputs.keras")
model_t1w.save("/kaggle/working/models/t1winputs.keras")
model_t1wce.save("/kaggle/working/models/t1wceinputs.keras")
model_t2w.save("/kaggle/working/models/t2winputs.keras")




## === cell 11
def valid_preds_for_model(model, subject_dirs, modality):
    ds = make_dataset(
        subject_dirs, y=None, modality=modality, batch_size=8, shuffle=False, seed=1
    )
    return model.predict(ds, verbose=0).reshape(-1)


preds_flair = valid_preds_for_model(model_flair, valid_dirs, "FLAIR")
preds_t1w = valid_preds_for_model(model_t1w, valid_dirs, "T1w")
preds_t1wce = valid_preds_for_model(model_t1wce, valid_dirs, "T1wCE")
preds_t2w = valid_preds_for_model(model_t2w, valid_dirs, "T2w")

mean_valid = (preds_flair + preds_t1w + preds_t2w + preds_t1wce) / 4.0
auc_score = roc_auc_score(y_valid, mean_valid)
print("Validation AUC:", float(auc_score))



## === cell 12
testimages = []
with os.scandir(TEST_DIR) as it:
    for e in it:
        if e.is_dir() and e.name.isdigit():
            testimages.append(e.path)
testimages.sort(key=natural_keys)

print("Num test subjects:", len(testimages))

test_dirs_norm = [os.path.normpath(p) for p in testimages]
central3_index_test = build_central3_index(test_dirs_norm, modalities=MODALITIES)

prebuild_npy_cache(test_dirs_norm, modalities=MODALITIES, index=central3_index_test)
print("Cache files after test prebuild:", len(glob(os.path.join(CACHE_DIR, "*.npy"))))




## === cell 13
def test_preds_for_model(model, subject_dirs, modality):
    ds = make_dataset(
        subject_dirs, y=None, modality=modality, batch_size=8, shuffle=False, seed=1
    )
    return model.predict(ds, verbose=0).reshape(-1)


preds_flair = test_preds_for_model(model_flair, testimages, "FLAIR")
preds_t1w = test_preds_for_model(model_t1w, testimages, "T1w")
preds_t2w = test_preds_for_model(model_t2w, testimages, "T2w")
preds_t1wce = test_preds_for_model(model_t1wce, testimages, "T1wCE")

mean_test = (preds_flair + preds_t1w + preds_t2w + preds_t1wce) / 4.0
mean_test = np.clip(mean_test, 0.0, 1.0)

print("Pred range:", float(mean_test.min()), float(mean_test.max()))

sample = pd.read_csv(SAMPLE_SUB_CSV)
sample_ids = sample["BraTS21ID"].astype(str).tolist()

test_ids = [os.path.basename(os.path.normpath(p)) for p in testimages]
id_to_pred = {tid: float(p) for tid, p in zip(test_ids, mean_test)}
preds_ordered = [id_to_pred.get(tid, 0.5) for tid in sample_ids]

submission = pd.DataFrame({"BraTS21ID": sample_ids, "MGMT_value": preds_ordered})
submission.to_csv("submission.csv", index=False)

print(submission.head())
print("Wrote submission.csv with shape:", submission.shape)
print("submission.csv exists:", os.path.isfile("submission.csv"))
