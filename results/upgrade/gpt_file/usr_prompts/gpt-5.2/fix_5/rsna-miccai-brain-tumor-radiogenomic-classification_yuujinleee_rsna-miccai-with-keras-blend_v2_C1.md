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

# 8. Previous improvement plans

- What this solution (achieved 0.61059) has done: 'I fix the import/runtime crash by avoiding the fragile `protobuf`/`pydicom` import path that triggers `MessageFactory.GetPrototype`, and I replace deprecated `pydicom.read_file` with `pydicom.dcmread`. Since the external pretrained `.h5` file path doesn’t exist, I keep the same overall approach (2D CNN on slices + patient-level averaging) but train a small Keras model inside the notebook so we can generate predictions end-to-end. I also correct the submission logic to output probabilities (not `argmax` class labels) because the metric is ROC AUC. Finally, I keep the I/O paths the same, exclude the known-bad cases, and ensure a valid `submission.csv` is written with the required columns.'
- What this solution (achieved 0.61647) has done: 'We fix the crash happening at import-time by forcing the pure-Python protobuf implementation before TensorFlow loads, which avoids the `MessageFactory.GetPrototype` error in this Kaggle image. Then we keep the rest of your pipeline intact (same DICOM loading, same 2D CNN, same slice-averaging inference), only adding small robustness guards so the script still produces a valid submission even if pydicom is unavailable or some subjects yield zero slices. These changes are score-neutral in intent and primarily unblock end-to-end execution and correct submission generation. The output still be a `submission.csv` with the required columns and probabilities for ROC AUC.'

# 9. Code solution

## === cell 0
import os
import random
import glob
import numpy as np
import pandas as pd

SEED = 42

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")

os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

import cv2
from tqdm.notebook import tqdm

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

try:
    tf.keras.utils.set_random_seed(SEED)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
BASE_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
TRAIN_LABELS_PATH = os.path.join(BASE_PATH, "train_labels.csv")
SAMPLE_SUB_PATH = os.path.join(BASE_PATH, "sample_submission.csv")
TRAIN_DIR = os.path.join(BASE_PATH, "train")
TEST_DIR = os.path.join(BASE_PATH, "test")

train_df = pd.read_csv(TRAIN_LABELS_PATH)
test_df = pd.read_csv(SAMPLE_SUB_PATH)

train_df.head()




## === cell 2
EXCLUDE = [109, 123, 709]
train_df = train_df[~train_df.BraTS21ID.isin(EXCLUDE)].reset_index(drop=True)
train_df.shape, test_df.shape




## === cell 3
TYPES = ["FLAIR", "T1w", "T1wCE", "T2w"]




## === cell 4
try:
    import pydicom

    _HAVE_PYDICOM = True
except Exception as e:
    print(
        "Warning: pydicom import failed, will fall back to cv2 for DICOM where possible.\n",
        repr(e),
    )
    pydicom = None
    _HAVE_PYDICOM = False


def load_dicom(path, size=64):
    if _HAVE_PYDICOM:
        dicom = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=True,  # skip non-essential parsing before pixel data
            specific_tags=[
                "PixelData",
                "RescaleIntercept",
                "RescaleSlope",
                "WindowCenter",
                "WindowWidth",
            ],
        )
        data = dicom.pixel_array.astype(np.float32)
        mx = float(np.max(data))
        if mx > 0:
            data = data / mx
        data = (data * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.resize(data, (size, size), interpolation=cv2.INTER_AREA)
    else:
        img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            raise RuntimeError(f"Could not read DICOM without pydicom: {path}")
        img = img.astype(np.float32)
        mx = float(np.max(img))
        if mx > 0:
            img = img / mx
        img = (img * 255.0).clip(0, 255).astype(np.uint8)
        return cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA)




## === cell 5
def _safe_slice_index_from_filename(p):
    base = os.path.splitext(os.path.basename(p))[0]
    try:
        return int(base.split("-")[-1])
    except Exception:
        return 10**18  # send unknowns to the end deterministically


def _build_dicom_index(base_dir):
    index = {}
    if not os.path.isdir(base_dir):
        return index
    with os.scandir(base_dir) as it:
        for ent in it:
            if not ent.is_dir():
                continue
            pid = ent.name
            pdir = ent.path
            index[pid] = {}
            for t in TYPES:
                tdir = os.path.join(pdir, t)
                if not os.path.isdir(tdir):
                    index[pid][t] = ()
                    continue
                files = []
                with os.scandir(tdir) as fit:
                    for f in fit:
                        if f.is_file():
                            files.append(f.path)
                files.sort(key=_safe_slice_index_from_filename)
                index[pid][t] = tuple(files)
    return index


_TRAIN_INDEX = _build_dicom_index(TRAIN_DIR)
_TEST_INDEX = _build_dicom_index(TEST_DIR)

_PATHS_CACHE = {("train", t): {} for t in TYPES}
_PATHS_CACHE.update({("test", t): {} for t in TYPES})


def get_all_image_paths(BraTS21ID, image_type, folder="train"):
    assert image_type in TYPES
    pid_str = str(BraTS21ID).zfill(5)

    cache = _PATHS_CACHE[(folder, image_type)]
    if pid_str in cache:
        return cache[pid_str]

    idx = _TRAIN_INDEX if folder == "train" else _TEST_INDEX
    paths = idx.get(pid_str, {}).get(image_type, ())

    num_images = len(paths)
    if num_images == 0:
        out = np.array([], dtype=object)
        cache[pid_str] = out
        return out

    start = int(num_images * 0.25)
    end = int(num_images * 0.75)
    jump = 1 if num_images < 10 else 3

    out = np.array(paths[start:end:jump], dtype=object)
    cache[pid_str] = out
    return out


def get_all_images(BraTS21ID, image_type, folder="train", size=128):
    paths = get_all_image_paths(BraTS21ID, image_type, folder)
    return [load_dicom(path, size) for path in paths]




## === cell 6
IMAGE_SIZE = 128


def get_all_data_train(image_type):
    X, y, train_ids = [], [], []
    ids = train_df["BraTS21ID"].to_numpy()
    labels = train_df["MGMT_value"].to_numpy(dtype=np.float32)

    for pid, label in tqdm(list(zip(ids, labels))):
        pid = int(pid)
        images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
        if not images:
            continue
        X.extend(images)
        y.extend([float(label)] * len(images))
        train_ids.extend([pid] * len(images))

    X = np.array(X, dtype=np.uint8)
    y = np.array(y, dtype=np.float32)
    train_ids = np.array(train_ids, dtype=np.int32)
    return X, y, train_ids


def get_all_data_test(image_type):
    X, test_ids = [], []
    ids = test_df["BraTS21ID"].to_numpy()

    for pid in tqdm(ids):
        pid = int(pid)
        images = get_all_images(pid, image_type, "test", IMAGE_SIZE)
        if not images:
            continue
        X.extend(images)
        test_ids.extend([pid] * len(images))

    X = np.array(X, dtype=np.uint8)
    test_ids = np.array(test_ids, dtype=np.int32)
    return X, test_ids




## === cell 7
X, y, train_idt = get_all_data_train("T1wCE")
X_test, test_idt = get_all_data_test("T1wCE")

X.shape, y.shape, train_idt.shape, X_test.shape, test_idt.shape




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/74597255.py in <cell line: 0>()
----> 1 X, y, train_idt = get_all_data_train("T1wCE")
      2 X_test, test_idt = get_all_data_test("T1wCE")
      3 
      4 X.shape, y.shape, train_idt.shape, X_test.shape, test_idt.shape
      5 

/tmp/ipykernel_11/2744093556.py in get_all_data_train(image_type)
     11     for pid, label in tqdm(list(zip(ids, labels))):
     12         pid = int(pid)
---> 13         images = get_all_images(pid, image_type, "train", IMAGE_SIZE)
     14         if not images:
     15             continue

/tmp/ipykernel_11/2066287064.py in get_all_images(BraTS21ID, image_type, folder, size)
     71 def get_all_images(BraTS21ID, image_type, folder="train", size=128):
     72     paths = get_all_image_paths(BraTS21ID, image_type, folder)
---> 73     return [load_dicom(path, size) for path in paths]
     74 
     75 

/tmp/ipykernel_11/2066287064.py in <listcomp>(.0)
     71 def get_all_images(BraTS21ID, image_type, folder="train", size=128):
     72     paths = get_all_image_paths(BraTS21ID, image_type, folder)
---> 73     return [load_dicom(path, size) for path in paths]
     74 
     75 

/tmp/ipykernel_11/1771603592.py in load_dicom(path, size)
     29         )
     30         # Accessing pixel_array triggers pixel data decode; semantics same as before.
---> 31         data = dicom.pixel_array.astype(np.float32)
     32         mx = float(np.max(data))
     33         if mx > 0:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in pixel_array(self)
   2191             that iterates through the image frames.
   2192         """
-> 2193         self.convert_pixel_data()
   2194         return cast("numpy.ndarray", self._pixel_array)
   2195 

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in convert_pixel_data(self, handler_name)
   1724             # Use 'pydicom.pixels' backend
   1725             opts["decoding_plugin"] = name
-> 1726             self._pixel_array = pixel_array(self, **opts)
   1727             self._pixel_id = get_image_pixel_ids(self)
   1728         else:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/utils.py in pixel_array(src, ds_out, specific_tags, index, raw, decoding_plugin, **kwargs)
   1428 
   1429         opts = as_pixel_options(ds, **kwargs)
-> 1430         return decoder.as_array(
   1431             ds,
   1432             index=index,

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in as_array(self, src, index, validate, raw, decoding_plugin, **kwargs)
    975 
    976         runner = DecodeRunner(self.UID)
--> 977         runner.set_source(src)
    978         runner.set_options(**kwargs)
    979         runner.set_decoders(

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in set_source(self, src)
    684 
    685         if isinstance(src, Dataset):
--> 686             self._set_options_ds(src)
    687             self._src = src[self.pixel_keyword].value
    688             if isinstance(self._src, BufferedIOBase):

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _set_options_ds(self, ds)
    657         px_keyword = [kw for kw in keywords if kw in ds]
    658         if not px_keyword:
--> 659             raise AttributeError(
    660                 "The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double "
    661                 "Float Pixel Data' element, no pixel data to decode"

AttributeError: The dataset has no 'Pixel Data', 'Float Pixel Data' or 'Double Float Pixel Data' element, no pixel data to decode

## === cell 8
if X.size == 0 or X_test.size == 0:
    raise RuntimeError(
        f"No images loaded. Train X size={X.size}, Test X size={X_test.size}. "
        "Check DICOM reading and paths."
    )

X = (X.astype(np.float32) / 255.0)[..., None]
X_test = (X_test.astype(np.float32) / 255.0)[..., None]

from sklearn.model_selection import train_test_split

X_train, X_val, y_train, y_val = train_test_split(
    X, y, test_size=0.2, random_state=SEED, stratify=y
)
X_train.shape, X_val.shape




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1369642963.py in <cell line: 0>()
----> 1 if X.size == 0 or X_test.size == 0:
      2     raise RuntimeError(
      3         f"No images loaded. Train X size={X.size}, Test X size={X_test.size}. "
      4         "Check DICOM reading and paths."
      5     )

NameError: name 'X' is not defined

## === cell 9
def build_model(input_shape=(IMAGE_SIZE, IMAGE_SIZE, 1)):
    inputs = keras.Input(shape=input_shape)
    x = layers.Conv2D(16, 3, padding="same", activation="relu")(inputs)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(x)
    x = layers.MaxPooling2D()(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)
    model = keras.Model(inputs, outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="binary_crossentropy")
    return model


model = build_model()
model.summary()




## === cell 10
BATCH_SIZE = 64
EPOCHS = (
    3  # keep small to fit time/memory constraints while producing a valid submission
)

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_val, y_val),
    epochs=EPOCHS,
    batch_size=BATCH_SIZE,
    verbose=2,
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1272012261.py in <cell line: 0>()
      5 
      6 history = model.fit(
----> 7     X_train,
      8     y_train,
      9     validation_data=(X_val, y_val),

NameError: name 'X_train' is not defined

## === cell 11
from sklearn.metrics import roc_auc_score

val_pred = model.predict(X_val, batch_size=256, verbose=0).reshape(-1)
try:
    auc = roc_auc_score(y_val, val_pred)
except Exception:
    auc = None
auc




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3153621253.py in <cell line: 0>()
      1 from sklearn.metrics import roc_auc_score
      2 
----> 3 val_pred = model.predict(X_val, batch_size=256, verbose=0).reshape(-1)
      4 try:
      5     auc = roc_auc_score(y_val, val_pred)

NameError: name 'X_val' is not defined

## === cell 12
test_slice_pred = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
test_slice_pred = np.clip(test_slice_pred, 1e-6, 1 - 1e-6)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/286566039.py in <cell line: 0>()
----> 1 test_slice_pred = model.predict(X_test, batch_size=256, verbose=0).reshape(-1)
      2 test_slice_pred = np.clip(test_slice_pred, 1e-6, 1 - 1e-6)
      3 
      4 

NameError: name 'X_test' is not defined

## === cell 13
result = pd.DataFrame(
    {"BraTS21ID": test_idt.astype(int), "MGMT_value": test_slice_pred.astype(float)}
)
result_final = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()

sample_sub = pd.read_csv(SAMPLE_SUB_PATH)
result_final = sample_sub[["BraTS21ID"]].merge(result_final, on="BraTS21ID", how="left")

result_final["MGMT_value"] = result_final["MGMT_value"].fillna(0.5).astype(float)
result_final["MGMT_value"] = result_final["MGMT_value"].clip(1e-6, 1 - 1e-6)

result_final.head(), result_final.shape




## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1369468922.py in <cell line: 0>()
      1 result = pd.DataFrame(
----> 2     {"BraTS21ID": test_idt.astype(int), "MGMT_value": test_slice_pred.astype(float)}
      3 )
      4 result_final = result.groupby("BraTS21ID", as_index=False)["MGMT_value"].mean()
      5 

NameError: name 'test_idt' is not defined

## === cell 14
SUB_PATH = "submission.csv"
result_final.to_csv(SUB_PATH, index=False)
print("Wrote", SUB_PATH)
print(result_final.head())

## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3441732069.py in <cell line: 0>()
      1 SUB_PATH = "submission.csv"
----> 2 result_final.to_csv(SUB_PATH, index=False)
      3 print("Wrote", SUB_PATH)
      4 print(result_final.head())

NameError: name 'result_final' is not defined
