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
import glob
import re
import math
import numpy as np
import pandas as pd
import cv2

from sklearn import model_selection as sk_model_selection

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.metrics import AUC

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

np.random.seed(12)
tf.random.set_seed(12)

print("TF version:", tf.__version__)



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
gpus = tf.config.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except Exception:
        pass



## === cell 2
data_directory = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

mri_types_orig = ["FLAIR", "T1w", "T1wCE", "T2w"]
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]

IMAGE_SIZE = 96
NUM_IMAGES_PER_TYPE = 30
NUM_IMAGES = NUM_IMAGES_PER_TYPE * len(mri_types)
BATCH_SIZE = 4

train_df = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv"
)

to_exclude = [109, 123, 709]
train_df = train_df[~train_df["BraTS21ID"].isin(to_exclude)].reset_index(drop=True)

train_df["BraTS21ID5"] = [format(int(x), "05d") for x in train_df.BraTS21ID]
print("Train rows:", len(train_df))
train_df.head(3)



## === cell 3
sample_submission = pd.read_csv(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv"
)
test = sample_submission.copy()
test["BraTS21ID5"] = [format(int(x), "05d") for x in test.BraTS21ID]
test.head(3)




## === cell 4
def _natural_key(path):
    return [
        int(x) if x.isdigit() else x
        for x in re.findall(r"[^0-9]|[0-9]+", os.path.basename(path))
    ]




## === cell 5
try:
    import pydicom
    from pydicom.pixel_data_handlers.util import apply_voi_lut
except Exception:
    pydicom = None
    apply_voi_lut = None

_DICOM_FILES_CACHE = {}  # key: (split, scan_id, mri_type) -> sorted list of file paths

_DISK_CACHE_DIR = "/kaggle/working/vol_cache_npy"
os.makedirs(_DISK_CACHE_DIR, exist_ok=True)

_RE_DICOM_NUM = re.compile(r"(\d+)")

_SLICE_CACHE = {}
_SLICE_CACHE_MAX = 8192  # bounded to prevent unbounded growth


def _slice_cache_get(key):
    return _SLICE_CACHE.get(key)


def _slice_cache_put(key, value):
    if len(_SLICE_CACHE) >= _SLICE_CACHE_MAX:
        _SLICE_CACHE.pop(next(iter(_SLICE_CACHE)))
    _SLICE_CACHE[key] = value


def _sorted_dcm_files_fast(folder):
    try:
        entries = []
        with os.scandir(folder) as it:
            for e in it:
                if e.is_file() and e.name.endswith(".dcm"):
                    m = _RE_DICOM_NUM.search(e.name)
                    idx = int(m.group(1)) if m else -1
                    entries.append((idx, e.name))
        entries.sort(key=lambda x: (x[0], x[1]))
        return [os.path.join(folder, name) for _, name in entries]
    except FileNotFoundError:
        return []


def _get_sorted_dicom_files(scan_id, split, mri_type):
    key = (split, scan_id, mri_type)
    files = _DICOM_FILES_CACHE.get(key)
    if files is None:
        folder = f"{data_directory}/{split}/{scan_id}/{mri_type}"
        files = _sorted_dcm_files_fast(folder)
        _DICOM_FILES_CACHE[key] = files
    return files


def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    """
    Speed: minimize pydicom header parsing overhead.
    Correctness: same pixel_array/VOI LUT application and resizing/rotation as before.
    """
    global pydicom, apply_voi_lut
    if pydicom is None or apply_voi_lut is None:
        import pydicom as _pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut as _apply_voi_lut

        pydicom = _pydicom
        apply_voi_lut = _apply_voi_lut

    cache_key = (path, img_size, int(voi_lut), int(rotate))
    cached = _slice_cache_get(cache_key)
    if cached is not None:
        return cached

    dicom = pydicom.dcmread(path, stop_before_pixels=True, force=True)
    dicom.Pixels = pydicom.dcmread(
        path, stop_before_pixels=False, force=True
    ).PixelData  # ensures PixelData loaded
    data = dicom.pixel_array

    if voi_lut:
        data = apply_voi_lut(data, dicom)

    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])

    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA).astype(
        np.float32, copy=False
    )
    _slice_cache_put(cache_key, data)
    return data


def load_dicom_images_3d(
    scan_id,
    split,
    mri_type="FLAIR",
    num_imgs=NUM_IMAGES_PER_TYPE,
    img_size=IMAGE_SIZE,
    rotate=0,
):
    files = _get_sorted_dicom_files(scan_id, split, mri_type)

    if len(files) == 0:
        return np.zeros((1, img_size, img_size, num_imgs), dtype=np.float32)

    middle = len(files) // 2
    num_imgs2 = num_imgs // 2
    p1 = max(0, middle - num_imgs2)
    p2 = min(len(files), middle + num_imgs2)

    sel = files[p1:p2]
    n = len(sel)

    out = np.zeros((img_size, img_size, num_imgs), dtype=np.float32)
    start = (num_imgs - n) // 2
    for k, f in enumerate(sel):
        out[:, :, start + k] = load_dicom_image(f, img_size=img_size, rotate=rotate)

    mn = float(out.min())
    mx = float(out.max())
    if mn < mx:
        out = (out - mn) / (mx - mn)

    return out[np.newaxis, ...]


def _disk_cache_path(split, scan_id):
    return os.path.join(
        _DISK_CACHE_DIR,
        f"{split}_{scan_id}_96_{NUM_IMAGES_PER_TYPE}x{len(mri_types)}.npy",
    )


def load_dicom_images_3d_all(scan_id, split):
    npy_path = _disk_cache_path(split, scan_id)
    if os.path.exists(npy_path):
        return np.load(npy_path, allow_pickle=False)

    img3d_all = np.empty((1, IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES), dtype=np.float32)
    offset = 0
    for mt in mri_types:
        vol = load_dicom_images_3d(scan_id, split, mri_type=mt)  # (1,H,W,30)
        img3d_all[..., offset : offset + NUM_IMAGES_PER_TYPE] = vol
        offset += NUM_IMAGES_PER_TYPE

    try:
        np.save(npy_path, img3d_all, allow_pickle=False)
    except Exception:
        pass
    return img3d_all


a = load_dicom_images_3d_all("00000", "train")
print("Sanity load shape:", a.shape)
print("min/max/mean/median:", np.min(a), np.max(a), np.mean(a), np.median(a))



## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4219934932.py in <cell line: 0>()
    160 
    161 
--> 162 a = load_dicom_images_3d_all("00000", "train")
    163 print("Sanity load shape:", a.shape)
    164 print("min/max/mean/median:", np.min(a), np.max(a), np.mean(a), np.median(a))

/tmp/ipykernel_11/4219934932.py in load_dicom_images_3d_all(scan_id, split)
    149     offset = 0
    150     for mt in mri_types:
--> 151         vol = load_dicom_images_3d(scan_id, split, mri_type=mt)  # (1,H,W,30)
    152         img3d_all[..., offset : offset + NUM_IMAGES_PER_TYPE] = vol
    153         offset += NUM_IMAGES_PER_TYPE

/tmp/ipykernel_11/4219934932.py in load_dicom_images_3d(scan_id, split, mri_type, num_imgs, img_size, rotate)
    124     start = (num_imgs - n) // 2
    125     for k, f in enumerate(sel):
--> 126         out[:, :, start + k] = load_dicom_image(f, img_size=img_size, rotate=rotate)
    127 
    128     mn = float(out.min())

/tmp/ipykernel_11/4219934932.py in load_dicom_image(path, img_size, voi_lut, rotate)
     79         path, stop_before_pixels=False, force=True
     80     ).PixelData  # ensures PixelData loaded
---> 81     data = dicom.pixel_array
     82 
     83     if voi_lut:

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

## === cell 6
from tensorflow.keras.utils import Sequence


class Dataset(Sequence):
    """
    Kept for compatibility/debug, but training/inference will use tf.data for speed.
    """

    def __init__(self, df, split, is_train=True, batch_size=BATCH_SIZE, shuffle=True):
        self.idx = df["BraTS21ID"].values
        self.paths = df["BraTS21ID5"].values
        self.y = (
            df["MGMT_value"].values
            if ("MGMT_value" in df.columns and is_train)
            else None
        )
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.split = split

        self.on_epoch_end()

    def __len__(self):
        return math.ceil(len(self.idx) / self.batch_size)

    def __getitem__(self, i):
        batch_paths = self.paths[i * self.batch_size : (i + 1) * self.batch_size]

        list_x = [load_dicom_images_3d_all(x, self.split) for x in batch_paths]
        x = np.concatenate(list_x, axis=0)  # (B,H,W,D)
        x = x[..., np.newaxis]  # (B,H,W,D,1)

        if self.is_train:
            batch_y = self.y[i * self.batch_size : (i + 1) * self.batch_size].astype(
                np.float32
            )
            return x.astype(np.float32), batch_y
        else:
            return x.astype(np.float32)

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.paths))
            self.paths = self.paths[perm]
            self.idx = self.idx[perm]
            if self.y is not None:
                self.y = self.y[perm]




## === cell 7
df_train, df_valid = sk_model_selection.train_test_split(
    train_df,
    test_size=0.2,
    random_state=12,
    stratify=train_df["MGMT_value"],
)



## === cell 8
pass




## === cell 9
def _precompute_file_lists_and_volumes():
    train_ids = df_train["BraTS21ID5"].values.astype(str).tolist()
    valid_ids = df_valid["BraTS21ID5"].values.astype(str).tolist()
    test_ids = test["BraTS21ID5"].values.astype(str).tolist()

    for split, ids in (("train", train_ids + valid_ids), ("test", test_ids)):
        for sid in ids:
            for mt in mri_types:
                _get_sorted_dicom_files(sid, split, mt)

    jobs = []
    for sid in train_ids + valid_ids:
        if not os.path.exists(_disk_cache_path("train", sid)):
            jobs.append((sid, "train"))
    for sid in test_ids:
        if not os.path.exists(_disk_cache_path("test", sid)):
            jobs.append((sid, "test"))

    if not jobs:
        print("Volume cache: all required .npy files already present.")
        return

    print(f"Precomputing {len(jobs)} volumes into {_DISK_CACHE_DIR} (parallel)...")

    from concurrent.futures import ProcessPoolExecutor, as_completed

    def _worker(sid_split):
        sid, split = sid_split
        import os as _os
        import numpy as _np
        import cv2 as _cv2
        import pydicom as _pydicom
        from pydicom.pixel_data_handlers.util import apply_voi_lut as _apply_voi_lut
        import re as _re

        _RE_DICOM_NUM_LOCAL = _re.compile(r"(\d+)")
        _DISK_CACHE_DIR_LOCAL = _DISK_CACHE_DIR
        _IMAGE_SIZE = IMAGE_SIZE
        _NUM_IMAGES_PER_TYPE = NUM_IMAGES_PER_TYPE
        _NUM_IMAGES = NUM_IMAGES
        _mri_types = list(mri_types)
        _data_directory = data_directory

        def _disk_cache_path_local(split_, scan_id_):
            return _os.path.join(
                _DISK_CACHE_DIR_LOCAL,
                f"{split_}_{scan_id_}_96_{_NUM_IMAGES_PER_TYPE}x{len(_mri_types)}.npy",
            )

        def _sorted_dcm_files_fast_local(folder):
            try:
                entries = []
                with _os.scandir(folder) as it:
                    for e in it:
                        if e.is_file() and e.name.endswith(".dcm"):
                            m = _RE_DICOM_NUM_LOCAL.search(e.name)
                            idx = int(m.group(1)) if m else -1
                            entries.append((idx, e.name))
                entries.sort(key=lambda x: (x[0], x[1]))
                return [_os.path.join(folder, name) for _, name in entries]
            except FileNotFoundError:
                return []

        def _get_sorted_dicom_files_local(scan_id_, split_, mri_type_):
            folder = f"{_data_directory}/{split_}/{scan_id_}/{mri_type_}"
            return _sorted_dcm_files_fast_local(folder)

        def _load_dicom_image_local(path, img_size=_IMAGE_SIZE, voi_lut=True, rotate=0):
            dcm = _pydicom.dcmread(path, stop_before_pixels=True, force=True)
            dcm.Pixels = _pydicom.dcmread(
                path, stop_before_pixels=False, force=True
            ).PixelData
            arr = dcm.pixel_array
            if voi_lut:
                arr = _apply_voi_lut(arr, dcm)
            if rotate > 0:
                rot_choices = [
                    0,
                    _cv2.ROTATE_90_CLOCKWISE,
                    _cv2.ROTATE_90_COUNTERCLOCKWISE,
                    _cv2.ROTATE_180,
                ]
                arr = _cv2.rotate(arr, rot_choices[rotate])
            arr = _cv2.resize(
                arr, (img_size, img_size), interpolation=_cv2.INTER_AREA
            ).astype(_np.float32, copy=False)
            return arr

        def _load_dicom_images_3d_local(scan_id_, split_, mri_type_):
            files = _get_sorted_dicom_files_local(scan_id_, split_, mri_type_)
            if len(files) == 0:
                return _np.zeros(
                    (1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_IMAGES_PER_TYPE),
                    dtype=_np.float32,
                )

            middle = len(files) // 2
            num_imgs2 = _NUM_IMAGES_PER_TYPE // 2
            p1 = max(0, middle - num_imgs2)
            p2 = min(len(files), middle + num_imgs2)
            sel = files[p1:p2]
            n = len(sel)

            out = _np.zeros(
                (_IMAGE_SIZE, _IMAGE_SIZE, _NUM_IMAGES_PER_TYPE), dtype=_np.float32
            )
            start = (_NUM_IMAGES_PER_TYPE - n) // 2
            for k, f in enumerate(sel):
                out[:, :, start + k] = _load_dicom_image_local(f, img_size=_IMAGE_SIZE)

            mn = float(out.min())
            mx = float(out.max())
            if mn < mx:
                out = (out - mn) / (mx - mn)
            return out[_np.newaxis, ...]

        npy_path = _disk_cache_path_local(split, sid)
        if _os.path.exists(npy_path):
            return 1

        img3d_all = _np.empty(
            (1, _IMAGE_SIZE, _IMAGE_SIZE, _NUM_IMAGES), dtype=_np.float32
        )
        offset = 0
        for mt in _mri_types:
            vol = _load_dicom_images_3d_local(sid, split, mt)
            img3d_all[..., offset : offset + _NUM_IMAGES_PER_TYPE] = vol
            offset += _NUM_IMAGES_PER_TYPE

        _np.save(npy_path, img3d_all, allow_pickle=False)
        return 1

    max_workers = max(1, min(8, (os.cpu_count() or 4)))
    done = 0
    with ProcessPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_worker, job) for job in jobs]
        for i, f in enumerate(as_completed(futs), 1):
            _ = f.result()
            done += 1
            if done % 25 == 0 or done == len(jobs):
                print(f"  cached {done}/{len(jobs)}")


_precompute_file_lists_and_volumes()



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
_RemoteTraceback                          Traceback (most recent call last)
_RemoteTraceback: 
"""
Traceback (most recent call last):
  File "/usr/lib/python3.11/multiprocessing/queues.py", line 244, in _feed
    obj = _ForkingPickler.dumps(obj)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/lib/python3.11/multiprocessing/reduction.py", line 51, in dumps
    cls(buf, protocol).dump(obj)
AttributeError: Can't pickle local object '_precompute_file_lists_and_volumes.<locals>._worker'
"""

The above exception was the direct cause of the following exception:

AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3216521135.py in <cell line: 0>()
    147 
    148 
--> 149 _precompute_file_lists_and_volumes()
    150 

/tmp/ipykernel_11/3216521135.py in _precompute_file_lists_and_volumes()
    141         futs = [ex.submit(_worker, job) for job in jobs]
    142         for i, f in enumerate(as_completed(futs), 1):
--> 143             _ = f.result()
    144             done += 1
    145             if done % 25 == 0 or done == len(jobs):

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    447                     raise CancelledError()
    448                 elif self._state == FINISHED:
--> 449                     return self.__get_result()
    450 
    451                 self._condition.wait(timeout)

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/multiprocessing/queues.py in _feed(buffer, notempty, send_bytes, writelock, reader_close, writer_close, ignore_epipe, onerror, queue_sem)
    242 
    243                         # serialize the data before acquiring the lock
--> 244                         obj = _ForkingPickler.dumps(obj)
    245                         if wacquire is None:
    246                             send_bytes(obj)

/usr/lib/python3.11/multiprocessing/reduction.py in dumps(cls, obj, protocol)
     49     def dumps(cls, obj, protocol=None):
     50         buf = io.BytesIO()
---> 51         cls(buf, protocol).dump(obj)
     52         return buf.getbuffer()
     53 

AttributeError: Can't pickle local object '_precompute_file_lists_and_volumes.<locals>._worker'

## === cell 10
AUTOTUNE = tf.data.AUTOTUNE

_TF_PARALLEL_CALLS = max(2, min(8, (os.cpu_count() or 4)))


def _make_tf_dataset(df, split, is_train, batch_size, shuffle):
    paths = df["BraTS21ID5"].values.astype(str)
    if is_train:
        labels = df["MGMT_value"].values.astype(np.float32)
        ds = tf.data.Dataset.from_tensor_slices((paths, labels))
    else:
        ds = tf.data.Dataset.from_tensor_slices(paths)

    if shuffle and is_train:
        ds = ds.shuffle(buffer_size=len(paths), seed=12, reshuffle_each_iteration=True)

    def _py_load_from_npy(scan_id):
        sid = scan_id.numpy().decode("utf-8")
        npy_path = _disk_cache_path(split, sid)
        x = np.load(npy_path, allow_pickle=False).astype(
            np.float32, copy=False
        )  # (1,H,W,D)
        x = x[..., np.newaxis]  # (1,H,W,D,1)
        return x[0]  # (H,W,D,1)

    def _map_x(scan_id):
        x = tf.py_function(func=_py_load_from_npy, inp=[scan_id], Tout=tf.float32)
        x.set_shape((IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES, 1))
        return x

    if is_train:

        def _map_xy(scan_id, y):
            x = _map_x(scan_id)
            y = tf.cast(y, tf.float32)
            return x, y

        ds = ds.map(_map_xy, num_parallel_calls=_TF_PARALLEL_CALLS, deterministic=True)
    else:
        ds = ds.map(_map_x, num_parallel_calls=_TF_PARALLEL_CALLS, deterministic=True)

    ds = ds.batch(batch_size, drop_remainder=False).prefetch(AUTOTUNE)
    return ds


train_dataset = _make_tf_dataset(
    df_train, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=True
)
valid_dataset = _make_tf_dataset(
    df_valid, "train", is_train=True, batch_size=BATCH_SIZE, shuffle=False
)



## === cell 11
del train_df



## === cell 12
for images, label in train_dataset.take(1):
    print("Train batch shape:", images.shape, "labels shape:", label.shape)




## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/2764837364.py in <cell line: 0>()
----> 1 for images, label in train_dataset.take(1):
      2     print("Train batch shape:", images.shape, "labels shape:", label.shape)
      3 
      4 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_2_device_/job:localhost/replica:0/task:0/device:CPU:0}} FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/train_00133_96_30x4.npy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/3578883989.py", line 21, in _py_load_from_npy
    x = np.load(npy_path, allow_pickle=False).astype(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/train_00133_96_30x4.npy'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 13
def get_model(width=IMAGE_SIZE, height=IMAGE_SIZE, depth=NUM_IMAGES):
    """Build a 3D convolutional neural network model."""
    inputs = keras.Input((width, height, depth, 1))

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(inputs)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=64, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=128, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.Conv3D(filters=256, kernel_size=3, activation="relu")(x)
    x = layers.BatchNormalization()(x)

    x = layers.MaxPool3D(pool_size=2)(x)
    x = layers.Dropout(0.5)(x)

    x = layers.GlobalAveragePooling3D()(x)
    x = layers.Dense(units=128, activation="relu")(x)
    x = layers.Dropout(0.6)(x)

    outputs = layers.Dense(units=1, activation="sigmoid")(x)

    model = keras.Model(inputs, outputs, name="3D_CNN")
    return model




## === cell 14
model = get_model()
model.summary()



## === cell 15
model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-4),
    loss="binary_crossentropy",
    metrics=[AUC(name="auc")],
)



## === cell 16
ckpt_path = "best_model.weights.h5"
callbacks = [
    ModelCheckpoint(
        ckpt_path,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    ),
]

history = model.fit(
    train_dataset,
    validation_data=valid_dataset,
    epochs=2,
    callbacks=callbacks,
    verbose=1,
)



## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/3033912107.py in <cell line: 0>()
     11 ]
     12 
---> 13 history = model.fit(
     14     train_dataset,
     15     validation_data=valid_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/execute.py in quick_execute(op_name, num_outputs, inputs, attrs, ctx, name)
     57       e.message += " name: " + name
     58     raise core._status_to_exception(e) from None
---> 59   except TypeError as e:
     60     keras_symbolic_tensors = [x for x in inputs if _is_keras_symbolic_tensor(x)]
     61     if keras_symbolic_tensors:

UnknownError: Graph execution error:

Detected at node EagerPyFunc defined at (most recent call last):
<stack traces unavailable>
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/train_00036_96_30x4.npy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/3578883989.py", line 21, in _py_load_from_npy
    x = np.load(npy_path, allow_pickle=False).astype(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/train_00036_96_30x4.npy'


	 [[{{node EagerPyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_6385]

## === cell 17
if os.path.exists(ckpt_path):
    model.load_weights(ckpt_path)



## === cell 18
test_dataset = _make_tf_dataset(
    test, "test", is_train=False, batch_size=1, shuffle=False
)



## === cell 19
for images in test_dataset.take(1):
    print("Test batch shape:", images.shape)



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/4030464152.py in <cell line: 0>()
----> 1 for images in test_dataset.take(1):
      2     print("Test batch shape:", images.shape)
      3 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in __next__(self)
    824   def __next__(self):
    825     try:
--> 826       return self._next_internal()
    827     except errors.OutOfRangeError:
    828       raise StopIteration

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/iterator_ops.py in _next_internal(self)
    774     # to communicate that there is no more data to iterate over.
    775     with context.execution_mode(context.SYNC):
--> 776       ret = gen_dataset_ops.iterator_get_next(
    777           self._iterator_resource,
    778           output_types=self._flat_output_types,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/gen_dataset_ops.py in iterator_get_next(iterator, output_types, output_shapes, name)
   3084       return _result
   3085     except _core._NotOkStatusException as e:
-> 3086       _ops.raise_from_not_ok_status(e, name)
   3087     except _core._FallbackException:
   3088       pass

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/test_00356_96_30x4.npy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/3578883989.py", line 21, in _py_load_from_npy
    x = np.load(npy_path, allow_pickle=False).astype(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/test_00356_96_30x4.npy'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 20
predictions = model.predict(test_dataset, verbose=1).reshape(-1)
predictions = np.clip(predictions, 0.0, 1.0)

print(
    "Predictions:",
    predictions[:5],
    "n=",
    len(predictions),
    "submission_n=",
    len(sample_submission),
)



## --- ERROR in cell 20, traceback:
---------------------------------------------------------------------------
UnknownError                              Traceback (most recent call last)
/tmp/ipykernel_11/3420703564.py in <cell line: 0>()
----> 1 predictions = model.predict(test_dataset, verbose=1).reshape(-1)
      2 predictions = np.clip(predictions, 0.0, 1.0)
      3 
      4 print(
      5     "Predictions:",

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/ops.py in raise_from_not_ok_status(e, name)
   6000 def raise_from_not_ok_status(e, name) -> NoReturn:
   6001   e.message += (" name: " + str(name if name is not None else ""))
-> 6002   raise core._status_to_exception(e) from None  # pylint: disable=protected-access
   6003 
   6004 

UnknownError: {{function_node __wrapped__IteratorGetNext_output_types_1_device_/job:localhost/replica:0/task:0/device:CPU:0}} FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/test_00356_96_30x4.npy'
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 267, in __call__
    return func(device, token, args)
           ^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 145, in __call__
    outputs = self._call(device, args)
              ^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 152, in _call
    ret = self._func(*args)
          ^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/3578883989.py", line 21, in _py_load_from_npy
    x = np.load(npy_path, allow_pickle=False).astype(
        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/numpy/lib/npyio.py", line 427, in load
    fid = stack.enter_context(open(os_fspath(file), "rb"))
                              ^^^^^^^^^^^^^^^^^^^^^^^^^^^

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_npy/test_00356_96_30x4.npy'


	 [[{{node EagerPyFunc}}]] [Op:IteratorGetNext] name: 

## === cell 21
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": predictions}
)

submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)
submission["MGMT_value"] = submission["MGMT_value"].astype(float)

submission.head()



## --- ERROR in cell 21, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/649597566.py in <cell line: 0>()
      1 submission = pd.DataFrame(
----> 2     {"BraTS21ID": sample_submission["BraTS21ID"].values, "MGMT_value": predictions}
      3 )
      4 
      5 submission["BraTS21ID"] = submission["BraTS21ID"].astype(int)

NameError: name 'predictions' is not defined

## === cell 22
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)
print(submission.head(3))

## --- ERROR in cell 22, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1435976667.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote submission.csv with shape:", submission.shape)
      3 print(submission.dtypes)
      4 print(submission.head(3))

NameError: name 'submission' is not defined
