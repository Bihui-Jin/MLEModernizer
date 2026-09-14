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

3.11

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

# 5. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import glob
import random
from enum import Enum
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
import hashlib
import time

import numpy as np
import pandas as pd

import cv2
import pydicom

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    BatchNormalization,
    MaxPool3D,
    Dense,
    Dropout,
    concatenate,
    GlobalAveragePooling3D,
    ReLU,
)

try:
    cv2.setNumThreads(0)
except Exception:
    pass



## === cell 1
SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

_intra = os.environ.get("TF_INTRA_OP_THREADS", "").strip()
_inter = os.environ.get("TF_INTER_OP_THREADS", "").strip()
try:
    if _intra:
        tf.config.threading.set_intra_op_parallelism_threads(int(_intra))
    if _inter:
        tf.config.threading.set_inter_op_parallelism_threads(int(_inter))
except Exception:
    pass




## === cell 2
class InternalDebug:
    def __init__(self, debug_mode=False, debug_prefix=None):
        self.__debug_mode = debug_mode
        self.__debug_prefix = debug_prefix

    @property
    def __prefix(self):
        if self.__debug_prefix is not None:
            return self.__debug_prefix
        else:
            return ""

    def log(self, *args):
        if self.__debug_mode:
            if self.__debug_prefix is not None:
                print(self.__debug_prefix, *args)
            else:
                print(*args)

    def separator(self, character="=", length=50):
        separator = character * length
        self.log(separator)

    def info(self, *args):
        if self.__debug_mode:
            prefix = self.__prefix + "[INFO]"
            self.log(prefix, *args)

    def warning(self, *args):
        if self.__debug_mode:
            prefix = self.__prefix + "[WARNING]"
            self.log(prefix, *args)

    def error(self, *args):
        if self.__debug_mode:
            prefix = self.__prefix + "[ERROR]"
            self.log(prefix, *args)

    def set_debug_mode(self, debug_mode):
        self.__debug_mode = debug_mode




## === cell 3
class ImageFormat(Enum):
    WHDC = "W-H-D-C"  # Format (W, H, D, C)
    DWHC = "D-W-H-C"  # Format (D, W, H, C)

    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.DWHC:
            image = np.transpose(image, (2, 1, 0, 3))
        elif image_format == ImageFormat.WHDC:
            image = np.transpose(image, (2, 1, 0, 3))
        return image


class DICOMLoader:
    def __init__(
        self,
        df,
        input_path,
        scan_categories,
        num_imgs=None,
        size=(224, 224),
        scale=1.0,
        rotate_angle=0,
        enable_center_focus=False,
        id_column_name="ID",
        label_column_name="Label",
        image_format=ImageFormat.WHDC,
        max_threads=8,
        image_file_sorter=lambda x: int(x[:-4].split("-")[-1]),
        debug_mode=False,
        require_labels=True,
        cache_dir=None,
        ram_cache_size=512,
    ):
        if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
            raise ValueError("num_imgs must be divisible by 2 for central image")

        if not (0 <= rotate_angle <= 360):
            raise ValueError("Rotation value must be between 0 and 360")

        if id_column_name not in df.columns:
            raise ValueError(f"Column {id_column_name} must be in dataset")
        if require_labels and (label_column_name not in df.columns):
            raise ValueError(f"Column {label_column_name} must be in dataset")

        self.__df = df.copy()
        self.__num_imgs = num_imgs
        self.__id_column_name = id_column_name
        self.__label_column_name = label_column_name
        self.__input_path = input_path
        self.__scan_categories = list(scan_categories)
        self.__max_threads = max_threads
        self.__size = size
        self.__scale = scale
        self.__rotate_angle = rotate_angle
        self.__image_format = image_format
        self.__image_file_sorter = image_file_sorter
        self.__enable_center_focus = enable_center_focus
        self.__debug = InternalDebug(debug_mode=debug_mode)

        self.__cache_dir = cache_dir
        if self.__cache_dir is not None:
            os.makedirs(self.__cache_dir, exist_ok=True)

        self.__ram_cache_size = int(ram_cache_size)
        self.__get_volume_cached = lru_cache(maxsize=self.__ram_cache_size)(
            self.__get_volume_uncached
        )

        self.__executor = ThreadPoolExecutor(max_workers=self.__max_threads)

        self.__file_index = {}
        for sc in self.__scan_categories:
            for r in range(len(self.__df)):
                pid = str(self.__df.loc[r, self.__id_column_name]).zfill(5)
                self.__file_index[(pid, sc)] = self._get_image_files_cached(pid, sc)

    def __del__(self):
        try:
            self.__executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    @property
    def scan_categories(self):
        return self.__scan_categories

    @property
    def num_imgs(self):
        return self.__num_imgs

    @property
    def image_format(self):
        return self.__image_format

    @property
    def df(self):
        return self.__df.copy()

    @property
    def len(self):
        return len(self.__df)

    def get_id(self, row):
        return self.__df.loc[row, self.__id_column_name]

    def gel_label(self, row):
        return self.__df.loc[row, self.__label_column_name]

    def format(self, images, type):
        if type == "normalize" and self.image_format == ImageFormat.WHDC:
            return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)
        elif type == "default" and self.image_format == ImageFormat.WHDC:
            return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
        else:
            return images

    @lru_cache(maxsize=8192)
    def _get_image_files_cached(self, patient_id: str, scan_category: str):
        scans_path = os.path.join(self.__input_path, patient_id, scan_category)
        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        image_files = sorted(
            glob.glob(os.path.join(scans_path, "*")),
            key=self.__image_file_sorter,
        )
        if not image_files:
            raise ValueError(f"No image files found in {scans_path}.")
        image_files = self._select_subset_image_files(image_files)
        return tuple(image_files)

    def _cache_key(self, patient_id: str, scan_category: str) -> str:
        params = (
            patient_id,
            scan_category,
            str(self.__size),
            str(self.__scale),
            str(self.__rotate_angle),
            str(self.__num_imgs),
            str(self.__enable_center_focus),
            str(self.__image_format.value),
            os.path.abspath(self.__input_path),
        )
        h = hashlib.md5("|".join(params).encode("utf-8")).hexdigest()
        return h

    def __get_volume_uncached(self, patient_id: str, scan_category: str):
        image_files = list(self.__file_index[(patient_id, scan_category)])

        target_len = (
            self.__num_imgs if self.__num_imgs is not None else len(image_files)
        )
        w, h = self.__size
        vol = np.zeros((target_len, h, w, 1), dtype=np.uint8)

        for idx, image_data in enumerate(
            self.__executor.map(self._load_dicom_image, image_files[:target_len])
        ):
            if image_data is None:
                continue
            vol[idx] = image_data  # already (H,W,1) uint8

        return self.format(vol, "default")

    def load_scan(self, row, scan_category, show_progress=True):
        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)

        if self.__cache_dir is not None:
            key = self._cache_key(patient_id, scan_category)
            cache_path = os.path.join(self.__cache_dir, f"{key}.npy")
            if os.path.exists(cache_path):
                return np.load(cache_path, mmap_mode="r")

            arr = self.__get_volume_cached(patient_id, scan_category)

            tmp_base = cache_path + f".tmp_{os.getpid()}_{time.time_ns()}"
            tmp_npy = tmp_base if tmp_base.endswith(".npy") else (tmp_base + ".npy")
            np.save(tmp_base, arr)
            try:
                os.replace(tmp_npy, cache_path)
            except Exception:
                try:
                    if os.path.exists(tmp_npy):
                        os.remove(tmp_npy)
                except Exception:
                    pass

            return np.load(cache_path, mmap_mode="r")

        return self.__get_volume_cached(patient_id, scan_category)

    def load_all_scans(self, row, show_progress=True):
        out = {}
        for scan_category in self.__scan_categories:
            out[scan_category] = self.load_scan(row, scan_category, False)
        return out

    def _load_dicom_image(self, dicom_path):
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File {dicom_path} does not exist.")

        try:
            dicom_file = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=False,
                force=True,
                specific_tags=[
                    "PixelData",
                    "Rows",
                    "Columns",
                    "BitsAllocated",
                    "BitsStored",
                    "HighBit",
                    "PixelRepresentation",
                    "RescaleIntercept",
                    "RescaleSlope",
                    "PhotometricInterpretation",
                    "SamplesPerPixel",
                    "PlanarConfiguration",
                    "NumberOfFrames",
                    "TransferSyntaxUID",
                ],
            )
        except Exception as e:
            raise IOError(f"An error occurred while reading the DICOM file: {e}")

        image = None

        try:
            rows = int(getattr(dicom_file, "Rows", 0) or 0)
            cols = int(getattr(dicom_file, "Columns", 0) or 0)
            bits_alloc = int(getattr(dicom_file, "BitsAllocated", 0) or 0)
            spp = int(getattr(dicom_file, "SamplesPerPixel", 1) or 1)
            nframes = int(getattr(dicom_file, "NumberOfFrames", 1) or 1)
            pix = getattr(dicom_file, "PixelData", None)

            if (
                pix is not None
                and rows > 0
                and cols > 0
                and spp == 1
                and nframes == 1
                and bits_alloc in (16, 8)
            ):
                dtype = np.uint16 if bits_alloc == 16 else np.uint8
                arr = np.frombuffer(pix, dtype=dtype)
                need = rows * cols
                if arr.size >= need:
                    image = arr[:need].reshape(rows, cols)
        except Exception:
            image = None

        if image is None:
            try:
                image = dicom_file.pixel_array
            except Exception:
                try:
                    rows = int(getattr(dicom_file, "Rows", 0) or 0)
                    cols = int(getattr(dicom_file, "Columns", 0) or 0)
                    pix = getattr(dicom_file, "PixelData", None)
                    if rows > 0 and cols > 0 and pix is not None:
                        arr = np.frombuffer(pix, dtype=np.uint16)
                        if arr.size >= rows * cols:
                            image = arr[: rows * cols].reshape(rows, cols)
                        else:
                            image = np.zeros((rows, cols), dtype=np.uint16)
                    else:
                        w, h = self.__size
                        return np.zeros((h, w, 1), dtype=np.uint8)
                except Exception:
                    w, h = self.__size
                    return np.zeros((h, w, 1), dtype=np.uint8)

        image = self._rotate_img(image)
        image = self._normalization_img(image)
        image = self._crop_img(image)
        image = self._resize_img(image)
        image = np.expand_dims(image, axis=-1)
        return image

    def _resize_img(self, image):
        w, h = self.__size
        return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

    def _crop_img(self, image):
        if self.__scale <= 0:
            return image
        center_x, center_y = image.shape[1] / 2, image.shape[0] / 2
        width_scaled, height_scaled = (
            image.shape[1] * self.__scale,
            image.shape[0] * self.__scale,
        )
        left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
        top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2
        return image[int(top_y) : int(bottom_y), int(left_x) : int(right_x)]

    def _rotate_img(self, image):
        if self.__rotate_angle <= 0:
            return image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def _normalization_img(self, image):
        img = image.astype(np.float32, copy=False)
        min_val = float(np.min(img))
        max_val = float(np.max(img))
        if max_val == 0:
            return np.zeros_like(image, dtype=np.uint8)
        img = img - min_val
        img = img / max_val
        img = img * 255.0
        return img.astype(np.uint8)

    def _select_subset_image_files(self, image_files):
        if self.__enable_center_focus and self.__num_imgs is not None:
            middle = len(image_files) // 2
            num_imgs2 = self.__num_imgs // 2
            p1 = max(0, middle - num_imgs2)
            p2 = min(len(image_files), middle + num_imgs2)
            return image_files[p1:p2]
        elif self.__num_imgs is not None:
            return image_files[: self.__num_imgs]
        else:
            return image_files




## === cell 4
class ScanDataset(Sequence):
    def __init__(
        self,
        dicom_loader,
        batch_size,
        subset="train",
        shuffle=True,
        debug_mode=False,
    ):
        self.__dicom_loader = dicom_loader
        self.__batch_size = batch_size

        self.__is_trainable = subset.lower() in ["validation", "train", "training"]

        self.__shuffle = shuffle
        self.__debug = InternalDebug(debug_mode=debug_mode)
        self.__indices = np.arange(self.__dicom_loader.len)

        self.__n_modalities = len(self.__dicom_loader.scan_categories)

        self.__vol_shape = None
        if self.__dicom_loader.len > 0:
            sc0 = self.__dicom_loader.scan_categories[0]
            v0 = self.__dicom_loader.load_scan(0, sc0, show_progress=False)
            self.__vol_shape = tuple(v0.shape)

        self.__executor = ThreadPoolExecutor(
            max_workers=min(8, self.__n_modalities * 2)
        )

        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __del__(self):
        try:
            self.__executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bs = len(batch_indices)

        if self.__vol_shape is None:
            raise RuntimeError("Failed to infer volume shape.")

        batch_x = [
            np.empty((bs, *self.__vol_shape), dtype=np.uint8)
            for _ in range(self.__n_modalities)
        ]
        if self.__is_trainable:
            batch_y = np.empty((bs,), dtype=np.float32)

        scan_categories = self.__dicom_loader.scan_categories

        tasks = []
        for k, i in enumerate(batch_indices):
            if self.__is_trainable:
                batch_y[k] = self.__dicom_loader.gel_label(i)
            for j, sc in enumerate(scan_categories):
                tasks.append((k, j, i, sc))

        def _load_task(t):
            k, j, i, sc = t
            arr = self.__dicom_loader.load_scan(i, sc, False)
            return k, j, arr

        for k, j, arr in self.__executor.map(
            _load_task,
            tasks,
            chunksize=max(1, len(tasks) // (self.__executor._max_workers * 4) or 1),
        ):
            batch_x[j][k] = arr

        batch_x = tuple(batch_x)
        if self.__is_trainable:
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))




## === cell 5
class MRIType(Enum):
    FLAIR = "FLAIR"
    T1w = "T1w"
    T1wCE = "T1wCE"
    T2w = "T2w"


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"




## === cell 6
VERSION = "V1"
SCAN_CATEGORIES = [mri_type.value for mri_type in MRIType]
EXCLUDED_IDS = [109, 123, 709]

RUN_DIR = "./run"
INPUT_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_DATASET_PATH = INPUT_PATH + "/train"
TRAIN_DATASET_DF_DIR = INPUT_PATH + "/train_labels.csv"

TEST_DATASET_PATH = INPUT_PATH + "/test"
TEST_DATASET_DF_DIR = INPUT_PATH + "/sample_submission.csv"

SUBMISSION_DATASET_DF_DIR = "/kaggle/working/submission.csv"

BEST_MODEL_PATH = f"{RUN_DIR}/models"
BEST_MODEL_H5_DIR = f"{BEST_MODEL_PATH}/model_{VERSION}.h5"

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 1
IMG_ROTATE = 0
IMG_ENABLE_CENTRAL_FOCUS = True

INPUT_SHAPE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN)

BATCH_SIZE = 8
EPOCHS = 26

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)
COMPILE_LOSS = "binary_crossentropy"
COMPILE_METRICS = [AUC(name="auc")]

SEQUENCE_WORKERS = int(os.environ.get("SEQUENCE_WORKERS", "4"))
SEQUENCE_MAX_QUEUE_SIZE = int(os.environ.get("SEQUENCE_MAX_QUEUE_SIZE", "8"))

CACHE_DIR = os.path.join("/kaggle/working", "dicom_cache_v1")



## === cell 7
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)




## === cell 8
class DeepScanModel(Model):
    def __init__(self, input_shape, model_name="My3DCNNModel"):
        self.input_layers = [Input(shape=input_shape) for _ in range(4)]
        self.cnn_models = [
            self.build_cnn_branch(input_layer) for input_layer in self.input_layers
        ]
        concatenated = concatenate(self.cnn_models)
        x = self.build_head(concatenated)
        super(DeepScanModel, self).__init__(
            inputs=self.input_layers, outputs=x, name=model_name
        )

    def build_cnn_branch(self, input_layer):
        x = Conv3D(64, 3)(input_layer)
        x = ReLU()(x)
        x = MaxPool3D(2)(x)
        x = BatchNormalization()(x)

        x = Conv3D(128, 3)(x)
        x = ReLU()(x)
        x = MaxPool3D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.1)(x)

        x = Conv3D(256, 3)(x)
        x = ReLU()(x)
        x = MaxPool3D(2)(x)
        x = BatchNormalization()(x)
        x = Dropout(0.2)(x)
        return x

    def build_head(self, x):
        x = GlobalAveragePooling3D()(x)
        x = Dense(1024)(x)
        x = ReLU()(x)
        x = Dropout(0.3)(x)
        x = Dense(1, activation="sigmoid")(x)
        return x




## === cell 9
perm = np.random.RandomState(SEED).permutation(len(train_df))
val_size = max(1, int(0.2 * len(train_df)))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_split_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

train_dicom_loader = DICOMLoader(
    train_split_df,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    require_labels=True,
    cache_dir=os.path.join(CACHE_DIR, "train"),
    ram_cache_size=512,
)

val_dicom_loader = DICOMLoader(
    val_split_df,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    require_labels=True,
    cache_dir=os.path.join(CACHE_DIR, "val"),
    ram_cache_size=256,
)

train_dataset = ScanDataset(
    dicom_loader=train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=True,
    debug_mode=False,
)

val_dataset = ScanDataset(
    dicom_loader=val_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.VALIDATION.value,
    shuffle=False,
    debug_mode=False,
)

test_dicom_loader = DICOMLoader(
    test_df[["ID"]].copy(),
    input_path=TEST_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    require_labels=False,
    cache_dir=os.path.join(CACHE_DIR, "test"),
    ram_cache_size=128,
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)



## === cell 10
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

model = DeepScanModel(input_shape=INPUT_SHAPE, model_name="Mult3DCNN4Input")
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

_monitor = "val_auc"
checkpoint_cb = ModelCheckpoint(
    filepath=BEST_MODEL_H5_DIR,
    monitor=_monitor,
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=0,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb],
    verbose=2,
)

if not os.path.exists(BEST_MODEL_H5_DIR):
    try:
        model.save(BEST_MODEL_H5_DIR, include_optimizer=False)
    except Exception:
        pass

if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )




## === cell 11
def generate_predictions(model, test_dataset, test_df):
    preds = model.predict(
        test_dataset,
        verbose=0,
    ).reshape(-1)

    preds = preds[: len(test_df)]

    submission = test_df.copy()
    submission["Label"] = preds.astype(np.float32)

    submission["ID"] = submission["ID"].astype(int).astype(str).str.zfill(5)

    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)
    submission = submission[["BraTS21ID", "MGMT_value"]]
    return submission


submission = generate_predictions(model, test_dataset, test_df)



## === cell 12
submission.info()
print(submission.head())



## === cell 13
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote:", os.path.abspath(SUBMISSION_DATASET_DF_DIR))
print("Rows:", len(submission))
print("Columns:", submission.columns.tolist())
