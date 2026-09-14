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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

os.environ.setdefault("TF_USE_LEGACY_KERAS", "1")
os.environ["PYTHONHASHSEED"] = "0"

import glob
import random
import numpy as np
import pandas as pd

import pydicom
import cv2

from enum import Enum

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    BatchNormalization,
    MaxPool3D,
    GlobalAveragePooling3D,
    Dense,
    Dropout,
    ReLU,
    concatenate,
)

plot_model = None

SEED = 123
tf.keras.utils.set_random_seed(SEED)
random.seed(SEED)
np.random.seed(SEED)

tf.get_logger().setLevel("ERROR")
try:
    tf.config.optimizer.set_jit(True)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)  # let TF decide
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    pydicom.config.pixel_data_handlers = pydicom.config.pixel_data_handlers  # no-op
except Exception:
    pass




## === cell 1
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




## === cell 2
from concurrent.futures import ThreadPoolExecutor
from collections import OrderedDict

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


class ImageFormat(Enum):
    """
    Enum to represent image formats.
    - WHDC = Width-Height-Depth-Channel
    - DWHC = Depth-Width-Height-Channel
    """

    WHDC = "W-H-D-C"
    DWHC = "D-W-H-C"

    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.DWHC:
            image = np.transpose(image, (2, 1, 0, 3))
        elif image_format == ImageFormat.WHDC:
            image = np.transpose(image, (2, 1, 0, 3))
        return image


_SHARED_DICOM_EXECUTOR = None


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
        enable_cache=True,
        cache_max_items=512,
        disk_cache_dir=None,
        enable_disk_cache=True,
        patient_disk_cache_dir=None,
        enable_patient_disk_cache=True,
    ):
        if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
            raise ValueError("num_imgs must be divisible by 2 for central image")

        if not (0 <= rotate_angle <= 360):
            raise ValueError("Rotation value must be between 0 and 360")

        for col in [id_column_name, label_column_name]:
            if col not in df.columns:
                raise ValueError(f"Columns {col} must be in dataset")

        self.__df = df.copy()
        self.__num_imgs = num_imgs
        self.__id_column_name = id_column_name
        self.__label_column_name = label_column_name
        self.__input_path = input_path
        self.__scan_categories = scan_categories
        self.__max_threads = max_threads
        self.__size = size
        self.__scale = scale
        self.__rotate_angle = rotate_angle
        self.__image_format = image_format
        self.__image_file_sorter = image_file_sorter
        self.__enable_center_focus = enable_center_focus
        self.__debug = InternalDebug(debug_mode=debug_mode)

        self.__ids_arr = self.__df[self.__id_column_name].to_numpy(copy=True)
        self.__labels_arr = self.__df[self.__label_column_name].to_numpy(copy=True)

        self.__file_index = self._build_file_index()

        global _SHARED_DICOM_EXECUTOR
        if _SHARED_DICOM_EXECUTOR is None:
            _SHARED_DICOM_EXECUTOR = ThreadPoolExecutor(max_workers=self.__max_threads)
        self.__executor = _SHARED_DICOM_EXECUTOR

        self.__enable_cache = bool(enable_cache)
        self.__cache_max_items = (
            int(cache_max_items) if cache_max_items is not None else 0
        )
        self.__scan_cache = (
            OrderedDict()
        )  # (patient_id_str, scan_category) -> np.ndarray

        self.__enable_disk_cache = bool(enable_disk_cache)
        self.__disk_cache_dir = disk_cache_dir
        if self.__enable_disk_cache and self.__disk_cache_dir:
            os.makedirs(self.__disk_cache_dir, exist_ok=True)

        self.__enable_patient_disk_cache = bool(enable_patient_disk_cache)
        self.__patient_disk_cache_dir = patient_disk_cache_dir
        if self.__enable_patient_disk_cache and self.__patient_disk_cache_dir:
            os.makedirs(self.__patient_disk_cache_dir, exist_ok=True)

        self.__pydicom_use_v2 = False
        try:
            if hasattr(pydicom, "pixel_array_options"):
                pydicom.pixel_array_options(use_v2_backend=True)
                self.__pydicom_use_v2 = True
        except Exception:
            self.__pydicom_use_v2 = False

    def __del__(self):
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
        return self.__ids_arr[row]

    def gel_label(self, row):
        return self.__labels_arr[row]

    def format(self, images, type):
        if type == "normalize" and self.image_format == ImageFormat.WHDC:
            return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)
        elif type == "default" and self.image_format == ImageFormat.WHDC:
            return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
        else:
            return images

    def _scandir_files(self, scans_path):
        try:
            with os.scandir(scans_path) as it:
                return [entry.path for entry in it if entry.is_file()]
        except FileNotFoundError:
            return []

    def _build_file_index(self):
        file_index = {}
        ids_unique = np.unique(self.__ids_arr)
        for pid in ids_unique:
            patient_id = str(int(pid)).zfill(5)
            file_index[patient_id] = {}
            base_patient_path = os.path.join(self.__input_path, patient_id)
            for scan_category in self.__scan_categories:
                scans_path = os.path.join(base_patient_path, scan_category)
                files = self._scandir_files(scans_path)
                if files:
                    files = sorted(files, key=self.__image_file_sorter)
                file_index[patient_id][scan_category] = files
        return file_index

    def _cache_get(self, cache_key):
        if not self.__enable_cache:
            return None
        try:
            val = self.__scan_cache.pop(cache_key)
            self.__scan_cache[cache_key] = val
            return val
        except KeyError:
            return None

    def _cache_put(self, cache_key, val):
        if not self.__enable_cache:
            return
        self.__scan_cache[cache_key] = val
        if self.__cache_max_items > 0:
            while len(self.__scan_cache) > self.__cache_max_items:
                self.__scan_cache.popitem(last=False)

    def _disk_cache_path(self, patient_id, scan_category):
        if not (self.__enable_disk_cache and self.__disk_cache_dir):
            return None
        fn = (
            f"{patient_id}_{scan_category}"
            f"_n{self.__num_imgs}"
            f"_s{self.__size[0]}x{self.__size[1]}"
            f"_sc{self.__scale}"
            f"_rot{self.__rotate_angle}"
            f"_cf{int(bool(self.__enable_center_focus))}"
            f".npy"
        )
        return os.path.join(self.__disk_cache_dir, fn)

    def _disk_cache_get(self, patient_id, scan_category):
        p = self._disk_cache_path(patient_id, scan_category)
        if p is None or (not os.path.exists(p)):
            return None
        try:
            return np.load(p, allow_pickle=False, mmap_mode=None)
        except Exception:
            return None

    def _disk_cache_put(self, patient_id, scan_category, arr):
        p = self._disk_cache_path(patient_id, scan_category)
        if p is None:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp_base = p + ".tmp"
        tmp_actual = tmp_base + ".npy"
        np.save(tmp_base, arr, allow_pickle=False)
        os.replace(tmp_actual, p)

    def _patient_cache_path(self, patient_id):
        if not (self.__enable_patient_disk_cache and self.__patient_disk_cache_dir):
            return None
        fn = (
            f"{patient_id}"
            f"_n{self.__num_imgs}"
            f"_s{self.__size[0]}x{self.__size[1]}"
            f"_sc{self.__scale}"
            f"_rot{self.__rotate_angle}"
            f"_cf{int(bool(self.__enable_center_focus))}"
            f".npz"
        )
        return os.path.join(self.__patient_disk_cache_dir, fn)

    def _patient_cache_get(self, patient_id):
        p = self._patient_cache_path(patient_id)
        if p is None or (not os.path.exists(p)):
            return None
        try:
            with np.load(p, allow_pickle=False) as z:
                out = {k: z[k] for k in self.__scan_categories if k in z.files}
            if len(out) != len(self.__scan_categories):
                return None
            return out
        except Exception:
            return None

    def _patient_cache_put(self, patient_id, all_images_dict):
        p = self._patient_cache_path(patient_id)
        if p is None:
            return
        os.makedirs(os.path.dirname(p), exist_ok=True)
        tmp_path = p + ".tmp.npz"
        np.savez_compressed(tmp_path, **all_images_dict)
        os.replace(tmp_path, p)

    def load_scan(self, row, scan_category, show_progress=True):
        patient_id = str(int(self.__ids_arr[row])).zfill(5)

        cache_key = (patient_id, scan_category)
        cached = self._cache_get(cache_key)
        if cached is not None:
            return cached

        disk_cached = self._disk_cache_get(patient_id, scan_category)
        if disk_cached is not None:
            self._cache_put(cache_key, disk_cached)
            return disk_cached

        image_files = self.__file_index.get(patient_id, {}).get(scan_category, [])
        if not image_files:
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)
        loaded_images = []

        image_data_iterable = self.__executor.map(self._load_dicom_image, image_files)
        if show_progress:
            image_data_iterable = tqdm(
                image_data_iterable, total=len(image_files), desc="Loading images"
            )

        for image_data in image_data_iterable:
            if image_data is not None:
                loaded_images.append(image_data)

        if not loaded_images:
            raise ValueError(
                "No images were loaded, and num_imgs is set. Cannot proceed."
            )

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                zero_image = np.zeros_like(loaded_images[0])
                loaded_images.append(zero_image)

        loaded_images = np.array(loaded_images)
        out = self.format(loaded_images, "default")

        self._disk_cache_put(patient_id, scan_category, out)
        self._cache_put(cache_key, out)
        return out

    def load_all_scans(self, row, show_progress=True):
        patient_id = str(int(self.__ids_arr[row])).zfill(5)

        patient_cached = self._patient_cache_get(patient_id)
        if patient_cached is not None:
            return {key: patient_cached.get(key, []) for key in self.__scan_categories}

        futures = {}
        for scan_category in self.__scan_categories:
            futures[scan_category] = self.__executor.submit(
                self.load_scan, row, scan_category, False
            )
        all_images = {k: f.result() for k, f in futures.items()}
        out = {key: all_images.get(key, []) for key in self.__scan_categories}

        self._patient_cache_put(patient_id, out)
        return out

    def _load_dicom_image(self, dicom_path):
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File {dicom_path} does not exist.")

        try:
            dicom_file = pydicom.dcmread(
                dicom_path,
                force=False,
                stop_before_pixels=False,
                specific_tags=[
                    "PixelData",
                    "Rows",
                    "Columns",
                    "BitsAllocated",
                    "BitsStored",
                    "HighBit",
                    "PixelRepresentation",
                    "PhotometricInterpretation",
                    "SamplesPerPixel",
                    "PlanarConfiguration",
                    "RescaleIntercept",
                    "RescaleSlope",
                ],
            )
        except Exception as e:
            raise IOError(f"An error occurred while reading the DICOM file: {e}")

        image = dicom_file.pixel_array
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
        min_val, max_val, _, _ = cv2.minMaxLoc(image)
        if max_val == 0:
            return np.zeros_like(image).astype(np.uint8)
        img = image.astype(np.float32, copy=False)
        img = img - float(min_val)
        img = img / float(max_val)
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




## === cell 3
class ScanDataset(Sequence):
    def __init__(
        self, dicom_loader, batch_size, subset="train", shuffle=True, debug_mode=False
    ):
        self.__dicom_loader = dicom_loader
        self.__batch_size = batch_size
        self.__is_trainable = subset.lower() in ["validation", "train"]
        self.__shuffle = shuffle
        self.__debug = InternalDebug(debug_mode=debug_mode)
        self.__indices = np.arange(self.__dicom_loader.len)

        df_ = self.__dicom_loader.df
        self.__labels_arr = df_["Label"].to_numpy(dtype=np.float32, copy=True)

        if self.__shuffle:
            np.random.shuffle(self.__indices)

        self.__scan_cats = list(self.__dicom_loader.scan_categories)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bsz = len(batch_indices)

        scan_cats = self.__scan_cats

        first_scans = self.__dicom_loader.load_all_scans(
            int(batch_indices[0]), show_progress=False
        )

        batches_x = []
        for sc in scan_cats:
            arr0 = first_scans[sc]
            bx = np.empty((bsz,) + arr0.shape, dtype=np.uint8)
            bx[0] = arr0
            batches_x.append(bx)

        if self.__is_trainable:
            batch_y = np.empty((bsz,), dtype=np.float32)
            batch_y[0] = self.__labels_arr[int(batch_indices[0])]

        for bi in range(1, bsz):
            i = int(batch_indices[bi])
            if self.__is_trainable:
                batch_y[bi] = self.__labels_arr[i]
            scans = self.__dicom_loader.load_all_scans(i, show_progress=False)
            for j, sc in enumerate(scan_cats):
                batches_x[j][bi] = scans[sc]

        batch_x = tuple(batches_x)
        if self.__is_trainable:
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))


def sequence_to_tfdata(seq, is_trainable, input_shape, batch_size, num_modalities=4):
    output_signature_x = tuple(
        tf.TensorSpec(shape=(None,) + tuple(input_shape), dtype=tf.uint8)
        for _ in range(num_modalities)
    )
    if is_trainable:
        output_signature = (
            output_signature_x,
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
        )
    else:
        output_signature = output_signature_x

    def gen():
        for i in range(len(seq)):
            yield seq[i]

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds




## === cell 4
class MRIType(Enum):
    FLAIR = "FLAIR"
    T1w = "T1w"
    T1wCE = "T1wCE"
    T2w = "T2w"


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"




## === cell 5
VERSION = "V1"
VERBOSITY = 2
SCAN_CATEGORIES = [mri_type.value for mri_type in MRIType]
EXCLUDED_IDS = [109, 123, 709]

RUN_DIR = "./run"
INPUT_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_DATASET_PATH = INPUT_PATH + "/train"
TRAIN_DATASET_DF_DIR = INPUT_PATH + "/train_labels.csv"

TEST_DATASET_PATH = INPUT_PATH + "/test"
TEST_DATASET_DF_DIR = INPUT_PATH + "/sample_submission.csv"

SUBMISSION_DATASET_DF_DIR = "/kaggle/working/submission.csv"

LOGS_PATH = f"{RUN_DIR}/logs"
BEST_MODEL_PATH = f"{RUN_DIR}/models"
BEST_MODEL_H5_DIR = f"{BEST_MODEL_PATH}/model_{VERSION}.h5"

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 0.95
IMG_ROTATE = 0
IMG_ENABLE_CENTRAL_FOCUS = True

SHUFFLE = True

INPUT_SHAPE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN)

MODEL_NAME = "Mult3DCNN4Input"
BATCH_SIZE = 8
EPOCHS = 26

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)
COMPILE_LOSS = "binary_crossentropy"
COMPILE_METRICS = [AUC(name="auc")]

TF_CALL_BACK_BEST_MODEL_MONITOR = "val_auc"
TF_CALL_BACK_EARLY_STOP_MONITOR = "auc"
TF_CALL_BACK_EARLY_STOP_PATIENTE = 6

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

DISK_CACHE_DIR = os.path.join("/kaggle/working", "dicom_cache_v1")
os.makedirs(DISK_CACHE_DIR, exist_ok=True)




## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
test_df["ID"] = test_df["ID"].astype(int)




## === cell 7
rng = np.random.RandomState(SEED)

labels = train_df["Label"].astype(int).values
pos_idx = np.where(labels == 1)[0]
neg_idx = np.where(labels == 0)[0]
rng.shuffle(pos_idx)
rng.shuffle(neg_idx)

val_pos = max(1, int(0.2 * len(pos_idx)))
val_neg = max(1, int(0.2 * len(neg_idx)))

val_idx = np.concatenate([pos_idx[:val_pos], neg_idx[:val_neg]])
trn_idx = np.concatenate([pos_idx[val_pos:], neg_idx[val_neg:]])

rng.shuffle(val_idx)
rng.shuffle(trn_idx)

train_split_df = train_df.iloc[trn_idx].reset_index(drop=True)
valid_split_df = train_df.iloc[val_idx].reset_index(drop=True)

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
    enable_cache=True,
    cache_max_items=512,
    disk_cache_dir=os.path.join(DISK_CACHE_DIR, "train"),
    enable_disk_cache=True,
    patient_disk_cache_dir=os.path.join(DISK_CACHE_DIR, "train_patient"),
    enable_patient_disk_cache=True,
)

valid_dicom_loader = DICOMLoader(
    valid_split_df,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    enable_cache=True,
    cache_max_items=256,
    disk_cache_dir=os.path.join(DISK_CACHE_DIR, "valid"),
    enable_disk_cache=True,
    patient_disk_cache_dir=os.path.join(DISK_CACHE_DIR, "valid_patient"),
    enable_patient_disk_cache=True,
)

train_dataset = ScanDataset(
    dicom_loader=train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=SHUFFLE,
    debug_mode=False,
)

valid_dataset = ScanDataset(
    dicom_loader=valid_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.VALIDATION.value,
    shuffle=False,
    debug_mode=False,
)

test_dicom_loader = DICOMLoader(
    test_df,
    input_path=TEST_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    enable_cache=True,
    cache_max_items=256,
    disk_cache_dir=os.path.join(DISK_CACHE_DIR, "test"),
    enable_disk_cache=True,
    patient_disk_cache_dir=os.path.join(DISK_CACHE_DIR, "test_patient"),
    enable_patient_disk_cache=True,
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)

train_tfds = sequence_to_tfdata(
    train_dataset,
    is_trainable=True,
    input_shape=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    num_modalities=len(SCAN_CATEGORIES),
)
valid_tfds = sequence_to_tfdata(
    valid_dataset,
    is_trainable=True,
    input_shape=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    num_modalities=len(SCAN_CATEGORIES),
)
test_tfds = sequence_to_tfdata(
    test_dataset,
    is_trainable=False,
    input_shape=INPUT_SHAPE,
    batch_size=BATCH_SIZE,
    num_modalities=len(SCAN_CATEGORIES),
)

test_tfds_for_predict = test_tfds




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

    def show_graph(self):
        if plot_model is None:
            print("plot_model unavailable in this environment.")
        else:
            display(plot_model(self, show_shapes=True, show_layer_names=True))




## === cell 9
model = DeepScanModel(input_shape=INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

callbacks = [
    ModelCheckpoint(
        BEST_MODEL_H5_DIR,
        monitor=TF_CALL_BACK_BEST_MODEL_MONITOR,
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    EarlyStopping(
        monitor=TF_CALL_BACK_EARLY_STOP_MONITOR,
        mode="max",
        patience=TF_CALL_BACK_EARLY_STOP_PATIENTE,
        restore_best_weights=False,
        verbose=1,
    ),
]

history = model.fit(
    train_tfds,
    validation_data=valid_tfds,
    epochs=EPOCHS,
    verbose=VERBOSITY,
    callbacks=callbacks,
)

if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )




## === cell 10
def generate_predictions(model, test_dataset_for_predict, test_df):
    preds = model.predict(
        test_dataset_for_predict,
        verbose=0,
    ).reshape(-1)

    submission = test_df.copy()
    submission["Label"] = preds[: len(submission)]
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["MGMT_value"] = submission["MGMT_value"].astype(float).clip(0.0, 1.0)
    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )

    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_tfds_for_predict, test_df)
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)

print(submission.head())
print("Wrote:", os.path.abspath(SUBMISSION_DATASET_DF_DIR), "rows:", len(submission))
print("Columns:", list(submission.columns))
