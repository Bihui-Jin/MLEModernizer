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

os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import random
import hashlib
from enum import Enum

import numpy as np
import pandas as pd

import cv2

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model, load_model
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



## === cell 1
SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
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
    """
    Enum to represent image formats.

    - WHDC: Width-Height-Depth-Channel
    - DWHC: Depth-Width-Height-Channel
    """

    WHDC = "W-H-D-C"  # e.g. (W,H,D,C)
    DWHC = "D-W-H-C"  # e.g. (D,W,H,C)

    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.DWHC:
            image = np.transpose(image, (2, 1, 0, 3))
        elif image_format == ImageFormat.WHDC:
            image = np.transpose(image, (2, 1, 0, 3))
        return image




## === cell 4
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import OrderedDict

CACHE_DIR = "/kaggle/working/dicom_cache_v1"
os.makedirs(CACHE_DIR, exist_ok=True)

import pydicom

_GLOBAL_DICOM_EXECUTOR = None


def _get_global_executor(max_workers: int):
    global _GLOBAL_DICOM_EXECUTOR
    if _GLOBAL_DICOM_EXECUTOR is None:
        _GLOBAL_DICOM_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_DICOM_EXECUTOR


def _stable_cache_key(path, size, scale, rotate_angle):
    s = f"{path}|{size[0]}x{size[1]}|scale={scale}|rot={rotate_angle}"
    return hashlib.md5(s.encode("utf-8")).hexdigest()


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
    ):
        if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
            raise ValueError("num_imgs must be divisible by 2 for central image")

        if not (0 <= rotate_angle <= 360):
            raise ValueError("Rotation value must be between 0 and 360")

        if id_column_name not in df.columns:
            raise ValueError(f"Column {id_column_name} must be in dataset")
        if label_column_name not in df.columns:
            df = df.copy()
            df[label_column_name] = 0

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

        self.__executor = _get_global_executor(self.__max_threads)

        self.__scan_file_index = self._build_scan_file_index_fast()

        self.__mem_cache = OrderedDict()
        self.__mem_cache_max = 4096  # slices

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

    def _build_scan_file_index_fast(self):
        index = {}
        ids = self.__df[self.__id_column_name].to_numpy()
        for row, pid in enumerate(ids):
            patient_id = str(int(pid)).zfill(5)
            patient_map = {}
            base = os.path.join(self.__input_path, patient_id)
            for scan_category in self.__scan_categories:
                scans_path = os.path.join(base, scan_category)
                if os.path.isdir(scans_path):
                    try:
                        files = [e.path for e in os.scandir(scans_path) if e.is_file()]
                    except FileNotFoundError:
                        files = []
                    if files:
                        files.sort(key=self.__image_file_sorter)
                    patient_map[scan_category] = files
                else:
                    patient_map[scan_category] = []
            index[row] = patient_map
        return index

    def load_scan(self, row, scan_category, show_progress=False):
        self.__debug.log("== load_scan ==")

        image_files = self.__scan_file_index[row].get(scan_category, [])
        if not image_files:
            patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)

        loaded_images = []
        for image_data in self.__executor.map(self._load_dicom_image, image_files):
            if image_data is not None:
                loaded_images.append(image_data)

        if not loaded_images:
            raise ValueError("No images were loaded. Cannot proceed.")

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                loaded_images.append(np.zeros_like(loaded_images[0]))

        loaded_images = np.array(loaded_images)
        return self.format(loaded_images, "default")

    def load_all_scans(self, row, show_progress=False):
        self.__debug.log("== load_all_scans ==")
        all_images = {}

        future_to_scan_category = {
            self.__executor.submit(
                self.load_scan, row, scan_category, False
            ): scan_category
            for scan_category in self.__scan_categories
        }
        for future in as_completed(future_to_scan_category):
            scan_category = future_to_scan_category[future]
            image_data = future.result()
            if image_data is not None:
                all_images[scan_category] = image_data

        return {key: all_images.get(key, []) for key in self.__scan_categories}

    def _mem_cache_get(self, cache_path):
        v = self.__mem_cache.get(cache_path, None)
        if v is not None:
            self.__mem_cache.move_to_end(cache_path)
        return v

    def _mem_cache_put(self, cache_path, value):
        self.__mem_cache[cache_path] = value
        self.__mem_cache.move_to_end(cache_path)
        if len(self.__mem_cache) > self.__mem_cache_max:
            self.__mem_cache.popitem(last=False)

    def _load_dicom_image(self, dicom_path):
        self.__debug.log("== _load_dicom_image ==")

        key = _stable_cache_key(
            dicom_path, self.__size, self.__scale, self.__rotate_angle
        )
        cache_path = os.path.join(CACHE_DIR, f"{key}.npy")

        mem = self._mem_cache_get(cache_path)
        if mem is not None:
            return mem

        try:
            if os.path.exists(cache_path):
                arr = np.load(cache_path, allow_pickle=False, mmap_mode="r")
                arr = np.array(arr, copy=True)
                self._mem_cache_put(cache_path, arr)
                return arr
        except Exception:
            pass

        try:
            dicom_file = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=False,
                force=True,
                specific_tags=[],
            )
            image = dicom_file.pixel_array
        except Exception:
            try:
                dicom_file = pydicom.dcmread(
                    dicom_path,
                    force=True,
                    stop_before_pixels=False,
                    specific_tags=[],
                )
                image = dicom_file.pixel_array
            except Exception:
                w, h = self.__size
                image = np.zeros((h, w), dtype=np.uint8)
                image = np.expand_dims(image, axis=-1)
                self._mem_cache_put(cache_path, image)
                return image

        image = self._rotate_img(image)
        image = self._normalization_img(image)
        image = self._crop_img(image)
        image = self._resize_img(image)
        image = np.expand_dims(image, axis=-1)

        tmp_path = None
        try:
            tmp_path = cache_path + f".tmp_{os.getpid()}_{random.randint(0, 1_000_000)}"
            with open(tmp_path, "wb") as f:
                np.save(f, image, allow_pickle=False)
            os.replace(tmp_path, cache_path)
        except Exception:
            try:
                if tmp_path is not None and os.path.exists(tmp_path):
                    os.remove(tmp_path)
            except Exception:
                pass

        self._mem_cache_put(cache_path, image)
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
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        denom = max_val - min_val
        if denom <= 0:
            return np.zeros_like(image).astype(np.uint8)
        image = (image - min_val) / denom
        return (image * 255.0).astype(np.uint8)

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




## === cell 5
class ScanDataset(Sequence):
    def __init__(
        self, dicom_loader, batch_size, subset="train", shuffle=True, debug_mode=False
    ):
        self.__dicom_loader = dicom_loader
        self.__batch_size = batch_size
        self.__subset = subset.lower()
        self.__is_trainable = self.__subset in ["validation", "train", "training"]
        self.__shuffle = shuffle
        self.__debug = InternalDebug(debug_mode=debug_mode)
        self.__indices = np.arange(self.__dicom_loader.len)
        if self.__shuffle:
            np.random.shuffle(self.__indices)

        self.__n_inputs = len(self.__dicom_loader.scan_categories)
        self.__h = None
        self.__w = None
        self.__d = self.__dicom_loader.num_imgs
        self.__c = 1

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bs = len(batch_indices)

        if self.__h is None or self.__w is None:
            i0 = int(batch_indices[0])
            sample = self.__dicom_loader.load_all_scans(i0, show_progress=False)
            first_key = self.__dicom_loader.scan_categories[0]
            arr = sample[first_key]
            self.__d, self.__w, self.__h, self.__c = arr.shape

        batches_x = [
            np.empty((bs, self.__d, self.__w, self.__h, self.__c), dtype=np.uint8)
            for _ in range(self.__n_inputs)
        ]
        batches_y = np.empty((bs,), dtype=np.float32) if self.__is_trainable else None

        for bi, i in enumerate(batch_indices):
            if self.__is_trainable:
                batches_y[bi] = self.__dicom_loader.gel_label(i)

            batch_x_images = self.__dicom_loader.load_all_scans(i, show_progress=False)
            for j, scan_category in enumerate(self.__dicom_loader.scan_categories):
                batches_x[j][bi] = batch_x_images[scan_category]

        batch_x = tuple(batches_x)
        if self.__is_trainable:
            return batch_x, batches_y
        return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))




## === cell 6
class MRIType(Enum):
    FLAIR = "FLAIR"
    T1w = "T1w"
    T1wCE = "T1wCE"
    T2w = "T2w"


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"




## === cell 7
VERSION = "V1"
VERBOSITY = 2
SCAN_CATEGORIES = [mri_type.value for mri_type in MRIType]

EXCLUDED_IDS = [109, 123, 709]

RUN_DIR = "./run"
INPUT_PATH = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_DATASET_PATH = INPUT_PATH + "/train"
TRAIN_DATASET_DF_DIR = INPUT_PATH + "/train_labels.csv"

TEST_DATASET_PATH = INPUT_PATH + "/test"
TEST_DATASET_DF_DIR = INPUT_PATH + "/sample_submission.csv"

SUBMISSION_DATASET_DF_DIR = "/kaggle/working/submission.csv"

LOGS_PATH = f"{RUN_DIR}/logs"
BEST_MODEL_PATH = f"{RUN_DIR}/models"
os.makedirs(BEST_MODEL_PATH, exist_ok=True)
BEST_MODEL_H5_DIR = f"{BEST_MODEL_PATH}/model_{VERSION}.h5"

NUM_SPLIT_FOLDS = 5
SELECTED_VALIDATION_FOLD = 1  # kept but not used for minimal changes

MAX_THREADS_DICOM_LOADER = max(8, min(32, (os.cpu_count() or 8)))

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 1
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

KERAS_WORKERS = min(8, os.cpu_count() or 2)
KERAS_USE_MULTIPROCESSING = False
KERAS_MAX_QUEUE_SIZE = 32



## === cell 8
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
test_df["ID"] = test_df["ID"].astype(int)




## === cell 9
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




## === cell 10
val_frac = 0.2
rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)

split = int(len(idx) * (1 - val_frac))
train_idx, val_idx = idx[:split], idx[split:]

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
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
)

train_dataset = ScanDataset(
    train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=True,
)
val_dataset = ScanDataset(
    val_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.VALIDATION.value,
    shuffle=False,
)




## === cell 11
def _precache_loader(dicom_loader: DICOMLoader, name: str):
    return  # intentionally disabled to meet the 600s timeout without changing data/model logic


_precache_loader(train_dicom_loader, "train")
_precache_loader(val_dicom_loader, "val")



## === cell 12
if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )
else:
    model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
    model.compile(
        optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS
    )

    model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=EPOCHS,
        verbose=VERBOSITY,
    )
    model.save(BEST_MODEL_H5_DIR)



## === cell 13
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
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)



## === cell 14
_precache_loader(test_dicom_loader, "test")


def generate_predictions(model, test_dataset, test_df):
    preds = model.predict(
        test_dataset,
        verbose=0,
    )
    preds = preds.reshape(-1)[: len(test_df)]

    submission = test_df.copy()
    submission["Label"] = preds.astype(np.float32)
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_dataset, test_df)



## === cell 15
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote:", SUBMISSION_DATASET_DF_DIR)
print(submission.head())
print(submission.shape)
