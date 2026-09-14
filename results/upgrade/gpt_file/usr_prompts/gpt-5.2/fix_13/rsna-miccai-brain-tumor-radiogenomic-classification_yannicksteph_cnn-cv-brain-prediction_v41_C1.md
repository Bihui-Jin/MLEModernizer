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
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import random
import numpy as np
import pandas as pd

import cv2
import pydicom

from enum import Enum
from concurrent.futures import ThreadPoolExecutor, as_completed

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
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

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass




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

    - 'W-H-D-C' = Width-Height-Depth-Channel
    - 'D-W-H-C' = Depth-Width-Height-Channel
    """

    WHDC = "W-H-D-C"  # Format (W, H, D, C) e.g. (128, 128, 32, 1)
    DWHC = "D-W-H-C"  # Format (D, W, H, C)

    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.DWHC:
            image = np.transpose(image, (2, 1, 0, 3))
        elif image_format == ImageFormat.WHDC:
            image = np.transpose(image, (2, 1, 0, 3))
        return image




## === cell 4
_GLOBAL_DICOM_EXECUTOR = None


def _get_global_executor(max_workers: int):
    global _GLOBAL_DICOM_EXECUTOR
    if _GLOBAL_DICOM_EXECUTOR is None:
        _GLOBAL_DICOM_EXECUTOR = ThreadPoolExecutor(max_workers=max_workers)
    return _GLOBAL_DICOM_EXECUTOR


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
        image_file_sorter=lambda x: int(
            os.path.splitext(os.path.basename(x))[0]
            .split("-")[-1]
            .replace("Image", "")
            .strip()
        ),
        debug_mode=False,
    ):
        if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
            raise ValueError("num_imgs must be divisible by 2 for central image")

        if not (0 <= rotate_angle <= 360):
            raise ValueError("Rotation value must be between 0 and 360")

        if id_column_name not in df.columns:
            raise ValueError(f"Column {id_column_name} must be in dataset")
        self.__has_label = label_column_name in df.columns

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

        self.__cache = {}  # in-memory per-row cache of already loaded volumes
        self.__dir_cache = {}  # in-memory cache of per (scans_path) file lists

        self.__cache_dir = os.path.join("/kaggle/working", "dicom_cache_v1_shared")
        os.makedirs(self.__cache_dir, exist_ok=True)

        self.__cache_key = (
            f"n{self.__num_imgs}_s{self.__size[0]}x{self.__size[1]}"
            f"_sc{self.__scale}_r{self.__rotate_angle}_cf{int(self.__enable_center_focus)}"
        )

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

    @property
    def has_label(self):
        return self.__has_label

    def get_id(self, row):
        return self.__df.loc[row, self.__id_column_name]

    def gel_label(self, row):
        if not self.__has_label:
            raise KeyError(
                f"Label column '{self.__label_column_name}' not available in this dataframe."
            )
        return self.__df.loc[row, self.__label_column_name]

    def format(self, images, type):
        if type == "default":
            if images.ndim == 4:
                return np.transpose(images, (1, 2, 0, 3))  # (D,H,W,C)->(H,W,D,C)
            return images
        elif type == "normalize":
            if images.ndim == 4:
                return np.transpose(images, (1, 2, 0, 3))
            return images
        else:
            return images

    def _list_sorted_files(self, scans_path):
        cached = self.__dir_cache.get(scans_path)
        if cached is not None:
            return cached
        with os.scandir(scans_path) as it:
            files = [entry.path for entry in it if entry.is_file()]
        files.sort(key=self.__image_file_sorter)
        self.__dir_cache[scans_path] = files
        return files

    def _get_image_files(self, row, scan_category):
        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
        scans_path = os.path.join(self.__input_path, patient_id, scan_category)

        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        image_files = self._list_sorted_files(scans_path)
        if not image_files:
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)
        return patient_id, image_files

    def _cache_path(self, patient_id, scan_category):
        fname = f"{patient_id}_{scan_category}_{self.__cache_key}.npz"
        return os.path.join(self.__cache_dir, fname)

    def load_scan(self, row, scan_category, show_progress=True):
        all_scans = self.load_all_scans(row, show_progress=show_progress)
        return all_scans[scan_category]

    def load_all_scans(self, row, show_progress=True):
        cached = self.__cache.get(row)
        if cached is not None:
            return cached

        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
        ordered = {}

        for scan_category in self.__scan_categories:
            cache_path = self._cache_path(patient_id, scan_category)

            if os.path.exists(cache_path):
                with np.load(cache_path) as z:
                    vol = z["vol"]
                ordered[scan_category] = vol
                continue

            _, image_files = self._get_image_files(row, scan_category)

            loaded_images = list(
                self.__executor.map(self._load_dicom_image, image_files)
            )
            loaded_images = [im for im in loaded_images if im is not None]

            if not loaded_images:
                raise ValueError("No images were loaded. Cannot proceed.")

            if self.__num_imgs is not None:
                while len(loaded_images) < self.__num_imgs:
                    zero_image = np.zeros_like(loaded_images[0])
                    loaded_images.append(zero_image)

            loaded_images = np.asarray(loaded_images, dtype=np.uint8)  # (D, H, W, 1)
            vol = self.format(loaded_images, "default")  # (H, W, D, 1)

            np.savez_compressed(cache_path, vol=vol)
            ordered[scan_category] = vol

        ordered = {key: ordered.get(key, []) for key in self.__scan_categories}
        self.__cache[row] = ordered
        return ordered

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
                    "PhotometricInterpretation",
                    "SamplesPerPixel",
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
        image = image.astype(np.float32)
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        denom = max_val - min_val
        if denom <= 0:
            return np.zeros_like(image, dtype=np.uint8)
        image = (image - min_val) / denom
        image = np.clip(image, 0.0, 1.0)
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
        self.__batch_size = int(batch_size)
        self.__subset = subset.lower()
        self.__is_trainable = self.__subset in ["validation", "train", "training"]
        self.__shuffle = shuffle
        self.__debug = InternalDebug(debug_mode=debug_mode)

        self.__n = self.__dicom_loader.len
        self.__indices = np.arange(self.__n, dtype=np.int32)
        if self.__shuffle:
            np.random.shuffle(self.__indices)

        self.__scan_categories = list(self.__dicom_loader.scan_categories)
        self.__num_scans = len(self.__scan_categories)
        _size = self.__dicom_loader._DICOMLoader__size
        self.__vol_shape = (_size[0], _size[1], self.__dicom_loader.num_imgs, 1)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        bs0 = ids * self.__batch_size
        bs1 = min((ids + 1) * self.__batch_size, self.__n)
        batch_indices = self.__indices[bs0:bs1]
        bs = int(batch_indices.shape[0])

        vol_shape = self.__vol_shape
        x_batches = [
            np.empty((bs,) + vol_shape, dtype=np.uint8) for _ in range(self.__num_scans)
        ]
        y = np.empty((bs,), dtype=np.float32) if self.__is_trainable else None

        loader = self.__dicom_loader
        scan_categories = self.__scan_categories
        is_trainable = self.__is_trainable

        for bi, i in enumerate(batch_indices):
            ii = int(i)
            if is_trainable:
                y[bi] = loader.gel_label(ii)

            scans = loader.load_all_scans(ii, show_progress=False)
            for sj, scan_type in enumerate(scan_categories):
                x_batches[sj][bi] = scans[scan_type]

        batch_x = tuple(x_batches)
        if is_trainable:
            return batch_x, y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__n / self.__batch_size))




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

KERAS_WORKERS = 4
KERAS_USE_MULTIPROCESSING = False
KERAS_MAX_QUEUE_SIZE = 16




## === cell 8
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

perm = np.random.RandomState(SEED).permutation(len(train_df))
val_size = max(1, int(0.2 * len(train_df)))
val_idx = perm[:val_size]
tr_idx = perm[val_size:]

train_split_df = train_df.iloc[tr_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)




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

train_dataset = ScanDataset(
    train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=True,
    debug_mode=False,
)
val_dataset = ScanDataset(
    val_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.VALIDATION.value,
    shuffle=False,
    debug_mode=False,
)
test_dataset = ScanDataset(
    test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)




## === cell 11
def _precache_all(loader: DICOMLoader, max_patients: int | None = None):
    n = loader.len if max_patients is None else min(loader.len, max_patients)

    def _one(i: int):
        loader.load_all_scans(i, show_progress=False)
        return i

    max_workers = min(16, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        futs = [ex.submit(_one, i) for i in range(n)]
        for _ in as_completed(futs):
            pass




## === cell 12
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

checkpoint_cb = tf.keras.callbacks.ModelCheckpoint(
    filepath=BEST_MODEL_H5_DIR,
    monitor=TF_CALL_BACK_BEST_MODEL_MONITOR,
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

if os.path.exists(BEST_MODEL_H5_DIR):
    model = tf.keras.models.load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )




## === cell 13
def generate_predictions(model, test_dataset, test_df):
    preds = (
        model.predict(
            test_dataset,
            verbose=0,
        )
        .reshape(-1)
        .astype(np.float32)
    )

    submission = test_df.copy()
    submission["Label"] = preds[: len(submission)]
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
    submission["MGMT_value"] = submission["MGMT_value"].astype(float).clip(0.0, 1.0)

    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_dataset, test_df)




## === cell 14
submission.info()
print(submission.head())




## === cell 15
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote submission.csv with shape:", submission.shape)
print("Saved to:", os.path.abspath(SUBMISSION_DATASET_DF_DIR))
print("Exists:", os.path.exists(SUBMISSION_DATASET_DF_DIR))
