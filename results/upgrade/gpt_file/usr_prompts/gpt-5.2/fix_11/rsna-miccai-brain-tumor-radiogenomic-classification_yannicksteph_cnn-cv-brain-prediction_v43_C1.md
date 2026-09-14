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

# 5. Target score

0.5498045215026347

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46706) has done: 'I fix the import-time crash by removing the legacy `keras` import that triggers the protobuf `MessageFactory.GetPrototype` error in this Kaggle Python 3.11 environment and rely on `tf.keras` only. Then I fix the missing external pretrained model file by training the provided 4-branch 3D CNN on the local train set and saving/using it from `/kaggle/working`, which keeps the core model architecture and training semantics intact while enabling end-to-end execution. I also correct a couple of logic issues that would break training/inference: the `ImageFormat.swap_dimensions` method signature and the loader’s normalization bug (division by `max_val` instead of `max_val-min_val`). Finally, I ensure predictions align exactly with `sample_submission.csv` ordering and write a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.46706) has done: 'The timeout is dominated by repeated DICOM decoding and per-batch Python overhead: every epoch re-reads the same slices from disk, and `load_all_scans()` also spawns nested thread pools (one for scan types and one for slices) which adds heavy overhead. I keep the same preprocessing steps, same slice selection, and the exact same model/training loop semantics, but make data loading provably equivalent and much faster by (1) removing the nested thread pool and instead loading scan categories sequentially while keeping the per-scan slice thread pool, and (2) adding an in-memory LRU cache for decoded/preprocessed slices keyed by file path + parameters, so subsequent epochs reuse identical arrays. I also enable `Sequence` multiprocessing/prefetch in `model.fit()` (does not change results) and reduce per-epoch overhead by precomputing/resolving file lists once per (patient, scan) inside the loader cache. These changes preserve accuracy because they do not alter which files are read, how they are transformed, or the training schedule—only eliminate redundant work and thread-management overhead.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import glob
import random
from enum import Enum

import numpy as np
import pandas as pd

import cv2
import pydicom

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




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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
IS_PYDICOM_IMPORTED = True
if IS_PYDICOM_IMPORTED:
    from concurrent.futures import ThreadPoolExecutor
    from collections import OrderedDict
    import threading

    class ImageFormat(Enum):
        """
        Enum to represent image formats.

        - 'W-H-D-C' = Width-Height-Depth-Channel
        - 'D-W-H-C' = Depth-Width-Height-Channel
        """

        WHDC = "W-H-D-C"  # Format (W, H, D, C)
        DWHC = "D-W-H-C"  # Format (D, W, H, C)

        @staticmethod
        def swap_dimensions(image, image_format):
            if image_format in (ImageFormat.DWHC, ImageFormat.WHDC):
                return np.transpose(image, (2, 1, 0, 3))
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
            image_file_sorter=lambda x: int(os.path.basename(x)[:-4].split("-")[-1]),
            debug_mode=False,
            cache_max_items=20000,
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
            self.__max_threads = int(max_threads) if max_threads else 1
            self.__size = size
            self.__scale = scale
            self.__rotate_angle = rotate_angle
            self.__image_format = image_format
            self.__image_file_sorter = image_file_sorter
            self.__enable_center_focus = enable_center_focus
            self.__debug = InternalDebug(debug_mode=debug_mode)

            self.__ids = self.__df[self.__id_column_name].to_numpy()
            self.__labels = self.__df[self.__label_column_name].to_numpy()

            self.__cache_max_items = int(cache_max_items) if cache_max_items else 0
            self.__slice_cache = OrderedDict()
            self.__slice_cache_lock = threading.Lock()

            self.__filelist_cache = {}
            self.__filelist_cache_lock = threading.Lock()

            self.__volume_cache = OrderedDict()
            self.__volume_cache_lock = threading.Lock()

            self.__rotation_matrix_cache = {}
            self.__rotation_matrix_lock = threading.Lock()

            self.__executor = ThreadPoolExecutor(max_workers=self.__max_threads)

        def __del__(self):
            try:
                if (
                    hasattr(self, "_DICOMLoader__executor")
                    and self.__executor is not None
                ):
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
            return self.__ids[row]

        def gel_label(self, row):
            return self.__labels[row]

        def format(self, images, type):
            if type == "normalize" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)
            elif type == "default" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
            else:
                return images

        def _cache_key_for_slice(self, dicom_path):
            return (
                dicom_path,
                self.__size,
                float(self.__scale),
                float(self.__rotate_angle),
            )

        def _get_cached_slice(self, key):
            if self.__cache_max_items <= 0:
                return None
            with self.__slice_cache_lock:
                v = self.__slice_cache.get(key, None)
                if v is not None:
                    self.__slice_cache.move_to_end(key)
                return v

        def _put_cached_slice(self, key, value):
            if self.__cache_max_items <= 0:
                return
            with self.__slice_cache_lock:
                self.__slice_cache[key] = value
                self.__slice_cache.move_to_end(key)
                while len(self.__slice_cache) > self.__cache_max_items:
                    self.__slice_cache.popitem(last=False)

        def _volume_cache_key(self, patient_id, scan_category):
            return (
                patient_id,
                scan_category,
                self.__enable_center_focus,
                self.__num_imgs,
                self.__size,
                float(self.__scale),
                float(self.__rotate_angle),
                self.__image_format.value,
            )

        def _get_cached_volume(self, key):
            if self.__cache_max_items <= 0:
                return None
            with self.__volume_cache_lock:
                v = self.__volume_cache.get(key, None)
                if v is not None:
                    self.__volume_cache.move_to_end(key)
                return v

        def _put_cached_volume(self, key, value):
            if self.__cache_max_items <= 0:
                return
            with self.__volume_cache_lock:
                self.__volume_cache[key] = value
                self.__volume_cache.move_to_end(key)
                max_vols = max(64, self.__cache_max_items // 256)
                while len(self.__volume_cache) > max_vols:
                    self.__volume_cache.popitem(last=False)

        def _get_image_files(self, patient_id, scan_category):
            cache_key = (
                patient_id,
                scan_category,
                self.__enable_center_focus,
                self.__num_imgs,
            )
            with self.__filelist_cache_lock:
                if cache_key in self.__filelist_cache:
                    return self.__filelist_cache[cache_key]

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

            with self.__filelist_cache_lock:
                self.__filelist_cache[cache_key] = image_files
            return image_files

        def load_scan(self, row, scan_category, show_progress=True):
            self.__debug.log("== load_scan ==")
            patient_id = str(self.get_id(row)).zfill(5)

            vol_key = self._volume_cache_key(patient_id, scan_category)
            cached_vol = self._get_cached_volume(vol_key)
            if cached_vol is not None:
                return cached_vol

            image_files = self._get_image_files(patient_id, scan_category)

            loaded_images = []
            loaded_images_extend = loaded_images.append
            for image_data in self.__executor.map(self._load_dicom_image, image_files):
                if image_data is not None:
                    loaded_images_extend(image_data)

            if not loaded_images:
                raise ValueError(
                    f"No images were loaded for patient={patient_id}, scan={scan_category}."
                )

            if self.__num_imgs is not None and len(loaded_images) < self.__num_imgs:
                zero_image = np.zeros_like(loaded_images[0])
                loaded_images.extend(
                    [zero_image] * (self.__num_imgs - len(loaded_images))
                )

            loaded_images = np.asarray(loaded_images, dtype=np.uint8)
            loaded_images = self.format(loaded_images, "default")

            self._put_cached_volume(vol_key, loaded_images)
            return loaded_images

        def load_all_scans(self, row, show_progress=True):
            self.__debug.log("== load_all_scans ==")
            futures = {
                scan_category: self.__executor.submit(
                    self.load_scan, row, scan_category, False
                )
                for scan_category in self.__scan_categories
            }
            all_images = {
                scan_category: futures[scan_category].result()
                for scan_category in self.__scan_categories
            }
            return {key: all_images.get(key, []) for key in self.__scan_categories}

        def _load_dicom_image(self, dicom_path):
            self.__debug.log("== _load_dicom_image ==")
            if not os.path.exists(dicom_path):
                return None

            key = self._cache_key_for_slice(dicom_path)
            cached = self._get_cached_slice(key)
            if cached is not None:
                return cached

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
                        "PixelRepresentation",
                        "SamplesPerPixel",
                        "PhotometricInterpretation",
                    ],
                )
                image = dicom_file.pixel_array
            except Exception:
                return None

            if self.__rotate_angle > 0:
                image = self._rotate_img(image)
            image = self._normalization_img(image)
            if self.__scale > 0:
                image = self._crop_img(image)
            image = self._resize_img(image)
            image = np.expand_dims(image, axis=-1)

            self._put_cached_slice(key, image)
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
            cache_key = (height, width, float(self.__rotate_angle))
            with self.__rotation_matrix_lock:
                rotation_matrix = self.__rotation_matrix_cache.get(cache_key)
                if rotation_matrix is None:
                    center = (width / 2, height / 2)
                    rotation_matrix = cv2.getRotationMatrix2D(
                        center, self.__rotate_angle, 1.0
                    )
                    self.__rotation_matrix_cache[cache_key] = rotation_matrix
            return cv2.warpAffine(image, rotation_matrix, (width, height))

        def _normalization_img(self, image):
            min_val = np.min(image)
            max_val = np.max(image)
            denom = max_val - min_val
            if denom <= 0:
                return np.zeros_like(image).astype(np.uint8)

            image = (image - min_val) / denom
            return (image * 255).astype(np.uint8)

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
        self,
        dicom_loader,
        batch_size,
        subset="train",
        shuffle=True,
        debug_mode=False,
    ):
        self.__dicom_loader = dicom_loader
        self.__batch_size = batch_size
        self.__is_trainable = subset.lower() in ["validation", "train"]
        self.__shuffle = shuffle
        self.__debug = InternalDebug(debug_mode=debug_mode)
        self.__indices = np.arange(self.__dicom_loader.len)

        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]

        bs = len(batch_indices)
        batches_y = np.empty((bs,), dtype=np.float32)

        batches_x = [None] * len(self.__dicom_loader.scan_categories)

        for bi, i in enumerate(batch_indices):
            batches_y[bi] = self.__dicom_loader.gel_label(i)
            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )
            for j, (_scan_type, images) in enumerate(batch_x_image_paths.items()):
                if batches_x[j] is None:
                    batches_x[j] = np.empty((bs,) + images.shape, dtype=images.dtype)
                batches_x[j][bi] = images

        batch_x = tuple(batches_x)
        if self.__is_trainable:
            return batch_x, batches_y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))




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
SEED = 123
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

NUM_SPLIT_FOLDS = 5
SELECTED_VALIDATION_FOLD = 1

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 0.85
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

TF_CALL_BACK_EARLY_STOP_MONITOR = "val_auc"
TF_CALL_BACK_EARLY_STOP_PATIENTE = 6

tf.keras.utils.set_random_seed(SEED)
try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

np.random.seed(SEED)
random.seed(SEED)

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)



## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)



## === cell 7
perm = np.random.RandomState(SEED).permutation(len(train_df))
folds = np.empty(len(train_df), dtype=int)
folds[perm] = (np.arange(len(train_df)) % NUM_SPLIT_FOLDS).astype(int)
train_df["fold"] = folds

train_split_df = train_df[train_df["fold"] != SELECTED_VALIDATION_FOLD].reset_index(
    drop=True
)
val_split_df = train_df[train_df["fold"] == SELECTED_VALIDATION_FOLD].reset_index(
    drop=True
)

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
    cache_max_items=20000,
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
    cache_max_items=20000,
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
    cache_max_items=20000,
)




## === cell 8
def _sequence_to_tf_dataset(seq: Sequence, input_shape, batch_size, is_trainable: bool):
    input_spec = tuple(
        tf.TensorSpec(shape=(None,) + input_shape, dtype=tf.uint8) for _ in range(4)
    )
    if is_trainable:
        out_sig = (input_spec, tf.TensorSpec(shape=(None,), dtype=tf.float32))
    else:
        out_sig = input_spec

    def gen():
        for i in range(len(seq)):
            yield seq[i]

    ds = tf.data.Dataset.from_generator(gen, output_signature=out_sig)

    def _cast(x, y=None):
        x = tuple(tf.cast(t, tf.float32) for t in x)
        if is_trainable:
            return x, y
        return x

    ds = ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_sequence = ScanDataset(
    train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset="train",
    shuffle=SHUFFLE,
    debug_mode=False,
)
val_sequence = ScanDataset(
    val_dicom_loader,
    batch_size=BATCH_SIZE,
    subset="validation",
    shuffle=False,
    debug_mode=False,
)
test_sequence = ScanDataset(
    test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset="test",
    shuffle=False,
    debug_mode=False,
)

train_dataset = _sequence_to_tf_dataset(
    train_sequence, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, is_trainable=True
)
val_dataset = _sequence_to_tf_dataset(
    val_sequence, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, is_trainable=True
)
test_dataset = _sequence_to_tf_dataset(
    test_sequence, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, is_trainable=False
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/522543869.py in <cell line: 0>()
     57     val_sequence, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, is_trainable=True
     58 )
---> 59 test_dataset = _sequence_to_tf_dataset(
     60     test_sequence, input_shape=INPUT_SHAPE, batch_size=BATCH_SIZE, is_trainable=False
     61 )

/tmp/ipykernel_11/522543869.py in _sequence_to_tf_dataset(seq, input_shape, batch_size, is_trainable)
     24         return x
     25 
---> 26     ds = ds.map(_cast, num_parallel_calls=tf.data.AUTOTUNE)
     27     ds = ds.prefetch(tf.data.AUTOTUNE)
     28     return ds

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/dataset_ops.py in map(self, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
   2339     from tensorflow.python.data.ops import map_op
   2340 
-> 2341     return map_op._map_v2(
   2342         self,
   2343         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in _map_v2(input_dataset, map_func, num_parallel_calls, deterministic, synchronous, use_unbounded_threadpool, name)
     55           num_parallel_calls,
     56       )
---> 57     return _ParallelMapDataset(
     58         input_dataset,
     59         map_func,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/map_op.py in __init__(self, input_dataset, map_func, num_parallel_calls, deterministic, use_inter_op_parallelism, preserve_cardinality, use_legacy_function, use_unbounded_threadpool, name)
    200     self._input_dataset = input_dataset
    201     self._use_inter_op_parallelism = use_inter_op_parallelism
--> 202     self._map_func = structured_function.StructuredFunctionWrapper(
    203         map_func,
    204         self._transformation_name(),

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in __init__(self, func, transformation_name, dataset, input_classes, input_shapes, input_types, input_structure, add_to_graph, use_legacy_function, defun_kwargs)
    263         fn_factory = trace_tf_function(defun_kwargs)
    264 
--> 265     self._function = fn_factory()
    266     # There is no graph to add in eager mode.
    267     add_to_graph &= not context.executing_eagerly()

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in get_concrete_function(self, *args, **kwargs)
   1249   def get_concrete_function(self, *args, **kwargs):
   1250     # Implements PolymorphicFunction.get_concrete_function.
-> 1251     concrete = self._get_concrete_function_garbage_collected(*args, **kwargs)
   1252     concrete._garbage_collector.release()  # pylint: disable=protected-access
   1253     return concrete

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _get_concrete_function_garbage_collected(self, *args, **kwargs)
   1219       if self._variable_creation_config is None:
   1220         initializers = []
-> 1221         self._initialize(args, kwargs, add_initializers_to=initializers)
   1222         self._initialize_uninitialized_variables(initializers)
   1223 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in _initialize(self, args, kwds, add_initializers_to)
    694     )
    695     # Force the definition of the function for these arguments
--> 696     self._concrete_variable_creation_fn = tracing_compilation.trace_function(
    697         args, kwds, self._variable_creation_config
    698     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in trace_function(args, kwargs, tracing_options)
    176       kwargs = {}
    177 
--> 178     concrete_function = _maybe_define_function(
    179         args, kwargs, tracing_options
    180     )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _maybe_define_function(args, kwargs, tracing_options)
    281         else:
    282           target_func_type = lookup_func_type
--> 283         concrete_function = _create_concrete_function(
    284             target_func_type, lookup_func_context, func_graph, tracing_options
    285         )

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/tracing_compilation.py in _create_concrete_function(function_type, type_context, func_graph, tracing_options)
    308       attributes_lib.DISABLE_ACD, False
    309   )
--> 310   traced_func_graph = func_graph_module.func_graph_from_py_func(
    311       tracing_options.name,
    312       tracing_options.python_function,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/framework/func_graph.py in func_graph_from_py_func(name, python_func, args, kwargs, signature, func_graph, add_control_dependencies, arg_names, op_return_value, collections, capture_by_value, create_placeholders)
   1057 
   1058     _, original_func = tf_decorator.unwrap(python_func)
-> 1059     func_outputs = python_func(*func_args, **func_kwargs)
   1060 
   1061     # invariant: `func_outputs` contains only Tensors, CompositeTensors,

/usr/local/lib/python3.11/dist-packages/tensorflow/python/eager/polymorphic_function/polymorphic_function.py in wrapped_fn(*args, **kwds)
    597         # the function a weak reference to itself to avoid a reference cycle.
    598         with OptionalXlaContext(compile_with_xla):
--> 599           out = weak_wrapped_fn().__wrapped__(*args, **kwds)
    600         return out
    601 

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapped_fn(*args)
    229       # Note: wrapper_helper will apply autograph based on context.
    230       def wrapped_fn(*args):  # pylint: disable=missing-docstring
--> 231         ret = wrapper_helper(*args)
    232         ret = structure.to_tensor_list(self._output_structure, ret)
    233         return [ops.convert_to_tensor(t) for t in ret]

/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/structured_function.py in wrapper_helper(*args)
    159       if not _should_unpack(nested_args):
    160         nested_args = (nested_args,)
--> 161       ret = autograph.tf_convert(self._func, ag_ctx)(*nested_args)
    162       ret = variable_utils.convert_variables_to_tensors(ret)
    163       if _should_pack(ret):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):
--> 693           raise e.ag_error_metadata.to_exception(e)
    694         else:
    695           raise

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in wrapper(*args, **kwargs)
    688       try:
    689         with conversion_ctx:
--> 690           return converted_call(f, args, kwargs, options=options)
    691       except Exception as e:  # pylint:disable=broad-except
    692         if hasattr(e, 'ag_error_metadata'):

/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py in converted_call(f, args, kwargs, caller_fn_scope, options)
    437     try:
    438       if kwargs is not None:
--> 439         result = converted_f(*effective_args, **kwargs)
    440       else:
    441         result = converted_f(*effective_args)

TypeError: in user code:


    TypeError: outer_factory.<locals>.inner_factory.<locals>.tf___cast() takes from 1 to 2 positional arguments but 4 were given


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
model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_H5_DIR,
        monitor=TF_CALL_BACK_BEST_MODEL_MONITOR,
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    ),
    tf.keras.callbacks.EarlyStopping(
        monitor=TF_CALL_BACK_EARLY_STOP_MONITOR,
        mode="max",
        patience=TF_CALL_BACK_EARLY_STOP_PATIENTE,
        restore_best_weights=True,
        verbose=1,
    ),
]

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=VERBOSITY,
    callbacks=callbacks,
)

if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR,
        custom_objects={"DeepScanModel": DeepScanModel},
        compile=False,
    )
    model.compile(
        optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS
    )




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
InvalidArgumentError                      Traceback (most recent call last)
/tmp/ipykernel_11/2280289531.py in <cell line: 0>()
     20 ]
     21 
---> 22 history = model.fit(
     23     train_dataset,
     24     validation_data=val_dataset,

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

InvalidArgumentError: Graph execution error:

Detected at node PyFunc defined at (most recent call last):
<stack traces unavailable>
ValueError: No images were loaded for patient=00350, scan=FLAIR.
Traceback (most recent call last):

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/ops/script_ops.py", line 269, in __call__
    ret = func(*args)
          ^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/autograph/impl/api.py", line 643, in wrapper
    return func(*args, **kwargs)
           ^^^^^^^^^^^^^^^^^^^^^

  File "/usr/local/lib/python3.11/dist-packages/tensorflow/python/data/ops/from_generator_op.py", line 198, in generator_py_func
    values = next(generator_state.get_iterator(iterator_id))
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/522543869.py", line 16, in gen
    yield seq[i]
          ~~~^^^

  File "/tmp/ipykernel_11/2242393413.py", line 36, in __getitem__
    batch_x_image_paths = self.__dicom_loader.load_all_scans(
                          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/444493261.py", line 258, in load_all_scans
    all_images = {
                 ^

  File "/tmp/ipykernel_11/444493261.py", line 259, in <dictcomp>
    scan_category: futures[scan_category].result()
                   ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/usr/lib/python3.11/concurrent/futures/_base.py", line 456, in result
    return self.__get_result()
           ^^^^^^^^^^^^^^^^^^^

  File "/usr/lib/python3.11/concurrent/futures/_base.py", line 401, in __get_result
    raise self._exception

  File "/usr/lib/python3.11/concurrent/futures/thread.py", line 58, in run
    result = self.fn(*self.args, **self.kwargs)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

  File "/tmp/ipykernel_11/444493261.py", line 234, in load_scan
    raise ValueError(

ValueError: No images were loaded for patient=00350, scan=FLAIR.


	 [[{{node PyFunc}}]]
	 [[IteratorGetNext]] [Op:__inference_multi_step_on_iterator_3238]

## === cell 11
def generate_predictions(model, test_dataset, test_df):
    preds = model.predict(test_dataset, verbose=0).reshape(-1)[: len(test_df)]

    submission = test_df.copy()
    submission["Label"] = preds.astype(np.float32)
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    sample_sub_path = TEST_DATASET_DF_DIR
    sample_sub = pd.read_csv(sample_sub_path)[["BraTS21ID"]]
    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    sample_sub["BraTS21ID"] = (
        sample_sub["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    submission = sample_sub.merge(submission, on="BraTS21ID", how="left")

    submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_dataset, test_df)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1204668977.py in <cell line: 0>()
     21 
     22 
---> 23 submission = generate_predictions(model, test_dataset, test_df)
     24 

NameError: name 'test_dataset' is not defined

## === cell 12
submission.info()
submission.head()



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4212896994.py in <cell line: 0>()
----> 1 submission.info()
      2 submission.head()
      3 

NameError: name 'submission' is not defined

## === cell 13
submission.to_csv("submission.csv", index=False)
print("Wrote:", os.path.abspath("submission.csv"))
print("Shape:", submission.shape)
print(submission.head(3).to_string(index=False))

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2391049210.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 print("Wrote:", os.path.abspath("submission.csv"))
      3 print("Shape:", submission.shape)
      4 print(submission.head(3).to_string(index=False))

NameError: name 'submission' is not defined
