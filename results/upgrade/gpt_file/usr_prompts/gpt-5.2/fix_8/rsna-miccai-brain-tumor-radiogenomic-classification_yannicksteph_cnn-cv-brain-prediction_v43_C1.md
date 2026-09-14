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
                    "No images were loaded, and num_imgs is set. Cannot proceed."
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
                raise FileNotFoundError(f"File {dicom_path} does not exist.")

            key = self._cache_key_for_slice(dicom_path)
            cached = self._get_cached_slice(key)
            if cached is not None:
                return cached

            try:
                dicom_file = pydicom.dcmread(
                    dicom_path,
                    stop_before_pixels=False,
                    specific_tags=["PixelData"],
                    force=True,
                )
            except Exception as e:
                raise IOError(f"An error occurred while reading the DICOM file: {e}")

            image = dicom_file.pixel_array
            image = self._rotate_img(image)
            image = self._normalization_img(image)
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
def _precompute_volumes_to_memmap(
    dicom_loader, df, out_dir, scan_categories, input_shape, seed
):
    os.makedirs(out_dir, exist_ok=True)
    n = len(df)
    x_paths = []
    for sc in scan_categories:
        x_paths.append(os.path.join(out_dir, f"x_{sc}.npy"))
    y_path = os.path.join(out_dir, "y.npy")
    id_path = os.path.join(out_dir, "ids.npy")

    expected_vol_shape = (
        input_shape[0],
        input_shape[1],
        input_shape[2],
        input_shape[3],
    )

    meta_path = os.path.join(out_dir, "meta.npz")
    meta = dict(
        n=n,
        scan_categories=np.array(scan_categories),
        vol_shape=np.array(expected_vol_shape),
        ids=np.array(df["ID"].to_numpy()),
        labels=np.array(df["Label"].to_numpy(), dtype=np.float32),
        seed=int(seed),
    )
    if os.path.exists(meta_path) and all(
        os.path.exists(p) for p in x_paths + [y_path, id_path]
    ):
        try:
            old = np.load(meta_path, allow_pickle=True)
            if (
                int(old["n"]) == n
                and tuple(old["vol_shape"]) == tuple(expected_vol_shape)
                and list(old["scan_categories"]) == list(scan_categories)
                and np.array_equal(old["ids"], meta["ids"])
            ):
                return x_paths, y_path, id_path
        except Exception:
            pass

    xs_mm = [
        np.lib.format.open_memmap(
            p, mode="w+", dtype=np.uint8, shape=(n,) + expected_vol_shape
        )
        for p in x_paths
    ]
    y_mm = np.lib.format.open_memmap(y_path, mode="w+", dtype=np.float32, shape=(n,))
    ids_mm = np.lib.format.open_memmap(id_path, mode="w+", dtype=np.int32, shape=(n,))

    for i in range(n):
        y_mm[i] = float(df.loc[i, "Label"])
        ids_mm[i] = int(df.loc[i, "ID"])
        scans = dicom_loader.load_all_scans(i, show_progress=False)
        for j, sc in enumerate(scan_categories):
            vol = scans[sc]
            if vol.shape != expected_vol_shape:
                raise ValueError(
                    f"Unexpected volume shape for {sc}: got {vol.shape}, expected {expected_vol_shape}"
                )
            xs_mm[j][i] = vol

    for mm in xs_mm:
        mm.flush()
    y_mm.flush()
    ids_mm.flush()
    np.savez_compressed(meta_path, **meta)
    return x_paths, y_path, id_path


def _make_tf_dataset_from_memmap(
    x_paths, y_path, batch_size, shuffle, seed, is_trainable, scan_categories
):
    xs = [np.load(p, mmap_mode="r") for p in x_paths]
    y = np.load(y_path, mmap_mode="r") if is_trainable else None
    n = xs[0].shape[0]

    def gen():
        idxs = np.arange(n)
        if shuffle:
            rng = np.random.RandomState(seed)
            rng.shuffle(idxs)
        for i in idxs:
            x_tuple = tuple(xs[j][i] for j in range(len(scan_categories)))
            if is_trainable:
                yield x_tuple, np.float32(y[i])
            else:
                yield x_tuple

    input_spec = tuple(
        tf.TensorSpec(shape=xs[j].shape[1:], dtype=tf.uint8)
        for j in range(len(scan_categories))
    )
    if is_trainable:
        output_signature = (input_spec, tf.TensorSpec(shape=(), dtype=tf.float32))
    else:
        output_signature = input_spec

    ds = tf.data.Dataset.from_generator(gen, output_signature=output_signature)

    def _cast_x(x, y=None):
        x = tuple(tf.cast(t, tf.float32) for t in x)
        if is_trainable:
            return x, y
        return x

    ds = ds.map(_cast_x, num_parallel_calls=tf.data.AUTOTUNE)
    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds, n


CACHE_DIR = os.path.join(RUN_DIR, "cache_volumes_v1")

train_x_paths, train_y_path, _ = _precompute_volumes_to_memmap(
    train_dicom_loader,
    train_split_df,
    out_dir=os.path.join(CACHE_DIR, "train"),
    scan_categories=SCAN_CATEGORIES,
    input_shape=INPUT_SHAPE,
    seed=SEED,
)
val_x_paths, val_y_path, _ = _precompute_volumes_to_memmap(
    val_dicom_loader,
    val_split_df,
    out_dir=os.path.join(CACHE_DIR, "val"),
    scan_categories=SCAN_CATEGORIES,
    input_shape=INPUT_SHAPE,
    seed=SEED,
)
test_x_paths, test_y_path, _ = _precompute_volumes_to_memmap(
    test_dicom_loader,
    test_df,
    out_dir=os.path.join(CACHE_DIR, "test"),
    scan_categories=SCAN_CATEGORIES,
    input_shape=INPUT_SHAPE,
    seed=SEED,
)

train_dataset, _n_train = _make_tf_dataset_from_memmap(
    train_x_paths,
    train_y_path,
    batch_size=BATCH_SIZE,
    shuffle=SHUFFLE,
    seed=SEED,
    is_trainable=True,
    scan_categories=SCAN_CATEGORIES,
)
val_dataset, _n_val = _make_tf_dataset_from_memmap(
    val_x_paths,
    val_y_path,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    is_trainable=True,
    scan_categories=SCAN_CATEGORIES,
)
test_dataset, _n_test = _make_tf_dataset_from_memmap(
    test_x_paths,
    test_y_path,
    batch_size=BATCH_SIZE,
    shuffle=False,
    seed=SEED,
    is_trainable=False,
    scan_categories=SCAN_CATEGORIES,
)




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/3023218981.py in <cell line: 0>()
    126 CACHE_DIR = os.path.join(RUN_DIR, "cache_volumes_v1")
    127 
--> 128 train_x_paths, train_y_path, _ = _precompute_volumes_to_memmap(
    129     train_dicom_loader,
    130     train_split_df,

/tmp/ipykernel_11/3023218981.py in _precompute_volumes_to_memmap(dicom_loader, df, out_dir, scan_categories, input_shape, seed)
     61         y_mm[i] = float(df.loc[i, "Label"])
     62         ids_mm[i] = int(df.loc[i, "ID"])
---> 63         scans = dicom_loader.load_all_scans(i, show_progress=False)
     64         # Ensure consistent modality order
     65         for j, sc in enumerate(scan_categories):

/tmp/ipykernel_11/707884816.py in load_all_scans(self, row, show_progress)
    255                 for scan_category in self.__scan_categories
    256             }
--> 257             all_images = {
    258                 scan_category: futures[scan_category].result()
    259                 for scan_category in self.__scan_categories

/tmp/ipykernel_11/707884816.py in <dictcomp>(.0)
    256             }
    257             all_images = {
--> 258                 scan_category: futures[scan_category].result()
    259                 for scan_category in self.__scan_categories
    260             }

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/707884816.py in load_scan(self, row, scan_category, show_progress)
    226             loaded_images = []
    227             loaded_images_extend = loaded_images.append
--> 228             for image_data in self.__executor.map(self._load_dicom_image, image_files):
    229                 if image_data is not None:
    230                     loaded_images_extend(image_data)

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    617                     # Careful not to keep a reference to the popped future
    618                     if timeout is None:
--> 619                         yield _result_or_cancel(fs.pop())
    620                     else:
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    315     try:
    316         try:
--> 317             return fut.result(timeout)
    318         finally:
    319             fut.cancel()

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    454                     raise CancelledError()
    455                 elif self._state == FINISHED:
--> 456                     return self.__get_result()
    457                 else:
    458                     raise TimeoutError()

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    399         if self._exception:
    400             try:
--> 401                 raise self._exception
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/707884816.py in _load_dicom_image(self, dicom_path)
    284                 raise IOError(f"An error occurred while reading the DICOM file: {e}")
    285 
--> 286             image = dicom_file.pixel_array
    287             image = self._rotate_img(image)
    288             image = self._normalization_img(image)

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
    988 
    989         if validate:
--> 990             runner.validate()
    991 
    992         if self.is_native:

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in validate(self)
    751     def validate(self) -> None:
    752         """Validate the decoding options and source buffer (if any)."""
--> 753         self._validate_options()
    754         if self.is_dataset or self.is_buffer:
    755             self._validate_buffer()

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/decoders/base.py in _validate_options(self)
    820     def _validate_options(self) -> None:
    821         """Validate the supplied options to ensure they meet minimum requirements."""
--> 822         super()._validate_options()
    823 
    824         # The Extended Offset Table is optional

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/common.py in _validate_options(self)
    573         prefix = "Missing required element: (0028"
    574         if self._opts.get("bits_allocated") is None:
--> 575             raise AttributeError(f"{prefix},0100) 'Bits Allocated'")
    576 
    577         if not 1 <= self.bits_allocated <= 64 or (

AttributeError: Missing required element: (0028,0100) 'Bits Allocated'

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
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2280289531.py in <cell line: 0>()
     21 
     22 history = model.fit(
---> 23     train_dataset,
     24     validation_data=val_dataset,
     25     epochs=EPOCHS,

NameError: name 'train_dataset' is not defined

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
