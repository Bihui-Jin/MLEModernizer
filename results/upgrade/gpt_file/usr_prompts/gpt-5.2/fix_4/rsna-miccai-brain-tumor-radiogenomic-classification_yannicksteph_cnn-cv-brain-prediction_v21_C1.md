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

0.542835288118307

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.46353) has done: 'I fix the immediate import/runtime failure by removing the incompatible `keras.utils.vis_utils` import that triggers the protobuf `MessageFactory.GetPrototype` error in this environment, while keeping TensorFlow/Keras usage intact. I also remove the hard dependency on an external pre-trained model file that doesn’t exist (`/kaggle/input/model-test/model_V1.h5`) by training the provided `DeepScanModel` on the competition training set and then using it for test inference (same architecture/loss/optimizer/metric). To ensure the pipeline finishes and produces a valid `.csv`, I add a small, deterministic train/validation split, compile+fit, and then write `submission.csv` with the exact required columns. These changes are directly to unblock execution and produce a legitimate AUC-based probability submission.'

# 9. Code solution

## === cell 0
import os

os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", "2")

import glob
import random
import numpy as np
import pandas as pd

import pydicom
from enum import Enum

import cv2

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
    GlobalAveragePooling3D,
    Dense,
    Dropout,
    ReLU,
    concatenate,
)

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(max(1, os.cpu_count() // 2))
except Exception:
    pass




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
from concurrent.futures import ThreadPoolExecutor, as_completed


class ImageFormat(Enum):
    WHDC = "W-H-D-C"  # (W, H, D, C)
    DWHC = "D-W-H-C"  # (D, W, H, C)

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
        cache_dir=None,
        use_cache=True,
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

        self.__use_cache = bool(use_cache)
        self.__cache_dir = cache_dir
        if self.__use_cache and self.__cache_dir is not None:
            os.makedirs(self.__cache_dir, exist_ok=True)

        self.__executor = ThreadPoolExecutor(max_workers=self.__max_threads)

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

    def _cache_key(self, row, scan_category):
        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
        w, h = self.__size
        return (
            f"id={patient_id}_scan={scan_category}_n={self.__num_imgs}_"
            f"size={w}x{h}_scale={self.__scale}_rot={self.__rotate_angle}_"
            f"center={int(self.__enable_center_focus)}_fmt={self.__image_format.value}.npy"
        )

    def _cache_path(self, row, scan_category):
        if self.__cache_dir is None:
            return None
        return os.path.join(self.__cache_dir, self._cache_key(row, scan_category))

    def load_scan(self, row, scan_category, show_progress=False):
        self.__debug.log("== load_scan ==")

        cache_path = self._cache_path(row, scan_category) if self.__use_cache else None
        if cache_path is not None and os.path.exists(cache_path):
            loaded_images = np.load(cache_path, allow_pickle=False)
            return self.format(loaded_images, "default")

        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
        scans_path = os.path.join(self.__input_path, patient_id, scan_category)

        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        try:
            image_files = [e.path for e in os.scandir(scans_path) if e.is_file()]
        except FileNotFoundError:
            image_files = []

        image_files = sorted(image_files, key=self.__image_file_sorter)

        if not image_files:
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)
        loaded_images = []

        image_data_iterable = self.__executor.map(self._load_dicom_image, image_files)
        for image_data in image_data_iterable:
            if image_data is not None:
                loaded_images.append(image_data)

        if not loaded_images:
            raise ValueError("No images were loaded. Cannot proceed.")

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                zero_image = np.zeros_like(loaded_images[0])
                loaded_images.append(zero_image)

        loaded_images = np.array(loaded_images)

        if cache_path is not None:
            tmp_path = cache_path + ".tmp"
            np.save(tmp_path, loaded_images)
            os.replace(tmp_path, cache_path)

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

    def _load_dicom_image(self, dicom_path):
        self.__debug.log("== _load_dicom_image ==")
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File {dicom_path} does not exist.")
        try:
            dicom_file = pydicom.dcmread(dicom_path)
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
        min_val = np.min(image)
        max_val = np.max(image)
        if max_val == 0:
            return np.zeros_like(image).astype(np.uint8)
        image = image - min_val
        image = image / max_val
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
        self, dicom_loader, batch_size, subset="train", shuffle=True, debug_mode=False
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

        batches_y = []
        batches_x = [[] for _ in range(len(self.__dicom_loader.scan_categories))]

        for i in batch_indices:
            label = self.__dicom_loader.gel_label(i)
            batches_y.append(label)

            batch_x_scans = self.__dicom_loader.load_all_scans(i, show_progress=False)
            for j, (_, images) in enumerate(batch_x_scans.items()):
                batches_x[j].append(images)

        batch_x = tuple(np.array(b, dtype=np.float32) for b in batches_x)
        batch_y = np.array(batches_y, dtype=np.float32)

        if self.__is_trainable:
            return batch_x, batch_y
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
BEST_MODEL_H5_DIR = f"{BEST_MODEL_PATH}/model_{VERSION}.h5"

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 0.90
IMG_ROTATE = 0
IMG_ENABLE_CENTRAL_FOCUS = True

BATCH_SIZE = 8
EPOCHS = 26

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)
COMPILE_LOSS = "binary_crossentropy"
COMPILE_METRICS = [AUC(name="auc")]

DICOM_CACHE_DIR = os.path.join("/kaggle/working", "dicom_cache_v1")




## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

val_frac = 0.2
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
val_size = int(len(train_df) * val_frac)
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_split_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

os.makedirs(BEST_MODEL_PATH, exist_ok=True)
os.makedirs(DICOM_CACHE_DIR, exist_ok=True)




## === cell 7
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




## === cell 8
train_cache = os.path.join(DICOM_CACHE_DIR, "train")
val_cache = os.path.join(DICOM_CACHE_DIR, "val")
test_cache = os.path.join(DICOM_CACHE_DIR, "test")
os.makedirs(train_cache, exist_ok=True)
os.makedirs(val_cache, exist_ok=True)
os.makedirs(test_cache, exist_ok=True)

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
    cache_dir=train_cache,
    use_cache=True,
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
    cache_dir=val_cache,
    use_cache=True,
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
    cache_dir=test_cache,
    use_cache=True,
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

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)




## === cell 9
INPUT_SHAPE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN)
model = DeepScanModel(input_shape=INPUT_SHAPE, model_name="Mult3DCNN4Input")

model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

checkpoint_cb = ModelCheckpoint(
    BEST_MODEL_H5_DIR,
    monitor="val_auc",
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint_cb],
    verbose=2,
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




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4211740682.py in <cell line: 0>()
     16 # Using workers>0 with Sequence introduces multiprocessing overhead and potential nondeterminism;
     17 # we keep default (single worker) for determinism, but caching makes it fast enough.
---> 18 history = model.fit(
     19     train_dataset,
     20     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/3351072055.py in __getitem__(self, ids)
     28             batches_y.append(label)
     29 
---> 30             batch_x_scans = self.__dicom_loader.load_all_scans(i, show_progress=False)
     31             for j, (_, images) in enumerate(batch_x_scans.items()):
     32                 batches_x[j].append(images)

/tmp/ipykernel_11/594569745.py in load_all_scans(self, row, show_progress)
    186         for future in as_completed(future_to_scan_category):
    187             scan_category = future_to_scan_category[future]
--> 188             image_data = future.result()
    189             if image_data is not None:
    190                 all_images[scan_category] = image_data

/usr/lib/python3.11/concurrent/futures/_base.py in result(self, timeout)
    459         finally:
    460             # Break a reference cycle with the exception in self._exception
--> 461             self = None
    462 
    463     def exception(self, timeout=None):

/usr/lib/python3.11/concurrent/futures/_base.py in __get_result(self)
    402             finally:
    403                 # Break a reference cycle with the exception in self._exception
--> 404                 self = None
    405         else:
    406             return self._result

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     60             self.future.set_exception(exc)
     61             # Break a reference cycle with the exception 'exc'
---> 62             self = None
     63         else:
     64             self.future.set_result(result)

/tmp/ipykernel_11/594569745.py in load_scan(self, row, scan_category, show_progress)
    168             tmp_path = cache_path + ".tmp"
    169             np.save(tmp_path, loaded_images)
--> 170             os.replace(tmp_path, cache_path)
    171 
    172         return self.format(loaded_images, "default")

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/dicom_cache_v1/train/id=00400_scan=T2w_n=32_size=128x128_scale=0.9_rot=0_center=1_fmt=W-H-D-C.npy.tmp' -> '/kaggle/working/dicom_cache_v1/train/id=00400_scan=T2w_n=32_size=128x128_scale=0.9_rot=0_center=1_fmt=W-H-D-C.npy'

## === cell 10
def generate_predictions(model, test_dataset, test_df):
    preds = []
    for batch_idx in range(len(test_dataset)):
        scan_type_1, scan_type_2, scan_type_3, scan_type_4 = test_dataset[batch_idx]
        batch_pred = model.predict(
            [scan_type_1, scan_type_2, scan_type_3, scan_type_4], verbose=0
        )
        preds.append(batch_pred.reshape(-1))

    preds = np.concatenate(preds, axis=0)[: len(test_df)]
    submission = test_df.copy()
    submission["Label"] = preds.astype(np.float32)
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission = submission[["BraTS21ID", "MGMT_value"]]
    return submission


submission = generate_predictions(model, test_dataset, test_df)
submission.head()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1576121163.py in <cell line: 0>()
     17 
     18 
---> 19 submission = generate_predictions(model, test_dataset, test_df)
     20 submission.head()
     21 

/tmp/ipykernel_11/1576121163.py in generate_predictions(model, test_dataset, test_df)
      2     preds = []
      3     for batch_idx in range(len(test_dataset)):
----> 4         scan_type_1, scan_type_2, scan_type_3, scan_type_4 = test_dataset[batch_idx]
      5         batch_pred = model.predict(
      6             [scan_type_1, scan_type_2, scan_type_3, scan_type_4], verbose=0

/tmp/ipykernel_11/3351072055.py in __getitem__(self, ids)
     28             batches_y.append(label)
     29 
---> 30             batch_x_scans = self.__dicom_loader.load_all_scans(i, show_progress=False)
     31             for j, (_, images) in enumerate(batch_x_scans.items()):
     32                 batches_x[j].append(images)

/tmp/ipykernel_11/594569745.py in load_all_scans(self, row, show_progress)
    186         for future in as_completed(future_to_scan_category):
    187             scan_category = future_to_scan_category[future]
--> 188             image_data = future.result()
    189             if image_data is not None:
    190                 all_images[scan_category] = image_data

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

/usr/lib/python3.11/concurrent/futures/thread.py in run(self)
     56 
     57         try:
---> 58             result = self.fn(*self.args, **self.kwargs)
     59         except BaseException as exc:
     60             self.future.set_exception(exc)

/tmp/ipykernel_11/594569745.py in load_scan(self, row, scan_category, show_progress)
    168             tmp_path = cache_path + ".tmp"
    169             np.save(tmp_path, loaded_images)
--> 170             os.replace(tmp_path, cache_path)
    171 
    172         return self.format(loaded_images, "default")

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/dicom_cache_v1/test/id=00356_scan=T2w_n=32_size=128x128_scale=0.9_rot=0_center=1_fmt=W-H-D-C.npy.tmp' -> '/kaggle/working/dicom_cache_v1/test/id=00356_scan=T2w_n=32_size=128x128_scale=0.9_rot=0_center=1_fmt=W-H-D-C.npy'

## === cell 11
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote:", SUBMISSION_DATASET_DF_DIR)
print(submission.info())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3363072990.py in <cell line: 0>()
----> 1 submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
      2 print("Wrote:", SUBMISSION_DATASET_DF_DIR)
      3 print(submission.info())

NameError: name 'submission' is not defined
