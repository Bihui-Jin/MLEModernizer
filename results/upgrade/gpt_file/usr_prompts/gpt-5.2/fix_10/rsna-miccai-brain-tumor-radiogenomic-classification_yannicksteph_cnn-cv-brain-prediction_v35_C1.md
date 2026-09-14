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
import numpy as np
import pandas as pd

import pydicom
import cv2

from enum import Enum
from concurrent.futures import ThreadPoolExecutor

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

plot_model = None

os.environ["PYTHONHASHSEED"] = "123"
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

print("TensorFlow:", tf.__version__)

try:
    tf.config.threading.set_intra_op_parallelism_threads(2)
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
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
class ImageFormat(Enum):
    """
    Enum to represent image formats.

    - 'W-H-D-C' = Width-Height-Depth-Channel
    - 'D-W-H-C' = Depth-Width-Height-Channel
    """

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

        self.__executor = ThreadPoolExecutor(max_workers=self.__max_threads)

        self.__scan_cache = {}
        self.__file_list_cache = {}

        w, h = self.__size
        self.__zero_slice = np.zeros((w, h, 1), dtype=np.uint8)

    def __del__(self):
        try:
            if hasattr(self, "_DICOMLoader__executor") and self.__executor is not None:
                self.__executor.shutdown(wait=False, cancel_futures=False)
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

    def load_scan(self, row, scan_category, show_progress=True):
        self.__debug.log("== load_scan ==")
        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)

        cache_key = (patient_id, scan_category)
        cached = self.__scan_cache.get(cache_key, None)
        if cached is not None:
            return cached

        scans_path = os.path.join(self.__input_path, patient_id, scan_category)
        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        file_key = (patient_id, scan_category)
        image_files = self.__file_list_cache.get(file_key)
        if image_files is None:
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")), key=self.__image_file_sorter
            )
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")
            self.__file_list_cache[file_key] = image_files

        selected_files = self._select_subset_image_files(image_files)

        w, h = self.__size
        if self.__num_imgs is None:
            out_depth = len(selected_files) if selected_files else 1
        else:
            out_depth = self.__num_imgs

        vol = np.empty((out_depth, w, h, 1), dtype=np.uint8)

        loaded_count = 0
        for img in self.__executor.map(self._load_dicom_image, selected_files):
            if img is None:
                continue
            if loaded_count >= out_depth:
                break
            vol[loaded_count] = img
            loaded_count += 1

        if loaded_count == 0:
            vol[0] = self.__zero_slice
            loaded_count = 1

        if loaded_count < out_depth:
            vol[loaded_count:out_depth] = self.__zero_slice

        loaded_images = self.format(vol, "default")
        self.__scan_cache[cache_key] = loaded_images
        return loaded_images

    def load_all_scans(self, row, show_progress=True):
        self.__debug.log("== load_all_scans ==")
        return {
            scan_category: self.load_scan(row, scan_category, False)
            for scan_category in self.__scan_categories
        }

    def _load_dicom_image(self, dicom_path):
        self.__debug.log("== _load_dicom_image ==")
        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File {dicom_path} does not exist.")
        try:
            ds = pydicom.dcmread(dicom_path, force=True, stop_before_pixels=True)
            ds_full = pydicom.dcmread(dicom_path, force=True)
            try:
                from pydicom.pixels import pixel_array as px_pixel_array

                image = px_pixel_array(ds_full)
            except Exception:
                try:
                    image = ds_full.pixel_array
                except Exception:
                    image = None

            if image is None:
                return None

            if image.ndim > 2:
                image = image[..., 0]

        except Exception:
            return None

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
        if max_val == min_val:
            return np.zeros_like(image).astype(np.uint8)
        image = image - min_val
        image = image / (max_val - min_val)
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
            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )
            for j, (_, images) in enumerate(batch_x_image_paths.items()):
                batches_x[j].append(images)

        batch_x = tuple(np.array(b) for b in batches_x)
        batch_y = np.array(batches_y).astype(np.float32)

        if self.__is_trainable:
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))


class PreloadedScanDataset(Sequence):
    def __init__(self, X_list, y, batch_size, subset="train", shuffle=True):
        self.X_list = X_list  # list of 4 arrays: [N, W, H, D, C]
        self.y = y
        self.batch_size = batch_size
        self.is_trainable = subset.lower() in ["validation", "train"]
        self.shuffle = shuffle
        self.indices = np.arange(self.X_list[0].shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.X_list[0].shape[0] / self.batch_size))

    def __getitem__(self, ids):
        sl = slice(ids * self.batch_size, (ids + 1) * self.batch_size)
        idx = self.indices[sl]
        batch_x = tuple(X[idx] for X in self.X_list)
        if self.is_trainable:
            return batch_x, self.y[idx]
        return batch_x




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
os.makedirs(BEST_MODEL_PATH, exist_ok=True)
BEST_MODEL_H5_DIR = f"{BEST_MODEL_PATH}/model_{VERSION}.h5"

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

print("Paths OK. Train labels:", TRAIN_DATASET_DF_DIR)




## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
test_df["ID"] = test_df["ID"].astype(int)

print("Train rows:", len(train_df), "Test rows:", len(test_df))




## === cell 7
class DeepScanModel(Model):
    def __init__(self, input_shape, model_name="My3DCNNModel"):
        self.input_layers = [Input(shape=input_shape) for _ in range(4)]
        self.cnn_models = [self.build_cnn_branch(inp) for inp in self.input_layers]
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
rng = np.random.RandomState(SEED)
idx = np.arange(len(train_df))
rng.shuffle(idx)

val_frac = 0.2
val_size = int(len(idx) * val_frac)

val_idx = idx[:val_size]
trn_idx = idx[val_size:]

train_df_trn = train_df.iloc[trn_idx].reset_index(drop=True)
train_df_val = train_df.iloc[val_idx].reset_index(drop=True)

train_dicom_loader = DICOMLoader(
    train_df_trn,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=8,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
)

val_dicom_loader = DICOMLoader(
    train_df_val,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=8,
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
    max_threads=8,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
)

CACHE_DIR = os.path.join(RUN_DIR, "cache_np")
os.makedirs(CACHE_DIR, exist_ok=True)


def _cache_paths(prefix):
    return [os.path.join(CACHE_DIR, f"{prefix}_{cat}.npy") for cat in SCAN_CATEGORIES]


def _maybe_load_memmap(paths, expected_shape, dtype=np.uint8):
    for p in paths:
        if not os.path.exists(p):
            return None
    arrs = [np.load(p, mmap_mode="r") for p in paths]
    ok = all(tuple(a.shape) == tuple(expected_shape) and a.dtype == dtype for a in arrs)
    return arrs if ok else None


def _build_memmap(paths, shape, dtype=np.uint8):
    from numpy.lib.format import open_memmap

    return [open_memmap(p, mode="w+", dtype=dtype, shape=shape) for p in paths]


def _precompute_split(dicom_loader, prefix):
    n = dicom_loader.len
    shape = (n, IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN)
    paths = _cache_paths(prefix)
    cached = _maybe_load_memmap(paths, shape, dtype=np.uint8)
    if cached is not None:
        print(f"Loaded cached precomputed tensors for {prefix}: {n} samples")
        y = dicom_loader.df["Label"].values.astype(np.float32)
        return cached, y

    print(f"Precomputing tensors for {prefix}: {n} samples (one-time DICOM decode)...")
    mmaps = _build_memmap(paths, shape, dtype=np.uint8)
    y = dicom_loader.df["Label"].values.astype(np.float32)

    def load_one(i):
        scans = dicom_loader.load_all_scans(i, show_progress=False)
        return i, scans

    max_workers = min(12, (os.cpu_count() or 2))
    chunksize = 8
    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for i, scans in ex.map(load_one, range(n), chunksize=chunksize):
            for j, cat in enumerate(SCAN_CATEGORIES):
                mmaps[j][i] = scans[cat]

    for m in mmaps:
        m.flush()
    arrs = [np.load(p, mmap_mode="r") for p in paths]
    print(f"Done precomputing {prefix}.")
    return arrs, y


X_trn_list, y_trn = _precompute_split(train_dicom_loader, "train")
X_val_list, y_val = _precompute_split(val_dicom_loader, "val")
X_tst_list, y_tst_dummy = _precompute_split(test_dicom_loader, "test")


def _make_tf_dataset(X_list, y, batch_size, subset, shuffle, seed=SEED):
    X_tuple = tuple(X_list)
    if subset.lower() in ["validation", "train"]:
        ds = tf.data.Dataset.from_tensor_slices((X_tuple, y))
    else:
        ds = tf.data.Dataset.from_tensor_slices(X_tuple)

    options = tf.data.Options()
    options.deterministic = True
    ds = ds.with_options(options)

    if shuffle:
        ds = ds.shuffle(
            buffer_size=int(X_list[0].shape[0]),
            seed=seed,
            reshuffle_each_iteration=True,
        )

    ds = ds.batch(batch_size, drop_remainder=False)
    ds = ds.prefetch(tf.data.AUTOTUNE)
    return ds


train_dataset = _make_tf_dataset(
    X_trn_list, y_trn, BATCH_SIZE, DatasetType.TRAIN.value, shuffle=True
)
val_dataset = _make_tf_dataset(
    X_val_list, y_val, BATCH_SIZE, DatasetType.VALIDATION.value, shuffle=False
)
test_dataset = _make_tf_dataset(
    X_tst_list, y_tst_dummy, BATCH_SIZE, DatasetType.TEST.value, shuffle=False
)

print(
    "Batches - train:",
    int(np.ceil(X_trn_list[0].shape[0] / BATCH_SIZE)),
    "val:",
    int(np.ceil(X_val_list[0].shape[0] / BATCH_SIZE)),
    "test:",
    int(np.ceil(X_tst_list[0].shape[0] / BATCH_SIZE)),
)




## === cell 9
model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

os.makedirs(os.path.dirname(BEST_MODEL_H5_DIR), exist_ok=True)

checkpoint = ModelCheckpoint(
    filepath=BEST_MODEL_H5_DIR,
    monitor=TF_CALL_BACK_BEST_MODEL_MONITOR,
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    callbacks=[checkpoint],
    verbose=2,
)

if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )
print("Model ready:", model.name)




## === cell 10
def generate_predictions(
    model, test_dataset, test_df, submission_template_path=TEST_DATASET_DF_DIR
):
    preds = model.predict(test_dataset, verbose=0).reshape(-1)[: len(test_df)]

    template = pd.read_csv(submission_template_path)
    template.rename(columns={"BraTS21ID": "ID"}, inplace=True)
    template["ID"] = template["ID"].astype(int)

    pred_df = pd.DataFrame(
        {"ID": test_df["ID"].values, "pred": preds.astype(np.float32)}
    )
    merged = template[["ID"]].merge(pred_df, on="ID", how="left")

    merged["pred"] = merged["pred"].fillna(0.5).clip(0.0, 1.0)

    submission = merged.rename(columns={"ID": "BraTS21ID", "pred": "MGMT_value"})
    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    submission = submission[["BraTS21ID", "MGMT_value"]]
    return submission


submission = generate_predictions(model, test_dataset, test_df)
print(submission.head())




## === cell 11
submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", submission.shape)
print(submission.dtypes)
print(submission.head())
