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
import glob
import random
from enum import Enum

import numpy as np
import pandas as pd

import cv2

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import Model
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

try:
    tf.config.threading.set_intra_op_parallelism_threads(
        max(1, (os.cpu_count() or 2) // 2)
    )
    tf.config.threading.set_inter_op_parallelism_threads(2)
except Exception:
    pass

print("TF version:", tf.__version__)




## === cell 1
from concurrent.futures import ThreadPoolExecutor, as_completed

try:
    from tqdm import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


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
    WHDC = "W-H-D-C"  # (W, H, D, C)
    DWHC = "D-W-H-C"  # (D, W, H, C)

    @staticmethod
    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.WHDC:
            return np.transpose(image, (1, 2, 0, 3))  # (W,H,D,C)
        elif image_format == ImageFormat.DWHC:
            return image  # already (D,W,H,C)
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
        image_format=ImageFormat.DWHC,  # ensure consistent (D,W,H,C) default
        max_threads=8,
        image_file_sorter=lambda x: int(os.path.basename(x)[:-4].split("-")[-1]),
        debug_mode=False,
        enable_cache=True,
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

        self.__enable_cache = bool(enable_cache)
        self.__cache = {}
        self.__cache_hits = 0
        self.__cache_misses = 0

        self.__ids_arr = self.__df[self.__id_column_name].astype(int).to_numpy()
        self.__labels_arr = self.__df[self.__label_column_name].to_numpy(
            dtype=np.float32
        )

        self.__file_index = self._build_file_index()
        self.__rot_mat_cache = {}
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
        return self.__ids_arr[row]

    def gel_label(self, row):
        return self.__labels_arr[row]

    def format(self, images, type):
        if type == "default":
            return ImageFormat.swap_dimensions(images, self.image_format)
        else:
            return images

    def _cache_key(self, row, scan_category):
        patient_id = str(int(self.__ids_arr[row])).zfill(5)
        return patient_id, scan_category

    def _build_file_index(self):
        index = {}
        try:
            with os.scandir(self.__input_path) as it:
                patient_dirs = [e.name for e in it if e.is_dir()]
        except Exception:
            patient_dirs = [
                os.path.basename(p)
                for p in glob.glob(os.path.join(self.__input_path, "*"))
                if os.path.isdir(p)
            ]

        df_patients = set(str(int(x)).zfill(5) for x in self.__ids_arr.tolist())
        patient_dirs = [p for p in patient_dirs if p in df_patients]

        for patient_id in patient_dirs:
            p_dir = os.path.join(self.__input_path, patient_id)
            for scan_category in self.__scan_categories:
                s_dir = os.path.join(p_dir, scan_category)
                if not os.path.exists(s_dir):
                    continue
                entries = []
                try:
                    with os.scandir(s_dir) as it2:
                        for e in it2:
                            if e.is_file():
                                entries.append(e.path)
                except Exception:
                    entries = glob.glob(os.path.join(s_dir, "*"))
                if not entries:
                    continue
                entries.sort(key=self.__image_file_sorter)
                index[(patient_id, scan_category)] = entries
        return index

    def load_scan(self, row, scan_category, show_progress=True):
        self.__debug.log("== load_scan ==")

        if self.__enable_cache:
            key = self._cache_key(row, scan_category)
            cached = self.__cache.get(key)
            if cached is not None:
                self.__cache_hits += 1
                return cached
            self.__cache_misses += 1

        patient_id = str(int(self.__ids_arr[row])).zfill(5)
        scans_path = os.path.join(self.__input_path, patient_id, scan_category)

        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        image_files = self.__file_index.get((patient_id, scan_category))
        if image_files is None:
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")),
                key=self.__image_file_sorter,
            )

        if not image_files:
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
            raise ValueError("No images were loaded. Cannot proceed.")

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                loaded_images.append(np.zeros_like(loaded_images[0]))

        loaded_images = np.asarray(loaded_images, dtype=np.uint8)  # (D, W, H, C)
        out = self.format(loaded_images, "default")

        if self.__enable_cache:
            self.__cache[self._cache_key(row, scan_category)] = out

        return out

    def load_all_scans(self, row, show_progress=True):
        all_images = {}
        for scan_category in self.__scan_categories:
            all_images[scan_category] = self.load_scan(
                row, scan_category, show_progress=False
            )
        return {key: all_images.get(key, []) for key in self.__scan_categories}

    def _load_dicom_image(self, dicom_path):
        image = cv2.imread(dicom_path, cv2.IMREAD_UNCHANGED)
        if image is None:
            image = np.zeros(self.__size[::-1], dtype=np.uint8)

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
        key = (height, width, self.__rotate_angle)
        rot = self.__rot_mat_cache.get(key)
        if rot is None:
            center = (width / 2, height / 2)
            rot = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)
            self.__rot_mat_cache[key] = rot
        return cv2.warpAffine(image, rot, (width, height))

    def _normalization_img(self, image):
        max_val = int(image.max(initial=0))
        if max_val == 0:
            return np.zeros_like(image, dtype=np.uint8)
        min_val = int(image.min(initial=0))
        img = image.astype(np.float32, copy=False)
        img = (img - float(min_val)) / float(max_val)
        return (img * 255.0).astype(np.uint8)

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

        batches_y = []
        batches_x = [[] for _ in range(len(self.__dicom_loader.scan_categories))]

        for i in batch_indices:
            if self.__is_trainable:
                label = self.__dicom_loader.gel_label(i)
                batches_y.append(label)

            batch_x_images = self.__dicom_loader.load_all_scans(i, show_progress=False)
            for j, (_, images) in enumerate(batch_x_images.items()):
                batches_x[j].append(images)

        batch_x = tuple(np.asarray(b, dtype=np.uint8) for b in batches_x)

        if self.__is_trainable:
            batch_y = np.asarray(batches_y, dtype=np.float32)
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))


class InMemoryScanDataset(Sequence):
    def __init__(self, X_tuple, y, batch_size, subset="train", shuffle=True):
        self.X = X_tuple
        self.y = y
        self.batch_size = batch_size
        self.is_trainable = subset.lower() in ["validation", "train"]
        self.shuffle = shuffle
        self.indices = np.arange(self.X[0].shape[0])
        if self.shuffle:
            np.random.shuffle(self.indices)

    def on_epoch_end(self):
        if self.shuffle:
            np.random.shuffle(self.indices)

    def __len__(self):
        return int(np.ceil(self.X[0].shape[0] / self.batch_size))

    def __getitem__(self, ids):
        from_id = ids * self.batch_size
        to_id = (ids + 1) * self.batch_size
        batch_idx = self.indices[from_id:to_id]
        batch_x = tuple(x[batch_idx] for x in self.X)
        if self.is_trainable:
            return batch_x, self.y[batch_idx]
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
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

BEST_MODEL_FILE = f"model_{VERSION}.keras"
BEST_MODEL_DIR = os.path.join(BEST_MODEL_PATH, BEST_MODEL_FILE)

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ = 32
IMG_SCALE = 0.90
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

print(
    "Paths OK:",
    os.path.exists(TRAIN_DATASET_DF_DIR),
    os.path.exists(TEST_DATASET_DF_DIR),
)




## === cell 5
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
test_df["ID"] = test_df["ID"].astype(int)

print("Train shape:", train_df.shape, "Test shape:", test_df.shape)




## === cell 6
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




## === cell 7
val_fraction = 0.2
idx = np.arange(len(train_df))
rng = np.random.RandomState(SEED)
rng.shuffle(idx)

val_size = int(len(idx) * val_fraction)
val_idx = idx[:val_size]
trn_idx = idx[val_size:]

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
    enable_cache=False,
    image_format=ImageFormat.DWHC,
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
    enable_cache=False,
    image_format=ImageFormat.DWHC,
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
    enable_cache=False,
    image_format=ImageFormat.DWHC,
)


def _precompute_all(loader: DICOMLoader, df: pd.DataFrame, subset_name: str):
    n = loader.len
    X = [
        np.empty(
            (n, IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN), dtype=np.float32
        )
        for _ in range(4)
    ]

    y = None
    if "Label" in df.columns:
        y = df["Label"].to_numpy(dtype=np.float32)

    def _load_one_patient(i: int):
        scans = loader.load_all_scans(i, show_progress=False)
        out_whdc = []
        for cat in SCAN_CATEGORIES:
            arr_dwhc = scans[cat]  # (D,W,H,C) because image_format=DWHC
            if arr_dwhc.shape != (IMG_SEQ, IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN):
                raise ValueError(
                    f"Unexpected scan shape for {subset_name} idx={i} cat={cat}: {arr_dwhc.shape}"
                )
            out_whdc.append(arr_dwhc.transpose(1, 2, 0, 3))  # (W,H,D,C)
        return i, out_whdc

    patient_workers = min(4, max(1, (os.cpu_count() or 2) // 2))
    with ThreadPoolExecutor(max_workers=patient_workers) as ex:
        mapped = ex.map(_load_one_patient, range(n), chunksize=1)
        mapped = tqdm(mapped, total=n, desc=f"Precomputing {subset_name}")
        for i, out_whdc in mapped:
            X[0][i, ...] = out_whdc[0].astype(np.float32, copy=False)
            X[1][i, ...] = out_whdc[1].astype(np.float32, copy=False)
            X[2][i, ...] = out_whdc[2].astype(np.float32, copy=False)
            X[3][i, ...] = out_whdc[3].astype(np.float32, copy=False)

    return tuple(X), y


X_train, y_train = _precompute_all(train_dicom_loader, train_split_df, "train")
X_val, y_val = _precompute_all(val_dicom_loader, val_split_df, "val")
X_test, _ = _precompute_all(test_dicom_loader, test_df, "test")

train_dataset = InMemoryScanDataset(
    X_tuple=X_train,
    y=y_train,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=SHUFFLE,
)

val_dataset = InMemoryScanDataset(
    X_tuple=X_val,
    y=y_val,
    batch_size=BATCH_SIZE,
    subset=DatasetType.VALIDATION.value,
    shuffle=False,
)

test_dataset = InMemoryScanDataset(
    X_tuple=X_test,
    y=None,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
)

print("Datasets:", len(train_dataset), len(val_dataset), len(test_dataset))




## === cell 8
model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

checkpoint_cb = ModelCheckpoint(
    filepath=BEST_MODEL_DIR,
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

if os.path.exists(BEST_MODEL_DIR):
    best_model = tf.keras.models.load_model(
        BEST_MODEL_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )
else:
    best_model = model

best_model.compile(
    optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS
)

print("Using model for inference:", getattr(best_model, "name", "model"))




## === cell 9
def generate_predictions(model, test_dataset, test_df):
    preds = model.predict(
        test_dataset,
        verbose=0,
    ).reshape(
        -1
    )[: len(test_df)]

    submission = test_df.copy()
    submission["Label"] = preds.astype(np.float32)
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).map(lambda x: f"{x:05d}")
    )
    submission = submission[["BraTS21ID", "MGMT_value"]]

    return submission


submission = generate_predictions(best_model, test_dataset, test_df)

print(submission.head())
print(submission.shape)




## === cell 10
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote:", os.path.abspath(SUBMISSION_DATASET_DF_DIR))
print(pd.read_csv(SUBMISSION_DATASET_DF_DIR).head())
