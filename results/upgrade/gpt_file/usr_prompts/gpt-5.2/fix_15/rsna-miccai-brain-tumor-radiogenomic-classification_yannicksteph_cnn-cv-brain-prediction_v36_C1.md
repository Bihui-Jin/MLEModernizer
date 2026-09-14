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

0.5568587455379909

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46706) has done: 'I fix the import/runtime crash by removing the incompatible `keras` (standalone) visualization import that triggers the protobuf `MessageFactory.GetPrototype` error, and rely only on `tf.keras`. Then I fix the missing external model file issue by training the existing `DeepScanModel` in-notebook (same architecture/loss/optimizer) and saving/loading from the defined `BEST_MODEL_H5_DIR` so the pipeline is self-contained. Finally, I ensure the test dataset doesn’t try to fetch labels (it currently does, because the sample submission’s `MGMT_value` column exists but is not a real label) and write a valid `submission.csv` with the required columns and ID formatting.'
- What this solution (achieved 0.46706) has done: 'The timeout is dominated by repeated DICOM decode + OpenCV preprocessing inside the `Sequence` generator, plus per-scan thread pools being created for every patient and repeated pandas `.loc` calls. I keep the exact same data, preprocessing steps, model, and training loop, but (1) add fast caching/lookup of IDs/labels, (2) avoid rebuilding thread pools per sample by reusing a single executor, (3) speed up DICOM reading by skipping unnecessary parsing (`stop_before_pixels=False` but `defer_size` and minimal tags) while preserving pixel decode, and (4) enable Keras multiprocessing prefetch for the existing `Sequence` (same batches/semantics) to overlap CPU DICOM work with GPU/TF execution. These changes are correctness-preserving (same slices selected, same transforms, same model inputs) and target the main constant-factor bottlenecks to fit within 600 seconds.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import random
from enum import Enum
import hashlib

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

os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")
tf.random.set_seed(123)
np.random.seed(123)
random.seed(123)

try:
    tf.config.threading.set_inter_op_parallelism_threads(2)
    tf.config.threading.set_intra_op_parallelism_threads(2)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
try:
    from concurrent.futures import ThreadPoolExecutor, as_completed
except ImportError:
    ThreadPoolExecutor = None
    as_completed = None

try:
    from tqdm.auto import tqdm
except Exception:

    def tqdm(x, **kwargs):
        return x


class ImageFormat(Enum):
    WHDC = "W-H-D-C"  # (W, H, D, C)
    DWHC = "D-W-H-C"  # (D, W, H, C)

    @staticmethod
    def swap_dimensions(image, image_format):
        if image_format == ImageFormat.DWHC:
            return np.transpose(image, (2, 0, 1, 3))
        elif image_format == ImageFormat.WHDC:
            return np.transpose(image, (1, 2, 0, 3))
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
        cache_version="v1",
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
        self.__scan_categories = list(scan_categories)
        self.__max_threads = max_threads
        self.__size = size
        self.__scale = scale
        self.__rotate_angle = rotate_angle
        self.__image_format = image_format
        self.__image_file_sorter = image_file_sorter
        self.__enable_center_focus = enable_center_focus
        self.__debug = InternalDebug(debug_mode=debug_mode)

        self.__ids_arr = self.__df[self.__id_column_name].to_numpy()
        self.__labels_arr = self.__df[self.__label_column_name].to_numpy()

        self.__cache_dir = cache_dir
        self.__cache_version = cache_version
        if self.__cache_dir is not None:
            os.makedirs(self.__cache_dir, exist_ok=True)

        self.__executor = None
        if ThreadPoolExecutor is not None and as_completed is not None:
            self.__executor = ThreadPoolExecutor(
                max_workers=min(self.__max_threads, len(self.__scan_categories))
            )

        self.__file_list_cache = {}

        self.__slice_executor = None
        if ThreadPoolExecutor is not None and as_completed is not None:
            self.__slice_executor = ThreadPoolExecutor(
                max_workers=max(1, int(self.__max_threads))
            )

    def __del__(self):
        try:
            if getattr(self, "_DICOMLoader__executor", None) is not None:
                self.__executor.shutdown(wait=False, cancel_futures=True)
        except Exception:
            pass
        try:
            if getattr(self, "_DICOMLoader__slice_executor", None) is not None:
                self.__slice_executor.shutdown(wait=False, cancel_futures=True)
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
        return self.__ids_arr[row]

    def gel_label(self, row):
        return self.__labels_arr[row]

    def format(self, images, type):
        if self.image_format == ImageFormat.WHDC:
            if type == "default":
                return images
            elif type == "normalize":
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
        return images

    def _cache_key(self, patient_id, scan_category):
        key = f"{self.__cache_version}|{patient_id}|{scan_category}|{self.__num_imgs}|{self.__size}|{self.__scale}|{self.__rotate_angle}|{int(self.__enable_center_focus)}"
        return hashlib.md5(key.encode("utf-8")).hexdigest()

    def _cache_path(self, patient_id, scan_category):
        if self.__cache_dir is None:
            return None
        return os.path.join(
            self.__cache_dir, f"{self._cache_key(patient_id, scan_category)}.npy"
        )

    def _list_sorted_dicom_files(self, scans_path):
        cached = self.__file_list_cache.get(scans_path)
        if cached is not None:
            return cached

        try:
            entries = []
            with os.scandir(scans_path) as it:
                for e in it:
                    if e.is_file():
                        entries.append(e.name)
        except FileNotFoundError:
            self.__file_list_cache[scans_path] = []
            return []

        if not entries:
            self.__file_list_cache[scans_path] = []
            return []

        def _key_from_name(name: str) -> int:
            base = name[:-4]
            i = base.rfind("-")
            return int(base[i + 1 :]) if i != -1 else 0

        entries.sort(key=_key_from_name)
        files = [os.path.join(scans_path, n) for n in entries]
        self.__file_list_cache[scans_path] = files
        return files

    def load_scan(self, row, scan_category, show_progress=True):
        self.__debug.log("== load_scan ==")
        patient_id = str(int(self.__ids_arr[row])).zfill(5)

        cache_path = self._cache_path(patient_id, scan_category)
        if cache_path is not None and os.path.exists(cache_path):
            arr = np.load(cache_path, allow_pickle=False, mmap_mode=None)
            return self.format(arr, "default")

        scans_path = os.path.join(self.__input_path, patient_id, scan_category)

        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        image_files = self._list_sorted_dicom_files(scans_path)

        if not image_files:
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)

        if self.__slice_executor is not None and len(image_files) > 1:
            loaded_images = list(
                self.__slice_executor.map(self._load_dicom_image, image_files)
            )
            loaded_images = [img for img in loaded_images if img is not None]
        else:
            loaded_images = []
            for fp in image_files:
                img = self._load_dicom_image(fp)
                if img is not None:
                    loaded_images.append(img)

        if not loaded_images:
            raise ValueError(
                "No images were loaded, and num_imgs is set. Cannot proceed."
            )

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                zero_image = np.zeros_like(loaded_images[0])
                loaded_images.append(zero_image)

        loaded_images = np.array(loaded_images)

        if cache_path is not None:
            tmp_path = cache_path + ".tmp.npy"
            np.save(tmp_path, loaded_images, allow_pickle=False)
            os.replace(tmp_path, cache_path)

        return self.format(loaded_images, "default")

    def load_all_scans(self, row, show_progress=True):
        self.__debug.log("== load_all_scans ==")
        all_images = {}

        if self.__executor is None or as_completed is None:
            for scan_category in self.__scan_categories:
                all_images[scan_category] = self.load_scan(
                    row, scan_category, show_progress=False
                )
            return {key: all_images.get(key, []) for key in self.__scan_categories}

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

        dicom_file = pydicom.dcmread(
            dicom_path,
            force=True,
            stop_before_pixels=False,
            defer_size=1 << 20,
            read_file_meta=False,
            specific_tags=[
                "PixelData",
                "Rows",
                "Columns",
                "BitsAllocated",
                "BitsStored",
                "HighBit",
                "PixelRepresentation",
                "SamplesPerPixel",
                "PhotometricInterpretation",
                "RescaleIntercept",
                "RescaleSlope",
            ],
        )

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
        h, w = image.shape[:2]
        cx, cy = w * 0.5, h * 0.5
        ws, hs = w * self.__scale, h * self.__scale
        left = int(cy - hs * 0.5)
        right = int(cy + hs * 0.5)
        top = int(cx - ws * 0.5)
        bottom = int(cx + ws * 0.5)
        return image[left:right, top:bottom]

    def _rotate_img(self, image):
        if self.__rotate_angle <= 0:
            return image
        height, width = image.shape[:2]
        center = (width / 2, height / 2)
        rotation_matrix = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)
        return cv2.warpAffine(image, rotation_matrix, (width, height))

    def _normalization_img(self, image):
        min_val = image.min()
        max_val = image.max()
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
        self.log(character * length)

    def info(self, *args):
        if self.__debug_mode:
            self.log(self.__prefix + "[INFO]", *args)

    def warning(self, *args):
        if self.__debug_mode:
            self.log(self.__prefix + "[WARNING]", *args)

    def error(self, *args):
        if self.__debug_mode:
            self.log(self.__prefix + "[ERROR]", *args)

    def set_debug_mode(self, debug_mode):
        self.__debug_mode = debug_mode




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

        self.__n_cats = len(self.__dicom_loader.scan_categories)

        self.__d = (
            int(self.__dicom_loader.num_imgs)
            if self.__dicom_loader.num_imgs is not None
            else None
        )
        self.__w = int(self.__dicom_loader._DICOMLoader__size[0])
        self.__h = int(self.__dicom_loader._DICOMLoader__size[1])
        self.__c = 1

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bs = len(batch_indices)

        if self.__is_trainable:
            batches_y = np.empty((bs,), dtype=np.float32)
        else:
            batches_y = None

        if self.__d is None:
            raise ValueError("num_imgs must be set for this model input pipeline.")

        batches_x = [
            np.empty((bs, self.__d, self.__w, self.__h, self.__c), dtype=np.uint8)
            for _ in range(self.__n_cats)
        ]

        for bi, i in enumerate(batch_indices):
            if self.__is_trainable:
                batches_y[bi] = self.__dicom_loader.gel_label(i)

            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )

            for j, cat in enumerate(self.__dicom_loader.scan_categories):
                batches_x[j][bi] = batch_x_image_paths[cat]

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

INPUT_SHAPE = (IMG_SEQ, IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN)

MODEL_NAME = "Mult3DCNN4Input"
BATCH_SIZE = 8
EPOCHS = 26

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)
COMPILE_LOSS = "binary_crossentropy"
COMPILE_METRICS = [AUC(name="auc")]

os.makedirs(LOGS_PATH, exist_ok=True)
os.makedirs(BEST_MODEL_PATH, exist_ok=True)

CACHE_DIR = os.path.join(RUN_DIR, "cache_preprocessed")
os.makedirs(CACHE_DIR, exist_ok=True)



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

test_df["Label"] = 0.0




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
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
val_size = max(1, int(0.2 * len(train_df)))
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_df_split = train_df.iloc[trn_idx].reset_index(drop=True)
val_df_split = train_df.iloc[val_idx].reset_index(drop=True)

train_dicom_loader = DICOMLoader(
    train_df_split,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    id_column_name="ID",
    label_column_name="Label",
    cache_dir=os.path.join(CACHE_DIR, "train"),
    cache_version=VERSION,
)

val_dicom_loader = DICOMLoader(
    val_df_split,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    id_column_name="ID",
    label_column_name="Label",
    cache_dir=os.path.join(CACHE_DIR, "val"),
    cache_version=VERSION,
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


def _warm_cache(_dicom_loader: DICOMLoader, desc: str = ""):
    for row in tqdm(range(_dicom_loader.len), desc=desc):
        for cat in _dicom_loader.scan_categories:
            _ = _dicom_loader.load_scan(row, cat, show_progress=False)


def _warm_cache_parallel(
    _dicom_loader: DICOMLoader, max_workers: int = 8, desc: str = ""
):
    if ThreadPoolExecutor is None:
        return _warm_cache(_dicom_loader, desc=desc)

    tasks = []
    for row in range(_dicom_loader.len):
        for cat in _dicom_loader.scan_categories:
            patient_id = str(int(_dicom_loader.get_id(row))).zfill(5)
            cache_path = _dicom_loader._cache_path(patient_id, cat)
            if cache_path is not None and os.path.exists(cache_path):
                continue
            tasks.append((row, cat))

    if not tasks:
        return

    def _one(t):
        r, c = t
        _dicom_loader.load_scan(r, c, show_progress=False)
        return None

    with ThreadPoolExecutor(max_workers=max_workers) as ex:
        for _ in tqdm(ex.map(_one, tasks), total=len(tasks), desc=desc):
            pass


_warm_cache_parallel(
    train_dicom_loader, max_workers=MAX_THREADS_DICOM_LOADER, desc="Caching train"
)
_warm_cache_parallel(
    val_dicom_loader, max_workers=MAX_THREADS_DICOM_LOADER, desc="Caching val"
)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2256192708.py in <cell line: 0>()
     97 
     98 # Speed: build cache once; massively reduces epoch time (no repeated DICOM decoding each epoch).
---> 99 _warm_cache_parallel(
    100     train_dicom_loader, max_workers=MAX_THREADS_DICOM_LOADER, desc="Caching train"
    101 )

/tmp/ipykernel_11/2256192708.py in _warm_cache_parallel(_dicom_loader, max_workers, desc)
     92 
     93     with ThreadPoolExecutor(max_workers=max_workers) as ex:
---> 94         for _ in tqdm(ex.map(_one, tasks), total=len(tasks), desc=desc):
     95             pass
     96 

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

/tmp/ipykernel_11/2256192708.py in _one(t)
     88     def _one(t):
     89         r, c = t
---> 90         _dicom_loader.load_scan(r, c, show_progress=False)
     91         return None
     92 

/tmp/ipykernel_11/2961905979.py in load_scan(self, row, scan_category, show_progress)
    203         # Speed: use executor.map instead of per-slice submit+result; reduces Python overhead while preserving order.
    204         if self.__slice_executor is not None and len(image_files) > 1:
--> 205             loaded_images = list(
    206                 self.__slice_executor.map(self._load_dicom_image, image_files)
    207             )

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

/tmp/ipykernel_11/2961905979.py in _load_dicom_image(self, dicom_path)
    266         # Speed: force=True + read_file_meta=False avoids extra parsing; specific_tags keeps I/O minimal.
    267         # Correctness: pixel_array decoding and subsequent preprocessing remain identical.
--> 268         dicom_file = pydicom.dcmread(
    269             dicom_path,
    270             force=True,

TypeError: dcmread() got an unexpected keyword argument 'read_file_meta'

## === cell 9
model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

callbacks = [
    tf.keras.callbacks.ModelCheckpoint(
        filepath=BEST_MODEL_H5_DIR,
        monitor="val_auc",
        mode="max",
        save_best_only=True,
        save_weights_only=False,
        verbose=1,
    )
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



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1897073094.py in <cell line: 0>()
     13 ]
     14 
---> 15 history = model.fit(
     16     train_dataset,
     17     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1110042724.py in __getitem__(self, ids)
     51                 batches_y[bi] = self.__dicom_loader.gel_label(i)
     52 
---> 53             batch_x_image_paths = self.__dicom_loader.load_all_scans(
     54                 i, show_progress=False
     55             )

/tmp/ipykernel_11/2961905979.py in load_all_scans(self, row, show_progress)
    253         for future in as_completed(future_to_scan_category):
    254             scan_category = future_to_scan_category[future]
--> 255             image_data = future.result()
    256             if image_data is not None:
    257                 all_images[scan_category] = image_data

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

/tmp/ipykernel_11/2961905979.py in load_scan(self, row, scan_category, show_progress)
    203         # Speed: use executor.map instead of per-slice submit+result; reduces Python overhead while preserving order.
    204         if self.__slice_executor is not None and len(image_files) > 1:
--> 205             loaded_images = list(
    206                 self.__slice_executor.map(self._load_dicom_image, image_files)
    207             )

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())
    622             finally:
--> 623                 for future in fs:
    624                     future.cancel()
    625         return result_iterator()

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    320     finally:
    321         # Break a reference cycle with the exception in self._exception
--> 322         del fut
    323 
    324 

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

/tmp/ipykernel_11/2961905979.py in _load_dicom_image(self, dicom_path)
    266         # Speed: force=True + read_file_meta=False avoids extra parsing; specific_tags keeps I/O minimal.
    267         # Correctness: pixel_array decoding and subsequent preprocessing remain identical.
--> 268         dicom_file = pydicom.dcmread(
    269             dicom_path,
    270             force=True,

TypeError: dcmread() got an unexpected keyword argument 'read_file_meta'

## === cell 10
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
    id_column_name="ID",
    label_column_name="Label",
    cache_dir=os.path.join(CACHE_DIR, "test"),
    cache_version=VERSION,
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)

_warm_cache_parallel(
    test_dicom_loader, max_workers=MAX_THREADS_DICOM_LOADER, desc="Caching test"
)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/353673365.py in <cell line: 0>()
     25 
     26 # Speed: cache test once before predict to avoid per-batch DICOM decode.
---> 27 _warm_cache_parallel(
     28     test_dicom_loader, max_workers=MAX_THREADS_DICOM_LOADER, desc="Caching test"
     29 )

/tmp/ipykernel_11/2256192708.py in _warm_cache_parallel(_dicom_loader, max_workers, desc)
     92 
     93     with ThreadPoolExecutor(max_workers=max_workers) as ex:
---> 94         for _ in tqdm(ex.map(_one, tasks), total=len(tasks), desc=desc):
     95             pass
     96 

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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

/tmp/ipykernel_11/2256192708.py in _one(t)
     88     def _one(t):
     89         r, c = t
---> 90         _dicom_loader.load_scan(r, c, show_progress=False)
     91         return None
     92 

/tmp/ipykernel_11/2961905979.py in load_scan(self, row, scan_category, show_progress)
    203         # Speed: use executor.map instead of per-slice submit+result; reduces Python overhead while preserving order.
    204         if self.__slice_executor is not None and len(image_files) > 1:
--> 205             loaded_images = list(
    206                 self.__slice_executor.map(self._load_dicom_image, image_files)
    207             )

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

/tmp/ipykernel_11/2961905979.py in _load_dicom_image(self, dicom_path)
    266         # Speed: force=True + read_file_meta=False avoids extra parsing; specific_tags keeps I/O minimal.
    267         # Correctness: pixel_array decoding and subsequent preprocessing remain identical.
--> 268         dicom_file = pydicom.dcmread(
    269             dicom_path,
    270             force=True,

TypeError: dcmread() got an unexpected keyword argument 'read_file_meta'

## === cell 11
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
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    submission = submission[["BraTS21ID", "MGMT_value"]]
    return submission


submission = generate_predictions(model, test_dataset, test_df)
submission.to_csv("submission.csv", index=False)
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)

print("Wrote submission to:", os.path.abspath("submission.csv"))
print("Also wrote submission to:", SUBMISSION_DATASET_DF_DIR)
print(submission.head())
print(submission.info())

## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1597220522.py in <cell line: 0>()
     18 
     19 
---> 20 submission = generate_predictions(model, test_dataset, test_df)
     21 submission.to_csv("submission.csv", index=False)
     22 submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)

/tmp/ipykernel_11/1597220522.py in generate_predictions(model, test_dataset, test_df)
      1 def generate_predictions(model, test_dataset, test_df):
----> 2     preds = model.predict(
      3         test_dataset,
      4         verbose=0,
      5     ).reshape(

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1110042724.py in __getitem__(self, ids)
     51                 batches_y[bi] = self.__dicom_loader.gel_label(i)
     52 
---> 53             batch_x_image_paths = self.__dicom_loader.load_all_scans(
     54                 i, show_progress=False
     55             )

/tmp/ipykernel_11/2961905979.py in load_all_scans(self, row, show_progress)
    253         for future in as_completed(future_to_scan_category):
    254             scan_category = future_to_scan_category[future]
--> 255             image_data = future.result()
    256             if image_data is not None:
    257                 all_images[scan_category] = image_data

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

/tmp/ipykernel_11/2961905979.py in load_scan(self, row, scan_category, show_progress)
    203         # Speed: use executor.map instead of per-slice submit+result; reduces Python overhead while preserving order.
    204         if self.__slice_executor is not None and len(image_files) > 1:
--> 205             loaded_images = list(
    206                 self.__slice_executor.map(self._load_dicom_image, image_files)
    207             )

/usr/lib/python3.11/concurrent/futures/_base.py in result_iterator()
    621                         yield _result_or_cancel(fs.pop(), end_time - time.monotonic())
    622             finally:
--> 623                 for future in fs:
    624                     future.cancel()
    625         return result_iterator()

/usr/lib/python3.11/concurrent/futures/_base.py in _result_or_cancel(***failed resolving arguments***)
    320     finally:
    321         # Break a reference cycle with the exception in self._exception
--> 322         del fut
    323 
    324 

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

/tmp/ipykernel_11/2961905979.py in _load_dicom_image(self, dicom_path)
    266         # Speed: force=True + read_file_meta=False avoids extra parsing; specific_tags keeps I/O minimal.
    267         # Correctness: pixel_array decoding and subsequent preprocessing remain identical.
--> 268         dicom_file = pydicom.dcmread(
    269             dicom_path,
    270             force=True,

TypeError: dcmread() got an unexpected keyword argument 'read_file_meta'
