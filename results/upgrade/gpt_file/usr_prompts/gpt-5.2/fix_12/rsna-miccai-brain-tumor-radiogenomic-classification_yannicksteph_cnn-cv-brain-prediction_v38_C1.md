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

0.5626381098079212

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.46235) has done: 'I remove the import that crashes Keras/TensorFlow in this environment (`plot_model` via `keras.utils.vis_utils` triggers the protobuf `MessageFactory` error) and keep the rest of the modeling pipeline unchanged. Because the external pretrained model file `/kaggle/input/model-v5/model_V5.h5` is missing, I switch to training the provided `DeepScanModel` on the available train set and then run inference on the test set to generate `submission.csv`. I also fix a few small issues that would otherwise break execution (typo `gel_label` usage for test, excluded-ID filtering type mismatch, and ensuring prediction length matches the submission rows). The output be written as `/kaggle/working/submission.csv` with the required columns.'
- What this solution (achieved 0.46235) has done: 'Main bottlenecks are (1) repeatedly reading and sorting DICOM filenames by opening every file to fetch `InstanceNumber`, and (2) `load_all_scans()` doing duplicate work (it spawns threaded loads, then immediately re-reads the same files again serially). I (a) sort using the already-encoded instance number in the filename (your existing `image_file_sorter`) to avoid header reads during sorting, (b) fix `load_all_scans()` to use only the threaded results (no second pass), and (c) make scan loading deterministic and ordered while still parallel. Additionally, I enable Keras `Sequence` multiprocessing workers for batch prefetching (keeps the same training loop/epochs/model, just overlaps CPU DICOM decode with GPU/TF compute), and slightly reduce Python overhead in the generator by avoiding redundant `.items()`/casts.'

# 9. Code solution

## === cell 0
import os

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)
os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION_VERSION", None)

import glob
import random
import numpy as np
import pandas as pd

import cv2
import pydicom
from enum import Enum

SEED = 123
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)

pydicom.config.convert_wrong_length_to_UN = True

try:
    import pydicom.pixel_data_handlers.gdcm_handler as _gdcm_handler

    if _gdcm_handler.is_available():
        pydicom.config.image_handlers = [
            _gdcm_handler,
            pydicom.pixel_data_handlers.pillow_handler,
            pydicom.pixel_data_handlers.numpy_handler,
            pydicom.pixel_data_handlers.rle_handler,
        ]
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


class ImageFormat(Enum):
    """
    Enum to represent image formats.

    - 'W-H-D-C' = Width-Height-Depth-Channel
    - 'D-W-H-C' = Depth-Width-Height-Channel
    """

    WHDC = "W-H-D-C"  # Format (128, 128, 64, 1)
    DWHC = "D-W-H-C"  # Format (64, 128, 128, 1)

    def swap_dimensions(image, image_format):
        """
        Swap dimensions of an image NumPy array based on the specified permutation type.
        """
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

        self.__series_files_cache = {}
        self.__dicom_image_cache = {}  # dicom_path -> np.ndarray (H,W,1) uint8
        self.__patient_scans_cache = {}  # row -> dict[scan_category] -> np.ndarray

        self.__executor = ThreadPoolExecutor(max_workers=self.__max_threads)

        self.__pydicom_specific_tags = [
            "PixelData",
            "Rows",
            "Columns",
            "BitsAllocated",
            "BitsStored",
            "HighBit",
            "PixelRepresentation",
            "SamplesPerPixel",
            "PhotometricInterpretation",
            "PlanarConfiguration",
            "RescaleIntercept",
            "RescaleSlope",
            "WindowCenter",
            "WindowWidth",
            "TransferSyntaxUID",
        ]

    def __del__(self):
        try:
            ex = getattr(self, "_DICOMLoader__executor", None)
            if ex is not None:
                ex.shutdown(wait=False, cancel_futures=True)
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

    def _get_sorted_image_files(self, scans_path):
        cached = self.__series_files_cache.get(scans_path)
        if cached is not None:
            return cached

        if not os.path.exists(scans_path):
            self.__series_files_cache[scans_path] = []
            return []

        entries = []
        with os.scandir(scans_path) as it:
            for e in it:
                if e.is_file():
                    entries.append(e.path)

        if not entries:
            self.__series_files_cache[scans_path] = []
            return []

        entries.sort(key=lambda fp: self.__image_file_sorter(os.path.basename(fp)))
        self.__series_files_cache[scans_path] = entries
        return entries

    def load_scan(self, row, scan_category, show_progress=True):
        self.__debug.log("== load_scan ==")
        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
        scans_path = os.path.join(self.__input_path, patient_id, scan_category)

        if not os.path.exists(scans_path):
            raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

        image_files = self._get_sorted_image_files(scans_path)
        if not image_files:
            raise ValueError(f"No image files found in {scans_path}.")

        image_files = self._select_subset_image_files(image_files)

        loaded_images = [
            im
            for im in self.__executor.map(self._load_dicom_image, image_files)
            if im is not None
        ]

        if not loaded_images:
            raise ValueError(
                "No images were loaded, and num_imgs is set. Cannot proceed."
            )

        if self.__num_imgs is not None:
            while len(loaded_images) < self.__num_imgs:
                loaded_images.append(np.zeros_like(loaded_images[0]))

        loaded_images = np.array(loaded_images)
        return self.format(loaded_images, "default")

    def load_all_scans(self, row, show_progress=True):
        self.__debug.log("== load_all_scans ==")

        cached = self.__patient_scans_cache.get(row)
        if cached is not None:
            return cached

        patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)

        series_files = {}
        for scan_category in self.__scan_categories:
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")
            image_files = self._get_sorted_image_files(scans_path)
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")
            series_files[scan_category] = self._select_subset_image_files(image_files)

        all_images = {}
        for scan_category in self.__scan_categories:
            fps = series_files[scan_category]
            imgs = [
                im
                for im in self.__executor.map(self._load_dicom_image, fps)
                if im is not None
            ]
            if not imgs:
                all_images[scan_category] = []
                continue
            if self.__num_imgs is not None:
                while len(imgs) < self.__num_imgs:
                    imgs.append(np.zeros_like(imgs[0]))
            all_images[scan_category] = self.format(np.asarray(imgs), "default")

        result = {key: all_images.get(key, []) for key in self.__scan_categories}
        self.__patient_scans_cache[row] = result
        return result

    def _load_dicom_image(self, dicom_path):
        self.__debug.log("== _load_dicom_image ==")

        cached = self.__dicom_image_cache.get(dicom_path)
        if cached is not None:
            return cached

        if not os.path.exists(dicom_path):
            raise FileNotFoundError(f"File {dicom_path} does not exist.")

        try:
            dicom_file = pydicom.dcmread(
                dicom_path,
                stop_before_pixels=True,
                force=True,
                specific_tags=self.__pydicom_specific_tags,
            )
            dicom_file.decompress(handler_name=None)
        except Exception as e:
            raise IOError(f"An error occurred while reading the DICOM file: {e}")

        image = dicom_file.pixel_array
        image = self._rotate_img(image)
        image = self._normalization_img(image)
        image = self._crop_img(image)
        image = self._resize_img(image)
        image = np.expand_dims(image, axis=-1)

        self.__dicom_image_cache[dicom_path] = image
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
            return np.zeros_like(image, dtype=np.uint8)
        image = image - min_val
        image = image / max_val
        return (image * 255).astype(np.uint8, copy=False)

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
import tensorflow as tf

tf.random.set_seed(SEED)
tf.config.optimizer.set_jit(True)

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 4
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

        self.__scan_categories = list(self.__dicom_loader.scan_categories)
        self.__n_scans = len(self.__scan_categories)

        self.__x_shape = (self.__batch_size, 128, 128, 32, 1)
        self.__x_dtype = np.float32

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bs = len(batch_indices)

        batch_x = [
            np.empty((bs, 128, 128, 32, 1), dtype=self.__x_dtype)
            for _ in range(self.__n_scans)
        ]
        batch_y = np.empty((bs,), dtype=np.float32) if self.__is_trainable else None

        loader = self.__dicom_loader
        scan_categories = self.__scan_categories

        for bi, i in enumerate(batch_indices):
            if self.__is_trainable:
                batch_y[bi] = loader.gel_label(i)

            scans = loader.load_all_scans(i, show_progress=False)
            for j, scan_type in enumerate(scan_categories):
                batch_x[j][bi, ...] = scans[scan_type]

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

os.makedirs(RUN_DIR, exist_ok=True)
os.makedirs(f"{RUN_DIR}/models", exist_ok=True)

BEST_MODEL_H5_DIR = f"{RUN_DIR}/models/model_{VERSION}.h5"

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
BATCH_SIZE = 2

EPOCHS = 6

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)

COMPILE_LOSS = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05)

COMPILE_METRICS = [AUC(name="auc")]

TF_CALL_BACK_BEST_MODEL_MONITOR = "val_auc"

KERAS_FIT_WORKERS = min(8, (os.cpu_count() or 8))
KERAS_FIT_USE_MULTIPROCESSING = False
KERAS_FIT_MAX_QUEUE_SIZE = 16



## === cell 7
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
test_df["ID"] = test_df["ID"].astype(int)

val_frac = 0.2
rng = np.random.RandomState(SEED)
perm = rng.permutation(len(train_df))
val_size = int(len(train_df) * val_frac)
val_idx = perm[:val_size]
trn_idx = perm[val_size:]

train_split_df = train_df.iloc[trn_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)




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
    dicom_loader=train_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TRAIN.value,
    shuffle=SHUFFLE,
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


def _warmup_loader_cache(dicom_loader: DICOMLoader, max_rows=None):
    n = dicom_loader.len if max_rows is None else min(dicom_loader.len, max_rows)
    for r in range(n):
        dicom_loader.load_all_scans(r, show_progress=False)





## === cell 10
model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

checkpoint_cb = ModelCheckpoint(
    BEST_MODEL_H5_DIR,
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
    callbacks=[checkpoint_cb],
    verbose=2,
    workers=KERAS_FIT_WORKERS,
    use_multiprocessing=KERAS_FIT_USE_MULTIPROCESSING,
    max_queue_size=KERAS_FIT_MAX_QUEUE_SIZE,
)

if os.path.exists(BEST_MODEL_H5_DIR):
    model = tf.keras.models.load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1025272441.py in <cell line: 0>()
     11 )
     12 
---> 13 history = model.fit(
     14     train_dataset,
     15     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.fit() got an unexpected keyword argument 'workers'

## === cell 11
def generate_predictions(model, test_dataset, test_df, sample_sub_path):
    sample_sub = pd.read_csv(sample_sub_path)
    sample_sub["BraTS21ID"] = sample_sub["BraTS21ID"].astype(str).str.zfill(5)
    ordered_ids = sample_sub["BraTS21ID"].tolist()

    flat_preds = model.predict(
        test_dataset,
        verbose=0,
        workers=KERAS_FIT_WORKERS,
        use_multiprocessing=KERAS_FIT_USE_MULTIPROCESSING,
        max_queue_size=KERAS_FIT_MAX_QUEUE_SIZE,
    ).reshape(-1)

    n_test = len(test_df)
    if len(flat_preds) < n_test:
        flat_preds = np.concatenate(
            [flat_preds, np.full(n_test - len(flat_preds), 0.5)]
        )
    flat_preds = flat_preds[:n_test]

    pred_df = pd.DataFrame(
        {
            "BraTS21ID": test_df["ID"].astype(int).astype(str).str.zfill(5).values,
            "MGMT_value": flat_preds.astype(float),
        }
    )

    pred_map = dict(zip(pred_df["BraTS21ID"].values, pred_df["MGMT_value"].values))
    submission = pd.DataFrame(
        {
            "BraTS21ID": ordered_ids,
            "MGMT_value": [float(pred_map.get(i, 0.5)) for i in ordered_ids],
        }
    )
    return submission


submission = generate_predictions(
    model,
    test_dataset,
    test_df,
    sample_sub_path=TEST_DATASET_DF_DIR,
)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
TypeError                                 Traceback (most recent call last)
/tmp/ipykernel_11/850052281.py in <cell line: 0>()
     36 
     37 
---> 38 submission = generate_predictions(
     39     model,
     40     test_dataset,

/tmp/ipykernel_11/850052281.py in generate_predictions(model, test_dataset, test_df, sample_sub_path)
      4     ordered_ids = sample_sub["BraTS21ID"].tolist()
      5 
----> 6     flat_preds = model.predict(
      7         test_dataset,
      8         verbose=0,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    117             return fn(*args, **kwargs)
    118         except Exception as e:
--> 119             filtered_tb = _process_traceback_frames(e.__traceback__)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`

TypeError: TensorFlowTrainer.predict() got an unexpected keyword argument 'workers'

## === cell 12
submission.info()
print(submission.head())



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/2863687828.py in <cell line: 0>()
----> 1 submission.info()
      2 print(submission.head())
      3 

NameError: name 'submission' is not defined

## === cell 13
submission.to_csv("submission.csv", index=False)
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print(f"Wrote submission to: {os.path.abspath('submission.csv')}")
print(f"Wrote submission to: {SUBMISSION_DATASET_DF_DIR}")
print(submission.head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3108479048.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)
      2 submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
      3 print(f"Wrote submission to: {os.path.abspath('submission.csv')}")
      4 print(f"Wrote submission to: {SUBMISSION_DATASET_DF_DIR}")
      5 print(submission.head())

NameError: name 'submission' is not defined
