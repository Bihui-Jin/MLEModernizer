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

0.5615332313445521

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.47412) has done: 'I fix the import/runtime crash by removing the unnecessary `keras.utils.vis_utils.plot_model` import that triggers a protobuf `MessageFactory.GetPrototype` error in this environment, while keeping TensorFlow/Keras usage intact. Then I fix the missing external model file issue by training the existing `DeepScanModel` (same architecture/loss/optimizer/loops semantics) on the provided training set and saving/loading from the existing `BEST_MODEL_H5_DIR` path. Finally, I ensure the test dataframe uses correct IDs and the submission is written to `/kaggle/working/submission.csv` with the required `BraTS21ID,MGMT_value` columns and correct row alignment.'

# 9. Code solution

## === cell 0
import os
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

PLOT_MODEL_AVAILABLE = False

os.environ["PYTHONHASHSEED"] = "0"
random.seed(123)
np.random.seed(123)
tf.random.set_seed(123)

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("TF_NUM_INTRAOP_THREADS", "1")
os.environ.setdefault("TF_NUM_INTEROP_THREADS", "1")
try:
    tf.config.threading.set_intra_op_parallelism_threads(1)
    tf.config.threading.set_inter_op_parallelism_threads(1)
except Exception:
    pass



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
IS_PYDICOM_IMPORTED = True
if IS_PYDICOM_IMPORTED:
    try:
        import os
        import glob
        import cv2
        import numpy as np
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed

        try:
            from tqdm import tqdm
        except Exception:

            def tqdm(x, total=None, desc=None):
                return x

    except ImportError:
        print(f"Missing some imports: {ImportError}")

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
            image_file_sorter=lambda x: int(
                os.path.splitext(os.path.basename(x))[0].split("-")[-1]
            ),
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
            if cache_dir is None:
                cache_dir = os.path.join("/kaggle/working", "dicom_cache_v1")
            self.__cache_dir = cache_dir
            os.makedirs(self.__cache_dir, exist_ok=True)

            self.__patient_ids = (
                self.__df[self.__id_column_name]
                .astype(int)
                .astype(str)
                .str.zfill(5)
                .tolist()
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

        def get_id(self, row):
            return self.__df.loc[row, self.__id_column_name]

        def get_label(self, row):
            return self.__df.loc[row, self.__label_column_name]

        def gel_label(self, row):
            return self.get_label(row)

        def format(self, images, type):
            if type == "normalize" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)
            elif type == "default" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
            else:
                return images

        def _cache_key(self, patient_id, scan_category):
            w, h = self.__size
            return (
                f"id={patient_id}"
                f"__scan={scan_category}"
                f"__n={self.__num_imgs}"
                f"__size={w}x{h}"
                f"__scale={self.__scale}"
                f"__rot={self.__rotate_angle}"
                f"__center={int(self.__enable_center_focus)}"
                f"__fmt={self.__image_format.value}"
                f".npy"
            )

        def _cache_path(self, patient_id, scan_category):
            return os.path.join(
                self.__cache_dir, self._cache_key(patient_id, scan_category)
            )

        def load_scan(self, row, scan_category, show_progress=True):
            self.__debug.log("== load_scan ==")
            patient_id = self.__patient_ids[row]
            cache_path = self._cache_path(patient_id, scan_category)

            if self.__use_cache and os.path.exists(cache_path):
                arr = np.load(cache_path, allow_pickle=False, mmap_mode=None)
                return arr

            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")), key=self.__image_file_sorter
            )
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")

            image_files = self._select_subset_image_files(image_files)

            with ThreadPoolExecutor(self.__max_threads) as executor:
                image_data_iterable = executor.map(self._load_dicom_image, image_files)
                if show_progress:
                    image_data_iterable = tqdm(
                        image_data_iterable,
                        total=len(image_files),
                        desc="Loading images",
                    )
                loaded_images = [im for im in image_data_iterable if im is not None]

            if not loaded_images:
                raise ValueError(
                    "No images were loaded, and num_imgs is set. Cannot proceed."
                )

            if self.__num_imgs is not None and len(loaded_images) < self.__num_imgs:
                zero_image = np.zeros_like(loaded_images[0])
                loaded_images.extend(
                    [zero_image] * (self.__num_imgs - len(loaded_images))
                )

            loaded_images = np.stack(loaded_images, axis=0)
            loaded_images = self.format(loaded_images, "default")

            if self.__use_cache:
                tmp = cache_path + ".tmp"
                np.save(tmp, loaded_images, allow_pickle=False)
                os.replace(tmp, cache_path)

            return loaded_images

        def load_all_scans(self, row, show_progress=True):
            self.__debug.log("== load_all_scans ==")

            all_images = {}
            with ThreadPoolExecutor(
                min(self.__max_threads, len(self.__scan_categories))
            ) as executor:
                future_to_scan_category = {
                    executor.submit(
                        self.load_scan, row, scan_category, False
                    ): scan_category
                    for scan_category in self.__scan_categories
                }

                if show_progress:
                    progress_bar = tqdm(
                        total=len(self.__scan_categories), desc="Loading scan types"
                    )

                for future in as_completed(future_to_scan_category):
                    scan_category = future_to_scan_category[future]
                    image_data = future.result()
                    if image_data is not None:
                        all_images[scan_category] = image_data
                        if show_progress:
                            progress_bar.update(1)

                if show_progress:
                    progress_bar.close()

            return {key: all_images.get(key, []) for key in self.__scan_categories}

        def _load_dicom_image(self, dicom_path):
            self.__debug.log("== _load_dicom_image ==")

            try:
                dicom_file = pydicom.dcmread(
                    dicom_path,
                    force=True,
                    stop_before_pixels=False,
                    specific_tags=[
                        "PixelData",
                        "Rows",
                        "Columns",
                        "BitsAllocated",
                        "PhotometricInterpretation",
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

        batches_y = np.empty((len(batch_indices),), dtype=np.float32)
        batches_x = [[] for _ in range(len(self.__dicom_loader.scan_categories))]

        for bi, i in enumerate(batch_indices):
            batches_y[bi] = self.__dicom_loader.get_label(i)
            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )
            for j, (_scan_type, images) in enumerate(batch_x_image_paths.items()):
                batches_x[j].append(images)

        batch_x = tuple(np.stack(b, axis=0) for b in batches_x)  # tuple, not list
        batch_y = batches_y

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

TF_CALL_BACK_BEST_MODEL_MONITOR = "val_auc"
TF_CALL_BACK_EARLY_STOP_MONITOR = "auc"
TF_CALL_BACK_EARLY_STOP_PATIENTE = 6

os.makedirs(BEST_MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

DICOM_CACHE_DIR = os.path.join("/kaggle/working", "dicom_cache_v1")
os.makedirs(DICOM_CACHE_DIR, exist_ok=True)



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




## === cell 7
def make_stratified_folds(df, label_col="Label", n_splits=5, seed=123):
    df = df.copy()
    rng = np.random.RandomState(seed)
    folds = np.full(len(df), -1, dtype=int)
    for cls in sorted(df[label_col].unique()):
        idx = np.where(df[label_col].values == cls)[0]
        rng.shuffle(idx)
        for j, ii in enumerate(idx):
            folds[ii] = j % n_splits
    df["fold"] = folds
    return df


train_df_folds = make_stratified_folds(
    train_df, label_col="Label", n_splits=NUM_SPLIT_FOLDS, seed=SEED
)

train_split_df = train_df_folds[
    train_df_folds["fold"] != SELECTED_VALIDATION_FOLD
].reset_index(drop=True)
val_split_df = train_df_folds[
    train_df_folds["fold"] == SELECTED_VALIDATION_FOLD
].reset_index(drop=True)

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
    cache_dir=DICOM_CACHE_DIR,
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
    cache_dir=DICOM_CACHE_DIR,
    use_cache=True,
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
earlystop_cb = EarlyStopping(
    monitor=TF_CALL_BACK_EARLY_STOP_MONITOR,
    mode="max",
    patience=TF_CALL_BACK_EARLY_STOP_PATIENTE,
    restore_best_weights=True,
    verbose=1,
)

history = model.fit(
    train_dataset,
    validation_data=val_dataset,
    epochs=EPOCHS,
    verbose=VERBOSITY,
    callbacks=[checkpoint_cb, earlystop_cb],
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
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1556348060.py in <cell line: 0>()
     18 )
     19 
---> 20 history = model.fit(
     21     train_dataset,
     22     validation_data=val_dataset,

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1461755730.py in __getitem__(self, ids)
     28         for bi, i in enumerate(batch_indices):
     29             batches_y[bi] = self.__dicom_loader.get_label(i)
---> 30             batch_x_image_paths = self.__dicom_loader.load_all_scans(
     31                 i, show_progress=False
     32             )

/tmp/ipykernel_11/2636414547.py in load_all_scans(self, row, show_progress)
    227                 for future in as_completed(future_to_scan_category):
    228                     scan_category = future_to_scan_category[future]
--> 229                     image_data = future.result()
    230                     if image_data is not None:
    231                         all_images[scan_category] = image_data

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

/tmp/ipykernel_11/2636414547.py in load_scan(self, row, scan_category, show_progress)
    180                         desc="Loading images",
    181                     )
--> 182                 loaded_images = [im for im in image_data_iterable if im is not None]
    183 
    184             if not loaded_images:

/tmp/ipykernel_11/2636414547.py in <listcomp>(.0)
    180                         desc="Loading images",
    181                     )
--> 182                 loaded_images = [im for im in image_data_iterable if im is not None]
    183 
    184             if not loaded_images:

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

/tmp/ipykernel_11/2636414547.py in _load_dicom_image(self, dicom_path)
    259                 raise IOError(f"An error occurred while reading the DICOM file: {e}")
    260 
--> 261             image = dicom_file.pixel_array
    262             image = self._rotate_img(image)
    263             image = self._normalization_img(image)

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
    585         if "Float" not in self.pixel_keyword:
    586             if self._opts.get("bits_stored") is None:
--> 587                 raise AttributeError(f"{prefix},0101) 'Bits Stored'")
    588 
    589             if not 1 <= self.bits_stored <= self.bits_allocated <= 64:

AttributeError: Missing required element: (0028,0101) 'Bits Stored'

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
    cache_dir=DICOM_CACHE_DIR,
    use_cache=True,
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)




## === cell 11
def generate_predictions(model, test_dataset, test_df):
    preds = model.predict(test_dataset, verbose=0).reshape(-1)
    preds = np.clip(preds, 0.0, 1.0)

    submission = test_df.copy()
    submission["Label"] = preds[: len(submission)]
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_dataset, test_df)



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/1136866462.py in <cell line: 0>()
     15 
     16 
---> 17 submission = generate_predictions(model, test_dataset, test_df)
     18 

/tmp/ipykernel_11/1136866462.py in generate_predictions(model, test_dataset, test_df)
      2     # Performance: use Keras' built-in predict over Sequence to reduce Python overhead
      3     # and allow internal prefetching; outputs are identical to manual batching.
----> 4     preds = model.predict(test_dataset, verbose=0).reshape(-1)
      5     preds = np.clip(preds, 0.0, 1.0)
      6 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/1461755730.py in __getitem__(self, ids)
     28         for bi, i in enumerate(batch_indices):
     29             batches_y[bi] = self.__dicom_loader.get_label(i)
---> 30             batch_x_image_paths = self.__dicom_loader.load_all_scans(
     31                 i, show_progress=False
     32             )

/tmp/ipykernel_11/2636414547.py in load_all_scans(self, row, show_progress)
    227                 for future in as_completed(future_to_scan_category):
    228                     scan_category = future_to_scan_category[future]
--> 229                     image_data = future.result()
    230                     if image_data is not None:
    231                         all_images[scan_category] = image_data

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

/tmp/ipykernel_11/2636414547.py in load_scan(self, row, scan_category, show_progress)
    180                         desc="Loading images",
    181                     )
--> 182                 loaded_images = [im for im in image_data_iterable if im is not None]
    183 
    184             if not loaded_images:

/tmp/ipykernel_11/2636414547.py in <listcomp>(.0)
    180                         desc="Loading images",
    181                     )
--> 182                 loaded_images = [im for im in image_data_iterable if im is not None]
    183 
    184             if not loaded_images:

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

/tmp/ipykernel_11/2636414547.py in _load_dicom_image(self, dicom_path)
    259                 raise IOError(f"An error occurred while reading the DICOM file: {e}")
    260 
--> 261             image = dicom_file.pixel_array
    262             image = self._rotate_img(image)
    263             image = self._normalization_img(image)

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
    585         if "Float" not in self.pixel_keyword:
    586             if self._opts.get("bits_stored") is None:
--> 587                 raise AttributeError(f"{prefix},0101) 'Bits Stored'")
    588 
    589             if not 1 <= self.bits_stored <= self.bits_allocated <= 64:

AttributeError: Missing required element: (0028,0101) 'Bits Stored'

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
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote submission to:", SUBMISSION_DATASET_DF_DIR)
print(pd.read_csv(SUBMISSION_DATASET_DF_DIR).head())

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3566789037.py in <cell line: 0>()
----> 1 submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
      2 print("Wrote submission to:", SUBMISSION_DATASET_DF_DIR)
      3 print(pd.read_csv(SUBMISSION_DATASET_DF_DIR).head())

NameError: name 'submission' is not defined
