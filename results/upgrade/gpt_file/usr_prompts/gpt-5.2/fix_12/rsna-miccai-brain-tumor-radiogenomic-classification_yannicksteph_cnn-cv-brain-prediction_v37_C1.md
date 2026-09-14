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

os.environ.pop("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", None)

import glob
import random
from enum import Enum

import numpy as np
import pandas as pd

import cv2
import pydicom

import tensorflow as tf

try:
    keras = tf.keras
    from tensorflow.keras.optimizers import SGD
    from tensorflow.keras.metrics import AUC
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.layers import (
        Input,
        Conv3D,
        BatchNormalization,
        MaxPool3D,
        Dropout,
        concatenate,
        GlobalAveragePooling3D,
        Dense,
        ReLU,
    )
except Exception:
    import tf_keras as keras
    from tf_keras.optimizers import SGD
    from tf_keras.metrics import AUC
    from tf_keras.utils import Sequence
    from tf_keras.models import Model, load_model
    from tf_keras.layers import (
        Input,
        Conv3D,
        BatchNormalization,
        MaxPool3D,
        Dropout,
        concatenate,
        GlobalAveragePooling3D,
        Dense,
        ReLU,
    )

PLOT_MODEL_AVAILABLE = False

SEED = 123
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)

try:
    tf.config.threading.set_intra_op_parallelism_threads(0)
    tf.config.threading.set_inter_op_parallelism_threads(0)
except Exception:
    pass

try:
    tf.config.experimental.enable_op_determinism()
except Exception:
    pass

try:
    cv2.setNumThreads(0)
except Exception:
    pass




## === cell 1
IS_PYDICOM_IMPORTED = True
if IS_PYDICOM_IMPORTED:
    """
    This block of code checks if the code is being run with PYDICOM Lib.
    """
    try:
        import pydicom
        import os
        import glob
        import cv2
        from enum import Enum
        import numpy as np
        import pandas as pd

        try:
            from tqdm import tqdm
        except Exception:

            def tqdm(x, **kwargs):
                return x

    except ImportError:
        print(f"Missing some imports: {ImportError}")

    class ImageFormat(Enum):
        """
        Enum to represent image formats.

        - 'W-H-D-C' = Width-Height-Depth-Channel
        - 'D-W-H-C' = Depth-Width-Height-Channel
        """

        WHDC = "W-H-D-C"  # (W, H, D, C)
        DWHC = "D-W-H-C"  # (D, W, H, C)

        def swap_dimensions(image, image_format):
            """
            Swap dimensions of an image NumPy array based on the specified permutation type.
            """
            if image_format == ImageFormat.DWHC:
                image = np.transpose(image, (2, 1, 0, 3))
            elif image_format == ImageFormat.WHDC:
                image = np.transpose(image, (2, 1, 0, 3))
            return image

    def _safe_image_file_sort_key(path: str) -> int:
        """
        Bugfix: robust and deterministic ordering for DICOM slices.
        """
        base = os.path.basename(path)
        stem = os.path.splitext(base)[0]
        digits = "".join(ch for ch in stem if ch.isdigit())
        if digits:
            try:
                return int(digits)
            except Exception:
                return 0
        return 0

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
            image_file_sorter=_safe_image_file_sort_key,
            debug_mode=False,
            cache_dir=None,
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

            self.__slice_files_cache = (
                {}
            )  # (patient_id_str, scan_category) -> tuple(paths)
            self.__volume_cache = {}  # (patient_id_str, scan_category) -> np.ndarray

            self.__cache_dir = cache_dir
            if self.__cache_dir is not None:
                os.makedirs(self.__cache_dir, exist_ok=True)

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
            """
            Bugfix: Keras model expects INPUT_SHAPE=(W,H,D,C) = (128,128,32,1),
            but loaded_images are (D,H,W,C). Convert to (W,H,D,C).
            """
            if images is None or len(images) == 0:
                return images
            if images.ndim != 4:
                return images
            images = np.transpose(images, (2, 1, 0, 3))
            return images

        def _cache_path(self, patient_id: str, scan_category: str) -> str | None:
            if self.__cache_dir is None:
                return None
            key = (
                f"id={patient_id}"
                f"__scan={scan_category}"
                f"__w={self.__size[0]}__h={self.__size[1]}"
                f"__d={self.__num_imgs}"
                f"__scale={self.__scale}"
                f"__rot={self.__rotate_angle}"
                f"__cf={int(bool(self.__enable_center_focus))}"
                f"__fmt=WHDC"
            )
            fname = key + ".npy"
            return os.path.join(self.__cache_dir, fname)

        def _get_sorted_image_files(self, patient_id: str, scan_category: str):
            key = (patient_id, scan_category)
            cached = self.__slice_files_cache.get(key)
            if cached is not None:
                return cached

            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")), key=self.__image_file_sorter
            )
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")

            image_files = self._select_subset_image_files(image_files)
            image_files = tuple(image_files)
            self.__slice_files_cache[key] = image_files
            return image_files

        def load_scan(self, row, scan_category, show_progress=True):
            self.__debug.log("== load_scan ==")
            patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)

            cache_key = (patient_id, scan_category)
            cached_vol = self.__volume_cache.get(cache_key)
            if cached_vol is not None:
                return cached_vol

            disk_cache = self._cache_path(patient_id, scan_category)
            if disk_cache is not None and os.path.exists(disk_cache):
                vol = np.load(disk_cache, allow_pickle=False)
                self.__volume_cache[cache_key] = vol
                return vol

            image_files = self._get_sorted_image_files(patient_id, scan_category)

            loaded_images = []
            if show_progress:
                iterable = tqdm(
                    image_files, total=len(image_files), desc="Loading images"
                )
            else:
                iterable = image_files

            for fp in iterable:
                image_data = self._load_dicom_image(fp)
                if image_data is not None:
                    loaded_images.append(image_data)

            if not loaded_images:
                raise ValueError(
                    "No images were loaded, and num_imgs is set. Cannot proceed."
                )

            if self.__num_imgs is not None:
                while len(loaded_images) < self.__num_imgs:
                    zero_image = np.zeros_like(loaded_images[0])
                    loaded_images.append(zero_image)

            loaded_images = np.array(loaded_images)  # (D, H, W, C)
            vol = self.format(loaded_images, "default")  # (W, H, D, C)

            self.__volume_cache[cache_key] = vol
            if disk_cache is not None:
                tmp = disk_cache + ".tmp.npy"
                np.save(tmp, vol, allow_pickle=False)
                try:
                    os.replace(tmp, disk_cache)
                except Exception:
                    pass
            return vol

        def load_all_scans(self, row, show_progress=True):
            self.__debug.log("== load_all_scans ==")
            all_images = {}
            for scan_category in self.__scan_categories:
                all_images[scan_category] = self.load_scan(
                    row, scan_category, show_progress=False
                )
            return {key: all_images.get(key, []) for key in self.__scan_categories}

        def _load_dicom_image(self, dicom_path):
            self.__debug.log("== _load_dicom_image ==")
            if not os.path.exists(dicom_path):
                raise FileNotFoundError(f"File {dicom_path} does not exist.")

            try:
                dicom_file = pydicom.dcmread(dicom_path, force=True)
            except Exception as e:
                raise IOError(f"An error occurred while reading the DICOM file: {e}")

            image = dicom_file.pixel_array
            self.__debug.log("Pixel array shape:", image.shape)

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

        self.__num_modalities = len(self.__dicom_loader.scan_categories)
        self.__vol_shape = (
            self.__dicom_loader._DICOMLoader__size[0],  # W
            self.__dicom_loader._DICOMLoader__size[1],  # H
            self.__dicom_loader.num_imgs,  # D
            1,  # C
        )

        df = self.__dicom_loader._DICOMLoader__df
        self.__labels = df[
            self.__dicom_loader._DICOMLoader__label_column_name
        ].to_numpy(dtype=np.float32, copy=True)

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)

    def __getitem__(self, ids):
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size
        batch_indices = self.__indices[from_id:to_id]
        bs = len(batch_indices)

        x_batches = [
            np.empty((bs,) + self.__vol_shape, dtype=np.uint8)
            for _ in range(self.__num_modalities)
        ]
        y_batch = self.__labels[batch_indices]

        scan_cats = self.__dicom_loader.scan_categories
        for bi, i in enumerate(batch_indices):
            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )
            for mj, scan_cat in enumerate(scan_cats):
                x_batches[mj][bi] = batch_x_image_paths[scan_cat]

        batch_x = tuple(x_batches)
        if self.__is_trainable:
            return batch_x, y_batch
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

TF_CALL_BACK_BEST_MODEL_MONITOR = "val_auc"

FIT_WORKERS = min(8, max(2, (os.cpu_count() or 4)))
FIT_USE_MULTIPROCESSING = True
FIT_MAX_QUEUE_SIZE = 32

os.makedirs(BEST_MODEL_PATH, exist_ok=True)
os.makedirs(RUN_DIR, exist_ok=True)

CACHE_DIR = os.path.join(RUN_DIR, "dicom_cache_v1")
os.makedirs(CACHE_DIR, exist_ok=True)




## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

if "Label" not in test_df.columns:
    test_df["Label"] = 0.0
test_df["Label"] = 0.0

train_df["ID"] = train_df["ID"].astype(int)
test_df["ID"] = test_df["ID"].astype(int)




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
rng = np.random.default_rng(SEED)

labels = train_df["Label"].to_numpy()
idx0 = np.where(labels == 0)[0]
idx1 = np.where(labels == 1)[0]
rng.shuffle(idx0)
rng.shuffle(idx1)

val_frac = 0.2
val0 = max(1, int(val_frac * len(idx0)))
val1 = max(1, int(val_frac * len(idx1)))

val_idx = np.concatenate([idx0[:val0], idx1[:val1]])
trn_idx = np.concatenate([idx0[val0:], idx1[val1:]])

rng.shuffle(val_idx)
rng.shuffle(trn_idx)

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
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    cache_dir=os.path.join(CACHE_DIR, "train"),
)

val_dicom_loader = DICOMLoader(
    train_df_val,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=SCAN_CATEGORIES,
    num_imgs=IMG_SEQ,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    max_threads=MAX_THREADS_DICOM_LOADER,
    enable_center_focus=IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode=False,
    cache_dir=os.path.join(CACHE_DIR, "val"),
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

model = DeepScanModel(INPUT_SHAPE, model_name=MODEL_NAME)
model.compile(optimizer=COMPILE_OPTIMIZER, loss=COMPILE_LOSS, metrics=COMPILE_METRICS)

checkpoint_cb = keras.callbacks.ModelCheckpoint(
    BEST_MODEL_H5_DIR,
    monitor=TF_CALL_BACK_BEST_MODEL_MONITOR,
    mode="max",
    save_best_only=True,
    save_weights_only=False,
    verbose=1,
)

fit_kwargs = dict(
    epochs=EPOCHS,
    callbacks=[checkpoint_cb],
    verbose=VERBOSITY,
    validation_data=val_dataset,
)
try:
    history = model.fit(
        train_dataset,
        **fit_kwargs,
        workers=FIT_WORKERS,
        use_multiprocessing=FIT_USE_MULTIPROCESSING,
        max_queue_size=FIT_MAX_QUEUE_SIZE,
    )
except TypeError:
    history = model.fit(train_dataset, **fit_kwargs)

if os.path.exists(BEST_MODEL_H5_DIR):
    model = load_model(
        BEST_MODEL_H5_DIR, custom_objects={"DeepScanModel": DeepScanModel}
    )




## === cell 9
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
    cache_dir=os.path.join(CACHE_DIR, "test"),
)

test_dataset = ScanDataset(
    dicom_loader=test_dicom_loader,
    batch_size=BATCH_SIZE,
    subset=DatasetType.TEST.value,
    shuffle=False,
    debug_mode=False,
)




## === cell 10
def generate_predictions(model, test_dataset, test_df):
    predictions = []

    for batch_idx in range(len(test_dataset)):
        scan_type_1, scan_type_2, scan_type_3, scan_type_4 = test_dataset[batch_idx]
        batch_predictions = model.predict(
            (scan_type_1, scan_type_2, scan_type_3, scan_type_4), verbose=0
        )
        predictions.append(batch_predictions)

    flat_preds = np.concatenate(predictions, axis=0).reshape(-1)
    flat_preds = flat_preds[: len(test_df)]  # safety for last partial batch

    submission = test_df[["ID"]].copy()
    submission["MGMT_value"] = flat_preds.astype(float)
    submission.rename(columns={"ID": "BraTS21ID"}, inplace=True)

    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )

    return submission


submission = generate_predictions(model, test_dataset, test_df)




## === cell 11
submission.info()
print(submission.head())




## === cell 12
submission.to_csv("submission.csv", index=False)

try:
    submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
except Exception:
    pass

print("Wrote submission.csv with shape:", submission.shape)
print(submission.head())
