# Goal

I want you to improve my Kaggle competition solution to increase the score toward a target. Here is the information you need.

# Requirements

- Keep changes minimal unless necessary.
- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (evaluation score improvement); avoid unrelated refactors or stylistic edits.
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

0.5

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.47412) has done: 'I fix the import/runtime crash by removing the unnecessary `keras.utils.vis_utils.plot_model` import that triggers a protobuf `MessageFactory.GetPrototype` error in this environment, while keeping TensorFlow/Keras usage intact. Then I fix the missing external model file issue by training the existing `DeepScanModel` (same architecture/loss/optimizer/loops semantics) on the provided training set and saving/loading from the existing `BEST_MODEL_H5_DIR` path. Finally, I ensure the test dataframe uses correct IDs and the submission is written to `/kaggle/working/submission.csv` with the required `BraTS21ID,MGMT_value` columns and correct row alignment.'
- What this solution (achieved 0.5) has done: 'I fix the two blockers that prevent any submission from being generated: (1) the protobuf/TensorFlow crash during imports by avoiding TensorFlow usage in this environment, and (2) the pydicom pixel decoding failure caused by reading only a small subset of DICOM tags (missing `BitsStored`). To keep core logic intact, I leave your model/dataloader structure in place but switch to a safe, deterministic baseline that still uses your existing train/validation split semantics and produces valid probabilities for the test IDs. This run end-to-end within the time limit and always write `/kaggle/working/submission.csv` with the required columns. Because the previous working score you referenced (~0.474) is below the target (0.5615), this baseline is chosen to be stable and often improves over a broken pipeline, without risky architecture changes.'
- What this solution (achieved 0.5) has done: 'You’re currently submitting a constant prior for every test case, which yields an AUC of ~0.5; to move toward the 0.5615 target with minimal risk, we keep the same split semantics and build a tiny amount of real signal by extracting a very cheap intensity feature from a small, fixed subset of DICOM slices per subject. We then fit a simple, deterministic logistic regression (implemented in NumPy; no new packages) on the training split and predict probabilities for test, falling back to the prior if any subject fails to load. This preserves your overall pipeline structure (dataframes, folds, and probability submission) while adding just enough learned variation to lift AUC above random without heavy compute. The submission file path and required columns remain unchanged.'
- What this solution (achieved 0.5) has done: 'I keep your current deterministic “mean-intensity + NumPy logistic regression” core logic, but add a tiny amount of extra real signal by using all 4 MRI sequences (FLAIR/T1w/T1wCE/T2w) instead of only FLAIR, still with the same slice sampling and the same Newton solver. This is a minimal extension of your existing feature extraction (just repeating the same mean computation per sequence) and typically lifts AUC above the ~0.5 plateau without changing evaluation semantics or adding new packages. I also fix a subtle normalization bug in `_normalization_img` (it should divide by `max-min`, not `max`) which can otherwise compress contrast and reduce separability, while keeping the rest of your DICOM pipeline unchanged. Finally, I ensure prediction-time feature building uses the provided loader argument (not a global), to avoid accidental mismatches and keep train/test processing consistent.'

# 9. Code solution

## === cell 0
import os
import glob
import random
import numpy as np
import pandas as pd

os.environ["PYTHONHASHSEED"] = "0"
random.seed(123)
np.random.seed(123)



## === cell 1
from enum import Enum


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
try:
    import pydicom
    import cv2
    from concurrent.futures import ThreadPoolExecutor, as_completed

    try:
        from tqdm import tqdm
    except Exception:

        def tqdm(x, total=None, desc=None):
            return x

    IS_PYDICOM_IMPORTED = True
except Exception:
    IS_PYDICOM_IMPORTED = False


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
        if not IS_PYDICOM_IMPORTED:
            raise ImportError("pydicom/cv2 not available in this environment.")

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
            loaded_images.extend([zero_image] * (self.__num_imgs - len(loaded_images)))

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

        dicom_file = pydicom.dcmread(
            dicom_path,
            force=True,
            stop_before_pixels=False,
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
        min_val = float(np.min(image))
        max_val = float(np.max(image))
        denom = max_val - min_val
        if denom <= 0:
            return np.zeros_like(image).astype(np.uint8)

        image = image.astype(np.float32) - min_val
        image = image / denom
        return (image * 255.0).clip(0, 255).astype(np.uint8)

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
class MRIType(Enum):
    FLAIR = "FLAIR"
    T1w = "T1w"
    T1wCE = "T1wCE"
    T2w = "T2w"


class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"




## === cell 4
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

os.makedirs(BEST_MODEL_PATH, exist_ok=True)
os.makedirs(LOGS_PATH, exist_ok=True)

DICOM_CACHE_DIR = os.path.join("/kaggle/working", "dicom_cache_v1")
os.makedirs(DICOM_CACHE_DIR, exist_ok=True)




## === cell 5
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

train_df["ID"] = train_df["ID"].astype(int)
index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)
train_df.reset_index(drop=True, inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
test_df["ID"] = test_df["ID"].astype(int)




## === cell 6
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

prior = float(train_split_df["Label"].mean())
prior = float(np.clip(prior, 1e-6, 1 - 1e-6))
print(
    "Train split prior:",
    prior,
    "  (n_train:",
    len(train_split_df),
    "n_val:",
    len(val_split_df),
    ")",
)




## === cell 7
class DeepScanModel:
    pass




## === cell 8
def _sigmoid(x):
    x = np.clip(x, -50.0, 50.0)
    return 1.0 / (1.0 + np.exp(-x))


def fit_logreg_newton(X, y, l2=1.0, max_iter=30):
    """
    Deterministic binary logistic regression with L2 using Newton-Raphson.
    X: (n, d), y: (n,)
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, d = X.shape

    w = np.zeros(d, dtype=np.float64)
    for _ in range(max_iter):
        z = X @ w
        p = _sigmoid(z)
        g = X.T @ (p - y) + l2 * w

        s = p * (1.0 - p)
        Xw = X * s[:, None]
        H = X.T @ Xw + l2 * np.eye(d, dtype=np.float64)

        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            step = g * 0.01
        w = w - step

        if float(np.max(np.abs(step))) < 1e-6:
            break
    return w


def predict_logreg(X, w):
    X = np.asarray(X, dtype=np.float64)
    w = np.asarray(w, dtype=np.float64)
    return _sigmoid(X @ w)


FEATURE_NUM_SLICES = 12  # keep small for runtime
FEATURE_SCANS = ["FLAIR", "T1w", "T1wCE", "T2w"]


def subject_feature_mean_intensity(dicom_loader, row_index, scan_category):
    arr = dicom_loader.load_scan(row_index, scan_category, show_progress=False)
    return float(np.mean(arr))


train_feat_loader = DICOMLoader(
    df=train_split_df,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=FEATURE_SCANS,
    num_imgs=FEATURE_NUM_SLICES,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    enable_center_focus=True,
    id_column_name="ID",
    label_column_name="Label",
    image_format=ImageFormat.WHDC,
    max_threads=MAX_THREADS_DICOM_LOADER,
    debug_mode=False,
    cache_dir=DICOM_CACHE_DIR,
    use_cache=True,
)

val_feat_loader = DICOMLoader(
    df=val_split_df,
    input_path=TRAIN_DATASET_PATH,
    scan_categories=FEATURE_SCANS,
    num_imgs=FEATURE_NUM_SLICES,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    enable_center_focus=True,
    id_column_name="ID",
    label_column_name="Label",
    image_format=ImageFormat.WHDC,
    max_threads=MAX_THREADS_DICOM_LOADER,
    debug_mode=False,
    cache_dir=DICOM_CACHE_DIR,
    use_cache=True,
)

test_feat_loader = DICOMLoader(
    df=test_df,
    input_path=TEST_DATASET_PATH,
    scan_categories=FEATURE_SCANS,
    num_imgs=FEATURE_NUM_SLICES,
    size=IMG_SIZE,
    scale=IMG_SCALE,
    rotate_angle=IMG_ROTATE,
    enable_center_focus=True,
    id_column_name="ID",
    label_column_name="Label",
    image_format=ImageFormat.WHDC,
    max_threads=MAX_THREADS_DICOM_LOADER,
    debug_mode=False,
    cache_dir=DICOM_CACHE_DIR,
    use_cache=True,
)


def build_features(df, loader, scan_categories, default_value):
    feats = np.full(
        (len(df), len(scan_categories)), float(default_value), dtype=np.float64
    )
    ok = np.zeros((len(df),), dtype=bool)

    for i in tqdm(
        range(len(df)), total=len(df), desc=f"Features:{'+'.join(scan_categories)}"
    ):
        row_ok = True
        for j, sc in enumerate(scan_categories):
            try:
                feats[i, j] = subject_feature_mean_intensity(loader, i, sc)
            except Exception:
                feats[i, j] = float(default_value)
                row_ok = False
        ok[i] = row_ok
    return feats, ok


train_feat_raw, train_ok = build_features(
    train_split_df, train_feat_loader, FEATURE_SCANS, default_value=128.0
)
val_feat_raw, val_ok = build_features(
    val_split_df, val_feat_loader, FEATURE_SCANS, default_value=128.0
)
test_feat_raw, test_ok = build_features(
    test_df, test_feat_loader, FEATURE_SCANS, default_value=128.0
)

mu = (
    np.mean(train_feat_raw[train_ok], axis=0)
    if np.any(train_ok)
    else np.mean(train_feat_raw, axis=0)
)
sigma = (
    np.std(train_feat_raw[train_ok], axis=0)
    if np.any(train_ok)
    else np.std(train_feat_raw, axis=0)
)
sigma = np.where(sigma > 1e-6, sigma, 1.0)

train_feat = (train_feat_raw - mu) / sigma
val_feat = (val_feat_raw - mu) / sigma
test_feat = (test_feat_raw - mu) / sigma

X_train = np.concatenate(
    [np.ones((len(train_feat), 1), dtype=np.float64), train_feat], axis=1
)
y_train = train_split_df["Label"].values.astype(np.float64)

w = fit_logreg_newton(X_train, y_train, l2=1.0, max_iter=30)

model = {
    "prior": prior,
    "mu": mu,
    "sigma": sigma,
    "w": w,
    "feature_scans": FEATURE_SCANS,
    "feature_num_slices": FEATURE_NUM_SLICES,
}

X_val = np.concatenate(
    [np.ones((len(val_feat), 1), dtype=np.float64), val_feat], axis=1
)
val_pred = predict_logreg(X_val, w)
print(
    "Val preds mean:",
    float(np.mean(val_pred)),
    "min/max:",
    float(np.min(val_pred)),
    float(np.max(val_pred)),
)




## === cell 9
test_dicom_loader = test_feat_loader
test_dataset = None




## === cell 10
def generate_predictions(model, test_df, loader):
    feats_raw, ok = build_features(
        test_df,
        loader,
        model["feature_scans"],
        default_value=128.0,
    )
    feats = (feats_raw - np.asarray(model["mu"], dtype=np.float64)) / np.asarray(
        model["sigma"], dtype=np.float64
    )
    X = np.concatenate([np.ones((len(feats), 1), dtype=np.float64), feats], axis=1)
    preds = predict_logreg(X, model["w"]).astype(np.float32)

    preds[~ok] = float(model["prior"])
    preds = np.clip(preds, 0.0, 1.0)

    submission = test_df.copy()
    submission["Label"] = preds[: len(submission)]
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    submission["BraTS21ID"] = (
        submission["BraTS21ID"].astype(int).astype(str).str.zfill(5)
    )
    return submission[["BraTS21ID", "MGMT_value"]]


submission = generate_predictions(model, test_df, test_feat_loader)




## === cell 11
submission.info()
print(submission.head())




## === cell 12
submission.to_csv(SUBMISSION_DATASET_DF_DIR, index=False)
print("Wrote submission to:", SUBMISSION_DATASET_DF_DIR)
print(pd.read_csv(SUBMISSION_DATASET_DF_DIR).head())
