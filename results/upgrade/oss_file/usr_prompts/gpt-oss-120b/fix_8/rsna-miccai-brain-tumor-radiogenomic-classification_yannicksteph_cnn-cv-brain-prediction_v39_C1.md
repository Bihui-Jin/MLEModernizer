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

# 8. Previous improvement plans

- What this solution (achieved 0.5) has done: 'The fix removes the failing TensorFlow import, replaces the missing pretrained model with a lightweight logistic‑regression model trained on simple mean‑intensity features extracted from the DICOM files, and updates the prediction code to use this model. This eliminates the file‑not‑found and protobuf errors while providing a reasonable baseline that moves the AUC toward the target score.'
- What this solution (achieved 0.5) has done: 'I fixed the import errors by providing a fallback `Sequence` class, removed the unused DICOM loading for the test set, and enriched the feature engineering to include both mean and standard‑deviation of each modality. These extra simple features should raise the AUC closer to the target while keeping the core logistic‑regression approach unchanged.'
- What this solution (achieved 0.5) has done: 'I’ve expanded the handcrafted MRI features by adding median, minimum and maximum intensity values for each modality and wrapped the logistic‑regression model in a `StandardScaler` pipeline so the features are properly normalised. These modest extensions give the classifier a richer view of the data and should lift the AUC closer to the target score.'
- What this solution (achieved 0.5) has done: 'I import the missing classes (`Sequence`, `make_pipeline`, `StandardScaler`) so the dataset wrapper and the sklearn pipeline are defined, then the model variable be available for generating predictions and writing the submission file.'

# 9. Code solution

## === cell 0
IS_PYDICOM_IMPORTED = True
if IS_PYDICOM_IMPORTED:
    """
    This block checks if the code is being run with PYDICOM lib.
    """
    try:
        import pydicom
        import os
        import glob
        import cv2
        from enum import Enum
        import numpy as np
        import pandas as pd
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

        from collections.abc import Sequence
    except ImportError as e:
        print(f"Missing some imports: {e}")

    class ImageFormat(Enum):
        WHDC = "W-H-D-C"  # (Width, Height, Depth, Channel)
        DWHC = "D-W-H-C"  # (Depth, Width, Height, Channel)

        @staticmethod
        def swap_dimensions(image, image_format):
            if image_format == ImageFormat.DWHC:
                return np.transpose(image, (2, 1, 0, 3))
            elif image_format == ImageFormat.WHDC:
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

        @property
        def scan_categories(self):
            return self.__scan_categories

        @property
        def len(self):
            return len(self.__df)

        def gel_label(self, row):
            return self.__df.loc[row, self.__label_column_name]

        def format(self, images, typ):
            if typ == "normalize" and self.__image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)
            elif typ == "default" and self.__image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)
            else:
                return images

        def load_all_scans(self, row, show_progress=True):
            all_images = {}
            with ThreadPoolExecutor(self.__max_threads) as executor:
                futures = {
                    executor.submit(self.load_scan, row, cat, False): cat
                    for cat in self.__scan_categories
                }
                if show_progress:
                    pbar = tqdm(
                        total=len(self.__scan_categories), desc="Loading scan types"
                    )
                for fut in as_completed(futures):
                    cat = futures[fut]
                    img = fut.result()
                    if img is not None:
                        all_images[cat] = img
                    if show_progress:
                        pbar.update(1)
                if show_progress:
                    pbar.close()
            return {k: all_images.get(k, []) for k in self.__scan_categories}

        def load_scan(self, row, scan_category, show_progress=True):
            patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)
            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")
            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")), key=self.__image_file_sorter
            )
            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")
            image_files = self._select_subset_image_files(image_files)
            loaded_images = []
            with ThreadPoolExecutor(self.__max_threads) as executor:
                iterator = executor.map(self._load_dicom_image, image_files)
                if show_progress:
                    iterator = tqdm(
                        iterator, total=len(image_files), desc="Loading images"
                    )
                for img in iterator:
                    if img is not None:
                        loaded_images.append(img)
            if not loaded_images:
                raise ValueError("No images were loaded.")
            if self.__num_imgs is not None:
                while len(loaded_images) < self.__num_imgs:
                    loaded_images.append(np.zeros_like(loaded_images[0]))
            loaded_images = np.array(loaded_images)
            return self.format(loaded_images, "default")

        def _select_subset_image_files(self, image_files):
            if self.__enable_center_focus and self.__num_imgs is not None:
                middle = len(image_files) // 2
                half = self.__num_imgs // 2
                return image_files[
                    max(0, middle - half) : min(len(image_files), middle + half)
                ]
            elif self.__num_imgs is not None:
                return image_files[: self.__num_imgs]
            return image_files

        def _load_dicom_image(self, dicom_path):
            if not os.path.exists(dicom_path):
                raise FileNotFoundError(f"{dicom_path} does not exist.")
            dicom_file = pydicom.dcmread(dicom_path)
            image = dicom_file.pixel_array.astype(np.float32)
            image = self._rotate_img(image)
            image = self._normalization_img(image)
            image = self._crop_img(image)
            image = self._resize_img(image)
            return np.expand_dims(image, axis=-1)

        def _resize_img(self, image):
            w, h = self.__size
            return cv2.resize(image, (w, h), interpolation=cv2.INTER_AREA)

        def _crop_img(self, image):
            if self.__scale <= 0:
                return image
            cx, cy = image.shape[1] / 2, image.shape[0] / 2
            w_s, h_s = image.shape[1] * self.__scale, image.shape[0] * self.__scale
            left, right = cx - w_s / 2, cx + w_s / 2
            top, bottom = cy - h_s / 2, cy + h_s / 2
            return image[int(top) : int(bottom), int(left) : int(right)]

        def _rotate_img(self, image):
            if self.__rotate_angle <= 0:
                return image
            h, w = image.shape[:2]
            center = (w / 2, h / 2)
            M = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)
            return cv2.warpAffine(image, M, (w, h))

        def _normalization_img(self, image):
            min_val, max_val = np.min(image), np.max(image)
            if max_val == 0:
                return np.zeros_like(image).astype(np.uint8)
            norm = (image - min_val) / max_val
            return (norm * 255).astype(np.uint8)




## === cell 1
SEED = 42

EXCLUDED_IDS = [109, 123, 709]  # corresponds to ['00109','00123','00709']

SCAN_CATEGORIES = ["FLAIR", "T1w", "T1wCE", "T2w"]

BASE_DIR = os.path.join("data", "rsna-miccai-brain-tumor-radiogenomic-classification")

TRAIN_DATASET_PATH = os.path.join(BASE_DIR, "train")
TEST_DATASET_PATH = os.path.join(BASE_DIR, "test")
TRAIN_DATASET_DF_DIR = os.path.join(BASE_DIR, "train_labels.csv")
TEST_DATASET_DF_DIR = os.path.join(BASE_DIR, "sample_submission.csv")




## === cell 2
class InternalDebug:
    def __init__(self, debug_mode=False, debug_prefix=None):
        self.__debug_mode = debug_mode
        self.__debug_prefix = debug_prefix

    @property
    def __prefix(self):
        return self.__debug_prefix if self.__debug_prefix is not None else ""

    def log(self, *args):
        if self.__debug_mode:
            if self.__debug_prefix:
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

    def __len__(self):
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))

    def __getitem__(self, ids):
        self.__debug.log("== __getitem__ ==")
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
            for j, (scan_type, images) in enumerate(batch_x_image_paths.items()):
                batches_x[j].append(images)
        batch_x = [np.array(b) for b in batches_x]
        batch_y = np.array(batches_y)
        if self.__is_trainable:
            return batch_x, batch_y
        else:
            return batch_x

    def on_epoch_end(self):
        if self.__shuffle:
            np.random.shuffle(self.__indices)




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
np.random.seed(SEED)


def extract_features(df, base_path):
    """
    Compute mean, std, median, min and max over **all** DICOM slices of each modality.
    Using information from every slice provides a more robust description of the scan,
    which is expected to improve the AUC toward the target score.
    """
    features = []
    for _, row in df.iterrows():
        patient_id = str(row["ID"]).zfill(5)
        modality_feats = []
        for mod in SCAN_CATEGORIES:
            folder = os.path.join(base_path, patient_id, mod)
            if not os.path.isdir(folder):
                modality_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            files = sorted(glob.glob(os.path.join(folder, "*")))
            if not files:
                modality_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            all_pixels = []
            for f in files:
                try:
                    dcm = pydicom.dcmread(f)
                    img = dcm.pixel_array.astype(np.float32)
                    all_pixels.append(img.ravel())
                except Exception:
                    continue
            if not all_pixels:
                modality_feats.extend([0.0, 0.0, 0.0, 0.0, 0.0])
                continue
            concat = np.concatenate(all_pixels)
            modality_feats.append(np.mean(concat))
            modality_feats.append(np.std(concat))
            modality_feats.append(np.median(concat))
            modality_feats.append(np.min(concat))
            modality_feats.append(np.max(concat))
        features.append(modality_feats)
    return np.array(features, dtype=np.float32)


train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
train_df["ID_int"] = train_df["ID"].astype(int)
train_df = train_df[~train_df["ID_int"].isin(EXCLUDED_IDS)].reset_index(drop=True)
train_df.drop(columns=["ID_int"], inplace=True)

test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

X_train = extract_features(train_df, TRAIN_DATASET_PATH)
y_train = train_df["Label"].values

X_test = extract_features(test_df, TEST_DATASET_PATH)

from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

simple_model = make_pipeline(
    StandardScaler(),
    LogisticRegression(
        max_iter=1000,
        n_jobs=-1,
        solver="lbfgs",
        C=5.0,  # slightly less regularisation to boost discrimination
        class_weight="balanced",
        random_state=SEED,
    ),
)

simple_model.fit(X_train, y_train)




## --- ERROR in cell 5, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/4105307955.py in <cell line: 0>()
     43 
     44 # Load training labels
---> 45 train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
     46 train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)
     47 train_df["ID_int"] = train_df["ID"].astype(int)

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in read_csv(filepath_or_buffer, sep, delimiter, header, names, index_col, usecols, dtype, engine, converters, true_values, false_values, skipinitialspace, skiprows, skipfooter, nrows, na_values, keep_default_na, na_filter, verbose, skip_blank_lines, parse_dates, infer_datetime_format, keep_date_col, date_parser, date_format, dayfirst, cache_dates, iterator, chunksize, compression, thousands, decimal, lineterminator, quotechar, quoting, doublequote, escapechar, comment, encoding, encoding_errors, dialect, on_bad_lines, delim_whitespace, low_memory, memory_map, float_precision, storage_options, dtype_backend)
   1024     kwds.update(kwds_defaults)
   1025 
-> 1026     return _read(filepath_or_buffer, kwds)
   1027 
   1028 

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _read(filepath_or_buffer, kwds)
    618 
    619     # Create the parser.
--> 620     parser = TextFileReader(filepath_or_buffer, **kwds)
    621 
    622     if chunksize or iterator:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in __init__(self, f, engine, **kwds)
   1618 
   1619         self.handles: IOHandles | None = None
-> 1620         self._engine = self._make_engine(f, self.engine)
   1621 
   1622     def close(self) -> None:

/usr/local/lib/python3.11/dist-packages/pandas/io/parsers/readers.py in _make_engine(self, f, engine)
   1878                 if "b" not in mode:
   1879                     mode += "b"
-> 1880             self.handles = get_handle(
   1881                 f,
   1882                 mode,

/usr/local/lib/python3.11/dist-packages/pandas/io/common.py in get_handle(path_or_buf, mode, encoding, compression, memory_map, is_text, errors, storage_options)
    871         if ioargs.encoding and "b" not in ioargs.mode:
    872             # Encoding
--> 873             handle = open(
    874                 handle,
    875                 ioargs.mode,

FileNotFoundError: [Errno 2] No such file or directory: 'data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv'

## === cell 6
model = simple_model




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4246081281.py in <cell line: 0>()
----> 1 model = simple_model
      2 
      3 

NameError: name 'simple_model' is not defined

## === cell 7
def generate_predictions(model, X_test, test_df):
    probs = model.predict_proba(X_test)[:, 1]
    submission = test_df.copy()
    submission["MGMT_value"] = probs
    submission.rename(columns={"ID": "BraTS21ID"}, inplace=True)
    submission = submission[["BraTS21ID", "MGMT_value"]]
    return submission


submission = generate_predictions(model, X_test, test_df)




## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/190355821.py in <cell line: 0>()
      8 
      9 
---> 10 submission = generate_predictions(model, X_test, test_df)
     11 
     12 

NameError: name 'model' is not defined

## === cell 8
submission.info()




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4243313552.py in <cell line: 0>()
----> 1 submission.info()
      2 
      3 

NameError: name 'submission' is not defined

## === cell 9
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
