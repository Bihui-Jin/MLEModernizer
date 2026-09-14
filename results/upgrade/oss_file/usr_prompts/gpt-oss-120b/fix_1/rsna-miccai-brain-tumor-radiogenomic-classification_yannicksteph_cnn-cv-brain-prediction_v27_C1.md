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

0.5035271120176781

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import numpy as np
import pandas as pd
import os
import glob
import numpy as np
import random

import pydicom

from enum import Enum

import cv2

import tensorflow as tf
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.metrics import AUC
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import Sequence
from tensorflow.keras.models import (
    Model,
    load_model
)
from tensorflow.keras.callbacks import (
    Callback, 
    ModelCheckpoint, 
    EarlyStopping
)
from tensorflow.keras.layers import (
    Input,
    Conv3D,
    BatchNormalization,
    MaxPooling3D,
    MaxPool3D,
    Flatten,
    Dense,
    Dropout,
    Resizing,
    Rescaling,
    RandomFlip,
    RandomRotation,
    concatenate,
    GlobalAveragePooling3D,
    Reshape,
    LeakyReLU,
    ReLU
)

import keras
from keras.utils.vis_utils import plot_model



## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 7
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
        from concurrent.futures import ThreadPoolExecutor, as_completed
        from tqdm import tqdm

    except ImportError:
        print(f"Missing some imports: {ImportError}")

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

            Parameters:
            ----------
            image : numpy.ndarray
                Input image with dimensions to be swapped.
            permutation_type : str
                The category of the scan to load.
                - 'WxHxDxC' = Width-Height-Depth-Channel'
                - 'DxWxHxC' = (Depth-Width-Height-Channel)

            Returns:
            numpy.ndarray: Image with swapped dimensions.
            """
            if image_format == ImageFormat.DWHC:
                image = np.transpose(image, (2, 1, 0, 3))

            elif image_format == ImageFormat.WHDC:
                image = np.transpose(image, (2, 1, 0, 3))

            return image

    class DICOMLoader():
        def __init__(self,
            df,
            input_path,
            scan_categories,
            num_imgs                        = None,
            size                            = (224, 224),
            scale                           = 1.0,
            rotate_angle                    = 0,
            enable_center_focus             = False,
            id_column_name                  = "ID",
            label_column_name               = "Label",
            image_format                    = ImageFormat.WHDC,
            max_threads                     = 8,
            image_file_sorter               = lambda x: int(x[:-4].split("-")[-1]),
            debug_mode                      = False
        ):
            """
            This class is designed for loading DICOM images from a given directory and
            creating a dataset for medical image analysis.

            Parameters:
            -----------
            df : DataFrame
                The DataFrame containing metadata and labels for DICOM images.
            input_path : str
                The path to the directory containing DICOM image files.
            scan_categories : list
                A list of scan categories to include in the dataset.
            num_imgs : int, optional
                The number of images to load per scan. If specified, must be divisible
                by 2 when 'enable_center_focus' is True. Default is None.
            size : tuple, optional
                The size to which images should be resized. Default is (224, 224).
            scale : float, optional
                The scaling factor applied to the images. Default is 1.0.
            rotate_angle : int, optional
                The angle in degrees by which images should be rotated. Default is 0.
            enable_center_focus : bool, optional
                If True, focus on the central images when 'num_imgs' is specified.
                Default is False.
            id_column_name : str, optional
                The name of the column containing unique IDs in 'df'. Default is "ID".
            label_column_name : str, optional
                The name of the column containing labels in 'df'. Default is "Label".
            image_format : ImageFormat, optional
                The format of the DICOM images (e.g., ImageFormat.WHDC). Default is
                ImageFormat.WHDC.
            max_threads : int, optional
                The maximum number of threads to use for image loading. Default is 8.
            image_file_sorter : function, optional
                A function used to sort image files. Default sorts by numeric value
                at the end of the filename.
            debug_mode: bool
                Debug mode

            Raises:
            -------
            ValueError
                - If 'num_imgs' is not divisible by 2 when 'enable_center_focus' is True.
                - If 'rotate_angle' is not in the range [0, 360].
                - If 'id_column_name' or 'label_column_name' is not in 'df.columns'.
            """
            if num_imgs is not None and num_imgs % 2 != 0 and enable_center_focus:
                raise ValueError("num_imgs must be divisible by 2 for central image")

            if not (0 <= rotate_angle <= 360):
                raise ValueError("Rotation value must be between 0 and 360")

            for col in [id_column_name, label_column_name]:
                if col not in df.columns:
                    raise ValueError(f"Columns {col} must be in dataset")

            self.__df                  = df.copy()
            self.__num_imgs            = num_imgs
            self.__id_column_name      = id_column_name
            self.__label_column_name   = label_column_name
            self.__input_path          = input_path
            self.__scan_categories     = scan_categories
            self.__max_threads         = max_threads
            self.__size                = size
            self.__scale               = scale
            self.__rotate_angle        = rotate_angle
            self.__image_format        = image_format
            self.__image_file_sorter   = image_file_sorter
            self.__enable_center_focus = enable_center_focus
            self.__debug               = InternalDebug(debug_mode=debug_mode)



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
            """
            Returns the length of the DataFrame.

            Returns
            -------
            int
                The number of rows in the DataFrame.
            """
            return len(self.__df)


        def get_id(self, row):
            """
            Retrieves the ID for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.

            Returns
            -------
            str
                The ID corresponding to the row.
            """
            return self.__df.loc[row, self.__id_column_name]

        def gel_label(self, row):
            """
            Retrieves the label for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.

            Returns
            -------
            str
                The label corresponding to the row.
            """
            return self.__df.loc[row, self.__label_column_name]


        def format(self, images, type):
            if type == "normalize" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.DWHC)

            elif type == "default" and self.image_format == ImageFormat.WHDC:
                return ImageFormat.swap_dimensions(images, ImageFormat.WHDC)

            else:
                return images


        def load_scan(self, row, scan_category, show_progress=True):
            """
            Loads a scan for a given row and scan category.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            scan_category : str
                The category of the scan to load.
            show_progress : bool, optional
                Whether to show a progress bar.

            Returns
            -------
            list
                A list of loaded images.
            """
            self.__debug.log("== load_scan ==")
            patient_id = str(self.__df.loc[row, self.__id_column_name]).zfill(5)
            scans_path = os.path.join(self.__input_path, patient_id, scan_category)

            if not os.path.exists(scans_path):
                raise FileNotFoundError(f"The folder {scans_path} doesn't exist.")

            image_files = sorted(
                glob.glob(os.path.join(scans_path, "*")),
                key = self.__image_file_sorter
            )

            if not image_files:
                raise ValueError(f"No image files found in {scans_path}.")

            image_files = self._select_subset_image_files(image_files)

            loaded_images = []

            with ThreadPoolExecutor(self.__max_threads) as executor:
                image_data_iterable = executor.map(self._load_dicom_image, image_files)

                if show_progress:
                    image_data_iterable = tqdm(image_data_iterable, total=len(image_files), desc="Loading images")

                for image_data in image_data_iterable:
                    if image_data is not None:
                        loaded_images.append(image_data)

            if not loaded_images:
                raise ValueError("No images were loaded, and num_imgs is set. Cannot proceed.")

            if self.__num_imgs is not None:
                while len(loaded_images) < self.__num_imgs:
                    zero_image = np.zeros_like(loaded_images[0])
                    loaded_images.append(zero_image)

            loaded_images = np.array(loaded_images)

            return self.format(loaded_images, "default")

        def load_all_scans(
            self,
            row,
            show_progress=True
        ):
            """
            Loads all scans for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            show_progress : bool, optional
                Whether to show a progress bar.

            Returns
            -------
            dict
                A dictionary containing all loaded images, categorized by scan type.
            """
            self.__debug.log("== load_all_scans ==")
            all_images = {}

            with ThreadPoolExecutor(self.__max_threads) as executor:
                future_to_scan_category = {
                    executor.submit(
                        self.load_scan,
                        row,
                        scan_category,
                        False
                    ): scan_category for scan_category in self.__scan_categories
                }

                if show_progress:
                    progress_bar = tqdm(total=len(self.__scan_categories), desc="Loading scan types")

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


        def _show_scan(self, scan_category, images, color_map):
            images = self.format(images, "normalize")
            show_text("h4", scan_category, False)
            show_images(images, color_map=color_map)

        def show(self, row, scan_category, color_map='gray'):
            """
            Displays the images for a given row and scan category.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            scan_category : str
                The category of the scan to display.
            color_map : str, optional
                The color map to use for displaying the images.

            """
            images = self.load_scan(row, scan_category)
            self._show_scan(scan_category, images, color_map)


        def show_all(self, row, color_map='gray'):
            """
            Displays all images for a given row in the DataFrame.

            Parameters
            ----------
            row : int
                The row index in the DataFrame.
            color_map : str, optional
                The color map to use for displaying the images.

            """
            loaders_images = self.load_all_scans(row)
            for scan_category, images in loaders_images.items():
                self._show_scan(scan_category, images, color_map)

        def summary(self, train_dataset=None):
            super().summary()
            print("Additional summary details specific:")
            scan_category = self.__scan_categories[0]

            if scan_category is not None:
                images = self.load_scan(0, scan_category)
                print("\n")

                if images is not None:
                    print("Size:", self.len)
                    print("Images Shape:", images.shape)

                else:
                    print("Error: Loading images are empty.")
            else:
                print("Error: scan_category is empty.")
            print("=" * 50)


        def _load_dicom_image(self, dicom_path):
            """
            Loads a DICOM image from a given path and applies various transformations
            such as VOI LUT, rotation, normalization, cropping, and resizing.

            Parameters
            ----------
            dicom_path : str
                The path to the DICOM file.

            Returns
            -------
            2D array
                The transformed DICOM image.

            Raises
            ------
            FileNotFoundError
                If the specified DICOM file does not exist.
            IOError
                If an error occurs while reading the DICOM file.

            Notes
            -----
            The method performs the following transformations in order:
            1. Applies Value of Interest Lookup Table (VOI LUT) for better visibility.
            2. Rotates the image based on the specified angle.
            3. Normalizes the pixel values in the image.
            4. Crops the image to focus on the region of interest.
            5. Resizes the image to the specified dimensions.
            """
            self.__debug.log("== _load_dicom_image ==")
            if not os.path.exists(dicom_path):
                raise FileNotFoundError(f"File {dicom_path} does not exist.")

            try:
                dicom_file = pydicom.dcmread(dicom_path)
            except Exception as e:
                raise IOError(f"An error occurred while reading the DICOM file: {e}")

            image = dicom_file.pixel_array
            self.__debug.log("Pixel array shape:", image.shape)

            image = self._rotate_img(image)

            image = self._normalization_img(image)

            image = self._crop_img(image)

            image = self._resize_img(image)

            image = np.expand_dims(image, axis=-1)

            self.__debug.log("Is normalized: ", self._is_normalized(image))
            self.__debug.log("Image shape: ", image.shape)

            return image

        def _resize_img(self, image):
            w, h = self.__size

            return cv2.resize(image, (w, h), interpolation = cv2.INTER_AREA)

        def _crop_img(self, image):
            """
            Crops and resizes a given image.

            Parameters
            ----------
            image : 2D array
                The image to be cropped and resized.

            Returns
            -------
            2D array
                The cropped and resized image.
            """
            if self.__scale <= 0:
                return image

            center_x, center_y = image.shape[1] / 2, image.shape[0] / 2

            width_scaled, height_scaled = image.shape[1] * self.__scale, image.shape[0] * self.__scale

            left_x, right_x = center_x - width_scaled / 2, center_x + width_scaled / 2
            top_y, bottom_y = center_y - height_scaled / 2, center_y + height_scaled / 2

            return image[int(top_y):int(bottom_y), int(left_x):int(right_x)]

        def _rotate_img(self, image):
          """
          Rotates the image array by the specified angle.

          Parameters
          ----------
          image : ndarray
              The original image array.

          Returns
          -------
          ndarray
              The rotated image array.
          """
          if self.__rotate_angle <= 0:
              return image

          height, width = image.shape[:2]
          center = (width / 2, height / 2)
          rotation_matrix = cv2.getRotationMatrix2D(center, self.__rotate_angle, 1.0)

          return cv2.warpAffine(image, rotation_matrix, (width, height))

        def _normalization_img(self, image):
            """
            Normalizes the image array to a range of 0 to 255.

            Parameters
            ----------
            image : ndarray
                The original image array.

            Returns
            -------
            ndarray
                The normalized image array.
            """
            min_val = np.min(image)
            max_val = np.max(image)

            if max_val == 0:
                return np.zeros_like(image).astype(np.uint8)

            image = image - min_val
            image = image / max_val

            return (image * 255).astype(np.uint8)

        def _select_subset_image_files(self, image_files):
            """
            Selects a subset of image files based on the object's attributes.

            Parameters
            ----------
            image_files : list
                List of image files to select from.

            Returns
            -------
            list
                A subset of the original list of image files.
            """
            if self.__enable_center_focus and self.__num_imgs is not None:
                middle = len(image_files) // 2
                num_imgs2 = self.__num_imgs // 2
                p1 = max(0, middle - num_imgs2)
                p2 = min(len(image_files), middle + num_imgs2)
                return image_files[p1:p2]

            elif self.__num_imgs is not None:
                return image_files[:self.__num_imgs]

            else:
                return image_files


        def _is_normalized(self, image):
            min_value = np.min(image)
            max_value = np.max(image)

            return min_value >= 0.0 and max_value <= 255


## === cell 8
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


## === cell 9
class ScanDataset(Sequence):
    def __init__(
        self,
        dicom_loader,
        batch_size,
        subset     = "train",
        shuffle    = True,
        debug_mode = False
    ):
        """
        Initializes the ScanDataset object.

        Parameters
        ----------
        dicom_loader : object
            The DICOMLoader object to load DICOM images.
        batch_size : int
            The size of each batch.
        subset: Subset of the data to return.
          One of "training", "validation" or "other".
          training and validation give the y_bath
        shuffle : bool, optional
            Whether to shuffle the dataset.
        debug_mode : bool, optional
            Whether to print debug information.
        """
        self.__dicom_loader = dicom_loader
        self.__batch_size   = batch_size
        self.__is_trainable = subset.lower() in ["validation", "train"]
        self.__shuffle      = shuffle
        self.__debug        = InternalDebug(debug_mode=debug_mode)
        self.__indices      = np.arange(self.__dicom_loader.len)
        
        if self.__shuffle:
            np.random.shuffle(self.__indices)
        

    def show_batch(
        self,
        row,
        columns=None,
        figure_size=(5, 5),
        color_map='hot'
    ):
        if self.__is_trainable:
            x_batch, y_batch = self[row]
        else:
            x_batch = self[row]
            y_batch = None

        if columns == None:
            columns = self.__dicom_loader.num_imgs
            
        num_input_tensors = len(self.__dicom_loader.scan_categories)

        for i in range(len(x_batch[0])):
            label_info = f" Label: {y_batch[i]}" if y_batch is not None else ""

            for j in range(num_input_tensors):
                images = x_batch[j][i]
                scan_type = self.__dicom_loader.scan_categories[j]

                if self.__dicom_loader.image_format == ImageFormat.WHDC:
                    images = ImageFormat.swap_dimensions(images, ImageFormat.DWHC)

                labels = [
                            f"Batch: {i + 1} \nImg: {k + 1} \nType: {scan_type} \nLabel:{label_info}"
                            for k in range(len(images))
                         ]

                show_images(
                    images,
                    y=labels,
                    columns=columns,
                    figure_size=figure_size,
                    color_map=color_map
                )

    def summary(self, train_dataset=None):
        super().summary()

        print("Additional summary details specific:")

        if self.__is_trainable:
            batch_x, batch_y = self[0]
        else:
            batch_x = self[0]

        print("Batch_x format:")
        for i, x in enumerate(batch_x):
          print(f"- Scan type {i+1}: {x.shape}")

        if self.__is_trainable:
            print(f"Batch_y format: {batch_y.shape}")
            
        print("=" * 40)

    def on_epoch_end(self):
        """
        Shuffles the dataset at the end of each epoch if shuffle is True.
        """
        if self.__shuffle:
            np.random.shuffle(self.__indices)


    def __getitem__(self, ids):
        self.__debug.log("== __getitem__ ==")
        """
        Retrieves a batch of data by batch index.

        Parameters
        ----------
        ids : int
            The batch index.

        Returns
        -------
        tuple
            A tuple containing the batch of images and labels.
        """
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size

        self.__debug.log(f"Batch ID: {ids}")
         
        batch_indices = self.__indices[from_id: to_id]
   
        self.__debug.log("batch_indices:", batch_indices.tolist())

        batches_y = []

        batches_x = [[] for _ in range(len(self.__dicom_loader.scan_categories))]

        for i in batch_indices:
            self.__debug.log("Processing batch index:", i)

            label = self.__dicom_loader.gel_label(i)
            batches_y.append(label)
            self.__debug.log("Label:", label)

            batch_x_image_paths = self.__dicom_loader.load_all_scans(i, show_progress=False)

            for j, (scan_type, images) in enumerate(batch_x_image_paths.items()):
                self.__debug.log("Processing Scan Type:", scan_type, "Number of Images Loaded:", len(images))
                batches_x[j].append(images)

        batch_x = [np.array(b) for b in batches_x]
        batch_y = np.array(batches_y)

        self.__debug.log(f"Final batch shapes - batch_x: {[x.shape for x in batch_x]}, batch_y: {batch_y}")

        if self.__is_trainable:
            return batch_x, batch_y
        else:
            return batch_x

    def __len__(self):
        """
        Calculates the number of batches in the dataset.

        Returns
        -------
        int
            The number of batches.
        """
        return int(np.ceil(self.__dicom_loader.len / self.__batch_size))

## === cell 10
class MRIType(Enum):
    FLAIR = "FLAIR"
    T1w = "T1w"
    T1wCE = "T1wCE"
    T2w = "T2w"
    
class DatasetType(Enum):
    TRAIN = "train"
    VALIDATION = "validation"
    TEST = "test"

## === cell 11
VERSION         = "V1"
VERBOSITY       = 2
SEED            = 123
SCAN_CATEGORIES = [mri_type.value for mri_type in MRIType]
EXCLUDED_IDS    = [109, 123, 709]

RUN_DIR = './run'
INPUT_PATH = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

TRAIN_DATASET_PATH = INPUT_PATH + "/train"
TRAIN_DATASET_DF_DIR = INPUT_PATH + "/train_labels.csv"

TEST_DATASET_PATH = INPUT_PATH + "/test"
TEST_DATASET_DF_DIR = INPUT_PATH + "/sample_submission.csv"

SUBMISSION_DATASET_DF_DIR = '/kaggle/working/submission.csv'

LOGS_PATH = f'{RUN_DIR}/logs'
BEST_MODEL_PATH = f'{RUN_DIR}/models'
BEST_MODEL_H5_DIR = f'{BEST_MODEL_PATH}/model_{VERSION}.h5'

NUM_SPLIT_FOLDS = 5
SELECTED_VALIDATION_FOLD = 1

MAX_THREADS_DICOM_LOADER = 8

IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_CHAN = (128, 128, 1)
IMG_SIZE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE)

IMG_SEQ                  = 32
IMG_SCALE                = .85
IMG_ROTATE               = 0 
IMG_ENABLE_CENTRAL_FOCUS = True

SHUFFLE    = True

AUGMENTATION_FRACTION               = 5
AUGMENTATION_CROP_LIMITS            = (0.85, 0.95)
AUGMENTATION_ROTATION_LIMITS        = (4, 12)
AUGMENTATION_TRANSLATION_X_Y_LIMITS = ((2, 6), (0, 2))
AUGMENTATION_BLUR                   = (0, 0.15)
AUGMENTATION_CONSTRAST_BRIGHT       = ((0.8, 1.2),(-2, 2))

INPUT_SHAPE = (IMG_WIDTH_SIZE, IMG_HEIGHT_SIZE, IMG_SEQ, IMG_CHAN) # Format sample: (128, 128, 64, 1)

MODEL_NAME = "Mult3DCNN4Input"
BATCH_SIZE = 8
EPOCHS     = 26

COMPILE_OPTIMIZER = SGD(learning_rate =0.001)
COMPILE_LOSS = 'binary_crossentropy'
COMPILE_METRICS = [AUC(name='auc')]

TF_CALL_BACK_BEST_MODEL_MONITOR  = "val_auc"
TF_CALL_BACK_EARLY_STOP_MONITOR  = "auc"
TF_CALL_BACK_EARLY_STOP_PATIENTE = 6

## === cell 12
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns = { "BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df['ID'].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)

train_df.reset_index(drop=True, inplace=True)


test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns = { "BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)



## === cell 13
class DeepScanModel(Model):
    def __init__(self, input_shape, model_name="My3DCNNModel"):
        self.input_layers = [Input(shape=input_shape) for _ in range(4)]

        self.cnn_models = [self.build_cnn_branch(input_layer) for input_layer in self.input_layers]

        concatenated = concatenate(self.cnn_models)

        x = self.build_head(concatenated)

        super(DeepScanModel, self).__init__(inputs=self.input_layers, outputs=x, name=model_name)

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

    def show_graph(self):
        display(plot_model(self, show_shapes=True, show_layer_names=True))

## === cell 14
test_dicom_loader = DICOMLoader(
    test_df,
    input_path          = TEST_DATASET_PATH,
    scan_categories     = SCAN_CATEGORIES,
    num_imgs            = IMG_SEQ,
    size                = IMG_SIZE,
    scale               = IMG_SCALE,
    rotate_angle        = IMG_ROTATE,
    max_threads         = MAX_THREADS_DICOM_LOADER,
    enable_center_focus = IMG_ENABLE_CENTRAL_FOCUS,
    debug_mode          = False
)

test_dataset = ScanDataset(
    dicom_loader = test_dicom_loader,
    batch_size   = BATCH_SIZE,
    subset       = DatasetType.TEST.value,
    shuffle      = False,
    debug_mode   = False
)

## === cell 15
model = load_model("/kaggle/input/model-v2/model_V2.h5", custom_objects={'DeepScanModel': DeepScanModel})

## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_11/1675406691.py in <cell line: 0>()
----> 1 model = load_model("/kaggle/input/model-v2/model_V2.h5", custom_objects={'DeepScanModel': DeepScanModel})

/usr/local/lib/python3.11/dist-packages/keras/src/saving/saving_api.py in load_model(filepath, custom_objects, compile, safe_mode)
    194         )
    195     if str(filepath).endswith((".h5", ".hdf5")):
--> 196         return legacy_h5_format.load_model_from_hdf5(
    197             filepath, custom_objects=custom_objects, compile=compile
    198         )

/usr/local/lib/python3.11/dist-packages/keras/src/legacy/saving/legacy_h5_format.py in load_model_from_hdf5(filepath, custom_objects, compile)
    114     opened_new_file = not isinstance(filepath, h5py.File)
    115     if opened_new_file:
--> 116         f = h5py.File(filepath, mode="r")
    117     else:
    118         f = filepath

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in __init__(self, name, mode, driver, libver, userblock_size, swmr, rdcc_nslots, rdcc_nbytes, rdcc_w0, track_order, fs_strategy, fs_persist, fs_threshold, fs_page_size, page_buf_size, min_meta_keep, min_raw_keep, locking, alignment_threshold, alignment_interval, meta_block_size, **kwds)
    562                                  fs_persist=fs_persist, fs_threshold=fs_threshold,
    563                                  fs_page_size=fs_page_size)
--> 564                 fid = make_fid(name, mode, userblock_size, fapl, fcpl, swmr=swmr)
    565 
    566             if isinstance(libver, tuple):

/usr/local/lib/python3.11/dist-packages/h5py/_hl/files.py in make_fid(name, mode, userblock_size, fapl, fcpl, swmr)
    236         if swmr and swmr_support:
    237             flags |= h5f.ACC_SWMR_READ
--> 238         fid = h5f.open(name, flags, fapl=fapl)
    239     elif mode == 'r+':
    240         fid = h5f.open(name, h5f.ACC_RDWR, fapl=fapl)

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/_objects.pyx in h5py._objects.with_phil.wrapper()

h5py/h5f.pyx in h5py.h5f.open()

FileNotFoundError: [Errno 2] Unable to synchronously open file (unable to open file: name = '/kaggle/input/model-v2/model_V2.h5', errno = 2, error message = 'No such file or directory', flags = 0, o_flags = 0)

## === cell 16
def generate_predictions(model, test_dataset, test_df):
    predictions = []

    for batch_idx in range(len(test_dataset)):
        scan_type_1, scan_type_2, scan_type_3, scan_type_4 = test_dataset[batch_idx]
        
        batch_predictions = model.predict([scan_type_1, scan_type_2, scan_type_3, scan_type_4])

        predictions.append(batch_predictions)

    submission = test_df.copy()
    submission["Label"] = [item[0] for sublist in predictions for item in sublist]
    submission.rename(columns={"ID": "BraTS21ID", "Label": "MGMT_value"}, inplace=True)

    return submission

submission = generate_predictions(model, test_dataset, test_df)

## --- ERROR in cell 16, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3216392060.py in <cell line: 0>()
     17     return submission
     18 
---> 19 submission = generate_predictions(model, test_dataset, test_df)

NameError: name 'model' is not defined

## === cell 17
submission.info()

## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/1426793162.py in <cell line: 0>()
----> 1 submission.info()

NameError: name 'submission' is not defined

## === cell 18
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
