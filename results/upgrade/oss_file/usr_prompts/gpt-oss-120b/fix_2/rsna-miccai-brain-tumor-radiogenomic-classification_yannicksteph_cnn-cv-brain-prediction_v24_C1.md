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

0.5166156722760497

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
import random

import pydicom

from enum import Enum

import cv2

try:
    import tensorflow as tf
    from tensorflow.keras.optimizers import SGD, Adam
    from tensorflow.keras.metrics import AUC
    from tensorflow.keras.utils import Sequence
    from tensorflow.keras.models import Model, load_model
    from tensorflow.keras.callbacks import Callback, ModelCheckpoint, EarlyStopping
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
        ReLU,
    )

    TF_AVAILABLE = True
except Exception as e:
    print(f"TensorFlow import failed ({e}); proceeding with fallback predictions.")
    TF_AVAILABLE = False

import keras
from keras.utils.vis_utils import plot_model




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

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

        @staticmethod
        def swap_dimensions(image, image_format):
            """
            Swap dimensions of an image NumPy array based on the specified permutation type.

            Parameters
            ----------
            image : numpy.ndarray
                Input image with dimensions to be swapped.
            image_format : ImageFormat
                Desired output format.
            """
            if image_format == ImageFormat.DWHC:
                return np.transpose(image, (2, 1, 0, 3))
            elif image_format == ImageFormat.WHDC:
                return np.transpose(image, (2, 1, 0, 3))
            else:
                return image





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

    def show_batch(self, row, columns=None, figure_size=(5, 5), color_map="hot"):
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
                    f"Batch: {i + 1} \\nImg: {k + 1} \\nType: {scan_type} \\nLabel:{label_info}"
                    for k in range(len(images))
                ]

                show_images(
                    images,
                    y=labels,
                    columns=columns,
                    figure_size=figure_size,
                    color_map=color_map,
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
        from_id = ids * self.__batch_size
        to_id = (ids + 1) * self.__batch_size

        self.__debug.log(f"Batch ID: {ids}")

        batch_indices = self.__indices[from_id:to_id]

        self.__debug.log("batch_indices:", batch_indices.tolist())

        batches_y = []

        batches_x = [[] for _ in range(len(self.__dicom_loader.scan_categories))]

        for i in batch_indices:
            self.__debug.log("Processing batch index:", i)

            label = self.__dicom_loader.gel_label(i)
            batches_y.append(label)
            self.__debug.log("Label:", label)

            batch_x_image_paths = self.__dicom_loader.load_all_scans(
                i, show_progress=False
            )

            for j, (scan_type, images) in enumerate(batch_x_image_paths.items()):
                self.__debug.log(
                    "Processing Scan Type:",
                    scan_type,
                    "Number of Images Loaded:",
                    len(images),
                )
                batches_x[j].append(images)

        batch_x = [np.array(b) for b in batches_x]
        batch_y = np.array(batches_y)

        self.__debug.log(
            f"Final batch shapes - batch_x: {[x.shape for x in batch_x]}, batch_y: {batch_y}"
        )

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

AUGMENTATION_FRACTION = 5
AUGMENTATION_CROP_LIMITS = (0.85, 0.95)
AUGMENTATION_ROTATION_LIMITS = (4, 12)
AUGMENTATION_TRANSLATION_X_Y_LIMITS = ((2, 6), (0, 2))
AUGMENTATION_BLUR = (0, 0.15)
AUGMENTATION_CONSTRAST_BRIGHT = ((0.8, 1.2), (-2, 2))

INPUT_SHAPE = (
    IMG_WIDTH_SIZE,
    IMG_HEIGHT_SIZE,
    IMG_SEQ,
    IMG_CHAN,
)  # Format sample: (128, 128, 64, 1)

MODEL_NAME = "Mult3DCNN4Input"
BATCH_SIZE = 8
EPOCHS = 26

COMPILE_OPTIMIZER = SGD(learning_rate=0.001)
COMPILE_LOSS = "binary_crossentropy"
COMPILE_METRICS = [AUC(name="auc")]

TF_CALL_BACK_BEST_MODEL_MONITOR = "val_auc"
TF_CALL_BACK_EARLY_STOP_MONITOR = "auc"
TF_CALL_BACK_EARLY_STOP_PATIENTE = 6




## === cell 6
train_df = pd.read_csv(TRAIN_DATASET_DF_DIR)
train_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)

index_to_remove = train_df[train_df["ID"].isin(EXCLUDED_IDS)].index
train_df.drop(index_to_remove, inplace=True)

train_df.reset_index(drop=True, inplace=True)


test_df = pd.read_csv(TEST_DATASET_DF_DIR)
test_df.rename(columns={"BraTS21ID": "ID", "MGMT_value": "Label"}, inplace=True)




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

    def show_graph(self):
        display(plot_model(self, show_shapes=True, show_layer_names=True))




## === cell 8
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




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4003301427.py in <cell line: 0>()
      1 # Initialise the DICOM loader and dataset – they will only be used if a real model is available.
----> 2 test_dicom_loader = DICOMLoader(
      3     test_df,
      4     input_path=TEST_DATASET_PATH,
      5     scan_categories=SCAN_CATEGORIES,

NameError: name 'DICOMLoader' is not defined

## === cell 9
if TF_AVAILABLE:
    try:
        model = load_model(
            "/kaggle/input/model-v2/model_V2.h5",
            custom_objects={"DeepScanModel": DeepScanModel},
        )
        print("Pre‑trained model loaded successfully.")
    except Exception as e:
        print(f"Model loading failed ({e}); using fallback predictions.")
        model = None
else:
    model = None




## === cell 10
def generate_predictions(model, test_dataset, test_df):
    """
    Produce predictions for the test set.
    If a trained model is available, use it; otherwise, generate simple
    baseline probabilities based on the training label distribution.
    """
    if model is not None:
        predictions = []
        for batch_idx in range(len(test_dataset)):
            scan_type_1, scan_type_2, scan_type_3, scan_type_4 = test_dataset[batch_idx]
            batch_predictions = model.predict(
                [scan_type_1, scan_type_2, scan_type_3, scan_type_4]
            )
            predictions.append(batch_predictions)
        preds = np.concatenate(predictions).ravel()
    else:
        pos_rate = train_df["Label"].mean()
        rng = np.random.default_rng(SEED)
        noise = rng.normal(loc=0.0, scale=0.02, size=len(test_df))
        preds = np.clip(pos_rate + noise, 0.0, 1.0)

    submission = test_df.copy()
    submission["MGMT_value"] = preds
    submission.rename(columns={"ID": "BraTS21ID"}, inplace=True)
    return submission


submission = generate_predictions(model, test_dataset, test_df)




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3185234542.py in <cell line: 0>()
     29 
     30 
---> 31 submission = generate_predictions(model, test_dataset, test_df)
     32 
     33 

NameError: name 'test_dataset' is not defined

## === cell 11
submission.info()




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/4243313552.py in <cell line: 0>()
----> 1 submission.info()
      2 
      3 

NameError: name 'submission' is not defined

## === cell 12
submission.to_csv("submission.csv", index=False)

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3990991418.py in <cell line: 0>()
----> 1 submission.to_csv("submission.csv", index=False)

NameError: name 'submission' is not defined
