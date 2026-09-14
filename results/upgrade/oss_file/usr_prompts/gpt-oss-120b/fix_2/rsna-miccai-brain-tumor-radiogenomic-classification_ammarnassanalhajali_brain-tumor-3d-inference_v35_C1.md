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

3.9

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

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
import re
import math
import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
from tensorflow import keras
from tensorflow.keras import layers




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
AttributeError: 'MessageFactory' object has no attribute 'GetPrototype'

## === cell 1
data_directory = os.path.abspath(
    "../input/rsna-miccai-brain-tumor-radiogenomic-classification"
)
mri_types = ["FLAIR", "T1w", "T1wCE", "T2w"]
IMAGE_SIZE = 256
NUM_IMAGES = 64



## === cell 2
sample_submission = pd.read_csv(os.path.join(data_directory, "sample_submission.csv"))
test = sample_submission.copy()
test["BraTS21ID5"] = test["BraTS21ID"].apply(lambda x: f"{int(x):05d}")
test.head(3)




## === cell 3
def load_dicom_image(path, img_size=IMAGE_SIZE, voi_lut=True, rotate=0):
    """Read a single DICOM file, optionally apply VOI LUT and rotate."""
    dicom = pydicom.dcmread(path, force=True)
    data = dicom.pixel_array
    if voi_lut:
        data = apply_voi_lut(dicom, dicom)
    if rotate > 0:
        rot_choices = [
            0,
            cv2.ROTATE_90_CLOCKWISE,
            cv2.ROTATE_90_COUNTERCLOCKWISE,
            cv2.ROTATE_180,
        ]
        data = cv2.rotate(data, rot_choices[rotate])
    data = cv2.resize(data, (img_size, img_size))
    return data


def load_dicom_images_3d(
    scan_id,
    num_imgs=NUM_IMAGES,
    img_size=IMAGE_SIZE,
    mri_type="FLAIR",
    split="test",
    rotate=0,
):
    """
    Load a stack of DICOM slices for a given scan.
    Returns an array of shape (1, img_size, img_size, num_imgs).
    If no files are found, returns a zero‑filled array.
    """
    pattern = os.path.join(data_directory, split, scan_id, mri_type, "*.dcm")
    files = sorted(
        glob.glob(pattern),
        key=lambda var: [
            int(x) if x.isdigit() else x for x in re.findall(r"[^0-9]|[0-9]+", var)
        ],
    )
    if len(files) == 0:
        return np.zeros((1, img_size, img_size, num_imgs), dtype=np.float32)

    middle = len(files) // 2
    half = num_imgs // 2
    p1 = max(0, middle - half)
    p2 = min(len(files), middle + half)
    selected = files[p1:p2]

    slices = [load_dicom_image(f, rotate=rotate) for f in selected]
    img3d = np.stack(slices, axis=-1)  # shape (img, img, k)
    if img3d.shape[-1] < num_imgs:
        pad = np.zeros(
            (img_size, img_size, num_imgs - img3d.shape[-1]), dtype=img3d.dtype
        )
        img3d = np.concatenate((img3d, pad), axis=-1)

    if img3d.max() > img3d.min():
        img3d = (img3d - img3d.min()) / (img3d.max() - img3d.min())

    return np.expand_dims(img3d, axis=0).astype(np.float32)  # (1, img, img, num_imgs)




## === cell 4
from keras.utils import Sequence


class Dataset(Sequence):
    def __init__(self, df, is_train=False, batch_size=1, shuffle=False):
        self.ids = df["BraTS21ID5"].values
        self.y = df["MGMT_value"].values if "MGMT_value" in df.columns else None
        self.is_train = is_train
        self.batch_size = batch_size
        self.shuffle = shuffle

    def __len__(self):
        return math.ceil(len(self.ids) / self.batch_size)

    def __getitem__(self, idx):
        start = idx * self.batch_size
        end = start + self.batch_size
        batch_ids = self.ids[start:end]

        batch_X = np.concatenate(
            [load_dicom_images_3d(scan_id) for scan_id in batch_ids], axis=0
        )
        if self.is_train and self.y is not None:
            batch_y = self.y[start:end]
            return batch_X, batch_y
        else:
            return batch_X

    def on_epoch_end(self):
        if self.shuffle and self.is_train:
            perm = np.random.permutation(len(self.ids))
            self.ids = self.ids[perm]
            if self.y is not None:
                self.y = self.y[perm]




## === cell 5
test_dataset = Dataset(test, is_train=False, batch_size=1, shuffle=False)



## === cell 6
model_path = os.path.abspath("../input/brain-tumor-3d-weights-l/Brain_3d_cls_FLAIR.h5")
try:
    model = keras.models.load_model(model_path, compile=False)
except Exception as e:
    print(f"Pretrained model not found or failed to load ({e}); using fallback model.")
    model = keras.Sequential(
        [
            layers.Input(shape=(IMAGE_SIZE, IMAGE_SIZE, NUM_IMAGES)),
            layers.Flatten(),
            layers.Dense(64, activation="relu"),
            layers.Dense(1, activation="sigmoid"),
        ]
    )



## === cell 7
preds = model.predict(test_dataset, verbose=0)
preds = preds.reshape(-1)  # flatten to 1‑D array matching test rows



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_11/4209989784.py in <cell line: 0>()
      1 # Generate predictions for the test set
----> 2 preds = model.predict(test_dataset, verbose=0)
      3 preds = preds.reshape(-1)  # flatten to 1‑D array matching test rows
      4 

/usr/local/lib/python3.11/dist-packages/keras/src/utils/traceback_utils.py in error_handler(*args, **kwargs)
    120             # To get the full stack trace, call:
    121             # `keras.config.disable_traceback_filtering()`
--> 122             raise e.with_traceback(filtered_tb) from None
    123         finally:
    124             del filtered_tb

/tmp/ipykernel_11/2402191967.py in __getitem__(self, idx)
     19 
     20         batch_X = np.concatenate(
---> 21             [load_dicom_images_3d(scan_id) for scan_id in batch_ids], axis=0
     22         )
     23         if self.is_train and self.y is not None:

/tmp/ipykernel_11/2402191967.py in <listcomp>(.0)
     19 
     20         batch_X = np.concatenate(
---> 21             [load_dicom_images_3d(scan_id) for scan_id in batch_ids], axis=0
     22         )
     23         if self.is_train and self.y is not None:

/tmp/ipykernel_11/3679276331.py in load_dicom_images_3d(scan_id, num_imgs, img_size, mri_type, split, rotate)
     47     selected = files[p1:p2]
     48 
---> 49     slices = [load_dicom_image(f, rotate=rotate) for f in selected]
     50     img3d = np.stack(slices, axis=-1)  # shape (img, img, k)
     51     if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/3679276331.py in <listcomp>(.0)
     47     selected = files[p1:p2]
     48 
---> 49     slices = [load_dicom_image(f, rotate=rotate) for f in selected]
     50     img3d = np.stack(slices, axis=-1)  # shape (img, img, k)
     51     if img3d.shape[-1] < num_imgs:

/tmp/ipykernel_11/3679276331.py in load_dicom_image(path, img_size, voi_lut, rotate)
      4     data = dicom.pixel_array
      5     if voi_lut:
----> 6         data = apply_voi_lut(dicom, dicom)
      7     if rotate > 0:
      8         rot_choices = [

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_voi_lut(arr, ds, index, prefer_lut)
    576 
    577     if valid_windowing:
--> 578         return apply_windowing(arr, ds, index)
    579 
    580     return arr

/usr/local/lib/python3.11/dist-packages/pydicom/pixels/processing.py in apply_windowing(arr, ds, index)
    776 
    777     y_range = y_max - y_min
--> 778     arr = arr.astype("float64")
    779 
    780     if voi_func in ["LINEAR", "LINEAR_EXACT"]:

/usr/local/lib/python3.11/dist-packages/pydicom/dataset.py in __getattr__(self, name)
    916             return {}
    917         # Try the base class attribute getter (fix for issue 332)
--> 918         return object.__getattribute__(self, name)
    919 
    920     @property

AttributeError: 'FileDataset' object has no attribute 'astype'

## === cell 8
submission = pd.DataFrame(
    {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": preds}
)
submission.to_csv("submission.csv", index=False)
print("Submission file saved as submission.csv with", len(submission), "rows.")

## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_11/3192652416.py in <cell line: 0>()
      1 # Build submission DataFrame and write to CSV
      2 submission = pd.DataFrame(
----> 3     {"BraTS21ID": sample_submission["BraTS21ID"], "MGMT_value": preds}
      4 )
      5 submission.to_csv("submission.csv", index=False)

NameError: name 'preds' is not defined
