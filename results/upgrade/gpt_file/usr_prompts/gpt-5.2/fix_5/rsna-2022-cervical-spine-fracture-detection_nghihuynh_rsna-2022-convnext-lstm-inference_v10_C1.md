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
Identify fractures in CT scans of the cervical spine (neck) at both the level of a single vertebrae and the entire patient.

## Metric
Weighted multi-label logarithmic loss. Each fracture sub-type is its own row for every exam, and you are expected to predict a probability for a fracture at each of the seven cervical vertebrae designated as C1, C2, C3, C4, C5, C6 and C7. There is also an any label, `patient_overall`, which indicates that a fracture of ANY kind described before exists in the examination. Fractures in the skull base, thoracic spine, ribs, and clavicles are ignored. The any label is weighted more highly than specific fracture level sub-types.

For each exam Id, you must submit a set of predicted probabilities (a separate row for each cervical level subtype). We then take the log loss for each predicted probability versus its true label.

The binary weighted log loss function for label j on exam i is specified as:

$$
L_{i j}=-w_j *\left[y_{i j} * \log \left(p_{i j}\right)+\left(1-y_{i j}\right) * \log \left(1-p_{i j}\right)\right]
$$

Finally, loss is averaged across all rows.

## Submission Format
There will be 8 rows per image Id. The label indicated by a particular row will look like [image Id]_[Sub-type Name], as follows. There is also a target column, `fractured`, indicating the probability of whether a fracture exists at the specified level. For each image ID in the test set, you must predict a probability for each of the different possible sub-types and the patient overall. The file should contain a header and have the following format:

```
row_id,fractured
1_C1,0
1_C2,0
1_C3,0
1_C4,0.6
1_C5,0
1_C6,0.9
1_C7,0.01
1_patient_overall,0.99
2_C1,0
etc.
```

## Dataset
**train.csv** Metadata for the train test set.

- `StudyInstanceUID` - The study ID. There is one unique study ID for each patient scan.
- `patient_overall` - One of the target columns. The patient level outcome, i.e. if any of the vertebrae are fractured.
- `C[1-7]` - The other target columns. Whether the given vertebrae is fractured. See [this diagram](https://en.wikipedia.org/wiki/Vertebral_column#/media/File:Gray_111_-_Vertebral_column-coloured.png) for the real location of each vertbrae in the spine.

**test.csv** Metadata for the test set prediction structure. Only the first few rows of the test set are available for download.

- `row_id` - The row ID. This will match the same column in the sample submission file.
- `StudyInstanceUID` - The study ID.
- `prediction_type` - Which one of the eight target columns needs a prediction in this row.

**[train/test]_images/[StudyInstanceUID]/[slice_number].dcm** The image data, organized with one folder per scan. Expect to see roughly 1,500 scans in the hidden test set.\

Each image is in [the dicom file format](https://www.dicomstandard.org/). The DICOM image files are ≤ 1 mm slice thickness, axial orientation, and bone kernel. Note that some of the DICOM files are JPEG compressed. You may require additional resources to read the pixel array of these files, such as GDCM and pylibjpeg.

**sample_submission.csv** A valid sample submission.

- `row_id` - The row ID. See the test.csv for what prediction needs to be filed in that row.
- `fractured` - The target column.

**train_bounding_boxes.csv** Bounding boxes for a subset of the training set.

**segmentations/** Pixel level annotations for a subset of the training set. This data is provided in the [nifti file format](https://nifti.nimh.nih.gov/).

A portion of the imaging datasets have been segmented automatically using a 3D UNET model, and radiologists modified and approved the segmentations. The provided segmentation labels have values of 1 to 7 for C1 to C7 (seven cervical vertebrae) and 8 to 19 for T1 to T12 (twelve thoracic vertebrae are located in the center of your upper and middle back), and 0 for everything else. As we focused on the cervical spine, all scans have C1 to C7 labels but not all thoracic labels.

Please be aware that the NIFTI files consist of segmentation in the sagittal plane, while the DICOM files are in the axial plane. Please use the NIFTI header information to determine the appropriate orientation such that the DICOM images and segmentation match. Otherwise, you run the risk of having the segmentations flipped in the Z axis and mirrored in the X axis.

# 2. Python version

3.10

# 3. Installed packages

No external packages required in the script and installed.

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        input/
            description.md (276 lines)
            sample_submission.csv (14537 lines)
            sample_submission.csv.zip (50.3 kB)
            segmentations.zip (3.0 MB)
            test.csv (14537 lines)
            test.csv.zip (94.5 kB)
            test.zip (160 Bytes)
            test_images.zip (181.9 GB)
            train.csv (203 lines)
            train.csv.zip (1.2 kB)
            train.zip (162 Bytes)
            train_bounding_boxes.csv (691 lines)
            train_bounding_boxes.csv.zip (11.5 kB)
            train_images.zip (20.3 GB)
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
            segmentations/
                1.2.826.0.1.3680043.12292.nii (89.4 MB)
                1.2.826.0.1.3680043.24617.nii (307.2 MB)
                ... and 7 other files
            test_images/
                1.2.826.0.1.3680043.10001/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 266 other files
                1.2.826.0.1.3680043.10005/
                    1.dcm (525.1 kB)
                    10.dcm (525.1 kB)
                    ... and 257 other files
                ... and 1816 other folders
            train_images/
                1.2.826.0.1.3680043.10014/
                    1.dcm (240.9 kB)
                    10.dcm (250.8 kB)
                    ... and 256 other files
                1.2.826.0.1.3680043.10058/
                    1.dcm (525.0 kB)
                    10.dcm (525.0 kB)
                    ... and 574 other files
                ... and 201 other folders
        working/
            rsna-2022-cervical-spine-fracture-detection/
                description.md (276 lines)
                sample_submission.csv (14537 lines)
                ... and 12 other files
                rsna-2022-cervical-spine-fracture-detection/
                segmentations/
                    1.2.826.0.1.3680043.12292.nii (89.4 MB)
                    1.2.826.0.1.3680043.24617.nii (307.2 MB)
                    ... and 7 other files
                test_images/
                    1.2.826.0.1.3680043.10001/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 266 other files
                    1.2.826.0.1.3680043.10005/
                        1.dcm (525.1 kB)
                        10.dcm (525.1 kB)
                        ... and 257 other files
                    ... and 1816 other folders
                train_images/
                    1.2.826.0.1.3680043.10014/
                        1.dcm (240.9 kB)
                        10.dcm (250.8 kB)
                        ... and 256 other files
                    1.2.826.0.1.3680043.10058/
                        1.dcm (525.0 kB)
                        10.dcm (525.0 kB)
                        ... and 574 other files
                    ... and 201 other folders
```

-> data/rsna-2022-cervical-spine-fracture-detection/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/rsna-2022-cervical-spine-fracture-detection/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/rsna-2022-cervical-spine-fracture-detection/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/rsna-2022-cervical-spine-fracture-detection/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> data/sample_submission.csv has 14536 rows and 2 columns.
The columns are: row_id, fractured

-> data/test.csv has 14536 rows and 3 columns.
The columns are: StudyInstanceUID, prediction_type, row_id

-> data/train.csv has 202 rows and 9 columns.
The columns are: StudyInstanceUID, patient_overall, C1, C2, C3, C4, C5, C6, C7

-> data/train_bounding_boxes.csv has 690 rows and 6 columns.
The columns are: StudyInstanceUID, x, y, width, height, slice_number

-> (stopped after 10 files for performance)

# 5. Target score

0.6117472871981601

# 6. Current score

0.69377

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 0.69315) has done: 'I (1) remove the failing offline pip-install logic and instead use the Kaggle-provided DICOM JPEG decoders already present in this environment, (2) make DICOM slice selection and loading robust so we never request non-existent “index-1/index+1” files and can decode compressed pixel data reliably, and (3) fix the missing “cell 3” issue and make model weight loading fault-tolerant by searching common Kaggle input locations (while keeping the same architectures and inference flow). These changes unblock end-to-end execution and ensure a correctly formatted `submission.csv` is always written. I also keep predictions clipped away from exactly 0/1 for log-loss stability (score-neutral correctness fix).'
- What this solution (achieved 0.69315) has done: 'I (1) remove the hard failure when external weight files aren’t present by falling back to the default (randomly initialized) models so the notebook always completes and writes `submission.csv`, and (2) fix DICOM JPEG decoding failures by using `torchvision.io.decode_image` (which supports JPEG/JPEG-LS in Kaggle’s base image) to decode encapsulated JPEG frames when `pydicom` can’t. These are minimal, directly targeted fixes to unblock end-to-end inference on all test studies and prevent missing feature arrays/KeyErrors. I keep the model architectures and inference flow unchanged, and keep probability clipping for log-loss stability. This should also improve the score versus the current 0.69315 (which is consistent with many 0.5 fallbacks due to failures) by producing real model outputs for all studies.'
- What this solution (achieved 0.69377) has done: 'The timeout is dominated by per-slice DICOM reads and repeated expensive setup inside `__getitem__` (notably building albumentations `Compose` every call) plus single-worker dataloading, causing GPU/CPU idle and huge overhead across ~1.8k studies. I keep the exact model and inference logic, but make I/O and preprocessing equivalent and faster by (1) precomputing selected triplets once per study, (2) using `dcmread(..., specific_tags=...)` to avoid parsing unnecessary metadata, (3) removing the redundant second read inside `window()` (so each slice is decoded once), (4) moving transforms out of `__getitem__`, and (5) enabling multi-worker loading with persistent workers to overlap DICOM decode with GPU compute. These are provably equivalent with only negligible floating-point differences, and they target constant-factor speedups needed to fit in 600s.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import time
import pickle
import numpy as np
import pandas as pd

import cv2
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut

from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, CenterCrop

from torchvision.models.convnext import convnext_base




## === cell 1
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 2
config = {
    "seq_len": 192,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 3
def _resolve_comp_path():
    candidates = [
        "../input/rsna-2022-cervical-spine-fracture-detection",
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
        "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    ]
    for p in candidates:
        if os.path.exists(p):
            return p
    p = "/kaggle/data/rsna-2022-cervical-spine-fracture-detection"
    return p


COMP_PATH = _resolve_comp_path()


def load_df_test():
    df_test = pd.read_csv(os.path.join(COMP_PATH, "test.csv"))

    if df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
        df_test = pd.DataFrame(
            {
                "row_id": [
                    "1.2.826.0.1.3680043.22327_C1",
                    "1.2.826.0.1.3680043.25399_C1",
                    "1.2.826.0.1.3680043.5876_C1",
                ],
                "StudyInstanceUID": [
                    "1.2.826.0.1.3680043.22327",
                    "1.2.826.0.1.3680043.25399",
                    "1.2.826.0.1.3680043.5876",
                ],
                "prediction_type": ["C1", "C1", "patient_overall"],
            }
        )
    return df_test




## === cell 4
test_df = load_df_test()
TEST_PATH = os.path.join(COMP_PATH, "test_images")
study_id_list = list(test_df.StudyInstanceUID.unique())  # uids

print("Num studies in test_df:", len(study_id_list))
print("TEST_PATH exists:", os.path.exists(TEST_PATH))



## === cell 5
selected_image_dict = {}
uid_to_slice_numbers = {}

for uid in study_id_list:
    dicom_files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    if len(dicom_files) == 0:
        uid_to_slice_numbers[uid] = []
        selected_image_dict[uid] = []
        continue

    slice_numbers = sorted(
        [int(os.path.splitext(os.path.basename(f))[0]) for f in dicom_files]
    )
    uid_to_slice_numbers[uid] = slice_numbers

    mid_idx = len(slice_numbers) // 2
    num_left = num_right = int(0.15 * len(slice_numbers))

    left_idx_start = max(0, mid_idx - num_left)
    left_idx_end = mid_idx  # exclusive
    right_idx_start = min(len(slice_numbers), mid_idx + 1)
    right_idx_end = min(len(slice_numbers), mid_idx + 1 + num_right)

    selected = (
        slice_numbers[left_idx_start:left_idx_end]
        + slice_numbers[right_idx_start:right_idx_end]
    )

    if len(selected) == 0:
        selected = [slice_numbers[mid_idx]]

    selected_image_dict[uid] = selected



## === cell 6
if len(study_id_list) > 0:
    print("Example selected slices:", selected_image_dict[study_id_list[0]][:10])




## === cell 7
def _decode_encapsulated_to_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Robust fallback decoder for encapsulated/compressed single-frame DICOM.
    Tries torchvision first, then Pillow. If both fail, returns a zero image.
    """
    from pydicom.encaps import generate_pixel_data_frame

    frame_bytes = next(generate_pixel_data_frame(ds.PixelData, nr_frames=1))

    try:
        from torchvision.io import decode_image

        t = decode_image(torch.frombuffer(frame_bytes, dtype=torch.uint8))  # [C,H,W]
        t = t.cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):
            if t.shape[0] == 1:
                return t[0].numpy()
            img = (
                0.2989 * t[0].float() + 0.5870 * t[1].float() + 0.1140 * t[2].float()
            ).numpy()
            return np.clip(img, 0, 255).astype(np.uint8)
    except Exception:
        pass

    try:
        from PIL import Image
        import io

        im = Image.open(io.BytesIO(frame_bytes))
        im = im.convert("L")
        return np.array(im)
    except Exception:
        pass

    rows = int(getattr(ds, "Rows", 512))
    cols = int(getattr(ds, "Columns", 512))
    return np.zeros((rows, cols), dtype=np.uint8)


_DCM_TAGS = [
    "RescaleSlope",
    "RescaleIntercept",
    "WindowCenter",
    "WindowWidth",
    "PhotometricInterpretation",
    "BitsStored",
    "BitsAllocated",
    "SamplesPerPixel",
    "PixelRepresentation",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "PixelData",
]


def read_dicom_pixel_array(dcm_path: str):
    try:
        ds = pydicom.dcmread(
            dcm_path,
            stop_before_pixels=False,
            force=True,
            specific_tags=_DCM_TAGS,
        )
    except Exception:
        ds = None
        return ds, np.zeros((512, 512), dtype=np.float32)

    try:
        arr = apply_voi_lut(ds.pixel_array, ds)
        return ds, arr
    except Exception:
        pass

    try:
        arr = ds.pixel_array
        return ds, arr
    except Exception:
        pass

    try:
        arr = _decode_encapsulated_to_array(ds)
        return ds, arr
    except Exception:
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        return ds, np.zeros((rows, cols), dtype=np.float32)




## === cell 8
def window_from_array(ds, img_arr, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = img_arr.astype(np.float32, copy=False)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)

    X = X - np.min(X)
    denom = np.max(X)
    if denom > 0:
        X = X / denom
    X = (X * 255.0).astype("uint8")
    return X


def window(data, WL=400, WW=1800):
    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    try:
        _, img_arr = read_dicom_pixel_array(data.filename)
        img = img_arr.astype(np.float32)
    except Exception:
        try:
            img = data.pixel_array.astype(np.float32)
        except Exception:
            img = np.zeros((512, 512), dtype=np.float32)

    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)

    X = X - np.min(X)
    denom = np.max(X)
    if denom > 0:
        X = X / denom
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))  # [H,W,C] -> [C,H,W]
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 9
def get_triplet_slices(uid: str, center_slice: int):
    slices = uid_to_slice_numbers.get(uid, [])
    if len(slices) == 0:
        return None

    import bisect

    pos = bisect.bisect_left(slices, center_slice)
    if pos == len(slices):
        pos = len(slices) - 1
    if slices[pos] != center_slice:
        if pos > 0 and (
            pos == len(slices)
            or abs(slices[pos - 1] - center_slice) <= abs(slices[pos] - center_slice)
        ):
            pos = pos - 1

    prev_idx = max(0, pos - 1)
    next_idx = min(len(slices) - 1, pos + 1)

    return slices[prev_idx], slices[pos], slices[next_idx]




## === cell 10
mean = np.array([0.456, 0.456, 0.456])
std = np.array([0.224, 0.224, 0.224])


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

        self._triplets = []
        for s in self.image_list:
            t = get_triplet_slices(self.uid, int(s))
            self._triplets.append(t)

        self._uid_path = os.path.join(TEST_PATH, self.uid)

        blank = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
        self._blank_tensor = img2tensor((blank / 255.0 - mean) / std)

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        triplet = self._triplets[index]
        if triplet is None:
            return self._blank_tensor

        s0, s1, s2 = triplet

        d0, a0 = read_dicom_pixel_array(os.path.join(self._uid_path, f"{s0}.dcm"))
        d1, a1 = read_dicom_pixel_array(os.path.join(self._uid_path, f"{s1}.dcm"))
        d2, a2 = read_dicom_pixel_array(os.path.join(self._uid_path, f"{s2}.dcm"))

        if d0 is None or d1 is None or d2 is None:
            return self._blank_tensor

        imgs = [
            window_from_array(d0, a0),
            window_from_array(d1, a1),
            window_from_array(d2, a2),
        ]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        stacked_img = self.inference_transform(image=stacked_img)
        X = stacked_img["image"]
        X = img2tensor((X / 255.0 - mean) / std)

        return X




## === cell 11
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict[uid]
        if len(feature_array) > self.seq_len:
            x = cv2.resize(
                feature_array,
                (feature_array.shape[1], self.seq_len),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            x = np.pad(
                feature_array,
                pad_width=[(0, self.seq_len - feature_array.shape[0]), (0, 0)],
                constant_values=0,
            )
        X = torch.tensor(x, dtype=torch.float32)  # (seq_len,1024)
        return X, uid




## === cell 12
def find_first_existing(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None


def find_weight_file(rel_candidates):
    base_candidates = [
        "../input",
        "/kaggle/input",
        "/kaggle/data",
    ]
    for base in base_candidates:
        for rel in rel_candidates:
            p = os.path.join(base, rel.lstrip("/"))
            if os.path.exists(p):
                return p
    return None




## === cell 13
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base()  # architecture preserved
        in_features = m.classifier[-1].in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=in_features, out_features=7)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.drop(out)
        feature = out.view(x.size(0), -1)
        out = self.fc(feature)
        return feature, out




## === cell 14
class CSFNet(nn.Module):
    def __init__(self, input_len, lstm_size):
        super().__init__()
        self.lstm1 = nn.GRU(input_len, lstm_size, bidirectional=True, batch_first=True)
        self.last_linear = nn.Linear(lstm_size * 2, 1)

    def forward(self, x):
        h_lstm1, _ = self.lstm1(x)
        max_pool, _ = torch.max(h_lstm1, 1)
        logits = self.last_linear(max_pool)
        return logits




## === cell 15
lv1_model = ConvNextCNN_B_Feature()
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])

lv1_w = find_weight_file(
    [
        "cnn-lstm-oct-21/run_5/run_5/model_3.pth",
        "cnn-lstm-oct-21/model_3.pth",
    ]
)
lv2_w = find_weight_file(
    [
        "cnn-lstm-oct-21/run_5/run_5/model_lstm_3.pth",
        "cnn-lstm-oct-21/model_lstm_3.pth",
    ]
)

if lv1_w is not None:
    lv1_model.load_state_dict(torch.load(lv1_w, map_location="cpu"))
if lv2_w is not None:
    lv2_model.load_state_dict(torch.load(lv2_w, map_location="cpu"))

lv1_model = lv1_model.to(device).eval()
lv2_model = lv2_model.to(device).eval()

print("Loaded weights:", lv1_w, lv2_w)
if lv1_w is None or lv2_w is None:
    print(
        "WARNING: One or more weight files missing; using default initialization for missing weights."
    )




## === cell 16
def clip_probs(p, eps=1e-5):
    return float(np.clip(p, eps, 1.0 - eps))




## === cell 17
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

cpu_count = os.cpu_count() or 2
num_workers_img = 4 if torch.cuda.is_available() else min(4, max(1, cpu_count // 2))
prefetch_factor = 2 if num_workers_img > 0 else None

for uid in study_id_list:
    image_list = selected_image_dict[uid]
    dataset = CSFImageDataset(
        uid=uid,
        image_list=image_list,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=num_workers_img,
        persistent_workers=(num_workers_img > 0),
        prefetch_factor=prefetch_factor if num_workers_img > 0 else None,
    )

    preds_uid = []
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in tqdm(
        enumerate(generator), total=len(generator), desc=f"Stage1 {uid}", leave=False
    ):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + config["batch_size_image_level"], len(generator.dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = np.squeeze(features.detach().cpu().numpy())
            preds = preds.sigmoid().detach().cpu()
            preds_uid.append(preds)

    feature_array_dict[uid] = feature_array

    if len(preds_uid) == 0:
        mean_preds = np.full((7,), 0.5, dtype=np.float32)
    else:
        mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

    submission_dict["row_id"].append(f"{uid}_C1")
    submission_dict["fractured"].append(clip_probs(mean_preds[0]))

    submission_dict["row_id"].append(f"{uid}_C2")
    submission_dict["fractured"].append(clip_probs(mean_preds[1]))

    submission_dict["row_id"].append(f"{uid}_C3")
    submission_dict["fractured"].append(clip_probs(mean_preds[2]))

    submission_dict["row_id"].append(f"{uid}_C4")
    submission_dict["fractured"].append(clip_probs(mean_preds[3]))

    submission_dict["row_id"].append(f"{uid}_C5")
    submission_dict["fractured"].append(clip_probs(mean_preds[4]))

    submission_dict["row_id"].append(f"{uid}_C6")
    submission_dict["fractured"].append(clip_probs(mean_preds[5]))

    submission_dict["row_id"].append(f"{uid}_C7")
    submission_dict["fractured"].append(clip_probs(mean_preds[6]))



## --- ERROR in cell 17, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_54/1697370317.py in <cell line: 0>()
     31 
     32     for i, images in tqdm(
---> 33         enumerate(generator), total=len(generator), desc=f"Stage1 {uid}", leave=False
     34     ):
     35         with torch.no_grad():

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __iter__(self)
    484         if self.persistent_workers and self.num_workers > 0:
    485             if self._iterator is None:
--> 486                 self._iterator = self._get_iterator()
    487             else:
    488                 self._iterator._reset(self)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_iterator(self)
    420         else:
    421             self.check_worker_number_rationality()
--> 422             return _MultiProcessingDataLoaderIter(self)
    423 
    424     @property

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __init__(self, loader)
   1144             #     before it starts, and __del__ tries to join but will get:
   1145             #     AssertionError: can only join a started process.
-> 1146             w.start()
   1147             self._index_queues.append(index_queue)
   1148             self._workers.append(w)

/usr/lib/python3.11/multiprocessing/process.py in start(self)
    119                'daemonic processes are not allowed to have children'
    120         _cleanup()
--> 121         self._popen = self._Popen(self)
    122         self._sentinel = self._popen.sentinel
    123         # Avoid a refcycle if the target function holds an indirect

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    222     @staticmethod
    223     def _Popen(process_obj):
--> 224         return _default_context.get_context().Process._Popen(process_obj)
    225 
    226     @staticmethod

/usr/lib/python3.11/multiprocessing/context.py in _Popen(process_obj)
    279         def _Popen(process_obj):
    280             from .popen_fork import Popen
--> 281             return Popen(process_obj)
    282 
    283     class SpawnProcess(process.BaseProcess):

/usr/lib/python3.11/multiprocessing/popen_fork.py in __init__(self, process_obj)
     17         self.returncode = None
     18         self.finalizer = None
---> 19         self._launch(process_obj)
     20 
     21     def duplicate_for_child(self, fd):

/usr/lib/python3.11/multiprocessing/popen_fork.py in _launch(self, process_obj)
     63         code = 1
     64         parent_r, child_w = os.pipe()
---> 65         child_r, parent_w = os.pipe()
     66         self.pid = os.fork()
     67         if self.pid == 0:

OSError: [Errno 24] Too many open files

## === cell 18
missing = [uid for uid in study_id_list if uid not in feature_array_dict]
print("Missing feature arrays:", len(missing))
if len(missing) > 0:
    raise KeyError(f"Missing extracted features for some studies, e.g. {missing[:5]}")



## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/3715722972.py in <cell line: 0>()
      2 print("Missing feature arrays:", len(missing))
      3 if len(missing) > 0:
----> 4     raise KeyError(f"Missing extracted features for some studies, e.g. {missing[:5]}")
      5 

KeyError: "Missing extracted features for some studies, e.g. ['1.2.826.0.1.3680043.21724', '1.2.826.0.1.3680043.13444', '1.2.826.0.1.3680043.24327', '1.2.826.0.1.3680043.23658', '1.2.826.0.1.3680043.22081']"

## === cell 19
dataset = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator = DataLoader(
    dataset=dataset,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=0,
)

for features, list_uid in tqdm(
    generator, total=len(generator), desc="Stage2 patient_overall"
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features)
        preds = np.squeeze(preds.sigmoid().detach().cpu().numpy())

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(clip_probs(preds[j]))



## --- ERROR in cell 19, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_54/511720755.py in <cell line: 0>()
     13 )
     14 
---> 15 for features, list_uid in tqdm(
     16     generator, total=len(generator), desc="Stage2 patient_overall"
     17 ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_54/1363661412.py in __getitem__(self, index)
     10     def __getitem__(self, index):
     11         uid = self.study_id_list[index]
---> 12         feature_array = self.feature_array_dict[uid]
     13         if len(feature_array) > self.seq_len:
     14             x = cv2.resize(

KeyError: '1.2.826.0.1.3680043.21724'

## === cell 20
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")

sub_df = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].fillna(0.5).astype(float).clip(1e-5, 1 - 1e-5)

print(sub_df.head())
print("Submission rows:", len(sub_df), "Expected:", len(test_df))



## === cell 21
assert list(sub_df.columns) == ["row_id", "fractured"]
assert len(sub_df) == len(test_df)
assert sub_df["fractured"].between(0.0, 1.0).all()

sub_df



## === cell 22
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print("Saved to:", os.path.abspath("submission.csv"))
