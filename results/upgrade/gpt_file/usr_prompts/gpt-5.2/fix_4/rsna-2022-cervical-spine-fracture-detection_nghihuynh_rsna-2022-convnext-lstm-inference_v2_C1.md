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

0.6366906402142058

# 6. Current score

1.65813

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 1.66104) has done: 'The timeout is dominated by Stage1 repeatedly reading/decompressing DICOMs and doing per-study DataLoader setup with single-threaded loading. I keep the exact same slice selection, preprocessing math, and model forward passes, but reduce Python and I/O overhead by (a) caching decoded/processed slices per study so the same slice isn’t read multiple times across neighboring triplets, (b) enabling multi-worker DataLoader with persistent workers to parallelize DICOM decoding, and (c) using faster inference settings (`inference_mode`, cuDNN benchmarking for fixed input sizes) while preserving numerical semantics. I also avoid expensive `torch.cat` accumulation by summing predictions incrementally (equivalent mean), and ensure the feature tensor shape remains identical. These changes are provably equivalent in outputs up to negligible FP differences and primarily remove redundant work and improve throughput.'
- What this solution (achieved 1.65813) has done: 'I fix the DataLoader crash by correcting the slice-cache “missing” check so it doesn’t compare numpy arrays to strings, and I harden it so failed DICOM decodes don’t break batching. Then I prevent the Stage2 KeyError by ensuring every UID has a feature array entry even if Stage1 fails/returns empty, keeping the same model/inference semantics. Finally, I add a safe fallback in Stage2 to generate zero features for any missing UID and still produce a complete, correctly ordered `submission.csv` with the required columns. These changes are score-neutral to slightly positive (they avoid accidental missing predictions and excessive 0.01 fallbacks), while preserving the core architecture and inference approach.'

# 9. Code solution

## === cell 0
import os
import sys
import time
import glob
import warnings

import numpy as np
import pandas as pd

import cv2
from tqdm import tqdm

import torch
import torch.nn as nn

from albumentations import Compose, CenterCrop

import pydicom
from pydicom.pixel_data_handlers.util import apply_modality_lut

from torchvision.models.convnext import convnext_base
from torch.utils.data import Dataset, DataLoader

warnings.filterwarnings("ignore")

torch.manual_seed(0)
np.random.seed(0)

DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_PATH = f"{DATA_ROOT}/test_images"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if torch.cuda.is_available():
    torch.backends.cudnn.benchmark = True



## === cell 1
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

LEVELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]




## === cell 2
def load_df_test():
    df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
    expected = {"row_id", "StudyInstanceUID", "prediction_type"}
    missing = expected - set(df_test.columns)
    if missing:
        raise ValueError(f"test.csv missing columns: {missing}")
    return df_test


test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())

print("Num test rows:", len(test_df))
print("Num unique studies:", len(study_id_list))




## === cell 3
def safe_dcmread(path: str):
    try:
        return pydicom.dcmread(path, force=True)
    except Exception:
        return None


def dcm_to_uint8(ds, WL=400, WW=1800):
    """
    Robust decode without relying on external JPEG plugins.
    We try pixel_array; if it fails, return None so caller can fallback.
    """
    try:
        arr = ds.pixel_array  # may fail if compressed decoder missing
    except Exception:
        return None

    try:
        arr = apply_modality_lut(arr, ds)
    except Exception:
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        arr = arr.astype(np.float32) * slope + intercept

    arr = arr.astype(np.float32)
    upper, lower = WL + WW // 2, WL - WW // 2
    arr = np.clip(arr, lower, upper)
    arr = arr - arr.min()
    mx = arr.max()
    if mx > 0:
        arr = arr / mx
    arr = (arr * 255.0).astype(np.uint8)
    return arr


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 4
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


def get_sorted_slice_numbers(uid: str):
    """
    Use actual existing numeric filenames to avoid FileNotFoundError.
    """
    dcm_files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    nums = []
    for p in dcm_files:
        base = os.path.splitext(os.path.basename(p))[0]
        if base.isdigit():
            nums.append(int(base))
    nums = sorted(nums)
    return nums


selected_image_dict = {}
slice_numbers_dict = {}

for uid in study_id_list:
    nums = get_sorted_slice_numbers(uid)
    slice_numbers_dict[uid] = nums
    if len(nums) == 0:
        selected_image_dict[uid] = []
        continue

    middle_idx = len(nums) // 2
    num_left = num_right = max(1, int(0.15 * len(nums)))
    left = list(range(max(0, middle_idx - num_left), middle_idx))
    right = list(range(middle_idx + 1, min(len(nums), middle_idx + num_right + 1)))
    selected_image_dict[uid] = left + right

print("Example UID:", study_id_list[0])
print("Num slices:", len(slice_numbers_dict[study_id_list[0]]))
print("Selected indices (first 10):", selected_image_dict[study_id_list[0]][:10])




## === cell 5
class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.slice_numbers = slice_numbers_dict[uid]

        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

        self._path = os.path.join(TEST_PATH, self.uid)

        self._slice_cache_uint8 = {}  # key: slice_number(int) -> uint8 2D array or None

    def __len__(self):
        return len(self.selected_indices)

    def _get_slice_uint8(self, slice_no: int):
        if slice_no in self._slice_cache_uint8:
            return self._slice_cache_uint8[slice_no]

        ds = safe_dcmread(os.path.join(self._path, f"{slice_no}.dcm"))
        if ds is None:
            self._slice_cache_uint8[slice_no] = None
            return None

        img = dcm_to_uint8(ds)
        self._slice_cache_uint8[slice_no] = img
        return img

    def __getitem__(self, index):
        center_pos = self.selected_indices[index]
        pos_m1 = max(0, center_pos - 1)
        pos_p1 = min(len(self.slice_numbers) - 1, center_pos + 1)

        s_m1 = self.slice_numbers[pos_m1]
        s_0 = self.slice_numbers[center_pos]
        s_p1 = self.slice_numbers[pos_p1]

        imgs = [
            self._get_slice_uint8(s_m1),
            self._get_slice_uint8(s_0),
            self._get_slice_uint8(s_p1),
        ]

        if any(im is None for im in imgs):
            stacked = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
        else:
            stacked = np.stack(imgs, axis=-1)
            stacked = cv2.resize(
                stacked,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )

        out = self.inference_transform(image=stacked)["image"]
        out = (out.astype(np.float32) / 255.0 - mean) / std
        return img2tensor(out, dtype=np.float32)




## === cell 6
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len, feature_size):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len
        self.feature_size = feature_size

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]

        feature_array = self.feature_array_dict.get(uid, None)
        if feature_array is None or len(feature_array) == 0:
            feature_array = np.zeros((1, self.feature_size), dtype=np.float32)

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
        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 7
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        m = convnext_base(weights=None)
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




## === cell 8
lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_ckpt = "../input/cnn-lstm-exp-6/run_1/run_1/model_0.pth"
lv2_ckpt = "../input/cnn-lstm-exp-6/model_lstm_0.pth"

loaded_any = False
if os.path.exists(lv1_ckpt):
    lv1_model.load_state_dict(torch.load(lv1_ckpt, map_location=device))
    loaded_any = True
else:
    print(f"WARNING: missing lv1 checkpoint: {lv1_ckpt}")

if os.path.exists(lv2_ckpt):
    lv2_model.load_state_dict(torch.load(lv2_ckpt, map_location=device))
    loaded_any = True
else:
    print(f"WARNING: missing lv2 checkpoint: {lv2_ckpt}")

if not loaded_any:
    print(
        "WARNING: No checkpoints loaded; will generate a valid but low-quality submission."
    )



## === cell 9
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

lv1_model.eval()
lv2_model.eval()


def _suggest_num_workers():
    if os.name == "nt":
        return 0
    cpu = os.cpu_count() or 2
    return max(2, min(8, cpu // 2))


NUM_WORKERS = _suggest_num_workers()
PERSISTENT = NUM_WORKERS > 0
PREFETCH_FACTOR = 2 if NUM_WORKERS > 0 else None

for uid in tqdm(study_id_list, desc="Stage1 per-study"):
    selected_indices = selected_image_dict[uid]
    if len(selected_indices) == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        mean_preds = np.full((7,), 0.01, dtype=np.float32)
    else:
        dataset = CSFImageDataset(
            uid=uid,
            selected_indices=selected_indices,
            target_size=config["target_size"],
            crop_size=config["crop_size"],
        )
        dl_kwargs = dict(
            dataset=dataset,
            batch_size=config["batch_size_image_level"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            num_workers=NUM_WORKERS,
            persistent_workers=PERSISTENT,
        )
        if PREFETCH_FACTOR is not None:
            dl_kwargs["prefetch_factor"] = PREFETCH_FACTOR
        generator = DataLoader(**dl_kwargs)

        pred_sum = torch.zeros((7,), dtype=torch.float32)
        pred_count = 0

        feature_array = np.zeros(
            (len(dataset), config["feature_size"]), dtype=np.float32
        )

        for i, images in enumerate(generator):
            start = i * config["batch_size_image_level"]
            end = min(start + images.size(0), len(dataset))

            with torch.inference_mode():
                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)
                probs = preds.sigmoid()

            feature_array[start:end] = features.detach().cpu().numpy()
            pred_sum += probs.detach().cpu().sum(dim=0)
            pred_count += probs.size(0)

        feature_array_dict[uid] = feature_array
        mean_preds = (pred_sum / max(1, pred_count)).numpy().astype(np.float32)

    for k, lvl in enumerate(LEVELS):
        submission_dict["row_id"].append(f"{uid}_{lvl}")
        submission_dict["fractured"].append(float(mean_preds[k]))

for uid in study_id_list:
    if uid not in feature_array_dict:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )



## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/3901321990.py in <cell line: 0>()
     51         )
     52 
---> 53         for i, images in enumerate(generator):
     54             start = i * config["batch_size_image_level"]
     55             end = min(start + images.size(0), len(dataset))

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

## === cell 10
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
    feature_size=config["feature_size"],
)
NUM_WORKERS2 = NUM_WORKERS
PERSISTENT2 = NUM_WORKERS2 > 0
PREFETCH_FACTOR2 = 2 if NUM_WORKERS2 > 0 else None

dl2_kwargs = dict(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=NUM_WORKERS2,
    drop_last=False,
    persistent_workers=PERSISTENT2,
)
if PREFETCH_FACTOR2 is not None:
    dl2_kwargs["prefetch_factor"] = PREFETCH_FACTOR2
generator2 = DataLoader(**dl2_kwargs)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
OSError                                   Traceback (most recent call last)
/tmp/ipykernel_55/172420327.py in <cell line: 0>()
     22 generator2 = DataLoader(**dl2_kwargs)
     23 
---> 24 for features, list_uid in tqdm(
     25     generator2, total=len(generator2), desc="Stage2 patient_overall"
     26 ):

/usr/local/lib/python3.11/dist-packages/tqdm/std.py in __iter__(self)
   1179 
   1180         try:
-> 1181             for obj in iterable:
   1182                 yield obj
   1183                 # Update and possibly print the progressbar.

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
   1105 
   1106         # No certainty which module multiprocessing_context is
-> 1107         self._worker_result_queue = multiprocessing_context.Queue()  # type: ignore[var-annotated]
   1108         self._worker_pids_set = False
   1109         self._shutdown = False

/usr/lib/python3.11/multiprocessing/context.py in Queue(self, maxsize)
    101         '''Returns a queue object'''
    102         from .queues import Queue
--> 103         return Queue(maxsize, ctx=self.get_context())
    104 
    105     def JoinableQueue(self, maxsize=0):

/usr/lib/python3.11/multiprocessing/queues.py in __init__(self, maxsize, ctx)
     40             from .synchronize import SEM_VALUE_MAX as maxsize
     41         self._maxsize = maxsize
---> 42         self._reader, self._writer = connection.Pipe(duplex=False)
     43         self._rlock = ctx.Lock()
     44         self._opid = os.getpid()

/usr/lib/python3.11/multiprocessing/connection.py in Pipe(duplex)
    542             c2 = Connection(s2.detach())
    543         else:
--> 544             fd1, fd2 = os.pipe()
    545             c1 = Connection(fd1, writable=False)
    546             c2 = Connection(fd2, readable=False)

OSError: [Errno 24] Too many open files

## === cell 11
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = test_df[["row_id"]].merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].astype(np.float32)
sub_df["fractured"] = sub_df["fractured"].fillna(0.01).clip(1e-6, 1 - 1e-6)

assert len(sub_df) == len(test_df)
assert list(sub_df.columns) == ["row_id", "fractured"]

sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Wrote submission.csv with shape:", sub_df.shape)
