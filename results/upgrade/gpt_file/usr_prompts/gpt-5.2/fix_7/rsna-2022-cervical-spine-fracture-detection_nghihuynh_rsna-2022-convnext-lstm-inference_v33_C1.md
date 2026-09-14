# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

3.11

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

# 5. Code solution

## === cell 0
import os, sys, glob, time
import numpy as np
import pandas as pd

try:
    import pydicom
except Exception as e:
    raise RuntimeError(
        "pydicom is required but not available in this environment."
    ) from e



## === cell 1
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, CenterCrop

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights



## === cell 2
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.benchmark = True

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def seed_worker(worker_id: int):
    base_seed = 0
    np.random.seed(base_seed + worker_id)
    torch.manual_seed(base_seed + worker_id)


_dl_generator = torch.Generator()
_dl_generator.manual_seed(0)



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 368,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}



## === cell 4
DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = os.path.join(DATA_ROOT, "test.csv")
TEST_PATH = os.path.join(DATA_ROOT, "test_images")


def load_df_test():
    df_test = pd.read_csv(TEST_CSV)

    if (
        len(df_test) > 0
        and str(df_test.iloc[0].row_id).endswith("_C1")
        and len(df_test) <= 10
    ):
        pass
    return df_test




## === cell 5
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
print("Num studies:", len(study_id_list))
print("First study:", study_id_list[0] if study_id_list else None)



## === cell 6
uid_to_paths = {}
selected_image_dict = {}

for uid in study_id_list:
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    uid_to_paths[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle = n // 2
    num_each_side = max(1, int(0.15 * n))  # keep original behavior; ensure >= 1
    left_start = max(0, middle - num_each_side)
    left_end = middle  # exclusive
    right_start = min(n - 1, middle + 1)
    right_end = min(n, middle + num_each_side + 1)  # exclusive
    selected_indices = list(range(left_start, left_end)) + list(
        range(right_start, right_end)
    )
    selected_image_dict[uid] = selected_indices

print(
    "Example selected indices:",
    selected_image_dict[study_id_list[0]][:10] if study_id_list else None,
)



## === cell 7
_DICOM_TAGS = [
    "PixelData",
    "RescaleSlope",
    "RescaleIntercept",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "PhotometricInterpretation",
    "SamplesPerPixel",
    "PlanarConfiguration",
    "Rows",
    "Columns",
    "TransferSyntaxUID",
]


def safe_pixel_array(ds):
    """Try to decode pixel data; if JPEG plugins missing/corrupt, return None (skip slice)."""
    try:
        return ds.pixel_array
    except Exception:
        return None


def window_from_dicom(ds, WL=400, WW=1800):
    img = safe_pixel_array(ds)
    if img is None:
        return None

    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = img.astype(np.float32, copy=False) * slope + intercept

    upper = WL + WW // 2
    lower = WL - WW // 2
    X = np.clip(img, lower, upper, out=img)  # reuse buffer
    mn = float(X.min())
    X = X - mn
    mx = float(X.max())
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype(np.uint8, copy=False)
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    arr = img.astype(dtype, copy=False)
    return torch.from_numpy(arr)




## === cell 8
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


def center_crop_np(img_hwc: np.ndarray, crop_size: int) -> np.ndarray:
    h, w = img_hwc.shape[:2]
    ch = crop_size
    cw = crop_size
    y0 = max(0, (h - ch) // 2)
    x0 = max(0, (w - cw) // 2)
    return img_hwc[y0 : y0 + ch, x0 : x0 + cw, :]


class GlobalCSFSliceDataset(Dataset):
    """
    One dataset over all (uid, selected_index_position) rows.
    Each item returns (image_tensor, global_row_index) so caller can scatter results.
    """

    def __init__(self, rows, target_size, crop_size):
        self.rows = rows  # list of tuples: (uid, pos_in_selected_list)
        self.target_size = target_size
        self.crop_size = crop_size
        self._cache = {}

    def __len__(self):
        return len(self.rows)

    def _get_windowed_slice(self, uid: str, j: int):
        c_uid = self._cache.get(uid)
        if c_uid is None:
            c_uid = {}
            self._cache[uid] = c_uid
        w = c_uid.get(j)
        if w is not None:
            return w

        paths = uid_to_paths[uid]
        ds = pydicom.dcmread(paths[j], force=True, specific_tags=_DICOM_TAGS)
        w = window_from_dicom(ds)
        if w is None:
            w = np.zeros((512, 512), dtype=np.uint8)
        c_uid[j] = w
        return w

    def __getitem__(self, index):
        uid, pos = self.rows[index]
        sel = selected_image_dict.get(uid, [])
        idx = sel[pos]
        paths = uid_to_paths[uid]
        n = len(paths)

        idxs = [max(0, idx - 1), idx, min(n - 1, idx + 1)]
        imgs = [self._get_windowed_slice(uid, j) for j in idxs]

        stacked_img = np.stack(imgs, axis=-1)  # H,W,3
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        stacked_img = center_crop_np(stacked_img, self.crop_size)
        X = stacked_img.astype(np.float32, copy=False)
        X = img2tensor((X / 255.0 - mean) / std)
        return X, index


class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict[uid]  # (n_slices, feat)

        if feature_array.shape[0] > self.seq_len:
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

        X = torch.from_numpy(x.astype(np.float32, copy=False))
        return X, uid




## === cell 9
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        m = convnext_base(weights=ConvNeXt_Base_Weights.IMAGENET1K_V1)
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




## === cell 10
lv1_model = ConvNextCNN_B_Feature().to(DEVICE).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(DEVICE)
    .eval()
)

if DEVICE == "cuda":
    lv1_model = lv1_model.to(memory_format=torch.channels_last)


def _try_load_weights(model, candidates):
    for p in candidates:
        if p and os.path.exists(p):
            ckpt = torch.load(p, map_location="cpu")
            state = ckpt.get("state_dict", ckpt)
            new_state = {}
            for k, v in state.items():
                nk = k
                if nk.startswith("module."):
                    nk = nk[len("module.") :]
                if nk.startswith("model."):
                    nk = nk[len("model.") :]
                new_state[nk] = v
            missing, unexpected = model.load_state_dict(new_state, strict=False)
            print(f"Loaded weights from: {p}")
            if missing:
                print("  Missing keys (showing up to 10):", missing[:10])
            if unexpected:
                print("  Unexpected keys (showing up to 10):", unexpected[:10])
            return True
    return False


lv1_candidates = [
    os.path.join(DATA_ROOT, "convnext_base_stage1.pth"),
    os.path.join(DATA_ROOT, "convnext_base.pth"),
    os.path.join(DATA_ROOT, "stage1.pth"),
    os.path.join(DATA_ROOT, "lv1.pth"),
]
lv2_candidates = [
    os.path.join(DATA_ROOT, "gru_stage2.pth"),
    os.path.join(DATA_ROOT, "stage2.pth"),
    os.path.join(DATA_ROOT, "lv2.pth"),
]

found1 = _try_load_weights(lv1_model, lv1_candidates)
found2 = _try_load_weights(lv2_model, lv2_candidates)
print("Stage1 weights found:", found1, "| Stage2 weights found:", found2)

lv1_model.eval()
lv2_model.eval()



## === cell 11
feature_array_dict = {}
uid_level_preds = {}  # uid -> 7 probs
uid_patient_preds = {}  # uid -> patient_overall prob


rows = []
uid_start = {}
uid_nslices = {}
for uid in study_id_list:
    sel = selected_image_dict.get(uid, [])
    uid_start[uid] = len(rows)
    uid_nslices[uid] = len(sel)
    for pos in range(len(sel)):
        rows.append((uid, pos))

for uid in study_id_list:
    n_slices = uid_nslices[uid]
    if n_slices == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        uid_level_preds[uid] = np.full((7,), 0.01, dtype=np.float32)
    else:
        feature_array_dict[uid] = np.zeros(
            (n_slices, config["feature_size"]), dtype=np.float32
        )

sum_preds_uid = {uid: torch.zeros((7,), dtype=torch.float32) for uid in study_id_list}
cnt_preds_uid = {uid: 0 for uid in study_id_list}

if len(rows) > 0:
    dataset = GlobalCSFSliceDataset(
        rows=rows, target_size=config["target_size"], crop_size=config["crop_size"]
    )

    cpu_cnt = os.cpu_count() or 2
    num_workers_img = min(4, max(1, cpu_cnt // 2))
    prefetch_factor = 2 if num_workers_img > 0 else None

    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=num_workers_img,
        persistent_workers=(num_workers_img > 0),
        prefetch_factor=prefetch_factor,
        worker_init_fn=seed_worker if num_workers_img > 0 else None,
        generator=_dl_generator,
    )

    with torch.inference_mode():
        for images, global_idx in tqdm(
            generator, desc="Stage1 (image-level)", total=len(generator)
        ):
            if DEVICE == "cuda":
                images = images.to(DEVICE, non_blocking=True).contiguous(
                    memory_format=torch.channels_last
                )
            else:
                images = images.to(DEVICE)

            features, preds = lv1_model(images)
            probs = preds.sigmoid().detach().cpu()  # (bs,7)
            feats_np = features.detach().cpu().numpy()

            global_idx_np = global_idx.detach().cpu().numpy()
            for b, g in enumerate(global_idx_np):
                uid, pos = rows[int(g)]
                feature_array_dict[uid][pos] = feats_np[b]
                sum_preds_uid[uid] += probs[b]
                cnt_preds_uid[uid] += 1

    for uid in study_id_list:
        if uid_nslices[uid] > 0:
            mean_preds = (
                (sum_preds_uid[uid] / max(1, cnt_preds_uid[uid]))
                .numpy()
                .astype(np.float32)
            )
            uid_level_preds[uid] = mean_preds

else:
    print("No selected slices found in test set; using defaults.")



## === cell 12
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)

cpu_cnt = os.cpu_count() or 2
num_workers_pat = min(2, max(0, cpu_cnt // 4))
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers_pat,
    persistent_workers=(num_workers_pat > 0),
    prefetch_factor=2 if num_workers_pat > 0 else None,
    worker_init_fn=seed_worker if num_workers_pat > 0 else None,
    generator=_dl_generator,
)

with torch.inference_mode():
    for features, list_uid in tqdm(
        generator2, desc="Stage2 (patient-level)", total=len(generator2)
    ):
        features = features.to(DEVICE, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        for j, uid in enumerate(list_uid):
            uid_patient_preds[uid] = float(preds[j])



## === cell 13
eps = 1e-6
prior_level = 0.01
alpha = (
    0.85  # keep predictions dominant; small shrink toward prior for logloss stability
)

uids = np.array(study_id_list, dtype=object)
level_mat = np.stack(
    [uid_level_preds.get(uid, np.full((7,), prior_level, np.float32)) for uid in uids],
    axis=0,
).astype(np.float32)

level_mat = np.clip(level_mat, eps, 1.0 - eps)
level_mat = alpha * level_mat + (1.0 - alpha) * prior_level
level_mat = np.clip(level_mat, eps, 1.0 - eps).astype(np.float32)

patient_from_level = np.max(level_mat, axis=1)
patient_stage2 = np.array(
    [uid_patient_preds.get(uid, np.nan) for uid in uids], dtype=np.float32
)
use_stage2 = ~np.isnan(patient_stage2)
patient_vec = patient_from_level
patient_vec[use_stage2] = 0.5 * patient_from_level[use_stage2] + 0.5 * np.clip(
    patient_stage2[use_stage2], eps, 1.0 - eps
)
patient_vec = np.clip(patient_vec, eps, 1.0 - eps).astype(np.float32)

pred_rows = []
pred_types = ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]
for k, p in enumerate(pred_types):
    pred_rows.append(
        pd.DataFrame(
            {
                "StudyInstanceUID": uids,
                "prediction_type": p,
                "fractured": level_mat[:, k].astype(np.float32),
            }
        )
    )
pred_rows.append(
    pd.DataFrame(
        {
            "StudyInstanceUID": uids,
            "prediction_type": "patient_overall",
            "fractured": patient_vec.astype(np.float32),
        }
    )
)
pred_df = pd.concat(pred_rows, axis=0, ignore_index=True)

sub_df = test_df.merge(pred_df, on=["StudyInstanceUID", "prediction_type"], how="left")
sub_df["fractured"] = sub_df["fractured"].fillna(prior_level).astype(np.float32)
sub_df = sub_df[["row_id", "fractured"]]

assert sub_df.shape[0] == test_df.shape[0]
assert list(sub_df.columns) == ["row_id", "fractured"]



## === cell 14
out_path = "submission.csv"
sub_df.to_csv(out_path, index=False)
print("Wrote", out_path, "with shape", sub_df.shape)
print(sub_df.head(10))
