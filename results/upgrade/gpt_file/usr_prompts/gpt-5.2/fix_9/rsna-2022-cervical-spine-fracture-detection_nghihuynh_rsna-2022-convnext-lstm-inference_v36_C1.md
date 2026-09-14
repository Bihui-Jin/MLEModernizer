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
import os, glob, sys, time, warnings

warnings.filterwarnings("ignore")
print("Python:", sys.version)



## === cell 1
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

try:
    import cv2
except Exception as e:
    cv2 = None
    print("cv2 not available:", e)

try:
    import pydicom
    from pydicom.uid import UID

    try:
        from pydicom.config import settings as _pyd_settings

        _pyd_settings.use_ds_cache = True
    except Exception:
        pass
except Exception as e:
    pydicom = None
    print("pydicom not available:", e)

try:
    from torchvision.models.convnext import convnext_base
except Exception as e:
    convnext_base = None
    print("torchvision convnext not available:", e)



## === cell 2
INPUT_DIR = "../input/rsna-2022-cervical-spine-fracture-detection"
TEST_CSV = os.path.join(INPUT_DIR, "test.csv")
SAMPLE_SUB = os.path.join(INPUT_DIR, "sample_submission.csv")
TEST_PATH = os.path.join(INPUT_DIR, "test_images")

assert os.path.exists(TEST_CSV), f"Missing {TEST_CSV}"
assert os.path.exists(SAMPLE_SUB), f"Missing {SAMPLE_SUB}"
assert os.path.exists(TEST_PATH), f"Missing {TEST_PATH}"



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 368,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 64,
    "batch_size_patient_level": 8,
}

LABELS = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

torch.manual_seed(0)
np.random.seed(0)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

torch.set_num_threads(max(1, (os.cpu_count() or 2)))




## === cell 4
def load_df_test():
    df_test = pd.read_csv(TEST_CSV)
    return df_test




## === cell 5
test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
print("n_test_rows:", len(test_df), "n_studies:", len(study_id_list))



## === cell 6
selected_image_dict = {}
dicom_index_dict = {}

for uid in study_id_list:
    folder = os.path.join(TEST_PATH, uid)
    nums = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                base = name[:-4]
                try:
                    nums.append(int(base))
                except Exception:
                    pass
    except FileNotFoundError:
        nums = []

    nums = sorted(set(nums))
    dicom_index_dict[uid] = nums

    if len(nums) == 0:
        selected_image_dict[uid] = []
        continue

    n_slices = len(nums)
    mid = n_slices // 2
    k = max(1, int(0.15 * n_slices))  # keep original intent

    start = max(1, mid - k)
    end = min(n_slices - 2, mid + k)  # -2 so idx+1 exists
    selected_image_dict[uid] = list(range(start, end + 1))

print("example uid:", study_id_list[0])
print(
    "example selected positions:",
    selected_image_dict[study_id_list[0]][:10],
    "count:",
    len(selected_image_dict[study_id_list[0]]),
)




## === cell 7
def _safe_resize(img, size: int):
    if cv2 is None:
        h, w = img.shape[:2]
        out = np.zeros((size, size, img.shape[2]), dtype=img.dtype)
        hh = min(h, size)
        ww = min(w, size)
        out[:hh, :ww] = img[:hh, :ww]
        return out
    return cv2.resize(img, (size, size))


def window_from_array(img, slope=1.0, intercept=0.0, WL=400, WW=1800):
    img = img.astype(np.float32) * float(slope) + float(intercept)
    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img, lower, upper)
    x = x - np.min(x)
    denom = np.max(x) - 1e-6
    x = x / denom
    x = (x * 255.0).astype("uint8")
    return x


_DICOM_TAGS = [
    "RescaleSlope",
    "RescaleIntercept",
    "PixelData",
    "BitsAllocated",
    "BitsStored",
    "HighBit",
    "PixelRepresentation",
    "SamplesPerPixel",
    "PhotometricInterpretation",
    "Rows",
    "Columns",
    "PlanarConfiguration",
    "NumberOfFrames",
    "TransferSyntaxUID",
]


def read_dicom_pixels(path):
    if pydicom is None:
        return None, 1.0, 0.0
    try:
        ds = pydicom.dcmread(
            path,
            force=True,
            stop_before_pixels=False,
            specific_tags=_DICOM_TAGS,
        )
        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        return ds.pixel_array, slope, intercept
    except Exception:
        return None, 1.0, 0.0


def _read_windowed_slice_uint8_uncached(dcm_path: str):
    arr, slope, intercept = read_dicom_pixels(dcm_path)
    if arr is None:
        return np.zeros((512, 512), dtype=np.uint8)
    return window_from_array(arr, slope=slope, intercept=intercept)


def img2tensor(img, dtype=np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 8
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_pos_list, target_size, crop_size):
        self.uid = uid
        self.image_pos_list = image_pos_list  # positions into dicom_index_dict[uid]
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_pos_list)

    def __getitem__(self, index):
        nums = dicom_index_dict.get(self.uid, [])
        if len(nums) < 3:
            stacked = np.zeros((self.target_size, self.target_size, 3), dtype=np.uint8)
            return img2tensor((stacked / 255.0 - mean) / std)

        pos = self.image_pos_list[index]
        pos = int(np.clip(pos, 1, len(nums) - 2))
        slice_nums = [nums[pos - 1], nums[pos], nums[pos + 1]]

        imgs = []
        for sn in slice_nums:
            dcm_path = os.path.join(TEST_PATH, self.uid, f"{sn}.dcm")
            img = _read_windowed_slice_uint8_uncached(dcm_path)
            imgs.append(img)

        stacked_img = np.stack(imgs, axis=-1)  # H,W,3
        stacked_img = _safe_resize(stacked_img, self.target_size)
        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X




## === cell 9
class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict.get(uid, None)
        if feature_array is None or len(feature_array) == 0:
            x = np.zeros((self.seq_len, config["feature_size"]), dtype=np.float32)
        else:
            fa = feature_array
            if fa.shape[0] > self.seq_len:
                if cv2 is not None:
                    x = cv2.resize(
                        fa, (fa.shape[1], self.seq_len), interpolation=cv2.INTER_LINEAR
                    )
                else:
                    idx = np.linspace(0, fa.shape[0] - 1, self.seq_len).astype(int)
                    x = fa[idx]
            else:
                x = np.pad(
                    fa,
                    pad_width=[(0, self.seq_len - fa.shape[0]), (0, 0)],
                    constant_values=0,
                )
        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 10
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super().__init__()
        if convnext_base is None:
            self.backbone = nn.Sequential(
                nn.Conv2d(3, 32, 3, stride=2, padding=1),
                nn.ReLU(inplace=True),
                nn.AdaptiveAvgPool2d(1),
            )
            self.in_features = 32
        else:
            m = convnext_base(weights=None)  # keep architecture; no external weights
            self.features = m.features
            self.avgpool = nn.AdaptiveAvgPool2d(1)
            self.in_features = m.classifier[-1].in_features

        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=self.in_features, out_features=7)

    def forward(self, x):
        if convnext_base is None:
            out = self.backbone(x)
        else:
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




## === cell 11
lv1_model = ConvNextCNN_B_Feature().to(device)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)

lv1_ckpt = "../input/cnn-lstm-oct-21/run_5/run_5/model_3.pth"
lv2_ckpt = "../input/cnn-lstm-oct-27/run_3/run_3/model_lstm_best.pth"


def try_load(model, path):
    if os.path.exists(path):
        sd = torch.load(path, map_location="cpu")
        model.load_state_dict(sd, strict=True)
        print("Loaded:", path)
        return True
    print("Checkpoint not found, using default init:", path)
    return False


_ = try_load(lv1_model, lv1_ckpt)
_ = try_load(lv2_model, lv2_ckpt)

lv1_model.eval()
lv2_model.eval()



## === cell 12
uid_to_i = {str(uid): i for i, uid in enumerate(study_id_list)}
nstudies = len(study_id_list)

uid_pos_pairs = []
nsel = {}
for uid in study_id_list:
    s = selected_image_dict.get(uid, [])
    nsel[uid] = len(s)
    for pos in s:
        uid_pos_pairs.append((uid, pos))

uid_to_slice_paths = {}
for uid in study_id_list:
    nums = dicom_index_dict.get(uid, [])
    if nums:
        base = os.path.join(TEST_PATH, uid)
        uid_to_slice_paths[uid] = [os.path.join(base, f"{sn}.dcm") for sn in nums]
    else:
        uid_to_slice_paths[uid] = []

uid_offsets = np.zeros((nstudies + 1,), dtype=np.int64)
for i, uid in enumerate(study_id_list):
    uid_offsets[i + 1] = uid_offsets[i] + int(nsel.get(uid, 0))
total_sel = int(uid_offsets[-1])

flat_probs = np.zeros((total_sel, 7), dtype=np.float32)
flat_feats = np.zeros((total_sel, config["feature_size"]), dtype=np.float32)

feature_array_dict = {}
for uid in study_id_list:
    c = int(nsel.get(uid, 0))
    if c <= 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
    else:
        feature_array_dict[uid] = np.zeros(
            (c, config["feature_size"]), dtype=np.float32
        )



## === cell 13
_inv255 = np.float32(1.0 / 255.0)
_mean = mean.reshape(1, 1, 3)
_std = std.reshape(1, 1, 3)

_sigmoid = torch.sigmoid

lv1 = lv1_model
fdim_cfg = int(config["feature_size"])
bs_img = int(config["batch_size_image_level"])


def _get_windowed_cached(path: str, cache: dict):
    v = cache.get(path)
    if v is not None:
        return v
    v = _read_windowed_slice_uint8_uncached(path)
    cache[path] = v
    return v


t0 = time.time()
with torch.no_grad():
    for si, uid in enumerate(study_id_list):
        pos_list = selected_image_dict.get(uid, [])
        if not pos_list:
            continue

        paths = uid_to_slice_paths.get(uid, [])
        if len(paths) < 3:
            continue

        cache = {}

        start = int(uid_offsets[si])
        batch_imgs = []
        batch_flat_idx = []

        for j, pos in enumerate(pos_list):
            pos = int(np.clip(int(pos), 1, len(paths) - 2))
            img0 = _get_windowed_cached(paths[pos - 1], cache)
            img1 = _get_windowed_cached(paths[pos], cache)
            img2 = _get_windowed_cached(paths[pos + 1], cache)

            stacked = np.stack((img0, img1, img2), axis=-1)  # H,W,3
            stacked = _safe_resize(stacked, config["target_size"])

            x = stacked.astype(np.float32, copy=False)
            x *= _inv255
            x -= _mean
            x /= _std
            X = img2tensor(x)  # C,H,W float32
            batch_imgs.append(X)
            batch_flat_idx.append(start + j)

            if len(batch_imgs) == bs_img:
                images = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
                features, logits = lv1(images)

                probs = (
                    _sigmoid(logits)
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32, copy=False)
                )
                feats = features.detach().cpu().numpy().astype(np.float32, copy=False)

                flat_idx_np = np.asarray(batch_flat_idx, dtype=np.int64)
                flat_probs[flat_idx_np] = probs
                fdim = feats.shape[1]
                if fdim >= fdim_cfg:
                    flat_feats[flat_idx_np, :fdim_cfg] = feats[:, :fdim_cfg]
                else:
                    flat_feats[flat_idx_np, :fdim] = feats

                batch_imgs.clear()
                batch_flat_idx.clear()

        if batch_imgs:
            images = torch.stack(batch_imgs, dim=0).to(device, non_blocking=True)
            features, logits = lv1(images)

            probs = (
                _sigmoid(logits).detach().cpu().numpy().astype(np.float32, copy=False)
            )
            feats = features.detach().cpu().numpy().astype(np.float32, copy=False)

            flat_idx_np = np.asarray(batch_flat_idx, dtype=np.int64)
            flat_probs[flat_idx_np] = probs
            fdim = feats.shape[1]
            if fdim >= fdim_cfg:
                flat_feats[flat_idx_np, :fdim_cfg] = feats[:, :fdim_cfg]
            else:
                flat_feats[flat_idx_np, :fdim] = feats

        del cache

t1 = time.time()

preds_level_dict = {}
for i, uid in enumerate(study_id_list):
    start = int(uid_offsets[i])
    end = int(uid_offsets[i + 1])
    c = end - start
    if c <= 0:
        preds_level_dict[uid] = np.full((7,), 0.5, dtype=np.float32)
    else:
        preds_level_dict[uid] = (
            flat_probs[start:end].mean(axis=0).astype(np.float32, copy=False)
        )
        feature_array_dict[uid][:c] = flat_feats[start:end]

print(
    f"Stage1 done in {t1-t0:.1f}s. Example preds:",
    study_id_list[0],
    preds_level_dict[study_id_list[0]],
)



## === cell 14
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)

_num_workers2 = min(2, max(0, (os.cpu_count() or 2) // 4))
generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=_num_workers2,
    persistent_workers=(_num_workers2 > 0),
)

preds_overall_dict = {}
with torch.no_grad():
    for features, list_uid in generator2:
        features = features.to(device, non_blocking=True)
        logits = lv2_model(features)
        probs = torch.sigmoid(logits).detach().cpu().numpy().reshape(-1)
        for u, p in zip(list_uid, probs):
            preds_overall_dict[str(u)] = float(p)

print(
    "Stage2 done. Example overall:",
    study_id_list[0],
    preds_overall_dict[study_id_list[0]],
)



## === cell 15
eps = 1e-4

label_to_idx = {"C1": 0, "C2": 1, "C3": 2, "C4": 3, "C5": 4, "C6": 5, "C7": 6}

level_mat = np.vstack(
    [preds_level_dict.get(uid, np.full((7,), 0.5, np.float32)) for uid in study_id_list]
).astype(np.float32)
uid_to_row = {str(uid): i for i, uid in enumerate(study_id_list)}

sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()
uids = sub["StudyInstanceUID"].astype(str).to_numpy()
ptypes = sub["prediction_type"].astype(str).to_numpy()

out = np.empty((len(sub),), dtype=np.float32)
overall_mask = ptypes == "patient_overall"

if overall_mask.any():
    out_overall = np.fromiter(
        (preds_overall_dict.get(u, 0.5) for u in uids[overall_mask]),
        dtype=np.float32,
        count=overall_mask.sum(),
    )
    out[overall_mask] = out_overall

if (~overall_mask).any():
    idxs = np.fromiter(
        (label_to_idx.get(t, 0) for t in ptypes[~overall_mask]),
        dtype=np.int64,
        count=(~overall_mask).sum(),
    )
    rows = np.fromiter(
        (uid_to_row.get(u, -1) for u in uids[~overall_mask]),
        dtype=np.int64,
        count=(~overall_mask).sum(),
    )
    vals = np.full((len(rows),), 0.5, dtype=np.float32)
    ok = rows >= 0
    vals[ok] = level_mat[rows[ok], idxs[ok]]
    out[~overall_mask] = vals

out = np.clip(out, eps, 1.0 - eps).astype(np.float32)
sub = pd.DataFrame({"row_id": sub["row_id"].values, "fractured": out})

sample = pd.read_csv(SAMPLE_SUB)
assert list(sample.columns) == ["row_id", "fractured"]
assert len(sub) == len(sample)
assert set(sub["row_id"]) == set(sample["row_id"])

sub = sample[["row_id"]].merge(sub, on="row_id", how="left")
sub["fractured"] = sub["fractured"].fillna(0.5).astype(np.float32)

sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head())
