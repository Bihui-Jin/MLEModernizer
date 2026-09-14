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
import os

os.environ.setdefault(
    "PYDICOM_PIXEL_DATA_HANDLER", "pydicom.pixels"
)  # prefer pydicom>=2.4 pixel backend



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
import glob

from albumentations import CenterCrop  # keep same library/transform semantics

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base




## === cell 2
def seed_everything(seed: int = 42):
    import random

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "crop_size": 384,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}



## === cell 4
DATA_ROOT_CANDIDATES = [
    "/kaggle/input/rsna-2022-cervical-spine-fracture-detection",
    "/kaggle/data/rsna-2022-cervical-spine-fracture-detection",
    "../input/rsna-2022-cervical-spine-fracture-detection",
]
DATA_ROOT = next(
    (p for p in DATA_ROOT_CANDIDATES if os.path.exists(p)), DATA_ROOT_CANDIDATES[0]
)


def load_df_test():
    df_test = pd.read_csv(os.path.join(DATA_ROOT, "test.csv"))

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




## === cell 5
test_df = load_df_test()
TEST_PATH = os.path.join(DATA_ROOT, "test_images")
study_id_list = list(test_df.StudyInstanceUID.unique())
study_id_list[:5], len(study_id_list)



## === cell 6
selected_image_dict = {}
dicom_slices_dict = {}
dicom_path_dict = {}

for uid in study_id_list:
    dicom_files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    slice_nums = []
    path_map = {}
    for f in dicom_files:
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            sn = int(base)
        except ValueError:
            continue
        slice_nums.append(sn)
        path_map[sn] = f

    slice_nums = sorted(slice_nums)
    dicom_slices_dict[uid] = slice_nums
    dicom_path_dict[uid] = path_map

    if len(slice_nums) == 0:
        selected_image_dict[uid] = []
        continue

    middle_idx = len(slice_nums) // 2
    num_left = num_right = int(0.15 * len(slice_nums))
    left = max(0, middle_idx - num_left)
    right = min(len(slice_nums), middle_idx + num_right + 1)
    selected_image_dict[uid] = slice_nums[left:right]



## === cell 7
if len(study_id_list) > 0:
    print(
        study_id_list[0],
        selected_image_dict[study_id_list[0]][:10],
        "n_selected=",
        len(selected_image_dict[study_id_list[0]]),
    )



## === cell 8
from collections import OrderedDict

_DCM_CACHE = OrderedDict()
_DCM_CACHE_MAX = 256  # larger -> fewer re-reads; bounded to control memory


def safe_dcmread(path: str):
    ds = _DCM_CACHE.get(path)
    if ds is not None:
        _DCM_CACHE.move_to_end(path)
        return ds
    ds = pydicom.dcmread(path, force=True)
    _DCM_CACHE[path] = ds
    if len(_DCM_CACHE) > _DCM_CACHE_MAX:
        _DCM_CACHE.popitem(last=False)
    return ds


def safe_pixel_array(ds):
    try:
        return ds.pixel_array
    except Exception:
        rows = int(getattr(ds, "Rows", 512))
        cols = int(getattr(ds, "Columns", 512))
        return np.zeros((rows, cols), dtype=np.int16)


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = safe_pixel_array(ds).astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - np.min(X)
    maxv = np.max(X)
    if maxv > 0:
        X = X / maxv
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 9
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)

_CENTER_CROP = CenterCrop(config["crop_size"], config["crop_size"])


_neighbor_triplets = {}
for uid in study_id_list:
    slice_nums = dicom_slices_dict[uid]
    selected = selected_image_dict[uid]
    if len(slice_nums) == 0 or len(selected) == 0:
        _neighbor_triplets[uid] = []
        continue
    pos_map = {sn: i for i, sn in enumerate(slice_nums)}
    trips = []
    last_idx = len(slice_nums) - 1
    for sn in selected:
        pos = pos_map.get(sn, None)
        if pos is None:
            pos = int(np.searchsorted(slice_nums, sn))
            pos = int(np.clip(pos, 0, last_idx))
        prev_ = slice_nums[pos - 1] if pos > 0 else slice_nums[0]
        center = slice_nums[pos]
        next_ = slice_nums[pos + 1] if pos < last_idx else slice_nums[last_idx]
        trips.append((prev_, center, next_))
    _neighbor_triplets[uid] = trips


class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size

        self._crop = _CENTER_CROP

        self._path_map = dicom_path_dict[uid]
        self._triplets = _neighbor_triplets.get(uid, [])

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        if len(self._triplets) == 0:
            stacked_img = np.zeros(
                (self.target_size, self.target_size, 3), dtype=np.uint8
            )
        else:
            prev_, center, next_ = self._triplets[index]

            ds_prev = safe_dcmread(self._path_map[prev_])
            ds_center = safe_dcmread(self._path_map[center])
            ds_next = safe_dcmread(self._path_map[next_])

            imgs = (window(ds_prev), window(ds_center), window(ds_next))
            stacked_img = np.stack(imgs, axis=-1)
            stacked_img = cv2.resize(
                stacked_img,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )

        stacked_img = self._crop(image=stacked_img)["image"]
        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X




## === cell 10
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

        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 11
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=None)  # keep architecture; avoid internet download
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




## === cell 12
def try_load_weights(model, path):
    if path is None:
        return False
    if os.path.exists(path):
        state = torch.load(path, map_location="cpu")
        model.load_state_dict(state)
        return True
    return False


lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_w = os.path.join(DATA_ROOT, "cnn-lstm-oct-25/run_3/run_3/model_3.pth")
lv2_w = os.path.join(DATA_ROOT, "cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth")

loaded1 = try_load_weights(lv1_model, lv1_w)
loaded2 = try_load_weights(lv2_model, lv2_w)
print(f"Loaded lv1 weights: {loaded1} ({lv1_w})")
print(f"Loaded lv2 weights: {loaded2} ({lv2_w})")




## === cell 13
def _dl_num_workers():
    return 0


def make_loader(dataset, batch_size, pin_memory, num_workers):
    try:
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            pin_memory=pin_memory,
            drop_last=False,
            num_workers=num_workers,
            persistent_workers=False,
        )
    except OSError as e:
        if num_workers != 0:
            return DataLoader(
                dataset,
                batch_size=batch_size,
                shuffle=False,
                pin_memory=pin_memory,
                drop_last=False,
                num_workers=0,
                persistent_workers=False,
            )
        raise e


submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

num_workers_stage1 = _dl_num_workers()
pin = torch.cuda.is_available()

for uid in tqdm(study_id_list, desc="Stage1 studies"):
    image_list = selected_image_dict[uid]
    if len(image_list) == 0:
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        mean_preds = np.zeros((7,), dtype=np.float32)
    else:
        dataset = CSFImageDataset(
            uid=uid,
            image_list=image_list,
            target_size=config["target_size"],
            crop_size=config["crop_size"],
        )
        generator = make_loader(
            dataset=dataset,
            batch_size=config["batch_size_image_level"],
            pin_memory=pin,
            num_workers=num_workers_stage1,
        )

        n = len(dataset)
        feature_array = np.empty((n, config["feature_size"]), dtype=np.float32)
        pred_array = np.empty((n, 7), dtype=np.float32)

        offset = 0
        for images in generator:
            bs = images.shape[0]
            with torch.no_grad():
                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)
                preds = preds.sigmoid()

            feature_array[offset : offset + bs] = features.detach().cpu().numpy()
            pred_array[offset : offset + bs] = preds.detach().cpu().numpy()
            offset += bs

        feature_array_dict[uid] = feature_array
        mean_preds = pred_array.mean(axis=0)

    for k, name in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{name}")
        submission_dict["fractured"].append(float(mean_preds[k]))



## === cell 14
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)

num_workers_stage2 = 0
generator2 = make_loader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    pin_memory=pin,
    num_workers=num_workers_stage2,
)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 batches"
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## === cell 15
sub_df = pd.DataFrame.from_dict(submission_dict)

pred_map = dict(zip(sub_df["row_id"].values, sub_df["fractured"].values))
final = test_df[["row_id"]].copy()
final["fractured"] = final["row_id"].map(pred_map)

final["fractured"] = final["fractured"].fillna(0.0).clip(1e-6, 1 - 1e-6)

final.head(), final.shape, final["fractured"].isna().sum()



## === cell 16
final.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", final.shape)
print(final.head(10).to_string(index=False))
