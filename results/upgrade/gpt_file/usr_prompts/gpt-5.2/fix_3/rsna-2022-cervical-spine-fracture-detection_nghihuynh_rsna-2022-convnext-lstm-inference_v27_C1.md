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

# 5. Target score

0.6726247095826395

# 6. Current score

0.5639

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.97796) has done: 'I remove the failing external-weight dependency by switching to a safe fallback that outputs constant probabilities when the referenced model checkpoints aren’t present, ensuring the notebook runs end-to-end and creates `submission.csv`. I also fix the JPEG DICOM decompression crash by forcing installation of compatible `pylibjpeg` + `pylibjpeg-libjpeg` + `gdcm` via pip when needed (the current versions in cell 1 are incompatible with the pydicom error). Finally, I fix multiple indexing/path issues in the DICOM slice selection/loading (sorting filenames, using real file paths rather than `index±1` naming) so feature extraction doesn’t crash on boundary slices and missing numbers, and I build the submission by merging with `test.csv` to guarantee all required `row_id` are present and aligned.'
- What this solution (achieved 0.5639) has done: 'I remove the failing `pip install` dependency for `gdcm/pylibjpeg` (which can’t be installed in this environment) and make DICOM decoding robust by using `pydicom.dcmread(..., force=True)` with a safe fallback when pixel decoding fails. This fixes the current runtime error while keeping the existing model/feature logic intact. To move the score toward your lower-is-better target (current 0.97796 is too high), I also stop defaulting to constant probabilities when checkpoints are missing and instead fit a tiny label-prior model from `train.csv` (global class priors), which is a minimal, valid calibration improvement without changing the core architecture/training loops. The submission writing and row alignment remain based on `test.csv` and always produce a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os, sys, glob, subprocess, textwrap

try:
    import pydicom  # noqa
except Exception as e:
    raise RuntimeError("pydicom is required but failed to import.") from e



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
import torch
import torch.nn as nn
import glob as _glob
from albumentations import Compose, CenterCrop



## === cell 2
torch.backends.cudnn.benchmark = True
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
DATA_ROOT = "../input/rsna-2022-cervical-spine-fracture-detection"


def load_df_test():
    df_test = pd.read_csv(f"{DATA_ROOT}/test.csv")
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
TEST_PATH = f"{DATA_ROOT}/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
study_id_list[:5], len(study_id_list)



## === cell 6
uid_to_dcm_paths = {}
selected_idx_dict = {}

for uid in study_id_list:
    dicom_files = sorted(_glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    uid_to_dcm_paths[uid] = dicom_files
    n = len(dicom_files)
    if n == 0:
        selected_idx_dict[uid] = []
        continue
    middle = n // 2
    k = int(0.15 * n)
    left = max(1, middle - k)
    right = min(n - 2, middle + k)  # keep room for +/-1 neighbor
    idxs = list(range(left, middle)) + list(range(middle + 1, right + 1))
    selected_idx_dict[uid] = idxs



## === cell 7
u0 = study_id_list[0]
print("Example uid:", u0)
print("Total slices:", len(uid_to_dcm_paths[u0]))
print("Selected indices (first 10):", selected_idx_dict[u0][:10])




## === cell 8
def _safe_dcmread(path):
    return pydicom.dcmread(path, force=True)


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    try:
        img = ds.pixel_array.astype(np.float32) * slope + intercept
    except Exception:
        img = np.zeros((512, 512), dtype=np.float32)

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - X.min()
    mx = X.max()
    if mx > 0:
        X = X / mx
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


class CSFImageDataset(torch.utils.data.Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size

        self.dcm_paths = uid_to_dcm_paths[uid]
        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        pos = self.selected_indices[index]
        pos0 = max(0, pos - 1)
        pos1 = pos
        pos2 = min(len(self.dcm_paths) - 1, pos + 1)

        data_list = [
            _safe_dcmread(self.dcm_paths[pos0]),
            _safe_dcmread(self.dcm_paths[pos1]),
            _safe_dcmread(self.dcm_paths[pos2]),
        ]

        imgs = [window(ds) for ds in data_list]
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        out = self.inference_transform(image=stacked_img)
        X = out["image"].astype(np.float32) / 255.0
        X = (X - mean) / std
        X = img2tensor(X)
        return X




## === cell 10
class CSFInstanceDataset(torch.utils.data.Dataset):
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
from torchvision.models.convnext import convnext_base


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




## === cell 12
LV1_CKPT = "../input/cnn-lstm-oct-25-exp-9/run_0/model_0.pth"
LV2_CKPT = "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth"

have_lv1 = os.path.exists(LV1_CKPT)
have_lv2 = os.path.exists(LV2_CKPT)

lv1_model = None
lv2_model = None

if have_lv1:
    lv1_model = ConvNextCNN_B_Feature()
    lv1_model.load_state_dict(torch.load(LV1_CKPT, map_location="cpu"))
    lv1_model = lv1_model.to(DEVICE)
    lv1_model.eval()

if have_lv2:
    lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    lv2_model.load_state_dict(torch.load(LV2_CKPT, map_location="cpu"))
    lv2_model = lv2_model.to(DEVICE)
    lv2_model.eval()

print(
    "Checkpoint availability:",
    {"lv1": have_lv1, "lv2": have_lv2, "device": str(DEVICE)},
)



## === cell 13
train_df = pd.read_csv(f"{DATA_ROOT}/train.csv")
target_cols = ["C1", "C2", "C3", "C4", "C5", "C6", "C7", "patient_overall"]
priors = train_df[target_cols].mean().astype(np.float32).to_dict()

DEFAULT_C_PROB = float(
    np.clip(np.mean([priors[f"C{i}"] for i in range(1, 8)]), 1e-3, 1 - 1e-3)
)
DEFAULT_PATIENT_PROB = float(np.clip(priors["patient_overall"], 1e-3, 1 - 1e-3))
DEFAULT_C_VEC = np.array(
    [float(np.clip(priors[f"C{i}"], 1e-3, 1 - 1e-3)) for i in range(1, 8)],
    dtype=np.float32,
)

print(
    "Using priors fallback:",
    {"C_mean": DEFAULT_C_PROB, "patient": DEFAULT_PATIENT_PROB},
)

c_preds_by_uid = {}
patient_pred_by_uid = {}

if (lv1_model is None) or (lv2_model is None):
    for uid in study_id_list:
        c_preds_by_uid[uid] = DEFAULT_C_VEC.copy()
        patient_pred_by_uid[uid] = float(DEFAULT_PATIENT_PROB)
else:
    feature_array_dict = {}
    for uid in tqdm(study_id_list, desc="Stage1 image-level"):
        idxs = selected_idx_dict[uid]
        if len(idxs) == 0:
            feature_array_dict[uid] = np.zeros(
                (1, config["feature_size"]), dtype=np.float32
            )
            c_preds_by_uid[uid] = DEFAULT_C_VEC.copy()
            continue

        dataset = CSFImageDataset(
            uid=uid,
            selected_indices=idxs,
            target_size=config["target_size"],
            crop_size=config["crop_size"],
        )
        generator = torch.utils.data.DataLoader(
            dataset,
            batch_size=config["batch_size_image_level"],
            shuffle=False,
            pin_memory=torch.cuda.is_available(),
            drop_last=False,
            num_workers=0,
        )

        preds_uid = []
        feature_array = np.zeros(
            (len(dataset), config["feature_size"]), dtype=np.float32
        )

        for i, images in enumerate(generator):
            with torch.no_grad():
                start = i * config["batch_size_image_level"]
                end = min(start + images.shape[0], len(dataset))
                images = images.to(DEVICE, non_blocking=True)
                features, preds = lv1_model(images)
                feature_array[start:end] = features.detach().cpu().numpy()
                preds_uid.append(preds.sigmoid().detach().cpu())

        feature_array_dict[uid] = feature_array
        mean_preds = (
            torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy().astype(np.float32)
        )
        c_preds_by_uid[uid] = mean_preds

    ds2 = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    gen2 = torch.utils.data.DataLoader(
        ds2,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
        drop_last=False,
    )

    for feats, uids in tqdm(gen2, desc="Stage2 patient-level"):
        with torch.no_grad():
            feats = feats.to(DEVICE, non_blocking=True)
            preds = lv2_model(feats).sigmoid().detach().cpu().numpy().reshape(-1)
        for j, uid in enumerate(uids):
            patient_pred_by_uid[uid] = float(preds[j])



## === cell 14
out = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()


def _get_pred(uid, ptype):
    if ptype == "patient_overall":
        return float(patient_pred_by_uid.get(uid, DEFAULT_PATIENT_PROB))
    idx = int(ptype[1:]) - 1
    return float(c_preds_by_uid.get(uid, DEFAULT_C_VEC)[idx])


out["fractured"] = [
    _get_pred(uid, ptype)
    for uid, ptype in zip(out["StudyInstanceUID"].values, out["prediction_type"].values)
]

out["fractured"] = out["fractured"].astype(np.float32).clip(1e-6, 1 - 1e-6)

sub_df = out[["row_id", "fractured"]].copy()
sub_df.to_csv("submission.csv", index=False)
print(sub_df.head())
print("Saved submission.csv with rows:", len(sub_df))
print("fractured summary:", sub_df["fractured"].describe())
