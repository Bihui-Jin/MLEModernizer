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

0.6065649570364632

# 6. Current score

0.86896

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 0.86896) has done: 'I make the notebook run end-to-end by (1) removing the offline `pip install` cell that breaks in this environment, (2) fixing DICOM JPEG decompression failures by using `pydicom.dcmread(..., force=True)` plus the built-in `pydicom.pixel_data_handlers.util.apply_voi_lut`/fallback decoding so `pixel_array` can be obtained reliably, and (3) fixing missing/incorrect CNN/LSTM weight paths by loading from a local “best available” checkpoint if present, otherwise falling back to a safe constant-probability submission so a valid CSV is always produced. I also fix slice indexing (your code used indices but read `"{idx}.dcm"`; DICOM filenames are not guaranteed contiguous) by sorting actual filenames and selecting neighbors by position, preventing out-of-range reads. Finally, I ensure the submission matches `test.csv` row ordering exactly by merging predictions back onto `test.csv`’s `row_id` list, guaranteeing correct alignment and a valid `submission.csv`.'
- What this solution (achieved 0.86896) has done: 'Your current score (0.86896, lower-is-better) is substantially worse than the target (0.60656), so we should improve predictions while keeping the exact two-stage CNN→GRU logic unchanged. The biggest likely issue is that ConvNeXt is instantiated with `weights=None`, so even if checkpoints exist, the feature backbone may not match what those checkpoints expect (or you’re missing a strong pretrained initialization), leading to weak predictions. I (1) load ImageNet pretrained ConvNeXt weights (using torchvision’s built-in weights, no extra packages), (2) load checkpoints with `strict=False` to avoid silent incompatibilities breaking loads, and (3) compute `patient_overall` in a metric-aligned way as `max(C1..C7, GRU_pred)` so the heavily-weighted overall label is at least consistent with per-level probabilities, typically reducing weighted log loss. These are minimal changes that preserve the architecture and inference flow while moving the score down toward your target.'
- What this solution (achieved 0.86896) has done: 'Your current score (0.86896, lower-is-better) is worse than the target (0.60656), so we should make a small, metric-aligned improvement without changing the 2-stage ConvNeXt→GRU core logic. The biggest safe gain is to correctly use the provided `crop_size` (it’s currently ignored) so the CNN sees the same center-cropped field-of-view it was likely trained on, improving feature consistency and reducing logloss. I also apply torchvision’s official ConvNeXt preprocessing (resize/interpolate + normalize) to better match the ImageNet-pretrained backbone you’re using, while keeping the same inputs (3-slice stack) and same models/checkpoints. Finally, I keep the patient_overall post-processing you already added (max with level preds), and still clamp probabilities for logloss safety, producing a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os

os.makedirs("/root/.cache/torch/hub/checkpoints/", exist_ok=True)



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
import glob
import torch.nn as nn
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

from torch.utils.data import Dataset, DataLoader
import torch

from pydicom.pixel_data_handlers.util import apply_voi_lut



## === cell 2
SEED = 42
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 3
config = {
    "seq_len": 150,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 368,
    "crop_size": 320,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
}




## === cell 4
def load_df_test():
    df_test = pd.read_csv(
        "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )
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
TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]



## === cell 6
uid_to_files = {}
selected_image_dict = {}

for uid in study_id_list:
    dicom_files = sorted(glob.glob(os.path.join(TEST_PATH, uid, "*.dcm")))
    uid_to_files[uid] = dicom_files

    n = len(dicom_files)
    if n == 0:
        selected_image_dict[uid] = []
        continue

    middle_slice = n // 2
    num_left_images = num_right_images = int(0.15 * n)  # 15% left/right
    left = max(1, middle_slice - num_left_images)  # ensure we can take neighbor -1
    right = min(
        n - 2, middle_slice + num_right_images
    )  # ensure we can take neighbor +1
    if right < left:
        safe = list(range(max(1, middle_slice - 1), min(n - 1, middle_slice + 2)))
        selected_image_dict[uid] = safe
    else:
        idxs = list(range(left, middle_slice)) + list(
            range(middle_slice + 1, right + 1)
        )
        selected_image_dict[uid] = idxs



## === cell 7
some_uid = study_id_list[0]
len(uid_to_files[some_uid]), selected_image_dict[some_uid][:10]




## === cell 8
def safe_dcmread(path):
    return pydicom.dcmread(path, force=True)


def window(ds, WL=400, WW=1800):
    """
    Robust windowing that avoids failing on JPEG-compressed pixel data.
    We try to decode pixel_array via pydicom; if it fails, return a neutral image.
    """
    try:
        arr = ds.pixel_array
        try:
            arr = apply_voi_lut(arr, ds)
        except Exception:
            pass

        slope = float(getattr(ds, "RescaleSlope", 1.0))
        intercept = float(getattr(ds, "RescaleIntercept", 0.0))
        img = arr.astype(np.float32) * slope + intercept

        upper, lower = WL + WW // 2, WL - WW // 2
        X = np.clip(img, lower, upper)
        X = X - np.min(X)
        denom = np.max(X)
        if denom > 0:
            X = X / denom
        X = (X * 255.0).astype("uint8")
        return X
    except Exception:
        return np.full((512, 512), 128, dtype=np.uint8)


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))  # HWC -> CHW
    return torch.from_numpy(img.astype(dtype, copy=False))


def center_crop(img_hwc: np.ndarray, crop_size: int) -> np.ndarray:
    """
    Change: actually use config['crop_size'] for a center crop before resizing.
    Why it helps score: this matches the common RSNA baseline preprocessing and
    reduces background/edge variability, usually improving CNN feature quality and logloss.
    """
    h, w = img_hwc.shape[:2]
    cs = int(min(crop_size, h, w))
    y0 = (h - cs) // 2
    x0 = (w - cs) // 2
    return img_hwc[y0 : y0 + cs, x0 : x0 + cs]


IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)




## === cell 9
class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size):
        self.uid = uid
        self.image_list = image_list  # list of integer indices into uid_to_files[uid]
        self.target_size = target_size
        self.crop_size = crop_size

    def __len__(self):
        return len(self.image_list)

    def __getitem__(self, index):
        files = uid_to_files[self.uid]
        pos = int(self.image_list[index])

        pos_m1 = max(0, pos - 1)
        pos_p1 = min(len(files) - 1, pos + 1)

        data_list = [
            safe_dcmread(files[pos_m1]),
            safe_dcmread(files[pos]),
            safe_dcmread(files[pos_p1]),
        ]

        imgs = [window(ds) for ds in data_list]
        stacked_img = np.stack(imgs, axis=-1)  # H,W,3

        stacked_img = center_crop(stacked_img, self.crop_size)

        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        X = stacked_img.astype(np.float32) / 255.0
        X = (X - IMAGENET_MEAN) / IMAGENET_STD
        X = img2tensor(X)
        return X


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
        if feature_array is None:
            feature_array = np.zeros((1, config["feature_size"]), dtype=np.float32)

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

        X = torch.tensor(x, dtype=torch.float32)  # (seq_len, 1024)
        return X, uid




## === cell 10
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




## === cell 11
def find_first_existing(paths):
    for p in paths:
        if p is not None and os.path.exists(p):
            return p
    return None


lv1_ckpt = find_first_existing(
    [
        "../input/cnn-lstm-oct-21/run_5/run_5/model_3.pth",
        "../input/cnn-lstm-oct-21/model_3.pth",
    ]
)
lv2_ckpt = find_first_existing(
    [
        "../input/cnn-lstm-oct-21/run_5b/run_5b/model_lstm_3.pth",
        "../input/cnn-lstm-oct-21/model_lstm_3.pth",
    ]
)

lv1_model = ConvNextCNN_B_Feature().to(device)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)

have_models = (lv1_ckpt is not None) and (lv2_ckpt is not None)
lv1_ckpt, lv2_ckpt, have_models



## === cell 12
if have_models:
    lv1_state = torch.load(lv1_ckpt, map_location=device)
    lv2_state = torch.load(lv2_ckpt, map_location=device)
    lv1_model.load_state_dict(lv1_state, strict=False)
    lv2_model.load_state_dict(lv2_state, strict=False)
    lv1_model.eval()
    lv2_model.eval()



## === cell 13
DEFAULT_C_PROB = 0.05
DEFAULT_ANY_PROB = 0.12  # slightly higher than per-level; conservative for logloss

submission_probs = {}  # key: row_id -> prob
feature_array_dict = {}

if have_models:
    for uid in tqdm(study_id_list, desc="Stage-1 (CNN) per study"):
        image_list = selected_image_dict.get(uid, [])
        if len(image_list) == 0:
            for lvl in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
                submission_probs[f"{uid}_{lvl}"] = DEFAULT_C_PROB
            feature_array_dict[uid] = np.zeros(
                (1, config["feature_size"]), dtype=np.float32
            )
            continue

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

                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)
                feature_array[start:end] = features.detach().cpu().numpy()
                preds_uid.append(torch.sigmoid(preds).detach().cpu())

        feature_array_dict[uid] = feature_array

        mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()
        for k, lvl in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_probs[f"{uid}_{lvl}"] = float(mean_preds[k])



## === cell 14
if have_models:
    dataset2 = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    generator2 = DataLoader(
        dataset=dataset2,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        num_workers=0,
    )

    for features, list_uid in tqdm(
        generator2, total=len(generator2), desc="Stage-2 (GRU) batch"
    ):
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            preds = torch.sigmoid(lv2_model(features)).squeeze(1).detach().cpu().numpy()

        for j, uid in enumerate(list_uid):
            submission_probs[f"{uid}_patient_overall"] = float(preds[j])

    for uid in study_id_list:
        c_probs = []
        for lvl in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            key = f"{uid}_{lvl}"
            if key in submission_probs:
                c_probs.append(submission_probs[key])
        if len(c_probs) > 0:
            any_from_levels = float(np.max(c_probs))
            key_any = f"{uid}_patient_overall"
            if key_any in submission_probs:
                submission_probs[key_any] = float(
                    max(submission_probs[key_any], any_from_levels)
                )
            else:
                submission_probs[key_any] = any_from_levels



## === cell 15
fractured = []
for row_id, ptype, uid in zip(
    test_df["row_id"].values,
    test_df["prediction_type"].values,
    test_df["StudyInstanceUID"].values,
):
    if row_id in submission_probs:
        prob = submission_probs[row_id]
    else:
        prob = DEFAULT_ANY_PROB if ptype == "patient_overall" else DEFAULT_C_PROB
    prob = float(np.clip(prob, 1e-6, 1 - 1e-6))
    fractured.append(prob)

sub_df = pd.DataFrame({"row_id": test_df["row_id"].values, "fractured": fractured})
sub_df.head(), sub_df.shape



## === cell 16
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.tail())
