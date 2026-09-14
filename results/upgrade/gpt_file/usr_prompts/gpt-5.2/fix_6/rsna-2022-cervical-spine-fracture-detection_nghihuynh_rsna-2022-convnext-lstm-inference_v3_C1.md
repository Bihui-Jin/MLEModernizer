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

0.6121877808283623

# 6. Current score

0.89301

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.15429) has done: 'I fix the two blockers preventing an end-to-end run: missing pretrained weight files (by loading them robustly from any available `/kaggle/input` location and falling back to zero/constant predictions if truly unavailable) and DICOM JPEG decompression failures (by using pydicom’s safest available pixel decoding plus a clean fallback when decompression plugins aren’t present). I also fix slice indexing bugs (DICOM filenames are not guaranteed to match `index±1`) by selecting slices by sorted filenames and safely clamping neighbors. Finally, I ensure the generated submission matches `test.csv` row ordering exactly (merge on `row_id`) and always writes `submission.csv` with the required columns.'
- What this solution (achieved 1.15429) has done: 'I make two minimal, score-relevant fixes that keep your model logic intact but remove avoidable inference-time randomness and improve the patient_overall calibration. First, I disable dropout during inference by making the ConvNeXt head’s dropout conditional on `self.training` (your current `Dropout(p=0.5)` is applied even in `eval()` because it’s not registered as an `nn.Dropout` module). Second, I compute `patient_overall` directly from the per-level probabilities using a noisy-OR (probability of any fracture) and blend it slightly with the LSTM output to stay close to your existing semantics while typically reducing weighted logloss on the heavily-weighted any label. These changes are small, deterministic, and should move the loss down toward your target without changing training/architecture.'
- What this solution (achieved 0.88934) has done: 'I make two minimal, score-relevant changes that preserve your model and inference flow but reduce weighted logloss by improving probability calibration and correctness for the heavily-weighted `patient_overall` label. First, I generate `patient_overall` for *all* studies (even when weights are missing) from the per-level probabilities using a noisy-OR, instead of a fixed constant, which should reduce loss noticeably versus `DEFAULT_ANY_P`. Second, I clamp/blend probabilities slightly more safely: compute a clipped noisy-OR, then blend with the LSTM output only when available, and clip the final `patient_overall` as well to avoid extreme logloss penalties. These are small post-processing changes that keep the core models unchanged and should move your loss down toward the 0.61 target.'
- What this solution (achieved 1.03952) has done: 'I make small post-processing changes to better match the competition’s weighted logloss without altering your model architectures or inference loops. Because your current score (0.88934, lower is better) is still far from the target (0.61219), we should improve calibration rather than change modeling: (1) compute `patient_overall` purely from the per-level probabilities via a clipped noisy-OR (this label is heavily weighted, and blending with the LSTM can hurt calibration), and (2) apply a very light “sharpening” (power transform) to per-level probabilities before noisy-OR to reduce overconfident midrange probabilities that typically inflate logloss. These changes keep all core logic intact (same models, same slice selection, same feature extraction) and only adjust probability aggregation/clipping to move the score down toward the target. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.89301) has done: 'I make two minimal, score-relevant calibration fixes that don’t change your model architectures or inference loops. First, I remove the per-level “sharpening” power transform (it can easily worsen logloss when probabilities are already miscalibrated, especially on a heavily-weighted label derived from them) and instead use a plain clipped noisy-OR for `patient_overall`. Second, I slightly regularize probabilities by blending each level prediction a small amount toward a low prior (label-smoothing at inference) before computing noisy-OR, which typically reduces overconfident errors and should lower weighted logloss toward your target. The submission writing/order logic stays identical and still produces a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import sys
import glob
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")



## === cell 1
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import pydicom

try:
    from albumentations import Compose, CenterCrop

    _HAS_ALB = True
except Exception:
    _HAS_ALB = False

try:
    import cv2
except Exception as e:
    raise RuntimeError("cv2 is required by this solution but is not available.") from e

from tqdm import tqdm

from torchvision.models.convnext import convnext_base



## === cell 2
BASE_INPUT = Path("../input/rsna-2022-cervical-spine-fracture-detection")
if not BASE_INPUT.exists():
    BASE_INPUT = Path("/kaggle/data/rsna-2022-cervical-spine-fracture-detection")
if not BASE_INPUT.exists():
    BASE_INPUT = Path("/kaggle/input/rsna-2022-cervical-spine-fracture-detection")

TEST_PATH = str(BASE_INPUT / "test_images")
TRAIN_CSV = str(BASE_INPUT / "train.csv")
TEST_CSV = str(BASE_INPUT / "test.csv")
SAMPLE_SUB = str(BASE_INPUT / "sample_submission.csv")

assert Path(TEST_CSV).exists(), f"Missing test.csv at {TEST_CSV}"
assert Path(TEST_PATH).exists(), f"Missing test_images at {TEST_PATH}"
assert Path(SAMPLE_SUB).exists(), f"Missing sample_submission.csv at {SAMPLE_SUB}"



## === cell 3
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




## === cell 4
def load_df_test():
    df_test = pd.read_csv(TEST_CSV)

    if (
        len(df_test) > 0
        and isinstance(df_test.iloc[0].row_id, str)
        and df_test.iloc[0].row_id.endswith("_C1")
    ):
        pass
    return df_test


test_df = load_df_test()
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]




## === cell 5
def _safe_int_stem(p: str):
    try:
        return int(Path(p).stem)
    except Exception:
        return None


dicom_files_by_uid = {}
selected_idx_by_uid = {}

for uid in study_id_list:
    files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    if len(files) == 0:
        dicom_files_by_uid[uid] = []
        selected_idx_by_uid[uid] = []
        continue

    stems = [_safe_int_stem(f) for f in files]
    if all(s is not None for s in stems):
        files = [f for _, f in sorted(zip(stems, files))]
    else:
        files = sorted(files)

    dicom_files_by_uid[uid] = files

    mid = len(files) // 2
    k = int(0.15 * len(files))
    left = max(0, mid - k)
    right = min(len(files) - 1, mid + k)
    idxs = list(range(left, mid)) + list(range(mid + 1, right + 1))
    selected_idx_by_uid[uid] = idxs

if len(study_id_list) > 0 and len(dicom_files_by_uid[study_id_list[0]]) > 0:
    print(
        "Example selected indices:",
        selected_idx_by_uid[study_id_list[0]][:10],
        " / n_files=",
        len(dicom_files_by_uid[study_id_list[0]]),
    )




## === cell 6
def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    try:
        img = ds.pixel_array.astype(np.float32)
    except Exception:
        img = np.zeros((512, 512), dtype=np.float32)

    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - np.min(X)
    mx = np.max(X)
    if mx > 0:
        X = X / mx
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 7
class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.files = dicom_files_by_uid[uid]

    def __len__(self):
        return len(self.selected_indices)

    def __getitem__(self, index):
        idx = self.selected_indices[index]
        n = len(self.files)
        if n == 0:
            stacked_img = np.zeros((self.crop_size, self.crop_size, 3), dtype=np.uint8)
            X = img2tensor((stacked_img / 255.0 - mean) / std)
            return X

        i0 = max(0, idx - 1)
        i1 = idx
        i2 = min(n - 1, idx + 1)

        ds0 = pydicom.dcmread(self.files[i0], force=True)
        ds1 = pydicom.dcmread(self.files[i1], force=True)
        ds2 = pydicom.dcmread(self.files[i2], force=True)

        imgs = [window(ds0), window(ds1), window(ds2)]
        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        if _HAS_ALB:
            inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])
            stacked_img = inference_transform(image=stacked_img)["image"]
        else:
            h, w = stacked_img.shape[:2]
            ch = cw = self.crop_size
            y0 = max(0, (h - ch) // 2)
            x0 = max(0, (w - cw) // 2)
            stacked_img = stacked_img[y0 : y0 + ch, x0 : x0 + cw]

        X = img2tensor((stacked_img.astype(np.float32) / 255.0 - mean) / std)
        return X




## === cell 8
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
        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 9
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




## === cell 10
def find_weight_file(preferred_rel_path: str, filename_fallback: str):
    preferred = Path(preferred_rel_path)
    if preferred.exists():
        return str(preferred)

    candidates = glob.glob(f"/kaggle/input/**/{filename_fallback}", recursive=True)
    if len(candidates) > 0:
        return candidates[0]
    candidates = glob.glob(f"../input/**/{filename_fallback}", recursive=True)
    if len(candidates) > 0:
        return candidates[0]
    return None


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lv1_weight = find_weight_file(
    "../input/cnn-lstm-exp-6/run_2/run_2/model_3.pth", "model_3.pth"
)
lv2_weight = find_weight_file(
    "../input/cnn-lstm-exp-6/run_2/run_2/model_lstm_3.pth", "model_lstm_3.pth"
)

lv1_model = ConvNextCNN_B_Feature().to(device)
lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)

HAS_WEIGHTS = True
if lv1_weight is None or lv2_weight is None:
    HAS_WEIGHTS = False
    print(
        "WARNING: Could not find required .pth weights. Will run with untrained models and safe defaults."
    )
else:
    lv1_model.load_state_dict(torch.load(lv1_weight, map_location="cpu"))
    lv2_model.load_state_dict(torch.load(lv2_weight, map_location="cpu"))

lv1_model.eval()
lv2_model.eval()

print("device:", device, "| HAS_WEIGHTS:", HAS_WEIGHTS)
if HAS_WEIGHTS:
    print("lv1 weights:", lv1_weight)
    print("lv2 weights:", lv2_weight)



## === cell 11
submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}
mean_level_preds_by_uid = {}

DEFAULT_LEVEL_P = 0.02
DEFAULT_ANY_P = 0.05

for uid in tqdm(study_id_list, total=len(study_id_list)):
    sel_idx = selected_idx_by_uid[uid]
    dataset = CSFImageDataset(
        uid=uid,
        selected_indices=sel_idx,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        drop_last=False,
        num_workers=0,
    )

    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    if len(dataset) == 0 or not HAS_WEIGHTS:
        feature_array_dict[uid] = feature_array
        mean_preds = np.array([DEFAULT_LEVEL_P] * 7, dtype=np.float32)
    else:
        preds_uid = []
        for i, images in enumerate(generator):
            with torch.no_grad():
                start = i * config["batch_size_image_level"]
                end = min(start + images.shape[0], len(generator.dataset))

                images = images.to(device, non_blocking=True)
                features, preds = lv1_model(images)

                feat_np = features.detach().cpu().numpy()
                feature_array[start:end] = feat_np.reshape(end - start, -1)

                preds = preds.sigmoid().detach().cpu()
                preds_uid.append(preds)

        feature_array_dict[uid] = feature_array
        mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

    mean_level_preds_by_uid[uid] = mean_preds.astype(np.float32, copy=False)

    for k, lvl in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{lvl}")
        submission_dict["fractured"].append(float(mean_preds[k]))



## === cell 12
EPS = 1e-6

PRIOR_LEVEL_P = 0.015
LEVEL_BLEND = 0.06  # small on purpose to avoid changing semantics too much


def _noisy_or_from_levels(level_probs: np.ndarray) -> float:
    p = level_probs.astype(np.float32, copy=False)
    p = np.clip(p, EPS, 1.0 - EPS)

    p = (1.0 - LEVEL_BLEND) * p + LEVEL_BLEND * PRIOR_LEVEL_P
    p = np.clip(p, EPS, 1.0 - EPS)

    return float(np.clip(1.0 - np.prod(1.0 - p), EPS, 1.0 - EPS))


if HAS_WEIGHTS:
    dataset2 = CSFInstanceDataset(
        feature_array_dict=feature_array_dict,
        study_id_list=study_id_list,
        seq_len=config["seq_len"],
    )
    generator2 = DataLoader(
        dataset=dataset2,
        batch_size=config["batch_size_patient_level"],
        shuffle=False,
        pin_memory=(device.type == "cuda"),
        num_workers=0,
    )

    for features, list_uid in tqdm(generator2, total=len(generator2)):
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            _ = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)

        for j in range(len(list_uid)):
            uid = list_uid[j]
            lvl_p = mean_level_preds_by_uid.get(
                uid, np.array([DEFAULT_LEVEL_P] * 7, dtype=np.float32)
            )
            p_any = _noisy_or_from_levels(lvl_p)
            submission_dict["row_id"].append(f"{uid}_patient_overall")
            submission_dict["fractured"].append(p_any)
else:
    for uid in study_id_list:
        lvl_p = mean_level_preds_by_uid.get(
            uid, np.array([DEFAULT_LEVEL_P] * 7, dtype=np.float32)
        )
        p_any = _noisy_or_from_levels(lvl_p)
        submission_dict["row_id"].append(f"{uid}_patient_overall")
        submission_dict["fractured"].append(p_any)



## === cell 13
pred_df = pd.DataFrame(submission_dict)

pred_df["fractured"] = pred_df["fractured"].astype(np.float32).clip(1e-6, 1 - 1e-6)

sub_template = pd.read_csv(TEST_CSV, usecols=["row_id"])
sub_df = sub_template.merge(pred_df, on="row_id", how="left")

missing = sub_df["fractured"].isna()
if missing.any():
    is_any = sub_df.loc[missing, "row_id"].str.endswith("_patient_overall")
    fill_vals = np.where(is_any.values, DEFAULT_ANY_P, DEFAULT_LEVEL_P).astype(
        np.float32
    )
    sub_df.loc[missing, "fractured"] = fill_vals

sub_df.head(), sub_df.shape, sub_df["fractured"].isna().sum()



## === cell 14
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_df.shape)
print(sub_df.head(10))
