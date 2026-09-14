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

0.6435436510572724

# 6. Current score

1.08162

# 7. Whether higher score is better

Lower is better.

# 8. Previous improvement plans

- What this solution (achieved 1.08162) has done: 'I first remove the broken on-the-fly wheel installs (they target cp37 and fail on Python 3.10) and instead rely on the Kaggle environment’s built-in DICOM decoders by explicitly using `pydicom.dcmread(..., force=True)` plus safe pixel decoding fallbacks. Next, I fix missing model-weight paths by adding a minimal, score-stable fallback that generates reasonable probabilities when the external `/kaggle/input/cnn-lstm-oct-21/...` files are not present, so the notebook always produces a valid `submission.csv`. I also fix DICOM slice indexing (the dataset was trying to read `0.dcm` / non-existent slices) by selecting actual slice numbers from filenames and clamping neighbor selection. Finally, I ensure the submission exactly matches `test.csv` row ordering/structure by merging predictions onto `test_df[['row_id','StudyInstanceUID','prediction_type']]` rather than sorting lexicographically.'
- What this solution (achieved 1.08162) has done: 'Your current score (1.08162, lower-is-better) is worse than the target (0.6435), and the biggest limiter is that you are almost certainly running in the “no pretrained weights” fallback path, which outputs constant priors and cannot be competitive. I keep your model and inference logic identical, but fix the weight-loading to automatically discover the correct `.pth` files anywhere under `/kaggle/input/` (including nested/differently-named dataset folders) so the real CNN+GRU predictions are used. I also ensure the CNN outputs 1024-d features (ConvNeXt-Base default) by removing the dropout-after-pool shape issue and using the standard pooled features; this preserves architecture intent but avoids subtle feature mismatch if weights were trained expecting pooled features. Finally, I keep your submission alignment logic unchanged and still fall back to priors only if weights truly cannot be found.'
- What this solution (achieved 1.08162) has done: 'Your current loss (1.08162, lower-is-better) is far above the target (0.6435), and the most likely reason is that you’re still running without the intended pretrained weights, so the fallback priors dominate. I keep the same ConvNeXt+GRU core inference, but make the weight discovery/load robust to common Kaggle packaging differences (nested folders, different key prefixes like `module.`/`model.`) so the real weights actually get used when present. I also ensure the ConvNeXt feature extractor returns the exact pooled feature vector the GRU expects (1024-d) without altering the model’s semantics, and I keep the submission alignment identical while adding a safety check that all 8 rows per study are produced. These are minimal, execution-safe changes that should materially reduce the logloss if weights exist anywhere under `/kaggle/input/`.'
- What this solution (achieved 1.08162) has done: 'Your current loss (1.08162, lower-is-better) is far worse than the target (0.6435), and the most likely cause is that the models are still effectively running untrained because the ConvNeXt feature head you return includes dropout, which changes the feature distribution the GRU expects (and also makes inference nondeterministic). I keep the same ConvNeXt+GRU architecture and inference flow, but return the *pre-dropout pooled features* to the GRU while still applying dropout only to the classification head, matching the common training intent and stabilizing patient-level predictions. I also clamp/clean any NaN/Inf probabilities right after sigmoid (rare but can happen) to prevent logloss blow-ups, without changing semantics. Everything else (slice selection, transforms, weight discovery, submission alignment) remains identical.'

# 9. Code solution

## === cell 0
import os, sys, glob, time, pickle
import numpy as np
import pandas as pd



## === cell 1
import warnings

warnings.filterwarnings("ignore")

import pydicom
import cv2
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from albumentations import Compose, CenterCrop

from torchvision.models.convnext import convnext_base



## === cell 2
torch.backends.cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



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
    df_test = pd.read_csv(
        "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )

    if len(df_test) > 0 and df_test.iloc[0].row_id == "1.2.826.0.1.3680043.10197_C1":
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
TEST_PATH = "/kaggle/input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]




## === cell 6
def list_dicom_slices(uid):
    files = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    nums = []
    for f in files:
        base = os.path.splitext(os.path.basename(f))[0]
        try:
            nums.append(int(base))
        except:
            pass
    nums = sorted(nums)
    return nums


selected_image_dict = {}
dicom_index_dict = {}  # uid -> sorted slice numbers present on disk
for uid in study_id_list:
    nums = list_dicom_slices(uid)
    dicom_index_dict[uid] = nums
    if len(nums) == 0:
        selected_image_dict[uid] = []
        continue
    mid_pos = len(nums) // 2
    k = max(1, int(0.15 * len(nums)))
    left = max(0, mid_pos - k)
    right = min(len(nums) - 1, mid_pos + k)
    selected_positions = list(range(left, right + 1))
    selected_image_dict[uid] = [nums[p] for p in selected_positions]



## === cell 7
example_uid = study_id_list[0]
print(example_uid)
print("n_slices:", len(dicom_index_dict[example_uid]))
print("selected slice numbers (first 10):", selected_image_dict[example_uid][:10])




## === cell 8
def safe_pixel_array(ds):
    try:
        arr = ds.pixel_array
        return arr
    except Exception:
        return None




## === cell 9
def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))

    img = safe_pixel_array(ds)
    if img is None:
        return np.zeros((512, 512), dtype=np.uint8)

    img = img.astype(np.float32) * slope + intercept

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
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 10
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)




## === cell 11
class CSFImageDataset(Dataset):
    def __init__(self, uid, image_list, target_size, crop_size, dicom_nums):
        self.uid = uid
        self.image_list = image_list
        self.target_size = target_size
        self.crop_size = crop_size
        self.dicom_nums = dicom_nums
        self.num_set = set(dicom_nums)

    def __len__(self):
        return len(self.image_list)

    def _nearest_existing(self, x):
        if x in self.num_set:
            return x
        x = min(max(x, self.dicom_nums[0]), self.dicom_nums[-1])
        import bisect

        i = bisect.bisect_left(self.dicom_nums, x)
        if i == 0:
            return self.dicom_nums[0]
        if i >= len(self.dicom_nums):
            return self.dicom_nums[-1]
        before = self.dicom_nums[i - 1]
        after = self.dicom_nums[i]
        return before if abs(x - before) <= abs(after - x) else after

    def __getitem__(self, index):
        PATH = os.path.join(TEST_PATH, self.uid)

        center = int(self.image_list[index])
        prev_n = self._nearest_existing(center - 1)
        curr_n = self._nearest_existing(center)
        next_n = self._nearest_existing(center + 1)

        data_list = [
            pydicom.dcmread(f"{PATH}/{prev_n}.dcm", force=True),
            pydicom.dcmread(f"{PATH}/{curr_n}.dcm", force=True),
            pydicom.dcmread(f"{PATH}/{next_n}.dcm", force=True),
        ]

        imgs = [window(ds) for ds in data_list]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_LINEAR,
        )

        inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])
        stacked_img = inference_transform(image=stacked_img)["image"]

        X = img2tensor((stacked_img / 255.0 - mean) / std)
        return X




## === cell 12
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




## === cell 13
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base(weights=None)
        in_features = m.classifier[-1].in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features=in_features, out_features=7)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        feature = out.view(x.size(0), -1)

        dropped = self.drop(feature)
        out = self.fc(dropped)
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




## === cell 14
lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

lv1_w_path = "/kaggle/input/cnn-lstm-oct-21/run_1/run_1/model_1.pth"
lv2_w_path = "/kaggle/input/cnn-lstm-oct-21/run_1/run_1/model_lstm_1.pth"


def _find_weight_file(filename):
    candidates = glob.glob(f"/kaggle/input/**/{filename}", recursive=True)
    if len(candidates) == 0:
        return None
    candidates = sorted(candidates, key=lambda p: (len(p), p))
    return candidates[0]


def _clean_state_dict(sd):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    elif isinstance(sd, dict) and "model" in sd and isinstance(sd["model"], dict):
        sd = sd["model"]

    if not isinstance(sd, dict):
        return sd

    new_sd = {}
    for k, v in sd.items():
        nk = k
        if nk.startswith("module."):
            nk = nk[len("module.") :]
        if nk.startswith("model."):
            nk = nk[len("model.") :]
        new_sd[nk] = v
    return new_sd


def _load_weights(model, path, device):
    raw = torch.load(path, map_location=device)
    sd = _clean_state_dict(raw)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if len(unexpected) > 0:
        print(
            f"Note: unexpected keys when loading {os.path.basename(path)}:",
            unexpected[:5],
        )
    if len(missing) > 0:
        print(f"Note: missing keys when loading {os.path.basename(path)}:", missing[:5])


if not (os.path.exists(lv1_w_path) and os.path.exists(lv2_w_path)):
    f1 = _find_weight_file("model_1.pth")
    f2 = _find_weight_file("model_lstm_1.pth")
    if f1 is not None:
        lv1_w_path = f1
    if f2 is not None:
        lv2_w_path = f2

have_weights = os.path.exists(lv1_w_path) and os.path.exists(lv2_w_path)
print("lv1_w_path:", lv1_w_path, "exists:", os.path.exists(lv1_w_path))
print("lv2_w_path:", lv2_w_path, "exists:", os.path.exists(lv2_w_path))

if have_weights:
    _load_weights(lv1_model, lv1_w_path, device)
    _load_weights(lv2_model, lv2_w_path, device)
else:
    print(
        "WARNING: pretrained weights not found. "
        "Will generate predictions from safe priors to produce a valid submission."
    )



## === cell 15
submission_rows = []
feature_array_dict = {}

prior_level = np.array([0.03, 0.03, 0.03, 0.03, 0.03, 0.03, 0.03], dtype=np.float32)
prior_patient = 0.06

for uid in tqdm(study_id_list, total=len(study_id_list)):
    image_list = selected_image_dict.get(uid, [])
    dicom_nums = dicom_index_dict.get(uid, [])

    if (not have_weights) or (len(image_list) == 0) or (len(dicom_nums) == 0):
        for k, name in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
            submission_rows.append((f"{uid}_{name}", float(prior_level[k])))
        feature_array_dict[uid] = np.zeros(
            (max(1, len(image_list)), config["feature_size"]), dtype=np.float32
        )
        continue

    dataset = CSFImageDataset(
        uid=uid,
        image_list=image_list,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
        dicom_nums=dicom_nums,
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
    feature_array = np.zeros((len(dataset), config["feature_size"]), dtype=np.float32)

    for i, images in enumerate(generator):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + images.size(0), len(dataset))

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = features.detach().cpu().numpy()

            p = preds.sigmoid().detach().cpu()
            p = torch.nan_to_num(p, nan=0.5, posinf=1.0, neginf=0.0).clamp(
                1e-6, 1 - 1e-6
            )
            preds_uid.append(p)

    feature_array_dict[uid] = feature_array
    mean_preds = torch.mean(torch.cat(preds_uid, dim=0), dim=0).numpy()

    for k, name in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_rows.append((f"{uid}_{name}", float(mean_preds[k])))



## === cell 16
if have_weights:
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

    for features, list_uid in tqdm(generator2, total=len(generator2)):
        with torch.no_grad():
            features = features.to(device, non_blocking=True)
            preds = lv2_model(features).sigmoid().detach().cpu().numpy().reshape(-1)
            preds = np.nan_to_num(preds, nan=0.5, posinf=1.0, neginf=0.0)
            preds = np.clip(preds, 1e-6, 1 - 1e-6)

        for j in range(len(list_uid)):
            submission_rows.append((f"{list_uid[j]}_patient_overall", float(preds[j])))
else:
    for uid in study_id_list:
        submission_rows.append((f"{uid}_patient_overall", float(prior_patient)))



## === cell 17
pred_map = {rid: p for rid, p in submission_rows}

sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()
sub["fractured"] = sub["row_id"].map(pred_map)

level_prior_map = {f"C{i}": float(prior_level[i - 1]) for i in range(1, 8)}
sub["fractured"] = sub.apply(
    lambda r: (
        level_prior_map.get(r["prediction_type"], float(prior_patient))
        if pd.isna(r["fractured"])
        else r["fractured"]
    ),
    axis=1,
)

sub["fractured"] = pd.to_numeric(sub["fractured"], errors="coerce").astype(np.float32)
sub["fractured"] = np.nan_to_num(
    sub["fractured"].values, nan=0.5, posinf=1.0, neginf=0.0
).astype(np.float32)
sub["fractured"] = np.clip(sub["fractured"], 1e-6, 1 - 1e-6)

print(sub[["row_id", "fractured"]].head())
print(
    "submission rows:",
    len(sub),
    "missing_after_fill:",
    int(pd.isna(sub["fractured"]).sum()),
)

counts = sub.groupby("StudyInstanceUID")["row_id"].count().value_counts().sort_index()
print("rows per study distribution:", counts.to_dict())

sub_out = sub[["row_id", "fractured"]].copy()



## === cell 18
sub_out.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub_out.shape)
print(sub_out.describe(include="all"))
