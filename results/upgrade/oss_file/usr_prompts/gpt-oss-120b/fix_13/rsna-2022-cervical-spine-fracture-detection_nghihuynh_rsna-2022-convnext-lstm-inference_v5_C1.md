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

# 5. Code solution

## === cell 0
try:
    import pylibjpeg  # noqa: F401
except Exception:
    pass



## === cell 1
import numpy as np
import pandas as pd
import pydicom

import cv2
import os
from tqdm import tqdm
import glob
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models.convnext import convnext_base

from torch.utils.data import Dataset, DataLoader
from collections import defaultdict



## === cell 2
config = {
    "seq_len": 192,
    "feature_size": 1024,
    "lstm_size": 128,
    "target_size": 512,
    "num_classes": 7,
    "batch_size_image_level": 32,
    "batch_size_patient_level": 8,
    "crop_size": 320,
}

torch.manual_seed(42)
np.random.seed(42)




## === cell 3
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




## === cell 4
test_df = load_df_test()
TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
study_id_list



## === cell 5
selected_image_dict = {}

for uid in tqdm(study_id_list, desc="Preparing slice selection"):
    dicom_files = sorted(
        glob.glob(os.path.join(f"{TEST_PATH}/{uid}", "*.dcm")),
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0]),
    )
    total_slices = len(dicom_files)
    middle_slice = total_slices // 2
    num_side = int(0.15 * total_slices)

    sel = list(np.arange(middle_slice - num_side, middle_slice)) + list(
        np.arange(middle_slice + 1, middle_slice + num_side + 1)
    )
    selected_image_dict[uid] = sel



## === cell 6
print(selected_image_dict[study_id_list[0]])




## === cell 7
def window(data, WL=400, WW=1800):
    slope = getattr(data, "RescaleSlope", 1.0)
    intercept = getattr(data, "RescaleIntercept", 0.0)
    img = data.pixel_array.astype(np.float32)
    img = img * slope + intercept
    upper, lower = WL + WW // 2, WL - WW // 2
    X = np.clip(img, lower, upper)
    X = X - X.min()
    if X.max() != 0:
        X = X / X.max()
    X = (X * 255.0).astype("uint8")
    return X


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))




## === cell 8
mean = np.array([0.456, 0.456, 0.456])
std = np.array([0.224, 0.224, 0.224])
mean_t = torch.tensor(mean).view(3, 1, 1)
std_t = torch.tensor(std).view(3, 1, 1)


class CSFAllImageDataset(Dataset):
    """
    Dataset that iterates over every selected slice of every study.
    Caches the needed DICOM slices per study to avoid repeated disk I/O.
    Performs a centre‑crop and normalisation; returns (tensor, uid).
    """

    def __init__(self, uid_slice_list, target_size, crop_size, selected_image_dict):
        self.uid_slice_list = uid_slice_list  # list of (uid, slice_idx)
        self.target_size = target_size
        self.crop_size = crop_size
        self.start = (target_size - crop_size) // 2
        self.sel_dict = selected_image_dict

        self._cached_uid = None
        self._cached_slices = {}

    def __len__(self):
        return len(self.uid_slice_list)

    def _load_slice(self, uid, s):
        """Load a single slice; on failure return a zero image."""
        try:
            dcm = pydicom.dcmread(os.path.join(TEST_PATH, uid, f"{s}.dcm"))
            img = window(dcm)
        except Exception:
            img = np.zeros((self.target_size, self.target_size), dtype=np.uint8)

        img_resized = cv2.resize(img, (self.target_size, self.target_size))
        return img_resized

    def _ensure_cached(self, uid):
        """Load once all slices required for the given uid."""
        if uid == self._cached_uid:
            return
        needed = set()
        for s in self.sel_dict.get(uid, []):
            needed.update([max(0, s - 1), s, s + 1])
        slices = {}
        for s in needed:
            slices[s] = self._load_slice(uid, s)
        self._cached_uid = uid
        self._cached_slices = slices

    def __getitem__(self, idx):
        uid, slice_idx = self.uid_slice_list[idx]
        self._ensure_cached(uid)

        ids = [max(0, slice_idx - 1), slice_idx, slice_idx + 1]
        imgs = [
            self._cached_slices.get(
                s, np.zeros((self.target_size, self.target_size), dtype=np.uint8)
            )
            for s in ids
        ]

        stacked = np.stack(imgs, axis=-1)  # H,W,3

        cs = self.start
        cropped = stacked[cs : cs + self.crop_size, cs : cs + self.crop_size, :]

        X = cropped.astype(np.float32) / 255.0
        X = (X - mean) / std
        X = img2tensor(X)
        return X, uid


class CSFInstanceDataset(Dataset):
    def __init__(self, feature_array_dict, study_id_list, seq_len):
        self.feature_array_dict = feature_array_dict
        self.study_id_list = study_id_list
        self.seq_len = seq_len

    def __len__(self):
        return len(self.study_id_list)

    def __getitem__(self, index):
        uid = self.study_id_list[index]
        feature_array = self.feature_array_dict.get(
            uid, np.zeros((self.seq_len, config["feature_size"]), dtype=np.float32)
        )
        if feature_array.shape[0] > self.seq_len:
            x = cv2.resize(
                feature_array,
                (feature_array.shape[1], self.seq_len),
                interpolation=cv2.INTER_LINEAR,
            )
        else:
            pad_len = self.seq_len - feature_array.shape[0]
            x = np.pad(
                feature_array,
                pad_width=[(0, pad_len), (0, 0)],
                mode="constant",
                constant_values=0,
            )
        X = torch.tensor(x, dtype=torch.float32)
        return X, uid




## === cell 9
class ConvNextCNN_B_Feature(nn.Module):
    def __init__(self):
        super(ConvNextCNN_B_Feature, self).__init__()
        m = convnext_base()
        in_features = m.classifier[-1].in_features
        self.features = m.features
        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(p=0.5)
        self.fc = nn.Linear(in_features, 7)

    def forward(self, x):
        out = self.features(x)
        out = self.avgpool(out)
        out = self.drop(out)
        feature = out.view(x.size(0), -1)
        out = self.fc(feature)
        return feature, out




## === cell 10
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
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

lv1_model = ConvNextCNN_B_Feature().to(device)
try:
    lv1_state = torch.load(
        "../input/cnn-lstm-oct-21/run_1/run_1/model_1.pth", map_location=device
    )
    lv1_model.load_state_dict(lv1_state)
except Exception:
    pass

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)
try:
    lv2_state = torch.load(
        "../input/cnn-lstm-oct-21/run_1/run_1/model_lstm_1.pth", map_location=device
    )
    lv2_model.load_state_dict(lv2_state)
except Exception:
    pass

lv1_model.eval()
lv2_model.eval()



## === cell 12
flat_uid_slice_list = [
    (uid, s) for uid in study_id_list for s in selected_image_dict[uid]
]

batch_size = min(64, config["batch_size_image_level"] * 2)
num_workers = min(12, max(1, os.cpu_count()))  # slightly more workers for I/O

dataset = CSFAllImageDataset(
    uid_slice_list=flat_uid_slice_list,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
    selected_image_dict=selected_image_dict,
)

loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    pin_memory=True,
    num_workers=num_workers,
    prefetch_factor=2,
    persistent_workers=True,
)

pred_sum_dict = defaultdict(lambda: torch.zeros(7, dtype=torch.float32))
count_dict = defaultdict(int)
features_dict = defaultdict(list)

with torch.no_grad():
    for batch_tensor, uid_batch in tqdm(loader, desc="Image‑level inference"):
        batch_tensor = batch_tensor.to(device)

        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                feats, preds = lv1_model(batch_tensor)
        else:
            feats, preds = lv1_model(batch_tensor)

        preds_sig = preds.sigmoid().cpu()
        feats_cpu = feats.cpu()

        for i, uid in enumerate(uid_batch):
            pred_sum_dict[uid] += preds_sig[i]
            count_dict[uid] += 1
            features_dict[uid].append(feats_cpu[i].numpy())

submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

for uid in study_id_list:
    total = max(count_dict[uid], 1)
    mean_preds = (pred_sum_dict[uid] / total).numpy()
    feature_array = (
        np.vstack(features_dict[uid])
        if features_dict[uid]
        else np.zeros((0, config["feature_size"]), dtype=np.float32)
    )
    feature_array_dict[uid] = feature_array

    for i, label in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{label}")
        submission_dict["fractured"].append(float(mean_preds[i]))



## === cell 13
dataset = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)
generator = DataLoader(
    dataset=dataset,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=True,
    num_workers=0,
)

for features, uid_batch in tqdm(generator, total=len(generator), desc="Patient level"):
    with torch.no_grad():
        features = features.to(device)
        if device.type == "cuda":
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                preds = lv2_model(features)
        else:
            preds = lv2_model(features)

        preds = preds.squeeze(1).sigmoid().cpu().numpy()

    for j, uid in enumerate(uid_batch):
        submission_dict["row_id"].append(f"{uid}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## === cell 14
sub_df = pd.DataFrame(submission_dict)
sub_df = sub_df.sort_values("row_id").reset_index(drop=True)



## === cell 15
sub_df.to_csv("submission.csv", index=False)
