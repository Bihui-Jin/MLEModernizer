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

0.6167917389474165

# 6. Current score

4.96465

# 7. Whether higher score is better

Lower is better

# 8. Previous improvement plans

- What this solution (achieved 4.92598) has done: 'The timeout is dominated by per-slice DICOM decoding done in pure Python loops with `num_workers=0`, plus redundant reads of adjacent slices (`i-1/i/i+1`) across the selected indices. I keep the exact model and inference logic, but (1) parallelize DataLoader reading with multiple workers, (2) add an in-dataset per-worker LRU cache so the same DICOM slice isn’t decoded multiple times, and (3) enable faster DICOM reads by skipping non-pixel metadata and forcing `pydicom` to avoid slow handler auto-detection. These changes preserve identical inputs to the models (same slices, same windowing, same transforms) while cutting repeated I/O/CPU work substantially. I also reduce Python overhead in aggregation by preallocating prediction arrays instead of concatenating many small tensors.'
- What this solution (achieved 4.96465) has done: 'I fix the “Too many open files” crash by explicitly raising the process file-descriptor limit where possible, switching the DataLoader multiprocessing start method to `spawn`, and (most importantly) ensuring workers are not kept alive across studies (disable `persistent_workers`) while also lowering worker counts and prefetch to reduce open pipes/files. I also make DICOM reads close cleanly and avoid holding extra references, which further reduces FD pressure without changing the model inputs. These changes are runtime/stability-focused and preserve the exact model/inference logic; they should allow the full pipeline to complete and produce `submission.csv`. With the full inference completing (instead of failing mid-run and effectively outputting mostly zeros), your leaderboard score should move substantially toward the target.'

# 9. Code solution

## === cell 0
import os, sys
import warnings

warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd

import glob
from tqdm import tqdm

import cv2

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

import pydicom

from albumentations import Compose, CenterCrop



## === cell 1
try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 4096), hard)
    if target_soft != soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
    print("RLIMIT_NOFILE:", resource.getrlimit(resource.RLIMIT_NOFILE))
except Exception as e:
    print("RLIMIT_NOFILE not set:", repr(e))

try:
    import multiprocessing as mp

    mp.set_start_method("spawn", force=True)
except Exception:
    pass

torch.backends.cudnn.benchmark = True
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Device:", device)



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
def load_df_test():
    df_test = pd.read_csv(
        f"../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
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
print("Unique studies in test:", len(study_id_list))
print("First 3:", study_id_list[:3])



## === cell 5
dicom_paths_by_uid = {}
selected_index_dict = {}

for uid in study_id_list:
    paths = glob.glob(os.path.join(TEST_PATH, uid, "*.dcm"))
    paths = sorted(paths, key=lambda p: int(os.path.splitext(os.path.basename(p))[0]))
    dicom_paths_by_uid[uid] = paths

    n = len(paths)
    if n == 0:
        selected_index_dict[uid] = []
        continue

    middle = n // 2
    num_left = num_right = max(1, int(0.15 * n))
    left_start = max(1, middle - num_left)  # keep room for -1 neighbor
    left_end = middle  # exclusive
    right_start = middle + 1
    right_end = min(n - 2, middle + num_right)  # keep room for +1 neighbor

    idxs = list(range(left_start, left_end)) + list(range(right_start, right_end + 1))
    idxs = [i for i in sorted(set(idxs)) if 1 <= i <= n - 2]
    selected_index_dict[uid] = idxs

print("Example selected indices:", selected_index_dict[study_id_list[0]][:20])



## === cell 6
try:
    pydicom.config.image_handlers = []
except Exception:
    pass




## === cell 7
def window(data, WL=400, WW=1800):
    try:
        if hasattr(data, "pixel_array_options"):
            data.pixel_array_options(use_v2_backend=True)
    except Exception:
        pass

    slope = float(getattr(data, "RescaleSlope", 1.0))
    intercept = float(getattr(data, "RescaleIntercept", 0.0))

    img = data.pixel_array.astype(np.float32)
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




## === cell 8
mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)



## === cell 9
from collections import OrderedDict


class CSFImageDataset(Dataset):
    def __init__(self, uid, selected_indices, target_size, crop_size, cache_size=256):
        self.uid = uid
        self.selected_indices = selected_indices
        self.target_size = target_size
        self.crop_size = crop_size
        self.paths = dicom_paths_by_uid[uid]
        self.inference_transform = Compose([CenterCrop(self.crop_size, self.crop_size)])

        self.cache_size = int(cache_size)
        self._cache = OrderedDict()  # path -> uint8 image

    def __len__(self):
        return len(self.selected_indices)

    def _cache_get(self, path):
        if self.cache_size <= 0:
            return None
        v = self._cache.get(path, None)
        if v is not None:
            self._cache.move_to_end(path, last=True)
        return v

    def _cache_put(self, path, value):
        if self.cache_size <= 0:
            return
        self._cache[path] = value
        self._cache.move_to_end(path, last=True)
        if len(self._cache) > self.cache_size:
            self._cache.popitem(last=False)

    def _safe_read_window(self, path):
        cached = self._cache_get(path)
        if cached is not None:
            return cached
        ds = None
        try:
            ds = pydicom.dcmread(
                path,
                stop_before_pixels=False,
                force=True,
                specific_tags=[
                    "PixelData",
                    "RescaleSlope",
                    "RescaleIntercept",
                    "BitsAllocated",
                    "BitsStored",
                    "HighBit",
                    "PixelRepresentation",
                    "SamplesPerPixel",
                    "PhotometricInterpretation",
                    "PlanarConfiguration",
                    "Rows",
                    "Columns",
                    "TransferSyntaxUID",
                ],
            )
            img = window(ds)
        except Exception:
            img = np.zeros((512, 512), dtype=np.uint8)
        finally:
            ds = None
        self._cache_put(path, img)
        return img

    def __getitem__(self, index):
        i = self.selected_indices[index]
        i0 = max(0, i - 1)
        i1 = i
        i2 = min(len(self.paths) - 1, i + 1)

        imgs = [
            self._safe_read_window(self.paths[i0]),
            self._safe_read_window(self.paths[i1]),
            self._safe_read_window(self.paths[i2]),
        ]

        stacked_img = np.stack(imgs, axis=-1)
        stacked_img = cv2.resize(
            stacked_img,
            (self.target_size, self.target_size),
            interpolation=cv2.INTER_AREA,
        )

        out = self.inference_transform(image=stacked_img)
        X = out["image"].astype(np.float32) / 255.0
        X = (X - mean) / std
        X = img2tensor(X)
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




## === cell 12
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




## === cell 13
lv1_model = ConvNextCNN_B_Feature().to(device)
lv1_model.eval()

lv2_model = CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"]).to(
    device
)
lv2_model.eval()

lv1_w_path = "../input/cnn-lstm-oct-21/run_3/run_3/model_3.pth"
lv2_w_path = "../input/cnn-lstm-oct-21/run_4/run_4/model_lstm_3.pth"

if os.path.exists(lv1_w_path):
    sd = torch.load(lv1_w_path, map_location="cpu")
    lv1_model.load_state_dict(sd, strict=True)

if os.path.exists(lv2_w_path):
    sd = torch.load(lv2_w_path, map_location="cpu")
    lv2_model.load_state_dict(sd, strict=True)

print("lv1 weights loaded:", os.path.exists(lv1_w_path))
print("lv2 weights loaded:", os.path.exists(lv2_w_path))




## === cell 14
def _seed_worker(worker_id):
    seed = 0 + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


num_workers_stage1 = 2 if torch.cuda.is_available() else 4
num_workers_stage1 = min(num_workers_stage1, max(1, (os.cpu_count() or 1) // 2))

submission_dict = {"row_id": [], "fractured": []}
feature_array_dict = {}

for uid in study_id_list:
    selected_indices = selected_index_dict[uid]
    if len(selected_indices) == 0:
        for c in ["C1", "C2", "C3", "C4", "C5", "C6", "C7"]:
            submission_dict["row_id"].append(f"{uid}_{c}")
            submission_dict["fractured"].append(0.0)
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        continue

    dataset = CSFImageDataset(
        uid=uid,
        selected_indices=selected_indices,
        target_size=config["target_size"],
        crop_size=config["crop_size"],
        cache_size=256,
    )
    generator = DataLoader(
        dataset,
        batch_size=config["batch_size_image_level"],
        shuffle=False,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        num_workers=num_workers_stage1,
        persistent_workers=False,  # critical to prevent FD accumulation across studies
        prefetch_factor=1 if num_workers_stage1 > 0 else None,
        worker_init_fn=_seed_worker if num_workers_stage1 > 0 else None,
    )

    n_items = len(dataset)
    feature_array = np.zeros((n_items, config["feature_size"]), dtype=np.float32)

    pred_sum = np.zeros((7,), dtype=np.float64)
    pred_count = 0

    for i, images in tqdm(
        enumerate(generator), total=len(generator), desc=f"Stage1 {uid[:18]}"
    ):
        with torch.no_grad():
            start = i * config["batch_size_image_level"]
            end = min(start + config["batch_size_image_level"], n_items)

            images = images.to(device, non_blocking=True)
            features, preds = lv1_model(images)

            feature_array[start:end] = features.detach().cpu().numpy()
            batch_probs = (
                torch.sigmoid(preds)
                .detach()
                .cpu()
                .numpy()
                .astype(np.float64, copy=False)
            )
            pred_sum += batch_probs.sum(axis=0)
            pred_count += batch_probs.shape[0]

    feature_array_dict[uid] = feature_array

    mean_preds = (pred_sum / max(pred_count, 1)).astype(np.float32)

    for k, c in enumerate(["C1", "C2", "C3", "C4", "C5", "C6", "C7"]):
        submission_dict["row_id"].append(f"{uid}_{c}")
        submission_dict["fractured"].append(float(mean_preds[k]))

    del dataset, generator



## --- ERROR in cell 14, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1250         try:
-> 1251             data = self._data_queue.get(timeout=timeout)
   1252             return (True, data)

/usr/lib/python3.11/queue.py in get(self, block, timeout)
    179                         raise Empty
--> 180                     self.not_empty.wait(remaining)
    181             item = self._get()

/usr/lib/python3.11/threading.py in wait(self, timeout)
    330                 if timeout > 0:
--> 331                     gotit = waiter.acquire(True, timeout)
    332                 else:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/signal_handling.py in handler(signum, frame)
     72         # Python can still get and update the process status successfully.
---> 73         _error_if_any_worker_fails()
     74         if previous_handler is not None:

RuntimeError: DataLoader worker (pid 103) exited unexpectedly with exit code 1. Details are lost due to multiprocessing. Rerunning with num_workers=0 may give better error trace.

The above exception was the direct cause of the following exception:

RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2939375896.py in <cell line: 0>()
     49     pred_count = 0
     50 
---> 51     for i, images in tqdm(
     52         enumerate(generator), total=len(generator), desc=f"Stage1 {uid[:18]}"
     53     ):

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
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0
-> 1458             idx, data = self._get_data()
   1459             self._tasks_outstanding -= 1
   1460             if self._dataset_kind == _DatasetKind.Iterable:

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _get_data(self)
   1408         elif self._pin_memory:
   1409             while self._pin_memory_thread.is_alive():
-> 1410                 success, data = self._try_get_data()
   1411                 if success:
   1412                     return data

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _try_get_data(self, timeout)
   1262             if len(failed_workers) > 0:
   1263                 pids_str = ", ".join(str(w.pid) for w in failed_workers)
-> 1264                 raise RuntimeError(
   1265                     f"DataLoader worker (pid(s) {pids_str}) exited unexpectedly"
   1266                 ) from e

RuntimeError: DataLoader worker (pid(s) 103) exited unexpectedly

## === cell 15
num_workers_stage2 = 0

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
    num_workers=num_workers_stage2,
    persistent_workers=False,
)

for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 patient_overall"
):
    with torch.no_grad():
        features = features.to(device, non_blocking=True)
        preds = torch.sigmoid(lv2_model(features)).detach().cpu().numpy().squeeze()

    preds = np.atleast_1d(preds)
    for j in range(len(list_uid)):
        submission_dict["row_id"].append(f"{list_uid[j]}_patient_overall")
        submission_dict["fractured"].append(float(preds[j]))



## --- ERROR in cell 15, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/659104918.py in <cell line: 0>()
     16 )
     17 
---> 18 for features, list_uid in tqdm(
     19     generator2, total=len(generator2), desc="Stage2 patient_overall"
     20 ):

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

/tmp/ipykernel_55/291513014.py in __getitem__(self, index)
     10     def __getitem__(self, index):
     11         uid = self.study_id_list[index]
---> 12         feature_array = self.feature_array_dict[uid]
     13         if len(feature_array) > self.seq_len:
     14             x = cv2.resize(

KeyError: '1.2.826.0.1.3680043.6200'

## === cell 16
sub_df = pd.DataFrame.from_dict(submission_dict)

sub_df = sub_df.drop_duplicates(subset=["row_id"], keep="last")

required = test_df[["row_id"]].copy()
sub_df = required.merge(sub_df, on="row_id", how="left")

sub_df["fractured"] = sub_df["fractured"].fillna(0.0).astype(np.float32)
sub_df["fractured"] = np.clip(sub_df["fractured"].values, 1e-6, 1 - 1e-6)

print(sub_df.head())
print("Submission shape:", sub_df.shape)
assert list(sub_df.columns) == ["row_id", "fractured"]
assert len(sub_df) == len(test_df)



## === cell 17
sub_df.to_csv("submission.csv", index=False)
print("Wrote submission.csv with rows:", len(sub_df))
print(sub_df.tail())
