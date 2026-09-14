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

import numpy as np
import pandas as pd
import pydicom

import cv2
from tqdm import tqdm
import glob

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from torchvision.models.convnext import convnext_base, ConvNeXt_Base_Weights

try:
    from pydicom.pixels import pixel_array as pyd_pixel_array
except Exception:
    pyd_pixel_array = None

try:
    import resource

    soft, hard = resource.getrlimit(resource.RLIMIT_NOFILE)
    target_soft = min(max(soft, 8192), hard)
    if target_soft > soft:
        resource.setrlimit(resource.RLIMIT_NOFILE, (target_soft, hard))
except Exception:
    pass



## === cell 1
torch.manual_seed(0)
np.random.seed(0)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device



## === cell 2
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




## === cell 3
def load_df_test():
    df_test = pd.read_csv(
        "../input/rsna-2022-cervical-spine-fracture-detection/test.csv"
    )
    return df_test


test_df = load_df_test()
TEST_PATH = "../input/rsna-2022-cervical-spine-fracture-detection/test_images"
study_id_list = list(test_df.StudyInstanceUID.unique())
len(study_id_list), study_id_list[:3]




## === cell 4
def _list_sorted_dicoms(folder: str):
    files = []
    try:
        with os.scandir(folder) as it:
            for e in it:
                if not e.is_file():
                    continue
                name = e.name
                if not name.endswith(".dcm"):
                    continue
                stem = name[:-4]
                try:
                    key = int(stem)
                except Exception:
                    key = stem
                files.append((key, e.path))
    except FileNotFoundError:
        return []
    files.sort(key=lambda x: x[0])
    return [p for _, p in files]


study_slices = {}
for uid in study_id_list:
    study_slices[uid] = _list_sorted_dicoms(os.path.join(TEST_PATH, uid))

selected_index_dict = {}
for uid in study_id_list:
    files = study_slices[uid]
    n = len(files)
    if n == 0:
        selected_index_dict[uid] = []
        continue
    middle = n // 2
    k = max(1, int(0.15 * n))
    left = max(0, middle - k)
    right = min(n - 1, middle + k)
    idxs = list(range(left, middle)) + list(range(middle + 1, right + 1))
    if len(idxs) == 0:
        idxs = [middle]
    selected_index_dict[uid] = idxs

selected_index_dict[study_id_list[0]][:10], len(selected_index_dict[study_id_list[0]])




## === cell 5
def _decode_encapsulated_jpeg_lossless_to_array(
    ds: pydicom.dataset.FileDataset,
) -> np.ndarray:
    """
    Bugfix: Kaggle environment may lack pydicom's JPEG plugins (gdcm/pylibjpeg),
    but OpenCV can still decode the encapsulated JPEG bitstream for many cases.
    This tries to extract the first encapsulated frame and decode it via cv2.imdecode.
    """
    try:
        ts = getattr(ds.file_meta, "TransferSyntaxUID", None)
        if ts is None or not getattr(ts, "is_compressed", False):
            return None

        from pydicom.encaps import generate_pixel_data_frame

        frame = next(generate_pixel_data_frame(ds.PixelData))
        buf = np.frombuffer(frame, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if img is None:
            return None

        if img.ndim == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        return img
    except Exception:
        return None


def _get_pixel_array(ds: pydicom.dataset.FileDataset) -> np.ndarray:
    """
    Robust pixel array extraction for compressed DICOMs.

    Order:
      1) pydicom.pixels backend (if available plugins exist)
      2) OpenCV decode of encapsulated JPEG frame (works without gdcm/pylibjpeg)
      3) ds.pixel_array (may still fail for compressed)
    """
    if pyd_pixel_array is not None:
        for plugin in (None, "pylibjpeg", "gdcm"):
            try:
                if plugin is None:
                    return pyd_pixel_array(ds)
                return pyd_pixel_array(ds, decoding_plugin=plugin)
            except Exception:
                pass

    arr = _decode_encapsulated_jpeg_lossless_to_array(ds)
    if arr is not None:
        return arr

    return ds.pixel_array


def window(ds, WL=400, WW=1800):
    slope = float(getattr(ds, "RescaleSlope", 1.0))
    intercept = float(getattr(ds, "RescaleIntercept", 0.0))
    img = _get_pixel_array(ds).astype(np.float32)
    img = img * slope + intercept

    upper, lower = WL + WW // 2, WL - WW // 2
    x = np.clip(img, lower, upper)
    x = x - x.min()
    m = x.max()
    if m > 0:
        x = x / m
    x = (x * 255.0).astype(np.uint8)
    return x


def img2tensor(img, dtype: np.dtype = np.float32):
    if img.ndim == 2:
        img = np.expand_dims(img, 2)
    img = np.transpose(img, (2, 0, 1))
    return torch.from_numpy(img.astype(dtype, copy=False))


mean = np.array([0.456, 0.456, 0.456], dtype=np.float32)
std = np.array([0.224, 0.224, 0.224], dtype=np.float32)



## === cell 6
from collections import OrderedDict


class _GlobalLRUCache:
    __slots__ = ("max_items", "_d")

    def __init__(self, max_items: int = 4096):
        self.max_items = int(max_items)
        self._d = OrderedDict()

    def get(self, k):
        try:
            v = self._d.pop(k)
        except KeyError:
            return None
        self._d[k] = v  # move to end (most recently used)
        return v

    def put(self, k, v):
        d = self._d
        if k in d:
            try:
                d.pop(k)
            except KeyError:
                pass
            d[k] = v
            return
        d[k] = v
        if len(d) > self.max_items:
            d.popitem(last=False)  # evict LRU


class CSFAllStudiesImageDataset(Dataset):
    def __init__(self, study_slices, selected_index_dict, target_size, crop_size):
        self.study_slices = study_slices
        self.selected_index_dict = selected_index_dict
        self.target_size = target_size
        self.crop_size = crop_size

        self.uid_minmax = {}
        items = []
        for uid, idxs in selected_index_dict.items():
            if not idxs:
                continue
            files = study_slices.get(uid, [])
            n = len(files)
            if n == 0:
                continue
            mn = max(0, int(min(idxs)) - 1)
            mx = min(n - 1, int(max(idxs)) + 1)
            self.uid_minmax[uid] = (mn, mx)
            for i in idxs:
                items.append((uid, int(i)))
        self.items = items

        self._cache = _GlobalLRUCache(max_items=8192)

        self._hot_uid = None
        self._hot_mnmx = None
        self._hot_imgs = None

    def __len__(self):
        return len(self.items)

    def _read_windowed(self, uid: str, file_idx: int) -> np.ndarray:
        k = (uid, int(file_idx))
        img = self._cache.get(k)
        if img is not None:
            return img

        fp = self.study_slices[uid][file_idx]
        ds = pydicom.dcmread(
            fp,
            force=True,
            stop_before_pixels=False,
            specific_tags=[
                "PixelData",
                "RescaleSlope",
                "RescaleIntercept",
                "TransferSyntaxUID",
            ],
        )
        img = window(ds)
        self._cache.put(k, img)
        return img

    def _ensure_uid_range_loaded(self, uid: str):
        mnmx = self.uid_minmax.get(uid)
        if mnmx is None:
            return
        if (
            self._hot_uid == uid
            and self._hot_mnmx == mnmx
            and self._hot_imgs is not None
        ):
            return

        mn, mx = mnmx
        imgs = [None] * (mx - mn + 1)
        for idx in range(mn, mx + 1):
            imgs[idx - mn] = self._read_windowed(uid, idx)

        self._hot_uid = uid
        self._hot_mnmx = mnmx
        self._hot_imgs = (mn, imgs)

    def __getitem__(self, index):
        uid, i = self.items[index]
        try:
            files = self.study_slices[uid]
            n = len(files)
            i0 = max(0, i - 1)
            i2 = min(n - 1, i + 1)

            self._ensure_uid_range_loaded(uid)
            if self._hot_uid == uid and self._hot_imgs is not None:
                mn, imgs = self._hot_imgs
                stacked = np.stack(
                    [imgs[i0 - mn], imgs[i - mn], imgs[i2 - mn]], axis=-1
                )  # H,W,3
            else:
                stacked = np.stack(
                    [
                        self._read_windowed(uid, i0),
                        self._read_windowed(uid, i),
                        self._read_windowed(uid, i2),
                    ],
                    axis=-1,
                )  # H,W,3

            stacked = cv2.resize(
                stacked,
                (self.target_size, self.target_size),
                interpolation=cv2.INTER_LINEAR,
            )

            cs = self.crop_size
            h, w = stacked.shape[:2]
            y0 = max(0, (h - cs) // 2)
            x0 = max(0, (w - cs) // 2)
            cropped = stacked[y0 : y0 + cs, x0 : x0 + cs, :]

            x = (cropped.astype(np.float32) / 255.0 - mean) / std
            x = img2tensor(x)
            return x, uid
        except Exception:
            cs = self.crop_size
            cropped = np.zeros((cs, cs, 3), dtype=np.uint8)
            x = (cropped.astype(np.float32) / 255.0 - mean) / std
            x = img2tensor(x)
            return x, uid


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
            uid, np.zeros((1, config["feature_size"]), dtype=np.float32)
        )
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




## === cell 8
lv1_model = ConvNextCNN_B_Feature().to(device).eval()
lv2_model = (
    CSFNet(input_len=config["feature_size"], lstm_size=config["lstm_size"])
    .to(device)
    .eval()
)

if device.type == "cuda":
    lv1_model = lv1_model.to(memory_format=torch.channels_last)



## === cell 9
feature_array_dict = {}
preds_level_by_uid = {}

use_cuda = torch.cuda.is_available()

uid_feat_writepos = {}
uid_counts = {}
uid_pred_sums = {}
for uid in study_id_list:
    idxs = selected_index_dict.get(uid, [])
    if (not idxs) or (len(study_slices.get(uid, [])) == 0):
        feature_array_dict[uid] = np.zeros(
            (1, config["feature_size"]), dtype=np.float32
        )
        preds_level_by_uid[uid] = np.full((7,), 0.01, dtype=np.float32)
        uid_feat_writepos[uid] = 0
        uid_counts[uid] = 0
        uid_pred_sums[uid] = np.zeros((7,), dtype=np.float32)
        continue
    n = len(idxs)
    feature_array_dict[uid] = np.zeros((n, config["feature_size"]), dtype=np.float32)
    uid_feat_writepos[uid] = 0
    uid_counts[uid] = n
    uid_pred_sums[uid] = np.zeros((7,), dtype=np.float32)

all_ds = CSFAllStudiesImageDataset(
    study_slices=study_slices,
    selected_index_dict=selected_index_dict,
    target_size=config["target_size"],
    crop_size=config["crop_size"],
)

cpu_cnt = os.cpu_count() or 1
num_workers_img = min(4, max(2, cpu_cnt // 2))

generator = DataLoader(
    all_ds,
    batch_size=config["batch_size_image_level"],
    shuffle=False,
    pin_memory=use_cuda,
    drop_last=False,
    num_workers=num_workers_img,
    persistent_workers=(num_workers_img > 0),
    prefetch_factor=2 if num_workers_img > 0 else None,
)

for images, uids in tqdm(generator, total=len(generator), desc="Stage1 batches"):
    with torch.inference_mode():
        if device.type == "cuda":
            images = images.to(device, non_blocking=True).contiguous(
                memory_format=torch.channels_last
            )
        else:
            images = images.to(device, non_blocking=True)
        features, preds = lv1_model(images)
        feats_np = features.detach().cpu().numpy().astype(np.float32, copy=False)
        probs_np = (
            torch.sigmoid(preds).detach().cpu().numpy().astype(np.float32, copy=False)
        )

    for j, uid in enumerate(uids):
        uid = str(uid)
        wp = uid_feat_writepos[uid]
        feature_array_dict[uid][wp] = feats_np[j]
        uid_feat_writepos[uid] = wp + 1
        uid_pred_sums[uid] += probs_np[j]

for uid in study_id_list:
    n = uid_counts.get(uid, 0)
    if n > 0:
        preds_level_by_uid[uid] = (uid_pred_sums[uid] / float(n)).astype(
            np.float32, copy=False
        )

len(feature_array_dict), list(feature_array_dict.keys())[:3]



## === cell 10
dataset2 = CSFInstanceDataset(
    feature_array_dict=feature_array_dict,
    study_id_list=study_id_list,
    seq_len=config["seq_len"],
)

num_workers_seq = 0

generator2 = DataLoader(
    dataset=dataset2,
    batch_size=config["batch_size_patient_level"],
    shuffle=False,
    pin_memory=torch.cuda.is_available(),
    num_workers=num_workers_seq,
    persistent_workers=False,
)

preds_patient = {}
for features, list_uid in tqdm(
    generator2, total=len(generator2), desc="Stage2 batches"
):
    with torch.inference_mode():
        features = features.to(device, non_blocking=True)
        logits = lv2_model(features)
        p = torch.sigmoid(logits).squeeze(1).detach().cpu().numpy().astype(np.float32)
    for j, uid in enumerate(list_uid):
        preds_patient[str(uid)] = float(p[j])

len(preds_patient), list(preds_patient.items())[:2]



## === cell 11
sub = test_df[["row_id", "StudyInstanceUID", "prediction_type"]].copy()

uids = sub["StudyInstanceUID"].to_numpy()
ptypes = sub["prediction_type"].to_numpy()

out = np.empty(len(sub), dtype=np.float32)

is_patient = ptypes == "patient_overall"
if is_patient.any():
    out[is_patient] = np.fromiter(
        (preds_patient.get(uid, 0.01) for uid in uids[is_patient]),
        dtype=np.float32,
        count=int(is_patient.sum()),
    )

if (~is_patient).any():
    level_idx = (
        pd.Series(ptypes[~is_patient]).str.slice(1).astype(np.int16).to_numpy() - 1
    )
    sel_uids = uids[~is_patient]

    default_levels = np.full((7,), 0.01, dtype=np.float32)
    levels_mat = np.vstack(
        [preds_level_by_uid.get(uid, default_levels) for uid in sel_uids]
    ).astype(np.float32, copy=False)
    out[~is_patient] = levels_mat[np.arange(level_idx.shape[0]), level_idx]

sub["fractured"] = out
sub = sub[["row_id", "fractured"]]
sub["fractured"] = sub["fractured"].clip(1e-4, 1 - 1e-4)

sub.head(), sub.shape



## === cell 12
sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sub.shape)
print(sub.head(10).to_string(index=False))
