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
Predict the genetic subtype of glioblastoma using MRI (magnetic resonance imaging) scans to detect for the presence of MGMT promoter methylation.

## Metric
Area under the ROC curve between the predicted probability and the observed target.

## Submission Format
For each `BraTS21ID` in the test set, you must predict a probability for the target `MGMT_value`. The file should contain a header and have the following format:

```
BraTS21ID,MGMT_value
00001,0.5
00013,0.5
00015,0.5
etc.
```

## Dataset
- **train/** - folder containing the training files, with each top-level folder representing a subject. **NOTE:** There are some unexpected issues with the following three cases in the training dataset, participants can exclude the cases during training: `[00109, 00123, 00709]`. We have checked and confirmed that the testing dataset is free from such issues.
- **train_labels.csv** - file containing the target `MGMT_value` for each subject in the training data (e.g. the presence of MGMT promoter methylation)
- **test/** - the test files, which use the same structure as `train/`; your task is to predict the `MGMT_value` for each subject in the test data. **NOTE**: the total size of the rerun test set (Public and Private) is ~5x the size of the Public test set
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.9

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pydicom==3.0.1
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        input/
            description.md (202 lines)
            sample_submission.csv (60 lines)
            sample_submission.csv.zip (382 Bytes)
            test.zip (1.3 GB)
            train.zip (10.2 GB)
            train_labels.csv (527 lines)
            train_labels.csv.zip (1.4 kB)
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
            test/
                00002/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 29 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 382 other files
                00019/
                    FLAIR/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 30 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-258.dcm (525.4 kB)
                        Image-259.dcm (525.4 kB)
                        ... and 127 other files
                ... and 58 other folders
            train/
                00000/
                    FLAIR/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 398 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.3 kB)
                        Image-10.dcm (525.3 kB)
                        ... and 406 other files
                00003/
                    FLAIR/
                        Image-387.dcm (525.4 kB)
                        Image-388.dcm (525.4 kB)
                        ... and 127 other files
                    T1w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 31 other files
                    T1wCE/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 127 other files
                    T2w/
                        Image-1.dcm (525.4 kB)
                        Image-10.dcm (525.4 kB)
                        ... and 406 other files
                ... and 525 other folders
        working/
            rsna-miccai-brain-tumor-radiogenomic-classification/
                description.md (202 lines)
                sample_submission.csv (60 lines)
                ... and 5 other files
                rsna-miccai-brain-tumor-radiogenomic-classification/
                test/
                    00002/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00019/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 58 other folders
                train/
                    00000/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    00003/
                        FLAIR/
                            ... (max depth reached)
                        T1w/
                            ... (max depth reached)
                        T1wCE/
                            ... (max depth reached)
                        T2w/
                            ... (max depth reached)
                    ... and 525 other folders
```

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> data/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/sample_submission.csv has 59 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> input/rsna-miccai-brain-tumor-radiogenomic-classification/train_labels.csv has 526 rows and 2 columns.
The columns are: BraTS21ID, MGMT_value

-> (stopped after 10 files for performance)

# 5. Target score

-1.0

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import glob
from tqdm.auto import tqdm
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt
import pydicom
from pydicom.pixel_data_handlers.util import apply_voi_lut
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

import warnings

warnings.filterwarnings("ignore")


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


seed_everything(42)



## === cell 1
import math
from functools import partial

dropout = 0.5


def get_inplanes():
    return [64, 128, 256, 512]


def conv3x3x3(in_planes, out_planes, stride=1):
    return nn.Conv3d(
        in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False
    )


def conv1x1x1(in_planes, out_planes, stride=1):
    return nn.Conv3d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(in_planes, planes, stride)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv1x1x1(in_planes, planes)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = conv3x3x3(planes, planes, stride)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = conv1x1x1(planes, planes * self.expansion)
        self.bn3 = nn.BatchNorm3d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv3(out)
        out = self.bn3(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        out = self.dropout(out)
        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        block_inplanes,
        n_input_channels=3,
        conv1_t_size=7,
        conv1_t_stride=1,
        no_max_pool=False,
        shortcut_type="B",
        widen_factor=1.0,
        n_classes=400,
    ):
        super().__init__()
        block_inplanes = [int(x * widen_factor) for x in block_inplanes]
        self.in_planes = block_inplanes[0]
        self.no_max_pool = no_max_pool

        self.conv1 = nn.Conv3d(
            n_input_channels,
            self.in_planes,
            kernel_size=(conv1_t_size, 7, 7),
            stride=(conv1_t_stride, 2, 2),
            padding=(conv1_t_size // 2, 3, 3),
            bias=False,
        )
        self.bn1 = nn.BatchNorm3d(self.in_planes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(
            block, block_inplanes[0], layers[0], shortcut_type
        )
        self.layer2 = self._make_layer(
            block, block_inplanes[1], layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, block_inplanes[2], layers[2], shortcut_type, stride=2
        )
        self.layer4 = self._make_layer(
            block, block_inplanes[3], layers[3], shortcut_type, stride=2
        )

        self.avgpool = nn.AdaptiveAvgPool3d((1, 1, 1))
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Linear(block_inplanes[3] * block.expansion, n_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm3d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def _downsample_basic_block(self, x, planes, stride):
        out = F.avg_pool3d(x, kernel_size=1, stride=stride)
        zero_pads = torch.zeros(
            out.size(0),
            planes - out.size(1),
            out.size(2),
            out.size(3),
            out.size(4),
            device=out.device,
        )
        out = torch.cat([out, zero_pads], dim=1)
        return out

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    self._downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                )
            else:
                downsample = nn.Sequential(
                    conv1x1x1(self.in_planes, planes * block.expansion, stride),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = []
        layers.append(
            block(
                in_planes=self.in_planes,
                planes=planes,
                stride=stride,
                downsample=downsample,
            )
        )
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        if not self.no_max_pool:
            x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.dropout(x)
        x = self.fc(x)
        return x


def resnet3d(model_depth, **kwargs):
    assert model_depth in [10, 18, 34, 50, 101, 152, 200]
    if model_depth == 10:
        model = ResNet(BasicBlock, [1, 1, 1, 1], get_inplanes(), **kwargs)
    elif model_depth == 18:
        model = ResNet(BasicBlock, [2, 2, 2, 2], get_inplanes(), **kwargs)
    elif model_depth == 34:
        model = ResNet(BasicBlock, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 50:
        model = ResNet(Bottleneck, [3, 4, 6, 3], get_inplanes(), **kwargs)
    elif model_depth == 101:
        model = ResNet(Bottleneck, [3, 4, 23, 3], get_inplanes(), **kwargs)
    elif model_depth == 152:
        model = ResNet(Bottleneck, [3, 8, 36, 3], get_inplanes(), **kwargs)
    elif model_depth == 200:
        model = ResNet(Bottleneck, [3, 24, 36, 3], get_inplanes(), **kwargs)
    return model




## === cell 2
path = "/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification"



## === cell 3
img_size = 200


def dicom2array(dcm_path, voi_lut=True, fix_monochrome=True):
    dicom = pydicom.dcmread(dcm_path)
    if voi_lut:
        data = apply_voi_lut(dicom.pixel_array, dicom)
    else:
        data = dicom.pixel_array

    if (
        fix_monochrome
        and getattr(dicom, "PhotometricInterpretation", None) == "MONOCHROME1"
    ):
        data = np.amax(data) - data

    data = data.astype(np.float32)
    data = data - np.min(data)
    mx = np.max(data)
    if mx > 0:
        data = data / mx
    data = (data * 255.0).astype(np.uint8)
    data = cv2.resize(data, (img_size, img_size), interpolation=cv2.INTER_AREA)
    return data


def get_n_value(lst):
    l = len(lst)
    return l // 50 + 1


def _safe_center_slices(files, n=50):
    if len(files) == 0:
        return []
    step = get_n_value(files)
    files = files[::step]
    mid = len(files) // 2
    half = n // 2
    start = max(0, mid - half)
    end = min(len(files), mid + half)
    return files[start:end]


def _sort_dicom_series(dcm_files):
    if not dcm_files:
        return dcm_files

    def _key(fp):
        try:
            ds = pydicom.dcmread(
                fp,
                stop_before_pixels=True,
                specific_tags=[
                    "InstanceNumber",
                    "ImagePositionPatient",
                    "SliceLocation",
                ],
            )
            if getattr(ds, "InstanceNumber", None) is not None:
                return (0, int(ds.InstanceNumber))
            ipp = getattr(ds, "ImagePositionPatient", None)
            if ipp is not None and len(ipp) >= 3:
                return (1, float(ipp[2]))
            sl = getattr(ds, "SliceLocation", None)
            if sl is not None:
                return (2, float(sl))
        except Exception:
            pass
        base = os.path.basename(fp)
        digits = "".join([c if c.isdigit() else " " for c in base]).split()
        num = int(digits[-1]) if digits else 0
        return (3, num)

    return sorted(dcm_files, key=_key)


def load_3d_dicom_images(scan_id, split="train"):
    flair = _sort_dicom_series(glob.glob(f"{path}/{split}/{scan_id}/FLAIR/*.dcm"))
    t1w = _sort_dicom_series(glob.glob(f"{path}/{split}/{scan_id}/T1w/*.dcm"))
    t1wce = _sort_dicom_series(glob.glob(f"{path}/{split}/{scan_id}/T1wCE/*.dcm"))
    t2w = _sort_dicom_series(glob.glob(f"{path}/{split}/{scan_id}/T2w/*.dcm"))

    flair = _safe_center_slices(flair, n=50)
    t1w = _safe_center_slices(t1w, n=50)
    t1wce = _safe_center_slices(t1wce, n=50)
    t2w = _safe_center_slices(t2w, n=50)

    def stack_and_pad(file_list):
        arr = np.array([dicom2array(a) for a in file_list], dtype=np.uint8)  # (D,H,W)
        if arr.size == 0:
            arr = np.zeros((0, img_size, img_size), dtype=np.uint8)
        arr = arr.transpose(1, 2, 0)  # (H,W,D)
        if arr.shape[-1] < 50:
            n_zero = 50 - arr.shape[-1]
            arr = np.concatenate(
                (arr, np.zeros((img_size, img_size, n_zero), dtype=np.uint8)), axis=-1
            )
        elif arr.shape[-1] > 50:
            arr = arr[..., :50]
        return arr

    flair_img = stack_and_pad(flair)
    t1w_img = stack_and_pad(t1w)
    t1wce_img = stack_and_pad(t1wce)
    t2w_img = stack_and_pad(t2w)

    return np.concatenate((flair_img, t1w_img, t1wce_img, t2w_img), axis=-1)




## === cell 4
class BrainTumor(Dataset):
    def __init__(
        self,
        path="/kaggle/input/rsna-miccai-brain-tumor-radiogenomic-classification",
        split="train",
        validation_split=0.2,
    ):
        self.path = path

        self.labels = {}
        if split in ("train", "val"):
            label_df = pd.read_csv(os.path.join(path, "train_labels.csv"))
            for b, m in zip(
                label_df["BraTS21ID"].astype(str).str.zfill(5),
                label_df["MGMT_value"].astype(float),
            ):
                self.labels[b] = float(m)

        if split == "val":
            self.split = "train"
            self.ids = [
                a.split("/")[-1] for a in sorted(glob.glob(path + f"/{self.split}/*"))
            ]
            self.ids = self.ids[-int(len(self.ids) * validation_split) :]
        elif split == "train":
            self.split = split
            self.ids = [
                a.split("/")[-1] for a in sorted(glob.glob(path + f"/{self.split}/*"))
            ]
            self.ids = self.ids[: -int(len(self.ids) * validation_split)]
        else:
            self.split = split
            self.ids = [
                a.split("/")[-1] for a in sorted(glob.glob(path + f"/{self.split}/*"))
            ]

    def __len__(self):
        return len(self.ids)

    def get_transforms(self):
        return A.Compose(
            [
                A.OneOf([A.HorizontalFlip(), A.VerticalFlip()], p=0.3),
                A.OneOf(
                    [
                        A.ShiftScaleRotate(
                            shift_limit=0.15, border_mode=cv2.BORDER_CONSTANT, value=0
                        ),
                        A.ShiftScaleRotate(
                            scale_limit=0.15, border_mode=cv2.BORDER_CONSTANT, value=0
                        ),
                        A.ShiftScaleRotate(
                            rotate_limit=15, border_mode=cv2.BORDER_CONSTANT, value=0
                        ),
                        A.IAAAffine(shear=(-15, 15), cval=0, mode="constant"),
                    ],
                    p=0.3,
                ),
                A.OneOf([A.GaussianBlur(), A.GaussNoise()], p=0.3),
                A.Normalize(mean=(0.0,) * 200, std=(1.0,) * 200),
                ToTensorV2(),
            ]
        )

    def __getitem__(self, idx):
        imgs = load_3d_dicom_images(self.ids[idx], self.split)  # (H,W,200) uint8
        x = torch.tensor(imgs, dtype=torch.float32)

        if self.split != "test":
            label = float(self.labels[self.ids[idx]])
            return x, torch.tensor(label, dtype=torch.float32)
        else:
            return x




## === cell 5
test_dir = os.path.join(path, "test")
test_ids = sorted(
    [
        os.path.basename(p)
        for p in glob.glob(os.path.join(test_dir, "*"))
        if os.path.isdir(p)
    ]
)

submission = pd.DataFrame(
    {"BraTS21ID": test_ids, "MGMT_value": np.zeros(len(test_ids), dtype=np.float32)}
)

test_dataset = BrainTumor(split="test", validation_split=0.2)
test_dataset.ids = test_ids

test_loader = DataLoader(
    test_dataset, batch_size=4, shuffle=False, num_workers=2, pin_memory=True
)

len(test_dataset), submission.shape



## === cell 6
gpu = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

model = resnet3d(50, n_input_channels=1, n_classes=1)

weights_path = "/kaggle/input/combined-sequences/resnet50.pt"

has_usable_weights = False
if os.path.exists(weights_path):
    try:
        state = torch.load(weights_path, map_location="cpu")
        if isinstance(state, dict) and all(isinstance(k, str) for k in state.keys()):
            model.load_state_dict(state, strict=False)
            has_usable_weights = True
        elif isinstance(state, nn.Module):
            model = state
            has_usable_weights = True
    except Exception:
        has_usable_weights = False

model.to(gpu)
model.eval()

label_df = pd.read_csv(os.path.join(path, "train_labels.csv"))
label_df["BraTS21ID"] = label_df["BraTS21ID"].astype(str).str.zfill(5)
bad_ids = set(["00109", "00123", "00709"])
label_df = label_df[~label_df["BraTS21ID"].isin(bad_ids)].copy()
prior_p = float(label_df["MGMT_value"].mean())
prior_p = min(max(prior_p, 1e-4), 1.0 - 1e-4)

has_usable_weights, prior_p



## === cell 7
y_pred = []

with torch.no_grad():
    for _, x in tqdm(enumerate(test_loader), total=len(test_loader)):
        if has_usable_weights:
            x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()
            x = x.to(gpu, non_blocking=True)
            output = model(x).detach().sigmoid().cpu().numpy().reshape(-1).tolist()
        else:
            bs = x.shape[0]
            output = [prior_p] * bs
        y_pred.extend(output)

len(y_pred), len(submission)



## === cell 8
if len(y_pred) != len(submission):
    raise RuntimeError(
        f"Pred length {len(y_pred)} != submission length {len(submission)}; cannot write submission."
    )

submission["MGMT_value"] = np.asarray(y_pred, dtype=np.float32)
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
submission.to_csv("submission.csv", index=False)

submission.head()
