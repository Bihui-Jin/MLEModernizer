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

0.5

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

- What this solution (achieved 0.5) has done: 'I fix the dataset split/label lookup bug causing the `KeyError: 'train'` by ensuring the dataset uses the requested logical split (`train`/`val`/`test`) for returning labels, while still reading images from the correct on-disk folder. Then I make the test DataLoader safe by removing the invalid `prefetch_factor=None` usage and ensuring `head` is always defined (even if calibration cannot be fit). Finally, I build the submission directly from the official `sample_submission.csv` to guarantee the exact expected row count/order and avoid the “same number of rows” mismatch error, while keeping the modeling logic unchanged.'

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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


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

from functools import lru_cache


@lru_cache(maxsize=200000)
def _dicom2array_cached(
    dcm_path, voi_lut=True, fix_monochrome=True, _img_size=img_size
):
    dicom = pydicom.dcmread(dcm_path, force=True)
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
    data = cv2.resize(data, (_img_size, _img_size), interpolation=cv2.INTER_AREA)
    return data


def dicom2array(dcm_path, voi_lut=True, fix_monochrome=True):
    return _dicom2array_cached(dcm_path, voi_lut, fix_monochrome)


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

    def _fast_num(fp):
        base = os.path.basename(fp)
        i = base.rfind("-")
        j = base.rfind(".")
        if i != -1 and j != -1 and i < j:
            num_str = base[i + 1 : j]
            if num_str.isdigit():
                return int(num_str)
        return None

    nums = []
    ok = True
    for fp in dcm_files:
        n = _fast_num(fp)
        if n is None:
            ok = False
            break
        nums.append(n)

    if ok:
        order = np.argsort(np.asarray(nums, dtype=np.int32), kind="mergesort")
        return [dcm_files[i] for i in order]

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
                force=True,
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


def _list_dcm(dir_path):
    try:
        with os.scandir(dir_path) as it:
            return [e.path for e in it if e.is_file() and e.name.endswith(".dcm")]
    except FileNotFoundError:
        return []


def load_3d_dicom_images(scan_id, split="train"):
    base = f"{path}/{split}/{scan_id}"
    flair = _sort_dicom_series(_list_dcm(f"{base}/FLAIR"))
    t1w = _sort_dicom_series(_list_dcm(f"{base}/T1w"))
    t1wce = _sort_dicom_series(_list_dcm(f"{base}/T1wCE"))
    t2w = _sort_dicom_series(_list_dcm(f"{base}/T2w"))

    flair = _safe_center_slices(flair, n=50)
    t1w = _safe_center_slices(t1w, n=50)
    t1wce = _safe_center_slices(t1wce, n=50)
    t2w = _safe_center_slices(t2w, n=50)

    def stack_and_pad(file_list):
        out = np.zeros((img_size, img_size, 50), dtype=np.uint8)
        if not file_list:
            return out
        d = min(len(file_list), 50)
        arr = np.stack([dicom2array(a) for a in file_list[:d]], axis=0)  # (D,H,W)
        arr = arr.transpose(1, 2, 0)  # (H,W,D)
        out[..., : arr.shape[-1]] = arr
        return out

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
        self.split = split  # logical split requested by caller

        self.labels = {}
        if split in ("train", "val"):
            label_df = pd.read_csv(os.path.join(path, "train_labels.csv"))
            for b, m in zip(
                label_df["BraTS21ID"].astype(str).str.zfill(5),
                label_df["MGMT_value"].astype(float),
            ):
                self.labels[b] = float(m)

        bad_ids = set(["00109", "00123", "00709"])

        def _list_ids(dir_path):
            try:
                with os.scandir(dir_path) as it:
                    ids = [e.name for e in it if e.is_dir()]
                ids.sort()
                return ids
            except FileNotFoundError:
                return []

        def _split_ids(all_ids):
            n_total = len(all_ids)
            n_val = int(n_total * validation_split)
            n_val = max(1, n_val) if n_total > 1 else 0
            rng = np.random.RandomState(42)
            all_ids = list(all_ids)
            rng.shuffle(all_ids)
            return all_ids[: n_total - n_val], all_ids[n_total - n_val :]

        if split in ("train", "val"):
            all_ids = _list_ids(os.path.join(path, "train"))
            all_ids = [i for i in all_ids if i not in bad_ids]
            tr_ids, va_ids = _split_ids(all_ids)
            self.ids = tr_ids if split == "train" else va_ids
            self._disk_split = "train"
        else:
            self.ids = _list_ids(os.path.join(path, "test"))
            self._disk_split = "test"

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        global path
        _old = path
        path = self.path
        try:
            imgs = load_3d_dicom_images(
                self.ids[idx], self._disk_split
            )  # (H,W,200) uint8
        finally:
            path = _old

        x = torch.as_tensor(imgs, dtype=torch.float32) / 255.0  # (H,W,200) in [0,1]

        if self.split != "test":
            label = float(self.labels[self.ids[idx]])
            return x, torch.tensor(label, dtype=torch.float32)
        else:
            return x




## === cell 5
sample_sub_path = os.path.join(path, "sample_submission.csv")
submission = pd.read_csv(sample_sub_path)
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)

test_ids = submission["BraTS21ID"].tolist()

test_dataset = BrainTumor(path=path, split="test", validation_split=0.2)
test_dataset.ids = test_ids  # enforce same order as submission file

_num_workers = min(4, (os.cpu_count() or 2))
test_loader_kwargs = dict(
    batch_size=2,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    test_loader_kwargs["prefetch_factor"] = 4

test_loader = DataLoader(test_dataset, **test_loader_kwargs)

len(test_dataset), submission.shape



## === cell 6
from sklearn.linear_model import LogisticRegression

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

prior_logit = float(np.log(prior_p / (1.0 - prior_p)))

has_usable_weights, prior_p



## === cell 7
train_dataset = BrainTumor(path=path, split="train", validation_split=0.2)

_num_workers = min(4, (os.cpu_count() or 2))
train_loader_kwargs = dict(
    batch_size=2,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
)
if _num_workers > 0:
    train_loader_kwargs["prefetch_factor"] = 4

train_loader = DataLoader(train_dataset, **train_loader_kwargs)

X_train = np.empty((len(train_dataset), 1), dtype=np.float32)
y_train = np.empty((len(train_dataset),), dtype=np.int64)

model.eval()
with torch.no_grad():
    row = 0
    for x, y in tqdm(train_loader, total=len(train_loader)):
        x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (B,1,D,H,W)
        x = x.to(gpu, non_blocking=True)
        logits = model(x).detach().float().cpu().numpy().reshape(-1, 1)
        bs = logits.shape[0]
        X_train[row : row + bs] = logits
        y_train[row : row + bs] = (
            y.detach().cpu().numpy().reshape(-1).astype(np.int64, copy=False)
        )
        row += bs

X_train = np.clip(X_train, -20.0, 20.0)

feature_std = float(np.std(X_train))
can_fit_head = (
    np.isfinite(feature_std) and feature_std > 1e-8 and len(np.unique(y_train)) > 1
)

head = None
if can_fit_head:
    head = LogisticRegression(
        solver="lbfgs",
        max_iter=200,
        random_state=42,
    )
    head.fit(X_train, y_train)

can_fit_head, feature_std, X_train.shape



## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
KeyError                                  Traceback (most recent call last)
/tmp/ipykernel_55/724816810.py in <cell line: 0>()
     20 with torch.no_grad():
     21     row = 0
---> 22     for x, y in tqdm(train_loader, total=len(train_loader)):
     23         x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (B,1,D,H,W)
     24         x = x.to(gpu, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/tqdm/notebook.py in __iter__(self)
    248         try:
    249             it = super().__iter__()
--> 250             for obj in it:
    251                 # return super(tqdm...) will not catch exception
    252                 yield obj

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
   1453                 data = self._task_info.pop(self._rcvd_idx)[1]
   1454                 self._rcvd_idx += 1
-> 1455                 return self._process_data(data)
   1456 
   1457             assert not self._shutdown and self._tasks_outstanding > 0

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

KeyError: Caught KeyError in DataLoader worker process 3.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/2258104590.py", line 69, in __getitem__
    label = float(self.labels[self.ids[idx]])
                  ~~~~~~~~~~~^^^^^^^^^^^^^^^
KeyError: 'train'


## === cell 8
y_pred = np.empty((len(test_dataset),), dtype=np.float32)

model.eval()
with torch.no_grad():
    row = 0
    for x in tqdm(test_loader, total=len(test_loader)):
        x = x.permute(0, 3, 1, 2).unsqueeze(1).contiguous()  # (B,1,D,H,W)
        x = x.to(gpu, non_blocking=True)
        logits = model(x).detach().float().cpu().numpy().reshape(-1, 1)
        logits = np.clip(logits, -20.0, 20.0)

        if head is not None:
            output = head.predict_proba(logits)[:, 1].astype(np.float32, copy=False)
        else:
            if has_usable_weights:
                output = (1.0 / (1.0 + np.exp(-logits.reshape(-1)))).astype(
                    np.float32, copy=False
                )
            else:
                output = np.full((logits.shape[0],), prior_p, dtype=np.float32)

        bs = output.shape[0]
        y_pred[row : row + bs] = output
        row += bs

len(y_pred), len(submission)



## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1585725905.py in <cell line: 0>()
     10         logits = np.clip(logits, -20.0, 20.0)
     11 
---> 12         if head is not None:
     13             output = head.predict_proba(logits)[:, 1].astype(np.float32, copy=False)
     14         else:

NameError: name 'head' is not defined

## === cell 9
if len(y_pred) != len(submission):
    raise RuntimeError(
        f"Pred length {len(y_pred)} != submission length {len(submission)}; cannot write submission."
    )

submission["MGMT_value"] = np.clip(
    y_pred.astype(np.float32, copy=False), 1e-6, 1 - 1e-6
)
submission["BraTS21ID"] = submission["BraTS21ID"].astype(str).str.zfill(5)
submission.to_csv("submission.csv", index=False)

submission.head()
