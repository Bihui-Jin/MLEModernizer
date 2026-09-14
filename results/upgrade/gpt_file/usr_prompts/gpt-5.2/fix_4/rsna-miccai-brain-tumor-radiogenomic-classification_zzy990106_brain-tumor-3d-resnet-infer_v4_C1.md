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

geopandas==0.14.4
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

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import sys
import os
import glob
import random
import time
import math
import numpy as np
import pandas as pd
import pydicom
import cv2
from tqdm import tqdm
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import roc_auc_score



## === cell 1
CFG = {
    "fold_num": 5,
    "seed": 42,
    "model": "resnet10",
    "epochs": 6,  # core training loop preserved
    "train_bs": 2,
    "valid_bs": 2,
    "lr": 1e-4,
    "num_workers": 2,
    "device": 0,
    "fp16": False,
}




## === cell 2
def seed_everything(seed):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(CFG["seed"])



## === cell 3
if torch.cuda.is_available():
    torch.cuda.set_device(CFG["device"])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)



## === cell 4
DATA_ROOT = "../input/rsna-miccai-brain-tumor-radiogenomic-classification"

train_df = pd.read_csv(os.path.join(DATA_ROOT, "train_labels.csv"))
sample_sub = pd.read_csv(os.path.join(DATA_ROOT, "sample_submission.csv"))

bad_ids = {109, 123, 709}
train_df["BraTS21ID_int"] = train_df["BraTS21ID"].astype(int)
train_df = train_df[~train_df["BraTS21ID_int"].isin(bad_ids)].reset_index(drop=True)

print("train rows:", len(train_df), "sample_submission rows:", len(sample_sub))
train_df.head()



## === cell 5


class MyDataset(Dataset):
    """
    Core logic preserved: build a (3, 64, 128, 128) volume from FLAIR/T1w/T1wCE,
    using linspace slice selection, min-max normalization, and cv2.resize.
    """

    def __init__(
        self,
        df,
        data_root=DATA_ROOT,
        split="test",
        with_target=False,
    ):
        self.df = df.reset_index(drop=True)
        self.data_root = data_root
        self.split = split
        self.with_target = with_target
        self.input_D = 64
        self.input_H = 128
        self.input_W = 128

        self.cache_dir = os.path.join("/kaggle/working", "vol_cache_d64_h128_w128_v1")
        os.makedirs(self.cache_dir, exist_ok=True)

        self._paths_cache = {}  # (id, series) -> list[str]

    def __len__(self):
        return len(self.df)

    @staticmethod
    def _sort_key(path):
        base = os.path.splitext(os.path.basename(path))[0]
        try:
            return int(base.split("-")[-1])
        except Exception:
            return base

    def _sorted_dicom_paths(self, series_dir, _id, series_name):
        key = (_id, series_name)
        if key in self._paths_cache:
            return self._paths_cache[key]
        t_paths = glob.glob(os.path.join(series_dir, "*.dcm"))
        t_paths = sorted(t_paths, key=self._sort_key)
        self._paths_cache[key] = t_paths
        return t_paths

    def load_dicom(self, path):
        dicom = pydicom.dcmread(path, stop_before_pixels=True)
        dicom.file_meta  # ensure file_meta is present for some decoders
        dicom = pydicom.dcmread(path)
        data = dicom.pixel_array.astype(np.float32)
        data = data - np.min(data)
        mx = np.max(data)
        if mx != 0:
            data = data / mx
        return data

    def _cache_path(self, _id):
        return os.path.join(self.cache_dir, f"{self.split}_{str(_id).zfill(5)}.npy")

    def _build_volume(self, _id):
        patient_path = os.path.join(self.data_root, self.split, str(_id).zfill(5))

        channels = []
        for t in ("FLAIR", "T1w", "T1wCE"):
            series_dir = os.path.join(patient_path, t)
            t_paths = self._sorted_dicom_paths(series_dir, _id, t)

            length = len(t_paths)
            if length == 0:
                channels.append(
                    [
                        np.zeros((self.input_H, self.input_W), np.float32)
                        for _ in range(self.input_D)
                    ]
                )
                continue

            idxs = np.linspace(0, length - 1, self.input_D).astype(int)

            channel = []
            for i in idxs:
                img2d = self.load_dicom(t_paths[int(i)])
                img2d = cv2.resize(img2d, (self.input_W, self.input_H)).astype(
                    np.float32
                )
                channel.append(img2d)
            channels.append(channel)

        image = np.array(channels, dtype=np.float32)  # (3, 64, 128, 128)
        return image

    def __getitem__(self, index):
        _id = int(self.df["BraTS21ID"].values[index])
        cpath = self._cache_path(_id)

        if os.path.exists(cpath):
            image = np.load(cpath)  # float32 exact cached volume
        else:
            image = self._build_volume(_id)
            tmp = cpath + f".tmp_{os.getpid()}"
            np.save(tmp, image)
            os.replace(tmp, cpath)

        if self.with_target:
            y = float(self.df["MGMT_value"].values[index])
            return image, torch.tensor(y, dtype=torch.float32), _id
        return image, _id




## === cell 6
x0, id0 = MyDataset(sample_sub, split="test", with_target=False)[0]
print("example id:", id0, "shape:", x0.shape, "dtype:", x0.dtype)



## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1488265000.py in <cell line: 0>()
----> 1 x0, id0 = MyDataset(sample_sub, split="test", with_target=False)[0]
      2 print("example id:", id0, "shape:", x0.shape, "dtype:", x0.dtype)
      3 

/tmp/ipykernel_55/111605171.py in __getitem__(self, index)
    116             tmp = cpath + f".tmp_{os.getpid()}"
    117             np.save(tmp, image)
--> 118             os.replace(tmp, cpath)
    119 
    120         if self.with_target:

FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_d64_h128_w128_v1/test_00356.npy.tmp_55' -> '/kaggle/working/vol_cache_d64_h128_w128_v1/test_00356.npy'

## === cell 7
from torch.autograd import Variable
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1, dilation=1):
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        dilation=dilation,
        stride=stride,
        padding=dilation,
        bias=False,
    )


def downsample_basic_block(x, planes, stride, no_cuda=False):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3), out.size(4)
    ).zero_()
    if not no_cuda:
        if isinstance(out.data, torch.cuda.FloatTensor):
            zero_pads = zero_pads.cuda()
    out = Variable(torch.cat([out.data, zero_pads], dim=1))
    return out


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3x3(inplanes, planes, stride=stride, dilation=dilation)
        self.bn1 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3x3(planes, planes, dilation=dilation)
        self.bn2 = nn.BatchNorm3d(planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, dilation=1, downsample=None):
        super().__init__()
        self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(
            planes,
            planes,
            kernel_size=3,
            stride=stride,
            dilation=dilation,
            padding=dilation,
            bias=False,
        )
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out = self.relu(out + residual)
        return out


class ResNet3D(nn.Module):
    def __init__(self, block, layers, shortcut_type="B", num_class=2, no_cuda=False):
        super().__init__()
        self.inplanes = 64
        self.no_cuda = no_cuda

        self.conv1 = nn.Conv3d(
            3, 64, kernel_size=7, stride=(2, 2, 2), padding=(3, 3, 3), bias=False
        )
        self.bn1 = nn.BatchNorm3d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool3d(kernel_size=(3, 3, 3), stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], shortcut_type)
        self.layer2 = self._make_layer(
            block, 64 * 2, layers[1], shortcut_type, stride=2
        )
        self.layer3 = self._make_layer(
            block, 128 * 2, layers[2], shortcut_type, stride=1, dilation=2
        )
        self.layer4 = self._make_layer(
            block, 256 * 2, layers[3], shortcut_type, stride=1, dilation=4
        )

        self.fea_dim = 256 * 2 * block.expansion
        self.dropout = nn.Dropout(0.5)
        self.fc = nn.Sequential(nn.Linear(self.fea_dim, num_class, bias=True))

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode="fan_out")
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, shortcut_type, stride=1, dilation=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            if shortcut_type == "A":
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride,
                    no_cuda=self.no_cuda,
                )
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=stride,
                        bias=False,
                    ),
                    nn.BatchNorm3d(planes * block.expansion),
                )

        layers = [
            block(
                self.inplanes,
                planes,
                stride=stride,
                dilation=dilation,
                downsample=downsample,
            )
        ]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, dilation=dilation))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.maxpool(self.relu(self.bn1(self.conv1(x))))
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = F.adaptive_avg_pool3d(x, (1, 1, 1))
        emb_3d = x.view((-1, self.fea_dim))
        emb_3d = self.dropout(emb_3d)
        out = self.fc(emb_3d)
        return out


def resnet10(**kwargs):
    return ResNet3D(BasicBlock, [1, 1, 1, 1], **kwargs)




## === cell 8
def generate_model(opt):
    assert opt.model in ["resnet"]
    if opt.model == "resnet":
        assert opt.model_depth in [10]
        model = resnet10(shortcut_type=opt.resnet_shortcut, no_cuda=opt.no_cuda)
    model = model.to(device)
    return model, model.parameters()


class Config:
    n_seg_classes = 2
    input_D = 64
    input_H = 128
    input_W = 128
    pretrain_path = None
    new_layer_names = ["conv_seg"]
    no_cuda = not torch.cuda.is_available()
    model = "resnet"
    model_depth = 10
    resnet_shortcut = "B"




## === cell 9
nw = CFG["num_workers"]
if os.name == "nt":
    nw = 0

skf = StratifiedKFold(n_splits=CFG["fold_num"], shuffle=True, random_state=CFG["seed"])
train_df["fold"] = -1
for f, (_, va_idx) in enumerate(skf.split(train_df, train_df["MGMT_value"].values)):
    train_df.loc[va_idx, "fold"] = f

train_df[["BraTS21ID", "MGMT_value", "fold"]].head()




## === cell 10
def train_one_epoch(model, loader, optimizer, criterion):
    model.train()
    losses = []
    for img, y, _id in tqdm(loader, total=len(loader), leave=False):
        img = img.to(device, non_blocking=True)
        y = y.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(img)
        loss = criterion(logits[:, 1], y)
        loss.backward()
        optimizer.step()

        losses.append(loss.item())
    return float(np.mean(losses)) if losses else 0.0


@torch.no_grad()
def valid_one_epoch(model, loader):
    model.eval()
    ys = []
    ps = []
    for img, y, _id in tqdm(loader, total=len(loader), leave=False):
        img = img.to(device, non_blocking=True)
        y_np = y.numpy().astype(np.float32)
        prob = model(img).softmax(1)[:, 1].detach().cpu().numpy().astype(np.float32)
        ys.append(y_np)
        ps.append(prob)
    ys = np.concatenate(ys) if ys else np.array([], dtype=np.float32)
    ps = np.concatenate(ps) if ps else np.array([], dtype=np.float32)
    if len(np.unique(ys)) < 2:
        return np.nan
    return roc_auc_score(ys, ps)




## === cell 11
def make_loader(ds, batch_size, shuffle):
    kwargs = dict(
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )
    if nw > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
    return DataLoader(ds, **kwargs)


models = []
oof_auc = []

for fold in range(CFG["fold_num"]):
    trn = train_df[train_df["fold"] != fold].reset_index(drop=True)
    val = train_df[train_df["fold"] == fold].reset_index(drop=True)

    train_set = MyDataset(trn, split="train", with_target=True)
    valid_set = MyDataset(val, split="train", with_target=True)

    train_loader = make_loader(train_set, CFG["train_bs"], shuffle=True)
    valid_loader = make_loader(valid_set, CFG["valid_bs"], shuffle=False)

    model, _ = generate_model(Config())
    optimizer = torch.optim.Adam(model.parameters(), lr=CFG["lr"])
    criterion = nn.BCEWithLogitsLoss()

    best_auc = -1.0
    best_state = None

    for epoch in range(CFG["epochs"]):
        tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
        va_auc = valid_one_epoch(model, valid_loader)
        if not np.isnan(va_auc) and va_auc > best_auc:
            best_auc = float(va_auc)
            best_state = {
                k: v.detach().cpu().clone() for k, v in model.state_dict().items()
            }
        print(
            f"fold {fold} epoch {epoch+1}/{CFG['epochs']} loss {tr_loss:.4f} val_auc {va_auc}"
        )

    if best_state is not None:
        model.load_state_dict(best_state, strict=True)
    model.eval()
    models.append(model)
    oof_auc.append(best_auc)

print("CV AUC (per fold):", oof_auc, "mean:", np.nanmean(oof_auc))



## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/1810634882.py in <cell line: 0>()
     37 
     38     for epoch in range(CFG["epochs"]):
---> 39         tr_loss = train_one_epoch(model, train_loader, optimizer, criterion)
     40         va_auc = valid_one_epoch(model, valid_loader)
     41         if not np.isnan(va_auc) and va_auc > best_auc:

/tmp/ipykernel_55/2907826714.py in train_one_epoch(model, loader, optimizer, criterion)
      2     model.train()
      3     losses = []
----> 4     for img, y, _id in tqdm(loader, total=len(loader), leave=False):
      5         img = img.to(device, non_blocking=True)
      6         y = y.to(device, non_blocking=True)

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
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

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

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/111605171.py", line 118, in __getitem__
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_d64_h128_w128_v1/train_00716.npy.tmp_99' -> '/kaggle/working/vol_cache_d64_h128_w128_v1/train_00716.npy'


## === cell 12
test_set = MyDataset(sample_sub, split="test", with_target=False)
test_loader = make_loader(test_set, CFG["valid_bs"], shuffle=False)


@torch.no_grad()
def predict_ensemble(models, loader):
    all_pred = []
    all_ids = []
    for img, _id in tqdm(loader, total=len(loader)):
        img = img.to(device, non_blocking=True)
        preds = []
        for m in models:
            p = m(img).softmax(1)[:, 1].detach().cpu().numpy().astype(np.float32)
            preds.append(p)
        p_mean = np.mean(np.stack(preds, axis=0), axis=0)
        all_pred.append(p_mean)
        all_ids.extend([int(x) for x in _id])
    return np.concatenate(all_pred), all_ids


ensemble_pred, ids = predict_ensemble(models, test_loader)
print("preds:", ensemble_pred.shape, "ids:", len(ids), "expected:", len(sample_sub))



## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2401124861.py in <cell line: 0>()
     19 
     20 
---> 21 ensemble_pred, ids = predict_ensemble(models, test_loader)
     22 print("preds:", ensemble_pred.shape, "ids:", len(ids), "expected:", len(sample_sub))
     23 

/usr/local/lib/python3.11/dist-packages/torch/utils/_contextlib.py in decorate_context(*args, **kwargs)
    114     def decorate_context(*args, **kwargs):
    115         with ctx_factory():
--> 116             return func(*args, **kwargs)
    117 
    118     return decorate_context

/tmp/ipykernel_55/2401124861.py in predict_ensemble(models, loader)
      7     all_pred = []
      8     all_ids = []
----> 9     for img, _id in tqdm(loader, total=len(loader)):
     10         img = img.to(device, non_blocking=True)
     11         preds = []

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
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

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

FileNotFoundError: Caught FileNotFoundError in DataLoader worker process 0.
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
  File "/tmp/ipykernel_55/111605171.py", line 118, in __getitem__
    os.replace(tmp, cpath)
FileNotFoundError: [Errno 2] No such file or directory: '/kaggle/working/vol_cache_d64_h128_w128_v1/test_00356.npy.tmp_112' -> '/kaggle/working/vol_cache_d64_h128_w128_v1/test_00356.npy'


## === cell 13
submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": ensemble_pred})
submission["BraTS21ID"] = submission["BraTS21ID"].map(lambda x: str(int(x)).zfill(5))

submission = (
    submission.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"]).reset_index()
)

submission["MGMT_value"] = submission["MGMT_value"].astype(np.float32)
submission["MGMT_value"] = submission["MGMT_value"].fillna(0.5).clip(0.0, 1.0)

assert submission.shape[0] == sample_sub.shape[0]
assert list(submission.columns) == ["BraTS21ID", "MGMT_value"]

submission.to_csv("submission.csv", index=False)
print(submission.head())
print("Wrote submission.csv with rows:", len(submission))

## --- ERROR in cell 13, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/958403144.py in <cell line: 0>()
----> 1 submission = pd.DataFrame({"BraTS21ID": ids, "MGMT_value": ensemble_pred})
      2 submission["BraTS21ID"] = submission["BraTS21ID"].map(lambda x: str(int(x)).zfill(5))
      3 
      4 submission = (
      5     submission.set_index("BraTS21ID").reindex(sample_sub["BraTS21ID"]).reset_index()

NameError: name 'ids' is not defined
