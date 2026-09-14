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
Classify each cassava image into four disease categories or a fifth category indicating a healthy leaf.

## Metric
Categorization accuracy.

## Submission Format
```
image_id,label
1000471002.jpg,4
1000840542.jpg,4
etc.
```

## Dataset
**[train/test]_images** the image files.

**train.csv**

- `image_id` the image file name.

- `label` the ID code for the disease.

**sample_submission.csv** A properly formatted sample submission, given the disclosed test set content.

- `image_id` the image file name.

- `label` the predicted ID code for the disease.

**[train/test]_tfrecords** the image files in tfrecord format.

**label_num_to_disease_map.json** The mapping between each disease code and the real disease name.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
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
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        input/
            description.md (124 lines)
            label_num_to_disease_map.json (1 lines)
            sample_submission.csv (2677 lines)
            sample_submission.csv.zip (13.4 kB)
            test.zip (160 Bytes)
            test_images.zip (319.5 MB)
            test_tfrecords.zip (451.9 MB)
            train.csv (18722 lines)
            train.csv.zip (100.0 kB)
            train.zip (162 Bytes)
            train_images.zip (2.2 GB)
            train_tfrecords.zip (3.2 GB)
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
            test_images/
                2574872277.jpg (183.5 kB)
                1449210447.jpg (100.8 kB)
                ... and 2674 other files
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
            test_tfrecords/
                ld_test00-1338.tfrec (225.9 MB)
                ld_test01-1338.tfrec (226.2 MB)
            train_images/
                478676678.jpg (90.6 kB)
                2315755156.jpg (59.5 kB)
                ... and 18719 other files
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
            train_tfrecords/
                ld_train00-1338.tfrec (227.2 MB)
                ld_train01-1338.tfrec (227.0 MB)
                ... and 12 other files
        working/
            cassava-leaf-disease-classification/
                description.md (124 lines)
                label_num_to_disease_map.json (1 lines)
                ... and 10 other files
                cassava-leaf-disease-classification/
                test_images/
                    2574872277.jpg (183.5 kB)
                    1449210447.jpg (100.8 kB)
                    ... and 2674 other files
                    test_images/
                test_tfrecords/
                    ld_test00-1338.tfrec (225.9 MB)
                    ld_test01-1338.tfrec (226.2 MB)
                train_images/
                    478676678.jpg (90.6 kB)
                    2315755156.jpg (59.5 kB)
                    ... and 18719 other files
                    train_images/
                train_tfrecords/
                    ld_train00-1338.tfrec (227.2 MB)
                    ld_train01-1338.tfrec (227.0 MB)
                    ... and 12 other files
```

-> data/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/cassava-leaf-disease-classification/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/cassava-leaf-disease-classification/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> data/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> data/sample_submission.csv has 2676 rows and 2 columns.
The columns are: image_id, label

-> data/train.csv has 18721 rows and 2 columns.
The columns are: image_id, label

-> input/cassava-leaf-disease-classification/label_num_to_disease_map.json has auto-generated json schema:
{
  "$schema": "http://json-schema.org/schema#",
  "type": "object",
  "properties": {
    "0": {
      "type": "string"
    },
    "1": {
      "type": "string"
    },
    "2": {
      "type": "string"
    },
    "3": {
      "type": "string"
    },
    "4": {
      "type": "string"
    }
  },
  "required": [
    "0",
    "1",
    "2",
    "3",
    "4"
  ]
}

-> (stopped after 10 files for performance)

# 5. Target score

0.8833484436385615

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.14312) has done: 'The changes lower the total training workload by reducing the cross‑validation folds to 1 and the number of epochs to 5, which cuts the number of forward/backward passes dramatically while keeping the same model architecture, loss, optimizer, and data handling. This preserves the original training logic and evaluation semantics, but finishes well within the 600 s limit.'
- What this solution (achieved 0.05643) has done: 'Implemented fixes to enable successful training, validation handling, and proper checkpoint loading:

1. Skipped the validation step when the validation dataloader is `None` (K_FOLD = 1) to prevent the `NoneType` length error.  
2. Guarded against division‑by‑zero when no validation samples are processed.  
3. Corrected the checkpoint file pattern in the inference cell so saved models are actually loaded (`resnext50_32x4d_kfold_*.pkl`).  

These changes let the script run end‑to‑end and produce a valid `submission.csv`, while preserving the original model architecture and training logic.'
- What this solution (achieved 0.6364) has done: 'The fixes address the `persistent_workers` requirement by disabling it when `num_workers` is zero, and correct the metric list initialization typo (`ACCMeter` vs `ACCMeter`). These changes let the script run end‑to‑end and produce a proper `submission.csv` while preserving the original model and training logic.'

# 9. Code solution

## === cell 0
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
import os
from PIL import Image


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_root, "*.jpg"))
        if not return_name:
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x])
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, self.image_paths[x].split("/")[-1]
        else:
            label = self.label.loc[self.image_paths[x].split("/")[-1]].label
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 1
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold

K_FOLD = 1  # use a single fold for speed
BATCH_SIZE = 32
NUM_WORKERS = 0
LR = 1e-3
WD = 1e-4
CUTMIX_PROB = 0.0
BETA = 0.0
TRAINING = True  # run a tiny training loop (1 epoch) to create a checkpoint
EPOCH = 5  # increased epochs to improve accuracy
PHASE = ["train", "val"]

train_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.RandomHorizontalFlip(),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transform.Compose(
    [
        transform.Resize((448, 448)),
        transform.ToTensor(),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True)

    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=False,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=False,
                ),
            }
        )
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=NUM_WORKERS,
                pin_memory=True,
                persistent_workers=False,
            ),
            "val": None,
        }
    )




## === cell 2
import torch
import torch.nn as nn
import os

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnext50_32x4d",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
]

model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ["downsample"]

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.0)) * groups
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x9(
            width, width, stride, groups, dilation
        )  # typo fixed to conv3x3
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
    ):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer

        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        ckpt_path = "../input/resnext50-32x4d/resnext50_32x4d.pth"
        if os.path.exists(ckpt_path):
            state_dict = torch.load(ckpt_path, map_location="cpu")
            for name, param in state_dict.items():
                if name in model.state_dict():
                    try:
                        model.state_dict()[name].copy_(param)
                    except Exception:
                        pass
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )




## === cell 3
torch.backends.cudnn.benchmark = True
torch.backends.cudnn.allow_tf32 = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.set_float32_matmul_precision("high")

if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)


def create_new_model():
    return resnext50_32x4d(num_classes=5, pretrained=False).to(device)




## === cell 4
def create_loss_opti(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    global scaler
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, lr_scheduler




## === cell 5
import numpy as np


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w = int(W * cut_rat)
    cut_h = int(H * cut_rat)

    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2




## === cell 6
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.acc = acc
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = value
        if self.acc:
            self.sum += value
        else:
            self.sum += value * batch
        self.count += batch
        self.avg = self.sum / self.count




## === cell 7
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device)
    b_label = label.to(device)

    r = np.random.rand(1)
    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0]).to(device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )
        with torch.cuda.amp.autocast():
            output = model(b_image)
            loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
                1.0 - lam
            )
    else:
        with torch.cuda.amp.autocast():
            output = model(b_image)
            loss = criterion(output, b_label)

    _, predicted = torch.max(output, dim=1)
    correct = (predicted == b_label).sum().item()

    if phase == "train":
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

    return correct, loss.item()




## === cell 8
import numpy as np
from tqdm import tqdm

ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model()
        criterion, optimizer, lr_scheduler = create_loss_opti(model)
        Best_ACC = 0.0
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)
        for epoch in range(1, EPOCH + 1):
            correct_t = 0
            total = 0
            loss_t = 0.0
            for phase in PHASE:
                if dataloader[phase] is None:
                    continue

                if phase == "train":
                    model.train(True)
                    all_train_dataset.set_transform(train_transform)
                else:
                    model.train(False)
                    all_train_dataset.set_transform(val_transform)

                for image, label in tqdm(
                    dataloader[phase],
                    total=len(dataloader[phase]),
                    position=0,
                    leave=False,
                ):
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase
                    )
                    total += label.size(0)
                    loss_t += loss * label.size(0)
                    correct_t += correct

            if total == 0:
                total = 1

            tmp_ACCMeter.update(correct_t, total)
            tmp_LOSSMeter.update(loss_t, total)

            if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                Best_ACC = tmp_ACCMeter.avg
                ACCMeter[index] = tmp_ACCMeter
                LOSSMeter[index] = tmp_LOSSMeter

            lr_scheduler.step()
            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1, K_FOLD, epoch, EPOCH, loss_t / total, correct_t / total
                )
            )
        torch.save(
            model.state_dict(),
            f"./resnext50_32x4d_kfold_{index+1}_final.pkl",
        )




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/3930326473.py in <cell line: 0>()
     10 if TRAINING:
     11     for index, dataloader in enumerate(fold_dataloader):
---> 12         model = create_new_model()
     13         criterion, optimizer, lr_scheduler = create_loss_opti(model)
     14         Best_ACC = 0.0

/tmp/ipykernel_55/800035312.py in create_new_model()
     12 
     13 def create_new_model():
---> 14     return resnext50_32x4d(num_classes=5, pretrained=False).to(device)
     15 
     16 

/tmp/ipykernel_55/3895259998.py in resnext50_32x4d(pretrained, progress, **kwargs)
    300     kwargs["groups"] = 32
    301     kwargs["width_per_group"] = 4
--> 302     return _resnet(
    303         "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    304     )

/tmp/ipykernel_55/3895259998.py in _resnet(arch, block, layers, pretrained, progress, **kwargs)
    275 
    276 def _resnet(arch, block, layers, pretrained, progress, **kwargs):
--> 277     model = ResNet(block, layers, **kwargs)
    278     if pretrained:
    279         ckpt_path = "../input/resnext50-32x4d/resnext50_32x4d.pth"

/tmp/ipykernel_55/3895259998.py in __init__(self, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
    185         self.relu = nn.ReLU(inplace=True)
    186         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
--> 187         self.layer1 = self._make_layer(block, 64, layers[0])
    188         self.layer2 = self._make_layer(
    189             block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]

/tmp/ipykernel_55/3895259998.py in _make_layer(self, block, planes, blocks, stride, dilate)
    227         layers = []
    228         layers.append(
--> 229             block(
    230                 self.inplanes,
    231                 planes,

/tmp/ipykernel_55/3895259998.py in __init__(self, inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
    118         self.conv1 = conv1x1(inplanes, width)
    119         self.bn1 = norm_layer(width)
--> 120         self.conv2 = conv3x9(
    121             width, width, stride, groups, dilation
    122         )  # typo fixed to conv3x3

NameError: name 'conv3x9' is not defined

## === cell 9
acc_sum = 0.0
loss_sum = 0.0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} LOSS : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))




## === cell 10
import pandas as pd
import glob
from torch.utils.data import DataLoader

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=train_transform,
    return_name=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=False,
)

ckpt_paths = glob.glob("resnext50_32x4d_kfold_*.pkl")
models = []
if ckpt_paths:
    for ckpt in ckpt_paths:
        m = create_new_model()
        state = torch.load(ckpt, map_location=device)
        m.load_state_dict(state)
        m.eval()
        models.append(m)
else:
    m = create_new_model()
    m.eval()
    models.append(m)

image_names = []
logits_list = []

for img_batch, name_batch in test_dataloader:
    b_img = img_batch.to(device)
    logits_ensemble = None
    for m in models:
        with torch.no_grad():
            out = m(b_img)  # (B, 5)
        if logits_ensemble is None:
            logits_ensemble = out.clone()
        else:
            logits_ensemble += out
    logits_list.append(logits_ensemble.cpu())
    image_names.extend(name_batch)

image_logits = torch.cat(logits_list, dim=0).numpy()

probs = np.exp(image_logits - np.max(image_logits, axis=1, keepdims=True))
probs = probs / probs.sum(axis=1, keepdims=True)
pred_labels = np.argmax(probs, axis=1)

df = pd.DataFrame({"image_id": image_names, "label": pred_labels.astype(int)})
print(df.head())
df.to_csv("/kaggle/working/submission.csv", index=False)

## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1684921980.py in <cell line: 0>()
     28         models.append(m)
     29 else:
---> 30     m = create_new_model()
     31     m.eval()
     32     models.append(m)

/tmp/ipykernel_55/800035312.py in create_new_model()
     12 
     13 def create_new_model():
---> 14     return resnext50_32x4d(num_classes=5, pretrained=False).to(device)
     15 
     16 

/tmp/ipykernel_55/3895259998.py in resnext50_32x4d(pretrained, progress, **kwargs)
    300     kwargs["groups"] = 32
    301     kwargs["width_per_group"] = 4
--> 302     return _resnet(
    303         "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    304     )

/tmp/ipykernel_55/3895259998.py in _resnet(arch, block, layers, pretrained, progress, **kwargs)
    275 
    276 def _resnet(arch, block, layers, pretrained, progress, **kwargs):
--> 277     model = ResNet(block, layers, **kwargs)
    278     if pretrained:
    279         ckpt_path = "../input/resnext50-32x4d/resnext50_32x4d.pth"

/tmp/ipykernel_55/3895259998.py in __init__(self, block, layers, num_classes, zero_init_residual, groups, width_per_group, replace_stride_with_dilation, norm_layer)
    185         self.relu = nn.ReLU(inplace=True)
    186         self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
--> 187         self.layer1 = self._make_layer(block, 64, layers[0])
    188         self.layer2 = self._make_layer(
    189             block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]

/tmp/ipykernel_55/3895259998.py in _make_layer(self, block, planes, blocks, stride, dilate)
    227         layers = []
    228         layers.append(
--> 229             block(
    230                 self.inplanes,
    231                 planes,

/tmp/ipykernel_55/3895259998.py in __init__(self, inplanes, planes, stride, downsample, groups, base_width, dilation, norm_layer)
    118         self.conv1 = conv1x1(inplanes, width)
    119         self.bn1 = norm_layer(width)
--> 120         self.conv2 = conv3x9(
    121             width, width, stride, groups, dilation
    122         )  # typo fixed to conv3x3

NameError: name 'conv3x9' is not defined
