# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


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

# 5. Code solution

## === cell 0
BATCH_SIZE = 16
EPOCH = 5
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 1.0

TRAINING = True

WEIGHT = "/kaggle/input/cutmix/resnext_kfold0_17_0.839"
K_FOLD = 1

BEST_WEIGHT_PATH = "/kaggle/working/best.pth"



## === cell 1
import os
import random
import numpy as np
import torch


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)



## === cell 2
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd

from torchvision.io import read_image, ImageReadMode


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))

        self.return_name = return_name
        self.label_map = None
        if not return_name:
            if label_path is None:
                raise ValueError("label_path must be provided when return_name=False")
            df = pd.read_csv(label_path)
            self.label_map = dict(
                zip(df["image_id"].tolist(), df["label"].astype(int).tolist())
            )

    def __getitem__(self, x):
        img_path = self.image_paths[x]
        img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8, CxHxW

        if self.transform is not None:
            img = self.transform(img)

        img_name = os.path.basename(img_path)
        if self.return_name:
            return img, img_name
        else:
            label = int(self.label_map[img_name])
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 3
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
import torchvision.transforms.v2 as T


train_transform = T.Compose(
    [
        T.Resize(
            (448, 448), interpolation=T.InterpolationMode.BILINEAR, antialias=True
        ),
        T.RandomHorizontalFlip(p=0.5),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

infer_transform = T.Compose(
    [
        T.Resize(
            (448, 448), interpolation=T.InterpolationMode.BILINEAR, antialias=True
        ),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)


def _make_loader(ds, shuffle):
    num_workers = min(8, (os.cpu_count() or 4))
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
    )


fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)
    index = 0
    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": _make_loader(train_dataset, shuffle=True),
                "val": _make_loader(val_dataset, shuffle=False),
            }
        )
        index += 1
    print(fold_dataloader)
else:
    fold_dataloader.append(
        {
            "train": _make_loader(all_train_dataset, shuffle=True),
            "val": None,
        }
    )



## === cell 4
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)



## === cell 5
import torch.nn as nn



## === cell 6
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
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
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
    load = []
    not_load = []

    if pretrained:
        try:
            import torchvision

            tv_arch = getattr(torchvision.models, arch)
            if arch == "resnext50_32x4d":
                from torchvision.models import ResNeXt50_32X4D_Weights

                tv_model = tv_arch(weights=ResNeXt50_32X4D_Weights.DEFAULT)
            elif arch == "resnext101_32x8d":
                from torchvision.models import ResNeXt101_32X8D_Weights

                tv_model = tv_arch(weights=ResNeXt101_32x8D_Weights.DEFAULT)
            else:
                tv_model = tv_arch(weights=None)

            state_dict = tv_model.state_dict()
        except Exception:
            state_dict = None

        if state_dict is not None:
            for name, param in state_dict.items():
                if name in model.state_dict():
                    try:
                        load.append(name)
                        model.state_dict()[name].copy_(param)
                    except Exception:
                        not_load.append(name)

    print("Load : {} layers".format(len(load)))
    print("Miss : {} layers".format(len(not_load)))
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )




## === cell 7
model = resnext50_32x4d(num_classes=5, pretrained=True).to(device)

if device.startswith("cuda"):
    model = model.to(memory_format=torch.channels_last)

try:
    if device.startswith("cuda"):
        model = torch.compile(model, mode="max-autotune")
except Exception as e:
    print("[WARN] torch.compile unavailable/failed, continuing uncompiled:", repr(e))



## === cell 8
model.eval()



## === cell 9
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 2, gamma=0.5, last_epoch=-1)




## === cell 10
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




## === cell 11
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.value = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, value, batch):
        self.value = float(value)
        self.sum += float(value) * batch
        self.count += batch
        self.avg = self.sum / self.count




## === cell 12
from tqdm import tqdm

val_max_acc = 0.0
ACCMeter = [
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
]
LOSSMeter = [
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
    AverageMeter(),
]

best_seen_loss = float("inf")

if TRAINING:
    for epoch in range(EPOCH):
        for index, dataloader in enumerate(fold_dataloader):
            for phase in PHASE:
                if dataloader[phase] is None:
                    continue

                correct = 0
                total = 0
                loss_t = 0.0

                if phase == "train":
                    model.train(True)
                else:
                    model.train(False)

                for image, label in tqdm(dataloader[phase], position=0, leave=False):
                    b_image = image.to(device, non_blocking=True)
                    if device.startswith("cuda"):
                        b_image = b_image.contiguous(memory_format=torch.channels_last)
                    b_label = label.to(device, non_blocking=True)

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
                            (bbx2 - bbx1)
                            * (bby2 - bby1)
                            / (b_image.size()[-1] * b_image.size()[-2])
                        )

                        output = model(b_image)
                        loss = criterion(output, target_a) * lam + criterion(
                            output, target_b
                        ) * (1.0 - lam)
                    else:
                        output = model(b_image)
                        loss = criterion(output, b_label)

                    _, predicted = torch.max(output.data, dim=1)
                    batch_correct = (predicted == b_label).sum().item()

                    if epoch == EPOCH - 1:
                        ACCMeter[index].update(batch_correct, b_label.size(0))
                        LOSSMeter[index].update(loss.item(), b_label.size(0))

                    total += b_label.size(0)
                    loss_t += loss.item() * b_label.size(0)

                    if phase == "train":
                        optimizer.zero_grad(set_to_none=True)
                        loss.backward()
                        optimizer.step()

                if phase == "train":
                    lr_scheduler.step()

                if phase == "train":
                    epoch_loss = loss_t / max(total, 1)
                    if epoch_loss < best_seen_loss:
                        best_seen_loss = epoch_loss
                        torch.save(model.state_dict(), BEST_WEIGHT_PATH)
                        print(
                            f"Saved best weights to {BEST_WEIGHT_PATH} (train loss {best_seen_loss:.6f})"
                        )

            if phase == "val":
                print(
                    "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                        index + 1,
                        K_FOLD,
                        epoch,
                        EPOCH,
                        loss_t / total,
                        batch_correct / total,
                    )
                )



## === cell 13
acc_sum = 0.0
loss_sum = 0.0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 14
model.eval()


def _find_candidate_weights(search_roots):
    exts = (".pth", ".pt", ".bin")
    candidates = []
    for search_root in search_roots:
        if not search_root or not os.path.exists(search_root):
            continue
        for root, _, files in os.walk(search_root):
            for fn in files:
                if fn.lower().endswith(exts):
                    candidates.append(os.path.join(root, fn))
    candidates = sorted(set(candidates), key=lambda p: os.path.getsize(p), reverse=True)
    return candidates


def _canonicalize_state_dict(sd: dict):
    if (
        isinstance(sd, dict)
        and "state_dict" in sd
        and isinstance(sd["state_dict"], dict)
    ):
        sd = sd["state_dict"]
    elif (
        isinstance(sd, dict)
        and "model_state_dict" in sd
        and isinstance(sd["model_state_dict"], dict)
    ):
        sd = sd["model_state_dict"]

    if not isinstance(sd, dict):
        return None

    new_sd = {}
    for k, v in sd.items():
        nk = k
        for prefix in ("module.", "model.", "net.", "backbone."):
            if nk.startswith(prefix):
                nk = nk[len(prefix) :]
        if nk.startswith("state_dict."):
            nk = nk[len("state_dict.") :]
        new_sd[nk] = v
    return new_sd


def _is_compatible_fc(model, sd):
    if sd is None:
        return False
    if "fc.weight" not in sd or "fc.bias" not in sd:
        return False
    try:
        return tuple(sd["fc.weight"].shape) == tuple(
            model.state_dict()["fc.weight"].shape
        )
    except Exception:
        return False


def _load_weights_strict_fc(model, weight_path):
    params = torch.load(weight_path, map_location="cpu")
    sd = _canonicalize_state_dict(params)
    if not _is_compatible_fc(model, sd):
        return 0, False

    loaded = 0
    msd = model.state_dict()
    for name, param in sd.items():
        if name in msd:
            try:
                if msd[name].shape == param.shape:
                    msd[name].copy_(param)
                    loaded += 1
            except Exception:
                pass
    return loaded, True


def _resolve_weight_path(weight_base):
    if weight_base is None:
        return None

    if os.path.exists(weight_base):
        return weight_base
    for ext in (".pth", ".pt", ".bin"):
        cand = weight_base + ext
        if os.path.exists(cand):
            return cand
    for suffix in ("_best", "-best", ".best"):
        for ext in (".pth", ".pt", ".bin"):
            cand = weight_base + suffix + ext
            if os.path.exists(cand):
                return cand

    if os.path.isdir(weight_base):
        files = []
        for fn in os.listdir(weight_base):
            if fn.lower().endswith((".pth", ".pt", ".bin")):
                files.append(os.path.join(weight_base, fn))
        files = sorted(files, key=lambda p: os.path.getsize(p), reverse=True)
        if files:
            return files[0]

    return None


def _resolve_weight_path_with_input_fallback(weight_base):
    p = _resolve_weight_path(weight_base)
    if p is not None:
        return p

    if weight_base is None:
        return None

    suffix = weight_base
    if suffix.startswith("/kaggle/input/"):
        suffix = suffix[len("/kaggle/input/") :]
        parts = suffix.split("/", 1)
        if len(parts) == 2:
            suffix = parts[1]  # drop dataset folder
        else:
            suffix = ""

    if not suffix:
        return None

    roots = []
    try:
        roots = [os.path.join("/kaggle/input", d) for d in os.listdir("/kaggle/input")]
    except Exception:
        roots = []

    for r in sorted(roots):
        cand = os.path.join(r, suffix)
        p2 = _resolve_weight_path(cand)
        if p2 is not None:
            return p2

    return None


loaded_any = False
resolved_weight = None

if TRAINING and os.path.exists(BEST_WEIGHT_PATH):
    resolved_weight = BEST_WEIGHT_PATH
    print(f"Using trained weights from: {resolved_weight}")
    loaded, ok = _load_weights_strict_fc(model, resolved_weight)
    if ok and loaded > 0:
        loaded_any = True
        print(f"Loaded {loaded} tensors from {resolved_weight}")
    else:
        print(
            f"[WARN] Could not load trained weights from {BEST_WEIGHT_PATH}; will fall back to WEIGHT search."
        )

if (not loaded_any) and (not TRAINING):
    resolved_weight = _resolve_weight_path_with_input_fallback(WEIGHT)

    if resolved_weight is None:
        likely_roots = []
        if WEIGHT is not None:
            likely_roots.append(os.path.dirname(WEIGHT))
        likely_roots.append("/kaggle/input")
        candidates = _find_candidate_weights(likely_roots)

        best = None
        for p in candidates[:400]:  # keep runtime safe
            try:
                params = torch.load(p, map_location="cpu")
                sd = _canonicalize_state_dict(params)
                if _is_compatible_fc(model, sd):
                    best = p
                    break
            except Exception:
                continue
        if best is not None:
            resolved_weight = best
            print(
                f"Configured weight missing; auto-selected compatible checkpoint: {resolved_weight}"
            )

    if resolved_weight is None:
        print(
            f"[WARN] No compatible 5-class cassava checkpoint found under /kaggle/input for WEIGHT='{WEIGHT}'."
        )
        print(
            "[WARN] Falling back to ImageNet-pretrained backbone weights with random 5-class head."
        )
        print(
            "[WARN] To reach the target score, provide a fine-tuned cassava checkpoint under /kaggle/input and set WEIGHT accordingly."
        )
    else:
        loaded, ok = _load_weights_strict_fc(model, resolved_weight)
        if not (ok and loaded > 0):
            print(
                f"[WARN] Found checkpoint but it was incompatible (missing/mismatched fc.*): {resolved_weight}"
            )
            print(
                "[WARN] Falling back to ImageNet-pretrained backbone weights with random 5-class head."
            )
        else:
            loaded_any = True
            print(f"Loaded {loaded} tensors from {resolved_weight}")
            print(
                "Model fc.weight shape after load:",
                tuple(model.state_dict()["fc.weight"].shape),
            )



## === cell 15
import pandas as pd
from torch.utils.data import Dataset

SAMPLE_PATH = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
TEST_ROOT = "/kaggle/input/cassava-leaf-disease-classification/test_images"

from torchvision.io import read_image, ImageReadMode


class CLD_TestDatasetByCSV(Dataset):
    def __init__(self, image_root, sample_csv_path, transform=None):
        self.image_root = image_root
        self.transform = transform
        sample = pd.read_csv(sample_csv_path)
        self.image_ids = sample["image_id"].tolist()

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        image_id = self.image_ids[idx]
        img_path = os.path.join(self.image_root, image_id)
        img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8, CxHxW
        if self.transform is not None:
            img = self.transform(img)
        return img, image_id


test_dataset = CLD_TestDatasetByCSV(TEST_ROOT, SAMPLE_PATH, transform=infer_transform)

_num_workers = min(8, (os.cpu_count() or 4))
test_dataloader = DataLoader(
    test_dataset,
    batch_size=32,
    shuffle=False,
    num_workers=_num_workers,
    pin_memory=True,
    persistent_workers=(_num_workers > 0),
    prefetch_factor=4 if _num_workers > 0 else None,
)

image_name = []
image_label = []

model.eval()
with torch.no_grad():
    for img, img_name in test_dataloader:
        b_img = img.to(device, non_blocking=True)
        if device.startswith("cuda"):
            b_img = b_img.contiguous(memory_format=torch.channels_last)
        output = model(b_img)
        _, predicted = torch.max(output, dim=1)

        image_name.extend(list(img_name))
        image_label.extend(predicted.detach().cpu().numpy().astype(int).tolist())

df = pd.DataFrame({"image_id": image_name, "label": image_label})

sample = pd.read_csv(SAMPLE_PATH)
df = sample[["image_id"]].merge(df, on="image_id", how="left")
if df["label"].isna().any():
    raise RuntimeError(
        "Some predictions are missing after merge; check image_id/path handling."
    )
df["label"] = df["label"].astype(int)

out_path = "/kaggle/working/submission.csv"
df.to_csv(out_path, index=False)
print("Wrote:", out_path, "rows:", len(df), "cols:", list(df.columns))
print(df.head())
