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
BATCH_SIZE = 16  # keep as provided
EPOCH = 6
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 0.0
CUTMIX_PROB = 0.0

TRAINING = True

K_FOLD = 5

IMG_SIZE = 384

import os
import glob
import random
import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset
from sklearn.model_selection import KFold
from tqdm import tqdm


def seed_everything(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


seed_everything(42)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4) // 2)))




## === cell 1
try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TV_READ_IMAGE = True
except Exception:
    _HAS_TV_READ_IMAGE = False


_TRAIN_CSV_PATH = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
_train_df = pd.read_csv(_TRAIN_CSV_PATH, usecols=["image_id", "label"])
_LABEL_MAP_GLOBAL = dict(zip(_train_df["image_id"].values, _train_df["label"].values))


class CLD_Dataset(Dataset):
    def __init__(
        self,
        image_root,
        label_path=None,
        transform=None,
        return_name=False,
        cache_images=True,
        shared_cache=None,
        cache_transformed: bool = False,
        max_transformed_cache: int = 0,
        label_map=None,
    ):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.return_name = return_name

        self.label_map = None
        if not return_name:
            if label_map is not None:
                self.label_map = label_map
            elif label_path is not None:
                df = pd.read_csv(label_path, usecols=["image_id", "label"])
                self.label_map = dict(zip(df["image_id"].values, df["label"].values))

        self.cache_images = cache_images
        if cache_images:
            self._img_cache = shared_cache if shared_cache is not None else {}
        else:
            self._img_cache = None

        self.cache_transformed = cache_transformed
        self.max_transformed_cache = (
            int(max_transformed_cache) if max_transformed_cache else 0
        )
        self._tensor_cache = {} if cache_transformed else None
        self._tensor_cache_order = (
            [] if cache_transformed and self.max_transformed_cache > 0 else None
        )

    def set_transform(self, transform):
        self.transform = transform
        if self._tensor_cache is not None:
            self._tensor_cache.clear()
            if self._tensor_cache_order is not None:
                self._tensor_cache_order.clear()

    def _load_rgb_chw_uint8(self, idx: int) -> torch.Tensor:
        if self.cache_images:
            cached = self._img_cache.get(idx, None)
            if cached is not None:
                return cached

        path = self.image_paths[idx]
        if _HAS_TV_READ_IMAGE:
            img = read_image(path, mode=ImageReadMode.RGB)  # uint8 CHW
        else:
            with Image.open(path) as im:
                arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
            img = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # uint8 CHW

        if self.cache_images:
            self._img_cache[idx] = img
        return img

    def __getitem__(self, x):
        if self._tensor_cache is not None:
            cached_t = self._tensor_cache.get(x, None)
            if cached_t is not None:
                img_t = cached_t
            else:
                img = self._load_rgb_chw_uint8(x)
                img_t = self.transform(img) if self.transform is not None else img
                self._tensor_cache[x] = img_t
                if self._tensor_cache_order is not None:
                    self._tensor_cache_order.append(x)
                    if len(self._tensor_cache_order) > self.max_transformed_cache:
                        ev = self._tensor_cache_order.pop(0)
                        self._tensor_cache.pop(ev, None)
        else:
            img = self._load_rgb_chw_uint8(x)
            img_t = self.transform(img) if self.transform is not None else img

        if self.return_name:
            return img_t, os.path.basename(self.image_paths[x])
        else:
            label = self.label_map[os.path.basename(self.image_paths[x])]
            return img_t, label

    def __len__(self):
        return len(self.image_paths)




## === cell 2
import torchvision.transforms.v2 as T

train_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.RandomHorizontalFlip(),
        T.ToDtype(torch.float32, scale=True),  # uint8 -> float in [0,1]
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = T.Compose(
    [
        T.Resize((IMG_SIZE, IMG_SIZE), antialias=True),
        T.ToDtype(torch.float32, scale=True),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

_shared_train_cache = {}

train_dataset_full = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
    cache_images=True,
    shared_cache=_shared_train_cache,
    cache_transformed=False,
    label_map=_LABEL_MAP_GLOBAL,
)
val_dataset_full = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    val_transform,
    cache_images=True,
    shared_cache=_shared_train_cache,
    cache_transformed=True,
    max_transformed_cache=4096,
    label_map=_LABEL_MAP_GLOBAL,
)
dataset_size = len(train_dataset_full)


def _fast_collate(batch):
    imgs, labels = zip(*batch)
    return torch.stack(imgs, 0), torch.as_tensor(labels, dtype=torch.long)


def _make_loader(ds, shuffle: bool):
    cpu = os.cpu_count() or 4
    num_workers = min(8, max(2, cpu // 2))
    return DataLoader(
        ds,
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=(num_workers > 0),
        prefetch_factor=4 if num_workers > 0 else None,
        collate_fn=_fast_collate,
    )


fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)

    for train_idx, val_idx in kf.split(range(dataset_size)):
        train_dataset = torch.utils.data.Subset(train_dataset_full, train_idx)
        val_dataset = torch.utils.data.Subset(val_dataset_full, val_idx)
        fold_dataloader.append(
            {
                "train": _make_loader(train_dataset, shuffle=True),
                "val": _make_loader(val_dataset, shuffle=False),
            }
        )
    print(f"Prepared {len(fold_dataloader)} folds")
else:
    fold_dataloader.append(
        {
            "train": _make_loader(train_dataset_full, shuffle=True),
            "val": None,
        }
    )




## === cell 3
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)




## === cell 4
import torchvision




## === cell 5
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
        state_dict = None
        local_pretrain = "../input/resnext50-32x4d/resnext50_32x4d.pth"
        if os.path.exists(local_pretrain):
            state_dict = torch.load(local_pretrain, map_location="cpu")
        else:
            try:
                tv = torchvision.models.resnext50_32x4d(
                    weights=torchvision.models.ResNeXt50_32X4D_Weights.DEFAULT
                )
                state_dict = tv.state_dict()
            except Exception as e:
                print(
                    f"Warning: could not load torchvision pretrained weights ({e}); continuing without pretrained."
                )
                state_dict = None

        if state_dict is not None:
            msd = model.state_dict()
            for name, param in state_dict.items():
                if name in msd:
                    try:
                        msd[name].copy_(param)
                        load.append(name)
                    except Exception:
                        not_load.append(name)

    print("Load pretrain : ")
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
        "resnext_50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
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




## === cell 6
_COMPILED_OK = None
ENABLE_TORCH_COMPILE = False


def create_new_model():
    global _COMPILED_OK
    m = resnext50_32x4d(num_classes=5, pretrained=True).to(device)
    if device.startswith("cuda"):
        m = m.to(memory_format=torch.channels_last)
        if ENABLE_TORCH_COMPILE:
            if _COMPILED_OK is None:
                try:
                    m = torch.compile(m, mode="reduce-overhead")
                    _COMPILED_OK = True
                except Exception as e:
                    print(
                        f"Warning: torch.compile failed ({e}); continuing without compile."
                    )
                    _COMPILED_OK = False
            elif _COMPILED_OK:
                m = torch.compile(m, mode="reduce-overhead")
    return m




## === cell 7
model = create_new_model()




## === cell 8
def create_loss_opti(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 9
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




## === cell 10
class AverageMeter:
    """Computes and stores the average and current value"""

    def __init__(self, acc):
        self.reset()
        self.acc = acc

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




## === cell 11
def train_step(model, criterion, optimizer, image, label, phase):
    if device.startswith("cuda"):
        b_image = image.to(device, non_blocking=True).to(
            memory_format=torch.channels_last
        )
    else:
        b_image = image.to(device, non_blocking=True)
    b_label = label.to(device, non_blocking=True)

    if BETA > 0 and CUTMIX_PROB > 0.0:
        r = np.random.rand(1)
    else:
        r = 1.0

    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0], device=device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )

        output = model(b_image)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
    else:
        output = model(b_image)
        loss = criterion(output, b_label)

    predicted = output.argmax(dim=1)
    correct = (predicted == b_label).sum().item()

    if phase == "train":
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 12
max_acc = 0.0
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
            tmp_ACCMeter.reset()
            tmp_LOSSMeter.reset()

            correct_t = 0
            total = 0
            loss_t = 0.0

            for phase in PHASE:
                is_train = phase == "train"
                model.train(is_train)
                loader = dataloader[phase]

                if not is_train:
                    with torch.no_grad():
                        for image, label in loader:
                            correct, loss = train_step(
                                model, criterion, optimizer, image, label, phase
                            )
                            bs = label.size(0)
                            tmp_ACCMeter.update(correct, bs)
                            tmp_LOSSMeter.update(loss, bs)
                            total += bs
                            loss_t += loss * bs
                            correct_t += correct
                else:
                    for image, label in loader:
                        _ = train_step(model, criterion, optimizer, image, label, phase)

                if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                    Best_ACC = tmp_ACCMeter.avg
                    ACCMeter[index] = tmp_ACCMeter
                    LOSSMeter[index] = tmp_LOSSMeter
                    torch.save(
                        model.state_dict(),
                        "./resnext50_32x4d_kfold_{}_{}_{:.6f}.pkl".format(
                            index + 1, epoch, tmp_ACCMeter.avg
                        ),
                    )

            lr_scheduler.step()
            if total > 0:
                print(
                    "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                        index + 1,
                        K_FOLD,
                        epoch,
                        EPOCH,
                        loss_t / total,
                        correct_t / total,
                    )
                )
            else:
                print(
                    "Fold : {}/ {} Epoch : {} / {} (no val)".format(
                        index + 1, K_FOLD, epoch, EPOCH
                    )
                )




## === cell 13
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} LOSS : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))




## === cell 14
import re
import torchvision.io as tvio
import torchvision.transforms.v2.functional as F


class CLD_TestBytesDataset(Dataset):
    def __init__(self, image_root):
        super().__init__()
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        p = self.image_paths[idx]
        with open(p, "rb") as f:
            b = f.read()
        return b, os.path.basename(p)


def _collate_test_bytes(batch):
    bs, names = zip(*batch)
    return list(bs), list(names)


TEST_BATCH_SIZE = 64

cpu = os.cpu_count() or 4
num_workers = min(8, max(2, cpu // 2))
test_dataset = CLD_TestBytesDataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images"
)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    collate_fn=_collate_test_bytes,
)

_WEIGHT_RE = re.compile(r"resnext50_32x4d_kfold_(\d+)_(\d+)_([0-9.]+)\.pkl$")


def _pick_best_per_fold(paths):
    by_fold = {}
    for p in paths:
        base = os.path.basename(p)
        m = _WEIGHT_RE.match(base)
        if m is None:
            continue
        fold = int(m.group(1))
        acc = float(m.group(3))
        prev = by_fold.get(fold, None)
        if prev is None or acc > prev[0]:
            by_fold[fold] = (acc, p)
    return [v[1] for k, v in sorted(by_fold.items(), key=lambda kv: kv[0])]


weight_paths_all = sorted(glob.glob("/kaggle/working/resnext50_32x4d_kfold_*.pkl"))
weight_paths = _pick_best_per_fold(weight_paths_all)
if len(weight_paths) == 0:
    weight_paths = sorted(glob.glob("../input/cassave-resnext50-30x4d/*.pkl"))

if len(weight_paths) == 0:
    print("Warning: No trained .pkl weights found in /kaggle/working or ../input.")
    print(
        "Falling back to a single model with (torchvision/local) pretrained backbone weights."
    )
    weight_paths = [None]
else:
    print(
        f"Using {len(weight_paths)} weight files for ensembling (best-per-fold if available)."
    )

n_test = len(test_dataset)
sum_probs = np.zeros((n_test, 5), dtype=np.float64)
image_names_ref = None
n_models = 0

mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

_test_uint8_cache = [None] * n_test
_test_names = [None] * n_test

write_pos = 0
for jpeg_bytes_list, img_name in test_dataloader:
    imgs = [
        tvio.decode_jpeg(
            torch.frombuffer(b, dtype=torch.uint8), mode=tvio.ImageReadMode.RGB
        )
        for b in jpeg_bytes_list
    ]
    imgs = [F.resize(im, (IMG_SIZE, IMG_SIZE), antialias=True) for im in imgs]  # uint8
    for j, (im, nm) in enumerate(zip(imgs, img_name)):
        _test_uint8_cache[write_pos + j] = im  # CHW uint8
        _test_names[write_pos + j] = nm
    write_pos += len(imgs)

image_names_ref = _test_names


class _CachedTestDataset(Dataset):
    def __init__(self, imgs_uint8, names):
        self.imgs_uint8 = imgs_uint8
        self.names = names

    def __len__(self):
        return len(self.imgs_uint8)

    def __getitem__(self, idx):
        return self.imgs_uint8[idx], self.names[idx]


def _collate_cached(batch):
    imgs, names = zip(*batch)
    return torch.stack(imgs, 0), list(names)


cached_test_dataset = _CachedTestDataset(_test_uint8_cache, _test_names)
cached_test_loader = DataLoader(
    cached_test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    num_workers=0,  # already decoded/resized; keep 0 to avoid IPC overhead
    pin_memory=True,
    collate_fn=_collate_cached,
)

for params_path in weight_paths:
    model = create_new_model()
    if params_path is not None:
        params = torch.load(params_path, map_location="cpu")
        load = []
        not_load = []
        msd = model.state_dict()
        for name, param in params.items():
            if name in msd:
                try:
                    load.append(name)
                    msd[name].copy_(param)
                except Exception:
                    not_load.append(name)
        print(f"Loaded weights: {os.path.basename(params_path)}")
        print("Trained weight load : {}".format(len(load)))
        print("Trained weight not load : {}".format(len(not_load)))
    else:
        print("No trained weights loaded for this fold (fallback).")

    model.eval()

    write_pos = 0
    with torch.inference_mode():
        for batch_uint8, _names in cached_test_loader:
            if device.startswith("cuda"):
                batch = batch_uint8.to(device, non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            else:
                batch = batch_uint8.to(device, non_blocking=True)

            batch = batch.to(torch.float32).div_(255.0)
            batch = (batch - mean) / std

            output = model(batch)  # [B, 5]
            prob = torch.softmax(output, dim=1).to("cpu").numpy()  # [B,5]
            bs = prob.shape[0]
            sum_probs[write_pos : write_pos + bs] += prob.astype(np.float64, copy=False)
            write_pos += bs

    n_models += 1

avg_probs = sum_probs / float(n_models)
image_labels = np.argmax(avg_probs, axis=1).astype(np.int64)

df = pd.DataFrame({"image_id": image_names_ref, "label": image_labels})

sample = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
df = sample[["image_id"]].merge(df, on="image_id", how="left")
df["label"] = df["label"].fillna(0).astype(int)

print(df.head())
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", df.shape)
