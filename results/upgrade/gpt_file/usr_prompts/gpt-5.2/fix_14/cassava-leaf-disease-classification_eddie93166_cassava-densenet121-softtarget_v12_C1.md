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
BATCH_SIZE = 8
EPOCH = 10
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 1.0
TRAINING = False
K_FOLD = 5

INFER_BATCH_SIZE = 64

MAX_ENSEMBLE_WEIGHTS = 5



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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


seed_everything(42)



## === cell 2
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
from PIL import Image

try:
    from torchvision.io import read_image, ImageReadMode

    _HAS_TV_READ_IMAGE = True
except Exception:
    _HAS_TV_READ_IMAGE = False


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.return_name = return_name

        self.label_map = None
        if (not return_name) and (label_path is not None):
            df = pd.read_csv(label_path)
            self.label_map = dict(
                zip(df["image_id"].values, df["label"].astype(int).values)
            )

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        path = self.image_paths[x]

        img = None
        if _HAS_TV_READ_IMAGE and self.transform is not None:
            try:
                img = read_image(path, mode=ImageReadMode.RGB)  # uint8, (C,H,W)
            except Exception:
                img = None

        if img is None:
            img = Image.open(path).convert("RGB")

        if self.transform is not None:
            img = self.transform(img)

        name = os.path.basename(path)
        if self.return_name:
            return img, name
        else:
            label = self.label_map[name]
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 3
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from sklearn.model_selection import KFold
from torchvision.transforms.functional import InterpolationMode

train_transform = transform.Compose(
    [
        transform.Resize(
            (448, 448), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transform.RandomHorizontalFlip(),
        transform.ConvertImageDtype(torch.float32),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transform.Compose(
    [
        transform.Resize(
            (448, 448), interpolation=InterpolationMode.BILINEAR, antialias=True
        ),
        transform.ConvertImageDtype(torch.float32),
        transform.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
dataset_size = len(all_train_dataset)

_dl_kwargs = dict(
    num_workers=4,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=2,
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
                "train": DataLoader(
                    train_dataset, batch_size=BATCH_SIZE, shuffle=True, **_dl_kwargs
                ),
                "val": DataLoader(
                    val_dataset, batch_size=BATCH_SIZE, shuffle=False, **_dl_kwargs
                ),
            }
        )
        index += 1
    print(f"Prepared {len(fold_dataloader)} folds")
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset, batch_size=BATCH_SIZE, shuffle=True, **_dl_kwargs
            ),
            "val": None,
        }
    )



## === cell 4
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor



## === cell 5
__all__ = ["DenseNet", "densenet121", "densenet169", "densenet201", "densenet161"]

model_urls = {
    "densenet121": "https://download.pytorch.org/models/densenet121-a639ec97.pth",
    "densenet169": "https://download.pytorch.org/models/densenet169-b2777c0a.pth",
    "densenet201": "https://download.pytorch.org/models/densenet201-c1103571.pth",
    "densenet161": "https://download.pytorch.org/models/densenet161-8d451a50.pth",
}


class _DenseLayer(nn.Module):
    def __init__(
        self,
        num_input_features,
        growth_rate,
        bn_size,
        drop_rate,
        memory_efficient=False,
    ):
        super(_DenseLayer, self).__init__()
        self.add_module("norm1", nn.BatchNorm2d(num_input_features)),
        self.add_module("relu1", nn.ReLU(inplace=True)),
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        ),
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate)),
        self.add_module("relu2", nn.ReLU(inplace=True)),
        self.add_module(
            "conv2",
            nn.Conv2d(
                bn_size * growth_rate,
                growth_rate,
                kernel_size=3,
                stride=1,
                padding=1,
                bias=False,
            ),
        ),
        self.drop_rate = float(drop_rate)
        self.memory_efficient = memory_efficient

    def bn_function(self, inputs):
        concated_features = torch.cat(inputs, 1)
        bottleneck_output = self.conv1(self.relu1(self.norm1(concated_features)))
        return bottleneck_output

    def any_requires_grad(self, input):
        for tensor in input:
            if tensor.requires_grad:
                return True
        return False

    @torch.jit.unused
    def call_checkpoint_bottleneck(self, input):
        def closure(*inputs):
            return self.bn_function(*inputs)

        return cp.checkpoint(closure, input)

    def forward(self, input):
        if isinstance(input, Tensor):
            prev_features = [input]
        else:
            prev_features = input

        if self.memory_efficient and self.any_requires_grad(prev_features):
            if torch.jit.is_scripting():
                raise Exception("Memory Efficient not supported in JIT")
            bottleneck_output = self.call_checkpoint_bottleneck(prev_features)
        else:
            bottleneck_output = self.bn_function(prev_features)

        new_features = self.conv2(self.relu2(self.norm2(bottleneck_output)))
        if self.drop_rate > 0:
            new_features = F.dropout(
                new_features, p=self.drop_rate, training=self.training
            )
        return new_features


class _DenseBlock(nn.ModuleDict):
    _version = 2

    def __init__(
        self,
        num_layers,
        num_input_features,
        bn_size,
        growth_rate,
        drop_rate,
        memory_efficient=False,
    ):
        super(_DenseBlock, self).__init__()
        for i in range(num_layers):
            layer = _DenseLayer(
                num_input_features + i * growth_rate,
                growth_rate=growth_rate,
                bn_size=bn_size,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.add_module("denselayer%d" % (i + 1), layer)

    def forward(self, init_features):
        features = [init_features]
        for _, layer in self.items():
            new_features = layer(features)
            features.append(new_features)
        return torch.cat(features, 1)


class _Transition(nn.Sequential):
    def __init__(self, num_input_features, num_output_features):
        super(_Transition, self).__init__()
        self.add_module("norm", nn.BatchNorm2d(num_input_features))
        self.add_module("relu", nn.ReLU(inplace=True))
        self.add_module(
            "conv",
            nn.Conv2d(
                num_input_features,
                num_output_features,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("pool", nn.AvgPool2d(kernel_size=2, stride=2))


class DenseNet(nn.Module):
    def __init__(
        self,
        growth_rate=32,
        block_config=(6, 12, 24, 16),
        num_init_features=64,
        bn_size=4,
        drop_rate=0,
        num_classes=1000,
        memory_efficient=False,
    ):
        super(DenseNet, self).__init__()

        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv0",
                        nn.Conv2d(
                            3,
                            num_init_features,
                            kernel_size=7,
                            stride=2,
                            padding=3,
                            bias=False,
                        ),
                    ),
                    ("norm0", nn.BatchNorm2d(num_init_features)),
                    ("relu0", nn.ReLU(inplace=True)),
                    ("pool0", nn.MaxPool2d(kernel_size=3, stride=2, padding=1)),
                ]
            )
        )

        num_features = num_init_features
        for i, num_layers in enumerate(block_config):
            block = _DenseBlock(
                num_layers=num_layers,
                num_input_features=num_features,
                bn_size=bn_size,
                growth_rate=growth_rate,
                drop_rate=drop_rate,
                memory_efficient=memory_efficient,
            )
            self.features.add_module("denseblock%d" % (i + 1), block)
            num_features = num_features + num_layers * growth_rate
            if i != len(block_config) - 1:
                trans = _Transition(
                    num_input_features=num_features,
                    num_output_features=num_features // 2,
                )
                self.features.add_module("transition%d" % (i + 1), trans)
                num_features = num_features // 2

        self.features.add_module("norm5", nn.BatchNorm2d(num_features))

        self.classifier = nn.Linear(num_features + 1024, num_classes)

        self.squeeze4 = nn.Conv2d(1664, 256, 1)
        self.fc4 = nn.Linear(256 * 256, 1024)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

    def channel_correlation(self, x):
        batch, channel, height, width = x.shape
        x = x.view(batch, channel, -1)
        x_t = x.permute(0, 2, 1)
        x_2 = torch.pow(x, 2)
        x_2 = torch.sqrt(torch.sum(x_2, dim=2, keepdim=True))
        x_t_2 = x_2.permute(0, 2, 1)
        Denominator_x = torch.bmm(x_2, x_t_2)
        Numerator_x = torch.bmm(x, x_t)
        norm_x = Numerator_x / (Denominator_x + 1e-12)
        return norm_x

    def feature_extract(self, x):
        x = self.features.conv0(x)
        x = self.features.norm0(x)
        x = self.features.relu0(x)
        x = self.features.pool0(x)
        x = self.features.denseblock1(x)
        x = self.features.transition1(x)
        x = self.features.denseblock2(x)
        x = self.features.transition2(x)
        x = self.features.denseblock3(x)
        x = self.features.transition3(x)
        x = self.features.denseblock4(x)
        x4 = self.channel_correlation(self.squeeze4(x))
        x = self.features.norm5(x)
        return x, x4

    def forward(self, x):
        features, x4 = self.feature_extract(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        x4 = self.fc4(x4.view(x.shape[0], -1))
        out = torch.flatten(out, 1)
        out = self.classifier(torch.cat([out, x4], dim=1))
        return out


def _load_state_dict(model, arch, progress):
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    pretrain_path = f"../../pretrain/{arch}.pth"
    if not os.path.exists(pretrain_path):
        print(
            f"Pretrain weights not found at {pretrain_path}; skipping custom pretrain load."
        )
        return

    state_dict = torch.load(pretrain_path, map_location="cpu")
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    load = []
    not_load = []
    for name, param in state_dict.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except Exception:
                not_load.append(name)

    print("Load pretrain : ")
    print("Load : {} layers".format(len(load)))
    print("Miss : {} layers".format(len(not_load)))


def _densenet(
    arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs
):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, arch, progress)
    return model


def densenet121(pretrained=False, progress=True, **kwargs):
    return _densenet(
        "densenet121", 32, (6, 12, 24, 16), 64, pretrained, progress, **kwargs
    )


def densenet161(pretrained=False, progress=True, **kwargs):
    return _densenet(
        "densenet161", 48, (6, 12, 36, 24), 96, pretrained, progress, **kwargs
    )


def densenet169(pretrained=False, progress=True, **kwargs):
    return _densenet(
        "densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs
    )


def densenet201(pretrained=False, progress=True, **kwargs):
    return _densenet(
        "densenet201", 32, (6, 12, 48, 32), 64, pretrained, progress, **kwargs
    )




## === cell 6
device = "cuda:0" if torch.cuda.is_available() else "cpu"
print(device)

import torchvision


def _init_custom_from_torchvision_densenet169(custom_model: nn.Module):
    tv = torchvision.models.densenet169(
        weights=torchvision.models.DenseNet169_Weights.IMAGENET1K_V1
    )
    tv_sd = tv.state_dict()
    cm_sd = custom_model.state_dict()

    copied = 0
    skipped = 0
    for k, v in tv_sd.items():
        if k in cm_sd and cm_sd[k].shape == v.shape:
            cm_sd[k].copy_(v)
            copied += 1
        else:
            skipped += 1
    custom_model.load_state_dict(cm_sd, strict=False)
    print(
        f"Initialized from torchvision DenseNet169: copied={copied}, skipped={skipped}"
    )


def create_new_model(pretrained=True):
    m = densenet169(num_classes=5, pretrained=False).to(device)
    if torch.cuda.is_available():
        m = m.to(memory_format=torch.channels_last)
    if pretrained:
        _init_custom_from_torchvision_densenet169(m)
    return m




## === cell 7
model = None




## === cell 8
def create_loss_opti():
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
    b_image = image.to(device, non_blocking=torch.cuda.is_available())
    b_label = label.to(device, non_blocking=torch.cuda.is_available())

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

        output = model(b_image)
        loss = criterion(output, target_a) * lam + criterion(output, target_b) * (
            1.0 - lam
        )
    else:
        output = model(b_image)
        loss = criterion(output, b_label)

    _, predicted = torch.max(output.data, dim=1)
    correct = (predicted.cpu() == label).sum().item()
    if phase == "train":
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    return correct, loss.item()




## === cell 12
from tqdm import tqdm

max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model()
        criterion, optimizer, lr_scheduler = create_loss_opti()
        Best_ACC = 0.0
        for epoch in range(1, EPOCH + 1):
            tmp_ACCMeter = AverageMeter(True)
            tmp_LOSSMeter = AverageMeter(False)
            correct_t = 0
            total = 0
            loss_t = 0.0
            for phase in PHASE:
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
                    leave=True,
                ):
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase
                    )

                    if phase == "val":
                        tmp_ACCMeter.update(correct, label.size(0))
                        tmp_LOSSMeter.update(loss, label.size(0))
                        total += label.size(0)
                        loss_t += loss * label.size(0)
                        correct_t += correct

                if phase == "val" and Best_ACC < tmp_ACCMeter.avg:
                    Best_ACC = tmp_ACCMeter.avg
                    ACCMeter[index] = tmp_ACCMeter
                    LOSSMeter[index] = tmp_LOSSMeter
                    torch.save(
                        model.state_dict(),
                        "./resnet50_kfold_{}_{}_{:.2f}.pkl".format(
                            index + 1, epoch, tmp_ACCMeter.avg
                        ),
                    )
            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1, K_FOLD, epoch, EPOCH, loss_t / total, correct_t / total
                )
            )
            lr_scheduler.step()



## === cell 13
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 14
import pandas as pd

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

test_dataloader = DataLoader(
    test_dataset,
    batch_size=INFER_BATCH_SIZE,
    shuffle=False,
    num_workers=min(12, os.cpu_count() or 12),
    pin_memory=torch.cuda.is_available(),
    persistent_workers=True,
    prefetch_factor=4,
)

preferred_dir = "/kaggle/input/densenet169-cutmix"
if os.path.isdir(preferred_dir):
    params_paths = sorted(glob.glob(os.path.join(preferred_dir, "*.pkl")))
else:
    weight_globs = [
        "../input/densenet169-cutmix/*.pkl",
        "/kaggle/input/densenet169-cutmix/*.pkl",
        "/kaggle/input/*densenet*cutmix*/*.pkl",
    ]
    params_paths = []
    for g in weight_globs:
        params_paths.extend(glob.glob(g))
    params_paths = sorted(list(dict.fromkeys(params_paths)))  # unique, stable order

print(f"Found {len(params_paths)} weight files (before limiting)")

if len(params_paths) > MAX_ENSEMBLE_WEIGHTS:
    params_paths = params_paths[-MAX_ENSEMBLE_WEIGHTS:]
print(f"Using {len(params_paths)} weight files (after limiting)")


def cache_test_batches(dataloader):
    cached_imgs = []
    names_flat = []
    batch_sizes = []
    for img, img_name in dataloader:
        if isinstance(img, torch.Tensor) and not img.is_contiguous():
            img = img.contiguous()
        cached_imgs.append(img)  # CPU tensors after val_transform
        names_flat.extend(list(img_name))
        batch_sizes.append(int(img.shape[0]))
    return cached_imgs, np.array(names_flat, dtype=object), batch_sizes


def predict_probs_from_cached_batches(
    model, cached_imgs, batch_sizes, use_cuda, amp_ctx, out_probs
):
    model.eval()
    offset = 0

    gpu_buf = {}

    with torch.inference_mode():
        for img, bs in zip(cached_imgs, batch_sizes):
            if use_cuda:
                buf = gpu_buf.get(bs)
                if buf is None or buf.shape != img.shape:
                    buf = torch.empty_like(
                        img, device=device, memory_format=torch.channels_last
                    )
                    gpu_buf[bs] = buf
                buf.copy_(img, non_blocking=True)
                inp = buf
            else:
                inp = img.to(device)

            with amp_ctx:
                output = model(inp)
                p = torch.softmax(output, dim=1)
            out_probs[offset : offset + bs, :] = p.float().cpu().numpy()
            offset += bs
    return out_probs


def _train_fallback_single_fold_and_return_best_path():
    global model
    dataloader = fold_dataloader[0]

    model = create_new_model(pretrained=True)
    criterion, optimizer, lr_scheduler = create_loss_opti()

    best_acc = -1.0
    best_path = "/kaggle/working/fallback_best.pkl"

    for epoch in range(1, EPOCH + 1):
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)
        correct_t = 0
        total = 0
        loss_t = 0.0

        for phase in PHASE:
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
                leave=True,
            ):
                correct, loss = train_step(
                    model, criterion, optimizer, image, label, phase
                )

                if phase == "val":
                    tmp_ACCMeter.update(correct, label.size(0))
                    tmp_LOSSMeter.update(loss, label.size(0))
                    total += label.size(0)
                    loss_t += loss * label.size(0)
                    correct_t += correct

            if phase == "val" and tmp_ACCMeter.avg > best_acc:
                best_acc = tmp_ACCMeter.avg
                torch.save(model.state_dict(), best_path)

        print(
            "Fallback Fold Epoch : {} / {} loss : {:.6f} ACC : {:.6f} (best {:.6f})".format(
                epoch,
                EPOCH,
                loss_t / max(total, 1),
                correct_t / max(total, 1),
                best_acc,
            )
        )
        lr_scheduler.step()

    print(f"Fallback training complete. Best val acc={best_acc:.6f}, saved={best_path}")
    return best_path


use_cuda = torch.cuda.is_available()
amp_ctx = (
    torch.amp.autocast(device_type="cuda", dtype=torch.float16)
    if use_cuda
    else torch.autocast(device_type="cpu", enabled=False)
)

cached_test_imgs, cached_test_names, cached_test_batch_sizes = cache_test_batches(
    test_dataloader
)

n_test = len(cached_test_names)


def maybe_compile_for_infer(m: torch.nn.Module):
    if not use_cuda:
        return m
    try:
        return torch.compile(m, mode="reduce-overhead", fullgraph=False)
    except Exception as e:
        print(f"torch.compile unavailable/failed; running eager. Reason: {e}")
        return m


if len(params_paths) == 0:
    best_path = _train_fallback_single_fold_and_return_best_path()
    model = create_new_model(pretrained=True)
    params = torch.load(best_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(params, strict=False)
    print(
        f"Loaded fallback trained weights. Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}"
    )
    model = maybe_compile_for_infer(model)
    tmp = np.empty((n_test, 5), dtype=np.float32)
    probs_sum = predict_probs_from_cached_batches(
        model, cached_test_imgs, cached_test_batch_sizes, use_cuda, amp_ctx, tmp
    )
else:
    probs_sum = np.zeros((n_test, 5), dtype=np.float32)
    tmp = np.empty((n_test, 5), dtype=np.float32)

    for params_path in params_paths:
        model = create_new_model(pretrained=False)
        params = torch.load(params_path, map_location="cpu")
        missing, unexpected = model.load_state_dict(params, strict=False)
        print(f"Trained weight loaded from: {params_path}")
        print(f"Missing keys: {len(missing)}; Unexpected keys: {len(unexpected)}")

        model = maybe_compile_for_infer(model)
        predict_probs_from_cached_batches(
            model, cached_test_imgs, cached_test_batch_sizes, use_cuda, amp_ctx, tmp
        )
        probs_sum += tmp

image_labels = probs_sum.argmax(axis=1).astype(np.int64)
pred_df = pd.DataFrame({"image_id": cached_test_names, "label": image_labels})

sample = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
pred_df = sample[["image_id"]].merge(pred_df, on="image_id", how="left")
if pred_df["label"].isna().any():
    pred_df["label"] = pred_df["label"].fillna(0).astype(np.int64)

pred_df.to_csv("/kaggle/working/submission.csv", index=False)
print(pred_df.head())
print(f"Wrote submission to /kaggle/working/submission.csv with shape {pred_df.shape}")
