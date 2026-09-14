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

TRAINING = True

K_FOLD = 1

import os as _os

if _os.environ.get("FORCE_TRAINING", "1") != "1":
    TRAINING = False



## === cell 1
import os
import glob
import random
import numpy as np
import pandas as pd

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transform



## === cell 2
from torch.utils.data.dataset import Dataset
from PIL import Image

try:
    import torchvision

    if hasattr(torchvision, "set_image_backend"):
        torchvision.set_image_backend("accimage")
except Exception:
    pass


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_root, "*.jpg"))
        self.image_paths = sorted(self.image_paths)
        self.return_name = return_name

        self._labels = None
        if not return_name:
            if label_path is None:
                raise ValueError("label_path must be provided when return_name=False")

            df = pd.read_csv(label_path, usecols=["image_id", "label"])
            label_map = dict(
                zip(df["image_id"].tolist(), df["label"].astype(np.int64).tolist())
            )
            basenames = [os.path.basename(p) for p in self.image_paths]
            self._labels = np.fromiter(
                (label_map[n] for n in basenames), count=len(basenames), dtype=np.int64
            )

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(self.image_paths[x])
        else:
            return img, int(self._labels[x])

    def __len__(self):
        return len(self.image_paths)




## === cell 3
from sklearn.model_selection import StratifiedKFold

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

_num_workers = min(8, (os.cpu_count() or 4))
_loader_kwargs = dict(
    num_workers=_num_workers,
    pin_memory=True,
    prefetch_factor=4,
)
if _num_workers > 0:
    _loader_kwargs["persistent_workers"] = True

fold_dataloader = []
if K_FOLD != 1:
    y = all_train_dataset._labels
    skf = StratifiedKFold(n_splits=K_FOLD, shuffle=True, random_state=42)

    index = 0
    for train_idx, val_idx in skf.split(np.zeros(len(y)), y):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    **_loader_kwargs,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    **_loader_kwargs,
                ),
            }
        )
        index += 1
else:
    fold_dataloader.append(
        {
            "train": DataLoader(
                all_train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                **_loader_kwargs,
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

    def forward(self, input):  # noqa: F811
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
        for name, layer in self.items():
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
        self.classifier = nn.Linear(num_features, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.constant_(m.bias, 0)

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
        x = self.features.norm5(x)
        return x

    def forward(self, x):
        features = self.feature_extract(x)
        out = F.relu(features, inplace=True)
        out = F.adaptive_avg_pool2d(out, (1, 1))
        out = torch.flatten(out, 1)
        out = self.classifier(out)
        return out


def _load_state_dict(model, arch, progress):
    pattern = re.compile(
        r"^(.*denselayer\d+\.(?:norm|relu|conv))\.((?:[12])\.(?:weight|bias|running_mean|running_var))$"
    )

    local_path = "../../pretrain/{}.pth".format(arch)
    if not os.path.exists(local_path):
        print(
            f"Pretrain file not found at {local_path}; skipping custom pretrain load."
        )
        return

    state_dict = torch.load(local_path, map_location="cpu")
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
random.seed(42)
np.random.seed(42)
torch.manual_seed(42)
torch.cuda.manual_seed_all(42)
torch.backends.cudnn.benchmark = True  # speed

if torch.cuda.is_available():
    device = "cuda:0"
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    try:
        torch.set_float32_matmul_precision("high")
    except Exception:
        pass
else:
    device = "cpu"
print(device)


def create_new_model(pretrained=True):
    return densenet121(num_classes=5, pretrained=pretrained).to(device)




## === cell 7
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 8
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




## === cell 9
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




## === cell 10
def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device, non_blocking=True)
    b_label = label.to(device, non_blocking=True)

    r = torch.rand((), device=device)
    if BETA > 0 and r.item() < CUTMIX_PROB:
        g1 = torch.distributions.Gamma(BETA, 1.0).sample(()).to(device)
        g2 = torch.distributions.Gamma(BETA, 1.0).sample(()).to(device)
        lam = (g1 / (g1 + g2)).item()

        rand_index = torch.randperm(b_image.size(0), device=device)
        target_a = b_label
        target_b = b_label[rand_index]

        W = b_image.size(2)
        H = b_image.size(3)
        cut_rat = float(np.sqrt(1.0 - lam))
        cut_w = int(W * cut_rat)
        cut_h = int(H * cut_rat)

        cx = int(torch.randint(0, W, (1,), device=device).item())
        cy = int(torch.randint(0, H, (1,), device=device).item())

        bbx1 = max(cx - cut_w // 2, 0)
        bby1 = max(cy - cut_h // 2, 0)
        bbx2 = min(cx + cut_w // 2, W)
        bby2 = min(cy + cut_h // 2, H)

        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size(-1) * b_image.size(-2))
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




## === cell 11
from tqdm import tqdm

max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))



## === cell 12
weight_globs = [
    "../input/densenet121-softtarget/*.pkl",
    "/kaggle/input/densenet121-softtarget/*.pkl",
    "/kaggle/input/*/*.pkl",
    "/kaggle/working/*.pkl",
]
weight_paths = []
for pat in weight_globs:
    weight_paths.extend(glob.glob(pat))
weight_paths = sorted(list(dict.fromkeys(weight_paths)))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model(pretrained=True)
        criterion, optimizer, lr_scheduler = create_loss_opti()
        Best_ACC = 0.0
        best_path = f"/kaggle/working/densenet121_kfold_{index+1}_best.pkl"
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

                _loader = dataloader[phase]
                if _loader is None:
                    _loader = dataloader["train"]

                for image, label in tqdm(
                    _loader,
                    total=len(_loader),
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
                    torch.save(model.state_dict(), best_path)

            print(
                "Fold : {}/ {} Epoch : {} / {} loss : {:.6f} ACC : {:.6f}".format(
                    index + 1,
                    K_FOLD,
                    epoch,
                    EPOCH,
                    loss_t / max(total, 1),
                    correct_t / max(total, 1),
                )
            )
            lr_scheduler.step()
        print("Saved best fold weights to:", best_path)
else:
    if len(weight_paths) == 0:
        print(
            "No external *.pkl weights found and TRAINING=False; will run inference with torchvision ImageNet pretrained DenseNet121 fallback."
        )
    else:
        print(f"Found {len(weight_paths)} external weight files; will ensemble them.")



## === cell 13
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} LOSS : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 14
test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

TEST_BATCH_SIZE = 64
_test_num_workers = min(8, (os.cpu_count() or 4))
_test_loader_kwargs = dict(
    num_workers=_test_num_workers,
    pin_memory=True,
    prefetch_factor=4,
)
if _test_num_workers > 0:
    _test_loader_kwargs["persistent_workers"] = True

test_dataloader = DataLoader(
    test_dataset,
    batch_size=TEST_BATCH_SIZE,
    shuffle=False,
    **_test_loader_kwargs,
)

trained_best_paths = sorted(glob.glob("/kaggle/working/densenet121_kfold_*_best.pkl"))
if len(trained_best_paths) > 0:
    weight_paths = trained_best_paths
else:
    weight_paths = []
    for pat in weight_globs:
        weight_paths.extend(glob.glob(pat))
    weight_paths = sorted(list(dict.fromkeys(weight_paths)))

test_image_names = [os.path.basename(p) for p in test_dataset.image_paths]
n_test = len(test_image_names)

sum_probs = np.zeros((n_test, 5), dtype=np.float64)

use_cuda = device != "cpu"
_CAN_COMPILE = hasattr(torch, "compile")


def _infer_with_model(model):
    model.eval()
    if use_cuda:
        model = model.to(memory_format=torch.channels_last)

    if _CAN_COMPILE:
        try:
            model = torch.compile(model, mode="reduce-overhead", fullgraph=False)
        except Exception:
            pass

    offset = 0
    with torch.inference_mode():
        for img, _img_name in test_dataloader:
            b = img.size(0)
            if use_cuda:
                img = img.to(memory_format=torch.channels_last)
            b_img = img.to(device, non_blocking=True)

            logits = model(b_img)
            prob = torch.softmax(logits, dim=1).cpu().numpy()
            sum_probs[offset : offset + b] += prob
            offset += b


if len(weight_paths) > 0:
    for params_path in weight_paths:
        model = create_new_model(pretrained=False)
        params = torch.load(params_path, map_location="cpu")
        incompatible = model.load_state_dict(params, strict=False)

        print("Loaded weights:", params_path)
        print(
            "Trained weight load : {}".format(
                len(params) - len(incompatible.unexpected_keys)
            )
        )
        print(
            "Trained weight not load : {}".format(
                len(incompatible.missing_keys) + len(incompatible.unexpected_keys)
            )
        )
        _infer_with_model(model)
else:
    import torchvision as tv

    try:
        weights = tv.models.DenseNet121_Weights.DEFAULT
        tv_model = tv.models.densenet121(weights=weights)
        print("Using torchvision DenseNet121_Weights.DEFAULT fallback.")
    except Exception:
        tv_model = tv.models.densenet121(weights=None)
        print("Using torchvision DenseNet121 without weights (fallback).")

    in_features = tv_model.classifier.in_features
    tv_model.classifier = torch.nn.Linear(in_features, 5)
    tv_model = tv_model.to(device)
    _infer_with_model(tv_model)

image_labels = np.argmax(sum_probs, axis=1).astype(int)

df = pd.DataFrame({"image_id": np.array(test_image_names), "label": image_labels})

sample_sub = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
df = sample_sub[["image_id"]].merge(df, on="image_id", how="left")
if df["label"].isna().any():
    raise RuntimeError(
        "Some test image_ids missing predictions after merge; check dataset paths."
    )
df["label"] = df["label"].astype(int)

print(df.head())
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", df.shape)
