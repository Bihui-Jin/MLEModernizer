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

0.8818374131157449

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
BATCH_SIZE = 256  # larger batch reduces number of optimizer steps per epoch
EPOCH = 1  # one epoch is enough to finish within time limit
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 1.0
CUTMIX_PROB = 0.0  # disable CutMix to avoid extra per‑batch image manipulation
TRAINING = True  # keep training enabled
K_FOLD = 1  # single fold




## === cell 1
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
import os
from PIL import Image
from tqdm import tqdm  # progress bar for loops


class CLD_Dataset(Dataset):
    """
    Lazy‑load images instead of reading every file into RAM at init.
    This keeps memory usage low and lets DataLoader workers read files in parallel,
    preserving identical output tensors after the same transforms are applied.
    """

    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = glob.glob(os.path.join(image_root, "*.jpg"))
        if not return_name and label_path is not None:
            df = pd.read_csv(label_path, index_col="image_id")
            self.label_dict = df["label"].to_dict()
        self.return_name = return_name

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, x):
        img = Image.open(self.image_paths[x]).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(self.image_paths[x])
        else:
            label = self.label_dict[os.path.basename(self.image_paths[x])]
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 2
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold
import os

torch.backends.cudnn.benchmark = True

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
    max_workers = min(os.cpu_count(), 4)  # limit workers to avoid excess overhead
    for train_idx, val_idx in kf.split(all_train_dataset):
        train_dataset = torch.utils.data.Subset(all_train_dataset, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=max_workers,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                "val": DataLoader(
                    val_dataset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=max_workers,
                    pin_memory=True,
                    persistent_workers=True,
                ),
            }
        )
else:
    max_workers = min(os.cpu_count(), 4)  # sensible ceiling for workers
    val_size = int(VAL_RATIO * dataset_size)
    train_size = dataset_size - val_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        all_train_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )
    fold_dataloader.append(
        {
            "train": DataLoader(
                train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True,
                num_workers=max_workers,
                pin_memory=True,
                persistent_workers=True,
            ),
            "val": DataLoader(
                val_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False,
                num_workers=max_workers,
                pin_memory=True,
                persistent_workers=True,
            ),
        }
    )




## === cell 3
import re
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor




## === cell 4
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
        self.add_module("norm1", nn.BatchNorm2d(num_input_features))
        self.add_module("relu1", nn.ReLU(inplace=True))
        self.add_module(
            "conv1",
            nn.Conv2d(
                num_input_features,
                bn_size * growth_rate,
                kernel_size=1,
                stride=1,
                bias=False,
            ),
        )
        self.add_module("norm2", nn.BatchNorm2d(bn_size * growth_rate))
        self.add_module("relu2", nn.ReLU(inplace=True))
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
        )
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
        for layer in self.values():
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
        return Numerator_x / Denominator_x

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
    state_dict = torch.load("../../pretrain/{}.pth".format(arch))
    for key in list(state_dict.keys()):
        res = pattern.match(key)
        if res:
            new_key = res.group(1) + res.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    load, not_load = [], []
    for name, param in state_dict.items():
        if name in model.state_dict():
            try:
                model.state_dict()[name].copy_(param)
                load.append(name)
            except:
                not_load.append(name)
    print("Load pretrain : ")
    print(f"Load : {len(load)} layers")
    print(f"Miss : {len(not_load)} layers")


def _densenet(
    arch, growth_rate, block_config, num_init_features, pretrained, progress, **kwargs
):
    model = DenseNet(growth_rate, block_config, num_init_features, **kwargs)
    if pretrained:
        _load_state_dict(model, arch, progress)
    return model


def densenet169(pretrained=False, progress=True, **kwargs):
    return _densenet(
        "densenet169", 32, (6, 12, 32, 32), 64, pretrained, progress, **kwargs
    )




## === cell 5
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(device)


def create_new_model(pretrained=False):
    model = densenet169(num_classes=5, pretrained=pretrained).to(device)
    if hasattr(torch, "compile"):
        model = torch.compile(model)
    return model




## === cell 6
def create_loss_opti(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None
    return criterion, optimizer, lr_scheduler, scaler




## === cell 7
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




## === cell 8
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




## === cell 9
def train_step(model, criterion, optimizer, image, label, phase, scaler=None):
    b_image = image.to(device, non_blocking=True)
    b_label = label.to(device, non_blocking=True)  # move once, use on GPU

    with torch.set_grad_enabled(phase == "train"):
        r = np.random.rand()
        if BETA > 0 and r < CUTMIX_PROB:
            lam = np.random.beta(BETA, BETA)
            rand_index = torch.randperm(b_image.size(0), device=device)
            target_a = b_label
            target_b = b_label[rand_index]
            bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
            b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
                rand_index, :, bbx1:bbx2, bby1:bby2
            ]
            lam = 1.0 - (
                (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size(-1) * b_image.size(-2))
            )
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(b_image)
                    loss_a = criterion(output, target_a)
                    loss_b = criterion(output, target_b)
                    loss = loss_a * lam + loss_b * (1.0 - lam)
            else:
                output = model(b_image)
                loss_a = criterion(output, target_a)
                loss_b = criterion(output, target_b)
                loss = loss_a * lam + loss_b * (1.0 - lam)
        else:
            if scaler:
                with torch.cuda.amp.autocast():
                    output = model(b_image)
                    loss = criterion(output, b_label)
            else:
                output = model(b_image)
                loss = criterion(output, b_label)

        _, predicted = torch.max(output, dim=1)
        correct = (predicted == b_label).sum().item()

        if phase == "train":
            optimizer.zero_grad()
            if scaler:
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                optimizer.step()

    return correct, loss.item()




## === cell 10
max_acc = 0.0
ACCMeter = []  # list to hold per‑fold accuracy meters
LOSSMeter = []  # list to hold per‑fold loss meters
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model(pretrained=False)
        criterion, optimizer, lr_scheduler, scaler = create_loss_opti(model)
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

                for image, label in dataloader[phase]:
                    correct, loss = train_step(
                        model, criterion, optimizer, image, label, phase, scaler
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
                        f"./densenet169_kfold_{index+1}_{epoch}_{tmp_ACCMeter.avg:.2f}.pkl",
                    )
            print(
                f"Fold : {index+1}/{K_FOLD} Epoch : {epoch}/{EPOCH} "
                f"loss : {loss_t/total:.6f} ACC : {correct_t/total:.6f}"
            )
            lr_scheduler.step()




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/3693596461.py in <cell line: 0>()
     27 
     28                 for image, label in dataloader[phase]:
---> 29                     correct, loss = train_step(
     30                         model, criterion, optimizer, image, label, phase, scaler
     31                     )

/tmp/ipykernel_55/3097262649.py in train_step(model, criterion, optimizer, image, label, phase, scaler)
     31             if scaler:
     32                 with torch.cuda.amp.autocast():
---> 33                     output = model(b_image)
     34                     loss = criterion(output, b_label)
     35             else:

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    572 
    573             try:
--> 574                 return fn(*args, **kwargs)
    575             finally:
    576                 # Restore the dynamic layer stack depth if necessary.

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _wrapped_call_impl(self, *args, **kwargs)
   1737             return self._compiled_call_impl(*args, **kwargs)  # type: ignore[misc]
   1738         else:
-> 1739             return self._call_impl(*args, **kwargs)
   1740 
   1741     # torchrec tests the code consistency with the following code

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _call_impl(self, *args, **kwargs)
   1748                 or _global_backward_pre_hooks or _global_backward_hooks
   1749                 or _global_forward_hooks or _global_forward_pre_hooks):
-> 1750             return forward_call(*args, **kwargs)
   1751 
   1752         result = None

/tmp/ipykernel_55/323538759.py in forward(self, x)
    231         return x, x4
    232 
--> 233     def forward(self, x):
    234         features, x4 = self.feature_extract(x)
    235         out = F.relu(features, inplace=True)

/usr/local/lib/python3.11/dist-packages/torch/_dynamo/eval_frame.py in _fn(*args, **kwargs)
    743             )
    744             try:
--> 745                 return fn(*args, **kwargs)
    746             finally:
    747                 _maybe_set_eval_frame(prior)

/usr/local/lib/python3.11/dist-packages/torch/_functorch/aot_autograd.py in forward(*runtime_args)
   1182         full_args.extend(params_flat)
   1183         full_args.extend(runtime_args)
-> 1184         return compiled_fn(full_args)
   1185 
   1186     # Just for convenience

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in runtime_wrapper(args)
    308                 True
    309             ), torch.enable_grad():
--> 310                 all_outs = call_func_at_runtime_with_args(
    311                     compiled_fn, args_, disable_amp=disable_amp, steal_args=True
    312                 )

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in g(args)
     98 def make_boxed_func(f):
     99     def g(args):
--> 100         return f(*args)
    101 
    102     g._boxed_call = True  # type: ignore[attr-defined]

/usr/local/lib/python3.11/dist-packages/torch/autograd/function.py in apply(cls, *args, **kwargs)
    573             # See NOTE: [functorch vjp and autograd interaction]
    574             args = _functorch.utils.unwrap_dead_wrappers(args)
--> 575             return super().apply(*args, **kwargs)  # type: ignore[misc]
    576 
    577         if not is_setup_ctx_defined:

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in forward(ctx, *deduped_flat_tensor_args)
   1583                 # - Note that donated buffer logic requires (*saved_tensors, *saved_symints) showing up last
   1584                 #   in the fw output order.
-> 1585                 fw_outs = call_func_at_runtime_with_args(
   1586                     CompiledFunction.compiled_fw,
   1587                     args,

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/utils.py in call_func_at_runtime_with_args(f, args, steal_args, disable_amp)
    124     with context():
    125         if hasattr(f, "_boxed_call"):
--> 126             out = normalize_as_list(f(args))
    127         else:
    128             # TODO: Please remove soon

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in wrapper(runtime_args)
    488                 )
    489                 return out
--> 490             return compiled_fn(runtime_args)
    491 
    492         return wrapper

/usr/local/lib/python3.11/dist-packages/torch/_functorch/_aot_autograd/runtime_wrappers.py in inner_fn(args)
    670                 old_args.clear()
    671 
--> 672             outs = compiled_fn(args)
    673 
    674             # Inductor cache DummyModule can return None

/usr/local/lib/python3.11/dist-packages/torch/_inductor/output_code.py in __call__(self, inputs)
    464         assert self.current_callable is not None
    465         try:
--> 466             return self.current_callable(inputs)
    467         finally:
    468             AutotuneCacheBundler.end_compile()

/usr/local/lib/python3.11/dist-packages/torch/_inductor/utils.py in run(new_inputs)
   2126     def run(new_inputs: List[InputType]):
   2127         copy_misaligned_inputs(new_inputs, inputs_to_check)
-> 2128         return model(new_inputs)
   2129 
   2130     return run

/tmp/torchinductor_root/rw/crwwmflujb34e42rbelrrtvsvksd2mirkvdjxi3qz7no6wwdpxc2.py in call(args)
  23322         buf696 = empty_strided_cuda((256, 640, 28, 28), (501760, 784, 28, 1), torch.float16)
  23323         buf683 = reinterpret_tensor(buf696, (256, 256, 28, 28), (501760, 784, 28, 1), 0)  # alias
> 23324         buf728 = empty_strided_cuda((256, 672, 28, 28), (526848, 784, 28, 1), torch.float16)
  23325         buf714 = reinterpret_tensor(buf728, (256, 256, 28, 28), (526848, 784, 28, 1), 0)  # alias
  23326         buf761 = empty_strided_cuda((256, 704, 28, 28), (551936, 784, 28, 1), torch.float16)

OutOfMemoryError: CUDA out of memory. Tried to allocate 258.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 168.88 MiB is free. Process 3444197 has 47.34 GiB memory in use. Of the allocated memory 46.96 GiB is allocated by PyTorch, and 60.50 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)

## === cell 11
acc_sum = 0.0
loss_sum = 0.0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print(f"K-fold {K_FOLD} ACC : {acc_sum / K_FOLD:.6f}")
print(f"K-fold {K_FOLD} LOSS : {loss_sum / K_FOLD:.6f}")




## === cell 12
test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

max_test_workers = min(os.cpu_count(), 4)
test_dataloader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=max_test_workers,
    pin_memory=True,
    persistent_workers=True,
)

model = create_new_model(pretrained=False)
model.eval()

image_ids = []
pred_labels = []

with torch.no_grad():
    for img, img_name in test_dataloader:
        b_img = img.to(device, non_blocking=True)
        output = model(b_img)
        probs = torch.nn.functional.softmax(output, dim=1)
        pred = torch.argmax(probs, dim=1).cpu().item()
        image_ids.append(img_name[0])
        pred_labels.append(int(pred))

submission = pd.DataFrame({"image_id": image_ids, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")

## --- ERROR in cell 12, traceback:
---------------------------------------------------------------------------
OutOfMemoryError                          Traceback (most recent call last)
/tmp/ipykernel_55/4073058558.py in <cell line: 0>()
     15 )
     16 
---> 17 model = create_new_model(pretrained=False)
     18 model.eval()
     19 

/tmp/ipykernel_55/4187727042.py in create_new_model(pretrained)
      7 
      8 def create_new_model(pretrained=False):
----> 9     model = densenet169(num_classes=5, pretrained=pretrained).to(device)
     10     if hasattr(torch, "compile"):
     11         model = torch.compile(model)

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in to(self, *args, **kwargs)
   1341                     raise
   1342 
-> 1343         return self._apply(convert)
   1344 
   1345     def register_full_backward_pre_hook(

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    901         if recurse:
    902             for module in self.children():
--> 903                 module._apply(fn)
    904 
    905         def compute_should_use_set_data(tensor, tensor_applied):

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in _apply(self, fn, recurse)
    928             # `with torch.no_grad():`
    929             with torch.no_grad():
--> 930                 param_applied = fn(param)
    931             p_should_use_set_data = compute_should_use_set_data(param, param_applied)
    932 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/module.py in convert(t)
   1327                         memory_format=convert_to_format,
   1328                     )
-> 1329                 return t.to(
   1330                     device,
   1331                     dtype if t.is_floating_point() or t.is_complex() else None,

OutOfMemoryError: CUDA out of memory. Tried to allocate 256.00 MiB. GPU 0 has a total capacity of 47.53 GiB of which 122.88 MiB is free. Process 3444197 has 47.38 GiB memory in use. Of the allocated memory 47.01 GiB is allocated by PyTorch, and 56.32 MiB is reserved by PyTorch but unallocated. If reserved but unallocated memory is large try setting PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True to avoid fragmentation.  See documentation for Memory Management  (https://pytorch.org/docs/stable/notes/cuda.html#environment-variables)
