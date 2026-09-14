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

0.8860682985796313

# 6. Current score

0.11323

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.08931) has done: 'I fix the runtime failure by removing the hard dependency on an unattached external weights dataset and instead using torchvision’s built-in DenseNet-121 ImageNet pretrained weights (same architecture) so inference can run end-to-end. I keep your dataset, transforms, batching, and submission formatting intact, only swapping the model factory to use torchvision weights and ensuring the test image ordering is deterministic. I also fix a small Dataset init bug (label_path can be None when return_name=False) and add safe CPU/GPU map_location handling. This produce a valid `/kaggle/working/submission.csv` and should improve accuracy versus random initialization, moving the score toward the target.'
- What this solution (achieved 0.11323) has done: 'The timeout is dominated by doing full 5-fold, 10-epoch training on 18k images at 448×448 (far beyond 600s), plus some avoidable Python overhead inside the train loop. I keep the exact model, transforms, CutMix, optimizer/scheduler, and loop semantics, but make the run finish by (1) auto-disabling training unless all fold checkpoints already exist (so the notebook runs inference-only under the timeout), and (2) accelerating inference by loading fold weights with `strict=False`, enabling cuDNN autotuning/TF32, and using pinned-memory, persistent DataLoader workers. I also remove per-layer copy loops when loading weights (equivalent to `load_state_dict` behavior) and reduce per-batch Python overhead while preserving identical outputs. Paths and core logic remain unchanged; this is purely eliminating work that cannot fit in 600 seconds and speeding up the remaining inference path.'

# 9. Code solution

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
K_FOLD = 5



## === cell 1
import numpy as np



## === cell 2
from torch.utils.data.dataset import Dataset
import glob
import pandas as pd
import os
from PIL import Image
from PIL import ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True


class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super().__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.return_name = return_name

        self._labels = None
        if not self.return_name:
            if label_path is None:
                raise ValueError("label_path must be provided when return_name=False")
            df = pd.read_csv(label_path)
            label_dict = dict(zip(df["image_id"].values, df["label"].values))
            self._labels = np.fromiter(
                (int(label_dict[os.path.basename(p)]) for p in self.image_paths),
                dtype=np.int64,
                count=len(self.image_paths),
            )

    def set_transform(self, transform):
        self.transform = transform

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        with Image.open(path) as im:
            img = im.convert("RGB")
            img.load()
        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(path)
        return img, int(self._labels[idx])

    def __len__(self):
        return len(self.image_paths)




## === cell 3
import torchvision.transforms as transform
from torch.utils.data import DataLoader
import torch
from sklearn.model_selection import KFold

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

all_train_dataset_train = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)
all_train_dataset_val = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    val_transform,
)
dataset_size = len(all_train_dataset_train)


def _make_loader(ds, shuffle):
    nw = min(4, max(1, (os.cpu_count() or 4) // 2))
    kwargs = dict(
        batch_size=BATCH_SIZE,
        shuffle=shuffle,
        num_workers=nw,
        pin_memory=True,
        persistent_workers=(nw > 0),
        prefetch_factor=2 if nw > 0 else None,
    )
    kwargs = {k: v for k, v in kwargs.items() if v is not None}
    return DataLoader(ds, **kwargs)


fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(K_FOLD, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(range(dataset_size)):
        train_dataset = torch.utils.data.Subset(all_train_dataset_train, train_idx)
        val_dataset = torch.utils.data.Subset(all_train_dataset_val, val_idx)
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
            "train": _make_loader(all_train_dataset_train, shuffle=True),
            "val": None,
        }
    )



## === cell 4
import random


def seed_everything(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


seed_everything(42)



## === cell 5
import re
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
from collections import OrderedDict
from torch import Tensor
from torch.jit.annotations import List



## === cell 6
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

    @torch.jit._overload_method
    def forward(self, input):
        pass

    @torch.jit._overload_method
    def forward(self, input):
        pass

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

    state_dict = torch.load("../../pretrain/{}.pth".format(arch))
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
            except:
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




## === cell 7
if torch.cuda.is_available():
    device = "cuda:0"
else:
    device = "cpu"
print(device)

import torchvision


def _maybe_compile(m: torch.nn.Module):
    if device.startswith("cuda") and hasattr(torch, "compile"):
        try:
            return torch.compile(m, mode="reduce-overhead", fullgraph=False)
        except Exception as e:
            print("torch.compile unavailable/fallback:", repr(e))
            return m
    return m


def create_new_model(pretrained=True):
    if pretrained:
        weights = torchvision.models.DenseNet121_Weights.IMAGENET1K_V1
        m = torchvision.models.densenet121(weights=weights)
        m.classifier = torch.nn.Linear(m.classifier.in_features, 5)
        m = m.to(device)
        m = _maybe_compile(m)
        return m
    else:
        m = densenet121(num_classes=5, pretrained=False).to(device)
        m = _maybe_compile(m)
        return m




## === cell 8
model = create_new_model(pretrained=True)




## === cell 9
def create_loss_opti():
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    return criterion, optimizer, lr_scheduler




## === cell 10
from tqdm import tqdm




## === cell 11
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




## === cell 12
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




## === cell 13
_amp_enabled = device.startswith("cuda")
_scaler = torch.cuda.amp.GradScaler(enabled=_amp_enabled)


def train_step(model, criterion, optimizer, image, label, phase):
    b_image = image.to(device, non_blocking=True)
    b_label = label.to(device, non_blocking=True)

    r = np.random.rand(1)
    if BETA > 0 and r < CUTMIX_PROB:
        lam = np.random.beta(BETA, BETA)
        rand_index = torch.randperm(b_image.size()[0], device=b_image.device)
        target_a = b_label
        target_b = b_label[rand_index]
        bbx1, bby1, bbx2, bby2 = rand_bbox(b_image.size(), lam)
        b_image[:, :, bbx1:bbx2, bby1:bby2] = b_image[
            rand_index, :, bbx1:bbx2, bby1:bby2
        ]
        lam = 1 - (
            (bbx2 - bbx1) * (bby2 - bby1) / (b_image.size()[-1] * b_image.size()[-2])
        )

        if phase == "train":
            with torch.cuda.amp.autocast(enabled=_amp_enabled):
                output = model(b_image)
                loss = criterion(output, target_a) * lam + criterion(
                    output, target_b
                ) * (1.0 - lam)
        else:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_amp_enabled):
                output = model(b_image)
                loss = criterion(output, target_a) * lam + criterion(
                    output, target_b
                ) * (1.0 - lam)
    else:
        if phase == "train":
            with torch.cuda.amp.autocast(enabled=_amp_enabled):
                output = model(b_image)
                loss = criterion(output, b_label)
        else:
            with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_amp_enabled):
                output = model(b_image)
                loss = criterion(output, b_label)

    predicted = output.argmax(dim=1)
    correct = (predicted.detach().cpu() == label).sum().item()

    if phase == "train":
        optimizer.zero_grad(set_to_none=True)
        _scaler.scale(loss).backward()
        _scaler.step(optimizer)
        _scaler.update()

    return correct, float(loss.detach().cpu())




## === cell 14
max_acc = 0.0
ACCMeter = []
LOSSMeter = []
for i in range(K_FOLD):
    ACCMeter.append(AverageMeter(True))
    LOSSMeter.append(AverageMeter(False))

best_ckpt_paths = [
    f"/kaggle/working/densenet121_fold{i+1}_best.pkl" for i in range(K_FOLD)
]


existing_ckpts = [p for p in best_ckpt_paths if os.path.exists(p)]
if TRAINING:
    if len(existing_ckpts) == K_FOLD:
        print(
            f"Found all {K_FOLD}/{K_FOLD} existing fold checkpoints in /kaggle/working; "
            "skipping training to meet timeout."
        )
        TRAINING = False
    else:
        print(
            "WARNING: Full 5-fold training is very unlikely to finish within 600 seconds. "
            "To preserve runtime limits, training is being disabled and inference will use "
            "available fold weights from /kaggle/input/densenet121-kfold5/ if present."
        )
        TRAINING = False

if TRAINING:
    for index, dataloader in enumerate(fold_dataloader):
        model = create_new_model(pretrained=True)

        for p in model.parameters():
            p.requires_grad = False
        for p in model.classifier.parameters():
            p.requires_grad = True

        criterion = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=LR,
            weight_decay=WD,
        )
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=10, T_mult=1, eta_min=1e-6, last_epoch=-1
        )

        Best_ACC = 0.0
        tmp_ACCMeter = AverageMeter(True)
        tmp_LOSSMeter = AverageMeter(False)

        for epoch in range(1, EPOCH + 1):
            correct_t = 0
            total = 0
            loss_t = 0.0

            for phase in PHASE:
                if phase == "train":
                    model.train(True)
                else:
                    model.train(False)

                loader = dataloader[phase]
                for image, label in tqdm(
                    loader, total=len(loader), position=0, leave=False
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
                    torch.save(model.state_dict(), best_ckpt_paths[index])

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



## === cell 15
acc_sum = 0
loss_sum = 0
for i in range(K_FOLD):
    acc_sum += ACCMeter[i].avg
    loss_sum += LOSSMeter[i].avg

print("K-fold {} ACC : {:.6f}".format(K_FOLD, acc_sum / K_FOLD))
print("K-fold {} ACC : {:.6f}".format(K_FOLD, loss_sum / K_FOLD))



## === cell 16
import pandas as pd
import os
import glob

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)

nw = min(4, max(1, (os.cpu_count() or 4) // 2))
dl_kwargs = dict(
    batch_size=128,
    shuffle=False,
    num_workers=nw,
    pin_memory=True,
    persistent_workers=(nw > 0),
    prefetch_factor=2 if nw > 0 else None,
)
dl_kwargs = {k: v for k, v in dl_kwargs.items() if v is not None}
test_dataloader = DataLoader(test_dataset, **dl_kwargs)

weight_paths = []
if TRAINING:
    weight_paths = [p for p in best_ckpt_paths if os.path.exists(p)]
else:
    weight_globs = [
        "../input/densenet121-kfold5/*.pkl",
        "/kaggle/input/densenet121-kfold5/*.pkl",
    ]
    for g in weight_globs:
        weight_paths.extend(sorted(glob.glob(g)))
    weight_paths.extend([p for p in best_ckpt_paths if os.path.exists(p)])
    weight_paths = sorted(list(dict.fromkeys(weight_paths)))

use_folds = len(weight_paths) > 0
if not use_folds:
    print(
        "WARNING: No fold weights found; using torchvision ImageNet-pretrained DenseNet-121 "
        "with randomly initialized 5-class classifier. Submission will be produced but score may be limited."
    )

sub_df = pd.read_csv(
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)
expected_names = sub_df["image_id"].values
n_test = len(expected_names)
n_classes = 5


def _fast_load_weights(model: torch.nn.Module, params_path: str):
    params = torch.load(params_path, map_location="cpu")
    missing, unexpected = model.load_state_dict(params, strict=False)
    print("Loaded weights from:", os.path.basename(params_path))
    print("Missing keys:", len(missing), "Unexpected keys:", len(unexpected))
    return model


if use_folds:
    probs_sum = np.zeros((n_test, n_classes), dtype=np.float64)
    names_once = None

    for fold_i, params_path in enumerate(weight_paths):
        model = create_new_model(pretrained=True)
        model = _fast_load_weights(model, params_path)
        model.eval()

        offset = 0
        with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_amp_enabled):
            for img, img_name in test_dataloader:
                if names_once is None:
                    if offset == 0:
                        names_once = []
                    names_once.extend(list(img_name))

                bs = img.shape[0]
                b_img = img.to(device, non_blocking=True)
                logits = model(b_img)  # (B, C)
                probs = torch.softmax(logits, dim=1).to(torch.float64).cpu().numpy()
                probs_sum[offset : offset + bs] += probs
                offset += bs

        if offset != n_test:
            raise RuntimeError(
                f"Test dataloader yielded {offset} samples, expected {n_test}"
            )

    image_labels = probs_sum.argmax(axis=1).astype(np.int64)
    df = pd.DataFrame({"image_id": np.array(names_once), "label": image_labels})
else:
    model = create_new_model(pretrained=True)
    model.eval()
    names = []
    labels = []
    with torch.inference_mode(), torch.cuda.amp.autocast(enabled=_amp_enabled):
        for img, img_name in test_dataloader:
            b_img = img.to(device, non_blocking=True)
            logits = model(b_img)
            preds = torch.argmax(logits, dim=1).detach().cpu().numpy().astype(np.int64)
            names.extend(list(img_name))
            labels.extend(list(preds))
    df = pd.DataFrame({"image_id": names, "label": np.array(labels, dtype=np.int64)})

print(df.head())
df.to_csv("/kaggle/working/submission.csv", index=False)
print("Wrote /kaggle/working/submission.csv with shape:", df.shape)
