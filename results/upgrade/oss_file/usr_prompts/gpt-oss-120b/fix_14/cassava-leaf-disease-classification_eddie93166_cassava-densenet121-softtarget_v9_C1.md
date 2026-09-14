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
import os
import glob
import re
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.model_selection import KFold
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as cp
import torchvision.transforms as transforms
from torchvision import io as tv_io
from torch.utils.data import Dataset, DataLoader, Subset
from collections import OrderedDict
from PIL import Image  # kept for potential backward compatibility

torch.backends.cudnn.benchmark = True
torch.manual_seed(42)
np.random.seed(42)

torch.set_num_threads(os.cpu_count() or 1)

BATCH_SIZE = 64  # increased batch size to cut number of iterations
EPOCH = 5
WD = 1e-4
LR = 0.0001
VAL_RATIO = 0.2
PHASE = ["train", "val"]
BETA = 0.0
CUTMIX_PROB = 0.0
TRAINING = True
K_FOLD = 1  # reduced to a single split to cut training time

NUM_WORKERS = min(8, os.cpu_count() or 1)




## === cell 1
class CLD_Dataset(Dataset):
    def __init__(self, image_root, label_path=None, transform=None, return_name=False):
        super(CLD_Dataset, self).__init__()
        self.transform = transform
        self.image_paths = sorted(glob.glob(os.path.join(image_root, "*.jpg")))
        self.cached_images = [None] * len(self.image_paths)

        if not return_name:
            self.label = pd.read_csv(label_path, index_col="image_id")
        self.return_name = return_name

    def set_transform(self, transform):
        self.transform = transform

    def _load_image(self, idx):
        img = tv_io.read_image(self.image_paths[idx]).float() / 255.0
        self.cached_images[idx] = img
        return img

    def __getitem__(self, idx):
        img = self.cached_images[idx]
        if img is None:
            img = self._load_image(idx)

        if self.transform is not None:
            img = self.transform(img)

        if self.return_name:
            return img, os.path.basename(self.image_paths[idx])
        else:
            basename = os.path.basename(self.image_paths[idx])
            label = int(self.label.loc[basename].label)
            return img, label

    def __len__(self):
        return len(self.image_paths)




## === cell 2
train_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.RandomHorizontalFlip(),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

val_transform = transforms.Compose(
    [
        transforms.Resize((448, 448)),
        transforms.ConvertImageDtype(torch.float32),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

all_train_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/train_images",
    "/kaggle/input/cassava-leaf-disease-classification/train.csv",
    train_transform,
)

fold_dataloader = []
if K_FOLD != 1:
    kf = KFold(n_splits=K_FOLD, shuffle=True, random_state=42)
    for train_idx, val_idx in kf.split(range(len(all_train_dataset))):
        train_subset = Subset(all_train_dataset, train_idx)
        val_subset = Subset(all_train_dataset, val_idx)
        fold_dataloader.append(
            {
                "train": DataLoader(
                    train_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=True,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
                ),
                "val": DataLoader(
                    val_subset,
                    batch_size=BATCH_SIZE,
                    shuffle=False,
                    num_workers=NUM_WORKERS,
                    pin_memory=True,
                    persistent_workers=True,
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
                persistent_workers=True,
            ),
            "val": None,
        }
    )




## === cell 3
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
                num_input_features, bn_size * growth_rate, kernel_size=1, bias=False
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

    def forward(self, input):
        if isinstance(input, torch.Tensor):
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
    state_dict = torch.load(f"../../pretrain/{arch}.pth")
    for key in list(state_dict.keys()):
        m = pattern.match(key)
        if m:
            new_key = m.group(1) + m.group(2)
            state_dict[new_key] = state_dict[key]
            del state_dict[key]
    for name, param in state_dict.items():
        if name in model.state_dict():
            model.state_dict()[name].copy_(param)


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




## === cell 4
if torch.cuda.is_available():
    device = torch.device("cuda:0")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

from torchvision.models import densenet121 as tv_densenet121


def create_new_model(pretrained=True):
    """
    Create a DenseNet-121 model. If pretrained=True, load torchvision ImageNet pretrained
    weights and replace the classifier to output 5 classes.
    """
    if pretrained:
        model = tv_densenet121(pretrained=True)
    else:
        model = tv_densenet121(pretrained=False)
    model.classifier = nn.Linear(model.classifier.in_features, 5)
    model = torch.compile(model)
    return model.to(device)




## === cell 5
def create_loss_opti(model):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WD)
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=5, T_mult=1, eta_min=1e-6, last_epoch=-1
    )
    scaler = torch.cuda.amp.GradScaler()
    return criterion, optimizer, lr_scheduler, scaler




## === cell 6
def rand_bbox(size, lam):
    """Generate random bounding box for CutMix."""
    W, H = size[2], size[3]
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




## === cell 7
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

    def update(self, value, n):
        self.value = value
        if self.acc:
            self.sum += value
        else:
            self.sum += value * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


acc_meters = []
loss_meters = []
for _ in range(K_FOLD):
    acc_meters.append(AverageMeter(True))
    loss_meters.append(AverageMeter(False))

if TRAINING:
    for fold_index, dataloader_dict in enumerate(fold_dataloader):
        print(f"\n=== Training fold {fold_index + 1}/{K_FOLD} ===")
        model = create_new_model(pretrained=True)
        criterion, optimizer, lr_scheduler, scaler = create_loss_opti(model)

        best_acc = 0.0
        best_acc_meter = AverageMeter(True)
        best_loss_meter = AverageMeter(False)

        for epoch in range(1, EPOCH + 1):
            epoch_loss = 0.0
            epoch_correct = 0
            epoch_total = 0

            for phase in PHASE:
                loader = dataloader_dict.get(phase)
                if loader is None:
                    continue  # skip missing validation loader
                if phase == "train":
                    model.train()
                    all_train_dataset.set_transform(train_transform)
                else:
                    model.eval()
                    all_train_dataset.set_transform(val_transform)

                for images, labels in tqdm(
                    loader,
                    total=len(loader),
                    leave=False,
                    desc=f"Epoch {epoch} {phase}",
                ):
                    images = images.to(device, non_blocking=True)
                    labels = labels.to(device, non_blocking=True)

                    with torch.cuda.amp.autocast():
                        if BETA > 0 and np.random.rand() < CUTMIX_PROB:
                            lam = np.random.beta(BETA, BETA)
                            rand_index = torch.randperm(images.size(0)).to(device)
                            target_a = labels
                            target_b = labels[rand_index]
                            bbx1, bby1, bbx2, bby2 = rand_bbox(images.size(), lam)
                            images[:, :, bbx1:bbx2, bby1:bby2] = images[
                                rand_index, :, bbx1:bbx2, bby1:bby2
                            ]
                            lam = 1 - (
                                (bbx2 - bbx1)
                                * (bby2 - bby1)
                                / (images.size(-1) * images.size(-2))
                            )
                            outputs = model(images)
                            loss = criterion(outputs, target_a) * lam + criterion(
                                outputs, target_b
                            ) * (1.0 - lam)
                        else:
                            outputs = model(images)
                            loss = criterion(outputs, labels)

                    _, preds = torch.max(outputs, 1)
                    correct = torch.sum(preds == labels).item()

                    if phase == "train":
                        optimizer.zero_grad()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        best_acc_meter.update(correct, labels.size(0))
                        best_loss_meter.update(loss.item(), labels.size(0))

                    epoch_loss += loss.item() * labels.size(0)
                    epoch_correct += correct
                    epoch_total += labels.size(0)

            lr_scheduler.step()

            avg_loss = epoch_loss / epoch_total if epoch_total else 0
            avg_acc = epoch_correct / epoch_total if epoch_total else 0
            print(
                f"Fold {fold_index + 1} Epoch {epoch}/{EPOCH} "
                f"Loss: {avg_loss:.4f} Acc: {avg_acc:.4f}"
            )

        torch.save(
            model.state_dict(),
            f"./densenet121_fold{fold_index + 1}_best.pkl",
        )
        acc_meters[fold_index] = best_acc_meter
        loss_meters[fold_index] = best_loss_meter




## === cell 8
ckpt_paths = sorted(glob.glob("./densenet121_fold*_best.pkl"))
if not ckpt_paths:
    raise FileNotFoundError(
        "No trained model checkpoints found. Ensure training completed."
    )

models = []
for path in ckpt_paths:
    m = create_new_model(pretrained=False)
    m.load_state_dict(torch.load(path, map_location=device))
    m.eval()
    models.append(m)

test_dataset = CLD_Dataset(
    "/kaggle/input/cassava-leaf-disease-classification/test_images",
    transform=val_transform,
    return_name=True,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=1,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
    persistent_workers=True,
)

pred_ids = []
pred_labels = []

with torch.no_grad():
    for imgs, img_names in tqdm(test_loader, desc="Predicting"):
        imgs = imgs.to(device, non_blocking=True)
        summed_logits = None
        for m in models:
            logits = m(imgs)
            summed_logits = logits if summed_logits is None else summed_logits + logits
        _, preds = torch.max(summed_logits, 1)
        pred_ids.append(img_names[0])
        pred_labels.append(int(preds.item()))

submission = pd.DataFrame({"image_id": pred_ids, "label": pred_labels})
submission_path = "/kaggle/working/submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}")
