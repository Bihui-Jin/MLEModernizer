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

No external packages required in the script and installed.

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
import math
import random
from copy import deepcopy

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import torch
from torch.utils.data import Dataset, DataLoader, IterableDataset
from torchvision import transforms
from PIL import Image

SEED = 42
os.environ["PYTHONHASHSEED"] = str(SEED)
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

torch.backends.cudnn.deterministic = True  # keep determinism
torch.backends.cudnn.benchmark = (
    True  # faster for fixed 512x512 convs; deterministic True keeps determinism
)

is_submission = False

data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
train_batch_size, val_batch_size, sub_batch_size = 4, 8, 8
train_rate = 0.7  # rate for train dataset
num_workers = 2
mean, std = 0.5, 0.5

use_balanced_sample = False
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([3.0, 1.5, 1.0, 1.0, 4.5])
learning_rate = 0.001
weight_decay = 0.00002
efficient_net_version = 3  # (1.4, 1.2)

train_epoch = (0, 20)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = (
    "../input/cassava-leaf-disease-classification-weight/weight.pth"
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
PIN_MEMORY = torch.cuda.is_available()
PERSISTENT_WORKERS = num_workers > 0
PREFETCH_FACTOR = 4 if num_workers > 0 else None


def seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


dl_generator = torch.Generator()
dl_generator.manual_seed(SEED)




## === cell 1
class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): csv file path
            root_dir (string): directory path of exist all image
            transform (callable, optional): Optional transform for sample
        """
        self.cassava_leaf_disease = pd.read_csv(csv_file).reset_index(drop=True)
        self.root_dir = root_dir
        self.transform = transform
        self.train = train
        self.folder = "train_images" if train else "test_images"

        self._image_ids = self.cassava_leaf_disease["image_id"].to_numpy()
        self._has_label = self.train and ("label" in self.cassava_leaf_disease.columns)
        self._labels = (
            self.cassava_leaf_disease["label"].to_numpy(dtype=np.int64)
            if self._has_label
            else None
        )

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = "/".join([self.root_dir, self.folder, self._image_ids[idx]])
        img = Image.open(img_name).convert("RGB")

        if self.transform:
            img = self.transform(img)

        label = int(self._labels[idx]) if self._has_label else 0
        sample = (img, label)
        return sample


def split_dataset(dataset, rate):
    train_length = int(len(dataset) * rate)
    val_length = len(dataset) - train_length
    g = torch.Generator()
    g.manual_seed(SEED)
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_length, val_length], generator=g
    )
    return train_set, val_set




## === cell 2
def _list_tfrec_files(root, train=True):
    folder = "train_tfrecords" if train else "test_tfrecords"
    p = os.path.join(root, folder)
    if not os.path.isdir(p):
        return []
    files = [os.path.join(p, f) for f in os.listdir(p) if f.endswith(".tfrec")]
    return sorted(files)


_HAS_TF = False


class CassavaTFRecordIterable(IterableDataset):
    def __init__(self, tfrec_files, batch_size, is_train, mean, std, seed=42):
        super().__init__()
        self.tfrec_files = list(tfrec_files)
        self.batch_size = int(batch_size)
        self.is_train = bool(is_train)
        self.mean = float(mean)
        self.std = float(std)
        self.seed = int(seed)
        raise RuntimeError(
            "TensorFlow/TFRecord pipeline disabled in this notebook to avoid runtime crashes."
        )

    def __iter__(self):
        return iter(())


try:
    from torchvision.transforms import v2 as T

    _HAS_V2 = True
except Exception:
    _HAS_V2 = False

if _HAS_V2:
    if is_submission:
        transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.RandomRotation(30),
                T.RandomHorizontalFlip(),
                T.RandomVerticalFlip(),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
else:
    if is_submission:
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )

dataset = CassavaLeafDiseaseDataset(
    csv_file="/".join([data_path, csv_file_name]),
    root_dir=data_path,
    transform=transform,
    train=not is_submission,
)

if is_submission:
    sub_loader = DataLoader(
        dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )
else:
    train_set, val_set = split_dataset(dataset, train_rate)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )



## === cell 3
if (not is_submission) and use_class_weight:

    def class_weight(label_count):
        total = sum(label_count)
        weight = [total / (2 * max(1, count)) for count in label_count]
        return torch.tensor(weight, dtype=torch.float32)

    vc = dataset.cassava_leaf_disease["label"].value_counts().to_dict()
    label_count = [int(vc.get(i, 0)) for i in range(5)]
    print("Label counts:", label_count)

    loss_weight = class_weight(label_count)
    loss_weight *= mul_weight

    if torch.cuda.is_available():
        loss_weight = loss_weight.cuda()
        print("Loss weight (cuda):", loss_weight)




## === cell 4
def progress_print(comment, current, total, num_of_print=50, slow=5):
    end = "\r" if current < total else "\n"
    symbol = ["\\", "/", "-"]
    done = int((current - 1) / total * num_of_print) if total > 0 else 0
    doing = symbol[(current % (len(symbol) * slow) // slow)] if current < total else ""
    yet = max(0, num_of_print - 1 - done)
    print(comment + ": " + "#" * done + doing + "*" * yet, end=end)


def num_of_correct(outputs, labels):
    _, predicted = torch.max(outputs, 1)
    c = (predicted == labels).squeeze()
    return c.sum().item()


def train(
    model=None,
    criterion=None,
    optimizer=None,
    data_loader=None,
    debug=True,
    epoch=0,
    debug_rate=0.01,
):
    model.train()
    try:
        data_loader_length = len(data_loader)
    except Exception:
        data_loader_length = -1

    correct, total = 0, 0
    running_loss = 0.0

    for i, data in enumerate(data_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        optimizer.zero_grad(set_to_none=True)

        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        correct += num_of_correct(outputs, labels)
        total += len(inputs)
        running_loss += loss.item() * len(inputs)

    if debug:
        print(
            "epoch: %d (%s/%s),\tloss: %.3f"
            % (
                epoch + 1,
                str(data_loader_length),
                str(data_loader_length),
                running_loss / max(1, total),
            )
        )

    return correct / total, running_loss / total


def valid(model=None, data_loader=None, debug=True):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            outputs = model(inputs)

            correct += num_of_correct(outputs, labels)
            total += len(inputs)
    if debug:
        try:
            n_batches = len(data_loader)
        except Exception:
            n_batches = -1
        print("validation done (%d batches)" % n_batches)
    return correct / total


def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    cm = torch.zeros((num_classes, num_classes), dtype=torch.int64)
    with torch.no_grad():
        for inputs, labels in data_loader:
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True)
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1).detach().cpu()
            labels = labels.detach().cpu()

            idx = predicted * num_classes + labels
            cm += torch.bincount(idx, minlength=num_classes * num_classes).view(
                num_classes, num_classes
            )
    return cm.numpy().astype(np.int32)




## === cell 5
class ConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        batch_norm=True,
        activation=True,
    ):
        super().__init__()
        modules = [
            torch.nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                padding=padding,
                stride=stride,
                groups=groups,
                bias=not batch_norm,
            )
        ]
        if batch_norm:
            modules.append(torch.nn.BatchNorm2d(out_channels))
        if activation:
            modules.append(torch.nn.LeakyReLU())
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


def DConvUnit(in_channels: int, kernel_size: int, stride: int = 1) -> torch.nn.Module:
    padding = kernel_size // 2
    return ConvUnit(in_channels, in_channels, kernel_size, padding, stride, in_channels)


def EConvUnit(in_channels: int, factor: int = 6) -> torch.nn.Module:
    return ConvUnit(in_channels, factor * in_channels, 1)


def PConvUnit(in_channels, out_channels, activation=False) -> torch.nn.Module:
    return ConvUnit(in_channels, out_channels, 1, activation=activation)


class DSConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation=True,
    ):
        super().__init__()
        modules = [
            DConvUnit(in_channels, kernel_size, stride),
            PConvUnit(in_channels, out_channels, activation),
        ]
        self.sequence = torch.nn.Sequential(*modules)
        self.out_channels = out_channels

    def forward(self, x):
        x = self.sequence(x)
        return x


class SEUnit(torch.nn.Module):
    def __init__(self, in_channels: int, se_ratio: float):
        super().__init__()
        se_channels = max(1, int(in_channels * se_ratio))
        self.sequence = torch.nn.Sequential(
            torch.nn.AdaptiveAvgPool2d((1, 1)),
            ConvUnit(in_channels, se_channels, 1, batch_norm=False),
            ConvUnit(se_channels, in_channels, 1, batch_norm=False, activation=False),
            torch.nn.Sigmoid(),
        )
        self.out_channels = in_channels

    def forward(self, x):
        y = self.sequence(x)
        z = x * y
        return z


class BottleneckUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        factor: int = 6,
        kernel_size: int = 3,
        stride: int = 1,
        has_se: bool = True,
        se_ratio: float = 0.25,
    ):
        super().__init__()
        modules = []
        self.residual = stride == 1 and in_channels == out_channels
        hid_channels = factor * in_channels
        if factor > 1:
            modules.append(EConvUnit(in_channels, factor))
        modules.append(DConvUnit(hid_channels, kernel_size, stride))
        if has_se:
            calced_se_ratio = se_ratio / factor
            modules.append(SEUnit(hid_channels, calced_se_ratio))
        modules.append(PConvUnit(hid_channels, out_channels, False))
        self.sequence = torch.nn.Sequential(*modules)
        self.activation = torch.nn.LeakyReLU()
        self.out_channels = in_channels

    def forward(self, x):
        y = self.sequence(x)
        z = self.activation(x + y if self.residual else y)
        return z




## === cell 6
def round_filters(filters, coef: float, divisor: int) -> int:
    filters *= coef
    filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    return filters


def round_repeats(repeats, coef):
    repeats = int(math.ceil(repeats * coef)) if repeats > 0 else -repeats
    return repeats


class BlockArgs:
    def __init__(
        self, module: torch.nn.Module, filters: int, num_repeats: int = 1, **kwargs
    ):
        self.module = module
        self.filters = filters
        self.num_repeats = num_repeats
        self.kwargs = kwargs

    def __call__(
        self,
        in_channels: int,
        width_coef: float = 1.0,
        depth_coef: float = 1.0,
        divisor: int = 8,
    ) -> torch.nn.Module:
        out_channels = round_filters(self.filters, width_coef, divisor)
        num_repeats = round_repeats(self.num_repeats, depth_coef)
        modules = []
        kwargs_local = dict(self.kwargs)
        for _ in range(num_repeats):
            modules.append(self.module(in_channels, out_channels, **kwargs_local))
            if "stride" in kwargs_local:
                del kwargs_local["stride"]
            in_channels = out_channels
        return torch.nn.Sequential(*modules), in_channels


EFFICIENT_NET_BLOCK_ARGS = [
    BlockArgs(ConvUnit, 32, -1, kernel_size=3, stride=2, padding=1),
    BlockArgs(BottleneckUnit, 16, 1, factor=1),
    BlockArgs(BottleneckUnit, 24, 2, stride=2),
    BlockArgs(BottleneckUnit, 40, 2, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 80, 3, stride=2),
    BlockArgs(BottleneckUnit, 112, 3, kernel_size=5),
    BlockArgs(BottleneckUnit, 192, 4, kernel_size=5, stride=2),
    BlockArgs(BottleneckUnit, 320, 1),
]

EFFICIENT_NET_COMPOUND_COEF = [
    (1.0, 1.0),
    (1.1, 1.0),
    (1.2, 1.1),
    (1.4, 1.2),
    (1.8, 1.4),
    (2.2, 1.6),
    (2.6, 1.8),
    (3.1, 2.0),
]


class EfficientNet(torch.nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        num_of_class: int = 1000,
        depth_coef: float = 1.0,
        width_coef: float = 1.0,
        hidden_channels: int = 1280,
        block_args=EFFICIENT_NET_BLOCK_ARGS,
    ):
        super().__init__()
        modules = []
        self.hidden_channels = round_filters(hidden_channels, width_coef, 8)
        for block_arg in block_args:
            module, in_channels = block_arg(in_channels, width_coef, depth_coef)
            modules.append(module)
        modules.append(PConvUnit(in_channels, self.hidden_channels, True))
        modules.append(torch.nn.AdaptiveAvgPool2d((1, 1)))
        self.sequence = torch.nn.Sequential(*modules)
        self.linear = torch.nn.Linear(self.hidden_channels, num_of_class)

    def forward(self, x):
        x = self.sequence(x)
        x = x.view(-1, self.hidden_channels)
        x = self.linear(x)
        return x




## === cell 7
model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])

if use_pre_trained_weight and os.path.exists(pre_trained_weight_path):
    pre_trained_weight = torch.load(pre_trained_weight_path, map_location="cpu")
    model.load_state_dict(pre_trained_weight)

model = model.to(DEVICE)

try:
    if hasattr(torch, "compile"):
        model = torch.compile(model, mode="reduce-overhead")
except Exception:
    pass



## === cell 8
if not is_submission:
    if use_class_weight:
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif mul_weight is not None:
        mul_weight = mul_weight.to(DEVICE) if torch.cuda.is_available() else mul_weight
        criterion = torch.nn.CrossEntropyLoss(weight=mul_weight)
    else:
        criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    train_accuracy = []
    train_losses = []
    val_accuracy = []
    best_accuracy = 0.0 if len(val_accuracy) == 0 else min(val_accuracy)
    best_model = deepcopy(model.state_dict())



## === cell 9
if not is_submission:
    for epoch in range(*train_epoch):
        train_acc, train_loss = train(
            model,
            criterion,
            optimizer,
            train_loader,
            epoch=epoch,
            debug=debug,
            debug_rate=0.1,
        )
        val_acc = valid(model, val_loader, debug=debug)
        train_accuracy.append(train_acc)
        train_losses.append(train_loss)
        val_accuracy.append(val_acc)
        if best_accuracy < val_acc:
            best_model = deepcopy(model.state_dict())
            best_accuracy = val_acc
            print("Best model is updated.")
        print(train_acc, val_acc)
    torch.save(best_model, "weight.pth")



## === cell 10
if not is_submission and debug:
    print(train_accuracy)
    print(train_losses)
    print(val_accuracy)



## === cell 11
if not is_submission and debug:
    plt.plot(train_accuracy, label="train")
    plt.plot(train_losses, label="loss")
    plt.plot(val_accuracy, label="validation")
    plt.legend(loc="upper left")
    plt.show()




## === cell 12
def show_confusion_matrix(confusion_matrix):
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    row_sums = diagonal_confusion_matrix.sum(axis=1)
    normalized_diagonal_confusion_matrix = (
        diagonal_confusion_matrix / row_sums[:, np.newaxis]
    )
    plt.figure(figsize=(2, 1))
    plt.matshow(normalized_confusion_matrix, cmap="gray")
    plt.matshow(normalized_diagonal_confusion_matrix, cmap="gray")
    plt.show()
    print(confusion_matrix)
    print(normalized_confusion_matrix)




## === cell 13
if not is_submission and debug:
    confusion_matrix = get_confusion_matrix(model, val_loader, 5)
    show_confusion_matrix(confusion_matrix)



## === cell 14
if not is_submission:
    if os.path.exists("weight.pth"):
        model.load_state_dict(torch.load("weight.pth", map_location="cpu"))
        model = model.to(DEVICE)

    if _HAS_V2:
        test_transform = T.Compose(
            [
                T.Resize((512, 512), antialias=True),
                T.ToImage(),
                T.ToDtype(torch.float32, scale=True),
                T.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
    else:
        test_transform = transforms.Compose(
            [
                transforms.Resize((512, 512)),
                transforms.ToTensor(),
                transforms.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )

    test_dataset = CassavaLeafDiseaseDataset(
        csv_file="/".join([data_path, "sample_submission.csv"]),
        root_dir=data_path,
        transform=test_transform,
        train=False,
    )

    sub_loader = DataLoader(
        test_dataset,
        batch_size=sub_batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=PIN_MEMORY,
        persistent_workers=PERSISTENT_WORKERS,
        prefetch_factor=PREFETCH_FACTOR,
        worker_init_fn=seed_worker,
        generator=dl_generator,
    )

    model.eval()
    submission = test_dataset.cassava_leaf_disease.copy()
    predicts = []
    with torch.no_grad():
        for i, data in enumerate(sub_loader):
            inputs, _ = data
            inputs = inputs.to(DEVICE, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicts.extend(predicted.detach().cpu().numpy().tolist())

    assert len(predicts) == len(
        submission
    ), f"Pred length {len(predicts)} != submission length {len(submission)}"

    submission["label"] = predicts
    submission[["image_id", "label"]].to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)
