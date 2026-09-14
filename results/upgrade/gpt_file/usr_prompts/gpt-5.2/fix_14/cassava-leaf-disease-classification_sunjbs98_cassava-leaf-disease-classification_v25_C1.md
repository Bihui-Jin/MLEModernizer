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

# 5. Target score

0.7796917497733454

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'I make the code reliably produce a valid `submission.csv` by fixing two execution blockers in submission mode: loading a missing pretrained weight file and using `skimage`/`torchvision` transforms in an incompatible order. I keep your model and inference logic the same, but switch image loading to PIL so `Resize->ToTensor->Normalize` works as intended and matches typical EfficientNet preprocessing behavior. I also make the pretrained weight loading conditional on the file actually existing, so the script runs end-to-end in the provided environment. These minimal fixes should move your score upward from “not yielded” to a real accuracy score, aiming toward your target.'
- What this solution (achieved 0.11024) has done: 'Your low score (0.10762) is consistent with running inference using a randomly initialized model because the pretrained `weight.pth` file is not available at the configured path, so the model outputs near-random labels. To move the score up toward your target with minimal changes and without altering the model architecture/training semantics, I (1) automatically train the same model when the pretrained weight file is missing, (2) save/load that weight locally, and (3) fix the dataset CSV choice in submission mode so it predicts for the real test set and aligns rows to `sample_submission.csv`. These changes keep your core model and training loop intact, but ensure you’re not submitting random predictions and that the submission format matches Kaggle expectations. The script still always write a valid `submission.csv` end-to-end.'
- What this solution (achieved 0.11024) has done: 'The timeout is most likely caused by expensive per-image CPU transforms (512×512 resize with antialias) and suboptimal input pipeline settings; training would be far over budget, but in submission mode you only need fast inference. I keep the exact model and prediction logic, but optimize data loading and transforms by moving resize+normalize into a single TorchVision v2 pipeline (fewer Python calls), using fast decode-to-RGB with PIL, and adding inference-only speedups (channels_last + inference_mode). I also avoid building the train/val loaders when `is_submission=True` (they aren’t needed for inference when weights exist), and I make the confusion-matrix computation vectorized (harmless even if not executed in submission). These changes preserve evaluation semantics (same inputs/outputs) with only negligible floating-point differences.'
- What this solution (achieved 0.11584) has done: 'The timeout is almost certainly coming from the training path: EfficientNet at 512×512 with batch size 4 over up to 60 epochs is far too slow on CPU and often still too slow on GPU under a 600s cap. The safest speed fix that preserves core logic and accuracy semantics is to ensure the “submission” run never trains: hard-fail if pretrained weights are missing, instead of silently training. Additionally, reduce input pipeline overhead by (1) avoiding TensorFlow TFRecord decoding (which adds a second framework and CPU cost) and (2) enabling faster DataLoader behavior for inference (larger batches, no persistent workers needed for a short inference). All changes keep the same model, transforms, and prediction logic (argmax over logits), just remove the accidental slow path and streamline I/O.'
- What this solution (achieved 0.11024) has done: 'I remove the hard failure when pretrained weights are missing and instead trigger training in that case, because your current 0.11584 score indicates you are effectively predicting with untrained/random weights. To keep your core model and training loop unchanged but make it fit the 600s budget, I switch the input pipeline to use the provided TFRecords (fast decoding) for training/validation/test and keep the same 512×512 normalization semantics. I also fix the CUDA dtype/device mismatch during training by ensuring the model is moved to GPU *after* any weight loading and before the optimizer is created. Finally, I make submission generation always align to `sample_submission.csv` row order and write a valid `submission.csv`.'

# 9. Code solution

## === cell 0
import os
import math
from copy import deepcopy

import torch
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import functional as TF

from PIL import Image

torch.manual_seed(42)
np.random.seed(42)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(42)
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = (
        False  # benchmark requires this False for speed; seeds keep stability.
    )

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    from torchvision.transforms import v2 as transforms_v2
except Exception:
    transforms_v2 = None

try:
    import torchvision.io as tvio
except Exception:
    tvio = None

tf = None




## === cell 1
is_submission = True
data_path = "/kaggle/input/cassava-leaf-disease-classification"

train_csv_path = "/".join([data_path, "train.csv"])
sample_sub_path = "/".join([data_path, "sample_submission.csv"])

train_batch_size, val_batch_size, sub_batch_size = 4, 8, 8
train_rate = 0.7  # rate for train dataset

num_workers = min(8, os.cpu_count() or 2)
mean, std = 0.5, 0.5

use_balanced_sample = True
use_class_weight = not use_balanced_sample
mul_weight = torch.tensor([4.0, 2.0, 1.0, 1.0, 4.5])
learning_rate = 0.001
weight_decay = 0.00002
efficient_net_version = 3  # (1.4, 1.2)

train_epoch = (40, 60)
debug = not is_submission

use_pre_trained_weight = True
pre_trained_weight_path = (
    "../input/cassava-leaf-disease-classification-weight/weight.pth"
)

local_weight_path = "weight.pth"




## === cell 2
class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        """
        Args:
            csv_file (string): csv file path
            root_dir (string): directory path of exist all image
            transform (callable, optional): Optional transform for sample
        """
        self.cassava_leaf_disease = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folder = "train_images" if train else "test_images"
        self.train = train

        self._image_ids = self.cassava_leaf_disease["image_id"].values
        self._has_label = "label" in self.cassava_leaf_disease.columns
        self._labels = (
            self.cassava_leaf_disease["label"].values.astype(np.int64)
            if self._has_label
            else None
        )
        folder_root = os.path.join(self.root_dir, self.folder)
        self._img_paths = np.array(
            [os.path.join(folder_root, fn) for fn in self._image_ids], dtype=object
        )

    def __len__(self):
        return len(self._image_ids)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_path = self._img_paths[idx]

        if tvio is not None:
            img = tvio.read_image(img_path, mode=tvio.ImageReadMode.RGB)  # uint8 CHW
        else:
            im = Image.open(img_path).convert("RGB")
            img = TF.pil_to_tensor(im)  # uint8, 3xHxW

        if self.transform:
            img = self.transform(img)

        label = int(self._labels[idx]) if self._has_label else 0
        sample = (img, label)
        return sample


def split_dataset(dataset, rate):
    train_length = int(len(dataset) * rate)
    val_length = len(dataset) - train_length
    train_set, val_set = torch.utils.data.random_split(
        dataset, [train_length, val_length]
    )
    return train_set, val_set




## === cell 3
def make_transform(is_train: bool, do_aug: bool):
    if transforms_v2 is not None:
        ops = [transforms_v2.Resize((512, 512), antialias=True)]
        if do_aug:
            ops.extend(
                [
                    transforms_v2.RandomRotation(30),
                    transforms_v2.RandomHorizontalFlip(),
                    transforms_v2.RandomVerticalFlip(),
                ]
            )
        ops.extend(
            [
                transforms_v2.ToDtype(torch.float32, scale=True),  # /255.0
                transforms_v2.Normalize((mean, mean, mean), (std, std, std)),
            ]
        )
        return transforms_v2.Compose(ops)
    else:
        from torchvision import transforms

        ops = []
        ops.append(
            transforms.Lambda(lambda x: TF.resize(x, (512, 512), antialias=True))
        )
        if do_aug:
            ops.extend(
                [
                    transforms.RandomRotation(30),
                    transforms.RandomHorizontalFlip(),
                    transforms.RandomVerticalFlip(),
                ]
            )
        ops.append(transforms.Lambda(lambda x: x.float().div(255.0)))
        ops.append(transforms.Normalize((mean, mean, mean), (std, std, std)))
        return transforms.Compose(ops)


def _seed_worker(worker_id: int):
    base_seed = 42
    seed = base_seed + worker_id
    np.random.seed(seed)
    torch.manual_seed(seed)


_dl_generator = torch.Generator()
_dl_generator.manual_seed(42)

_persistent_workers = num_workers > 0

_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
    persistent_workers=_persistent_workers,
    prefetch_factor=(8 if num_workers > 0 else None),
    worker_init_fn=_seed_worker if num_workers > 0 else None,
)
_loader_kwargs = {k: v for k, v in _loader_kwargs.items() if v is not None}

transform_sub = make_transform(is_train=False, do_aug=False)
sub_dataset = CassavaLeafDiseaseDataset(
    csv_file=sample_sub_path,
    root_dir=data_path,
    transform=transform_sub,
    train=False,
)
sub_loader = DataLoader(
    sub_dataset,
    batch_size=sub_batch_size,
    shuffle=False,
    generator=_dl_generator,
    **_loader_kwargs,
)

train_loader = None
val_loader = None
train_dataset_full = None
dataset = None




## === cell 4
if use_class_weight:

    def class_weight(label_count):
        total = sum(label_count)
        weight = [total / (2 * count) for count in label_count]
        return torch.tensor(weight)




## === cell 5
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
    data_loader_length = len(data_loader)
    correct, total = 0, 0
    running_loss = 0.0
    running_debug_rate = debug_rate
    running_debug_step = max(1, int(data_loader_length * running_debug_rate))
    for i, data in enumerate(data_loader):
        inputs, labels = data
        if torch.cuda.is_available():
            inputs = inputs.cuda(non_blocking=True).to(
                memory_format=torch.channels_last
            )
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
            progress_print("train", i, running_debug_step, slow=3)
        if debug and i == running_debug_step - 1:
            print(
                "epoch: %d (%d/%d),\tloss: %.3f"
                % (epoch + 1, i + 1, data_loader_length, running_loss / total)
            )
            running_debug_rate += debug_rate
            running_debug_step = min(
                data_loader_length, max(1, int(data_loader_length * running_debug_rate))
            )
    return correct / total, running_loss / total


def valid(model=None, data_loader=None, debug=True):
    model.eval()
    correct, total = 0, 0
    with torch.inference_mode():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True).to(
                    memory_format=torch.channels_last
                )
                labels = labels.cuda(non_blocking=True)
            outputs = model(inputs)

            correct += num_of_correct(outputs, labels)
            total += len(inputs)
            if debug:
                progress_print("validation", i, len(data_loader), slow=3)
    return correct / total


def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    with torch.inference_mode():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            if torch.cuda.is_available():
                inputs = inputs.cuda(non_blocking=True).to(
                    memory_format=torch.channels_last
                )
            outputs = model(inputs)
            predicted = outputs.argmax(dim=1).detach().cpu().to(torch.int64)
            labels = labels.detach().cpu().to(torch.int64)
            idx = (predicted * num_classes + labels).numpy()
            binc = np.bincount(idx, minlength=num_classes * num_classes).reshape(
                num_classes, num_classes
            )
            confusion_matrix += binc.astype(np.int32)

            progress_print("confusion_matrix", i, len(data_loader), slow=3)
    return confusion_matrix




## === cell 6
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




## === cell 7
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
        for _ in range(num_repeats):
            modules.append(self.module(in_channels, out_channels, **self.kwargs))
            if "stride" in self.kwargs:
                del self.kwargs["stride"]
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




## === cell 8
model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])

loaded_weight = False
candidate_weight_paths = [
    pre_trained_weight_path,
    local_weight_path,
    "/kaggle/input/cassava-leaf-disease-classification-weight/weight.pth",
    "/kaggle/input/cassava-leaf-disease-classification-weight/weights.pth",
]
for wp in candidate_weight_paths:
    if wp and os.path.exists(wp):
        state = torch.load(wp, map_location="cpu")
        model.load_state_dict(state)
        loaded_weight = True
        print(f"Loaded weight from: {wp}")
        break

if is_submission and (not loaded_weight):
    raise RuntimeError(
        "Submission mode is enabled but no pretrained weight file was found. "
        "Training would exceed the 600s timeout; please attach the weight dataset or "
        "place weight.pth in the working directory."
    )

need_train = not loaded_weight

if torch.cuda.is_available():
    model = model.cuda()
    model = model.to(memory_format=torch.channels_last)

_can_compile = hasattr(torch, "compile")
if _can_compile:
    try:
        model = torch.compile(model, mode="reduce-overhead")
    except Exception:
        pass




## --- ERROR in cell 8, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/4265167794.py in <cell line: 0>()
     19 # Correctness: preserves intended submission semantics (inference with pretrained weights).
     20 if is_submission and (not loaded_weight):
---> 21     raise RuntimeError(
     22         "Submission mode is enabled but no pretrained weight file was found. "
     23         "Training would exceed the 600s timeout; please attach the weight dataset or "

RuntimeError: Submission mode is enabled but no pretrained weight file was found. Training would exceed the 600s timeout; please attach the weight dataset or place weight.pth in the working directory.

## === cell 9
def _list_tfrec_files(split: str):
    tfrec_dir = os.path.join(data_path, f"{split}_tfrecords")
    if not os.path.isdir(tfrec_dir):
        return []
    return sorted(
        os.path.join(tfrec_dir, f)
        for f in os.listdir(tfrec_dir)
        if f.endswith(".tfrec")
    )


def _build_tf_dataset(tfrec_files, batch_size: int, training: bool):
    return None


class _TorchLoaderFromTFDS:
    def __init__(self, tf_ds, length_hint: int):
        self.tf_ds = tf_ds
        self.length_hint = max(1, int(length_hint))

    def __iter__(self):
        for imgs, labels in self.tf_ds:
            imgs_np = imgs.numpy().astype(np.float32)
            labels_np = labels.numpy().astype(np.int64)
            yield torch.from_numpy(imgs_np), torch.from_numpy(labels_np)

    def __len__(self):
        return self.length_hint


if need_train:
    train_transform = make_transform(is_train=True, do_aug=True)
    train_dataset_full = CassavaLeafDiseaseDataset(
        csv_file=train_csv_path,
        root_dir=data_path,
        transform=train_transform,
        train=True,
    )
    dataset = train_dataset_full  # for compatibility with existing weighting logic

    if use_balanced_sample:
        labels_np = train_dataset_full._labels
        indices_for_each_label = [
            np.where(labels_np == i)[0].tolist() for i in range(5)
        ]

        dataset_for_each_label = [
            torch.utils.data.dataset.Subset(train_dataset_full, indices)
            for indices in indices_for_each_label
        ]

        if len(dataset_for_each_label[3]) > 2500:
            dataset_for_each_label[3], _ = torch.utils.data.random_split(
                dataset_for_each_label[3],
                [2500, len(dataset_for_each_label[3]) - 2500],
                generator=_dl_generator,
            )

        train_set_for_each_label = []
        val_set_for_each_label = []
        for sub_dataset_ in dataset_for_each_label:
            sub_train_set, sub_val_set = split_dataset(sub_dataset_, train_rate)
            train_set_for_each_label.append(sub_train_set)
            val_set_for_each_label.append(sub_val_set)

        train_set_for_each_label[0] = torch.utils.data.ConcatDataset(
            [train_set_for_each_label[0], train_set_for_each_label[0]]
        )
        val_set_for_each_label[0] = torch.utils.data.ConcatDataset(
            [val_set_for_each_label[0], val_set_for_each_label[0]]
        )

        train_set = torch.utils.data.ConcatDataset(train_set_for_each_label)
        val_set = torch.utils.data.ConcatDataset(val_set_for_each_label)
    else:
        train_set, val_set = split_dataset(train_dataset_full, train_rate)

    train_loader = DataLoader(
        train_set,
        batch_size=train_batch_size,
        shuffle=True,
        generator=_dl_generator,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_set,
        batch_size=val_batch_size,
        shuffle=False,
        generator=_dl_generator,
        **_loader_kwargs,
    )




## --- ERROR in cell 9, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2792719899.py in <cell line: 0>()
     29 
     30 
---> 31 if need_train:
     32     train_transform = make_transform(is_train=True, do_aug=True)
     33     train_dataset_full = CassavaLeafDiseaseDataset(

NameError: name 'need_train' is not defined

## === cell 10
if need_train:
    if use_class_weight:
        _dataset_for_weight = train_dataset_full
        label_count = np.bincount(_dataset_for_weight._labels, minlength=5).tolist()
        print(label_count)

        loss_weight = class_weight(label_count)
        loss_weight *= mul_weight
        if torch.cuda.is_available():
            loss_weight = loss_weight.cuda()
            print(loss_weight)
        criterion = torch.nn.CrossEntropyLoss(weight=loss_weight)
    elif mul_weight is not None:
        if torch.cuda.is_available():
            mul_weight = mul_weight.cuda()
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

    best_model_path = "best_weight_tmp.pth"
    if os.path.exists(best_model_path):
        try:
            os.remove(best_model_path)
        except Exception:
            pass




## --- ERROR in cell 10, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2831885619.py in <cell line: 0>()
----> 1 if need_train:
      2     if use_class_weight:
      3         _dataset_for_weight = train_dataset_full
      4         label_count = np.bincount(_dataset_for_weight._labels, minlength=5).tolist()
      5         print(label_count)

NameError: name 'need_train' is not defined

## === cell 11
if need_train:
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
            torch.save(model.state_dict(), best_model_path)
            best_accuracy = val_acc
            print("Best model is updated.")
        print(train_acc, val_acc)

    if os.path.exists(best_model_path):
        best_state = torch.load(best_model_path, map_location="cpu")
        model.load_state_dict(best_state)

    torch.save(model.state_dict(), local_weight_path)
    print(f"Saved trained weight to: {local_weight_path}")




## --- ERROR in cell 11, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1206873429.py in <cell line: 0>()
----> 1 if need_train:
      2     for epoch in range(*train_epoch):
      3         train_acc, train_loss = train(
      4             model,
      5             criterion,

NameError: name 'need_train' is not defined

## === cell 12
if (not is_submission) and (not need_train):
    pass




## === cell 13
if not is_submission:
    print(train_accuracy)
    print(train_losses)
    print(val_accuracy)




## === cell 14
if not is_submission:
    plt.plot(train_accuracy, label="train")
    plt.plot(train_losses, label="loss")
    plt.plot(val_accuracy, label="validation")
    plt.legend(loc="upper left")
    plt.show()




## === cell 15
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




## === cell 16
if not is_submission and (val_loader is not None):
    confusion_matrix = get_confusion_matrix(model, val_loader, 5)
    show_confusion_matrix(confusion_matrix)




## === cell 17
def _iter_test_batches_from_tfrecords(batch_size: int):
    return None




## === cell 18
if is_submission:
    model.eval()

    submission = pd.read_csv(sample_sub_path).copy()

    infer_batch_size = max(sub_batch_size, 64 if torch.cuda.is_available() else 16)

    if infer_batch_size != sub_loader.batch_size:
        sub_loader = DataLoader(
            sub_dataset,
            batch_size=infer_batch_size,
            shuffle=False,
            generator=_dl_generator,
            **_loader_kwargs,
        )

    image_ids = submission["image_id"].values
    predicts = np.empty(len(submission), dtype=np.int64)

    used_tfrecord = False
    tfrec_iter = _iter_test_batches_from_tfrecords(infer_batch_size)
    if tfrec_iter is not None:
        used_tfrecord = True
        name_to_index = {n.encode(): i for i, n in enumerate(image_ids)}
        with torch.inference_mode():
            for imgs_np, names_np in tfrec_iter:
                inputs = torch.from_numpy(imgs_np)
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                outputs = model(inputs)
                pred = outputs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)

                for j, nm in enumerate(names_np.tolist()):
                    predicts[name_to_index[nm]] = pred[j]

    if not used_tfrecord:
        write_pos = 0
        with torch.inference_mode():
            for inputs, _ in sub_loader:  # labels are dummy in sample_submission.csv
                if torch.cuda.is_available():
                    inputs = inputs.cuda(non_blocking=True).to(
                        memory_format=torch.channels_last
                    )
                outputs = model(inputs)
                pred = outputs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
                predicts[write_pos : write_pos + pred.shape[0]] = pred
                write_pos += pred.shape[0]

    if len(predicts) != len(submission):
        raise RuntimeError(
            f"Prediction length mismatch: got {len(predicts)} preds but submission has {len(submission)} rows."
        )

    submission["label"] = predicts.tolist()
    submission.to_csv("submission.csv", index=False)
    print("Wrote submission.csv with shape:", submission.shape)
    print(submission.head())

## --- ERROR in cell 18, traceback:
---------------------------------------------------------------------------
RuntimeError                              Traceback (most recent call last)
/tmp/ipykernel_55/2631452766.py in <cell line: 0>()
     46                         memory_format=torch.channels_last
     47                     )
---> 48                 outputs = model(inputs)
     49                 pred = outputs.argmax(dim=1).detach().cpu().numpy().astype(np.int64)
     50                 predicts[write_pos : write_pos + pred.shape[0]] = pred

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

/tmp/ipykernel_55/1860382837.py in forward(self, x)
     82 
     83     def forward(self, x):
---> 84         x = self.sequence(x)
     85         x = x.view(-1, self.hidden_channels)
     86         x = self.linear(x)

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/tmp/ipykernel_55/703435372.py in forward(self, x)
     31 
     32     def forward(self, x):
---> 33         x = self.sequence(x)
     34         return x
     35 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/container.py in forward(self, input)
    248     def forward(self, input):
    249         for module in self:
--> 250             input = module(input)
    251         return input
    252 

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

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in forward(self, input)
    552 
    553     def forward(self, input: Tensor) -> Tensor:
--> 554         return self._conv_forward(input, self.weight, self.bias)
    555 
    556 

/usr/local/lib/python3.11/dist-packages/torch/nn/modules/conv.py in _conv_forward(self, input, weight, bias)
    547                 self.groups,
    548             )
--> 549         return F.conv2d(
    550             input, weight, bias, self.stride, self.padding, self.dilation, self.groups
    551         )

RuntimeError: Input type (torch.cuda.FloatTensor) and weight type (torch.FloatTensor) should be the same
