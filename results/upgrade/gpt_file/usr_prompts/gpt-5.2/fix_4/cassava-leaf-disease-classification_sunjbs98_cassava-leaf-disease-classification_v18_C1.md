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
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-image==0.25.2
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

0.8173164097914778

# 6. Current score

0.11024

# 7. Whether higher score is better

Higher is better.

# 8. Previous improvement plans

- What this solution (achieved 0.10762) has done: 'Your run didn’t yield a Kaggle score because the notebook can’t execute end-to-end: the cell numbering skips (missing cell 8/10) and, more importantly, `torch.load` fail because the pretrained weight path points to a dataset that isn’t present in your provided `/kaggle/input/...` tree. I keep your model and inference logic identical, but make the pretrained weight loading robust (only load if the file exists, map to the right device), and I fix the image transform order (Resize must happen before ToTensor to avoid type/shape issues), which also prevents silent preprocessing bugs that can hurt accuracy. Finally, I ensure the submission uses `sample_submission.csv` ordering and writes a valid `submission.csv` with the required columns.'
- What this solution (achieved 0.10762) has done: 'Your low score is consistent with running inference using randomly initialized weights because the pretrained checkpoint path doesn’t exist in this environment, so the model never loads meaningful weights. I keep your model and inference logic the same, but (1) point the weight path to a location that can actually exist under your provided `/kaggle/input/...` tree and make the loader tolerant to common checkpoint formats, and (2) fix the submission row alignment by using a direct `image_id -> prediction` map (no per-row boolean indexing inside the loop), ensuring every sample_submission row gets the correct prediction. These are minimal changes that should move accuracy up substantially toward your target without changing architecture/loss/training semantics. The script still run end-to-end and write a valid `submission.csv`.'
- What this solution (achieved 0.11024) has done: 'Your current score (0.10762) strongly suggests the model is doing inference with random weights because the referenced pretrained checkpoint folder doesn’t exist in your available `/kaggle/input/...` tree. I keep your model and inference logic identical, but make the script automatically load weights from a real, available source (torchvision’s EfficientNet-B3 pretrained) by mapping its weights into your EfficientNet module (same architecture family), which is a minimal change focused purely on getting meaningful features. I also fix a critical bug in `BottleneckUnit` (`self.out_channels` is wrong) that can silently break compatibility and weight loading. Finally, I keep the submission alignment logic but ensure the dataset used for inference is definitely `test_images` and matches `sample_submission.csv` order.'

# 9. Code solution

## === cell 0
import os
import math
from copy import deepcopy

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from skimage import io
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms



## === cell 1
is_submission = True
data_path = "/kaggle/input/cassava-leaf-disease-classification"
csv_file_name = "train.csv" if not is_submission else "sample_submission.csv"
batch_size = 8
num_workers = 2

mean, std = 0.5, 0.5
efficient_net_version = 3  # (1.4, 1.2)

use_pre_trained_weight = True

pre_trained_weight_path = os.path.join(
    "/kaggle/input", "cassava-leaf-disease-classification-weight", "weight.pth"
)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")




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

    def __len__(self):
        return len(self.cassava_leaf_disease)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = "/".join(
            [self.root_dir, self.folder, self.cassava_leaf_disease.image_id[idx]]
        )
        img = io.imread(img_name)

        if self.transform:
            img = self.transform(img)

        label = self.cassava_leaf_disease.label[idx]
        sample = (img, label)
        return sample




## === cell 3
transform = transforms.Compose(
    [
        transforms.ToPILImage(),
        transforms.Resize((512, 512)),
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

data_loader = DataLoader(
    dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)




## === cell 4
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

        self.out_channels = out_channels

    def forward(self, x):
        y = self.sequence(x)
        z = self.activation(x + y if self.residual else y)
        return z




## === cell 5
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




## === cell 6
def _try_load_local_checkpoint(model, path: str) -> bool:
    if not (path and os.path.exists(path)):
        return False
    ckpt = torch.load(path, map_location="cpu")
    if isinstance(ckpt, dict) and "state_dict" in ckpt:
        ckpt = ckpt["state_dict"]
    if isinstance(ckpt, dict) and "model" in ckpt:
        ckpt = ckpt["model"]
    if isinstance(ckpt, dict):
        ckpt = {k.replace("module.", ""): v for k, v in ckpt.items()}
    model.load_state_dict(ckpt, strict=True)
    return True


def _load_torchvision_efficientnet_b3_as_fallback(model: torch.nn.Module) -> bool:
    """
    Minimal, score-relevant fallback:
    If the local cassava checkpoint doesn't exist, load torchvision EfficientNet-B3 pretrained weights
    and map the matching parameter tensors by (name, shape). This keeps your architecture/forward the same,
    but prevents random-weight inference (which causes ~0.10 accuracy).
    """
    try:
        import torchvision
        from torchvision.models import efficientnet_b3, EfficientNet_B3_Weights
    except Exception:
        return False

    try:
        tv = efficientnet_b3(weights=EfficientNet_B3_Weights.IMAGENET1K_V1)
    except Exception:
        return False

    src = tv.state_dict()
    dst = model.state_dict()

    copied = 0
    used_src_keys = set()

    for dk, dv in dst.items():
        if dk.startswith("linear."):
            continue

        candidates = []
        if dk in src and src[dk].shape == dv.shape:
            candidates = [dk]
        else:
            for sk, sv in src.items():
                if sk in used_src_keys:
                    continue
                if sv.shape != dv.shape:
                    continue
                if sk.endswith(dk) or dk.endswith(sk):
                    candidates.append(sk)

        if candidates:
            sk = candidates[0]
            dst[dk] = src[sk].detach().clone()
            used_src_keys.add(sk)
            copied += 1

    model.load_state_dict(dst, strict=False)
    return copied > 50


model = EfficientNet(3, 5, *EFFICIENT_NET_COMPOUND_COEF[efficient_net_version])

loaded = False
if use_pre_trained_weight:
    loaded = _try_load_local_checkpoint(model, pre_trained_weight_path)
    if not loaded:
        loaded = _load_torchvision_efficientnet_b3_as_fallback(model)

model = model.to(device)



## === cell 7
if is_submission:
    model.eval()

    submission = pd.read_csv(os.path.join(data_path, "sample_submission.csv"))
    submission_ids = submission["image_id"].tolist()

    pred_by_id = {}

    with torch.no_grad():
        offset = 0
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            batch_preds = predicted.detach().cpu().numpy().tolist()

            batch_ids = (
                dataset.cassava_leaf_disease["image_id"]
                .iloc[offset : offset + len(batch_preds)]
                .tolist()
            )
            for img_id, pred in zip(batch_ids, batch_preds):
                pred_by_id[img_id] = int(pred)
            offset += len(batch_preds)

    submission["label"] = [pred_by_id.get(img_id, 0) for img_id in submission_ids]
    submission.to_csv("submission.csv", index=False)




## === cell 8
def progress_print(comment, current, total, num_of_print=50, slow=5):
    end = "\r" if current < total else "\n"
    symbol = ["\\", "/", "-"]
    done = int((current - 1) / total * num_of_print)
    doing = symbol[(current % (len(symbol) * slow) // slow)] if current < total else ""
    yet = max(0, num_of_print - 1 - done)
    print(comment + ": " + "#" * done + doing + "*" * yet, end=end)


def get_confusion_matrix(model, data_loader, num_classes):
    model.eval()
    confusion_matrix = np.zeros((num_classes, num_classes), dtype=np.int32)
    with torch.no_grad():
        for i, data in enumerate(data_loader):
            inputs, labels = data
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            predicts = predicted.squeeze()
            for p, l in zip(predicts, labels):
                confusion_matrix[p.item()][int(l)] += 1

            progress_print("confusion_matrix", i, len(data_loader), slow=3)
    return confusion_matrix




## === cell 9
def show_confusion_matrix(confusion_matrix):
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    plt.figure(figsize=(2, 1))
    plt.matshow(confusion_matrix, cmap="gray")
    plt.matshow(diagonal_confusion_matrix, cmap="gray")
    plt.show()
    print(confusion_matrix)




## === cell 10
if not is_submission:
    confusion_matrix = get_confusion_matrix(model, data_loader, 5)



## === cell 11
if not is_submission:
    row_sums = confusion_matrix.sum(axis=1)
    normalized_confusion_matrix = confusion_matrix / row_sums[:, np.newaxis]
    show_confusion_matrix(normalized_confusion_matrix)
