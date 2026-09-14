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

# 5. Code solution

## === cell 0
import os
import torch
import pandas as pd
from skimage import io, transform
import numpy as np
import matplotlib.pyplot as plt
import math
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from copy import deepcopy
from PIL import Image

torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

torch.manual_seed(42)
np.random.seed(42)

candidate_paths = [
    "./kaggle/data/cassava-leaf-disease-classification",
    "./input/cassava-leaf-disease-classification",
    "/kaggle/input/cassava-leaf-disease-classification",
    "/kaggle/working/cassava-leaf-disease-classification",
]
data_path = None
for p in candidate_paths:
    if os.path.isdir(p):
        data_path = p
        break
if data_path is None:
    raise FileNotFoundError("Dataset directory not found in expected locations.")

csv_file_name = "train.csv"
sample_submission_path = os.path.join(data_path, "sample_submission.csv")

for fp in [os.path.join(data_path, csv_file_name), sample_submission_path]:
    if not os.path.isfile(fp):
        raise FileNotFoundError(f"Required file not found: {fp}")

batch_size = 64
num_workers = min(16, os.cpu_count() or 1)

mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.2),
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)

test_transform = transforms.Compose(
    [
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
)


class CassavaLeafDiseaseDataset(Dataset):
    "Cassava Leaf Disease"

    def __init__(self, csv_file, root_dir, transform=None, train=True):
        self.cassava_leaf_disease = pd.read_csv(csv_file)
        self.root_dir = root_dir
        self.transform = transform
        self.folder = "train_images" if train else "test_images"

    def __len__(self):
        return len(self.cassava_leaf_disease)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(
            self.root_dir, self.folder, self.cassava_leaf_disease.image_id.iloc[idx]
        )
        img = Image.open(img_name).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.cassava_leaf_disease.label.iloc[idx]
        return img, label


train_dataset = CassavaLeafDiseaseDataset(
    csv_file=os.path.join(data_path, csv_file_name),
    root_dir=data_path,
    transform=train_transform,
    train=True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,  # pin memory for faster GPU transfer
    persistent_workers=True,  # keep workers alive across epochs
)




## === cell 1
import math


class ConvUnit(torch.nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        padding: int = 0,
        stride: int = 1,
        groups: int = 1,
        batch_norm: bool = True,
        activation: bool = True,
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
        return self.sequence(x)


def DConvUnit(in_channels: int, kernel_size: int, stride: int = 1) -> torch.nn.Module:
    padding = kernel_size // 2
    return ConvUnit(in_channels, in_channels, kernel_size, padding, stride, in_channels)


def EConvUnit(in_channels: int, factor: int = 6) -> torch.nn.Module:
    return ConvUnit(in_channels, factor * in_channels, 1)


def PConvUnit(in_channels, out_channels, activation: bool = False) -> torch.nn.Module:
    return ConvUnit(in_channels, out_channels, 1, activation=activation)


class DSConvUnit(torch.nn.Module):
    """
    Fixed depth‑wise separable convolution:
    - Depthwise conv keeps the channel count.
    - Point‑wise conv projects to the desired output channels.
    - BatchNorm and optional activation are retained.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int = 3,
        stride: int = 1,
        activation: bool = True,
    ):
        super().__init__()
        self.depthwise = torch.nn.Conv2d(
            in_channels,
            in_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=in_channels,
            bias=False,
        )
        self.bn_dw = torch.nn.BatchNorm2d(in_channels)
        self.act = torch.nn.LeakyReLU() if activation else torch.nn.Identity()
        self.pointwise = torch.nn.Conv2d(
            in_channels, out_channels, kernel_size=1, bias=False
        )
        self.bn_pw = torch.nn.BatchNorm2d(out_channels)

    def forward(self, x):
        x = self.depthwise(x)
        x = self.bn_dw(x)
        x = self.act(x)
        x = self.pointwise(x)
        x = self.bn_pw(x)
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
        return x * y


class BottleneckUnit(torch.nn.Module):
    """
    Corrected bottleneck:
    - The projection (1x1) after depthwise convolution now uses the
      *output* channels from DSConvUnit, avoiding a channel‑mismatch.
    - `self.out_channels` reflects the true output channel count.
    """

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
        modules.append(DSConvUnit(hid_channels, out_channels, kernel_size, stride))
        if has_se:
            calced_se_ratio = se_ratio / factor
            modules.append(SEUnit(hid_channels, calced_se_ratio))
        modules.append(PConvUnit(out_channels, out_channels, False))
        self.sequence = torch.nn.Sequential(*modules)
        self.activation = torch.nn.LeakyReLU()
        self.out_channels = out_channels

    def forward(self, x):
        y = self.sequence(x)
        z = self.activation(x + y if self.residual else y)
        return z




## === cell 2
def round_filters(filters, coef: float, divisor: int) -> int:
    filters *= coef
    filters = max(divisor, int(filters + divisor / 2) // divisor * divisor)
    return filters


def round_repeats(repeats, coef):
    return int(math.ceil(repeats * coef)) if repeats > 0 else -repeats


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
        return self.linear(x)




## === cell 3
import torchvision

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torchvision.models.efficientnet_b0(
    weights=torchvision.models.EfficientNet_B0_Weights.DEFAULT
)

in_features = model.classifier[1].in_features
model.classifier[1] = torch.nn.Linear(in_features, 5)

model = model.to(device)




## === cell 4
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
scaler = torch.cuda.amp.GradScaler()  # <-- added for AMP

scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.5)

model.train()
num_epochs = 30  # unchanged – core training schedule preserved
for epoch in range(num_epochs):
    running_loss = 0.0
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.cuda.amp.autocast():
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item()
    scheduler.step()
    print(f"Epoch {epoch+1}/{num_epochs} – loss: {running_loss/len(train_loader):.4f}")




## === cell 5
is_submission = True  # switch flag for inference path

submission_df = pd.read_csv(sample_submission_path)

test_csv_path = "tmp_test.csv"
submission_df.to_csv(test_csv_path, index=False)

test_dataset = CassavaLeafDiseaseDataset(
    csv_file=test_csv_path,
    root_dir=data_path,
    transform=test_transform,
    train=False,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

model.eval()
predicts = []
with torch.no_grad():
    for inputs, _ in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        outputs = model(inputs)
        _, predicted = torch.max(outputs, 1)
        predicts.extend(predicted.cpu().numpy())

submission_df["label"] = predicts
submission_df.to_csv("submission.csv", index=False)
print("Submission file 'submission.csv' written with", len(predicts), "predictions.")




## === cell 6
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
        for i, (inputs, labels) in enumerate(data_loader, 1):
            inputs = inputs.to(device, non_blocking=True)
            outputs = model(inputs)
            _, predicted = torch.max(outputs, 1)
            for p, l in zip(predicted, labels):
                confusion_matrix[p.item()][l] += 1
            progress_print("confusion_matrix", i, len(data_loader), slow=3)
    return confusion_matrix




## === cell 7
def show_confusion_matrix(confusion_matrix):
    diagonal_confusion_matrix = deepcopy(confusion_matrix)
    np.fill_diagonal(diagonal_confusion_matrix, 0)
    plt.figure(figsize=(6, 5))
    plt.matshow(confusion_matrix, cmap="gray")
    plt.matshow(diagonal_confusion_matrix, cmap="gray")
    plt.show()
    print(confusion_matrix)


if not is_submission:
    confusion_matrix = get_confusion_matrix(model, train_loader, 5)
    row_sums = confusion_matrix.sum(axis=1, keepdims=True)
    normalized_confusion_matrix = confusion_matrix / row_sums
    show_confusion_matrix(normalized_confusion_matrix)
