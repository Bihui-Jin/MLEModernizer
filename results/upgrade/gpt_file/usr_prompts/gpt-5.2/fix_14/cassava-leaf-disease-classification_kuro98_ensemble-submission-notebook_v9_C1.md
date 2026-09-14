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

3.12

# 3. Installed packages

geopandas==0.14.4
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
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
import random

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("device:", device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(DATA_DIR, "train.csv")
train_dir = os.path.join(DATA_DIR, "train_images")
test_dir = os.path.join(DATA_DIR, "test_images")
sample_sub_path = os.path.join(DATA_DIR, "sample_submission.csv")

eff_img_size = 528
vit_img_size = 384
batch_size = 16

num_workers = min(8, (os.cpu_count() or 4))
num_classes = 5
tta = True

train_epochs = 3
lr = 3e-4

weight_decay = 1e-2


class TinyBackbone(nn.Module):
    def __init__(self, out_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, out_dim),
        )

    def forward(self, x):
        return self.net(x)


vit_out_dim = 5
eff_out_dim = 5

vit_model = TinyBackbone(vit_out_dim).to(device)
eff_model = TinyBackbone(eff_out_dim).to(device)
linear_head = nn.Linear(vit_out_dim + eff_out_dim, num_classes).to(device)

if device.type == "cuda":
    vit_model = vit_model.to(memory_format=torch.channels_last)
    eff_model = eff_model.to(memory_format=torch.channels_last)


def _maybe_channels_last(x: torch.Tensor, enabled: bool) -> torch.Tensor:
    """
    Bugfix: channels_last is only valid for 4D (NCHW) tensors (and 5D for 3D conv).
    Dataset returns 3D CHW tensors; applying channels_last there crashes.
    We only convert after batching in the training/inference loops (rank 4/5).
    """
    if not enabled:
        return x
    if x.ndim == 4:
        return x.contiguous(memory_format=torch.channels_last)
    return x.contiguous()




## === cell 1
import pathlib
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as Fv2

try:
    from torchvision.io import decode_jpeg, read_file, ImageReadMode

    _HAS_TV_DECODE = True
except Exception:
    _HAS_TV_DECODE = False


class CassavaDataset(VisionDataset):
    """Dataset for Cassava images (train or test).

    If labels are provided, returns (vit_img, eff_img, label).
    Otherwise returns (vit_img, eff_img, filename).

    If TTA is enabled, vit_img/eff_img will be list[tensor] for inference dataset.
    """

    def __init__(
        self,
        data_dir,
        vit_size,
        efficient_size,
        transform=None,
        ttas=None,
        labels_df=None,
        channels_last: bool = False,  # retained for compatibility; not used at sample-level
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.labels_df = labels_df

        if labels_df is None:
            self.images = sorted(
                [
                    f
                    for f in os.listdir(data_dir)
                    if f.lower().endswith((".jpg", ".jpeg", ".png"))
                ]
            )
        else:
            self.images = labels_df["image_id"].tolist()
            self.labels = labels_df["label"].astype(int).tolist()

        self.vit_size = int(vit_size)
        self.efficient_size = int(efficient_size)

        self.center_crop_h = 600
        self.center_crop_w = 600

        self._tta_fastpath = (
            (labels_df is None)
            and (ttas is not None)
            and (transform is not None)
            and (len(ttas) == 4)
        )

        self._norm_mean = None
        self._norm_std = None
        for t in getattr(transform, "transforms", []):
            if isinstance(t, v2.Normalize):
                self._norm_mean = torch.tensor(t.mean, dtype=torch.float32).view(
                    3, 1, 1
                )
                self._norm_std = torch.tensor(t.std, dtype=torch.float32).view(3, 1, 1)
                break
        if self._norm_mean is None or self._norm_std is None:
            self._tta_fastpath = False

        self._big_size = int(max(self.vit_size, self.efficient_size))
        self._small_size = int(min(self.vit_size, self.efficient_size))
        self._vit_is_big = self.vit_size == self._big_size

    def _load_rgb_uint8_chw(self, img_path: str) -> torch.Tensor:
        if _HAS_TV_DECODE:
            data = read_file(img_path)  # uint8 1D tensor
            return decode_jpeg(data, mode=ImageReadMode.RGB)  # uint8 [C,H,W]
        img = Image.open(img_path).convert("RGB")
        return Fv2.to_image(img)  # uint8 tensor [C,H,W]

    def _center_crop_chw(self, img: torch.Tensor) -> torch.Tensor:
        _, h, w = img.shape
        th, tw = self.center_crop_h, self.center_crop_w
        if h == th and w == tw:
            return img
        i = max(0, int(round((h - th) / 2.0)))
        j = max(0, int(round((w - tw) / 2.0)))
        return img[:, i : i + th, j : j + tw]

    @staticmethod
    def _resize_chw_bicubic(img_uint8_chw: torch.Tensor, size: int) -> torch.Tensor:
        x = img_uint8_chw.unsqueeze(0).to(dtype=torch.float32)  # 1,C,H,W
        x = F.interpolate(x, size=(size, size), mode="bicubic", align_corners=False)
        x = x.clamp_(0.0, 255.0).to(dtype=torch.uint8).squeeze(0)
        return x

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)

        img = self._load_rgb_uint8_chw(img_path)
        img = self._center_crop_chw(img)

        big = self._resize_chw_bicubic(img, self._big_size)
        if self._small_size == self._big_size:
            small = big
        else:
            small = self._resize_chw_bicubic(big, self._small_size)

        if self._vit_is_big:
            vit_img = big
            eff_img = small
        else:
            vit_img = small
            eff_img = big

        transform = self.transform
        ttas = self.ttas

        if self._tta_fastpath:
            mean = self._norm_mean
            std = self._norm_std

            vit_base = vit_img.to(torch.float32).div_(255.0)
            vit_base = vit_base.sub(mean).div(std)

            eff_base = eff_img.to(torch.float32).div_(255.0)
            eff_base = eff_base.sub(mean).div(std)

            vit_img = [
                vit_base,
                torch.flip(vit_base, dims=[2]),  # horizontal flip (W)
                torch.flip(vit_base, dims=[1]),  # vertical flip (H)
                torch.rot90(vit_base, k=1, dims=[1, 2]),
            ]
            eff_img = [
                eff_base,
                torch.flip(eff_base, dims=[2]),
                torch.flip(eff_base, dims=[1]),
                torch.rot90(eff_base, k=1, dims=[1, 2]),
            ]
        else:
            if ttas is not None and transform is not None:
                vit_img = [transform(t(vit_img)) for t in ttas]
                eff_img = [transform(t(eff_img)) for t in ttas]
            elif transform is not None:
                vit_img = transform(vit_img)
                eff_img = transform(eff_img)

        if self.labels_df is None:
            return vit_img, eff_img, filename
        else:
            return vit_img, eff_img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 2
train_transforms = v2.Compose(
    [
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.2),
        v2.RandomRotation(degrees=15, interpolation=InterpolationMode.BILINEAR),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

test_transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90), interpolation=InterpolationMode.BILINEAR),
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)

use_channels_last = device.type == "cuda"

train_dataset = CassavaDataset(
    train_dir,
    vit_img_size,
    eff_img_size,
    transform=train_transforms,
    ttas=None,
    labels_df=train_df,
    channels_last=use_channels_last,
)

test_dataset = CassavaDataset(
    test_dir,
    vit_img_size,
    eff_img_size,
    transform=test_transforms,
    ttas=ttas,
    labels_df=None,
    channels_last=use_channels_last,
)


def _seed_worker(worker_id: int):
    worker_seed = (SEED + worker_id) % (2**32)
    np.random.seed(worker_seed)
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(SEED)


def _tta_collate(batch):
    bsz = len(batch)
    vit0, eff0, _ = batch[0]
    tta_n = len(vit0)

    vit = torch.empty((bsz, tta_n, *vit0[0].shape), dtype=vit0[0].dtype)  # [B,T,C,H,W]
    eff = torch.empty((bsz, tta_n, *eff0[0].shape), dtype=eff0[0].dtype)  # [B,T,C,H,W]
    names = [None] * bsz

    for i, (v_list, e_list, name) in enumerate(batch):
        names[i] = name
        vit[i].copy_(torch.stack(v_list, 0))
        eff[i].copy_(torch.stack(e_list, 0))

    return vit, eff, names


torch.set_num_threads(max(1, min(4, (os.cpu_count() or 4))))

_prefetch = 8 if num_workers > 0 else None
_pin = torch.cuda.is_available()

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=_pin,
    pin_memory_device="cuda" if _pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    generator=g,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=_pin,
    pin_memory_device="cuda" if _pin else "",
    persistent_workers=(num_workers > 0),
    prefetch_factor=_prefetch,
    worker_init_fn=_seed_worker if num_workers > 0 else None,
    collate_fn=_tta_collate if tta else None,
)




## === cell 3
vit_model.train()
eff_model.train()
linear_head.train()

criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

optimizer = torch.optim.AdamW(
    list(vit_model.parameters())
    + list(eff_model.parameters())
    + list(linear_head.parameters()),
    lr=lr,
    weight_decay=weight_decay,
)

if device.type == "cuda":
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True


for epoch in range(train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for vit_inputs, eff_inputs, labels in train_loader:
        vit_inputs = vit_inputs.to(device, non_blocking=True)
        eff_inputs = eff_inputs.to(device, non_blocking=True)
        vit_inputs = _maybe_channels_last(vit_inputs, use_channels_last)
        eff_inputs = _maybe_channels_last(eff_inputs, use_channels_last)

        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)

        vit_outputs = vit_model(vit_inputs)
        eff_outputs = eff_model(eff_inputs)
        logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
        logits = linear_head(logit_inputs)

        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

    epoch_loss = running_loss / max(1, total)
    epoch_acc = correct / max(1, total)
    print(
        f"epoch {epoch+1}/{train_epochs} - loss: {epoch_loss:.4f} - acc: {epoch_acc:.4f}"
    )

vit_model.eval()
eff_model.eval()
linear_head.eval()




## === cell 4
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.inference_mode():
    for vit_inputs, eff_inputs, filenames in test_loader:
        bs = len(filenames)

        if tta:
            n_tta = vit_inputs.shape[1]
            vit_inputs = vit_inputs.flatten(0, 1).to(
                device, non_blocking=True
            )  # [B*T,C,H,W]
            eff_inputs = eff_inputs.flatten(0, 1).to(device, non_blocking=True)
            vit_inputs = _maybe_channels_last(vit_inputs, use_channels_last)
            eff_inputs = _maybe_channels_last(eff_inputs, use_channels_last)

            vit_outputs = vit_model(vit_inputs)  # [B*T, D]
            eff_outputs = eff_model(eff_inputs)  # [B*T, D]

            vit_mean = vit_outputs.view(bs, n_tta, -1).mean(dim=1)
            eff_mean = eff_outputs.view(bs, n_tta, -1).mean(dim=1)

            logit_inputs = torch.cat([vit_mean, eff_mean], dim=1)
            logits = linear_head(logit_inputs)
            pred_labels = torch.argmax(logits, dim=1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            vit_inputs = _maybe_channels_last(vit_inputs, use_channels_last)
            eff_inputs = _maybe_channels_last(eff_inputs, use_channels_last)

            vit_outputs = vit_model(vit_inputs)
            eff_outputs = eff_model(eff_inputs)
            logit_inputs = torch.cat([vit_outputs, eff_outputs], dim=1)
            logits = linear_head(logit_inputs)
            pred_labels = torch.argmax(logits, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("preds:", len(all_preds), "names:", len(all_names), "dataset:", len(test_dataset))




## === cell 5
sample_sub = pd.read_csv(sample_sub_path)
pred_map = dict(zip(all_names, all_preds))

sample_sub["label"] = sample_sub["image_id"].map(pred_map).fillna(0).astype(int)

assert (
    sample_sub.shape[0] == 2676
), f"Unexpected submission length: {sample_sub.shape[0]}"
assert list(sample_sub.columns) == [
    "image_id",
    "label",
], f"Bad columns: {sample_sub.columns.tolist()}"
missing = sample_sub["label"].isna().sum()
print("missing labels in submission after map:", missing)

sample_sub.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", sample_sub.shape)
print(sample_sub.head())
