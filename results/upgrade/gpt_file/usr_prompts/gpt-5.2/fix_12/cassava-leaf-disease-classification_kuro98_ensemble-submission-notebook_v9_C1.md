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

# 5. Target score

0.8986098519190088

# 6. Current score

0.05531

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plans

- What this solution (achieved 0.61099) has done: 'I fix the pipeline so it runs end-to-end by (1) removing the dependency on missing external model files and replacing them with a minimal, self-contained PyTorch model that preserves the same “vit + efficient + linear head” fusion structure, (2) fixing non-deterministic/incorrect inference ordering by disabling shuffle and sorting filenames to match `sample_submission.csv`, and (3) ensuring test-time augmentation works with torchvision v2 by using deterministic flip/rotate transforms and correct batching. This produce a valid `submission.csv` with exactly the required rows and columns. Since no current score exists, the goal is correctness and a reasonable baseline; the main score-affecting change is only the unavoidable replacement of unavailable weights.'
- What this solution (achieved 0.6364) has done: 'Your current score is low because the models are randomly initialized and never trained, so predictions are essentially noise; the smallest change that legitimately increases accuracy is to train the existing tiny backbones + linear head on `train.csv` using the same image preprocessing. To preserve your core “vit backbone + efficient backbone + linear head fusion” logic and inference/TTA semantics, I’m only adding a short supervised training loop (no architecture/loss changes) and then running the same inference code to write `submission.csv`. I also keep determinism and ensure filenames align with `sample_submission.csv` exactly as you already do. This should move the score substantially upward toward the 0.8986 target without changing the model structure.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by CPU image decoding + multiple expensive resize/augment ops per sample, plus per-iteration `channels_last` conversions inside the train/infer loops. I keep the exact model/training/inference logic, but speed up input processing by (1) switching to `torchvision.io.read_image` with an explicit backend, (2) resizing directly to each target size (avoids an extra “big” resize and then two more resizes), (3) computing the center-crop box once and using a fast slice-based crop, and (4) moving `channels_last` conversion into the Dataset/Collate so tensors arrive already in the right memory format (removing repeated per-batch conversions). I also cache the sorted test filenames list to avoid repeated `os.listdir` overhead and add `torch.no_grad()`-safe minor DataLoader tuning without changing semantics.'
- What this solution (achieved 0.05531) has done: 'The timeout is dominated by CPU-side image decoding + resizing twice per sample (train+test) plus expensive per-sample TTA transform application, which keeps the GPU underfed. I keep the exact same model/training/inference logic, but speed up input by (1) switching to `torchvision.io.decode_jpeg` from raw bytes (fast libjpeg-turbo path), (2) replacing `Fv2.resize` with `torch.nn.functional.interpolate` on tensors (same geometry, much faster), and (3) making the TTA fast-path fully tensor-based using `torch.flip/rot90` (exactly equivalent to the current chosen TTAs). I also tune DataLoader worker behavior (more aggressive prefetch/persistent workers, avoiding accimage) while keeping determinism via the same seeds/worker seeding. No epochs, batch sizes, augmentation set, loss, or architecture are changed.'

# 9. Code solution

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
import torch.nn.functional as F
from torchvision.transforms.v2 import functional as Fv2

try:
    from torchvision.io import decode_jpeg, ImageReadMode

    _HAS_TV_DECODE = True
except Exception:
    _HAS_TV_DECODE = False


def _read_file_bytes(path: str) -> torch.Tensor:
    return torch.fromfile(path, dtype=torch.uint8)


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

    def _load_rgb_uint8_chw(self, img_path: str) -> torch.Tensor:
        if _HAS_TV_DECODE:
            data = _read_file_bytes(img_path)
            return decode_jpeg(data, mode=ImageReadMode.RGB)
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

        vit_img = self._resize_chw_bicubic(img, self.vit_size)
        eff_img = self._resize_chw_bicubic(img, self.efficient_size)

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
        vit[i] = torch.stack(v_list, 0)
        eff[i] = torch.stack(e_list, 0)

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




## --- ERROR in cell 3, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3590187757.py in <cell line: 0>()
     24     total = 0
     25 
---> 26     for vit_inputs, eff_inputs, labels in train_loader:
     27         vit_inputs = vit_inputs.to(device, non_blocking=True)
     28         eff_inputs = eff_inputs.to(device, non_blocking=True)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

AttributeError: Caught AttributeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 115, in __getitem__
    img = self._load_rgb_uint8_chw(img_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 87, in _load_rgb_uint8_chw
    data = _read_file_bytes(img_path)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 19, in _read_file_bytes
    return torch.fromfile(path, dtype=torch.uint8)
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/__init__.py", line 2681, in __getattr__
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
AttributeError: module 'torch' has no attribute 'fromfile'


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




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/3201309118.py in <cell line: 0>()
      7 
      8 with torch.inference_mode():
----> 9     for vit_inputs, eff_inputs, filenames in test_loader:
     10         bs = len(filenames)
     11 

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
   1478                 del self._task_info[idx]
   1479                 self._rcvd_idx += 1
-> 1480                 return self._process_data(data)
   1481 
   1482     def _try_put_index(self):

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _process_data(self, data)
   1503         self._try_put_index()
   1504         if isinstance(data, ExceptionWrapper):
-> 1505             data.reraise()
   1506         return data
   1507 

/usr/local/lib/python3.11/dist-packages/torch/_utils.py in reraise(self)
    731             # instantiate since we don't know how to
    732             raise RuntimeError(msg) from None
--> 733         raise exception
    734 
    735 

AttributeError: Caught AttributeError in DataLoader worker process 0.
Original Traceback (most recent call last):
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/worker.py", line 349, in _worker_loop
    data = fetcher.fetch(index)  # type: ignore[possibly-undefined]
           ^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in fetch
    data = [self.dataset[idx] for idx in possibly_batched_index]
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py", line 52, in <listcomp>
    data = [self.dataset[idx] for idx in possibly_batched_index]
            ~~~~~~~~~~~~^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 115, in __getitem__
    img = self._load_rgb_uint8_chw(img_path)
          ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 87, in _load_rgb_uint8_chw
    data = _read_file_bytes(img_path)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/tmp/ipykernel_55/608671023.py", line 19, in _read_file_bytes
    return torch.fromfile(path, dtype=torch.uint8)
           ^^^^^^^^^^^^^^
  File "/usr/local/lib/python3.11/dist-packages/torch/__init__.py", line 2681, in __getattr__
    raise AttributeError(f"module '{__name__}' has no attribute '{name}'")
AttributeError: module 'torch' has no attribute 'fromfile'


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
