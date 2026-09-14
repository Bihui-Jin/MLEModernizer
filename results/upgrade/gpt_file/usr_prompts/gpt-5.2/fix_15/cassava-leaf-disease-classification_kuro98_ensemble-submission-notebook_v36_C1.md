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

import pandas as pd
import torch
from torch.backends import cudnn
from torch.utils.data import DataLoader, Subset, WeightedRandomSampler
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision.io import read_image, ImageReadMode



## === cell 1
torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
try:
    torch.use_deterministic_algorithms(False)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"

model_b_img_size = 384
model_a_img_size = 384
batch_size = 16
num_workers = 4
num_classes = 5

tta = True

train_epochs = 6
train_lr = 2e-4

val_frac = 0.10


class FallbackTinyNet(torch.nn.Module):
    def __init__(self, num_classes: int = 5):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            torch.nn.ReLU(inplace=True),
            torch.nn.AdaptiveAvgPool2d(1),
            torch.nn.Flatten(),
            torch.nn.Linear(32, num_classes),
        )

    def forward(self, x):
        return self.net(x)


model_a = FallbackTinyNet(num_classes=num_classes).to(device)
model_b = FallbackTinyNet(num_classes=num_classes).to(device)
linear_head = torch.nn.Identity().to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        img_size=384,
    ):
        super().__init__(root=data_dir)

        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )
        self.ttas = ttas

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    @staticmethod
    def _pad_to_min600(img_t: torch.Tensor) -> torch.Tensor:
        _, h, w = img_t.shape
        if w >= 600 and h >= 600:
            return img_t
        pad_w = max(0, 600 - w)
        pad_h = max(0, 600 - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return v2.functional.pad(img_t, padding=padding, fill=0)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)

        img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 tensor (C,H,W)

        img = self._pad_to_min600(img)
        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        return model_a_img, model_b_img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, csv_path, img_dir, img_size, transform=None):
        super().__init__(root=img_dir)
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir
        self.transform = transform

        self._image_ids = self.df["image_id"].astype(str).to_numpy()
        self._labels = self.df["label"].astype("int64").to_numpy()

        self.cc = v2.CenterCrop((600, 600))
        self.resize = v2.Resize(
            (img_size, img_size), interpolation=InterpolationMode.BICUBIC
        )

    @staticmethod
    def _pad_to_min600(img_t: torch.Tensor) -> torch.Tensor:
        _, h, w = img_t.shape
        if w >= 600 and h >= 600:
            return img_t
        pad_w = max(0, 600 - w)
        pad_h = max(0, 600 - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return v2.functional.pad(img_t, padding=padding, fill=0)

    def __getitem__(self, idx):
        filename = self._image_ids[idx]
        label = int(self._labels[idx])
        img_path = os.path.join(self.img_dir, filename)

        img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 tensor (C,H,W)
        img = self._pad_to_min600(img)

        img = self.cc(img)

        if self.transform:
            img = self.transform(img)
        else:
            img = self.resize(img)

        return img, label

    def __len__(self):
        return len(self._labels)


class CassavaTrainPairedDataset(VisionDataset):
    def __init__(
        self,
        csv_path,
        img_dir,
        model_a_size,
        model_b_size,
        transform_a=None,
        transform_b=None,
    ):
        super().__init__(root=img_dir)
        self.df = pd.read_csv(csv_path)
        self.img_dir = img_dir

        self._image_ids = self.df["image_id"].astype(str).to_numpy()
        self._labels = self.df["label"].astype("int64").to_numpy()

        self.transform_a = transform_a
        self.transform_b = transform_b

        self.cc = v2.CenterCrop((600, 600))
        self.resize_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    @staticmethod
    def _pad_to_min600(img_t: torch.Tensor) -> torch.Tensor:
        _, h, w = img_t.shape
        if w >= 600 and h >= 600:
            return img_t
        pad_w = max(0, 600 - w)
        pad_h = max(0, 600 - h)
        padding = (pad_w // 2, pad_h // 2, pad_w - pad_w // 2, pad_h - pad_h // 2)
        return v2.functional.pad(img_t, padding=padding, fill=0)

    def __getitem__(self, idx):
        filename = self._image_ids[idx]
        label = int(self._labels[idx])
        img_path = os.path.join(self.img_dir, filename)

        img = read_image(img_path, mode=ImageReadMode.RGB)  # uint8 tensor (C,H,W)
        img = self._pad_to_min600(img)
        img = self.cc(img)

        img_a = self.resize_a(img)
        img_b = self.resize_b(img)

        if self.transform_a is not None:
            img_a = self.transform_a(img_a)
        if self.transform_b is not None:
            img_b = self.transform_b(img_b)

        return img_a, img_b, label

    def __len__(self):
        return len(self._labels)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms_a = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(model_a_img_size, model_a_img_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        v2.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms_b = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomResizedCrop(
            size=(model_b_img_size, model_b_img_size),
            scale=(0.85, 1.0),
            ratio=(0.9, 1.1),
            interpolation=InterpolationMode.BICUBIC,
            antialias=True,
        ),
        v2.ColorJitter(brightness=0.15, contrast=0.15, saturation=0.10, hue=0.02),
        v2.RandomRotation(degrees=10, interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)

_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **_loader_kwargs,
)

train_dataset_full_paired = CassavaTrainPairedDataset(
    csv_path=train_csv_path,
    img_dir=train_dir,
    model_a_size=model_a_img_size,
    model_b_size=model_b_img_size,
    transform_a=train_transforms_a,
    transform_b=train_transforms_b,
)

n = len(train_dataset_full_paired)
g = torch.Generator()
g.manual_seed(3407)
perm = torch.randperm(n, generator=g).tolist()
val_n = int(n * val_frac)
val_idx = perm[:val_n]
train_idx = perm[val_n:]

train_dataset_paired = Subset(train_dataset_full_paired, train_idx)
val_dataset_paired = Subset(train_dataset_full_paired, val_idx)

train_labels = (
    train_dataset_full_paired.df.iloc[train_idx]["label"].astype("int64").to_numpy()
)
label_counts = pd.Series(train_labels).value_counts().to_dict()

inv_counts = torch.zeros(num_classes, dtype=torch.double)
for k, v in label_counts.items():
    inv_counts[int(k)] = 1.0 / float(v)

train_weights = inv_counts[torch.from_numpy(train_labels)]
sampler = WeightedRandomSampler(
    weights=train_weights, num_samples=len(train_weights), replacement=True
)

train_loader = DataLoader(
    train_dataset_paired,
    batch_size=batch_size,
    shuffle=False,
    sampler=sampler,
    drop_last=False,
    **_loader_kwargs,
)

val_loader = DataLoader(
    val_dataset_paired,
    batch_size=batch_size,
    shuffle=False,
    drop_last=False,
    **_loader_kwargs,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 4
criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer_a = torch.optim.AdamW(model_a.parameters(), lr=train_lr, weight_decay=1e-4)
optimizer_b = torch.optim.AdamW(model_b.parameters(), lr=train_lr, weight_decay=1e-4)

scheduler_a = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_a, T_max=max(1, train_epochs)
)
scheduler_b = torch.optim.lr_scheduler.CosineAnnealingLR(
    optimizer_b, T_max=max(1, train_epochs)
)

scaler = torch.cuda.amp.GradScaler(enabled=False)

infer_mode = "ensemble"

model_a.train()
model_b.train()

for epoch in range(train_epochs):
    running_loss_a = 0.0
    running_loss_b = 0.0
    correct_a_t = torch.zeros((), device=device, dtype=torch.long)
    correct_b_t = torch.zeros((), device=device, dtype=torch.long)
    seen_t = torch.zeros((), device=device, dtype=torch.long)

    for imgs_a, imgs_b, labels in train_loader:
        imgs_a = imgs_a.to(device, non_blocking=True)
        imgs_b = imgs_b.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer_a.zero_grad(set_to_none=True)
        optimizer_b.zero_grad(set_to_none=True)

        logits_a = model_a(imgs_a)
        logits_b = model_b(imgs_b)
        loss_a = criterion(logits_a, labels)
        loss_b = criterion(logits_b, labels)
        loss = loss_a + loss_b

        loss.backward()
        optimizer_a.step()
        optimizer_b.step()

        running_loss_a += float(loss_a.detach())
        running_loss_b += float(loss_b.detach())
        with torch.no_grad():
            pred_a = torch.argmax(logits_a, dim=1)
            pred_b = torch.argmax(logits_b, dim=1)
            correct_a_t += (pred_a == labels).sum()
            correct_b_t += (pred_b == labels).sum()
            seen_t += labels.numel()

    scheduler_a.step()
    scheduler_b.step()

    correct_a = int(correct_a_t.item())
    correct_b = int(correct_b_t.item())
    seen = int(seen_t.item())

    print(
        f"epoch={epoch+1}/{train_epochs} "
        f"loss_a={running_loss_a/max(1,len(train_loader)):.4f} "
        f"loss_b={running_loss_b/max(1,len(train_loader)):.4f} "
        f"acc_a={correct_a/max(1,seen):.4f} "
        f"acc_b={correct_b/max(1,seen):.4f} "
        f"lr_a={scheduler_a.get_last_lr()[0]:.2e}"
    )

model_a.eval()
model_b.eval()
with torch.no_grad():
    va_seen_t = torch.zeros((), device=device, dtype=torch.long)
    va_ca_t = torch.zeros((), device=device, dtype=torch.long)
    va_cb_t = torch.zeros((), device=device, dtype=torch.long)
    va_cens_t = torch.zeros((), device=device, dtype=torch.long)

    for imgs_a, imgs_b, labels in val_loader:
        labels = labels.to(device, non_blocking=True)

        imgs_a = imgs_a.to(device, non_blocking=True)
        imgs_b = imgs_b.to(device, non_blocking=True)

        if tta:
            imgs_a_flip = torch.flip(imgs_a, dims=[3])
            imgs_b_flip = torch.flip(imgs_b, dims=[3])

            la = 0.5 * model_a(imgs_a) + 0.5 * model_a(imgs_a_flip)
            lb = 0.5 * model_b(imgs_b) + 0.5 * model_b(imgs_b_flip)
        else:
            la = model_a(imgs_a)
            lb = model_b(imgs_b)

        pa = torch.argmax(la, dim=1)
        pb = torch.argmax(lb, dim=1)
        pens = torch.argmax(0.5 * la + 0.5 * lb, dim=1)

        va_ca_t += (pa == labels).sum()
        va_cb_t += (pb == labels).sum()
        va_cens_t += (pens == labels).sum()
        va_seen_t += labels.numel()

val_acc_a = int(va_ca_t.item()) / max(1, int(va_seen_t.item()))
val_acc_b = int(va_cb_t.item()) / max(1, int(va_seen_t.item()))
val_acc_ens = int(va_cens_t.item()) / max(1, int(va_seen_t.item()))

if val_acc_ens >= val_acc_a and val_acc_ens >= val_acc_b:
    infer_mode = "ensemble"
elif val_acc_a >= val_acc_b:
    infer_mode = "a"
else:
    infer_mode = "b"

print(
    f"val_acc_a={val_acc_a:.4f} val_acc_b={val_acc_b:.4f} val_acc_ens={val_acc_ens:.4f} -> infer_mode={infer_mode}"
)



## === cell 5
all_names = []
all_preds = []

model_a.eval()
model_b.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bsz = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(
                device, non_blocking=True
            )
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(
                device, non_blocking=True
            )
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)  # (T*B, C)
            model_b_outputs = model_b(model_b_inputs)  # (T*B, C)

            t = model_a_outputs.shape[0] // bsz
            model_a_mean_logits = model_a_outputs.view(t, bsz, -1).mean(dim=0)
            model_b_mean_logits = model_b_outputs.view(t, bsz, -1).mean(dim=0)

            if infer_mode == "a":
                outputs = model_a_mean_logits
            elif infer_mode == "b":
                outputs = model_b_mean_logits
            else:
                outputs = 0.5 * model_a_mean_logits + 0.5 * model_b_mean_logits

            mean_preds = normalizer(outputs)
            pred_labels = torch.argmax(mean_preds, 1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device, non_blocking=True)
            model_b_inputs = model_b_inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            model_a_outputs = model_a(model_a_inputs)
            model_b_outputs = model_b(model_b_inputs)

            if infer_mode == "a":
                outputs = model_a_outputs
            elif infer_mode == "b":
                outputs = model_b_outputs
            else:
                outputs = 0.5 * model_a_outputs + 0.5 * model_b_outputs

            preds = normalizer(outputs)
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 6
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
default_label = 0
aligned_labels = [
    int(pred_map.get(img_id, default_label))
    for img_id in sample_sub["image_id"].tolist()
]

my_submission = pd.DataFrame(
    {"image_id": sample_sub["image_id"].tolist(), "label": aligned_labels}
)
my_submission.to_csv("submission.csv", index=False)

print(
    "Preds:",
    len(all_preds),
    "Unique files:",
    len(set(all_names)),
    "Submission rows:",
    len(my_submission),
)
print(my_submission.head())
