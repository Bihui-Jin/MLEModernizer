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

import pandas as pd
import torch
from PIL import Image, ImageFile
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

ImageFile.LOAD_TRUNCATED_IMAGES = True

try:
    Image.MAX_IMAGE_PIXELS = None
except Exception:
    pass

torch.manual_seed(3407)
random.seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.set_float32_matmul_precision("high")
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

train_epochs = 2
train_lr = 3e-4
weight_decay = 0.05

import torchvision


def _build_vit_5cls(img_size: int):
    m = torchvision.models.vit_b_16(weights=None, image_size=img_size)
    if hasattr(m, "heads") and hasattr(m.heads, "head"):
        in_features = m.heads.head.in_features
        m.heads.head = torch.nn.Linear(in_features, num_classes)
    else:
        in_features = m.classifier.in_features
        m.classifier = torch.nn.Linear(in_features, num_classes)
    return m


model_a = _build_vit_5cls(model_a_img_size).to(device)
model_b = _build_vit_5cls(model_b_img_size).to(device)

linear_head = torch.nn.Identity()


def _maybe_compile(m):
    try:
        return torch.compile(m, mode="reduce-overhead")
    except Exception as e:
        print("torch.compile unavailable/fallback:", repr(e))
        return m


if torch.cuda.is_available():
    model_a = _maybe_compile(model_a)
    model_b = _maybe_compile(model_b)

if torch.cuda.is_available():
    model_a = model_a.to(memory_format=torch.channels_last)
    model_b = model_b.to(memory_format=torch.channels_last)



## === cell 1
from functools import lru_cache


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data."""

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        transform=None,
        ttas=None,
        base_seed: int = 3407,
    ):
        super().__init__(root=data_dir)

        self.transform = transform

        exts = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
        self.images = []
        for de in os.scandir(data_dir):
            if de.is_file():
                ext = os.path.splitext(de.name)[1].lower()
                if ext in exts:
                    self.images.append(de.name)
        self.images.sort()

        self.ttas = ttas
        self.base_seed = int(base_seed)

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

        self._to_image = v2.ToImage()

        self.paths = [os.path.join(self.root, fn) for fn in self.images]

    @lru_cache(maxsize=4096)
    def _get_base_pil_cached(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.cc(img)
        return img

    def _get_base_pair(self, idx: int):
        idx = int(idx)
        img = self._get_base_pil_cached(idx)
        a_img = self.resize_model_a(img)
        b_img = self.resize_model_b(img)
        a = self._to_image(a_img)  # uint8 CHW
        b = self._to_image(b_img)  # uint8 CHW
        return a, b

    def __getitem__(self, idx):
        filename = self.images[idx]
        model_a_img, model_b_img = self._get_base_pair(idx)
        return model_a_img, model_b_img, filename, idx

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    def __init__(self, csv_path, images_dir, model_a_size, transform=None):
        super().__init__(root=images_dir)
        df = pd.read_csv(csv_path)

        self.image_ids = df["image_id"].tolist()
        self.labels = df["label"].astype("int64").tolist()

        self.transform = transform
        self.cc = v2.CenterCrop((600, 600))
        self.resize = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self._to_image = v2.ToImage()

        self.paths = [os.path.join(self.root, fn) for fn in self.image_ids]

    @lru_cache(maxsize=8192)
    def _get_base_cached(self, idx: int):
        img = Image.open(self.paths[idx]).convert("RGB")
        img = self.cc(img)
        img = self.resize(img)
        x = self._to_image(img)  # uint8 tensor
        return x

    def _get_base(self, idx: int):
        return self._get_base_cached(int(idx))

    def __getitem__(self, idx):
        label = int(self.labels[idx])
        img = self._get_base(idx)
        if self.transform:
            img = self.transform(img)
        return img, label

    def __len__(self):
        return len(self.labels)




## === cell 2
test_transforms = v2.Compose(
    [
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180, interpolation=InterpolationMode.BILINEAR),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

test_dataset = CassavaDataset(
    test_dir, model_a_img_size, model_b_img_size, transform=test_transforms, ttas=ttas
)


def _seed_worker(worker_id: int):
    seed = 3407 + worker_id
    random.seed(seed)
    torch.manual_seed(seed)


def _make_test_collate_fn(dataset: CassavaDataset):
    base_seed = dataset.base_seed
    transform = dataset.transform
    ttas_local = dataset.ttas

    def _collate(batch):
        a_imgs, b_imgs, fnames, idxs = zip(*batch)
        B = len(a_imgs)

        if ttas_local is not None and transform is not None:
            T = len(ttas_local)

            tmp = transform(a_imgs[0])
            out_shape = (B, T) + tuple(tmp.shape)
            a_out = torch.empty(out_shape, dtype=tmp.dtype)
            b_out = torch.empty(out_shape, dtype=tmp.dtype)

            for i in range(B):
                idx_i = int(idxs[i])
                seed_base = base_seed + idx_i * 1000
                a_img = a_imgs[i]
                b_img = b_imgs[i]
                for j, t in enumerate(ttas_local):
                    seed = int(seed_base + j)

                    torch.manual_seed(seed)
                    random.seed(seed)
                    a_out[i, j].copy_(transform(t(a_img)))

                    torch.manual_seed(seed)
                    random.seed(seed)
                    b_out[i, j].copy_(transform(t(b_img)))

            return a_out, b_out, list(fnames)

        elif transform is not None:
            a_out = torch.stack([transform(x) for x in a_imgs], dim=0)
            b_out = torch.stack([transform(x) for x in b_imgs], dim=0)
            return a_out, b_out, list(fnames)
        else:
            return list(a_imgs), list(b_imgs), list(fnames)

    return _collate


_eff_workers = min(8, max(2, (os.cpu_count() or 4) - 1))
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=_eff_workers,
    pin_memory=True,
    persistent_workers=(_eff_workers > 0),
    prefetch_factor=(4 if _eff_workers > 0 else None),
    worker_init_fn=_seed_worker if _eff_workers > 0 else None,
    collate_fn=_make_test_collate_fn(test_dataset),
)

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandomHorizontalFlip(p=0.5),
        v2.RandomVerticalFlip(p=0.5),
        v2.RandomRotation(20, interpolation=InterpolationMode.BILINEAR),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_dataset = CassavaTrainDataset(
    csv_path=train_csv_path,
    images_dir=train_dir,
    model_a_size=model_a_img_size,
    transform=train_transforms,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=_eff_workers,
    pin_memory=True,
    persistent_workers=(_eff_workers > 0),
    prefetch_factor=(4 if _eff_workers > 0 else None),
    worker_init_fn=_seed_worker if _eff_workers > 0 else None,
)

normalizer = torch.nn.Softmax(dim=1)



## === cell 3
model_a.train()
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.AdamW(
    model_a.parameters(), lr=train_lr, weight_decay=weight_decay
)

for epoch in range(train_epochs):
    running_loss = 0.0
    correct = 0
    total = 0
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if torch.cuda.is_available():
            imgs = imgs.contiguous(memory_format=torch.channels_last)

        optimizer.zero_grad(set_to_none=True)
        logits = model_a(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * imgs.size(0)
        preds = torch.argmax(logits.detach(), dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.numel())

    print(
        f"epoch {epoch+1}/{train_epochs} "
        f"loss={running_loss/max(total,1):.4f} "
        f"train_acc={correct/max(total,1):.4f}"
    )

model_a.eval()
model_b.eval()
if hasattr(linear_head, "eval"):
    linear_head.eval()



## === cell 4
all_names = []
all_preds = []

with torch.inference_mode():
    for batch_idx, (model_a_inputs, model_b_inputs, filenames) in enumerate(
        test_loader
    ):
        bs = len(filenames)

        if tta:
            a = model_a_inputs.reshape(-1, *model_a_inputs.shape[2:]).to(
                device, non_blocking=True
            )
            b = model_b_inputs.reshape(-1, *model_b_inputs.shape[2:]).to(
                device, non_blocking=True
            )
            if torch.cuda.is_available():
                a = a.contiguous(memory_format=torch.channels_last)
                b = b.contiguous(memory_format=torch.channels_last)

            ao = model_a(a)
            bo = model_b(b)

            ao = ao.reshape(-1, bs, ao.shape[-1]).mean(dim=0)
            bo = bo.reshape(-1, bs, bo.shape[-1]).mean(dim=0)

            outputs = 0.95 * ao + 0.05 * bo

            pred_labels = torch.argmax(outputs, 1).tolist()
        else:
            a = model_a_inputs.to(device, non_blocking=True)
            b = model_b_inputs.to(device, non_blocking=True)
            if torch.cuda.is_available():
                a = a.contiguous(memory_format=torch.channels_last)
                b = b.contiguous(memory_format=torch.channels_last)

            ao = model_a(a)
            bo = model_b(b)
            outputs = 0.95 * ao + 0.05 * bo

            pred_labels = torch.argmax(outputs, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 5
sample_path = "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
sample_sub = pd.read_csv(sample_path)

pred_map = dict(zip(all_names, all_preds))
sub = sample_sub.copy()
sub["label"] = sub["image_id"].map(pred_map)

if sub["label"].isna().any():
    if len(all_preds) > 0:
        fallback = int(pd.Series(all_preds).mode().iloc[0])
    else:
        fallback = 0
    sub["label"] = sub["label"].fillna(fallback).astype(int)
else:
    sub["label"] = sub["label"].astype(int)

sub.to_csv("submission.csv", index=False)
print(sub.shape)
print(sub.head())
