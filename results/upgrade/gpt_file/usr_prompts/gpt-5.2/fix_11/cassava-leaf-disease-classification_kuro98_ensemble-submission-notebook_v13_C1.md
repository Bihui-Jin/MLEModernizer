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
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import v2

torch.set_num_threads(max(1, (os.cpu_count() or 4) // 2))

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

test_dir = "/kaggle/input/cassava-leaf-disease-classification/test_images/"
train_dir = "/kaggle/input/cassava-leaf-disease-classification/train_images/"
train_csv_path = "/kaggle/input/cassava-leaf-disease-classification/train.csv"
sample_sub_path = (
    "/kaggle/input/cassava-leaf-disease-classification/sample_submission.csv"
)

batch_size = 32
num_workers = 4
num_classes = 5
tta = False  # unchanged

try:
    import torchvision

    torchvision.io.set_image_backend("accimage")
except Exception:
    pass

from torchvision.models import resnet18, ResNet18_Weights

weights = ResNet18_Weights.DEFAULT
model = resnet18(weights=weights)
model.fc = torch.nn.Linear(model.fc.in_features, num_classes)
model = model.to(device)

if device.type == "cuda":
    try:
        model = torch.compile(model, mode="max-autotune", fullgraph=False)
    except Exception:
        pass

resnet_eval_preprocess = weights.transforms()

_mean = getattr(resnet_eval_preprocess, "mean", None)
_std = getattr(resnet_eval_preprocess, "std", None)
if _mean is None or _std is None:
    _mean = (0.485, 0.456, 0.406)
    _std = (0.229, 0.224, 0.225)

IMG_SIZE = 384
RESIZE_SIZE = 438  # common choice: int(IMG_SIZE / 0.875)

eval_transform = v2.Compose(
    [
        v2.Resize(RESIZE_SIZE, interpolation=v2.InterpolationMode.BILINEAR),
        v2.CenterCrop((IMG_SIZE, IMG_SIZE)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
    ]
)

normalizer = torch.nn.Softmax(dim=1)


def seed_worker(worker_id: int):
    worker_seed = 3407 + worker_id
    torch.manual_seed(worker_seed)
    np.random.seed(worker_seed)
    random.seed(worker_seed)




## === cell 1
from torchvision.io import read_image, ImageReadMode


class CassavaTestDataset(VisionDataset):
    """Custom dataset for Cassava test images (no labels)."""

    def __init__(self, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform

        exts = (".jpg", ".jpeg", ".png", ".bmp", ".webp")
        entries = os.listdir(data_dir)
        files = [f for f in entries if f.lower().endswith(exts)]
        self.images = sorted(files)  # deterministic ordering

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = read_image(os.path.join(self.root, filename), mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Custom dataset for Cassava train images with labels from train.csv."""

    def __init__(self, df, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self._image_ids = self.df["image_id"].to_numpy()
        self._labels = self.df["label"].to_numpy(dtype=np.int64)

    def __getitem__(self, idx):
        filename = self._image_ids[idx]
        label = int(self._labels[idx])
        img = read_image(os.path.join(self.root, filename), mode=ImageReadMode.RGB)
        if self.transform:
            img = self.transform(img)
        return img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)




## === cell 2
train_df = pd.read_csv(train_csv_path)

g_split = torch.Generator().manual_seed(3407)
perm = torch.randperm(len(train_df), generator=g_split).numpy()
train_df_shuf = train_df.iloc[perm].reset_index(drop=True)

val_frac = 0.1
labels_arr = train_df_shuf["label"].to_numpy(dtype=np.int64)

val_mask = np.zeros(len(train_df_shuf), dtype=bool)
for c in range(num_classes):
    idx_c = np.flatnonzero(labels_arr == c)
    v_c = int(round(len(idx_c) * val_frac))
    if v_c > 0:
        val_mask[idx_c[:v_c]] = True

val_df = (
    train_df_shuf.loc[val_mask]
    .sample(frac=1.0, random_state=3407)
    .reset_index(drop=True)
)
tr_df = (
    train_df_shuf.loc[~val_mask]
    .sample(frac=1.0, random_state=3407)
    .reset_index(drop=True)
)

train_transform = v2.Compose(
    [
        v2.RandomResizedCrop(
            size=(IMG_SIZE, IMG_SIZE), scale=(0.8, 1.0), ratio=(0.9, 1.1)
        ),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.05),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=_mean, std=_std),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ]
)

val_transform = eval_transform

train_dataset = CassavaTrainDataset(tr_df, train_dir, transform=train_transform)
val_dataset = CassavaTrainDataset(val_df, train_dir, transform=val_transform)

g_loader = torch.Generator().manual_seed(3407)

_common_loader_kwargs = dict(
    num_workers=num_workers,
    pin_memory=(device.type == "cuda"),
    pin_memory_device="cuda" if device.type == "cuda" else "",
    worker_init_fn=seed_worker,
    persistent_workers=(num_workers > 0),
    prefetch_factor=4 if num_workers > 0 else None,
    in_order=False if num_workers > 0 else True,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    generator=g_loader,
    **{k: v for k, v in _common_loader_kwargs.items() if v not in (None, "")},
)
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in _common_loader_kwargs.items() if v not in (None, "")},
)

label_counts = (
    tr_df["label"].value_counts().reindex(range(num_classes), fill_value=0).values
)
label_counts = torch.tensor(label_counts, dtype=torch.float32)
class_weights = (label_counts.sum() / torch.clamp(label_counts, min=1.0)).to(device)
class_weights = class_weights / class_weights.mean()

for p in model.parameters():
    p.requires_grad = False
for p in model.layer3.parameters():
    p.requires_grad = True
for p in model.layer4.parameters():
    p.requires_grad = True
for p in model.fc.parameters():
    p.requires_grad = True

criterion = torch.nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

optimizer = torch.optim.AdamW(
    list(model.layer3.parameters())
    + list(model.layer4.parameters())
    + list(model.fc.parameters()),
    lr=2e-4,
    weight_decay=1e-2,
)

epochs = 4
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)


def evaluate_accuracy(m, loader):
    m.eval()
    correct = 0
    seen = 0
    with torch.inference_mode():
        for imgs, labels in loader:
            imgs = imgs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            logits = m(imgs)
            pred = logits.argmax(1)
            correct += (pred == labels).sum().item()
            seen += labels.size(0)
    return correct / max(seen, 1)


best_val = -1.0
best_state = None

for epoch in range(epochs):
    running_loss = 0.0
    seen = 0
    correct = 0

    model.train()
    for imgs, labels in train_loader:
        imgs = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        logits = model(imgs)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        running_loss += float(loss.item()) * labels.size(0)
        seen += labels.size(0)
        correct += (logits.argmax(1) == labels).sum().item()

    scheduler.step()

    train_acc = correct / max(seen, 1)
    val_acc = evaluate_accuracy(model, val_loader)

    if val_acc > best_val:
        best_val = val_acc
        best_state = {
            k: v.detach().cpu().clone() for k, v in model.state_dict().items()
        }

    print(
        f"epoch {epoch+1}/{epochs} - loss {running_loss/seen:.4f} - train_acc {train_acc:.4f} - val_acc {val_acc:.4f} - best_val {best_val:.4f}"
    )

if best_state is not None:
    model.load_state_dict(best_state)
model.eval()




## === cell 3
test_dataset = CassavaTestDataset(test_dir, transform=eval_transform)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    **{k: v for k, v in _common_loader_kwargs.items() if v not in (None, "")},
)

all_names = []
all_preds = []

model.eval()
with torch.inference_mode():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        logits = model(inputs)
        pred_labels = torch.argmax(logits, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("preds:", len(all_preds), "names:", len(all_names))




## === cell 4
sample_sub = pd.read_csv(sample_sub_path)
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", merged.shape)
merged.head()
