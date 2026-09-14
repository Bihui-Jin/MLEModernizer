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
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2

torch.manual_seed(3407)
torch.cuda.manual_seed(3407)
random.seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

data_root = "/kaggle/input/cassava-leaf-disease-classification/"
train_csv_path = os.path.join(data_root, "train.csv")
train_dir = os.path.join(data_root, "train_images/")
test_dir = os.path.join(data_root, "test_images/")
sample_sub_path = os.path.join(data_root, "sample_submission.csv")

eff_img_size = 224
vit_img_size = 224

batch_size = 16
num_workers = 4
num_classes = 5
tta = True

train_head = True
head_epochs = 10
head_lr = 2e-3

label_smoothing = 0.05

val_frac = 0.10
split_seed = 3407

import torchvision

vit_weights = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
eff_weights = torchvision.models.EfficientNet_B0_Weights.IMAGENET1K_V1

vit_model = torchvision.models.vit_b_16(weights=vit_weights).to(device)
eff_model = torchvision.models.efficientnet_b0(weights=eff_weights).to(device)

linear_head = torch.nn.Linear(768 + 1280, num_classes).to(device)

use_amp = torch.cuda.is_available()
amp_dtype = torch.float16


def seed_worker(worker_id: int):
    worker_seed = (torch.initial_seed() + worker_id) % 2**32
    random.seed(worker_seed)
    torch.manual_seed(worker_seed)


g = torch.Generator()
g.manual_seed(3407)




## === cell 1
class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (vit_img, eff_img, label)."""

    def __init__(self, df, data_dir, vit_size, efficient_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.train_crop = v2.RandomResizedCrop(
            size=(600, 600),
            scale=(0.70, 1.0),
            ratio=(0.90, 1.10),
            interpolation=InterpolationMode.BICUBIC,
        )

        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.train_crop(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaValDataset(VisionDataset):
    """Validation dataset returning (vit_img, eff_img, label) with deterministic center-crop."""

    def __init__(self, df, data_dir, vit_size, efficient_size, transform=None):
        super().__init__(root=data_dir)
        self.df = df.reset_index(drop=True)
        self.transform = transform

        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        label = int(row["label"])

        img = Image.open(os.path.join(self.root, filename)).convert("RGB")
        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.transform:
            vit_img = self.transform(vit_img)
            eff_img = self.transform(eff_img)

        return vit_img, eff_img, torch.tensor(label, dtype=torch.long)

    def __len__(self):
        return len(self.df)


class CassavaDataset(VisionDataset):
    """Custom dataset for the Cassava test data.

    Returns:
        vit_img: PIL.Image or list[PIL.Image] if TTA enabled
        eff_img: PIL.Image or list[PIL.Image] if TTA enabled
        filename: image filename
    """

    def __init__(self, data_dir, vit_size, efficient_size, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = (
            transform  # kept for backward-compat; collate applies normalization
        )
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

        self.ttas = ttas
        self.cc = v2.CenterCrop((600, 600))
        self.resize_vit = v2.Resize(
            (vit_size, vit_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_efficient = v2.Resize(
            (efficient_size, efficient_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        vit_img = self.resize_vit(img)
        eff_img = self.resize_efficient(img)

        if self.ttas is not None:
            vit_img = [t(vit_img) for t in self.ttas]
            eff_img = [t(eff_img) for t in self.ttas]

        return vit_img, eff_img, filename

    def __len__(self):
        return len(self.images)




## === cell 2
def _norm_only_from_weights(weights):
    mean = weights.meta.get("mean", [0.485, 0.456, 0.406])
    std = weights.meta.get("std", [0.229, 0.224, 0.225])
    return v2.Compose(
        [
            v2.ToImage(),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=mean, std=std),
        ]
    )


vit_transforms = _norm_only_from_weights(vit_weights)
eff_transforms = _norm_only_from_weights(eff_weights)


class DualTransform:
    def __init__(self, vit_t, eff_t):
        self.vit_t = vit_t
        self.eff_t = eff_t

    def __call__(self, x):
        return x


dual_transform = DualTransform(vit_transforms, eff_transforms)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
    ]
else:
    ttas = None

train_df_full = pd.read_csv(train_csv_path)

train_parts = []
val_parts = []
for lbl, grp in train_df_full.groupby("label", sort=False):
    grp = grp.sample(frac=1.0, random_state=split_seed).reset_index(drop=True)
    n_val = max(1, int(round(len(grp) * val_frac)))
    val_parts.append(grp.iloc[:n_val])
    train_parts.append(grp.iloc[n_val:])
train_df = (
    pd.concat(train_parts, axis=0)
    .sample(frac=1.0, random_state=split_seed)
    .reset_index(drop=True)
)
val_df = pd.concat(val_parts, axis=0).reset_index(drop=True)

print("train/val sizes:", len(train_df), len(val_df))
print(
    "train label dist:\n", train_df["label"].value_counts(normalize=True).sort_index()
)
print("val label dist:\n", val_df["label"].value_counts(normalize=True).sort_index())

train_dataset = CassavaTrainDataset(
    train_df, train_dir, vit_img_size, eff_img_size, transform=None
)
val_dataset = CassavaValDataset(
    val_df, train_dir, vit_img_size, eff_img_size, transform=None
)
test_dataset = CassavaDataset(
    test_dir, vit_img_size, eff_img_size, transform=None, ttas=ttas
)


def train_collate(batch):
    vit_imgs, eff_imgs, labels = zip(*batch)
    vit_imgs = torch.stack([dual_transform.vit_t(im) for im in vit_imgs], dim=0)
    eff_imgs = torch.stack([dual_transform.eff_t(im) for im in eff_imgs], dim=0)
    labels = torch.stack(labels, dim=0)
    return vit_imgs, eff_imgs, labels


def test_collate(batch):
    vit_items, eff_items, fnames = zip(*batch)
    if tta:
        T = len(vit_items[0])
        vit_out = [
            torch.stack(
                [dual_transform.vit_t(vit_items[b][t]) for b in range(len(batch))],
                dim=0,
            )
            for t in range(T)
        ]
        eff_out = [
            torch.stack(
                [dual_transform.eff_t(eff_items[b][t]) for b in range(len(batch))],
                dim=0,
            )
            for t in range(T)
        ]
        return vit_out, eff_out, list(fnames)
    else:
        vit_imgs = torch.stack([dual_transform.vit_t(im) for im in vit_items], dim=0)
        eff_imgs = torch.stack([dual_transform.eff_t(im) for im in eff_items], dim=0)
        return vit_imgs, eff_imgs, list(fnames)


train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=train_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    collate_fn=test_collate,
    worker_init_fn=seed_worker,
    generator=g,
    persistent_workers=(num_workers > 0),
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 3
def vit_features(x: torch.Tensor) -> torch.Tensor:
    if hasattr(vit_model, "forward_features"):
        out = vit_model.forward_features(x)
        return out[:, 0] if out.ndim == 3 else out

    n = x.shape[0]
    x = vit_model._process_input(x)  # [B, seq_len, hidden_dim]
    cls = vit_model.class_token.expand(n, -1, -1)  # [B,1,hidden_dim]
    x = torch.cat([cls, x], dim=1)  # [B, 1+seq_len, hidden_dim]
    x = vit_model.encoder(x)  # [B, 1+seq_len, hidden_dim]
    return x[:, 0]  # CLS token [B,768]


def eff_features(x: torch.Tensor) -> torch.Tensor:
    return eff_model.features(x).mean(dim=(2, 3))


for p in vit_model.parameters():
    p.requires_grad = False
for p in eff_model.parameters():
    p.requires_grad = False

vit_model.eval()
eff_model.eval()

best_state = None
best_val_acc = -1.0

if train_head:
    linear_head.train()
    optimizer = torch.optim.AdamW(
        linear_head.parameters(), lr=head_lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, threshold=1e-4, min_lr=1e-5
    )

    criterion = torch.nn.CrossEntropyLoss(label_smoothing=label_smoothing)

    for epoch in range(head_epochs):
        running_loss = 0.0
        running_correct = 0
        running_total = 0

        for vit_inputs, eff_inputs, labels in train_loader:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            with torch.no_grad(), torch.autocast(
                device_type="cuda", dtype=amp_dtype, enabled=use_amp
            ):
                vit_out = vit_features(vit_inputs)  # [B,768]
                eff_out = eff_features(eff_inputs)  # [B,1280]

            logit_inputs = torch.cat(
                [vit_out.float(), eff_out.float()], dim=1
            )  # [B,2048]
            outputs = linear_head(logit_inputs)  # [B,5]

            loss = criterion(outputs, labels)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            running_loss += float(loss.item()) * labels.size(0)
            preds = outputs.argmax(dim=1)
            running_correct += int((preds == labels).sum().item())
            running_total += int(labels.size(0))

        train_loss = running_loss / max(running_total, 1)
        train_acc = running_correct / max(running_total, 1)

        linear_head.eval()
        val_correct = 0
        val_total = 0
        with torch.no_grad():
            for vit_inputs, eff_inputs, labels in val_loader:
                vit_inputs = vit_inputs.to(device, non_blocking=True)
                eff_inputs = eff_inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                with torch.autocast(
                    device_type="cuda", dtype=amp_dtype, enabled=use_amp
                ):
                    vit_out = vit_features(vit_inputs)
                    eff_out = eff_features(eff_inputs)

                logit_inputs = torch.cat([vit_out.float(), eff_out.float()], dim=1)
                outputs = linear_head(logit_inputs)
                preds = outputs.argmax(dim=1)
                val_correct += int((preds == labels).sum().item())
                val_total += int(labels.size(0))

        val_acc = val_correct / max(val_total, 1)
        scheduler.step(val_acc)

        if val_acc > best_val_acc:
            best_val_acc = val_acc
            best_state = {
                k: v.detach().cpu().clone() for k, v in linear_head.state_dict().items()
            }

        current_lr = optimizer.param_groups[0]["lr"]
        print(
            f"epoch {epoch+1}/{head_epochs} "
            f"lr={current_lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_acc={val_acc:.4f} best_val_acc={best_val_acc:.4f}"
        )
        linear_head.train()

linear_head.eval()
if best_state is not None:
    linear_head.load_state_dict(best_state)



## === cell 4
all_names = []
all_preds = []

vit_model.eval()
eff_model.eval()
linear_head.eval()

with torch.no_grad():
    for batch_idx, (vit_inputs, eff_inputs, filenames) in enumerate(test_loader):
        base_batch_size = len(filenames)

        if tta:
            vit_inputs = torch.cat(vit_inputs, dim=0).to(device, non_blocking=True)
            eff_inputs = torch.cat(eff_inputs, dim=0).to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_out = vit_features(vit_inputs)  # [T*B,768]
                eff_out = eff_features(eff_inputs)  # [T*B,1280]

            vit_out = vit_out.float()
            eff_out = eff_out.float()

            vit_batch = torch.stack(
                torch.split(vit_out, base_batch_size), dim=0
            )  # [T,B,768]
            vit_mean = torch.mean(vit_batch, dim=0)  # [B,768]

            eff_batch = torch.stack(
                torch.split(eff_out, base_batch_size), dim=0
            )  # [T,B,1280]
            eff_mean = torch.mean(eff_batch, dim=0)  # [B,1280]

            logit_inputs = torch.cat([vit_mean, eff_mean], dim=1)  # [B,2048]
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()
        else:
            vit_inputs = vit_inputs.to(device, non_blocking=True)
            eff_inputs = eff_inputs.to(device, non_blocking=True)

            with torch.autocast(device_type="cuda", dtype=amp_dtype, enabled=use_amp):
                vit_out = vit_features(vit_inputs)
                eff_out = eff_features(eff_inputs)

            logit_inputs = torch.cat([vit_out.float(), eff_out.float()], dim=1)
            outputs = linear_head(logit_inputs)

            probs = normalizer(outputs)
            pred_labels = torch.argmax(probs, 1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

print("Preds:", len(all_preds), "Names:", len(all_names))



## === cell 5
sample_sub = pd.read_csv(sample_sub_path)

pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

merged = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
if merged["label"].isna().any():
    fill_label = int(pred_df["label"].mode().iloc[0]) if len(pred_df) else 0
    merged["label"] = merged["label"].fillna(fill_label).astype(int)
else:
    merged["label"] = merged["label"].astype(int)

merged.to_csv("submission.csv", index=False)
print(merged.head())
print("submission.csv rows:", len(merged))
print("label value counts:\n", merged["label"].value_counts(dropna=False).sort_index())
