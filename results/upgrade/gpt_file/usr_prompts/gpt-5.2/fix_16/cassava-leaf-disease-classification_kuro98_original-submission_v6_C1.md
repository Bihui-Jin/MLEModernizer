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
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2
from torchvision import models

SEED = 3407
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

cudnn.deterministic = True
cudnn.benchmark = False

torch.set_num_threads(max(1, min(8, os.cpu_count() or 1)))
torch.set_num_interop_threads(1)

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

try:
    torch.backends.cuda.enable_flash_sdp(True)
    torch.backends.cuda.enable_mem_efficient_sdp(True)
    torch.backends.cuda.enable_math_sdp(True)
except Exception:
    pass

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
train_dir = f"{DATA_DIR}/train_images/"
test_dir = f"{DATA_DIR}/test_images/"
train_csv_path = f"{DATA_DIR}/train.csv"
sample_path = f"{DATA_DIR}/sample_submission.csv"

batch_size = 32
num_workers = int(max(2, min(8, (os.cpu_count() or 4) // 2)))

num_classes = 5
tta = True

vit_weights = models.ViT_L_16_Weights.IMAGENET1K_SWAG_E2E_V1

img_size = int(getattr(vit_weights.transforms(), "crop_size", (512, 512))[0])
print("Using img_size =", img_size)

vit_imagenet = models.vit_l_16(weights=vit_weights).to(device)
if device.type == "cuda":
    vit_imagenet = vit_imagenet.to(memory_format=torch.channels_last)
vit_imagenet.eval()




## === cell 1
class CassavaDataset(VisionDataset):
    """Custom dataset returning (image_tensor, filename)."""

    def __init__(self, data_dir, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.images = sorted(
            [f for f in os.listdir(data_dir) if f.lower().endswith(".jpg")]
        )

    def __getitem__(self, idx):
        filename = self.images[idx]
        path = os.path.join(self.root, filename)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, filename

    def __len__(self):
        return len(self.images)


class CassavaTrainDataset(VisionDataset):
    """Train dataset returning (image_tensor, label). Only used to build class prototypes."""

    def __init__(self, data_dir, df, transform=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.df = df.reset_index(drop=True)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        filename = row["image_id"]
        y = int(row["label"])
        path = os.path.join(self.root, filename)
        with Image.open(path) as im:
            img = im.convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, y

    def __len__(self):
        return len(self.df)




## === cell 2
mean = vit_weights.transforms().mean
std = vit_weights.transforms().std

test_transforms = v2.Compose(
    [
        v2.Resize((img_size, img_size), interpolation=InterpolationMode.BICUBIC),
        v2.CenterCrop((img_size, img_size)),
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=mean, std=std),
    ]
)

if tta:
    ttas = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(180, 180)),
        v2.RandomRotation(degrees=(270, 270)),
    ]
else:
    ttas = None

if tta:
    proto_views = [
        v2.Identity(),
        v2.RandomHorizontalFlip(p=1.0),
        v2.RandomVerticalFlip(p=1.0),
        v2.RandomRotation(degrees=(90, 90)),
        v2.RandomRotation(degrees=(180, 180)),
        v2.RandomRotation(degrees=(270, 270)),
    ]
else:
    proto_views = [v2.Identity()]

train_df = pd.read_csv(train_csv_path)
train_dataset = CassavaTrainDataset(train_dir, df=train_df, transform=test_transforms)
test_dataset = CassavaDataset(test_dir, transform=test_transforms)

loader_kwargs = dict(
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
)
if num_workers > 0:
    loader_kwargs.update(dict(persistent_workers=True, prefetch_factor=4))

train_loader = DataLoader(train_dataset, **loader_kwargs)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    drop_last=False,
    persistent_workers=(num_workers > 0),
    prefetch_factor=(4 if num_workers > 0 else 2),
)




## === cell 3
@torch.no_grad()
def vit_embeddings(model, x: torch.Tensor) -> torch.Tensor:
    """
    Extract the normalized CLS embedding using torchvision ViT internals:
      _process_input -> prepend class token -> encoder -> take CLS -> final LayerNorm.
    """
    x = model._process_input(x)  # (B, num_patches, hidden_dim)
    n = x.shape[0]

    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)  # (B, 1+num_patches, hidden_dim)

    x = model.encoder(x)  # (B, 1+num_patches, hidden_dim)
    x = x[:, 0]

    if hasattr(model, "ln"):
        x = model.ln(x)

    x = torch.nn.functional.normalize(x, p=2, dim=1)
    return x


embed_dim = int(
    getattr(vit_imagenet, "hidden_dim", None)
    or getattr(vit_imagenet, "hidden_size", None)
    or vit_imagenet.heads.head.in_features
)

prototypes_sum = torch.zeros(
    (num_classes, embed_dim), device=device, dtype=torch.float32
)
prototypes_count = torch.zeros((num_classes,), device=device, dtype=torch.float32)

feat_sum = torch.zeros((embed_dim,), device=device, dtype=torch.float32)
feat_sumsq = torch.zeros((embed_dim,), device=device, dtype=torch.float32)
feat_count_total = torch.zeros((), device=device, dtype=torch.float32)

normalize = torch.nn.functional.normalize
proto_views_list = (
    proto_views if (proto_views is not None and len(proto_views) > 1) else None
)
V_proto = len(proto_views) if proto_views_list is not None else 1

_view_buf = None

vit_imagenet.eval()
with torch.inference_mode():
    for inputs, labels in train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        if device.type == "cuda":
            inputs = inputs.to(memory_format=torch.channels_last)

        if proto_views_list is not None:
            B, C, H, W = inputs.shape
            need = (V_proto * B, C, H, W)
            if (
                (_view_buf is None)
                or (_view_buf.shape != need)
                or (_view_buf.device != inputs.device)
            ):
                _view_buf = torch.empty(
                    need,
                    device=inputs.device,
                    dtype=inputs.dtype,
                    memory_format=(
                        torch.channels_last
                        if device.type == "cuda"
                        else torch.contiguous_format
                    ),
                )
            for vi, t in enumerate(proto_views_list):
                _view_buf[vi * B : (vi + 1) * B].copy_(t(inputs))
            feats = vit_embeddings(vit_imagenet, _view_buf)  # (V*B, D)
            feats = feats.view(V_proto, B, -1).mean(dim=0)  # (B, D)
            feats = normalize(feats, p=2, dim=1)
        else:
            feats = vit_embeddings(vit_imagenet, inputs)  # (B, D)

        prototypes_sum.index_add_(0, labels, feats)
        prototypes_count.index_add_(
            0, labels, torch.ones_like(labels, dtype=torch.float32, device=device)
        )

        feat_sum += feats.sum(dim=0)
        feat_sumsq += (feats * feats).sum(dim=0)
        feat_count_total += float(feats.shape[0])

prototypes = prototypes_sum / prototypes_count.clamp_min(1.0).unsqueeze(1)

global_proto = prototypes.mean(dim=0, keepdim=True)  # (1, D)

base_alpha = 0.05
max_count = prototypes_count.max().clamp_min(1.0)
alpha_per_class = base_alpha * (1.0 - (prototypes_count / max_count))  # (C,)
alpha_per_class = alpha_per_class.clamp(0.0, base_alpha).unsqueeze(1)  # (C, 1)
prototypes = (1.0 - alpha_per_class) * prototypes + alpha_per_class * global_proto

mu = feat_sum / feat_count_total.clamp_min(1.0)
var = feat_sumsq / feat_count_total.clamp_min(1.0) - mu * mu
std_diag = torch.sqrt(var.clamp_min(1e-6))  # (D,)

prototypes = (prototypes - mu.unsqueeze(0)) / std_diag.unsqueeze(0)
prototypes = normalize(prototypes, p=2, dim=1)

print(
    "Built prototypes with counts:",
    prototypes_count.detach().cpu().numpy().astype(int).tolist(),
)



## === cell 4
all_names = []
all_preds = []

normalize = torch.nn.functional.normalize
ttas_list = ttas if ttas is not None else None
T_tta = len(ttas) if ttas_list is not None else 1

_tta_buf = None

vit_imagenet.eval()
with torch.inference_mode():
    for inputs, filenames in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        if device.type == "cuda":
            inputs = inputs.to(memory_format=torch.channels_last)

        if ttas_list is not None:
            B, C, H, W = inputs.shape
            need = (T_tta * B, C, H, W)
            if (
                (_tta_buf is None)
                or (_tta_buf.shape != need)
                or (_tta_buf.device != inputs.device)
            ):
                _tta_buf = torch.empty(
                    need,
                    device=inputs.device,
                    dtype=inputs.dtype,
                    memory_format=(
                        torch.channels_last
                        if device.type == "cuda"
                        else torch.contiguous_format
                    ),
                )
            for ti, t in enumerate(ttas_list):
                _tta_buf[ti * B : (ti + 1) * B].copy_(t(inputs))
            feats = vit_embeddings(vit_imagenet, _tta_buf)  # (T*B, D)
            feats = feats.view(T_tta, B, -1).mean(dim=0)  # (B, D)
            feats = normalize(feats, p=2, dim=1)
        else:
            feats = vit_embeddings(vit_imagenet, inputs)  # (B, D)

        feats = (feats - mu.unsqueeze(0)) / std_diag.unsqueeze(0)
        feats = normalize(feats, p=2, dim=1)

        sims = feats @ prototypes.T  # (B, 5)
        pred_labels = torch.argmax(sims, dim=1).to(torch.int64).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)



## === cell 5
sample_sub = pd.read_csv(sample_path)
pred_map = dict(zip(all_names, all_preds))

missing = [
    img_id for img_id in sample_sub["image_id"].tolist() if img_id not in pred_map
]
if missing:
    for m in missing:
        pred_map[m] = 0

my_submission = sample_sub.copy()
my_submission["label"] = my_submission["image_id"].map(pred_map).astype(int)

assert len(my_submission) == len(sample_sub)
assert my_submission["image_id"].isna().sum() == 0
assert my_submission["label"].isna().sum() == 0

my_submission.to_csv("submission.csv", index=False)
print("Wrote submission.csv with shape:", my_submission.shape)



## === cell 6
my_submission.head(10)
