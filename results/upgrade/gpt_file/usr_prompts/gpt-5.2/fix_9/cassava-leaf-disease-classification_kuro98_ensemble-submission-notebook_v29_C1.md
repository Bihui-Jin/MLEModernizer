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
from PIL import Image
from torch.backends import cudnn
from torch.utils.data import DataLoader
from torchvision.datasets import VisionDataset
from torchvision.transforms import InterpolationMode, v2



## === cell 1
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = False
cudnn.benchmark = True
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_DIR = "/kaggle/input/cassava-leaf-disease-classification"
test_dir = f"{DATA_DIR}/test_images/"
train_dir = f"{DATA_DIR}/train_images/"
sample_sub_path = f"{DATA_DIR}/sample_submission.csv"
train_csv_path = f"{DATA_DIR}/train.csv"

model_a_img_size = 224
model_b_img_size = 384

batch_size = 32
num_workers = 4
num_classes = 5
tta = False

import torchvision

weights_a = torchvision.models.ViT_B_16_Weights.IMAGENET1K_V1
weights_b = torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1

backbone_a = torchvision.models.vit_b_16(weights=weights_a).to(device)
backbone_b = torchvision.models.vit_b_16(weights=weights_b).to(device)

in_features = backbone_a.heads.head.in_features
backbone_a.heads.head = torch.nn.Identity()
backbone_b.heads.head = torch.nn.Identity()

linear_head = torch.nn.Linear(in_features, num_classes, bias=True).to(device)

model_a = backbone_a
model_b = backbone_b

use_amp = torch.cuda.is_available()

ens_w_a, ens_w_b = 0.94, 0.06




## === cell 2
class CassavaDataset(VisionDataset):
    """Custom dataset for Cassava data (train or test).

    For test: uses image_ids from sample_submission.csv to guarantee exact order.
    For train: uses image_ids from train.csv.
    """

    def __init__(
        self,
        data_dir,
        model_a_size,
        model_b_size,
        image_ids,
        labels=None,
        transform=None,
        ttas=None,
    ):
        super().__init__(root=data_dir)
        self.transform = transform
        self.image_ids = list(image_ids)
        self.labels = None if labels is None else list(labels)
        self.ttas = ttas

        self.cc = v2.CenterCrop((600, 600))
        self.resize_model_a = v2.Resize(
            (model_a_size, model_a_size), interpolation=InterpolationMode.BICUBIC
        )
        self.resize_model_b = v2.Resize(
            (model_b_size, model_b_size), interpolation=InterpolationMode.BICUBIC
        )

    def __getitem__(self, idx):
        filename = self.image_ids[idx]
        img = Image.open(os.path.join(self.root, filename)).convert("RGB")

        img = self.cc(img)
        model_a_img = self.resize_model_a(img)
        model_b_img = self.resize_model_b(img)

        if self.ttas is not None and self.transform is not None:
            model_a_img = [self.transform(t(model_a_img)) for t in self.ttas]
            model_b_img = [self.transform(t(model_b_img)) for t in self.ttas]
        elif self.transform is not None:
            model_a_img = self.transform(model_a_img)
            model_b_img = self.transform(model_b_img)

        if self.labels is None:
            return model_a_img, model_b_img, filename
        return model_a_img, model_b_img, int(self.labels[idx])

    def __len__(self):
        return len(self.image_ids)




## === cell 3
test_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ]
)

train_transforms = v2.Compose(
    [
        v2.ToImage(),
        v2.RandAugment(num_ops=2, magnitude=9),
        v2.RandomHorizontalFlip(p=0.5),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        v2.RandomErasing(p=0.25, scale=(0.02, 0.12), ratio=(0.3, 3.3), value=0.0),
    ]
)

if tta:
    ttas = [
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

sample_df = pd.read_csv(sample_sub_path)
test_image_ids = sample_df["image_id"].tolist()

train_df = pd.read_csv(train_csv_path)

from sklearn.model_selection import StratifiedShuffleSplit

sss = StratifiedShuffleSplit(n_splits=1, test_size=0.1, random_state=3407)
idx = torch.arange(len(train_df)).numpy()
train_idx, val_idx = next(sss.split(idx, train_df["label"].values))

train_split_df = train_df.iloc[train_idx].reset_index(drop=True)
val_split_df = train_df.iloc[val_idx].reset_index(drop=True)

train_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=train_split_df["image_id"].tolist(),
    labels=train_split_df["label"].tolist(),
    transform=train_transforms,
    ttas=None,
)

val_dataset = CassavaDataset(
    train_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=val_split_df["image_id"].tolist(),
    labels=val_split_df["label"].tolist(),
    transform=test_transforms,  # eval uses test-like preprocessing
    ttas=None,
)

test_dataset = CassavaDataset(
    test_dir,
    model_a_img_size,
    model_b_img_size,
    image_ids=test_image_ids,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)



## === cell 4
for p in model_a.parameters():
    p.requires_grad = False
for p in model_b.parameters():
    p.requires_grad = False

linear_head.train()
model_a.eval()
model_b.eval()

criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.05)

optimizer = torch.optim.AdamW(linear_head.parameters(), lr=2e-3, weight_decay=1e-2)

epochs = 6
steps_per_epoch = len(train_loader)
total_steps = max(1, epochs * steps_per_epoch)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=total_steps)

scaler = torch.cuda.amp.GradScaler(enabled=use_amp)

global_step = 0
for epoch in range(epochs):
    running_loss = 0.0
    correct = 0
    total = 0

    for model_a_inputs, model_b_inputs, y in train_loader:
        xa = model_a_inputs.to(device, non_blocking=True)
        xb = model_b_inputs.to(device, non_blocking=True)
        y = torch.as_tensor(y, device=device)

        optimizer.zero_grad(set_to_none=True)

        with torch.no_grad(), torch.cuda.amp.autocast(enabled=use_amp):
            feats_a = model_a(xa)  # (B, 768)
            feats_b = model_b(xb)  # (B, 768)
            feats = ens_w_a * feats_a + ens_w_b * feats_b

        with torch.cuda.amp.autocast(enabled=use_amp):
            logits = linear_head(feats)  # (B, 5)
            loss = criterion(logits, y)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        scheduler.step()
        global_step += 1

        running_loss += float(loss.item()) * xa.size(0)
        pred = torch.argmax(logits, dim=1)
        correct += (pred == y).sum().item()
        total += xa.size(0)

    linear_head.eval()
    val_correct = 0
    val_total = 0
    with torch.no_grad():
        for model_a_inputs, model_b_inputs, y in val_loader:
            xa = model_a_inputs.to(device, non_blocking=True)
            xb = model_b_inputs.to(device, non_blocking=True)
            y = torch.as_tensor(y, device=device)
            with torch.cuda.amp.autocast(enabled=use_amp):
                feats_a = model_a(xa)
                feats_b = model_b(xb)
                feats = ens_w_a * feats_a + ens_w_b * feats_b
                logits = linear_head(feats)
            pred = torch.argmax(logits, dim=1)
            val_correct += (pred == y).sum().item()
            val_total += xa.size(0)
    val_acc = val_correct / max(1, val_total)

    print(
        f"epoch {epoch+1}/{epochs} | loss={running_loss/total:.4f} | train_acc={correct/total:.4f} | val_acc={val_acc:.4f}"
    )
    linear_head.train()

linear_head.eval()
model_a.eval()
model_b.eval()

all_names = []
all_preds = []

with torch.no_grad():
    for model_a_inputs, model_b_inputs, filenames in test_loader:
        bs = len(filenames)

        if tta:
            model_a_inputs = torch.cat(model_a_inputs, dim=0).to(device)
            model_b_inputs = torch.cat(model_b_inputs, dim=0).to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_a_feats = model_a(model_a_inputs)
                model_b_feats = model_b(model_b_inputs)

            model_a_batch_feats = torch.stack(torch.split(model_a_feats, bs), dim=0)
            model_b_batch_feats = torch.stack(torch.split(model_b_feats, bs), dim=0)

            model_a_mean = torch.mean(model_a_batch_feats, dim=0)
            model_b_mean = torch.mean(model_b_batch_feats, dim=0)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_a_out = linear_head(model_a_mean)
                model_b_out = linear_head(model_b_mean)

            outputs = ens_w_a * model_a_out + ens_w_b * model_b_out
            pred_labels = torch.argmax(outputs, dim=1).tolist()
        else:
            model_a_inputs = model_a_inputs.to(device)
            model_b_inputs = model_b_inputs.to(device)

            with torch.cuda.amp.autocast(enabled=use_amp):
                model_a_feats = model_a(model_a_inputs)
                model_b_feats = model_b(model_b_inputs)

                model_a_out = linear_head(model_a_feats)
                model_b_out = linear_head(model_b_feats)

                outputs = ens_w_a * model_a_out + ens_w_b * model_b_out

            pred_labels = torch.argmax(outputs, dim=1).tolist()

        all_names.extend(list(filenames))
        all_preds.extend(pred_labels)

assert len(all_names) == len(sample_df), (len(all_names), len(sample_df))
assert len(all_preds) == len(sample_df), (len(all_preds), len(sample_df))

my_submission = pd.DataFrame({"image_id": all_names, "label": all_preds})

my_submission = sample_df[["image_id"]].merge(my_submission, on="image_id", how="left")
assert (
    my_submission["label"].isna().sum() == 0
), "Missing predictions for some test images."
my_submission["label"] = my_submission["label"].astype(int)

my_submission.to_csv("submission.csv", index=False)
my_submission
