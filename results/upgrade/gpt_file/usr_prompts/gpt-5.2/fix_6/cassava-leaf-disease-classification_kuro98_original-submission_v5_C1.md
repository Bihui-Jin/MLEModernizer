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
from torchvision.transforms import v2



## === cell 1
torch.manual_seed(3407)
if torch.cuda.is_available():
    torch.cuda.manual_seed(3407)

cudnn.deterministic = True
cudnn.benchmark = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

DATA_ROOT = "/kaggle/input/cassava-leaf-disease-classification"
train_csv_path = f"{DATA_ROOT}/train.csv"
train_dir = f"{DATA_ROOT}/train_images/"
test_dir = f"{DATA_ROOT}/test_images/"
sample_sub_path = f"{DATA_ROOT}/sample_submission.csv"

batch_size = 32
num_workers = 4
num_classes = 5
tta = False

use_amp = torch.cuda.is_available()
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)

from torchvision.models import vit_b_16, ViT_B_16_Weights

weights = ViT_B_16_Weights.IMAGENET1K_V1
vit_model = vit_b_16(weights=weights)

vit_model.heads.head = torch.nn.Linear(vit_model.heads.head.in_features, num_classes)

for p in vit_model.parameters():
    p.requires_grad = False
for p in vit_model.heads.head.parameters():
    p.requires_grad = True

vit_model.to(device)




## === cell 2
class CassavaDataset(VisionDataset):
    """Dataset for Cassava images.

    - robust RGB loading
    - supports optional labels (train) and deterministic ordering (test)
    """

    def __init__(self, data_dir, images, labels=None, transform=None, ttas=None):
        super().__init__(root=data_dir)
        self.transform = transform
        self.ttas = ttas
        self.images = list(images)
        self.labels = None if labels is None else list(labels)

    def __getitem__(self, idx):
        filename = self.images[idx]
        img_path = os.path.join(self.root, filename)
        img = Image.open(img_path).convert("RGB")

        if self.ttas is not None and self.transform is not None:
            img = [self.transform(t(img)) for t in self.ttas]
        elif self.transform:
            img = self.transform(img)

        if self.labels is None:
            return img, filename
        return img, int(self.labels[idx])

    def __len__(self):
        return len(self.images)




## === cell 3
train_transforms = weights.transforms(antialias=True)
test_transforms = weights.transforms(antialias=True)

if tta:
    ttas = [
        v2.RandomResizedCrop((224, 224), (0.5, 1)),
        v2.RandomRotation(180),
        v2.RandomVerticalFlip(1),
        v2.RandomAffine(180),
        v2.RandomPerspective(p=1),
    ]
else:
    ttas = None

train_df = pd.read_csv(train_csv_path)

sample_sub = pd.read_csv(sample_sub_path)
test_images_ordered = sample_sub["image_id"].tolist()
test_dataset = CassavaDataset(
    test_dir,
    images=test_images_ordered,
    labels=None,
    transform=test_transforms,
    ttas=ttas,
)
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,  # maintain 1:1 alignment and stable ordering
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

normalizer = torch.nn.Softmax(dim=1)




## === cell 4
def _stratified_split_indices(labels, val_frac=0.1, seed=3407):
    g = torch.Generator()
    g.manual_seed(seed)
    labels = torch.as_tensor(labels, dtype=torch.long)
    idx_all = torch.arange(labels.numel())

    tr_idx = []
    va_idx = []
    for c in range(int(labels.max().item()) + 1):
        cls_idx = idx_all[labels == c]
        if cls_idx.numel() == 0:
            continue
        perm = cls_idx[torch.randperm(cls_idx.numel(), generator=g)]
        n_val = max(1, int(round(val_frac * perm.numel())))
        va_idx.append(perm[:n_val])
        tr_idx.append(perm[n_val:])
    tr_idx = torch.cat(tr_idx).tolist()
    va_idx = torch.cat(va_idx).tolist()
    return tr_idx, va_idx


def _make_loader(df, indices, shuffle):
    sub = df.iloc[indices].reset_index(drop=True)
    ds = CassavaDataset(
        train_dir,
        images=sub["image_id"].tolist(),
        labels=sub["label"].tolist(),
        transform=train_transforms,
        ttas=None,
    )
    return DataLoader(
        ds,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


def _forward_frozen_backbone(model, x):
    x = model._process_input(x)
    n = x.shape[0]
    batch_class_token = model.class_token.expand(n, -1, -1)
    x = torch.cat([batch_class_token, x], dim=1)

    with torch.no_grad():
        x = model.encoder(x)

    x = x[:, 0]
    x = model.heads(x)
    return x


def _eval_acc(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = model(inputs)
            preds = torch.argmax(logits, dim=1)
            correct += int((preds == labels).sum().item())
            total += int(labels.size(0))
    return correct / max(total, 1)


criterion = torch.nn.CrossEntropyLoss()

tr_idx, va_idx = _stratified_split_indices(
    train_df["label"].tolist(), val_frac=0.1, seed=3407
)
train_loader_split = _make_loader(train_df, tr_idx, shuffle=True)
val_loader_split = _make_loader(train_df, va_idx, shuffle=False)


def _reset_head_parameters():
    torch.manual_seed(3407)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(3407)
    vit_model.heads.head.reset_parameters()


candidates = [
    {"lr": 3e-4, "epochs": 3},
    {"lr": 2e-4, "epochs": 4},
]

best = None
best_acc = -1.0

for cfg in candidates:
    _reset_head_parameters()
    optimizer = torch.optim.AdamW(
        vit_model.heads.head.parameters(), lr=cfg["lr"], weight_decay=1e-2
    )

    vit_model.eval()
    vit_model.heads.head.train()

    for epoch in range(cfg["epochs"]):
        running_loss = 0.0
        total = 0
        for inputs, labels in train_loader_split:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=use_amp):
                logits = _forward_frozen_backbone(vit_model, inputs)
                loss = criterion(logits, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            running_loss += float(loss.item()) * labels.size(0)
            total += int(labels.size(0))

        print(
            f"trial lr={cfg['lr']} epoch={epoch+1}/{cfg['epochs']} train_loss={running_loss/max(total,1):.4f}"
        )

    val_acc = _eval_acc(vit_model, val_loader_split)
    print(f"trial lr={cfg['lr']} epochs={cfg['epochs']} val_acc={val_acc:.5f}")

    if val_acc > best_acc:
        best_acc = val_acc
        best = cfg

print("Selected head-train config:", best, "val_acc=", best_acc)

_reset_head_parameters()
full_train_dataset = CassavaDataset(
    train_dir,
    images=train_df["image_id"].tolist(),
    labels=train_df["label"].tolist(),
    transform=train_transforms,
    ttas=None,
)
full_train_loader = DataLoader(
    full_train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=torch.cuda.is_available(),
)

optimizer = torch.optim.AdamW(
    vit_model.heads.head.parameters(), lr=best["lr"], weight_decay=1e-2
)

vit_model.eval()
vit_model.heads.head.train()

for epoch in range(best["epochs"]):
    running_loss = 0.0
    correct = 0
    total = 0
    for inputs, labels in full_train_loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.amp.autocast("cuda", enabled=use_amp):
            logits = _forward_frozen_backbone(vit_model, inputs)
            loss = criterion(logits, labels)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += float(loss.item()) * labels.size(0)
        preds = torch.argmax(logits, dim=1)
        correct += int((preds == labels).sum().item())
        total += int(labels.size(0))

    print(
        f"final epoch={epoch+1}/{best['epochs']} "
        f"train_loss={running_loss/max(total,1):.4f} "
        f"train_acc={correct/max(total,1):.4f}"
    )



## === cell 5
all_names = []
all_preds = []

vit_model.eval()

with torch.no_grad():
    for batch_idx, (inputs, filenames) in enumerate(test_loader):
        if tta:
            n_imgs = len(filenames)
            n_tta = len(ttas)
            inputs = torch.cat(inputs, dim=0).to(device, non_blocking=True)
            filenames = list(filenames)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = normalizer(vit_model(inputs))

            preds = preds.view(n_tta, n_imgs, -1).mean(dim=0)
            pred_labels = torch.argmax(preds, 1).tolist()
        else:
            inputs = inputs.to(device, non_blocking=True)
            filenames = list(filenames)

            with torch.amp.autocast("cuda", enabled=use_amp):
                preds = normalizer(vit_model(inputs))
            pred_labels = torch.argmax(preds, 1).tolist()

        all_names.extend(filenames)
        all_preds.extend(pred_labels)



## === cell 6
pred_df = pd.DataFrame({"image_id": all_names, "label": all_preds})

pred_df = sample_sub[["image_id"]].merge(pred_df, on="image_id", how="left")
pred_df["label"] = pred_df["label"].fillna(0).astype(int)

pred_df.to_csv("submission.csv", index=False)

print("Wrote submission.csv with shape:", pred_df.shape)
print(pred_df.head())



## === cell 7
pred_df
