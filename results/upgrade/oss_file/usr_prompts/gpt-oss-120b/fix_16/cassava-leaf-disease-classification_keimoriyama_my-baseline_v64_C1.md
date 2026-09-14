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
pandas==2.2.3
pandas-datareader==0.10.0
pandas-gbq==0.29.2
pandas-profiling==3.6.6
pandas-stubs==2.2.2.240909
pandasql==0.7.3
pillow==11.3.0
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
sklearn-pandas==2.2.0
timm==1.0.19
torch==2.6.0+cu124
torchao==0.10.0
torchaudio==2.6.0+cu124
torchdata==0.11.0
torchinfo==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
torchtune==0.6.1
torchvision==0.21.0+cu124
tqdm==4.67.1

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
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image, ImageDraw
from sklearn.model_selection import train_test_split

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import autocast, GradScaler

import timm
import torchvision.transforms as transforms

base_path = "../input/cassava-leaf-disease-classification"
train_csv_path = os.path.join(base_path, "train.csv")
train_images_path = os.path.join(base_path, "train_images")
test_images_path = os.path.join(base_path, "test_images")
nested = os.path.join(test_images_path, "test_images")
if os.path.isdir(nested):
    test_images_path = nested



## === cell 1
df = pd.read_csv(train_csv_path)
df["path"] = df["image_id"].map(lambda x: os.path.join(train_images_path, x))
df = df.sample(frac=1, random_state=42).reset_index(drop=True)



## === cell 2
train_df, valid_df = train_test_split(
    df, test_size=0.2, random_state=42, stratify=df["label"].values
)



## === cell 3
image_size = 256
mean = [0.485, 0.456, 0.406]
std = [0.229, 0.224, 0.225]

train_transform = transforms.Compose(
    [
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.5),
        transforms.RandomResizedCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)

valid_transform = transforms.Compose(
    [
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std),
    ]
)




## === cell 4
class CassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img, label


class CachedCassavaDataset(Dataset):
    def __init__(self, dataframe, transform=None):
        self.df = dataframe.reset_index(drop=True)
        self.transform = transform
        self._cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        if idx in self._cache:
            return self._cache[idx]
        path = self.df.loc[idx, "path"]
        label = self.df.loc[idx, "label"]
        with open(path, "rb") as f:
            img = Image.open(f).convert("RGB")
        if self.transform:
            img = self.transform(img)
        self._cache[idx] = (img, label)
        return img, label




## === cell 5
class make_mask_image:
    def __init__(self, p, mask_size=50):
        self.p = p
        self.mask_size = mask_size

    def __call__(self, image):
        if random.random() < self.p:
            draw = ImageDraw.Draw(image)
            w, h = image.size
            for _ in range(10):
                x = random.randrange(0, w - self.mask_size)
                y = random.randrange(0, self.mask_size)  # kept original logic
                draw.rectangle(
                    (x, y, x + self.mask_size, y + self.mask_size),
                    fill=(0, 0, 0),
                    outline=(0, 0, 0),
                )
        return image




## === cell 6
class Unnormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


unnorm = Unnormalize(mean, std)



## === cell 7
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True
data_loader_workers = min(8, os.cpu_count() or 1)
batch_size = 96
epoch_num = 1  # unchanged
num_classes = 5



## === cell 8
resNet = timm.create_model("resnet50", pretrained=True)
resNet.fc = nn.Linear(resNet.fc.in_features, num_classes)
resNet = resNet.to(device)

ef_model = timm.create_model("tf_efficientnet_b2_ns", pretrained=True)
ef_model.classifier = nn.Linear(ef_model.classifier.in_features, num_classes)
ef_model = ef_model.to(device)

if hasattr(torch, "compile"):
    resNet = torch.compile(resNet, mode="max-autotune")
    ef_model = torch.compile(ef_model, mode="max-autotune")



## === cell 9
resNet_optimizer = torch.optim.AdamW(resNet.parameters(), lr=1e-4, weight_decay=1e-4)
ef_optimizer = torch.optim.AdamW(ef_model.parameters(), lr=1e-4, weight_decay=1e-4)

resNet_scheduler = torch.optim.lr_scheduler.StepLR(
    resNet_optimizer, step_size=2, gamma=0.1
)
ef_scheduler = torch.optim.lr_scheduler.StepLR(ef_optimizer, step_size=2, gamma=0.1)

criterion = nn.CrossEntropyLoss()




## === cell 10
def calc_correction(model, dataset, batch_sz=64):
    """Batched validation accuracy."""
    model.eval()
    loader = DataLoader(
        dataset,
        batch_sz,
        shuffle=False,
        num_workers=data_loader_workers,
        pin_memory=True,
        persistent_workers=True,
    )
    correct = 0
    total = 0
    pred_counts = torch.zeros(num_classes, dtype=torch.long, device=device)
    with torch.no_grad():
        for imgs, targets in loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            outputs = model(imgs)
            preds = outputs.argmax(1)
            pred_counts += torch.bincount(preds, minlength=num_classes)
            correct += (preds == targets).sum().item()
            total += targets.size(0)
    return correct / total if total > 0 else 0.0, pred_counts.cpu().tolist()




## === cell 11
def plot_losses(epochs, title, train_losses, valid_losses):
    y = list(range(len(train_losses)))
    plt.plot(y, train_losses, label="train loss")
    plt.plot(y, valid_losses, label="valid loss")
    plt.title(title)
    plt.xlabel("epoch")
    plt.ylabel("loss")
    plt.legend()
    plt.show()




## === cell 12
def train_two_models(
    model_a,
    model_b,
    train_dataset,
    valid_dataset,
    batch_sz,
    optimizer_a,
    optimizer_b,
    scheduler_a,
    scheduler_b,
    loss_fn,
    epochs,
    save_path_a,
    save_path_b,
):
    best_state_a = None
    best_state_b = None
    best_loss_a = float("inf")
    best_loss_b = float("inf")
    best_acc_a = 0.0
    best_acc_b = 0.0

    train_losses_a, valid_losses_a = [], []
    train_losses_b, valid_losses_b = [], []

    scaler_a = GradScaler()
    scaler_b = GradScaler()

    train_loader = DataLoader(
        train_dataset,
        batch_sz,
        shuffle=True,
        num_workers=data_loader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_sz,
        shuffle=False,
        num_workers=data_loader_workers,
        pin_memory=True,
        persistent_workers=True,
        prefetch_factor=2,
    )

    for epoch in range(1, epochs + 1):
        model_a.train()
        model_b.train()
        epoch_train_loss_a = 0.0
        epoch_train_loss_b = 0.0

        for imgs, targets in train_loader:
            imgs = imgs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            optimizer_a.zero_grad()
            with autocast():
                outputs_a = model_a(imgs)
                loss_a = loss_fn(outputs_a, targets)
            scaler_a.scale(loss_a).backward()
            scaler_a.step(optimizer_a)
            scaler_a.update()
            epoch_train_loss_a += loss_a.item() * imgs.size(0)

            optimizer_b.zero_grad()
            with autocast():
                outputs_b = model_b(imgs)
                loss_b = loss_fn(outputs_b, targets)
            scaler_b.scale(loss_b).backward()
            scaler_b.step(optimizer_b)
            scaler_b.update()
            epoch_train_loss_b += loss_b.item() * imgs.size(0)

        epoch_train_loss_a /= len(train_loader.dataset)
        epoch_train_loss_b /= len(train_loader.dataset)
        train_losses_a.append(epoch_train_loss_a)
        train_losses_b.append(epoch_train_loss_b)

        model_a.eval()
        model_b.eval()
        epoch_valid_loss_a = 0.0
        epoch_valid_loss_b = 0.0
        correct_a = 0
        correct_b = 0
        total = 0

        with torch.no_grad():
            for imgs, targets in valid_loader:
                imgs = imgs.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)

                with autocast():
                    out_a = model_a(imgs)
                    out_b = model_b(imgs)
                    loss_a = loss_fn(out_a, targets)
                    loss_b = loss_fn(out_b, targets)

                epoch_valid_loss_a += loss_a.item() * imgs.size(0)
                epoch_valid_loss_b += loss_b.item() * imgs.size(0)

                preds_a = out_a.argmax(1)
                preds_b = out_b.argmax(1)
                correct_a += (preds_a == targets).sum().item()
                correct_b += (preds_b == targets).sum().item()
                total += targets.size(0)

        epoch_valid_loss_a /= len(valid_loader.dataset)
        epoch_valid_loss_b /= len(valid_loader.dataset)
        valid_losses_a.append(epoch_valid_loss_a)
        valid_losses_b.append(epoch_valid_loss_b)

        val_acc_a = correct_a / total if total > 0 else 0.0
        val_acc_b = correct_b / total if total > 0 else 0.0

        if epoch_valid_loss_a < best_loss_a:
            best_loss_a = epoch_valid_loss_a
            best_state_a = {k: v.cpu() for k, v in model_a.state_dict().items()}
            best_acc_a = val_acc_a

        if epoch_valid_loss_b < best_loss_b:
            best_loss_b = epoch_valid_loss_b
            best_state_b = {k: v.cpu() for k, v in model_b.state_dict().items()}
            best_acc_b = val_acc_b

        scheduler_a.step()
        scheduler_b.step()

        print(
            f"Epoch {epoch}/{epochs} – "
            f"ResNet loss: {epoch_train_loss_a:.4f}/{epoch_valid_loss_a:.4f} acc {val_acc_a:.4f} – "
            f"EffNet loss: {epoch_train_loss_b:.4f}/{epoch_valid_loss_b:.4f} acc {val_acc_b:.4f}"
        )

    torch.save(best_state_a, save_path_a)
    torch.save(best_state_b, save_path_b)

    model_a.load_state_dict(best_state_a)
    model_b.load_state_dict(best_state_b)

    return (
        model_a,
        model_b,
        train_losses_a,
        train_losses_b,
        valid_losses_a,
        valid_losses_b,
        best_acc_a,
        best_acc_b,
    )




## === cell 13
train_dataset = CachedCassavaDataset(train_df, transform=train_transform)
valid_dataset = CachedCassavaDataset(valid_df, transform=valid_transform)



## === cell 14
(
    resNet,
    ef_model,
    res_train_losses,
    ef_train_losses,
    res_valid_losses,
    ef_valid_losses,
    res_val_acc,
    ef_val_acc,
) = train_two_models(
    resNet,
    ef_model,
    train_dataset,
    valid_dataset,
    batch_size,
    resNet_optimizer,
    ef_optimizer,
    resNet_scheduler,
    ef_scheduler,
    criterion,
    epoch_num,
    "./res_model.pth",
    "./ef_model.pth",
)
print("ResNet validation accuracy:", res_val_acc)
print("EfficientNet validation accuracy:", ef_val_acc)



## === cell 15
if os.path.isfile("./res_model.pth"):
    resNet.load_state_dict(torch.load("./res_model.pth", map_location=device))
if os.path.isfile("./ef_model.pth"):
    ef_model.load_state_dict(torch.load("./ef_model.pth", map_location=device))




## === cell 16
class CassavaEnsemble(nn.Module):
    def __init__(self, model_a, model_b):
        super().__init__()
        self.model_a = model_a
        self.model_b = model_b

    def forward(self, x):
        return 0.5 * self.model_a(x) + 0.5 * self.model_b(x)


ensemble = CassavaEnsemble(resNet, ef_model).to(device)



## === cell 17
test_files = []
test_ids = []
for fname in os.listdir(test_images_path):
    fpath = os.path.join(test_images_path, fname)
    if os.path.isfile(fpath):
        test_ids.append(fname)
        test_files.append(fpath)




## === cell 18
class TestDataset(Dataset):
    def __init__(self, file_paths, transform=None):
        self.paths = file_paths
        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        img = Image.open(path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


test_dataset = TestDataset(test_files, transform=valid_transform)
test_loader = DataLoader(
    test_dataset,
    batch_size=64,
    shuffle=False,
    num_workers=data_loader_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

predictions = []
with torch.no_grad():
    for batch in tqdm(test_loader, desc="Predicting"):
        batch = batch.to(device, non_blocking=True)
        out = ensemble(batch)
        preds = out.argmax(1).cpu().numpy()
        predictions.extend(preds.tolist())



## === cell 19
submission = pd.DataFrame({"image_id": test_ids, "label": predictions})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission file written to {submission_path}, shape: {submission.shape}")
