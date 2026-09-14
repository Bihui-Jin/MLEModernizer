# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Given a dataset of images from digital pathology scans, predict if the center 32x32px region of a patch contains at least one pixel of tumor tissue. Tumor tissue in the outer region of the patch does not influence the label. 

## Metric
Area under the ROC curve.

## Submission Format
For each `id` in the test set, you must predict a probability that center 32x32px region of a patch contains at least one pixel of tumor tissue. The file should contain a header and have the following format:

```
id,label
0b2ea2a822ad23fdb1b5dd26653da899fbd2c0d5,0
95596b92e5066c5c52466c90b69ff089b39f2737,0
248e6738860e2ebcf6258cdc1f32f299e0c76914,0
etc.
```

## Dataset
Files are named with an image `id`. The `train_labels.csv` file provides the ground truth for the images in the `train` folder. You are predicting the labels for the images in the `test` folder.

# 2. Python version

3.8

# 3. Installed packages

albumentations==2.0.8
geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
opencv-python==4.12.0.88
opencv-python-headless==4.12.0.88
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
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
        input/
            description.md (63 lines)
            sample_submission.csv (45562 lines)
            sample_submission.csv.zip (1.1 MB)
            test.zip (1.1 GB)
            train.zip (4.2 GB)
            train_labels.csv (174465 lines)
            train_labels.csv.zip (4.2 MB)
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
            test/
                7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                ... and 45559 other files
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
            train/
                bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                ... and 174462 other files
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
        working/
            histopathologic-cancer-detection/
                description.md (63 lines)
                sample_submission.csv (45562 lines)
                ... and 5 other files
                histopathologic-cancer-detection/
                test/
                    7d1637c3535cd849727c50dff5fb0efd42f500a7.tif (27.9 kB)
                    c66203935db093d22a62c667636345dab7ee67ba.tif (27.9 kB)
                    ... and 45559 other files
                    test/
                train/
                    bc9b47c5fd125f59519a4f719bf459f919164104.tif (27.9 kB)
                    0874a429121cea137156954353d1b287022a6f65.tif (27.9 kB)
                    ... and 174462 other files
                    train/
```

-> data/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> data/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> data/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/sample_submission.csv has 45561 rows and 2 columns.
The columns are: id, label

-> input/histopathologic-cancer-detection/train_labels.csv has 174464 rows and 2 columns.
The columns are: id, label

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import os
import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torchvision.models as models
import albumentations as A
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

torch.backends.cudnn.benchmark = True

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size_test = 128
batch_size_train = 64
num_workers = 8  # increase workers for faster I/O
num_epochs = 50
learning_rate = 1e-5




## === cell 1
class MyDataset(Dataset):
    """
    Simple dataset that reads images given a list of IDs.
    For the test set we do not have ground‑truth labels, so a dummy
    label (0) is returned.
    """

    def __init__(self, ids, dataloc, transform=None, labels=None):
        self.ids = ids
        self.dataloc = dataloc
        self.transform = transform
        self.labels = labels

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        img_id = self.ids[idx]
        img_path = os.path.join(self.dataloc, f"{img_id}.tif")
        img = cv2.imread(img_path)
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        if self.transform:
            img = self.transform(image=img)["image"]
        if self.labels is None:
            label = torch.tensor(0.0, dtype=torch.float32)
        else:
            label = torch.tensor(float(self.labels[idx]), dtype=torch.float32)
        return img, label




## === cell 2
data_transforms_test = A.Compose(
    [
        A.Resize(224, 224),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

data_transforms_train = A.Compose(
    [
        A.RandomResizedCrop(size=(224, 224), scale=(0.8, 1.0)),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class Densenet169(nn.Module):
    """
    Wrapper around torchvision's DenseNet‑169.
    Concatenates max‑ and avg‑pooled features before a small head.
    """

    def __init__(self, pretrained=True):
        super(Densenet169, self).__init__()
        self.backbone = models.densenet169(pretrained=pretrained)
        self.backbone.classifier = nn.Identity()
        self.linear = nn.Linear(3328, 128)
        self.bn = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.2)
        self.elu = nn.ELU()
        self.out = nn.Linear(128, 1)

    def forward(self, x):
        features = self.backbone.features(x)
        max_pool = torch.max(features.view(x.size(0), 1664, -1), dim=2)[0]
        avg_pool = torch.mean(features.view(x.size(0), 1664, -1), dim=2)
        conc = torch.cat((avg_pool, max_pool), dim=1)
        conc = self.linear(conc)
        conc = self.elu(conc)
        conc = self.bn(conc)
        conc = self.dropout(conc)
        out = self.out(conc)
        return out




## === cell 4
model = Densenet169(pretrained=True).to(device)
model.train()



## === cell 5
train_labels_path = "../input/histopathologic-cancer-detection/train_labels.csv"
train_df = pd.read_csv(train_labels_path)
train_ids = train_df["id"].astype(str).values
train_labels = train_df["label"].values

ids_train, ids_val, y_train, y_val = train_test_split(
    train_ids, train_labels, test_size=0.1, stratify=train_labels, random_state=42
)

train_dataset = MyDataset(
    ids=ids_train,
    dataloc="../input/histopathologic-cancer-detection/train/",
    transform=data_transforms_train,
    labels=y_train,
)

val_dataset = MyDataset(
    ids=ids_val,
    dataloc="../input/histopathologic-cancer-detection/train/",
    transform=data_transforms_test,
    labels=y_val,
)

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size_train,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size_train,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

pos_ratio = np.mean(y_train)
neg_ratio = 1.0 - pos_ratio
pos_weight = neg_ratio / pos_ratio if pos_ratio > 0 else 1.0

criterion = nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, dtype=torch.float32).to(device)
)
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

use_amp = device.type == "cuda"
scaler = torch.cuda.amp.GradScaler() if use_amp else None

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_losses = []
    for imgs, lbls in tqdm(train_loader, desc=f"Epoch {epoch}/{num_epochs} - Training"):
        imgs = imgs.to(device, non_blocking=True)
        lbls = lbls.to(device, non_blocking=True).unsqueeze(1)

        optimizer.zero_grad()
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
                loss = criterion(logits, lbls)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            logits = model(imgs)
            loss = criterion(logits, lbls)
            loss.backward()
            optimizer.step()
        epoch_losses.append(loss.item())
    avg_loss = np.mean(epoch_losses)

    model.eval()
    val_preds = []
    val_targets = []
    with torch.no_grad():
        for imgs, lbls in tqdm(
            val_loader, desc=f"Epoch {epoch}/{num_epochs} - Validation"
        ):
            imgs = imgs.to(device, non_blocking=True)
            if use_amp:
                with torch.cuda.amp.autocast():
                    logits = model(imgs)
            else:
                logits = model(imgs)
            probs = torch.sigmoid(logits).cpu().numpy().flatten()
            val_preds.extend(probs)
            val_targets.extend(lbls.numpy())
    val_auc = roc_auc_score(val_targets, val_preds)
    print(f"Epoch {epoch}: Train loss={avg_loss:.4f}, Val AUC={val_auc:.4f}")



## === cell 6
model.eval()
sample_sub_path = "../input/histopathologic-cancer-detection/sample_submission.csv"
test_df = pd.read_csv(sample_sub_path)
test_ids = test_df["id"].astype(str).values.tolist()

test_dataset = MyDataset(
    ids=test_ids,
    dataloc="../input/histopathologic-cancer-detection/test/",
    transform=data_transforms_test,
    labels=None,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size_test,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
)

preds = []
with torch.no_grad():
    for imgs, _ in tqdm(test_loader, desc="Inference"):
        imgs = imgs.to(device, non_blocking=True)
        if use_amp:
            with torch.cuda.amp.autocast():
                logits = model(imgs)
        else:
            logits = model(imgs)
        probs = torch.sigmoid(logits).squeeze().cpu().numpy().tolist()
        preds.extend(probs)

submission = pd.DataFrame({"id": test_ids, "label": preds})
submission_path = "sub.csv"
submission.to_csv(submission_path, index=False)
print(f"Submission saved to {submission_path}")
