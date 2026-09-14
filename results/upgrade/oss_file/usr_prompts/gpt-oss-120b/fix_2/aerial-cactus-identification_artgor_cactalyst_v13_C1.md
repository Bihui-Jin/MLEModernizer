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
Create a classifier to predict whether an image contains a cactus.

## Metric
Area under the ROC curve.

## Submission Format
For each ID in the test set, you must predict a probability for the `has_cactus` variable. The file should contain a header and have the following format:

```
id,has_cactus
000940378805c44108d287872b2f04ce.jpg,0.5
0017242f54ececa4512b4d7937d1e21e.jpg,0.5
001ee6d8564003107853118ab87df407.jpg,0.5
etc.
```

## Dataset
This dataset contains a large number of 32 x 32 thumbnail images containing aerial photos of a cactus. The file name of an image corresponds to its `id`.

- **train/** - the training set images
- **test/** - the test set images (you must predict the labels of these)
- **train.csv** - the training set labels, indicates whether the image has a cactus (`has_cactus = 1`)
- **sample_submission.csv** - a sample submission file in the correct format

# 2. Python version

3.7

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

# 4. Data file paths

```
/
    kaggle/
        data/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
        input/
            description.md (56 lines)
            sample_submission.csv (3326 lines)
            sample_submission.csv.zip (67.3 kB)
            test.zip (3.5 MB)
            train.csv (14176 lines)
            train.csv.zip (285.6 kB)
            train.zip (15.0 MB)
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
            test/
                76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                ... and 3323 other files
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
            train/
                775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                ... and 14173 other files
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
        working/
            aerial-cactus-identification/
                description.md (56 lines)
                sample_submission.csv (3326 lines)
                ... and 5 other files
                aerial-cactus-identification/
                test/
                    76bad42ebc1ed65f7f50c06fd17849db.jpg (1.2 kB)
                    f620bd2745d51c25cd05eca4f7c4da94.jpg (1.1 kB)
                    ... and 3323 other files
                    test/
                train/
                    775da0be6da934cb05d6bc7955931dd9.jpg (1.0 kB)
                    65a52562f1ebce1166d9737ac9d1c0e5.jpg (960 Bytes)
                    ... and 14173 other files
                    train/
```

-> data/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> data/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> data/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/sample_submission.csv has 3325 rows and 2 columns.
The columns are: id, has_cactus

-> input/aerial-cactus-identification/train.csv has 14175 rows and 2 columns.
The columns are: id, has_cactus

-> (stopped after 10 files for performance)

# 5. Target score

0.9996

# 6. Current score

Not yielded

# 7. Whether higher score is better

Higher is better

# 8. Previous improvement plan

N/A

# 9. Code solution

## === cell 0
import os
import numpy as np
import pandas as pd
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler
import albumentations as A
from albumentations.pytorch import ToTensorV2
import pretrainedmodels
from collections import OrderedDict


def train_val_split(df, val_frac=0.1, seed=42):
    np.random.seed(seed)
    idx = np.random.permutation(len(df))
    split = int(len(df) * (1 - val_frac))
    train_idx, val_idx = idx[:split], idx[split:]
    return df.iloc[train_idx].reset_index(drop=True), df.iloc[val_idx].reset_index(
        drop=True
    )


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)




## --- ERROR in cell 0, traceback:
---------------------------------------------------------------------------
ModuleNotFoundError                       Traceback (most recent call last)
/tmp/ipykernel_55/1569564841.py in <cell line: 0>()
     11 import albumentations as A
     12 from albumentations.pytorch import ToTensorV2
---> 13 import pretrainedmodels
     14 from collections import OrderedDict
     15 

ModuleNotFoundError: No module named 'pretrainedmodels'

## === cell 1
data_transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightness(),
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

data_transforms_test = A.Compose(
    [A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), ToTensorV2()]
)




## --- ERROR in cell 1, traceback:
---------------------------------------------------------------------------
AttributeError                            Traceback (most recent call last)
/tmp/ipykernel_55/1096174640.py in <cell line: 0>()
      4         A.HorizontalFlip(),
      5         A.VerticalFlip(),
----> 6         A.RandomBrightness(),
      7         A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
      8         ToTensorV2(),

AttributeError: module 'albumentations' has no attribute 'RandomBrightness'

## === cell 2
train_path = os.path.join("..", "input", "aerial-cactus-identification", "train.csv")
train_df = pd.read_csv(train_path)

train_split, valid_split = train_val_split(train_df, val_frac=0.1)

img_class_dict = dict(zip(train_df["id"], train_df["has_cactus"]))




## --- ERROR in cell 2, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2062701740.py in <cell line: 0>()
      4 
      5 # split into train / valid
----> 6 train_split, valid_split = train_val_split(train_df, val_frac=0.1)
      7 
      8 # mapping from filename to label

NameError: name 'train_val_split' is not defined

## === cell 3
class CactusDataset(Dataset):
    def __init__(self, folder, file_list, transform, labels_dict=None):
        self.folder = folder
        self.file_list = file_list
        self.transform = transform
        self.labels_dict = labels_dict

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        img_name = self.file_list[idx]
        img_path = os.path.join(self.folder, img_name)
        img = cv2.imread(img_path)
        img = img[:, :, ::-1]
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]
        if self.labels_dict is not None:
            label = float(self.labels_dict[img_name])
        else:
            label = 0.0
        return img_tensor, torch.tensor(label, dtype=torch.float32)




## === cell 4
train_folder = os.path.join(
    "..", "input", "aerial-cactus-identification", "train", "train"
)
test_folder = os.path.join(
    "..", "input", "aerial-cactus-identification", "test", "test"
)

train_dataset = CactusDataset(
    folder=train_folder,
    file_list=train_split["id"].tolist(),
    transform=data_transforms,
    labels_dict=img_class_dict,
)

valid_dataset = CactusDataset(
    folder=train_folder,
    file_list=valid_split["id"].tolist(),
    transform=data_transforms_test,
    labels_dict=img_class_dict,
)

test_dataset = CactusDataset(
    folder=test_folder,
    file_list=os.listdir(test_folder),
    transform=data_transforms_test,
    labels_dict=None,
)

batch_size = 512
num_workers = 0

train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(list(range(len(train_dataset)))),
    num_workers=num_workers,
    pin_memory=True,
)

valid_loader = DataLoader(
    valid_dataset,
    batch_size=batch_size,
    sampler=SubsetRandomSampler(list(range(len(valid_dataset)))),
    num_workers=num_workers,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)




## --- ERROR in cell 4, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2595529704.py in <cell line: 0>()
     10 train_dataset = CactusDataset(
     11     folder=train_folder,
---> 12     file_list=train_split["id"].tolist(),
     13     transform=data_transforms,
     14     labels_dict=img_class_dict,

NameError: name 'train_split' is not defined

## === cell 5
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, num_classes=1, p=0.2, arch="densenet169", pretrained="imagenet"):
        super().__init__()
        backbone = pretrainedmodels.__dict__[arch](pretrained=pretrained)
        modules = list(backbone.children())[:-1]  # remove original classifier
        modules += [
            nn.Sequential(
                Flatten(),
                nn.BatchNorm1d(1664),
                nn.Dropout(p),
                nn.Linear(1664, num_classes),
            )
        ]
        self.net = nn.Sequential(*modules)

    def forward(self, x):
        logits = self.net(x)
        return torch.squeeze(logits)




## === cell 6
num_epochs = 10
model = Net(num_classes=1).to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)
scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.1, patience=2)

best_auc = 0.0

for epoch in range(1, num_epochs + 1):
    model.train()
    epoch_loss = 0.0
    for imgs, targets in train_loader:
        imgs = imgs.to(device)
        targets = targets.to(device).unsqueeze(1)
        optimizer.zero_grad()
        outputs = model(imgs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()
        epoch_loss += loss.item() * imgs.size(0)
    epoch_loss /= len(train_loader.dataset)

    model.eval()
    all_preds = []
    all_targets = []
    with torch.no_grad():
        for imgs, targets in valid_loader:
            imgs = imgs.to(device)
            targets = targets.to(device).unsqueeze(1)
            outputs = model(imgs)
            probs = torch.sigmoid(outputs)
            all_preds.append(probs.cpu().numpy())
            all_targets.append(targets.cpu().numpy())
    val_preds = np.concatenate(all_preds).ravel()
    val_targets = np.concatenate(all_targets).ravel()
    order = np.argsort(val_preds)
    sorted_targets = val_targets[order]
    pos = np.sum(sorted_targets)
    if pos == 0 or pos == len(sorted_targets):
        auc = 0.5
    else:
        cum_pos = np.cumsum(sorted_targets)
        auc = (np.sum(cum_pos[sorted_targets == 0]) / pos) / (len(sorted_targets) - pos)
    scheduler.step(auc)

    if auc > best_auc:
        best_auc = auc
        torch.save(model.state_dict(), "best_model.pth")

    print(f"Epoch {epoch}/{num_epochs} - loss: {epoch_loss:.4f} - val_auc: {auc:.4f}")

print("Best validation AUC:", best_auc)




## --- ERROR in cell 6, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/1407955170.py in <cell line: 0>()
      1 # training hyper‑parameters
      2 num_epochs = 10
----> 3 model = Net(num_classes=1).to(device)
      4 criterion = nn.BCEWithLogitsLoss()
      5 optimizer = optim.SGD(model.parameters(), lr=1e-2, momentum=0.99)

/tmp/ipykernel_55/3116984622.py in __init__(self, num_classes, p, arch, pretrained)
      7     def __init__(self, num_classes=1, p=0.2, arch="densenet169", pretrained="imagenet"):
      8         super().__init__()
----> 9         backbone = pretrainedmodels.__dict__[arch](pretrained=pretrained)
     10         modules = list(backbone.children())[:-1]  # remove original classifier
     11         modules += [

NameError: name 'pretrainedmodels' is not defined

## === cell 7
model.load_state_dict(torch.load("best_model.pth", map_location=device))
model.eval()

test_ids = test_dataset.file_list
all_test_preds = []
with torch.no_grad():
    for imgs, _ in test_loader:
        imgs = imgs.to(device)
        outputs = model(imgs)
        probs = torch.sigmoid(outputs)
        all_test_preds.append(probs.cpu().numpy())
test_preds = np.concatenate(all_test_preds).ravel()

submission = pd.DataFrame({"id": test_ids, "has_cactus": test_preds})
submission_path = "submission.csv"
submission.to_csv(submission_path, index=False)
print("Submission saved to", submission_path)

## --- ERROR in cell 7, traceback:
---------------------------------------------------------------------------
NameError                                 Traceback (most recent call last)
/tmp/ipykernel_55/2535906873.py in <cell line: 0>()
      1 # load best model for inference
----> 2 model.load_state_dict(torch.load("best_model.pth", map_location=device))
      3 model.eval()
      4 
      5 test_ids = test_dataset.file_list

NameError: name 'model' is not defined
