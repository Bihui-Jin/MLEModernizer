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
import torchvision.models as models
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




## === cell 1
data_transforms = A.Compose(
    [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.RandomBrightnessContrast(),  # replaced missing RandomBrightness
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)

data_transforms_test = A.Compose(
    [
        A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ToTensorV2(),
    ]
)




## === cell 2
train_path = os.path.join("..", "input", "aerial-cactus-identification", "train.csv")
train_df = pd.read_csv(train_path)

train_split, valid_split = train_val_split(train_df, val_frac=0.1)

img_class_dict = dict(zip(train_df["id"], train_df["has_cactus"]))




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
        if img is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        img = img[:, :, ::-1]  # BGR to RGB
        augmented = self.transform(image=img)
        img_tensor = augmented["image"]
        if self.labels_dict is not None:
            label = float(self.labels_dict[img_name])
        else:
            label = 0.0
        return img_tensor, torch.tensor(label, dtype=torch.float32)




## === cell 4
train_folder = os.path.join("..", "input", "aerial-cactus-identification", "train")
test_folder = os.path.join("..", "input", "aerial-cactus-identification", "test")

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

test_files = sorted(os.listdir(test_folder))
test_dataset = CactusDataset(
    folder=test_folder,
    file_list=test_files,
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




## === cell 5
class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Net(nn.Module):
    def __init__(self, num_classes=1, p=0.2):
        super().__init__()
        backbone = models.densenet169(pretrained=True)
        features = backbone.features
        self.net = nn.Sequential(
            features,
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.BatchNorm1d(1664),
            nn.Dropout(p),
            nn.Linear(1664, num_classes),
        )

    def forward(self, x):
        logits = self.net(x)
        return logits




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
        targets = targets.to(device).unsqueeze(1)  # shape (batch, 1)
        optimizer.zero_grad()
        outputs = model(imgs)  # shape (batch, 1)
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
FileNotFoundError                         Traceback (most recent call last)
/tmp/ipykernel_55/2448797596.py in <cell line: 0>()
      6 all_test_preds = []
      7 with torch.no_grad():
----> 8     for imgs, _ in test_loader:
      9         imgs = imgs.to(device)
     10         outputs = model(imgs)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in __next__(self)
    706                 # TODO(https://github.com/pytorch/pytorch/issues/76750)
    707                 self._reset()  # type: ignore[call-arg]
--> 708             data = self._next_data()
    709             self._num_yielded += 1
    710             if (

/usr/local/lib/python3.11/dist-packages/torch/utils/data/dataloader.py in _next_data(self)
    762     def _next_data(self):
    763         index = self._next_index()  # may raise StopIteration
--> 764         data = self._dataset_fetcher.fetch(index)  # may raise StopIteration
    765         if self._pin_memory:
    766             data = _utils.pin_memory.pin_memory(data, self._pin_memory_device)

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in fetch(self, possibly_batched_index)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/usr/local/lib/python3.11/dist-packages/torch/utils/data/_utils/fetch.py in <listcomp>(.0)
     50                 data = self.dataset.__getitems__(possibly_batched_index)
     51             else:
---> 52                 data = [self.dataset[idx] for idx in possibly_batched_index]
     53         else:
     54             data = self.dataset[possibly_batched_index]

/tmp/ipykernel_55/1772422767.py in __getitem__(self, idx)
     14         img = cv2.imread(img_path)
     15         if img is None:
---> 16             raise FileNotFoundError(f"Image not found: {img_path}")
     17         img = img[:, :, ::-1]  # BGR to RGB
     18         augmented = self.transform(image=img)

FileNotFoundError: Image not found: ../input/aerial-cactus-identification/test/test
