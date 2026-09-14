# Goal

Make the code finish within a 600-second timeout. The last attempt timed out after 10 minutes. Optimize for speed WITHOUT harming result accuracy and WITHOUT changing the core logic.

# Requirements

- Preserve the core logic, including model architecture, layers, training approach/loops, feature extraction, or loss function. Maintain identical core logic and evaluation semantics; only allow negligible floating-point differences.
- Every change must be directly relevant to the stated issue (timeout fix); avoid unrelated refactors or stylistic edits.
- Do NOT introduce extra approximations, sampling, reduced precision, early stopping, or relaxed convergence criteria.
- Keep file paths unchanged.


# 1. Kaggle task description

## Task
Detect apple diseases from images.

## Metric
Mean F1-Score

## Submission Format
labels should be a space-delimited list.

The file should contain a header and have the following format:

```
image, labels
85f8cb619c66b863.jpg,healthy
ad8770db05586b59.jpg,healthy
c7b03e718489f3ca.jpg,healthy
```

## Dataset
**train.csv** - the training set metadata.

- `image` - the image ID.
- `labels` - the target classes, a space delimited list of all diseases found in the image. Unhealthy leaves with too many diseases to classify visually will have the `complex` class, and may also have a subset of the diseases identified.

**sample_submission.csv** - A sample submission file in the correct format.

- `image`
- `labels`

**train_images** - The training set images.

**test_images** - The test set images. This competition has a hidden test set: only three images are provided here as samples while the remaining 5,000 images will be available to your notebook once it is submitted.

# 2. Python version

3.9

# 3. Installed packages

geopandas==0.14.4
matplotlib==3.7.2
matplotlib-inline==0.1.7
matplotlib-venn==1.1.2
numpy==1.26.4
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
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
        input/
            description.md (101 lines)
            sample_submission.csv (3728 lines)
            sample_submission.csv.zip (39.6 kB)
            test.zip (160 Bytes)
            test_images.zip (3.2 GB)
            train.csv (14906 lines)
            train.csv.zip (171.3 kB)
            train.zip (162 Bytes)
            train_images.zip (12.7 GB)
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
            test_images/
                df98c83c4d383c2d.jpg (802.8 kB)
                817e97dad0c33ae0.jpg (667.3 kB)
                ... and 3725 other files
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
            train_images/
                c19a7aca95e54c35.jpg (1.1 MB)
                8476bd24bd4b89a5.jpg (985.0 kB)
                ... and 14903 other files
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
        working/
            plant-pathology-2021-fgvc8/
                description.md (101 lines)
                sample_submission.csv (3728 lines)
                ... and 7 other files
                plant-pathology-2021-fgvc8/
                test_images/
                    df98c83c4d383c2d.jpg (802.8 kB)
                    817e97dad0c33ae0.jpg (667.3 kB)
                    ... and 3725 other files
                    test_images/
                train_images/
                    c19a7aca95e54c35.jpg (1.1 MB)
                    8476bd24bd4b89a5.jpg (985.0 kB)
                    ... and 14903 other files
                    train_images/
```

-> data/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> data/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> data/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/sample_submission.csv has 3727 rows and 2 columns.
The columns are: image, labels

-> input/plant-pathology-2021-fgvc8/train.csv has 14905 rows and 2 columns.
The columns are: image, labels

-> (stopped after 10 files for performance)

# 5. Code solution

## === cell 0
import glob
import os
import os.path as osp

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from tqdm.notebook import tqdm
from PIL import Image

import matplotlib.pyplot as plt


import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision
from torchvision import models, transforms

torch.manual_seed(0)
np.random.seed(0)




## === cell 1
df_train = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/train.csv")
df_sub = pd.read_csv("/kaggle/input/plant-pathology-2021-fgvc8/sample_submission.csv")




## === cell 2
def to_label(df):
    le = LabelEncoder()
    df["labels_n"] = le.fit_transform(df.labels.values)
    return df


df_train = to_label(df_train)
df_labels_idx = (
    df_train.loc[df_train.duplicated(["labels", "labels_n"]) == False][
        ["labels_n", "labels"]
    ]
    .set_index("labels_n")
    .sort_index()
)
num_classes = df_labels_idx.shape[0]




## === cell 3
def make_datapath_list(phase="train", val_size=0.25):
    if phase in ["train", "val"]:
        phase_path = "train_images"
    elif phase == "test":
        phase_path = "test_images"
    else:
        raise ValueError(f"{phase} not recognized")
    rootpath = "/kaggle/input/plant-pathology-2021-fgvc8/"
    target_path = osp.join(rootpath, phase_path, "*.jpg")
    path_list = glob.glob(target_path)

    if phase in ["train", "val"]:
        train, val = train_test_split(
            path_list, test_size=val_size, random_state=0, shuffle=True
        )
        path_list = train if phase == "train" else val
    return path_list




## === cell 4
train_list = make_datapath_list(phase="train")
val_list = make_datapath_list(phase="val")
test_list = make_datapath_list(phase="test")




## === cell 5
class ImageTransform:
    def __init__(self, resize, mean, std):
        self.data_transform = {
            "train": transforms.Compose(
                [
                    transforms.RandomResizedCrop(resize, scale=(0.5, 1.0)),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "val": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
            "test": transforms.Compose(
                [
                    transforms.Resize(resize),
                    transforms.CenterCrop(resize),
                    transforms.ToTensor(),
                    transforms.Normalize(mean, std),
                ]
            ),
        }

    def __call__(self, img, phase="train"):
        return self.data_transform[phase](img)


size = 224
mean = (0.485, 0.456, 0.406)
std = (0.229, 0.224, 0.225)
transform = ImageTransform(size, mean, std)




## === cell 6
class PlantDataset(data.Dataset):
    """
    For validation and test phases we cache the transformed tensors to avoid
    repeating disk I/O and CPU image preprocessing. Training samples keep the
    original lazy loading to preserve random augmentations.
    """

    def __init__(self, df_train, file_list, transform=None, phase="train"):
        self.df_train = df_train
        self.df_labels_idx = df_labels_idx
        self.file_list = file_list
        self.transform = transform
        self.phase = phase
        self.label_map = dict(zip(df_train["image"], df_train["labels_n"]))

        self._cached = None
        if self.phase in ["val", "test"]:
            self._cached = []
            for p in self.file_list:
                img = Image.open(p).convert("RGB")
                tensor = self.transform(img, self.phase)
                self._cached.append(tensor)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        if self._cached is not None:
            img_tensor = self._cached[index]
        else:
            img_path = self.file_list[index]
            img = Image.open(img_path).convert("RGB")
            img_tensor = self.transform(img, self.phase)

        image_name = os.path.basename(self.file_list[index])
        if self.phase in ["train", "val"]:
            label = self.label_map[image_name]
        else:
            label = -1

        return img_tensor, label, image_name




## === cell 7
train_dataset = PlantDataset(df_train, train_list, transform=transform, phase="train")
val_dataset = PlantDataset(df_train, val_list, transform=transform, phase="val")
test_dataset = PlantDataset(df_train, test_list, transform=transform, phase="test")




## === cell 8
batch_size = 512
num_workers = min(4, os.cpu_count() or 1)

train_dataloader = data.DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
val_dataloader = data.DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)
test_dataloader = data.DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
    persistent_workers=True,
    prefetch_factor=2,
)

dataloaders_dict = {
    "train": train_dataloader,
    "val": val_dataloader,
    "test": test_dataloader,
}




## === cell 9
net = models.vgg16(pretrained=False)
net.classifier[6] = nn.Linear(in_features=4096, out_features=num_classes)
net.train()

for param in net.parameters():
    param.requires_grad = True

if hasattr(torch, "compile"):
    net = torch.compile(net)




## === cell 10
criterion = nn.CrossEntropyLoss()




## === cell 11
optimizer = optim.SGD(net.parameters(), lr=1e-3, momentum=0.9)




## === cell 12
def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    net.to(device)
    torch.backends.cudnn.benchmark = True

    scaler = torch.cuda.amp.GradScaler() if device.type == "cuda" else None

    autocast_ctx = (
        torch.cuda.amp.autocast
        if device.type == "cuda"
        else lambda **kwargs: torch.autocast(device_type="cpu", **kwargs)
    )

    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        print("-" * 30)
        for phase in ["train", "val"]:
            net.train() if phase == "train" else net.eval()
            epoch_loss = 0.0
            epoch_corrects = 0

            for inputs, labels, _ in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device, non_blocking=True)
                labels = labels.to(device, non_blocking=True)

                if phase == "train":
                    optimizer.zero_grad()
                    with autocast_ctx():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                    if scaler:
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        loss.backward()
                        optimizer.step()
                else:
                    with autocast_ctx():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)
                epoch_loss += loss.item() * inputs.size(0)
                epoch_corrects += torch.sum(preds == labels.data)

            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double() / len(dataloaders_dict[phase].dataset)
            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")




## === cell 13
num_epochs = 10  # keep original epoch count
train_model(net, dataloaders_dict, criterion, optimizer, num_epochs)




## === cell 14
class PlantPredictor:
    def __init__(self, net, df_labels_idx, dataloaders_dict):
        self.net = net
        self.df_labels_idx = df_labels_idx
        self.dataloaders_dict = dataloaders_dict
        self.df_submit = pd.DataFrame()

    def __predict_max(self, out_tensor):
        max_ids = torch.argmax(out_tensor, dim=1).cpu().numpy()
        return self.df_labels_idx.iloc[max_ids]

    def inference(self):
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(device)
        self.net.eval()
        pred_batches = []

        for inputs, _, image_names in tqdm(self.dataloaders_dict["test"]):
            inputs = inputs.to(device, non_blocking=True)
            with torch.no_grad():
                with (
                    torch.cuda.amp.autocast()
                    if device.type == "cuda"
                    else torch.autocast(device_type="cpu")
                ):
                    outputs = self.net(inputs)
            df_pred = self.__predict_max(outputs).reset_index(drop=True)
            df_pred["image"] = image_names
            pred_batches.append(df_pred)

        self.df_submit = pd.concat(pred_batches, axis=0)[
            ["image", "labels"]
        ].reset_index(drop=True)




## === cell 15
predictor = PlantPredictor(net, df_labels_idx, dataloaders_dict)
predictor.inference()




## === cell 16
df_submit = predictor.df_submit.copy()
df_submit.to_csv("/kaggle/working/submission.csv", index=False)
