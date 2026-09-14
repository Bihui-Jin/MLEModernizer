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
pytorch-ignite==0.5.3
pytorch-lightning==2.5.5
scikit-learn==1.2.2
scikit-learn-intelex==2025.9.0
seaborn==0.12.2
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
import numpy as np
import pandas as pd
import cv2
import glob
import gc
import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2
from sklearn.preprocessing import MultiLabelBinarizer
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import os  # added for robust path handling
from torch.cuda.amp import autocast, GradScaler  # mixed‑precision

torch.backends.cudnn.benchmark = True
torch.set_float32_matmul_precision("high")



## === cell 1
torch.cuda.empty_cache()
gc.collect()
DEBUG = False
DIMENTION = (256, 171)  # (height, width) image size
THRESHOLD = 0.5  # lowered from 0.75 to capture more labels
BATCH_SIZE = 128
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_csv_path = "/kaggle/input/plant-pathology-2021-fgvc8/train.csv"
test_images_path = "/kaggle/input/plant-pathology-2021-fgvc8/test_images/"
train_data = pd.read_csv(train_csv_path)
test_images_names = glob.glob(test_images_path + "*.jpg")

train_data["labels"] = train_data["labels"].apply(lambda s: s.split(" "))
mlb = MultiLabelBinarizer()
train_labels = pd.DataFrame(
    mlb.fit_transform(train_data["labels"].tolist()),
    columns=mlb.classes_,
    index=train_data.index,
)
labels_size = len(train_labels.columns)

train_df = pd.concat([train_data[["image"]], train_labels], axis=1)



## === cell 2
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.Resize(height=DIMENTION[0], width=DIMENTION[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

test_transform = A.Compose(
    [
        A.Resize(height=DIMENTION[0], width=DIMENTION[1]),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)




## === cell 3
class PlantDataSet(Dataset):
    """
    Pre‑loads all images (and applies transforms) into RAM during initialization.
    This removes per‑epoch cv2 reads and Albumentations processing, preserving
    exact tensors and labels while speeding up training and inference.
    """

    def __init__(self, dataframe, images_path, transform=None, is_test=False):
        self.is_test = is_test
        self.transform = transform
        self.images_path = images_path

        if self.is_test:
            img_paths = dataframe
            self.labels = None
        else:
            img_paths = [
                os.path.join(images_path, img_name)
                for img_name in dataframe["image"].values
            ]
            label_arrays = dataframe.drop(columns=["image"]).values.astype(np.float32)
            self.labels = torch.from_numpy(label_arrays)  # shape (N, C)

        self.tensors = []
        for path in img_paths:
            img = cv2.imread(path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            if self.transform:
                img = self.transform(image=img)["image"]
            else:
                img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            self.tensors.append(img)
        self.tensors = tuple(self.tensors)  # immutable for safety

    def __len__(self):
        return len(self.tensors)

    def __getitem__(self, idx):
        img_tensor = self.tensors[idx]
        if self.is_test:
            return img_tensor, torch.tensor([])  # empty target placeholder
        else:
            return img_tensor, self.labels[idx]




## === cell 4
num_workers = 0

train_dataset = PlantDataSet(
    train_df,
    "/kaggle/input/plant-pathology-2021-fgvc8/train_images/",
    transform=train_transform,
    is_test=False,
)
train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=True,
)

test_dataset = PlantDataSet(
    test_images_names, None, transform=test_transform, is_test=True
)
test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=True,
)


def train_one_epoch(model, loader, criterion, optimizer, scaler):
    model.train()
    running_loss = 0.0
    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad()
        with autocast():
            outputs = model(images)
            loss = criterion(outputs, targets)
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * images.size(0)

    return running_loss / len(loader.dataset)




## === cell 5
def test(test_dataloader, model):
    model.eval()
    preds_list = []
    with torch.no_grad():
        for images, _ in test_dataloader:
            images = images.to(device, non_blocking=True)
            with autocast():
                output = model(images)
                probabilities = torch.sigmoid(output)
            preds_list.append(probabilities.cpu().numpy())
    predictions = np.concatenate(preds_list, axis=0)
    return predictions




## === cell 6
def create_submission(test_images_path_list, predictions):
    rows = []
    for img_path, pred_vec in zip(test_images_path_list, predictions):
        name = img_path.split("/")[-1]
        selected = [
            lbl for prob, lbl in zip(pred_vec, train_labels.columns) if prob > THRESHOLD
        ]
        if not selected:
            selected = ["healthy"]
        prediction_labels = " ".join(selected)
        rows.append([name, prediction_labels])

    submission_df = pd.DataFrame(rows, columns=["image", "labels"])
    if DEBUG:
        display(submission_df.head())
    submission_df.to_csv("submission.csv", index=False)




## === cell 7
resnext50 = torchvision.models.resnext50_32x4d(pretrained=True)
in_features = resnext50.fc.in_features
resnext50.fc = nn.Linear(in_features, labels_size)



## === cell 8
resnext50 = resnext50.to(device)
criterion = nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(resnext50.parameters(), lr=1e-4)
scaler = GradScaler()  # mixed‑precision scaler

EPOCHS = 3
for epoch in range(EPOCHS):
    loss = train_one_epoch(resnext50, train_loader, criterion, optimizer, scaler)
    if DEBUG:
        print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {loss:.4f}")

test_predictions = test(test_loader, resnext50)
print("Test predictions shape:", test_predictions.shape)
create_submission(test_images_names, test_predictions)
